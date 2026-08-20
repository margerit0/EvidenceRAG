"""The embedding provider client, in one place.

Extracted from ``scripts/probe_mrl_quality.py`` and
``scripts/verify_embedding_api.py``, which had grown a copy each of ``_load_env``
and ``_post``. Three scripts would have been three chances for the retry ladder
to drift, and the retry ladder is not incidental -- see :func:`backoff_seconds`.

Everything that can be a pure function is one, so the parts that decide
correctness (which HTTP codes are retried, how long to wait, how the cache
merges, how the base URL is resolved) are unit-testable without a network. The
one method that does I/O takes its transport as an argument for the same reason.

The provider is an OpenAI-compatible relay, not SiliconFlow, and it has three
measured quirks that this module encodes rather than documents elsewhere:

* **A missing User-Agent is a 403.** Cloudflare rejects ``Python-urllib/3.x``
  with ``error code 1010``, which is indistinguishable from a bad key.
* **429 means the upstream pool is saturated**, not that a quota was hit, so it
  clears on the provider's schedule rather than ours. The backoff is long on
  purpose.
* **The same text embeds differently across calls** (cosine 0.999931). Caching
  is therefore not just an optimisation: it is what makes a run reproducible.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from zhrag.io_utils import append_jsonl, read_json, read_jsonl, read_text, write_json

__all__ = [
    "MAX_RETRY_AFTER",
    "QUERY_PROMPT",
    "RETRY_STATUS",
    "BatchEmbedder",
    "EmbeddingClient",
    "EmbeddingConfig",
    "backoff_seconds",
    "explain_http_error",
    "load_env",
    "load_or_embed",
    "resolve_embeddings_url",
]

#: Asymmetric by design. Qwen3's document prompt is the empty string; adding a
#: prefix to both sides silently costs several points of R@1 and raises nothing.
#: The instruction is English even though the corpus is Chinese -- Qwen's own
#: advice, because the training-time instructions were English.
QUERY_PROMPT = (
    "Instruct: Given a Chinese question, retrieve the news passage that answers it\nQuery:"
)

#: Cloudflare 403s the stdlib default UA with error 1010.
USER_AGENT = "zhrag/0.1 (+https://github.com/margerit0/zhrag)"

#: How much of a provider error body to surface. Generous on purpose: the body
#: is the only place an unsupported parameter or a gated key group announces
#: itself, and truncating it turns a diagnosable failure into a status code.
ERROR_DETAIL_CHARS = 600

#: Transient upstream conditions. Everything else is a bug in the request and
#: retrying it just spends the same money seven times.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

#: Cap on a server-sent ``Retry-After``. The server knows more than we do about
#: when it will be ready, but not enough to be handed an unbounded sleep.
MAX_RETRY_AFTER = 300.0


def _flush_print(message: str) -> None:
    print(message, flush=True)


def explain_http_error(detail: str) -> str:
    """Translate the two 403s this relay returns, which look identical to a caller.

    Both arrive as ``HTTP 403`` with a body, and both look exactly like a bad
    key until the body is read. One is Cloudflare rejecting the User-Agent; the
    other is this relay gating the key group to a time-of-day window, which no
    amount of retrying inside the window will fix and no amount of code will
    either.
    """
    if "1010" in detail:
        return "\n  -> Cloudflare rejected the User-Agent, not your key."
    if "可调用时段" in detail:
        return (
            "\n  -> Not an auth or code failure: this relay's key group is gated to a"
            "\n     time-of-day window. Re-run inside the window shown in the message."
        )
    return ""


def load_env(path: str | Path) -> dict[str, str]:
    """Parse a ``.env`` file into a dict.

    The leading BOM is stripped explicitly rather than by decoding as
    ``utf-8-sig``, so that every read in this codebase still goes through the
    one port in :mod:`zhrag.io_utils` with one encoding. Windows editors write
    the BOM freely, and left in place it turns the first variable's name into
    ``\\ufeffEmbedding_API_KEY`` -- which then reads as simply absent.
    """
    out: dict[str, str] = {}
    for raw in read_text(path).lstrip("﻿").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            out[name.strip()] = value.strip().strip("'\"")
    return out


def resolve_embeddings_url(base_url: str) -> str:
    """Append the right suffix to whatever form the base URL was written in.

    Both ``https://host`` and ``https://host/v1`` appear in the wild and in this
    project's own ``.env`` history; concatenating ``/v1/embeddings`` onto the
    second gives ``/v1/v1/embeddings`` and a 404 that reads like a dead host.
    """
    base = base_url.rstrip("/")
    return f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Where to send embedding requests, and as whom."""

    url: str
    key: str
    model: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> EmbeddingConfig:
        """Read the ``Embedding_*`` variables.

        The names are capitalised irregularly (``Embedding_API_KEY``, not
        ``EMBEDDING_API_KEY``) because that is how the provider's own dashboard
        emits them; matching it exactly avoids a silent fallback to an unset
        value. A missing variable raises here rather than producing a request
        with ``Bearer None``.
        """
        missing = [
            name
            for name in ("Embedding_API_KEY", "Embedding_BASE_URL", "Embedding_MODEL_NAME")
            if not env.get(name)
        ]
        if missing:
            raise ValueError(f".env is missing or empty for: {', '.join(missing)}")
        return cls(
            url=resolve_embeddings_url(env["Embedding_BASE_URL"]),
            key=env["Embedding_API_KEY"],
            model=env["Embedding_MODEL_NAME"],
        )


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    Deliberately long. The observed 429 on this relay is
    ``"当前分组上游负载已饱和"`` -- upstream saturation, not a per-key quota, so
    it clears on the provider's timescale rather than ours. A 1/2/4-second
    ladder gives up after seven seconds and throws away a run that is otherwise
    fine; this one spends up to ~4 minutes in total, still far cheaper than
    re-embedding a corpus. A server-sent ``Retry-After`` always wins, because
    the server knows something we do not.
    """
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    if retry_after is not None:
        return retry_after
    return min(60.0, 5.0 * float(2**attempt))


def _parse_retry_after(raw: str | None) -> float | None:
    """``Retry-After`` in delta-seconds form; the HTTP-date form is ignored."""
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    # isfinite, not just >= 0: float("inf") and float("nan") both parse, and an
    # infinite Retry-After would hand the run to time.sleep and never return.
    # The original used raw.isdigit(), which excluded them by accident.
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER)


Transport = Callable[[urllib.request.Request], bytes]


def _urlopen(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=300) as response:
        body: bytes = response.read()
        return body


@dataclass(frozen=True, slots=True)
class EmbeddingClient:
    """A retrying client for one OpenAI-compatible embeddings endpoint."""

    config: EmbeddingConfig
    retries: int = 7
    transport: Transport = _urlopen
    sleep: Callable[[float], None] = time.sleep
    #: Progress goes through here. The default flushes, because both scripts
    #: this was extracted from did: a quarter-hour run whose output is being
    #: teed to a file shows nothing at all until the buffer fills otherwise.
    log: Callable[[str], None] = _flush_print

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST once, retrying transient failures with :func:`backoff_seconds`."""
        request_body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.config.url,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.config.key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                parsed: dict[str, Any] = json.loads(self.transport(request).decode("utf-8"))
                return parsed
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                detail, retry_after = "", None
                if isinstance(exc, urllib.error.HTTPError):
                    detail = exc.read().decode("utf-8", errors="replace")[:ERROR_DETAIL_CHARS]
                    if exc.code not in RETRY_STATUS:
                        raise SystemExit(
                            f"! HTTP {exc.code} from {self.config.url}\n"
                            f"  {detail}{explain_http_error(detail)}"
                        ) from exc
                    retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                if attempt == self.retries - 1:
                    raise SystemExit(
                        f"! giving up after {self.retries} attempts: {exc} {detail}"
                    ) from exc
                wait = backoff_seconds(attempt, retry_after)
                self.log(f"    retry {attempt + 1}/{self.retries} in {wait:.0f}s ({exc})")
                self.sleep(wait)
        raise SystemExit("unreachable")

    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        """Embed in batches, invoking ``on_batch`` after each so callers can checkpoint.

        A full corpus pass is a quarter-hour of wall clock against a relay with
        undocumented rate limits. Accumulating everything in memory and writing
        once at the end means a failure at batch 124 of 125 discards all of it,
        so the caller gets each batch as it lands.
        """
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        out: list[list[float]] = []
        total = (len(texts) + batch - 1) // batch
        start = time.perf_counter()
        for i in range(0, len(texts), batch):
            sent = list(texts[i : i + batch])
            response = self.post({"model": self.config.model, "input": sent})
            raw_rows = response["data"]
            if len(raw_rows) != len(sent):
                # A short batch is worse than an error, because the caller pairs
                # these vectors positionally against its own id list: every id
                # after the gap takes the next id's vector and the mislabelled
                # row is then written to an append-only cache, where a resumed
                # run sees it as already done and never revisits it. The run
                # would surface this as a KeyError on the last id -- which reads
                # like a transient failure, not like corruption.
                raise SystemExit(
                    f"! batch at offset {i} sent {len(sent)} texts and got "
                    f"{len(raw_rows)} vectors.\n"
                    f"  Refusing to continue: pairing them positionally would write the\n"
                    f"  wrong vector under every subsequent id. Nothing has been cached\n"
                    f"  for this batch; re-run to resume from the last complete one."
                )
            indices = [row.get("index") for row in raw_rows]
            expected_indices = list(range(len(sent)))
            valid_indices = all(
                isinstance(index, int) and not isinstance(index, bool) for index in indices
            )
            if not valid_indices or sorted(indices) != expected_indices:
                # Checking only the row count would admit a response like
                # [index=0, index=0]: it has the right length but duplicates one
                # vector and omits another. A missing index used to default to 0,
                # which let the same corruption through when it occurred first.
                raise SystemExit(
                    f"! batch at offset {i} sent {len(sent)} texts but provider "
                    f"indices were {indices!r}; expected {expected_indices!r}.\n"
                    f"  Refusing to cache a duplicated, missing, non-integer, or "
                    f"out-of-range row."
                )
            rows = sorted(raw_rows, key=lambda row: row["index"])
            got = [row["embedding"] for row in rows]
            out.extend(got)
            if on_batch is not None:
                on_batch(i, got)
            done = i // batch + 1
            rate = (time.perf_counter() - start) / done
            self.log(
                f"  {label} {done:>4}/{total}  ({len(out):,} vectors, "
                f"~{rate * (total - done):.0f}s left)"
            )
        return out


class BatchEmbedder(Protocol):
    """The one method :func:`load_or_embed` needs, so tests can supply a fake."""

    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]: ...


def _meta_path(cache: Path) -> Path:
    return cache.with_suffix(cache.suffix + ".meta.json")


def _check_or_write_meta(cache: Path, model: str, prompt: str, log: Callable[[str], None]) -> None:
    """Refuse to mix vectors from two models, or two prompt conventions, in one cache.

    The cache is keyed on the item id alone. That is fine as long as one cache
    file holds one model's output under one prompt convention, and catastrophic
    otherwise: switching ``Embedding_MODEL_NAME`` would silently serve the old
    model's vectors and every downstream number would be wrong while looking
    entirely healthy. Rewriting the key to include the model would invalidate
    the 523 MB already on disk, so the check lives in a sidecar instead.

    A pre-existing cache with no sidecar is adopted, because that is the state
    this project's own caches were created in. The adoption records
    ``"assumed": true`` rather than writing a bare certificate: the guard exists
    to catch an unrecorded model switch, and a sidecar that cannot distinguish
    "observed" from "assumed" would launder exactly that into a fact. A later
    run under a different model still trips the drift check, so the guard works
    from the adoption point forward; it simply cannot speak for what came
    before, and now says so.
    """
    meta = _meta_path(cache)
    current: dict[str, object] = {"model": model, "prompt": prompt}
    if meta.exists():
        try:
            recorded = read_json(meta)
        except ValueError as exc:
            raise SystemExit(
                f"! {meta.name} is not readable JSON: {exc}\n"
                f"  It records which model wrote {cache.name}. Delete both and re-embed,"
                f" or restore the sidecar."
            ) from exc
        if not isinstance(recorded, dict):
            raise SystemExit(f"! {meta.name} should hold a JSON object, found {type(recorded)}.")
        drift = [(k, recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v]
        if drift:
            detail = "; ".join(
                f"{k}: cache has {was!r}, config says {now!r}" for k, was, now in drift
            )
            provenance = (
                "  Note the recorded settings were assumed at adoption, not observed.\n"
                if recorded.get("assumed")
                else ""
            )
            raise SystemExit(
                f"! {cache.name} was written under different settings.\n"
                f"  {detail}\n"
                f"{provenance}"
                f"  Vectors are keyed on id alone, so reusing this cache would mix them.\n"
                f"  Use a different cache path, or delete {cache.name} and {meta.name}."
            )
        return
    if cache.exists():
        log(f"  adopting existing {cache.name} as {model!r} output (assumed, not verified)")
        current["assumed"] = True
    write_json(meta, current)


def load_or_embed(
    cache: Path,
    items: Mapping[str, str],
    client: BatchEmbedder,
    *,
    model: str,
    prompt: str = "",
    batch: int = 16,
    log: Callable[[str], None] = print,
) -> dict[str, list[float]]:
    """Return a vector per item, embedding and caching only what is missing.

    ``prompt`` is prepended to every text before sending and is recorded in the
    cache sidecar. It is not merely cosmetic: Qwen3 is asymmetric, so the same
    string embeds to two different vectors depending on whether it arrived as a
    query or as a document, and one cache serving both would be wrong in a way
    no assertion downstream could catch.

    The cache is append-only and flushed per batch, so an interrupted run
    resumes where it stopped. Readers must therefore tolerate duplicate ids; the
    convention here, as everywhere in this codebase, is last-write-wins.
    """
    _check_or_write_meta(cache, model, prompt, log)

    cached: dict[str, list[float]] = {}
    if cache.exists():
        cached = {r["doc_id"]: r["embedding"] for r in read_jsonl(cache)}
        log(f"cache {cache.name}: {len(cached):,} vectors on disk")

    missing = [key for key in items if key not in cached]
    if missing:
        log(f"embedding {len(missing):,} new items ...")

        def flush(offset: int, vectors: list[list[float]]) -> None:
            batch_ids = missing[offset : offset + len(vectors)]
            cached.update(zip(batch_ids, vectors, strict=True))
            # Append only the new rows. Rewriting the whole file per batch is
            # O(n^2) in json.dumps, which at full corpus width costs more wall
            # clock than the network calls it is checkpointing.
            append_jsonl(cache, ({"doc_id": d, "embedding": cached[d]} for d in batch_ids))

        client.embed_all(
            [prompt + items[key] for key in missing],
            batch=batch,
            label=cache.stem[-10:],
            on_batch=flush,
        )
        log(f"cache {cache.name}: {len(cached):,} vectors")
    return {key: cached[key] for key in items}
