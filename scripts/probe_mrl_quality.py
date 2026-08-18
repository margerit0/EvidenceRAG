"""Measure what slicing a 4096-d embedding actually costs, on real labelled data.

    uv run python scripts/probe_mrl_quality.py            # 2,000-doc probe
    uv run python scripts/probe_mrl_quality.py --docs 5681  # full eval corpus

``scripts/verify_embedding_api.py`` established that this provider implements
``dimensions=n`` as a prefix slice + renormalise, which is also what Qwen3's
own ``truncate_dim`` does (``modules.json`` slices before ``Normalize``, and
scalar normalisation commutes with slice-then-renormalise). That answers *how*
truncation is implemented. It says nothing about what truncation **costs**.

Two measurements that do say something, neither of which is the L2 energy
profile:

1. **Per-dimension variance across documents.** Magnitude is the wrong signal --
   after L2 normalisation ``||v[:n]|| ~ sqrt(n/N)`` almost by construction, so a
   "front-loading ratio" near 1.0 is an artefact rather than a finding. What
   distinguishes an MRL-trained model is whether early dimensions *discriminate*
   between documents, and that is variance across a corpus, not norm within one
   vector.

2. **Retrieval quality per width.** The only measurement whose outcome changes a
   decision. Reported against a stated ``corpus_size`` because R@1 is meaningless
   without one -- see the saturation table in README.

The whole sweep runs off **one** embedding pass: documents are embedded once at
full width and cached, and every narrower row is a client-side slice of that
cache. That is not a shortcut, it is the finding from ``verify_embedding_api.py``
being cashed in -- and re-running this script costs nothing.

Held to ``questanswer_1doc`` so R@1 is not capped by gold arity (a 3-gold query
cannot exceed 1/3 at k=1; see :func:`zhrag.eval.metrics.recall_at_k`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import Query, sample_corpus
from zhrag.eval.metrics import (
    bootstrap_ci,
    holm_bonferroni,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_test,
    recall_at_k,
)
from zhrag.io_utils import append_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
CACHE = EXPANDED / "emb_cache_4096.jsonl"
#: Queries need their own cache because Qwen3 is asymmetric: a query carries the
#: instruct prefix and a document carries none, so the *same* string embeds to
#: two different vectors depending on which side it is on. One shared cache
#: keyed on raw text would silently serve the wrong one.
QUERY_CACHE = EXPANDED / "emb_cache_queries_4096.jsonl"

#: Widths to sweep. 4096 is the model's native width; 64 is the smallest the
#: Qwen3 MRL whitelist mentions. Going below the useful range on purpose: a
#: curve that only degrades at the far end is far more convincing than three
#: points that all look fine.
DIMS = (4096, 2048, 1024, 512, 256, 128, 64)

#: Asymmetric by design. Qwen3's document prompt is the empty string; adding a
#: prefix to both sides silently costs several points of R@1 and raises nothing.
#: Instruction in English even though the corpus is Chinese -- Qwen's own advice,
#: because the training-time instructions were English.
QUERY_PROMPT = (
    "Instruct: Given a Chinese question, retrieve the news passage that answers it\nQuery:"
)


def _load_env() -> dict[str, str]:
    out: dict[str, str] = {}
    raw_text = (ROOT / ".env").read_text(encoding="utf-8-sig")
    for raw in raw_text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            out[name.strip()] = value.strip().strip("'\"")
    return out


def _post(url: str, key: str, payload: dict[str, Any], *, retries: int = 7) -> dict[str, Any]:
    """POST with exponential backoff.

    Backoff is deliberately long. The observed 429 on this relay is
    ``"当前分组上游负载已饱和"`` -- upstream saturation, not a per-key quota, so
    it clears on the provider's timescale rather than ours. A 1/2/4-second
    ladder gives up after 7 seconds and throws away a run that is otherwise
    fine; this one waits up to ~4 minutes in total, which is still far cheaper
    than re-embedding. ``Retry-After`` is honoured when sent.
    """
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Cloudflare 403s the stdlib default UA with error 1010.
                "User-Agent": "zhrag/0.1 (+https://github.com/margerit0/zhrag)",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            detail, retry_after = "", None
            if isinstance(exc, urllib.error.HTTPError):
                detail = exc.read().decode("utf-8", errors="replace")[:200]
                if exc.code not in (429, 500, 502, 503, 504):
                    raise SystemExit(f"! HTTP {exc.code}: {detail}") from exc
                raw = exc.headers.get("Retry-After")
                retry_after = float(raw) if raw and raw.isdigit() else None
            if attempt == retries - 1:
                raise SystemExit(f"! giving up after {retries} attempts: {exc} {detail}") from exc
            wait = retry_after if retry_after is not None else min(60.0, 5.0 * 2**attempt)
            print(f"    retry {attempt + 1}/{retries} in {wait:.0f}s ({exc})", flush=True)
            time.sleep(wait)
    raise SystemExit("unreachable")


def _l2(vector: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in vector))


def _slice(vector: list[float], dim: int) -> list[float]:
    """Prefix slice + L2 renormalise -- identical to Qwen3's own truncate_dim."""
    if dim >= len(vector):
        return vector
    prefix = vector[:dim]
    scale = _l2(prefix)
    return [x / scale for x in prefix]


def _embed_all(
    url: str,
    key: str,
    model: str,
    texts: list[str],
    *,
    batch: int,
    label: str,
    on_batch: Callable[[int, list[list[float]]], None] | None = None,
) -> list[list[float]]:
    """Embed in batches, invoking ``on_batch`` after each so callers can checkpoint.

    A full corpus pass is a quarter-hour of wall clock against a relay with
    undocumented rate limits. Accumulating everything in memory and writing once
    at the end means a failure at batch 124 of 125 discards all of it, so the
    caller gets each batch as it lands.
    """
    out: list[list[float]] = []
    total = (len(texts) + batch - 1) // batch
    start = time.perf_counter()
    for i in range(0, len(texts), batch):
        response = _post(url, key, {"model": model, "input": texts[i : i + batch]})
        rows = sorted(response["data"], key=lambda d: d.get("index", 0))
        got = [row["embedding"] for row in rows]
        out.extend(got)
        if on_batch is not None:
            on_batch(i, got)
        done = i // batch + 1
        rate = (time.perf_counter() - start) / done
        print(
            f"  {label} {done:>4}/{total}  ({len(out):,} vectors, "
            f"~{rate * (total - done):.0f}s left)",
            flush=True,
        )
    return out


def load_or_embed(
    url: str,
    key: str,
    model: str,
    items: dict[str, str],
    *,
    batch: int,
    cache: Path = CACHE,
) -> dict[str, list[float]]:
    """Embed ``{id: text}`` once at full width; the cache makes re-runs free.

    The cache is flushed as batches complete, so an interrupted run resumes
    where it stopped instead of starting over.
    """
    cached: dict[str, list[float]] = {}
    if cache.exists():
        # Last-write-wins: the cache is append-only, so a resumed run may hold
        # more than one row per id.
        cached = {r["doc_id"]: r["embedding"] for r in read_jsonl(cache)}
        print(f"cache {cache.name}: {len(cached):,} vectors on disk")

    missing = [d for d in items if d not in cached]
    if missing:
        print(f"embedding {len(missing):,} new items at 4096-d ...")

        def flush(offset: int, vectors: list[list[float]]) -> None:
            batch_ids = missing[offset : offset + len(vectors)]
            cached.update(zip(batch_ids, vectors, strict=True))
            # Append only the new rows. Rewriting the whole file per batch is
            # O(n^2) in json.dumps, which at full corpus width costs more wall
            # clock than the network calls it is checkpointing.
            append_jsonl(cache, ({"doc_id": d, "embedding": cached[d]} for d in batch_ids))

        _embed_all(
            url,
            key,
            model,
            [items[d] for d in missing],
            batch=batch,
            label=cache.stem[-10:],
            on_batch=flush,
        )
        print(f"cache {cache.name}: {len(cached):,} vectors")
    return {d: cached[d] for d in items}


def report_variance(vectors: NDArray[np.float32]) -> None:
    """Do early dimensions discriminate between documents more than late ones?"""
    print("\n\n## 1. Per-dimension variance across documents\n")
    print("   Magnitude is the wrong probe: after L2 normalisation ||v[:n]|| ~ sqrt(n/N)")
    print("   holds almost by construction. Variance measures whether a dimension")
    print("   actually separates documents -- which is what MRL training targets.\n")

    variances = vectors.var(axis=0)
    width = variances.size
    band = width // 8
    overall = float(variances.mean())

    print(f"   {'dim range':>13} {'mean variance':>15} {'vs overall':>12}")
    print(f"   {'-' * 13} {'-' * 15} {'-' * 12}")
    for i in range(8):
        lo, hi = i * band, (i + 1) * band
        mean_var = float(variances[lo:hi].mean())
        print(f"   {f'{lo:,}-{hi:,}':>13} {mean_var:>15.3e} {mean_var / overall:>11.3f}x")

    first, last = float(variances[:band].mean()), float(variances[-band:].mean())
    print(f"\n   first {band}/last {band} variance ratio = {first / last:.3f}")
    if first / last > 1.5:
        print("   => early dimensions carry more discriminative signal (MRL-consistent).")
    elif first / last > 0.9:
        print("   => variance is flat across dimensions. Information is NOT ordered by")
        print("      index, so truncation discards signal roughly in proportion to width.")
        print("      Whether that costs accuracy is an empirical question -- see table 2.")
    else:
        print("   => later dimensions carry MORE variance than earlier ones.")


def _rank(
    queries: NDArray[np.float32], docs: NDArray[np.float32], doc_ids: list[str], k: int = 10
) -> list[list[str]]:
    """Top-k document ids per query, by cosine on unit-norm slices.

    One matmul rather than a Python loop. At 800 x 5,681 x 4,096 this is ~19
    billion multiply-adds; the pure-Python version of the same sweep would run
    for over half an hour and hold 745 MB of float objects.
    """
    sims = queries @ docs.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, 1), axis=1), 1)
    return [[doc_ids[i] for i in row] for row in ordered]


def _slice_matrix(matrix: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Prefix slice + row-wise L2 renormalise -- Qwen3's own truncate_dim."""
    if dim >= matrix.shape[1]:
        return matrix
    prefix = matrix[:, :dim]
    return prefix / np.linalg.norm(prefix, axis=1, keepdims=True)


def report_retrieval(
    doc_matrix: NDArray[np.float32],
    doc_ids: list[str],
    query_matrix: NDArray[np.float32],
    queries: list[Query],
) -> None:
    print("\n\n## 2. Retrieval quality per width (the measurement that decides)\n")
    print(f"   corpus_size = {len(doc_ids):,} documents, {len(queries):,} queries")
    print("   task = questanswer_1doc only, so R@1 is not capped by gold arity.")
    print("   R@1 carries a 95% bootstrap CI; p is a one-sided paired bootstrap for")
    print("   'dim=4096 beats this row', Holm-corrected across the whole family.\n")
    print(
        f"   {'dim':>6} {'R@1 [95% CI]':>22} {'MRR@10':>8} {'nDCG@10':>8} "
        f"{'vs 4096':>9} {'p(Holm)':>9} {'MB':>7}"
    )
    print(f"   {'-' * 6} {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 7}")

    rows: dict[int, dict[str, Any]] = {}
    per_query: dict[int, list[float]] = {}

    for dim in DIMS:
        ranked = _rank(_slice_matrix(query_matrix, dim), _slice_matrix(doc_matrix, dim), doc_ids)
        pairs = list(zip(ranked, queries, strict=True))
        n = len(pairs)
        # Keep the per-query vector: the paired test needs the same queries in
        # both arms, and that pairing is what makes it sensitive enough to
        # resolve differences this small.
        per_query[dim] = [recall_at_k(r, q.gold_doc_ids, 1) for r, q in pairs]
        rows[dim] = {
            "ci": bootstrap_ci(per_query[dim]),
            "mrr": sum(mrr_at_k(r, q.gold_doc_ids, 10) for r, q in pairs) / n,
            "ndcg": sum(ndcg_at_k(r, q.gold_doc_ids, 10) for r, q in pairs) / n,
            "mb": len(doc_ids) * dim * 4 / 1e6,
        }

    full = DIMS[0]
    raw_p = {
        str(dim): paired_bootstrap_test(per_query[dim], per_query[full])
        for dim in DIMS
        if dim != full
    }
    adjusted = holm_bonferroni(raw_p)

    for dim in DIMS:
        row, ci = rows[dim], rows[dim]["ci"]
        delta = (ci.mean - rows[full]["ci"].mean) * 100
        if dim == full:
            verdict = "baseline"
        else:
            p, reject = adjusted[str(dim)]
            verdict = f"{p:.3f}{'*' if reject else ''}"
        print(
            f"   {dim:>6} {ci.mean:>7.1%} [{ci.low:.1%}, {ci.high:.1%}] {row['mrr']:>8.3f} "
            f"{row['ndcg']:>8.3f} {delta:>+8.2f}pp {verdict:>9} {row['mb']:>7.1f}"
        )

    print("\n   * = significant at family-wise alpha=0.05 after Holm correction.")
    significant = [d for d in DIMS if d != full and adjusted[str(d)][1]]
    if not significant:
        print("   => NO width is significantly worse than 4096 on this query set.")
        print(f"      Read that as 'not resolvable at n={len(queries)}', not 'proven equal'.")
    else:
        print(f"   => significantly worse than {full}: {significant}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=2000, help="corpus size (gold always kept)")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if not (EXPANDED / "corpus.jsonl").exists():
        print(f"! {EXPANDED} not built -- run scripts/build_eval_corpus.py first")
        return 1

    env = _load_env()
    key, model = env["Embedding_API_KEY"], env["Embedding_MODEL_NAME"]
    base = env["Embedding_BASE_URL"].rstrip("/")
    url = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"

    pool = {r["doc_id"]: r["text"] for r in read_jsonl(EXPANDED / "corpus.jsonl")}
    all_queries = [
        Query(
            query_id=r["query_id"],
            question=r["question"],
            answer=r["answer"],
            gold_doc_ids=tuple(r["gold_doc_ids"]),
            task=r["task"],
        )
        for r in read_jsonl(EXPANDED / "qrels.jsonl")
        if r["task"] == "questanswer_1doc"
    ]
    queries = all_queries[: args.queries]
    corpus = sample_corpus(pool, queries, size=args.docs)
    print(f"corpus {len(corpus):,} docs | {len(queries):,} queries | model {model}\n")

    doc_vectors = load_or_embed(url, key, model, corpus, batch=args.batch)

    # Queries are cached on the *prefixed* text, which is what was actually sent.
    prefixed = {q.query_id: QUERY_PROMPT + q.question for q in queries}
    query_map = load_or_embed(url, key, model, prefixed, batch=args.batch, cache=QUERY_CACHE)

    # float32, not float64: storage error is ~1e-7 relative, five orders of
    # magnitude below this provider's own reproducibility floor of cos 0.99993
    # (see scripts/verify_embedding_api.py). Doubling the width would record
    # noise more precisely, nothing else. Also halves 5,681x4096 to 93 MB.
    doc_ids = list(doc_vectors)
    doc_matrix = np.asarray([doc_vectors[d] for d in doc_ids], dtype=np.float32)
    query_matrix = np.asarray([query_map[q.query_id] for q in queries], dtype=np.float32)
    print(f"matrices: docs {doc_matrix.shape} queries {query_matrix.shape} float32")

    report_variance(doc_matrix)
    report_retrieval(doc_matrix, doc_ids, query_matrix, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
