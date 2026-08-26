"""Tests for the embedding provider client.

None of these touch the network. That is the point of the module's shape: the
transport, the clock and the logger are all injected, so the parts that decide
correctness -- which HTTP codes are retried, how long the backoff waits, how the
cache merges, whether a model switch is caught -- are testable without spending
money or depending on a relay being up.

The behaviours pinned here were each learned the expensive way; see the module
docstring for the three provider quirks.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from zhrag.io_utils import read_json, read_jsonl, write_json, write_text
from zhrag.providers.embedding import (
    MAX_RETRY_AFTER,
    RETRY_STATUS,
    EmbeddingClient,
    EmbeddingConfig,
    backoff_seconds,
    load_embedding_provenance,
    load_env,
    load_or_embed,
    resolve_embeddings_url,
    validate_embedding_cache,
)

CONFIG = EmbeddingConfig(url="https://relay.example/v1/embeddings", key="k", model="test-model")


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        CONFIG.url, code, "boom", headers, io.BytesIO(b'{"error":"boom"}')
    )


def _ok(vectors: Sequence[Sequence[float]], *, shuffled: bool = False) -> bytes:
    data = [{"index": i, "embedding": list(v)} for i, v in enumerate(vectors)]
    if shuffled:
        data.reverse()
    return json.dumps({"data": data}).encode("utf-8")


class Recorder:
    """A fake transport that replays a scripted sequence of outcomes."""

    def __init__(self, *outcomes: bytes | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[urllib.request.Request] = []
        self.slept: list[float] = []

    def __call__(self, request: urllib.request.Request) -> bytes:
        self.requests.append(request)
        outcome = self.outcomes[min(len(self.requests) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def client(self, **kwargs: Any) -> EmbeddingClient:
        return EmbeddingClient(
            config=CONFIG,
            transport=self,
            sleep=self.slept.append,
            log=lambda _msg: None,
            **kwargs,
        )


class TestLoadEnv:
    def test_parses_comments_blanks_and_quotes(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        write_text(path, "# comment\n\nA=1\nB='two'\nC=\"three\"\n")
        assert load_env(path) == {"A": "1", "B": "two", "C": "three"}

    def test_a_value_may_contain_equals(self, tmp_path: Path) -> None:
        path = tmp_path / ".env"
        write_text(path, "URL=https://host/v1?a=b\n")
        assert load_env(path)["URL"] == "https://host/v1?a=b"

    def test_strips_the_byte_order_mark(self, tmp_path: Path) -> None:
        """A BOM renames the first variable to '﻿Embedding_API_KEY'.

        Windows editors add it silently, and the symptom is an authentication
        failure rather than a parse error -- the variable simply reads as unset.
        """
        path = tmp_path / ".env"
        write_text(path, "﻿Embedding_API_KEY=secret\n")
        assert load_env(path) == {"Embedding_API_KEY": "secret"}


class TestUrlResolution:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("https://host", "https://host/v1/embeddings"),
            ("https://host/", "https://host/v1/embeddings"),
            ("https://host/v1", "https://host/v1/embeddings"),
            ("https://host/v1/", "https://host/v1/embeddings"),
        ],
    )
    def test_both_written_forms_reach_the_same_endpoint(self, base: str, expected: str) -> None:
        """Naive concatenation onto a '/v1' base gives /v1/v1/embeddings and a
        404 that reads exactly like a dead host."""
        assert resolve_embeddings_url(base) == expected


class TestConfigFromEnv:
    def test_reads_the_irregularly_capitalised_names(self) -> None:
        config = EmbeddingConfig.from_env(
            {
                "Embedding_API_KEY": "k",
                "Embedding_BASE_URL": "https://host/v1",
                "Embedding_MODEL_NAME": "m",
            }
        )
        assert (config.key, config.model) == ("k", "m")
        assert config.url == "https://host/v1/embeddings"

    @pytest.mark.parametrize("missing", ["Embedding_API_KEY", "Embedding_BASE_URL"])
    def test_a_missing_variable_raises_rather_than_sending_bearer_none(self, missing: str) -> None:
        env = {
            "Embedding_API_KEY": "k",
            "Embedding_BASE_URL": "https://host",
            "Embedding_MODEL_NAME": "m",
        }
        del env[missing]
        with pytest.raises(ValueError, match=missing):
            EmbeddingConfig.from_env(env)

    def test_an_empty_value_counts_as_missing(self) -> None:
        with pytest.raises(ValueError, match="Embedding_API_KEY"):
            EmbeddingConfig.from_env(
                {
                    "Embedding_API_KEY": "",
                    "Embedding_BASE_URL": "https://host",
                    "Embedding_MODEL_NAME": "m",
                }
            )


class TestBackoff:
    def test_ladder_doubles_and_caps_at_sixty(self) -> None:
        waits = [backoff_seconds(i) for i in range(8)]
        assert waits == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0]

    def test_total_wait_is_minutes_not_seconds(self) -> None:
        """The 1/2/4-second ladder gives up in 7s and discards a good run.

        The observed 429 here is upstream saturation, which clears on the
        provider's schedule; waiting minutes is far cheaper than re-embedding.
        """
        assert sum(backoff_seconds(i) for i in range(7)) > 180

    def test_server_retry_after_wins(self) -> None:
        assert backoff_seconds(0, retry_after=3.0) == 3.0
        assert backoff_seconds(6, retry_after=1.0) == 1.0

    def test_rejects_negative_attempt(self) -> None:
        with pytest.raises(ValueError, match="attempt must be"):
            backoff_seconds(-1)


class TestClientRetries:
    def test_sends_a_user_agent(self) -> None:
        """Cloudflare 403s the stdlib default UA with error 1010, which is
        indistinguishable from a bad key."""
        recorder = Recorder(_ok([[1.0]]))
        recorder.client().post({"input": ["x"]})
        assert recorder.requests[0].get_header("User-agent", "").startswith("zhrag/")

    @pytest.mark.parametrize("code", sorted(RETRY_STATUS))
    def test_transient_codes_are_retried(self, code: int) -> None:
        recorder = Recorder(_http_error(code), _ok([[1.0]]))
        assert recorder.client().post({}) == {"data": [{"index": 0, "embedding": [1.0]}]}
        assert recorder.slept == [5.0]

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_permanent_codes_are_not_retried(self, code: int) -> None:
        """Retrying a malformed request just spends the same money seven times."""
        recorder = Recorder(_http_error(code))
        with pytest.raises(SystemExit, match=f"HTTP {code}"):
            recorder.client().post({})
        assert len(recorder.requests) == 1
        assert recorder.slept == []

    def test_gives_up_after_the_retry_budget(self) -> None:
        recorder = Recorder(_http_error(429))
        with pytest.raises(SystemExit, match="giving up after 3"):
            recorder.client(retries=3).post({})
        assert len(recorder.requests) == 3
        assert recorder.slept == [5.0, 10.0]

    def test_honours_a_numeric_retry_after(self) -> None:
        recorder = Recorder(_http_error(429, retry_after="2"), _ok([[1.0]]))
        recorder.client().post({})
        assert recorder.slept == [2.0]

    def test_ignores_an_http_date_retry_after(self) -> None:
        """The date form is unparsed rather than crashing the run mid-corpus."""
        recorder = Recorder(
            _http_error(429, retry_after="Wed, 21 Oct 2026 07:28:00 GMT"), _ok([[1.0]])
        )
        recorder.client().post({})
        assert recorder.slept == [5.0]

    def test_caps_an_excessive_retry_after(self) -> None:
        recorder = Recorder(_http_error(429, retry_after="86400"), _ok([[1.0]]))
        recorder.client().post({})
        assert recorder.slept == [MAX_RETRY_AFTER]

    @pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
    def test_ignores_a_nonfinite_or_negative_retry_after(self, value: str) -> None:
        recorder = Recorder(_http_error(429, retry_after=value), _ok([[1.0]]))
        recorder.client().post({})
        assert recorder.slept == [5.0]

    def test_network_errors_are_retried_too(self) -> None:
        recorder = Recorder(urllib.error.URLError("connection reset"), _ok([[1.0]]))
        recorder.client().post({})
        assert len(recorder.requests) == 2


class TestEmbedAll:
    def test_batches_and_concatenates(self) -> None:
        recorder = Recorder(_ok([[1.0], [2.0]]), _ok([[3.0]]))
        got = recorder.client().embed_all(["a", "b", "c"], batch=2)
        assert got == [[1.0], [2.0], [3.0]]
        assert len(recorder.requests) == 2

    def test_reorders_a_permuted_response_by_index(self) -> None:
        """The API is not required to preserve input order, and a silently
        permuted batch would mislabel every vector in it."""
        recorder = Recorder(_ok([[1.0], [2.0], [3.0]], shuffled=True))
        assert recorder.client().embed_all(["a", "b", "c"], batch=3) == [[1.0], [2.0], [3.0]]

    def test_refuses_a_short_response_before_checkpointing(self) -> None:
        recorder = Recorder(_ok([[1.0], [2.0]]))
        checkpointed: list[list[list[float]]] = []
        with pytest.raises(SystemExit, match="sent 3 texts and got 2 vectors"):
            recorder.client().embed_all(
                ["a", "b", "c"],
                batch=3,
                on_batch=lambda _i, vectors: checkpointed.append(vectors),
            )
        assert checkpointed == []

    def test_refuses_duplicate_indices_even_when_the_row_count_matches(self) -> None:
        body = json.dumps(
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [2.0]},
                ]
            }
        ).encode("utf-8")
        with pytest.raises(SystemExit, match=r"indices were \[0, 0\]"):
            Recorder(body).client().embed_all(["a", "b"], batch=2)

    @pytest.mark.parametrize(
        "bad_index",
        [None, 0.0, True],
        ids=["missing", "float", "boolean"],
    )
    def test_refuses_a_missing_or_noninteger_index(self, bad_index: object) -> None:
        first: dict[str, object] = {"embedding": [1.0]}
        if bad_index is not None:
            first["index"] = bad_index
        body = json.dumps(
            {
                "data": [
                    first,
                    {"index": 1, "embedding": [2.0]},
                ]
            }
        ).encode("utf-8")
        with pytest.raises(SystemExit, match="missing, non-integer"):
            Recorder(body).client().embed_all(["a", "b"], batch=2)

    def test_on_batch_fires_per_batch_with_its_offset(self) -> None:
        """A failure at batch 124 of 125 must not discard the first 123."""
        recorder = Recorder(_ok([[1.0], [2.0]]), _ok([[3.0]]))
        seen: list[tuple[int, int]] = []
        recorder.client().embed_all(
            ["a", "b", "c"], batch=2, on_batch=lambda i, v: seen.append((i, len(v)))
        )
        assert seen == [(0, 2), (2, 1)]

    def test_rejects_a_nonpositive_batch(self) -> None:
        with pytest.raises(ValueError, match="batch must be"):
            Recorder().client().embed_all(["a"], batch=0)


class FakeEmbedder:
    """Returns a deterministic vector per text and records what it was asked."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        self.seen.extend(texts)
        vectors = [[float(len(t))] for t in texts]
        for i in range(0, len(vectors), batch):
            if on_batch is not None:
                on_batch(i, vectors[i : i + batch])
        return vectors


class TestLoadOrEmbed:
    def test_embeds_everything_on_a_cold_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        embedder = FakeEmbedder()
        got = load_or_embed(cache, {"a": "xx", "b": "y"}, embedder, model="m", log=lambda _m: None)
        assert got == {"a": [2.0], "b": [1.0]}
        assert [r["doc_id"] for r in read_jsonl(cache)] == ["a", "b"]

    def test_only_missing_items_are_sent(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        first = FakeEmbedder()
        load_or_embed(cache, {"a": "xx"}, first, model="m", log=lambda _m: None)
        second = FakeEmbedder()
        got = load_or_embed(cache, {"a": "xx", "b": "yyy"}, second, model="m", log=lambda _m: None)
        assert second.seen == ["yyy"]  # 'a' came from disk
        assert got == {"a": [2.0], "b": [3.0]}

    def test_the_prompt_is_applied_to_the_sent_text_not_the_key(self, tmp_path: Path) -> None:
        """Qwen3 is asymmetric: the same string is a different vector as a query
        than as a document, so the prefix must reach the API while the cache key
        stays the stable id."""
        cache = tmp_path / "c.jsonl"
        embedder = FakeEmbedder()
        load_or_embed(cache, {"q1": "ab"}, embedder, model="m", prompt="P:", log=lambda _m: None)
        assert embedder.seen == ["P:ab"]
        assert [r["doc_id"] for r in read_jsonl(cache)] == ["q1"]

    def test_duplicate_rows_resolve_last_write_wins(self, tmp_path: Path) -> None:
        """The cache is append-only and resumable, so ids can repeat."""
        cache = tmp_path / "c.jsonl"
        write_json(cache.with_suffix(".jsonl.meta.json"), {"model": "m", "prompt": ""})
        write_text(
            cache,
            '{"doc_id": "a", "embedding": [1.0]}\n{"doc_id": "a", "embedding": [9.0]}\n',
        )
        got = load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)
        assert got == {"a": [9.0]}

    def test_returns_only_what_was_asked_for(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        write_json(cache.with_suffix(".jsonl.meta.json"), {"model": "m", "prompt": ""})
        write_text(
            cache,
            '{"doc_id": "a", "embedding": [1.0]}\n{"doc_id": "z", "embedding": [2.0]}\n',
        )
        got = load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)
        assert got == {"a": [1.0]}


class TestCacheSidecar:
    def test_read_only_validation_loads_exact_model_and_prompt(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        write_json(cache.with_suffix(".jsonl.meta.json"), {"model": "m", "prompt": "P:"})

        assert load_embedding_provenance(cache) == {"model": "m", "prompt": "P:"}
        validate_embedding_cache(cache, model="m", prompt="P:")

    def test_read_only_validation_rejects_absent_or_drifted_provenance(
        self, tmp_path: Path
    ) -> None:
        cache = tmp_path / "c.jsonl"
        with pytest.raises(SystemExit, match="provenance is absent"):
            validate_embedding_cache(cache, model="m", prompt="")
        assert not cache.with_suffix(".jsonl.meta.json").exists()

        write_json(cache.with_suffix(".jsonl.meta.json"), {"model": "m", "prompt": "old"})
        with pytest.raises(SystemExit, match="different settings"):
            validate_embedding_cache(cache, model="m", prompt="new")

    def test_a_model_switch_is_refused_rather_than_silently_mixed(self, tmp_path: Path) -> None:
        """The killer case: vectors are keyed on id alone, so a changed
        Embedding_MODEL_NAME would serve the previous model's output and every
        downstream number would be wrong while looking entirely healthy."""
        cache = tmp_path / "c.jsonl"
        load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="old", log=lambda _m: None)
        with pytest.raises(SystemExit, match="written under different settings"):
            load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="new", log=lambda _m: None)

    def test_a_prompt_switch_is_refused_too(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)
        with pytest.raises(SystemExit, match="prompt"):
            load_or_embed(
                cache, {"a": "x"}, FakeEmbedder(), model="m", prompt="P:", log=lambda _m: None
            )

    def test_an_existing_cache_without_a_sidecar_is_adopted_loudly(self, tmp_path: Path) -> None:
        """The state the project's own 523 MB caches were created in. Adopting
        them is necessary; doing it silently is not."""
        cache = tmp_path / "c.jsonl"
        write_text(cache, '{"doc_id": "a", "embedding": [1.0]}\n')
        messages: list[str] = []
        load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=messages.append)
        assert any("adopting" in m for m in messages)
        assert cache.with_suffix(".jsonl.meta.json").exists()
        meta = read_json(cache.with_suffix(".jsonl.meta.json"))
        assert meta == {"model": "m", "prompt": "", "assumed": True}

    def test_a_corrupt_sidecar_is_refused(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        write_text(cache, '{"doc_id": "a", "embedding": [1.0]}\n')
        write_text(cache.with_suffix(".jsonl.meta.json"), "{not json")
        with pytest.raises(SystemExit, match="not readable JSON"):
            load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)

    def test_a_non_object_sidecar_is_refused(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        write_text(cache, '{"doc_id": "a", "embedding": [1.0]}\n')
        write_text(cache.with_suffix(".jsonl.meta.json"), "[]")
        with pytest.raises(SystemExit, match="should hold a JSON object"):
            load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)

    def test_a_matching_sidecar_is_silent(self, tmp_path: Path) -> None:
        cache = tmp_path / "c.jsonl"
        load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=lambda _m: None)
        messages: list[str] = []
        load_or_embed(cache, {"a": "x"}, FakeEmbedder(), model="m", log=messages.append)
        assert not any("adopting" in m for m in messages)
