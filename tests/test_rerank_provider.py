"""Offline contract tests for the rerank provider and score cache."""

from __future__ import annotations

import email.message
import io
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from zhrag.io_utils import read_json, read_jsonl, read_text, write_text
from zhrag.providers.cache import (
    PairScore,
    append_pair_scores,
    load_pair_score_provenance,
    load_pair_scores,
    prepare_pair_score_cache,
    validate_pair_score_cache,
)
from zhrag.providers.http import JsonClient
from zhrag.providers.rerank import (
    DEFAULT_RERANK_INSTRUCTION,
    RerankClient,
    RerankConfig,
    estimate_rerank_tokens,
    resolve_rerank_url,
)

CONFIG = RerankConfig(
    api_key="secret",
    base_url="https://relay.example/v1",
    model="rerank-model",
)


def _response(results: Any, *, prompt_tokens: Any = 123) -> bytes:
    return json.dumps({"results": results, "usage": {"prompt_tokens": prompt_tokens}}).encode()


def _row(index: Any, score: Any, *, echo: str | None = None) -> dict[str, Any]:
    row = {"index": index, "relevance_score": score}
    if echo is not None:
        row["document"] = {"text": echo}
    return row


class Recorder:
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

    def client(self, *, retries: int = 7) -> RerankClient:
        http = JsonClient(
            url=CONFIG.endpoint,
            key=CONFIG.api_key,
            retries=retries,
            transport=self,
            sleep=self.slept.append,
            log=lambda _message: None,
        )
        return RerankClient(CONFIG, http)


class TestRerankConfig:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("https://host", "https://host/v1/rerank"),
            ("https://host/", "https://host/v1/rerank"),
            ("https://host/v1", "https://host/v1/rerank"),
            ("https://host/v1/", "https://host/v1/rerank"),
            ("https://host/v1/rerank", "https://host/v1/rerank"),
            (
                "https://host/v1/rerank?api-version=1#anchor",
                "https://host/v1/rerank?api-version=1#anchor",
            ),
            (
                "https://host/v1?tenant=x#anchor",
                "https://host/v1/rerank?tenant=x#anchor",
            ),
        ],
    )
    def test_resolves_supported_base_url_forms(self, base: str, expected: str) -> None:
        assert resolve_rerank_url(base) == expected

    def test_reads_irregularly_capitalised_names(self) -> None:
        config = RerankConfig.from_env(
            {
                "ReRank_API_KEY": "k",
                "ReRank_BASE_URL": "https://host/v1",
                "ReRank_MODEL_NAME": "m",
            }
        )
        assert (config.api_key, config.model) == ("k", "m")
        assert config.endpoint == "https://host/v1/rerank"

    @pytest.mark.parametrize("missing", ["ReRank_API_KEY", "ReRank_BASE_URL", "ReRank_MODEL_NAME"])
    def test_rejects_missing_or_empty_values(self, missing: str) -> None:
        env = {
            "ReRank_API_KEY": "k",
            "ReRank_BASE_URL": "https://host",
            "ReRank_MODEL_NAME": "m",
        }
        env[missing] = ""
        with pytest.raises(ValueError, match=missing):
            RerankConfig.from_env(env)

    def test_create_targets_the_resolved_rerank_endpoint(self) -> None:
        client = RerankClient.create(CONFIG)
        assert client.http.url == CONFIG.endpoint


class TestTokenEstimate:
    def test_is_one_scaffold_per_document(self) -> None:
        one = estimate_rerank_tokens("q", ["d"])
        two = estimate_rerank_tokens("q", ["d", "d"])
        assert two == 2 * one

    def test_longer_inputs_cost_more(self) -> None:
        assert estimate_rerank_tokens("q", ["short"]) < estimate_rerank_tokens(
            "long query" * 10, ["long document" * 100]
        )


class TestRerankClient:
    def test_sends_the_frozen_payload_and_user_agent(self) -> None:
        recorder = Recorder(_response([_row(0, 0.75)]))
        result = recorder.client().score("question", ["document"])

        request = recorder.requests[0]
        payload = json.loads(request.data or b"{}")
        assert request.full_url == CONFIG.endpoint
        assert request.get_header("User-agent", "").startswith("zhrag/")
        assert payload == {
            "model": CONFIG.model,
            "query": "question",
            "documents": ["document"],
            "instruction": DEFAULT_RERANK_INSTRUCTION,
            "top_n": 1,
            "return_documents": False,
        }
        assert result.scores == (0.75,)
        assert result.prompt_tokens == 123

    def test_reorders_results_and_discards_echoed_documents(self) -> None:
        recorder = Recorder(
            _response(
                [
                    _row(2, 0.3, echo="third text"),
                    _row(0, 0.9, echo="first text"),
                    _row(1, 0.6, echo="second text"),
                ]
            )
        )
        result = recorder.client().score("q", ["a", "b", "c"])
        assert result.scores == (0.9, 0.6, 0.3)
        assert not hasattr(result, "documents")

    def test_delegates_transient_retries_to_the_shared_transport(self) -> None:
        headers = email.message.Message()
        error = urllib.error.HTTPError(
            CONFIG.endpoint,
            429,
            "busy",
            headers,
            io.BytesIO(b'{"error":"busy"}'),
        )
        recorder = Recorder(error, _response([_row(0, 0.5)]))
        assert recorder.client().score("q", ["d"]).scores == (0.5,)
        assert recorder.slept == [5.0]

    def test_allows_missing_or_malformed_usage(self) -> None:
        recorder = Recorder(_response([_row(0, 0.5)], prompt_tokens=True))
        assert recorder.client().score("q", ["d"]).prompt_tokens is None

    def test_ignores_negative_prompt_tokens(self) -> None:
        recorder = Recorder(_response([_row(0, 0.5)], prompt_tokens=-1))
        assert recorder.client().score("q", ["d"]).prompt_tokens is None

    def test_rejects_empty_document_batches(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Recorder().client().score("q", [])

    def test_rejects_a_scalar_document_string_before_requesting(self) -> None:
        recorder = Recorder()
        with pytest.raises(TypeError, match="scalar string"):
            recorder.client().score("q", "document")
        assert recorder.requests == []

    def test_rejects_a_non_string_document_before_requesting(self) -> None:
        recorder = Recorder()
        with pytest.raises(TypeError, match="every document"):
            recorder.client().score("q", ["document", 1])  # type: ignore[list-item]
        assert recorder.requests == []

    def test_rejects_a_non_list_results_field(self) -> None:
        with pytest.raises(SystemExit, match="results is not a list"):
            Recorder(_response({})).client().score("q", ["d"])

    def test_rejects_a_short_response(self) -> None:
        with pytest.raises(SystemExit, match="1 results for 2 documents"):
            Recorder(_response([_row(0, 0.5)])).client().score("q", ["a", "b"])

    @pytest.mark.parametrize("index", [None, 0.0, True, "0"])
    def test_rejects_a_non_integer_index(self, index: Any) -> None:
        with pytest.raises(SystemExit, match="index is not an integer"):
            Recorder(_response([_row(index, 0.5)])).client().score("q", ["d"])

    @pytest.mark.parametrize(
        "rows",
        [
            [_row(0, 0.5), _row(0, 0.4)],
            [_row(0, 0.5), _row(2, 0.4)],
            [_row(-1, 0.5), _row(1, 0.4)],
        ],
    )
    def test_rejects_duplicate_or_out_of_range_indices(self, rows: list[dict[str, Any]]) -> None:
        with pytest.raises(SystemExit, match="duplicate or out-of-range"):
            Recorder(_response(rows)).client().score("q", ["a", "b"])

    @pytest.mark.parametrize("score", [None, True, "0.5"])
    def test_rejects_a_non_numeric_score(self, score: Any) -> None:
        with pytest.raises(SystemExit, match="not numeric"):
            Recorder(_response([_row(0, score)])).client().score("q", ["d"])

    @pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, 10**1000])
    def test_rejects_a_nonfinite_score(self, score: int | float) -> None:
        with pytest.raises(SystemExit, match="not finite"):
            Recorder(_response([_row(0, score)])).client().score("q", ["d"])


class TestPairScoreCache:
    def test_creates_and_revalidates_exact_provenance(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        provenance = {"model": "m", "instruction": "i", "candidate_run": "rrf"}
        prepare_pair_score_cache(cache, provenance)
        prepare_pair_score_cache(cache, provenance)
        assert read_json(Path(f"{cache}.meta.json")) == provenance

    def test_updates_empty_sidecar_when_provenance_changes(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        prepare_pair_score_cache(cache, {"model": "old"})
        prepare_pair_score_cache(cache, {"model": "new"})
        assert read_json(Path(f"{cache}.meta.json")) == {"model": "new"}

    def test_read_only_validation_does_not_create_or_rewrite_sidecars(self, tmp_path: Path) -> None:
        absent = tmp_path / "absent.jsonl"
        with pytest.raises(SystemExit, match="provenance is absent"):
            validate_pair_score_cache(absent, {"model": "m"})
        assert not Path(f"{absent}.meta.json").exists()

        cache = tmp_path / "scores.jsonl"
        prepare_pair_score_cache(cache, {"model": "old"})
        sidecar = Path(f"{cache}.meta.json")
        before = read_text(sidecar)
        with pytest.raises(SystemExit, match="metadata drift"):
            validate_pair_score_cache(cache, {"model": "new"})
        assert read_text(sidecar) == before
        assert load_pair_score_provenance(cache) == {"model": "old"}

    def test_rejects_provenance_drift_once_scores_exist(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        prepare_pair_score_cache(cache, {"model": "old"})
        append_pair_scores(cache, [PairScore("q", "d", 0.5)])
        with pytest.raises(SystemExit, match="metadata drift"):
            prepare_pair_score_cache(cache, {"model": "new"})

    def test_refuses_to_adopt_nonempty_scores_without_a_sidecar(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        write_text(cache, '{"query_id":"q","doc_id":"d","score":0.5}\n')
        with pytest.raises(SystemExit, match="without provenance"):
            prepare_pair_score_cache(cache, {"model": "m"})

    def test_appends_only_ids_and_scores(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        append_pair_scores(cache, [PairScore("q", "d", 0.75)])
        assert list(read_jsonl(cache)) == [{"query_id": "q", "doc_id": "d", "score": 0.75}]

    def test_load_is_last_write_wins(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        append_pair_scores(
            cache,
            [PairScore("q", "d", 0.25), PairScore("q", "d", 0.75)],
        )
        assert load_pair_scores(cache) == {("q", "d"): 0.75}

    def test_missing_cache_is_empty(self, tmp_path: Path) -> None:
        assert load_pair_scores(tmp_path / "absent.jsonl") == {}

    def test_rejects_a_non_object_cache_row(self, tmp_path: Path) -> None:
        cache = tmp_path / "scores.jsonl"
        write_text(cache, "[]\n")
        with pytest.raises(SystemExit, match="expected an object"):
            load_pair_scores(cache)

    @pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
    def test_refuses_to_append_nonfinite_scores(self, tmp_path: Path, score: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            append_pair_scores(tmp_path / "scores.jsonl", [PairScore("q", "d", score)])
