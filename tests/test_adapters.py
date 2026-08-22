"""Adapter tests: provider seams stay hermetic and instruction-explicit."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest

from zhrag.lexical import build_sparse_index
from zhrag.providers.rerank import RerankClient, RerankConfig
from zhrag.retrieval import (
    DenseQueryAdapter,
    OnlineSettings,
    RerankAdapter,
    SparseQueryAdapter,
)


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors if vectors is not None else [[1.0, 0.0]]
        self.calls: list[tuple[list[str], int, str]] = []

    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        self.calls.append((list(texts), batch, label))
        return self.vectors


class FakeHttp:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.response


def rerank_client(scores: Sequence[float]) -> tuple[RerankClient, FakeHttp]:
    http = FakeHttp(
        {
            "results": [
                {"index": index, "relevance_score": score} for index, score in enumerate(scores)
            ]
        }
    )
    client = RerankClient(
        config=RerankConfig(api_key="k", base_url="https://example.invalid", model="m"),
        http=cast(Any, http),
    )
    return client, http


class TestDenseQueryAdapter:
    def test_prepends_the_instruction_and_returns_an_immutable_vector(self) -> None:
        embedder = FakeEmbedder()
        adapter = DenseQueryAdapter(embedder, prompt="Instruct: x\nQuery:", dimensions=2)

        vector = adapter.encode("中文问题")

        assert vector == (1.0, 0.0)
        texts, batch, _ = embedder.calls[0]
        assert texts == ["Instruct: x\nQuery:中文问题"]
        assert batch == 1

    def test_a_benchmark_prompt_is_never_the_default(self) -> None:
        # The adapter has no prompt default at all, so a product profile cannot
        # inherit the frozen "news passage" instruction by omission.
        with pytest.raises(TypeError):
            DenseQueryAdapter(FakeEmbedder(), dimensions=2)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "vectors, message",
        [
            ([[1.0, 0.0], [0.0, 1.0]], "2 vectors"),
            ([[1.0, 0.0, 0.0]], "3 dimensions"),
            ([[1.0, math.nan]], "non-finite"),
        ],
    )
    def test_malformed_provider_responses_fail_closed(
        self,
        vectors: list[list[float]],
        message: str,
    ) -> None:
        adapter = DenseQueryAdapter(FakeEmbedder(vectors), prompt="p", dimensions=2)
        with pytest.raises(RuntimeError, match=message):
            adapter.encode("q")


class TestSparseQueryAdapter:
    def test_delegates_to_the_collection_vocabulary(self) -> None:
        build = build_sparse_index({"a": "向量检索", "b": "集群部署"})
        adapter = SparseQueryAdapter(build.index)
        assert adapter.encode_query("向量检索") == build.index.encode_query("向量检索")
        assert adapter.encode_query("完全不相关的词") == ()


class TestRerankAdapter:
    def test_sends_the_given_instruction_and_preserves_request_order(self) -> None:
        client, http = rerank_client([0.2, 0.9])
        adapter = RerankAdapter(client, instruction="TiDB 文档指令")

        scores = adapter.score("q", ["甲", "乙"])

        assert scores == (0.2, 0.9)
        payload = http.payloads[0]
        assert payload["instruction"] == "TiDB 文档指令"
        assert payload["documents"] == ["甲", "乙"]
        assert payload["return_documents"] is False


class TestComposition:
    def test_adapters_satisfy_the_online_protocols(self) -> None:
        settings = OnlineSettings.benchmark_exact()
        dense = DenseQueryAdapter(
            FakeEmbedder([[0.0] * settings.dense_dimensions]),
            prompt="Instruct: x\nQuery:",
            dimensions=settings.dense_dimensions,
        )
        sparse = SparseQueryAdapter(build_sparse_index({"a": "向量"}).index)
        client, _ = rerank_client([1.0])
        reranker = RerankAdapter(client, instruction="i")

        assert len(dense.encode("q")) == settings.dense_dimensions
        assert sparse.encode_query("向量") != ()
        assert reranker.score("q", ["甲"]) == (1.0,)
