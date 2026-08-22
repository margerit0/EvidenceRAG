"""Pure online retrieval tests with no provider, corpus, or Milvus dependency."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from zhrag.lexical import SparseVector
from zhrag.retrieval import OnlineRetriever, OnlineSettings
from zhrag.store import ArmHit, DenseVector, Passage

SHA256 = "a" * 64


class FakeDenseEncoder:
    def __init__(self, vector: Sequence[float] = (1.0, 0.0)) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def encode(self, query: str) -> Sequence[float]:
        self.queries.append(query)
        return self.vector


class FakeSparseEncoder:
    def __init__(self, vector: SparseVector = ((3, 1.0),)) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def encode_query(self, query: str) -> SparseVector:
        self.queries.append(query)
        return self.vector


class FakeStore:
    def __init__(
        self,
        *,
        dense_hits: Sequence[ArmHit] = (
            ArmHit("a", 0.9),
            ArmHit("b", 0.8),
            ArmHit("c", 0.7),
        ),
        sparse_hits: Sequence[ArmHit] = (
            ArmHit("d", 4.0),
            ArmHit("b", 3.0),
            ArmHit("e", 2.0),
        ),
        dimensions: int = 2,
    ) -> None:
        self._dense_hits = dense_hits
        self._sparse_hits = sparse_hits
        self._dense_dimensions = dimensions
        self.dense_calls: list[tuple[DenseVector, int]] = []
        self.sparse_calls: list[tuple[SparseVector, int]] = []
        self.fetch_calls: list[tuple[str, ...]] = []
        self.fetch_override: Sequence[Passage] | None = None

    @property
    def collection_name(self) -> str:
        return "fake"

    @property
    def dense_dimensions(self) -> int:
        return self._dense_dimensions

    def search_dense(self, vector: DenseVector, *, limit: int) -> Sequence[ArmHit]:
        self.dense_calls.append((vector, limit))
        return self._dense_hits

    def search_sparse(self, vector: SparseVector, *, limit: int) -> Sequence[ArmHit]:
        self.sparse_calls.append((vector, limit))
        return self._sparse_hits

    def fetch(self, doc_ids: Sequence[str]) -> Sequence[Passage]:
        ids = tuple(doc_ids)
        self.fetch_calls.append(ids)
        if self.fetch_override is not None:
            return self.fetch_override
        return [passage(doc_id) for doc_id in reversed(ids)]


class FakeReranker:
    def __init__(self, scores: Sequence[float] = (0.1, 0.5, 0.4, 100.0)) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        self.calls.append((query, tuple(documents)))
        return self.scores


class StepClock:
    def __init__(self, *, step: float = 1.0) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def passage(doc_id: str) -> Passage:
    return Passage(
        doc_id=doc_id,
        text=f"text:{doc_id}",
        source_key=f"source:{doc_id}",
        document_sha256=SHA256,
        ordinal=0,
        metadata={"doc": doc_id},
    )


def settings(**overrides: object) -> OnlineSettings:
    values: dict[str, object] = {
        "profile_name": "test-profile",
        "embedding_profile": "test-embedding",
        "rerank_profile": "test-reranker",
        "dense_dimensions": 2,
        "arm_depth": 3,
        "fusion_depth": 3,
        "rrf_k": 10,
        "rerank_request_depth": 4,
        "rerank_apply_depth": 3,
        "output_limit": 2,
    }
    values.update(overrides)
    return OnlineSettings(**values)  # type: ignore[arg-type]


def retriever(
    *,
    configured: OnlineSettings | None = None,
    dense: FakeDenseEncoder | None = None,
    sparse: FakeSparseEncoder | None = None,
    store: FakeStore | None = None,
    reranker: FakeReranker | None = None,
    clock: StepClock | None = None,
) -> OnlineRetriever:
    return OnlineRetriever(
        settings=configured or settings(),
        dense_encoder=dense or FakeDenseEncoder(),
        sparse_encoder=sparse or FakeSparseEncoder(),
        store=cast(Any, store or FakeStore()),
        reranker=reranker or FakeReranker(),
        clock=clock or StepClock(),
    )


class TestFrozenProfile:
    def test_benchmark_profile_records_exact_evidence_shape(self) -> None:
        profile = OnlineSettings.benchmark_exact()
        assert profile.dense_dimensions == 4096
        assert profile.arm_depth == 100
        assert profile.fusion_depth == 100
        assert profile.rrf_k == 10
        assert profile.rerank_request_depth == 100
        assert profile.rerank_apply_depth == 50
        assert profile.output_limit == 10
        assert "news" in profile.embedding_profile
        assert "news" in profile.rerank_profile

    def test_product_profile_requires_separate_names(self) -> None:
        profile = OnlineSettings.product(
            profile_name="tidb-product-v1",
            embedding_profile="qwen3-tidb-query-4096-v1",
            rerank_profile="qwen3-tidb-v1",
            dense_dimensions=4096,
        )
        assert profile.profile_name == "tidb-product-v1"
        assert profile.embedding_profile != OnlineSettings.benchmark_exact().embedding_profile

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"fusion_depth": 4}, "fusion_depth"),
            ({"rerank_request_depth": 7}, "maximum fused"),
            ({"rerank_apply_depth": 5}, "rerank_apply_depth"),
            ({"output_limit": 4}, "output_limit"),
            ({"rrf_k": -1}, "rrf_k"),
            ({"dense_dimensions": 0}, "dense_dimensions"),
        ],
    )
    def test_rejects_incoherent_settings(
        self,
        overrides: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            settings(**overrides)


class TestPipeline:
    def test_exact_local_rrf_fetch_and_request_to_apply_split(self) -> None:
        dense = FakeDenseEncoder()
        sparse = FakeSparseEncoder()
        store = FakeStore()
        reranker = FakeReranker()
        result = retriever(
            dense=dense,
            sparse=sparse,
            store=store,
            reranker=reranker,
        ).retrieve("原样查询？")

        assert dense.queries == ["原样查询？"]
        assert sparse.queries == ["原样查询？"]
        assert store.dense_calls == [((1.0, 0.0), 3)]
        assert store.sparse_calls == [(((3, 1.0),), 3)]
        assert result.fused_candidates == ("b", "a", "d", "c", "e")
        assert result.rerank_request == ("b", "a", "d", "c")
        assert store.fetch_calls == [("b", "a", "d", "c")]
        assert reranker.calls == [("原样查询？", ("text:b", "text:a", "text:d", "text:c"))]
        # c has the largest returned score, but it is outside apply_depth=3.
        assert [row.passage.doc_id for row in result.passages] == ["a", "d"]
        assert [(row.rank, row.fused_rank, row.rerank_score) for row in result.passages] == [
            (1, 2, 0.5),
            (2, 3, 0.4),
        ]
        assert result.profile_name == "test-profile"
        assert result.embedding_profile == "test-embedding"
        assert result.rerank_profile == "test-reranker"

    def test_equal_rerank_scores_preserve_fused_order(self) -> None:
        result = retriever(reranker=FakeReranker((1.0, 1.0, 1.0, 999.0))).retrieve("q")
        assert [row.passage.doc_id for row in result.passages] == ["b", "a"]

    def test_timings_are_non_overlapping_stage_durations(self) -> None:
        result = retriever(clock=StepClock(step=0.25)).retrieve("q")
        timings = result.timings
        assert timings.dense_encode_seconds == 0.25
        assert timings.sparse_encode_seconds == 0.25
        assert timings.dense_search_seconds == 0.25
        assert timings.sparse_search_seconds == 0.25
        assert timings.fusion_seconds == 0.25
        assert timings.fetch_seconds == 0.25
        assert timings.rerank_seconds == 0.25
        assert timings.total_seconds == 3.75


class TestFailures:
    @pytest.mark.parametrize("query", ["", "  \n"])
    def test_rejects_empty_queries_before_any_dependency_call(self, query: str) -> None:
        dense = FakeDenseEncoder()
        with pytest.raises(ValueError, match="query"):
            retriever(dense=dense).retrieve(query)
        assert dense.queries == []

    def test_rejects_store_profile_dimension_mismatch_at_composition(self) -> None:
        with pytest.raises(ValueError, match="store dense dimensions"):
            retriever(store=FakeStore(dimensions=3))

    @pytest.mark.parametrize("vector", [(1.0,), (1.0, math.nan)])
    def test_rejects_malformed_dense_encoding(self, vector: Sequence[float]) -> None:
        with pytest.raises(ValueError, match="dense encoder"):
            retriever(dense=FakeDenseEncoder(vector)).retrieve("q")

    def test_rejects_malformed_sparse_encoding(self) -> None:
        malformed = cast(SparseVector, ((2, 1.0), (1, 1.0)))
        with pytest.raises(ValueError, match="sparse encoder"):
            retriever(sparse=FakeSparseEncoder(malformed)).retrieve("q")

    def test_rejects_duplicate_arm_hits(self) -> None:
        store = FakeStore(dense_hits=(ArmHit("a", 1.0), ArmHit("a", 0.9)))
        with pytest.raises(RuntimeError, match="duplicate"):
            retriever(store=store).retrieve("q")

    def test_rejects_insufficient_fused_union(self) -> None:
        hits = (ArmHit("a", 1.0), ArmHit("b", 0.5))
        store = FakeStore(dense_hits=hits, sparse_hits=hits)
        with pytest.raises(RuntimeError, match="only 2 candidates"):
            retriever(store=store).retrieve("q")

    def test_rejects_missing_and_unrequested_fetched_rows(self) -> None:
        store = FakeStore()
        store.fetch_override = [passage("b"), passage("a"), passage("d")]
        with pytest.raises(RuntimeError, match="omitted 1"):
            retriever(store=store).retrieve("q")

        store = FakeStore()
        store.fetch_override = [passage("b"), passage("a"), passage("d"), passage("z")]
        with pytest.raises(RuntimeError, match="unrequested"):
            retriever(store=store).retrieve("q")

    @pytest.mark.parametrize(
        "scores, error, message",
        [
            ((1.0,), RuntimeError, "1 scores for 4"),
            ((1.0, 2.0, 3.0, math.nan), ValueError, "non-finite"),
        ],
    )
    def test_rejects_malformed_rerank_scores(
        self,
        scores: Sequence[float],
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            retriever(reranker=FakeReranker(scores)).retrieve("q")

    def test_rejects_a_decreasing_clock(self) -> None:
        values = iter([1.0, 2.0, 1.0])
        with pytest.raises(RuntimeError, match="clock"):
            retriever(clock=cast(StepClock, lambda: next(values))).retrieve("q")
