"""Pure online dense+sparse retrieval with exact local RRF and reranking.

The benchmark profile intentionally mirrors the offline evidence: each retrieval
arm contributes 100 candidates, local RRF uses ``k=10`` at depth 100, the
reranker receives the first 100 fused passages, and only its first 50 scores are
allowed to alter the ranking. Requesting only 50 passages would be a different
provider experiment, not an implementation shortcut.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from zhrag.eval.rerank import rerank_prefix
from zhrag.lexical.sparse import SparseVector
from zhrag.retrieval.fusion import reciprocal_rank_fusion
from zhrag.store.base import ArmHit, DenseVector, Passage, VectorStore

__all__ = [
    "DenseQueryEncoder",
    "OnlineRetrievalResult",
    "OnlineRetriever",
    "OnlineSettings",
    "PassageReranker",
    "RankedPassage",
    "SparseQueryEncoder",
    "StageTimings",
]


class DenseQueryEncoder(Protocol):
    """Encode one exact query under a named asymmetric embedding profile."""

    def encode(self, query: str) -> Sequence[float]: ...


class SparseQueryEncoder(Protocol):
    """Encode one query with the collection's frozen lexical vocabulary."""

    def encode_query(self, query: str) -> SparseVector: ...


class PassageReranker(Protocol):
    """Return one relevance score per document in request order."""

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class OnlineSettings:
    """One explicit, fingerprintable online retrieval profile."""

    profile_name: str
    embedding_profile: str
    rerank_profile: str
    dense_dimensions: int
    arm_depth: int
    fusion_depth: int
    rrf_k: int
    rerank_request_depth: int
    rerank_apply_depth: int
    output_limit: int
    rerank_enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("profile_name", "embedding_profile", "rerank_profile"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        positive = (
            "dense_dimensions",
            "arm_depth",
            "fusion_depth",
            "rerank_request_depth",
            "rerank_apply_depth",
            "output_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer")
        if not isinstance(self.rerank_enabled, bool):
            raise TypeError("rerank_enabled must be boolean")
        if self.fusion_depth > self.arm_depth:
            raise ValueError("fusion_depth cannot exceed arm_depth")
        if self.rerank_request_depth > 2 * self.fusion_depth:
            raise ValueError("rerank_request_depth exceeds the maximum fused union")
        if self.rerank_apply_depth > self.rerank_request_depth:
            raise ValueError("rerank_apply_depth cannot exceed rerank_request_depth")
        if self.output_limit > self.rerank_apply_depth:
            raise ValueError("output_limit cannot exceed rerank_apply_depth")

    @classmethod
    def benchmark_exact(cls, *, output_limit: int = 10) -> OnlineSettings:
        """Return the frozen dense-4096 news benchmark deployment shape."""
        return cls(
            profile_name="benchmark-exact-dense4096-rrf10-rerank100to50-v1",
            embedding_profile="qwen3-embedding-8b-news-query-4096-v1",
            rerank_profile="qwen3-reranker-8b-news-v1",
            dense_dimensions=4096,
            arm_depth=100,
            fusion_depth=100,
            rrf_k=10,
            rerank_request_depth=100,
            rerank_apply_depth=50,
            output_limit=output_limit,
        )

    @classmethod
    def product(
        cls,
        *,
        profile_name: str,
        embedding_profile: str,
        rerank_profile: str,
        dense_dimensions: int,
        output_limit: int = 10,
        rerank_enabled: bool = True,
    ) -> OnlineSettings:
        """Name a product prompt/profile separately from benchmark evidence."""
        return cls(
            profile_name=profile_name,
            embedding_profile=embedding_profile,
            rerank_profile=rerank_profile,
            dense_dimensions=dense_dimensions,
            arm_depth=100,
            fusion_depth=100,
            rrf_k=10,
            rerank_request_depth=100,
            rerank_apply_depth=50,
            output_limit=output_limit,
            rerank_enabled=rerank_enabled,
        )


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Monotonic wall-clock seconds for every online stage."""

    dense_encode_seconds: float
    sparse_encode_seconds: float
    dense_search_seconds: float
    sparse_search_seconds: float
    fusion_seconds: float
    fetch_seconds: float
    rerank_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RankedPassage:
    """One final passage with both retrieval and reranker provenance."""

    rank: int
    fused_rank: int
    rerank_score: float
    passage: Passage

    def __post_init__(self) -> None:
        if self.rank < 1 or self.fused_rank < 1:
            raise ValueError("ranks must be positive")
        if not math.isfinite(self.rerank_score):
            raise ValueError("rerank_score must be finite")


@dataclass(frozen=True, slots=True)
class OnlineRetrievalResult:
    """Final passages plus enough immutable diagnostics to audit one query."""

    query: str
    profile_name: str
    embedding_profile: str
    rerank_profile: str
    dense_hits: tuple[ArmHit, ...]
    sparse_hits: tuple[ArmHit, ...]
    fused_candidates: tuple[str, ...]
    rerank_request: tuple[str, ...]
    passages: tuple[RankedPassage, ...]
    timings: StageTimings


@dataclass(slots=True)
class OnlineRetriever:
    """Compose injected encoders, store, and reranker without I/O policy."""

    settings: OnlineSettings
    dense_encoder: DenseQueryEncoder
    sparse_encoder: SparseQueryEncoder
    store: VectorStore
    reranker: PassageReranker
    clock: Callable[[], float] = time.perf_counter

    def __post_init__(self) -> None:
        if self.store.dense_dimensions != self.settings.dense_dimensions:
            raise ValueError(
                "store dense dimensions do not match the online profile: "
                f"{self.store.dense_dimensions} vs {self.settings.dense_dimensions}"
            )

    def retrieve(self, query: str) -> OnlineRetrievalResult:
        """Run exact two-arm retrieval and fail closed on incomplete boundaries."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")

        total_start = self.clock()
        dense_raw, dense_encode = _timed(self.clock, lambda: self.dense_encoder.encode(query))
        dense = _validated_dense(dense_raw, self.settings.dense_dimensions)
        sparse, sparse_encode = _timed(self.clock, lambda: self.sparse_encoder.encode_query(query))
        _validate_sparse(sparse)

        dense_raw_hits, dense_search = _timed(
            self.clock,
            lambda: self.store.search_dense(dense, limit=self.settings.arm_depth),
        )
        dense_hits = _validated_hits(
            dense_raw_hits,
            limit=self.settings.arm_depth,
            arm="dense",
        )
        sparse_raw_hits, sparse_search = _timed(
            self.clock,
            lambda: self.store.search_sparse(sparse, limit=self.settings.arm_depth),
        )
        sparse_hits = _validated_hits(
            sparse_raw_hits,
            limit=self.settings.arm_depth,
            arm="sparse",
        )

        fused, fusion_seconds = _timed(
            self.clock,
            lambda: reciprocal_rank_fusion(
                [
                    [hit.doc_id for hit in dense_hits],
                    [hit.doc_id for hit in sparse_hits],
                ],
                k=self.settings.rrf_k,
                depth=self.settings.fusion_depth,
            ),
        )
        if len(fused) < self.settings.rerank_request_depth:
            raise RuntimeError(
                f"fused run has only {len(fused)} candidates; "
                f"the profile requires {self.settings.rerank_request_depth}"
            )
        request_ids = tuple(fused[: self.settings.rerank_request_depth])
        fetched_raw, fetch_seconds = _timed(self.clock, lambda: self.store.fetch(request_ids))
        fetched = _passages_by_id(fetched_raw, request_ids)
        documents = [fetched[doc_id].text for doc_id in request_ids]

        raw_scores, rerank_seconds = _timed(
            self.clock,
            lambda: self.reranker.score(query, documents),
        )
        scores = _validated_scores(raw_scores, len(request_ids))
        scores_by_id = dict(zip(request_ids, scores, strict=True))
        reranked = rerank_prefix(
            fused,
            scores_by_id,
            depth=self.settings.rerank_apply_depth,
        )
        selected = reranked[: self.settings.output_limit]
        fused_ranks = {doc_id: rank for rank, doc_id in enumerate(fused, start=1)}
        passages = tuple(
            RankedPassage(
                rank=rank,
                fused_rank=fused_ranks[doc_id],
                rerank_score=scores_by_id[doc_id],
                passage=fetched[doc_id],
            )
            for rank, doc_id in enumerate(selected, start=1)
        )
        total_seconds = _duration(total_start, self.clock())
        return OnlineRetrievalResult(
            query=query,
            profile_name=self.settings.profile_name,
            embedding_profile=self.settings.embedding_profile,
            rerank_profile=self.settings.rerank_profile,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_candidates=tuple(fused),
            rerank_request=request_ids,
            passages=passages,
            timings=StageTimings(
                dense_encode_seconds=dense_encode,
                sparse_encode_seconds=sparse_encode,
                dense_search_seconds=dense_search,
                sparse_search_seconds=sparse_search,
                fusion_seconds=fusion_seconds,
                fetch_seconds=fetch_seconds,
                rerank_seconds=rerank_seconds,
                total_seconds=total_seconds,
            ),
        )


def _timed[T](clock: Callable[[], float], operation: Callable[[], T]) -> tuple[T, float]:
    start = clock()
    result = operation()
    return result, _duration(start, clock())


def _duration(start: float, end: float) -> float:
    duration = end - start
    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError("clock returned a non-finite or decreasing timestamp")
    return duration


def _validated_dense(raw: Sequence[float], dimensions: int) -> DenseVector:
    if isinstance(raw, (str, bytes)):
        raise TypeError("dense encoder returned a scalar string")
    values: list[float] = []
    for raw_value in raw:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError("dense encoder returned a non-numeric value")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("dense encoder returned a non-finite value")
        values.append(value)
    if len(values) != dimensions:
        raise ValueError(f"dense encoder returned {len(values)} dimensions, expected {dimensions}")
    return tuple(values)


def _validate_sparse(vector: SparseVector) -> None:
    previous = -1
    for index, value in vector:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index <= previous:
            raise ValueError("sparse encoder returned unsorted or duplicate indices")
        if not math.isfinite(value) or value == 0:
            raise ValueError("sparse encoder returned a non-finite or zero value")
        previous = index


def _validated_hits(
    raw: Sequence[ArmHit],
    *,
    limit: int,
    arm: str,
) -> tuple[ArmHit, ...]:
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"{arm} search returned a scalar string")
    hits = tuple(raw)
    if len(hits) > limit:
        raise RuntimeError(f"{arm} search returned more than its requested limit")
    if any(not isinstance(hit, ArmHit) for hit in hits):
        raise TypeError(f"{arm} search returned a non-ArmHit value")
    ids = [hit.doc_id for hit in hits]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{arm} search returned duplicate document ids")
    return hits


def _passages_by_id(
    raw: Sequence[Passage],
    requested: Sequence[str],
) -> dict[str, Passage]:
    if isinstance(raw, (str, bytes)):
        raise TypeError("fetch returned a scalar string")
    requested_set = set(requested)
    passages: dict[str, Passage] = {}
    for passage in raw:
        if not isinstance(passage, Passage):
            raise TypeError("fetch returned a non-Passage value")
        if passage.doc_id not in requested_set:
            raise RuntimeError(f"fetch returned unrequested passage {passage.doc_id!r}")
        if passage.doc_id in passages:
            raise RuntimeError(f"fetch returned duplicate passage {passage.doc_id!r}")
        passages[passage.doc_id] = passage
    missing = [doc_id for doc_id in requested if doc_id not in passages]
    if missing:
        raise RuntimeError(f"fetch omitted {len(missing)} requested passages")
    return passages


def _validated_scores(raw: Sequence[float], expected: int) -> tuple[float, ...]:
    if isinstance(raw, (str, bytes)):
        raise TypeError("reranker returned a scalar string")
    scores: list[float] = []
    for raw_score in raw:
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise TypeError("reranker returned a non-numeric score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("reranker returned a non-finite score")
        scores.append(score)
    if len(scores) != expected:
        raise RuntimeError(f"reranker returned {len(scores)} scores for {expected} passages")
    return tuple(scores)
