"""Build the frozen retrieval runs used to pool TiDB relevance judgements.

The pool must be a union of *all systems that will later be compared*. Otherwise
one system's blind spots become the benchmark's blind spots, and that system is
rewarded twice: first for choosing what gets judged, then for being measured
against those judgements. This module therefore names four runs explicitly:
character-bigram BM25, Qwen3 dense-4096, equal-weight RRF k=10, and the product
reranker applied to the first 50 of the fused top 100.

Only pure ranking and validation live here. Embedding and reranker HTTP calls,
append-only caches and artifact I/O stay in ``scripts/build_tidb_pool.py``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.rerank import rerank_prefix
from zhrag.eval.retrieval import dense_runs
from zhrag.lexical import SparseIndex, sparse_dot
from zhrag.retrieval import reciprocal_rank_fusion

__all__ = [
    "DENSE_LABEL",
    "LEXICAL_LABEL",
    "RERANK_LABEL",
    "RRF_LABEL",
    "TIDB_RUNS_SCHEMA",
    "TiDBRuns",
    "build_dense_runs",
    "build_lexical_runs",
    "build_reranked_runs",
    "build_rrf_runs",
    "runs_fingerprint",
]

TIDB_RUNS_SCHEMA = "zhrag-tidb-runs-v1"
LEXICAL_LABEL = "bm25-char-bigram"
DENSE_LABEL = "dense-qwen3-4096"
RRF_LABEL = "rrf-k10-depth100"
RERANK_LABEL = "rerank-qwen3-top50"


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


@dataclass(frozen=True, slots=True)
class TiDBRuns:
    """Aligned named runs for one exact query sequence."""

    query_ids: tuple[str, ...]
    runs: Mapping[str, tuple[tuple[str, ...], ...]]

    def __post_init__(self) -> None:
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query ids must be non-empty and unique")
        required = {LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL}
        if set(self.runs) != required:
            raise ValueError(
                f"run labels differ from the frozen set: {sorted(set(self.runs) ^ required)}"
            )
        for label, rows in self.runs.items():
            if len(rows) != len(self.query_ids):
                raise ValueError(f"{label}: run count does not match query count")
            for query_id, run in zip(self.query_ids, rows, strict=True):
                if len(set(run)) != len(run):
                    raise ValueError(f"{label}/{query_id}: a run must contain unique ids")
                if label != LEXICAL_LABEL and not run:
                    raise ValueError(f"{label}/{query_id}: a run must be non-empty")

    def for_query(self, query_id: str) -> dict[str, tuple[str, ...]]:
        try:
            position = self.query_ids.index(query_id)
        except ValueError as exc:
            raise KeyError(query_id) from exc
        return {label: rows[position] for label, rows in self.runs.items()}

    @property
    def fingerprint(self) -> str:
        return runs_fingerprint(self.query_ids, self.runs)


def build_lexical_runs(
    index: SparseIndex,
    corpus: Mapping[str, str],
    questions: Sequence[str],
    *,
    depth: int,
) -> tuple[tuple[str, ...], ...]:
    """Rank with the persisted BM25 vocabulary used by the published index.

    Rebuilding a local BM25 index from text would normally be equivalent, but
    using the persisted sparse statistics makes the candidate run bind to the
    exact published collection and avoids a second tokenizer/statistics path.
    """
    _validate_depth(depth)
    if set(corpus) == set():
        raise ValueError("corpus must be non-empty")
    encoded = {doc_id: index.encode_document(text) for doc_id, text in corpus.items()}
    selected = min(depth, len(corpus))
    runs: list[tuple[str, ...]] = []
    for question in questions:
        query = index.encode_query(question)
        scored = (
            (doc_id, score)
            for doc_id, vector in encoded.items()
            if (score := sparse_dot(query, vector)) > 0.0
        )
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:selected]
        runs.append(tuple(doc_id for doc_id, _score in ranked))
    return tuple(runs)


def build_dense_runs(
    query_matrix: NDArray[np.float32],
    doc_matrix: NDArray[np.float32],
    doc_ids: Sequence[str],
    *,
    depth: int,
) -> tuple[tuple[str, ...], ...]:
    """Rank L2-normalised query/document embeddings by cosine inner product."""
    _validate_depth(depth)
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("document ids must be unique")
    return tuple(tuple(run) for run in dense_runs(query_matrix, doc_matrix, doc_ids, depth=depth))


def build_rrf_runs(
    lexical: Sequence[Sequence[str]],
    dense: Sequence[Sequence[str]],
    *,
    depth: int,
    rrf_k: int,
) -> tuple[tuple[str, ...], ...]:
    """Fuse aligned lexical and dense runs under the frozen product settings."""
    _validate_depth(depth)
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("rrf_k must be a non-negative integer")
    if len(lexical) != len(dense):
        raise ValueError("lexical and dense query counts differ")
    return tuple(
        tuple(reciprocal_rank_fusion([left, right], k=rrf_k, depth=depth))
        for left, right in zip(lexical, dense, strict=True)
    )


def build_reranked_runs(
    fused: Sequence[Sequence[str]],
    query_ids: Sequence[str],
    scores: Mapping[tuple[str, str], float],
    *,
    request_depth: int,
    apply_depth: int,
) -> tuple[tuple[str, ...], ...]:
    """Apply cached scores to the frozen prefix, leaving the fused tail intact."""
    _validate_depth(request_depth)
    _validate_depth(apply_depth)
    if apply_depth > request_depth:
        raise ValueError("apply_depth cannot exceed request_depth")
    if len(fused) != len(query_ids):
        raise ValueError("fused run and query counts differ")
    rows: list[tuple[str, ...]] = []
    for query_id, run in zip(query_ids, fused, strict=True):
        if len(run) < request_depth:
            raise ValueError(
                f"{query_id}: fused run has {len(run)} candidates, below request depth"
            )
        expected = run[:request_depth]
        missing = [doc_id for doc_id in expected if (query_id, doc_id) not in scores]
        if missing:
            raise ValueError(f"{query_id}: missing {len(missing)} rerank scores")
        by_doc = {doc_id: scores[(query_id, doc_id)] for doc_id in expected}
        rows.append(tuple(rerank_prefix(run, by_doc, depth=apply_depth)))
    return tuple(rows)


def runs_fingerprint(
    query_ids: Sequence[str],
    runs: Mapping[str, Sequence[Sequence[str]]],
) -> str:
    """Hash every named ordered run without including corpus-derived text."""
    digest = hashlib.sha256()
    _update(digest, TIDB_RUNS_SCHEMA)
    for label in sorted(runs):
        rows = runs[label]
        if len(rows) != len(query_ids):
            raise ValueError(f"{label}: run count does not match query count")
        _update(digest, label)
        for query_id, run in zip(query_ids, rows, strict=True):
            _update(digest, query_id)
            for doc_id in run:
                _update(digest, doc_id)
    return digest.hexdigest()


def _validate_depth(depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
