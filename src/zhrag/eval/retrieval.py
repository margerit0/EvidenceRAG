"""Shared offline retrieval runs used by evaluation scripts.

The dense/BM25 comparison and the rerank experiment must start from the same
rankings. Keeping their loaders and rankers here prevents a later script from
quietly changing a tie-break, corpus order, or cache-miss policy while still
calling its baseline "RRF k=10/depth=100".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import Query
from zhrag.eval.metrics import (
    all_gold_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from zhrag.io_utils import read_jsonl
from zhrag.lexical import BM25, char_ngram

__all__ = [
    "bm25_runs",
    "dense_runs",
    "load_embedding_matrix",
    "load_queries",
    "per_query_metrics",
    "prefix_l2_normalize",
]


def load_queries(
    qrels_path: Path,
    tasks: Sequence[str],
    limit: int | None = None,
) -> list[Query]:
    """Load selected CRUD-RAG tasks in their stable JSONL order."""
    selected = set(tasks)
    queries = [
        Query(
            query_id=row["query_id"],
            question=row["question"],
            answer=row["answer"],
            gold_doc_ids=tuple(row["gold_doc_ids"]),
            task=row["task"],
        )
        for row in read_jsonl(qrels_path)
        if row["task"] in selected
    ]
    return queries[:limit] if limit else queries


def load_embedding_matrix(
    cache: Path,
    ids: Sequence[str],
    *,
    width: int,
    require_all: bool = True,
) -> tuple[NDArray[np.float32], list[str]]:
    """Stream an append-only embedding cache into a float32 matrix.

    Duplicate ids resolve last-write-wins. When ``require_all`` is false, the
    returned missing-id list tells the caller whether an optional analysis can
    run; callers must not score the corresponding zero rows.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if len(set(ids)) != len(ids) or any(
        not isinstance(item_id, str) or not item_id for item_id in ids
    ):
        raise ValueError("embedding ids must be unique non-empty strings")
    position = {item_id: i for i, item_id in enumerate(ids)}
    matrix = np.zeros((len(ids), width), dtype=np.float32)
    seen: set[str] = set()
    for lineno, row in enumerate(read_jsonl(cache), 1):
        item_id = row.get("doc_id")
        if not isinstance(item_id, str) or not item_id:
            raise SystemExit(f"! {cache.name}:{lineno}: malformed doc_id {item_id!r}")
        i = position.get(item_id)
        if i is None:
            continue
        vector = row.get("embedding")
        if not isinstance(vector, list):
            raise SystemExit(f"! {cache.name}:{lineno}: embedding is not an array")
        if len(vector) != width:
            raise SystemExit(f"! {cache.name}: {item_id} has width {len(vector)}, not {width}")
        try:
            values = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(f"! {cache.name}:{lineno}: embedding is not numeric") from exc
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"! {cache.name}:{lineno}: embedding contains non-finite values")
        matrix[i] = values
        seen.add(item_id)

    missing = [item_id for item_id in ids if item_id not in seen]
    if missing and require_all:
        raise SystemExit(
            f"! {cache.name} is missing {len(missing):,} of {len(ids):,} vectors.\n"
            "  Populate the embedding cache before running this analysis."
        )
    return matrix, missing


def prefix_l2_normalize(
    matrix: NDArray[np.float32],
    width: int,
) -> NDArray[np.float32]:
    """Prefix-slice every row and L2-normalize it without mutating the source.

    Qwen3's MRL truncation applies the dimensional prefix before the final
    normalization.  Loading the full cache first is deliberate: a cache file
    containing native 1024-d rows would be a different paid run, not this
    experiment's client-side ablation.
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if matrix.ndim != 2:
        raise ValueError("embedding matrix must be two-dimensional")
    if width > matrix.shape[1]:
        raise ValueError(f"requested width {width} exceeds source width {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding matrix must contain only finite values")

    prefix = np.array(matrix[:, :width], dtype=np.float32, copy=True)
    if not np.all(np.isfinite(prefix)):
        raise ValueError("embedding prefix is not representable as finite float32")
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError("embedding prefix has a non-finite L2 norm")
    if np.any(norms == 0):
        raise ValueError("embedding prefix has a zero L2 norm")
    normalized = prefix / norms
    if normalized.dtype != np.float32:
        normalized = normalized.astype(np.float32)
    return normalized


def bm25_runs(
    corpus: Mapping[str, str],
    queries: Sequence[Query],
    *,
    depth: int,
) -> list[list[str]]:
    """Rank every query with the frozen character-bigram BM25 baseline."""
    index = BM25(analyzer=char_ngram(2)).index(list(corpus), list(corpus.values()))
    return [[doc_id for doc_id, _ in index.search(query.question, k=depth)] for query in queries]


def dense_runs(
    query_matrix: NDArray[np.float32],
    doc_matrix: NDArray[np.float32],
    doc_ids: Sequence[str],
    *,
    depth: int,
) -> list[list[str]]:
    """Rank every query by cosine with a deterministic document-id tie-break.

    ``argpartition`` alone chooses an arbitrary member when a score tie straddles
    the requested cut-off, and its subsequent quicksort is not stable. Exact ties
    are uncommon for dense floats but real for zero/OOV vectors and quantized
    backends; allowing them to depend on array layout makes candidate pools drift
    across corpus order and NumPy versions. Lexicographic sort on ``(-score,
    doc_id)`` matches the lexical and RRF contracts and keeps the top-N boundary
    reproducible.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if query_matrix.ndim != 2 or doc_matrix.ndim != 2:
        raise ValueError("query and document matrices must be two-dimensional")
    if len(query_matrix) == 0:
        return []
    selected = min(depth, len(doc_ids))
    if selected < 1:
        raise ValueError("doc_ids must be non-empty")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("doc_ids must be unique")
    if doc_matrix.shape[0] != len(doc_ids):
        raise ValueError("doc_matrix row count does not match doc_ids")
    if query_matrix.shape[1] != doc_matrix.shape[1]:
        raise ValueError("query and document embedding widths differ")
    if not np.all(np.isfinite(query_matrix)) or not np.all(np.isfinite(doc_matrix)):
        raise ValueError("query and document matrices must contain only finite values")
    similarities = query_matrix @ doc_matrix.T
    ids = np.asarray(doc_ids, dtype=object)
    return [
        [doc_ids[index] for index in np.lexsort((ids, -scores))[:selected]]
        for scores in similarities
    ]


def per_query_metrics(
    runs: Sequence[Sequence[str]],
    queries: Sequence[Query],
) -> dict[str, list[float]]:
    """Compute aligned retrieval metrics, including arity-safe binary views."""
    pairs = list(zip(runs, queries, strict=True))
    return {
        "R@1": [recall_at_k(run, query.gold_doc_ids, 1) for run, query in pairs],
        "MRR@10": [mrr_at_k(run, query.gold_doc_ids, 10) for run, query in pairs],
        "nDCG@10": [ndcg_at_k(run, query.gold_doc_ids, 10) for run, query in pairs],
        "hit@1": [hit_at_k(run, query.gold_doc_ids, 1) for run, query in pairs],
        "ALL@10": [all_gold_at_k(run, query.gold_doc_ids, 10) for run, query in pairs],
    }
