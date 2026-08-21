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
    position = {item_id: i for i, item_id in enumerate(ids)}
    matrix = np.zeros((len(ids), width), dtype=np.float32)
    seen: set[str] = set()
    for row in read_jsonl(cache):
        item_id = row["doc_id"]
        i = position.get(item_id)
        if i is None:
            continue
        vector = row["embedding"]
        if len(vector) != width:
            raise SystemExit(f"! {cache.name}: {item_id} has width {len(vector)}, not {width}")
        matrix[i] = vector
        seen.add(item_id)

    missing = [item_id for item_id in ids if item_id not in seen]
    if missing and require_all:
        raise SystemExit(
            f"! {cache.name} is missing {len(missing):,} of {len(ids):,} vectors.\n"
            "  Populate the embedding cache before running this analysis."
        )
    return matrix, missing


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
    """Rank every query by cosine over already L2-normalised vectors."""
    if len(query_matrix) == 0:
        return []
    selected = min(depth, len(doc_ids))
    if selected < 1:
        raise ValueError("doc_ids must be non-empty")
    similarities = query_matrix @ doc_matrix.T
    top = np.argpartition(-similarities, selected - 1, axis=1)[:, :selected]
    ordered = np.take_along_axis(
        top,
        np.argsort(-np.take_along_axis(similarities, top, 1), axis=1),
        1,
    )
    return [[doc_ids[i] for i in row] for row in ordered]


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
