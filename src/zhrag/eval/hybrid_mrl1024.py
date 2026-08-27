"""Offline evaluation for the frozen CRUD-RAG H hybrid baseline.

H is deliberately narrow: char-bigram BM25 plus a client-side 1024-dimensional
prefix of the existing 4096-dimensional Qwen3 cache, fused by equal-weight exact
RRF.  The module imports no provider code and has no path that can populate a
missing cache.  G is rebuilt from the same full matrices as a retention
reference; the paid rerank cache is outside this module's inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from zhrag.embedding_contract import (
    CRUD_EMBEDDING_MODEL,
    DOCUMENT_PROMPT,
    QUERY_PROMPT,
    load_embedding_provenance,
    validate_embedding_cache,
)
from zhrag.eval.crud import QA_TASKS, Query
from zhrag.eval.metrics import (
    bootstrap_p_floor,
    holm_bonferroni,
    holm_floor_flags,
    mcnemar_exact,
    win_loss_tie,
)
from zhrag.eval.retrieval import (
    bm25_runs,
    dense_runs,
    load_embedding_matrix,
    per_query_metrics,
    prefix_l2_normalize,
)
from zhrag.io_utils import read_json, read_jsonl
from zhrag.retrieval import reciprocal_rank_fusion

__all__ = [
    "ARM_LABELS",
    "A_LABEL",
    "E_LABEL",
    "G_LABEL",
    "HYBRID_MRL1024_SCHEMA",
    "H_LABEL",
    "HybridInputs",
    "build_hybrid_mrl1024_runs",
    "evaluate_hybrid_mrl1024",
    "load_hybrid_mrl1024_inputs",
    "validate_hybrid_mrl1024_report",
]

HYBRID_MRL1024_SCHEMA = "zhrag-crud-h-hybrid-rrf-mrl1024-v1"
A_LABEL = "A-bm25-char-bigram"
E_LABEL = "E-dense-1024"
H_LABEL = "H-hybrid-a-plus-e"
G_LABEL = "G-hybrid-a-plus-dense-4096"
ARM_LABELS = (A_LABEL, E_LABEL, H_LABEL, G_LABEL)

SOURCE_WIDTH = 4096
EFFECTIVE_WIDTH = 1024
RETRIEVAL_DEPTH = 100
RRF_K = 10
CONFIDENCE = 0.95
ALPHA = 0.05
HEADLINE_TASK = "questanswer_1doc"
HEADLINE_METRICS = ("R@1", "MRR@10", "nDCG@10")
ARITY_METRICS = ("R@1", "hit@1", "ALL@10", "MRR@10", "nDCG@10")
BINARY_METRICS = ("hit@1", "ALL@10")
CONTINUOUS_METRICS = ("MRR@10", "nDCG@10")
ARITIES = (1, 2, 3)

_FROZEN_DOCUMENTS = 5_681
_FROZEN_QUERIES = 2_394
_FROZEN_HEADLINE = 800
_FROZEN_ARITY_COUNTS = {1: 809, 2: 802, 3: 783}
_FROZEN_TASK_COUNTS = {
    "questanswer_1doc": 800,
    "questanswer_2docs": 797,
    "questanswer_3docs": 797,
}

_MANIFEST_KEYS = {
    "distractors",
    "documents",
    "gold_documents",
    "queries",
    "queries_per_gold_arity",
    "queries_per_task",
    "records_per_task",
    "source_file",
    "source_sha256",
}
_SUMMARY_KEYS = {"mean", "low", "high", "n"}
_CONTRAST_KEYS = {
    "adjusted_p",
    "adjusted_p_inherits_floor",
    "arity",
    "ci_high",
    "ci_low",
    "comparator",
    "delta",
    "losses",
    "metric",
    "raw_p",
    "raw_p_at_floor",
    "reject",
    "ties_nonzero",
    "ties_zero",
    "treatment",
    "wins",
}
_FAMILY_ORDER = (
    "efficacy-binary",
    "efficacy-continuous",
    "retention-binary",
    "retention-continuous",
)
_FORBIDDEN_REPORT_KEYS = {
    "answer",
    "doc_id",
    "embedding",
    "per_query",
    "query_id",
    "question",
    "raw_rows",
    "runs",
    "scores",
    "text",
    "vectors",
}
_SEED_DERIVATION = "sha256(schema, base_seed, namespace), first unsigned 64 bits"
_BOOTSTRAP_METHOD = "query bootstrap via multinomial counts; percentile 95% CI"
_CONTINUOUS_TEST = "two-sided centred paired query bootstrap with add-one p-value"
_BINARY_TEST = "two-sided exact McNemar"


@dataclass(frozen=True, slots=True)
class HybridInputs:
    """Validated in-memory inputs in the exact row order used by the run."""

    manifest: Mapping[str, object]
    corpus: Mapping[str, str]
    queries: tuple[Query, ...]
    doc_ids: tuple[str, ...]
    doc_matrix: NDArray[np.float32]
    query_matrix: NDArray[np.float32]
    document_provenance: Mapping[str, object]
    query_provenance: Mapping[str, object]


def _require_keys(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} keys differ; missing={missing}, extra={extra}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: object, path: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{path} must be finite and in [{low}, {high}]")
    return result


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _fingerprint(value: object, path: str) -> str:
    rendered = _string(value, path)
    if len(rendered) != 64 or any(char not in "0123456789abcdef" for char in rendered):
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return rendered


def _update_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _update_text(digest: Any, value: str) -> None:
    _update_bytes(digest, value.encode("utf-8"))


def _semantic_digest(namespace: str, values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    _update_text(digest, namespace)
    for value in values:
        _update_text(digest, value)
    return digest.hexdigest()


def _json_fingerprint(namespace: str, value: Mapping[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _semantic_digest(namespace, [canonical])


def _matrix_fingerprint(
    ids: Sequence[str],
    matrix: NDArray[np.float32],
    *,
    namespace: str,
) -> str:
    if matrix.shape[0] != len(ids):
        raise ValueError("matrix fingerprint row count differs from ids")
    digest = hashlib.sha256()
    _update_text(digest, namespace)
    _update_text(digest, str(matrix.shape[0]))
    _update_text(digest, str(matrix.shape[1]))
    for item_id, row in zip(ids, matrix, strict=True):
        _update_text(digest, item_id)
        little_endian = np.asarray(row, dtype="<f4")
        _update_bytes(digest, little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _corpus_fingerprint(corpus: Mapping[str, str]) -> str:
    values: list[str] = []
    for doc_id, text in corpus.items():
        values.extend((doc_id, text))
    return _semantic_digest("zhrag-crud-expanded-corpus-v1", values)


def _qrels_fingerprint(queries: Sequence[Query]) -> str:
    values: list[str] = []
    for query in queries:
        values.extend(
            (
                query.query_id,
                query.task,
                query.question,
                query.answer,
                str(len(query.gold_doc_ids)),
                *query.gold_doc_ids,
            )
        )
    return _semantic_digest("zhrag-crud-expanded-qrels-v1", values)


def _run_fingerprint(queries: Sequence[Query], rankings: Sequence[Sequence[str]]) -> str:
    if len(queries) != len(rankings):
        raise ValueError("query/run length mismatch")
    digest = hashlib.sha256()
    _update_text(digest, "zhrag-crud-h-run-v1")
    for query, ranking in zip(queries, rankings, strict=True):
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"{query.query_id}: run contains duplicate document ids")
        _update_text(digest, query.query_id)
        _update_text(digest, str(len(ranking)))
        for doc_id in ranking:
            _update_text(digest, doc_id)
    return digest.hexdigest()


def _parse_corpus(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for lineno, row in enumerate(rows, 1):
        _require_keys(row, {"doc_id", "text"}, f"corpus:{lineno}")
        doc_id = _string(row["doc_id"], f"corpus:{lineno}.doc_id")
        text = _string(row["text"], f"corpus:{lineno}.text")
        if doc_id in corpus:
            raise ValueError(f"corpus:{lineno}: duplicate document id {doc_id!r}")
        corpus[doc_id] = text
    if not corpus:
        raise ValueError("corpus must be non-empty")
    return corpus


def _parse_queries(
    rows: Sequence[Mapping[str, object]],
    corpus: Mapping[str, str],
) -> tuple[Query, ...]:
    queries: list[Query] = []
    seen: set[str] = set()
    for lineno, row in enumerate(rows, 1):
        _require_keys(
            row,
            {"answer", "gold_doc_ids", "query_id", "question", "task"},
            f"qrels:{lineno}",
        )
        query_id = _string(row["query_id"], f"qrels:{lineno}.query_id")
        if query_id in seen:
            raise ValueError(f"qrels:{lineno}: duplicate query id {query_id!r}")
        seen.add(query_id)
        task = _string(row["task"], f"qrels:{lineno}.task")
        if task not in QA_TASKS or not query_id.startswith(f"{task}:"):
            raise ValueError(f"qrels:{lineno}: query/task contract is invalid")
        question = _string(row["question"], f"qrels:{lineno}.question")
        answer = row["answer"]
        if not isinstance(answer, str):
            raise ValueError(f"qrels:{lineno}.answer must be a string")
        raw_gold = row["gold_doc_ids"]
        if not isinstance(raw_gold, list) or not raw_gold:
            raise ValueError(f"qrels:{lineno}.gold_doc_ids must be a non-empty array")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in raw_gold):
            raise ValueError(f"qrels:{lineno}.gold_doc_ids must contain non-empty strings")
        gold = tuple(raw_gold)
        if len(set(gold)) != len(gold) or len(gold) not in ARITIES:
            raise ValueError(f"qrels:{lineno}: gold arity must be a unique 1/2/3 set")
        absent = [doc_id for doc_id in gold if doc_id not in corpus]
        if absent:
            raise ValueError(f"qrels:{lineno}: {len(absent)} gold documents are absent")
        queries.append(
            Query(
                query_id=query_id,
                question=question,
                answer=answer,
                gold_doc_ids=gold,
                task=task,
            )
        )
    if not queries:
        raise ValueError("qrels must be non-empty")
    return tuple(queries)


def _count_map(value: object, path: str) -> dict[str, int]:
    mapping = _mapping(value, path)
    return {key: _integer(item, f"{path}.{key}") for key, item in mapping.items()}


def _validate_manifest(
    manifest: Mapping[str, object],
    corpus: Mapping[str, str],
    queries: Sequence[Query],
    *,
    require_frozen: bool,
) -> None:
    _require_keys(manifest, _MANIFEST_KEYS, "manifest")
    documents = _integer(manifest["documents"], "manifest.documents", minimum=1)
    query_count = _integer(manifest["queries"], "manifest.queries", minimum=1)
    gold_documents = _integer(manifest["gold_documents"], "manifest.gold_documents", minimum=1)
    distractors = _integer(manifest["distractors"], "manifest.distractors")
    if documents != len(corpus) or query_count != len(queries):
        raise ValueError("manifest document/query counts do not match the frozen rows")
    unique_gold = {doc_id for query in queries for doc_id in query.gold_doc_ids}
    if gold_documents != len(unique_gold) or distractors != documents - len(unique_gold):
        raise ValueError("manifest gold/distractor counts do not match qrels")

    arity_counts = {str(arity): 0 for arity in ARITIES}
    task_counts = {task: 0 for task in QA_TASKS}
    for query in queries:
        arity_counts[str(len(query.gold_doc_ids))] += 1
        task_counts[query.task] += 1
    manifest_arity = _count_map(
        manifest["queries_per_gold_arity"],
        "manifest.queries_per_gold_arity",
    )
    if manifest_arity != arity_counts:
        raise ValueError("manifest arity counts do not match qrels")
    if _count_map(manifest["queries_per_task"], "manifest.queries_per_task") != task_counts:
        raise ValueError("manifest task counts do not match qrels")
    records = _count_map(manifest["records_per_task"], "manifest.records_per_task")
    if not records:
        raise ValueError("manifest.records_per_task must be non-empty")
    _string(manifest["source_file"], "manifest.source_file")
    _fingerprint(manifest["source_sha256"], "manifest.source_sha256")

    if require_frozen:
        frozen_arity = {str(key): value for key, value in _FROZEN_ARITY_COUNTS.items()}
        if (
            documents != _FROZEN_DOCUMENTS
            or query_count != _FROZEN_QUERIES
            or task_counts != _FROZEN_TASK_COUNTS
            or arity_counts != frozen_arity
        ):
            raise ValueError("expanded artifacts do not match the frozen H evaluation counts")


def _validate_provenance(
    provenance: Mapping[str, object],
    *,
    prompt: str,
    path: str,
) -> None:
    _require_keys(provenance, {"model", "prompt"}, path)
    if provenance["model"] != CRUD_EMBEDDING_MODEL or provenance["prompt"] != prompt:
        raise ValueError(f"{path} does not match the frozen model/prompt contract")


def _validate_inputs(inputs: HybridInputs) -> None:
    if tuple(inputs.corpus) != inputs.doc_ids:
        raise ValueError("doc_ids must preserve corpus mapping order")
    if inputs.doc_matrix.shape != (len(inputs.doc_ids), SOURCE_WIDTH):
        raise ValueError("document matrix shape does not match the frozen source width")
    if inputs.query_matrix.shape != (len(inputs.queries), SOURCE_WIDTH):
        raise ValueError("query matrix shape does not match the frozen source width")
    if inputs.doc_matrix.dtype != np.float32 or inputs.query_matrix.dtype != np.float32:
        raise ValueError("embedding matrices must be float32")
    if not np.all(np.isfinite(inputs.doc_matrix)) or not np.all(np.isfinite(inputs.query_matrix)):
        raise ValueError("embedding matrices must contain only finite values")
    _validate_manifest(inputs.manifest, inputs.corpus, inputs.queries, require_frozen=False)
    _validate_provenance(
        inputs.document_provenance,
        prompt=DOCUMENT_PROMPT,
        path="document_provenance",
    )
    _validate_provenance(
        inputs.query_provenance,
        prompt=QUERY_PROMPT,
        path="query_provenance",
    )


def load_hybrid_mrl1024_inputs(
    expanded: str | Path,
    *,
    require_frozen: bool = True,
) -> HybridInputs:
    """Load every H input locally and fail rather than filling a cache miss."""
    root = Path(expanded)
    manifest_raw = read_json(root / "manifest.json")
    if not isinstance(manifest_raw, dict):
        raise ValueError("manifest must be a JSON object")
    manifest: dict[str, object] = dict(manifest_raw)
    corpus_rows = tuple(read_jsonl(root / "corpus.jsonl"))
    qrel_rows = tuple(read_jsonl(root / "qrels.jsonl"))
    corpus = _parse_corpus(corpus_rows)
    queries = _parse_queries(qrel_rows, corpus)
    _validate_manifest(manifest, corpus, queries, require_frozen=require_frozen)

    document_cache = root / "emb_cache_4096.jsonl"
    query_cache = root / "emb_cache_queries_4096.jsonl"
    validate_embedding_cache(
        document_cache,
        model=CRUD_EMBEDDING_MODEL,
        prompt=DOCUMENT_PROMPT,
    )
    validate_embedding_cache(
        query_cache,
        model=CRUD_EMBEDDING_MODEL,
        prompt=QUERY_PROMPT,
    )
    document_provenance = load_embedding_provenance(document_cache)
    query_provenance = load_embedding_provenance(query_cache)
    _validate_provenance(
        document_provenance,
        prompt=DOCUMENT_PROMPT,
        path=document_cache.name,
    )
    _validate_provenance(
        query_provenance,
        prompt=QUERY_PROMPT,
        path=query_cache.name,
    )

    doc_ids = tuple(corpus)
    doc_matrix, doc_missing = load_embedding_matrix(
        document_cache,
        doc_ids,
        width=SOURCE_WIDTH,
        require_all=True,
    )
    query_matrix, query_missing = load_embedding_matrix(
        query_cache,
        [query.query_id for query in queries],
        width=SOURCE_WIDTH,
        require_all=True,
    )
    if doc_missing or query_missing:
        raise AssertionError("require_all=True returned missing cache rows")
    inputs = HybridInputs(
        manifest=manifest,
        corpus=corpus,
        queries=queries,
        doc_ids=doc_ids,
        doc_matrix=doc_matrix,
        query_matrix=query_matrix,
        document_provenance=document_provenance,
        query_provenance=query_provenance,
    )
    _validate_inputs(inputs)
    return inputs


def build_hybrid_mrl1024_runs(inputs: HybridInputs) -> dict[str, list[list[str]]]:
    """Build A/E/H and the full-width G retention reference in memory."""
    _validate_inputs(inputs)
    lexical = bm25_runs(inputs.corpus, inputs.queries, depth=RETRIEVAL_DEPTH)
    dense_1024 = dense_runs(
        prefix_l2_normalize(inputs.query_matrix, EFFECTIVE_WIDTH),
        prefix_l2_normalize(inputs.doc_matrix, EFFECTIVE_WIDTH),
        inputs.doc_ids,
        depth=RETRIEVAL_DEPTH,
    )
    dense_4096 = dense_runs(
        inputs.query_matrix,
        inputs.doc_matrix,
        inputs.doc_ids,
        depth=RETRIEVAL_DEPTH,
    )
    h = [
        reciprocal_rank_fusion([left, right], k=RRF_K, depth=RETRIEVAL_DEPTH)
        for left, right in zip(lexical, dense_1024, strict=True)
    ]
    g = [
        reciprocal_rank_fusion([left, right], k=RRF_K, depth=RETRIEVAL_DEPTH)
        for left, right in zip(lexical, dense_4096, strict=True)
    ]
    return {A_LABEL: lexical, E_LABEL: dense_1024, H_LABEL: h, G_LABEL: g}


def _derived_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256()
    _update_text(digest, HYBRID_MRL1024_SCHEMA)
    _update_text(digest, str(base_seed))
    _update_text(digest, namespace)
    return int.from_bytes(digest.digest()[:8], "big")


def _bootstrap_weights(n: int, resamples: int, seed: int) -> NDArray[np.int64]:
    if n < 1 or resamples < 1:
        raise ValueError("bootstrap n and resamples must be positive")
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    return np.random.default_rng(seed).multinomial(n, probabilities, size=resamples)


def _bootstrap_columns(
    values: NDArray[np.float64],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, float | int]], NDArray[np.int64]]:
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("bootstrap values must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite")
    weights = _bootstrap_weights(values.shape[0], resamples, seed)
    means = weights @ values / values.shape[0]
    ordered = np.sort(means, axis=0)
    low_index = int((1.0 - CONFIDENCE) / 2.0 * resamples)
    high_index = min(int((1.0 + CONFIDENCE) / 2.0 * resamples), resamples - 1)
    summaries = [
        {
            "mean": math.fsum(values[:, column]) / values.shape[0],
            "low": float(ordered[low_index, column]),
            "high": float(ordered[high_index, column]),
            "n": values.shape[0],
        }
        for column in range(values.shape[1])
    ]
    return summaries, weights


def _metric_block(
    scored: Mapping[str, Mapping[str, Sequence[float]]],
    indices: Sequence[int],
    metrics: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    columns = [(arm, metric) for arm in ARM_LABELS for metric in metrics]
    values = np.asarray(
        [[scored[arm][metric][index] for arm, metric in columns] for index in indices],
        dtype=np.float64,
    )
    summaries, _weights = _bootstrap_columns(values, resamples=resamples, seed=seed)
    out: dict[str, dict[str, dict[str, float | int]]] = {arm: {} for arm in ARM_LABELS}
    for (arm, metric), summary in zip(columns, summaries, strict=True):
        out[arm][metric] = summary
    return out


def _family_specs(
    family: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], bool]:
    if family == "efficacy-binary":
        return ((A_LABEL, H_LABEL), (E_LABEL, H_LABEL)), BINARY_METRICS, True
    if family == "efficacy-continuous":
        return ((A_LABEL, H_LABEL), (E_LABEL, H_LABEL)), CONTINUOUS_METRICS, False
    if family == "retention-binary":
        return ((G_LABEL, H_LABEL),), BINARY_METRICS, True
    if family == "retention-continuous":
        return ((G_LABEL, H_LABEL),), CONTINUOUS_METRICS, False
    raise ValueError(f"unknown family {family!r}")


def _contrast_key(row: Mapping[str, object]) -> str:
    return f"{row['arity']}\0{row['comparator']}\0{row['treatment']}\0{row['metric']}"


def _build_family(
    family: str,
    scored: Mapping[str, Mapping[str, Sequence[float]]],
    groups: Mapping[int, Sequence[int]],
    *,
    resamples: int,
    base_seed: int,
) -> list[dict[str, object]]:
    comparisons, metrics, binary = _family_specs(family)
    rows: list[dict[str, object]] = []
    raw: dict[str, float] = {}
    raw_floors: dict[str, bool] = {}
    floor = bootstrap_p_floor(resamples)

    for arity in ARITIES:
        indices = groups[arity]
        columns = [
            (comparator, treatment, metric)
            for comparator, treatment in comparisons
            for metric in metrics
        ]
        diffs = np.asarray(
            [
                [
                    scored[treatment][metric][index] - scored[comparator][metric][index]
                    for comparator, treatment, metric in columns
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        summaries, weights = _bootstrap_columns(
            diffs,
            resamples=resamples,
            seed=_derived_seed(base_seed, f"contrast:{family}:arity:{arity}"),
        )
        for column, ((comparator, treatment, metric), summary) in enumerate(
            zip(columns, summaries, strict=True)
        ):
            before = [scored[comparator][metric][index] for index in indices]
            after = [scored[treatment][metric][index] for index in indices]
            if binary:
                raw_p = mcnemar_exact(before, after)
                raw_at_floor = False
            else:
                observed = float(summary["mean"])
                centred = diffs[:, column] - observed
                null_means = weights @ centred / len(indices)
                exceedances = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
                raw_p = (exceedances + 1) / (resamples + 1)
                raw_at_floor = raw_p <= floor + 1e-15
            counts = win_loss_tie(before, after)
            row: dict[str, object] = {
                "arity": arity,
                "comparator": comparator,
                "treatment": treatment,
                "metric": metric,
                "delta": summary["mean"],
                "ci_low": summary["low"],
                "ci_high": summary["high"],
                "wins": counts.wins,
                "losses": counts.losses,
                "ties_nonzero": counts.ties_nonzero,
                "ties_zero": counts.ties_zero,
                "raw_p": raw_p,
                "raw_p_at_floor": raw_at_floor,
                "adjusted_p": 1.0,
                "adjusted_p_inherits_floor": False,
                "reject": False,
            }
            key = _contrast_key(row)
            raw[key] = raw_p
            raw_floors[key] = raw_at_floor
            rows.append(row)

    adjusted = holm_bonferroni(raw, alpha=ALPHA)
    adjusted_floors = holm_floor_flags(raw, raw_floors)
    for row in rows:
        key = _contrast_key(row)
        adjusted_p, reject = adjusted[key]
        row["adjusted_p"] = adjusted_p
        row["adjusted_p_inherits_floor"] = adjusted_floors[key]
        row["reject"] = reject
    return rows


def _input_report(inputs: HybridInputs) -> dict[str, object]:
    headline = sum(query.task == HEADLINE_TASK for query in inputs.queries)
    arity_counts = {
        str(arity): sum(len(query.gold_doc_ids) == arity for query in inputs.queries)
        for arity in ARITIES
    }
    return {
        "source_sha256": inputs.manifest["source_sha256"],
        "documents": len(inputs.doc_ids),
        "queries": len(inputs.queries),
        "headline_queries": headline,
        "arity_counts": arity_counts,
        "corpus_fingerprint": _corpus_fingerprint(inputs.corpus),
        "qrels_fingerprint": _qrels_fingerprint(inputs.queries),
        "document_provenance_fingerprint": _json_fingerprint(
            "zhrag-embedding-provenance-v1", inputs.document_provenance
        ),
        "query_provenance_fingerprint": _json_fingerprint(
            "zhrag-embedding-provenance-v1", inputs.query_provenance
        ),
        "document_matrix_fingerprint": _matrix_fingerprint(
            inputs.doc_ids,
            inputs.doc_matrix,
            namespace="zhrag-crud-document-matrix-f32-v1",
        ),
        "query_matrix_fingerprint": _matrix_fingerprint(
            [query.query_id for query in inputs.queries],
            inputs.query_matrix,
            namespace="zhrag-crud-query-matrix-f32-v1",
        ),
    }


def evaluate_hybrid_mrl1024(
    inputs: HybridInputs,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Build the frozen runs and return a strict aggregate-only report."""
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_inputs(inputs)
    rankings = build_hybrid_mrl1024_runs(inputs)
    scored = {arm: per_query_metrics(rankings[arm], inputs.queries) for arm in ARM_LABELS}
    headline_indices = [
        index for index, query in enumerate(inputs.queries) if query.task == HEADLINE_TASK
    ]
    groups = {
        arity: [
            index for index, query in enumerate(inputs.queries) if len(query.gold_doc_ids) == arity
        ]
        for arity in ARITIES
    }
    if not headline_indices or any(not groups[arity] for arity in ARITIES):
        raise ValueError("headline and every arity scope must be non-empty")

    metrics: dict[str, object] = {
        "headline": {
            "n": len(headline_indices),
            "systems": _metric_block(
                scored,
                headline_indices,
                HEADLINE_METRICS,
                resamples=resamples,
                seed=_derived_seed(seed, "metrics:headline"),
            ),
        },
        "by_arity": {
            str(arity): {
                "n": len(groups[arity]),
                "systems": _metric_block(
                    scored,
                    groups[arity],
                    ARITY_METRICS,
                    resamples=resamples,
                    seed=_derived_seed(seed, f"metrics:arity:{arity}"),
                ),
            }
            for arity in ARITIES
        },
    }
    contrasts = {
        family: _build_family(
            family,
            scored,
            groups,
            resamples=resamples,
            base_seed=seed,
        )
        for family in _FAMILY_ORDER
    }
    prompt_fingerprint = _semantic_digest("zhrag-crud-query-prompt-v1", [QUERY_PROMPT])
    report: dict[str, object] = {
        "schema": HYBRID_MRL1024_SCHEMA,
        "inputs": _input_report(inputs),
        "configuration": {
            "dense_model": CRUD_EMBEDDING_MODEL,
            "source_width": SOURCE_WIDTH,
            "effective_width": EFFECTIVE_WIDTH,
            "document_prompt": "empty",
            "query_prompt_fingerprint": prompt_fingerprint,
            "evaluation_unit": "complete news document",
            "chunk_target": "N/A",
            "retrieval_depth_per_arm": RETRIEVAL_DEPTH,
            "rrf_k": RRF_K,
            "rrf_weights": [1.0, 1.0],
            "rerank": False,
        },
        "run_fingerprints": {
            arm: _run_fingerprint(inputs.queries, rankings[arm]) for arm in ARM_LABELS
        },
        "evaluation_design": {
            "confidence": CONFIDENCE,
            "alpha": ALPHA,
            "resamples": resamples,
            "base_seed": seed,
            "seed_derivation": _SEED_DERIVATION,
            "bootstrap_method": _BOOTSTRAP_METHOD,
            "binary_test": _BINARY_TEST,
            "continuous_test": _CONTINUOUS_TEST,
            "headline_in_test_families": False,
            "families": [
                {"name": "efficacy-binary", "tests": 12},
                {"name": "efficacy-continuous", "tests": 12},
                {"name": "retention-binary", "tests": 6},
                {"name": "retention-continuous", "tests": 6},
            ],
        },
        "metrics": metrics,
        "contrasts": contrasts,
        "limitations": {
            "benchmark_role": "exploratory; all 2394 queries have informed prior analysis",
            "confidence_intervals": "pointwise query-bootstrap intervals",
            "retention_claim": "difference test only; not non-inferiority or equivalence",
            "non_significant_wording": "no difference detected; never lossless or equivalent",
            "scope": "this frozen CRUD-RAG expanded benchmark and cached vector realization",
            "paid_calls": "none; no embedding, rerank, chat, or provider fallback",
        },
    }
    validate_hybrid_mrl1024_report(report)
    return report


def _validate_summary(value: object, path: str, *, expected_n: int) -> None:
    summary = _mapping(value, path)
    _require_keys(summary, _SUMMARY_KEYS, path)
    _number(summary["mean"], f"{path}.mean", low=0.0, high=1.0)
    low = _number(summary["low"], f"{path}.low", low=0.0, high=1.0)
    high = _number(summary["high"], f"{path}.high", low=0.0, high=1.0)
    if low > high:
        raise ValueError(f"{path} has an inverted confidence interval")
    if _integer(summary["n"], f"{path}.n", minimum=1) != expected_n:
        raise ValueError(f"{path}.n does not match its scope")


def _expected_rows(family: str) -> list[tuple[int, str, str, str]]:
    comparisons, metrics, _binary = _family_specs(family)
    return [
        (arity, comparator, treatment, metric)
        for arity in ARITIES
        for comparator, treatment in comparisons
        for metric in metrics
    ]


def _validate_family(  # noqa: PLR0912
    family: str,
    value: object,
    *,
    arity_counts: Mapping[int, int],
    resamples: int,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"contrasts.{family} must be an array")
    expected = _expected_rows(family)
    if len(value) != len(expected):
        raise ValueError(f"contrasts.{family} has {len(value)} rows, expected {len(expected)}")
    raw: dict[str, float] = {}
    floors: dict[str, bool] = {}
    binary = _family_specs(family)[2]
    floor = bootstrap_p_floor(resamples)
    rows: list[Mapping[str, object]] = []
    for index, (item, descriptor) in enumerate(zip(value, expected, strict=True)):
        row = _mapping(item, f"contrasts.{family}[{index}]")
        _require_keys(row, _CONTRAST_KEYS, f"contrasts.{family}[{index}]")
        arity, comparator, treatment, metric = descriptor
        if (
            row["arity"] != arity
            or row["comparator"] != comparator
            or row["treatment"] != treatment
            or row["metric"] != metric
        ):
            raise ValueError(f"contrasts.{family}[{index}] descriptor/order drifted")
        _number(row["delta"], f"contrasts.{family}[{index}].delta", low=-1.0, high=1.0)
        ci_low = _number(
            row["ci_low"],
            f"contrasts.{family}[{index}].ci_low",
            low=-1.0,
            high=1.0,
        )
        ci_high = _number(
            row["ci_high"],
            f"contrasts.{family}[{index}].ci_high",
            low=-1.0,
            high=1.0,
        )
        if ci_low > ci_high:
            raise ValueError(f"contrasts.{family}[{index}] has an inverted CI")
        counts = [
            _integer(row[name], f"contrasts.{family}[{index}].{name}")
            for name in ("wins", "losses", "ties_nonzero", "ties_zero")
        ]
        if sum(counts) != arity_counts[arity]:
            raise ValueError(f"contrasts.{family}[{index}] W/L/T does not match n")
        raw_p = _number(row["raw_p"], f"contrasts.{family}[{index}].raw_p", low=0.0, high=1.0)
        _number(
            row["adjusted_p"],
            f"contrasts.{family}[{index}].adjusted_p",
            low=0.0,
            high=1.0,
        )
        for name in ("raw_p_at_floor", "adjusted_p_inherits_floor", "reject"):
            if not isinstance(row[name], bool):
                raise ValueError(f"contrasts.{family}[{index}].{name} must be boolean")
        if binary and (row["raw_p_at_floor"] or row["adjusted_p_inherits_floor"]):
            raise ValueError(f"contrasts.{family}[{index}] exact test cannot have a MC floor")
        if not binary and bool(row["raw_p_at_floor"]) != (raw_p <= floor + 1e-15):
            raise ValueError(f"contrasts.{family}[{index}] raw floor provenance is inconsistent")
        key = _contrast_key(row)
        raw[key] = raw_p
        floors[key] = bool(row["raw_p_at_floor"])
        rows.append(row)

    adjusted = holm_bonferroni(raw, alpha=ALPHA)
    adjusted_floors = holm_floor_flags(raw, floors)
    for index, row in enumerate(rows):
        key = _contrast_key(row)
        expected_p, expected_reject = adjusted[key]
        reported_p = _number(
            row["adjusted_p"],
            f"contrasts.{family}[{index}].adjusted_p",
            low=0.0,
            high=1.0,
        )
        if not math.isclose(reported_p, expected_p, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"contrasts.{family}[{index}] Holm p-value is inconsistent")
        if row["reject"] is not expected_reject:
            raise ValueError(f"contrasts.{family}[{index}] reject flag is inconsistent")
        if row["adjusted_p_inherits_floor"] is not adjusted_floors[key]:
            raise ValueError(f"contrasts.{family}[{index}] adjusted floor flag is inconsistent")


def _scan_forbidden_keys(value: object, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _FORBIDDEN_REPORT_KEYS:
                raise ValueError(f"{path} contains forbidden raw field {key!r}")
            _scan_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, f"{path}[{index}]")


def validate_hybrid_mrl1024_report(  # noqa: PLR0912, PLR0915
    report: Mapping[str, Any],
) -> None:
    """Reject structural drift, invalid inference metadata, and raw-content leaks."""
    _scan_forbidden_keys(report)
    _require_keys(
        report,
        {
            "configuration",
            "contrasts",
            "evaluation_design",
            "inputs",
            "limitations",
            "metrics",
            "run_fingerprints",
            "schema",
        },
        "report",
    )
    if report["schema"] != HYBRID_MRL1024_SCHEMA:
        raise ValueError("report.schema is not the H quality schema")

    inputs = _mapping(report["inputs"], "inputs")
    _require_keys(
        inputs,
        {
            "arity_counts",
            "corpus_fingerprint",
            "document_matrix_fingerprint",
            "document_provenance_fingerprint",
            "documents",
            "headline_queries",
            "qrels_fingerprint",
            "queries",
            "query_matrix_fingerprint",
            "query_provenance_fingerprint",
            "source_sha256",
        },
        "inputs",
    )
    documents = _integer(inputs["documents"], "inputs.documents", minimum=1)
    queries = _integer(inputs["queries"], "inputs.queries", minimum=1)
    headline_n = _integer(inputs["headline_queries"], "inputs.headline_queries", minimum=1)
    if documents < RETRIEVAL_DEPTH:
        raise ValueError("inputs.documents is smaller than the frozen retrieval depth")
    arity_raw = _mapping(inputs["arity_counts"], "inputs.arity_counts")
    _require_keys(arity_raw, {"1", "2", "3"}, "inputs.arity_counts")
    arity_counts = {
        arity: _integer(arity_raw[str(arity)], f"inputs.arity_counts.{arity}", minimum=1)
        for arity in ARITIES
    }
    if sum(arity_counts.values()) != queries:
        raise ValueError("inputs.arity_counts do not sum to inputs.queries")
    for name in (
        "source_sha256",
        "corpus_fingerprint",
        "qrels_fingerprint",
        "document_provenance_fingerprint",
        "query_provenance_fingerprint",
        "document_matrix_fingerprint",
        "query_matrix_fingerprint",
    ):
        _fingerprint(inputs[name], f"inputs.{name}")

    configuration = _mapping(report["configuration"], "configuration")
    _require_keys(
        configuration,
        {
            "chunk_target",
            "dense_model",
            "document_prompt",
            "effective_width",
            "evaluation_unit",
            "query_prompt_fingerprint",
            "rerank",
            "retrieval_depth_per_arm",
            "rrf_k",
            "rrf_weights",
            "source_width",
        },
        "configuration",
    )
    expected_configuration: dict[str, object] = {
        "dense_model": CRUD_EMBEDDING_MODEL,
        "source_width": SOURCE_WIDTH,
        "effective_width": EFFECTIVE_WIDTH,
        "document_prompt": "empty",
        "evaluation_unit": "complete news document",
        "chunk_target": "N/A",
        "retrieval_depth_per_arm": RETRIEVAL_DEPTH,
        "rrf_k": RRF_K,
        "rrf_weights": [1.0, 1.0],
        "rerank": False,
    }
    for key, expected in expected_configuration.items():
        if configuration[key] != expected:
            raise ValueError(f"configuration.{key} drifted from the frozen H contract")
    _fingerprint(
        configuration["query_prompt_fingerprint"],
        "configuration.query_prompt_fingerprint",
    )

    run_fingerprints = _mapping(report["run_fingerprints"], "run_fingerprints")
    _require_keys(run_fingerprints, set(ARM_LABELS), "run_fingerprints")
    for arm in ARM_LABELS:
        _fingerprint(run_fingerprints[arm], f"run_fingerprints.{arm}")

    design = _mapping(report["evaluation_design"], "evaluation_design")
    _require_keys(
        design,
        {
            "alpha",
            "base_seed",
            "binary_test",
            "bootstrap_method",
            "confidence",
            "continuous_test",
            "families",
            "headline_in_test_families",
            "resamples",
            "seed_derivation",
        },
        "evaluation_design",
    )
    if design["confidence"] != CONFIDENCE or design["alpha"] != ALPHA:
        raise ValueError("evaluation_design confidence/alpha drifted")
    resamples = _integer(design["resamples"], "evaluation_design.resamples", minimum=1)
    _integer(design["base_seed"], "evaluation_design.base_seed")
    if (
        design["seed_derivation"] != _SEED_DERIVATION
        or design["bootstrap_method"] != _BOOTSTRAP_METHOD
        or design["binary_test"] != _BINARY_TEST
        or design["continuous_test"] != _CONTINUOUS_TEST
        or design["headline_in_test_families"] is not False
    ):
        raise ValueError("evaluation_design method metadata drifted")
    expected_families = [
        {"name": "efficacy-binary", "tests": 12},
        {"name": "efficacy-continuous", "tests": 12},
        {"name": "retention-binary", "tests": 6},
        {"name": "retention-continuous", "tests": 6},
    ]
    if design["families"] != expected_families:
        raise ValueError("evaluation_design.families drifted")

    metrics = _mapping(report["metrics"], "metrics")
    _require_keys(metrics, {"by_arity", "headline"}, "metrics")
    headline = _mapping(metrics["headline"], "metrics.headline")
    _require_keys(headline, {"n", "systems"}, "metrics.headline")
    if _integer(headline["n"], "metrics.headline.n", minimum=1) != headline_n:
        raise ValueError("metrics.headline.n does not match inputs")
    headline_systems = _mapping(headline["systems"], "metrics.headline.systems")
    _require_keys(headline_systems, set(ARM_LABELS), "metrics.headline.systems")
    for arm in ARM_LABELS:
        arm_metrics = _mapping(headline_systems[arm], f"metrics.headline.systems.{arm}")
        _require_keys(arm_metrics, set(HEADLINE_METRICS), f"metrics.headline.systems.{arm}")
        for metric in HEADLINE_METRICS:
            _validate_summary(
                arm_metrics[metric],
                f"metrics.headline.systems.{arm}.{metric}",
                expected_n=headline_n,
            )

    by_arity = _mapping(metrics["by_arity"], "metrics.by_arity")
    _require_keys(by_arity, {"1", "2", "3"}, "metrics.by_arity")
    for arity in ARITIES:
        block = _mapping(by_arity[str(arity)], f"metrics.by_arity.{arity}")
        _require_keys(block, {"n", "systems"}, f"metrics.by_arity.{arity}")
        if _integer(block["n"], f"metrics.by_arity.{arity}.n", minimum=1) != arity_counts[arity]:
            raise ValueError(f"metrics.by_arity.{arity}.n does not match inputs")
        systems = _mapping(block["systems"], f"metrics.by_arity.{arity}.systems")
        _require_keys(systems, set(ARM_LABELS), f"metrics.by_arity.{arity}.systems")
        for arm in ARM_LABELS:
            arm_metrics = _mapping(systems[arm], f"metrics.by_arity.{arity}.systems.{arm}")
            _require_keys(
                arm_metrics,
                set(ARITY_METRICS),
                f"metrics.by_arity.{arity}.systems.{arm}",
            )
            for metric in ARITY_METRICS:
                _validate_summary(
                    arm_metrics[metric],
                    f"metrics.by_arity.{arity}.systems.{arm}.{metric}",
                    expected_n=arity_counts[arity],
                )

    contrasts = _mapping(report["contrasts"], "contrasts")
    _require_keys(contrasts, set(_FAMILY_ORDER), "contrasts")
    for family in _FAMILY_ORDER:
        _validate_family(
            family,
            contrasts[family],
            arity_counts=arity_counts,
            resamples=resamples,
        )

    limitations = _mapping(report["limitations"], "limitations")
    expected_limitations: Final[dict[str, object]] = {
        "benchmark_role": "exploratory; all 2394 queries have informed prior analysis",
        "confidence_intervals": "pointwise query-bootstrap intervals",
        "retention_claim": "difference test only; not non-inferiority or equivalence",
        "non_significant_wording": "no difference detected; never lossless or equivalent",
        "scope": "this frozen CRUD-RAG expanded benchmark and cached vector realization",
        "paid_calls": "none; no embedding, rerank, chat, or provider fallback",
    }
    _require_keys(limitations, set(expected_limitations), "limitations")
    if dict(limitations) != expected_limitations:
        raise ValueError("limitations drifted from the frozen wording")
