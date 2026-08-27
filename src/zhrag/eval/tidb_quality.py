"""Strict offline quality evaluation for the frozen TiDB retrieval benchmark.

The benchmark has two correlated query surfaces per generating chunk. Point
estimates may be read per surface, but pooled inference first reduces each pair
to one mean observation. This module keeps that rule next to the final-qrels
parser so a later report cannot accidentally bootstrap 980 surfaces as if they
were 980 independent examples.

Everything here is pure and in-memory. Artifact I/O and atomic publication live
in ``scripts/evaluate_tidb_retrieval.py``; provider modules are intentionally not
dependencies of this evaluation path.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from zhrag.eval.metrics import (
    bootstrap_p_floor,
    clustered_bootstrap_ci,
    clustered_paired_bootstrap_test,
    graded_ndcg_at_k,
    hit_at_k,
    holm_bonferroni,
    holm_floor_flags,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    win_loss_tie,
)
from zhrag.eval.pool import PooledQuery, build_pool, pool_fingerprint
from zhrag.eval.qgen import QUESTION_TYPES, VARIANTS
from zhrag.eval.tidb_runs import (
    DENSE_LABEL,
    LEXICAL_LABEL,
    RERANK_LABEL,
    RRF_LABEL,
    TIDB_RUNS_SCHEMA,
    TiDBRuns,
)
from zhrag.retrieval.fusion import reciprocal_rank_fusion

__all__ = [
    "METRIC_LABELS",
    "PRIMARY_CONTRASTS",
    "PRIMARY_METRIC",
    "QRELS_FINGERPRINT_SCHEMA",
    "RUN_LABELS",
    "TIDB_QUALITY_SCHEMA",
    "QrelPair",
    "QrelSurface",
    "QrelsBundle",
    "assign_overlap_strata",
    "evaluate_tidb_quality",
    "parse_qrels",
    "parse_runs",
    "published_chunk_ids",
    "qrels_semantic_fingerprint",
    "score_tidb_pairs",
    "validate_quality_report",
]

TIDB_QUALITY_SCHEMA = "zhrag-tidb-retrieval-quality-v2"
QRELS_FINGERPRINT_SCHEMA = "zhrag-tidb-qrels-semantics-v1"
STATE_SCHEMA = "zhrag-ingest-state-v1"
QRELS_SCHEMA = "zhrag-tidb-qrels-v1"
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
QUERY_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-query-4096-v1"
RERANK_PROFILE = "qwen3-reranker-8b-tidb-v1"

RUN_DEPTH = 100
POOL_DEPTH = 20
RRF_K = 10
RERANK_REQUEST_DEPTH = 100
RERANK_APPLY_DEPTH = 50
EVALUATION_DEPTH = 10
CONFIDENCE = 0.95
FAMILYWISE_ALPHA = 0.05
SEED_DERIVATION = "sha256-first-8-bytes(zhrag-tidb-quality-seed-v1, base-seed, namespace)"

RUN_LABELS = (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)
METRIC_LABELS = (
    "full_hit_at_1",
    "full_recall_at_1",
    "full_mrr_at_10",
    "full_binary_ndcg_at_10",
    "graded_ndcg_at_10",
)
PRIMARY_METRIC = "full_binary_ndcg_at_10"
PRIMARY_CONTRASTS = (
    ("dense-vs-bm25", LEXICAL_LABEL, DENSE_LABEL),
    ("rrf-vs-bm25", LEXICAL_LABEL, RRF_LABEL),
    ("rrf-vs-dense", DENSE_LABEL, RRF_LABEL),
    ("rerank-vs-rrf", RRF_LABEL, RERANK_LABEL),
)
OVERLAP_STRATA = ("low", "middle", "high")

_QREL_FIELDS = frozenset(
    {
        "query_id",
        "question",
        "answer",
        "gold_doc_ids",
        "gold_source_key",
        "task",
        "question_type",
        "theme",
        "bigram_containment",
        "partial_doc_ids",
        "judged_doc_ids",
        "generating_chunk_id",
        "generating_chunk_grade",
    }
)
_METRIC_DEFINITIONS = {
    "full_hit_at_1": "any operational full-answer alternative at rank 1",
    "full_recall_at_1": "fraction of operational full-answer alternatives at rank 1",
    "full_mrr_at_10": "reciprocal rank of first operational full answer",
    "full_binary_ndcg_at_10": "binary nDCG over operational full answers",
    "graded_ndcg_at_10": "nDCG with full gain 3 and partial gain 1",
}
_INFERENTIAL_UNIT = (
    "one generating-chunk pair observation per view; direct/paraphrase use one "
    "surface score and overall averages both; whole gold_source_key clusters resampled"
)
_PAIRED_TEST = "two-sided centred paired source-cluster bootstrap with add-one p-value"
_CORRECTION = "Holm-Bonferroni within each predeclared four-test family"
_OVERLAP_RULE = "within-surface tie-preserving mid-CDF tertiles; descriptive only"
_LIMITATIONS = (
    "synthetic-same-model-self-agreement",
    "pool-outside-unjudged",
    "frozen-query-set-not-used-for-tuning",
    "source-clustered-pointwise-confidence-intervals",
    "cross-source-theme-dependence-not-modelled",
    "frozen-benchmark-only",
)

type ScoreCube = dict[str, dict[str, dict[str, tuple[float, ...]]]]
type Strata = dict[str, dict[str, tuple[int, ...]]]


@dataclass(frozen=True, slots=True)
class QrelSurface:
    """One final query surface and its pair-shared operational judgements."""

    query_id: str
    task: str
    pair_id: str
    question: str
    answer: str
    full_doc_ids: tuple[str, ...]
    partial_doc_ids: tuple[str, ...]
    judged_doc_ids: tuple[str, ...]
    generating_chunk_grade: int
    bigram_containment: float
    theme: str
    question_type: str
    source_key: str


@dataclass(frozen=True, slots=True)
class QrelPair:
    """Exactly one direct and one paraphrase surface for one generating chunk."""

    pair_id: str
    direct: QrelSurface
    paraphrase: QrelSurface

    def surface(self, task: str) -> QrelSurface:
        if task == "direct":
            return self.direct
        if task == "paraphrase":
            return self.paraphrase
        raise KeyError(task)


@dataclass(frozen=True, slots=True)
class QrelsBundle:
    pairs: tuple[QrelPair, ...]
    by_query: Mapping[str, QrelSurface]
    query_set_fingerprint: str
    semantic_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ValidationSummary:
    chunks: int
    pool_total: int
    pool_min: int
    pool_mean: float
    pool_max: int
    raw_grades: Mapping[str, int]
    gold_arity: Mapping[str, int | float]
    generating_grades: Mapping[str, int]
    promotions: int
    batches: int
    run_coverage: Mapping[str, Mapping[str, int | float | Mapping[str, int]]]
    pool_fingerprint: str
    runs_fingerprint: str


def _exact_keys(row: Mapping[str, Any], expected: set[str] | frozenset[str], context: str) -> None:
    actual = set(row)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{context}: keys differ: missing={missing} extra={extra}")


def _string(row: Mapping[str, Any], name: str, context: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {name} must be a non-empty string")
    return value


def _integer(
    row: Mapping[str, Any],
    name: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context}: {name} must be an integer >= {minimum}")
    return value


def _number(row: Mapping[str, Any], name: str, context: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: {name} must be finite")
    return result


def _mapping(row: Mapping[str, Any], name: str, context: str) -> Mapping[str, Any]:
    value = row.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{context}: {name} must be an object")
    return value


def _string_ids(value: object, *, name: str, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(doc_id, str) or not doc_id for doc_id in value
    ):
        raise ValueError(f"{context}: {name} must be a list of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{context}: {name} contains duplicate ids")
    if result != tuple(sorted(result)):
        raise ValueError(f"{context}: {name} must be in canonical sorted order")
    return result


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _query_set_fingerprint(surfaces: Sequence[QrelSurface]) -> str:
    digest = hashlib.sha256()
    for surface in sorted(surfaces, key=lambda item: item.query_id):
        for value in (
            surface.query_id,
            surface.question,
            surface.answer,
            surface.task,
            surface.pair_id,
        ):
            _update(digest, value)
    return digest.hexdigest()


def qrels_semantic_fingerprint(
    surfaces: Sequence[QrelSurface],
    *,
    query_set_fingerprint: str,
) -> str:
    """Bind every metric-affecting final qrels field without retaining raw text."""
    digest = hashlib.sha256()
    _update(digest, QRELS_FINGERPRINT_SCHEMA)
    _update(digest, query_set_fingerprint)
    for surface in sorted(surfaces, key=lambda item: item.query_id):
        for name, values in (
            ("query", (surface.query_id,)),
            ("task", (surface.task,)),
            ("pair", (surface.pair_id,)),
            ("full", surface.full_doc_ids),
            ("partial", surface.partial_doc_ids),
            ("judged", surface.judged_doc_ids),
            ("generator-grade", (str(surface.generating_chunk_grade),)),
            ("overlap", (surface.bigram_containment.hex(),)),
            ("theme", (surface.theme,)),
            ("question-type", (surface.question_type,)),
            ("source", (surface.source_key,)),
        ):
            _update(digest, name)
            for value in values:
                _update(digest, value)
    return digest.hexdigest()


def _parse_surface(row: Mapping[str, Any], row_number: int) -> QrelSurface:
    context = f"qrels row {row_number}"
    _exact_keys(row, _QREL_FIELDS, context)
    query_id = _string(row, "query_id", context)
    task = _string(row, "task", context)
    if task not in VARIANTS or query_id.partition(":")[0] != task:
        raise ValueError(f"{context}: task/query-id prefix is not direct or paraphrase")
    question = _string(row, "question", context)
    answer = _string(row, "answer", context)
    pair_id = _string(row, "generating_chunk_id", context)
    full = _string_ids(row.get("gold_doc_ids"), name="gold_doc_ids", context=context)
    partial = _string_ids(
        row.get("partial_doc_ids"),
        name="partial_doc_ids",
        context=context,
    )
    judged = _string_ids(
        row.get("judged_doc_ids"),
        name="judged_doc_ids",
        context=context,
    )
    if not full:
        raise ValueError(f"{context}: gold_doc_ids must be non-empty")
    if set(full) & set(partial):
        raise ValueError(f"{context}: full and partial judgements overlap")
    if not set(full) <= set(judged) or not set(partial) <= set(judged):
        raise ValueError(f"{context}: full/partial judgements must be judged")
    if pair_id not in full or pair_id not in judged or pair_id in partial:
        raise ValueError(f"{context}: generating chunk is not verified operational gold")
    grade = _integer(row, "generating_chunk_grade", context)
    if grade not in (0, 1, 2):
        raise ValueError(f"{context}: generating_chunk_grade must be 0, 1, or 2")
    overlap = _number(row, "bigram_containment", context)
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"{context}: bigram_containment must be in [0, 1]")
    question_type = _string(row, "question_type", context)
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"{context}: unknown question_type {question_type!r}")
    return QrelSurface(
        query_id=query_id,
        task=task,
        pair_id=pair_id,
        question=question,
        answer=answer,
        full_doc_ids=full,
        partial_doc_ids=partial,
        judged_doc_ids=judged,
        generating_chunk_grade=grade,
        bigram_containment=overlap,
        theme=_string(row, "theme", context),
        question_type=question_type,
        source_key=_string(row, "gold_source_key", context),
    )


def parse_qrels(rows: Iterable[Mapping[str, Any]]) -> QrelsBundle:
    """Parse final qrels and recover complete pairs by generating chunk id."""
    surfaces: list[QrelSurface] = []
    seen_queries: set[str] = set()
    grouped: dict[str, dict[str, QrelSurface]] = {}
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"qrels row {row_number}: expected an object")
        surface = _parse_surface(row, row_number)
        if surface.query_id in seen_queries:
            raise ValueError(f"duplicate qrels query id {surface.query_id!r}")
        seen_queries.add(surface.query_id)
        pair = grouped.setdefault(surface.pair_id, {})
        if surface.task in pair:
            raise ValueError(f"{surface.pair_id}: duplicate {surface.task} surface")
        pair[surface.task] = surface
        surfaces.append(surface)
    if not surfaces:
        raise ValueError("qrels must be non-empty")

    pairs: list[QrelPair] = []
    for pair_id in sorted(grouped):
        group = grouped[pair_id]
        if set(group) != set(VARIANTS):
            raise ValueError(f"{pair_id}: qrels pair must contain direct and paraphrase")
        direct, paraphrase = group["direct"], group["paraphrase"]
        shared = (
            ("answer", direct.answer, paraphrase.answer),
            ("full judgements", direct.full_doc_ids, paraphrase.full_doc_ids),
            ("partial judgements", direct.partial_doc_ids, paraphrase.partial_doc_ids),
            ("judged set", direct.judged_doc_ids, paraphrase.judged_doc_ids),
            (
                "generating grade",
                direct.generating_chunk_grade,
                paraphrase.generating_chunk_grade,
            ),
            ("theme", direct.theme, paraphrase.theme),
            ("question type", direct.question_type, paraphrase.question_type),
            ("source", direct.source_key, paraphrase.source_key),
        )
        drift = [name for name, left, right in shared if left != right]
        if drift:
            raise ValueError(f"{pair_id}: pair-shared controls drift: {', '.join(drift)}")
        pairs.append(QrelPair(pair_id, direct, paraphrase))

    query_fingerprint = _query_set_fingerprint(surfaces)
    semantic_fingerprint = qrels_semantic_fingerprint(
        surfaces,
        query_set_fingerprint=query_fingerprint,
    )
    return QrelsBundle(
        pairs=tuple(pairs),
        by_query={surface.query_id: surface for surface in surfaces},
        query_set_fingerprint=query_fingerprint,
        semantic_fingerprint=semantic_fingerprint,
    )


def parse_runs(rows: Iterable[Mapping[str, Any]]) -> TiDBRuns:
    """Parse the frozen JSONL run rows without relying on qrels file order."""
    query_ids: list[str] = []
    per_query: dict[str, dict[str, tuple[str, ...]]] = {}
    required = set(RUN_LABELS)
    for row_number, row in enumerate(rows, 1):
        context = f"runs row {row_number}"
        if not isinstance(row, dict):
            raise ValueError(f"{context}: expected an object")
        _exact_keys(row, {"query_id", "runs"}, context)
        query_id = _string(row, "query_id", context)
        if query_id in per_query:
            raise ValueError(f"duplicate run query id {query_id!r}")
        raw_runs = _mapping(row, "runs", context)
        if set(raw_runs) != required:
            raise ValueError(f"{context}: labels differ from the frozen systems")
        parsed: dict[str, tuple[str, ...]] = {}
        for label in RUN_LABELS:
            value = raw_runs[label]
            if not isinstance(value, list) or any(
                not isinstance(doc_id, str) or not doc_id for doc_id in value
            ):
                raise ValueError(f"{context}: {label} must be a list of non-empty strings")
            parsed[label] = tuple(value)
        query_ids.append(query_id)
        per_query[query_id] = parsed
    if not query_ids:
        raise ValueError("runs must be non-empty")
    aligned = {
        label: tuple(per_query[query_id][label] for query_id in query_ids) for label in RUN_LABELS
    }
    try:
        return TiDBRuns(tuple(query_ids), aligned)
    except ValueError as exc:
        raise ValueError(f"malformed TiDB runs: {exc}") from exc


def published_chunk_ids(state: Mapping[str, Any]) -> frozenset[str]:
    """Extract the exact globally unique chunk set from published state."""
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"state schema must be {STATE_SCHEMA!r}")
    _string(state, "collection_name", "state")
    if state.get("embedding_profile") != DOCUMENT_EMBEDDING_PROFILE:
        raise ValueError("state has the wrong document embedding profile")
    documents = _mapping(state, "documents", "state")
    if not documents:
        raise ValueError("state documents must be non-empty")
    chunks: list[str] = []
    for document_id, raw_document in documents.items():
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("state document ids must be non-empty strings")
        if not isinstance(raw_document, dict):
            raise ValueError(f"state document {document_id!r} must be an object")
        raw_ids = raw_document.get("chunk_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in raw_ids)
        ):
            raise ValueError(f"state document {document_id!r} has malformed chunk ids")
        chunks.extend(raw_ids)
    if len(set(chunks)) != len(chunks):
        raise ValueError("published chunk ids must be globally unique")
    return frozenset(chunks)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    return value


def _same(name: str, *values: object) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"aggregate artifacts disagree on {name}: {values}")


def _validate_frozen_runs(
    runs: TiDBRuns,
    *,
    published: frozenset[str],
) -> dict[str, dict[str, int]]:
    depths: dict[str, list[int]] = {label: [] for label in RUN_LABELS}
    for position, query_id in enumerate(runs.query_ids):
        by_label = {label: runs.runs[label][position] for label in RUN_LABELS}
        for label, run in by_label.items():
            unknown = set(run) - published
            if unknown:
                raise ValueError(f"{label}/{query_id}: run contains unpublished chunks")
            depths[label].append(len(run))
        lexical = by_label[LEXICAL_LABEL]
        dense = by_label[DENSE_LABEL]
        fused = by_label[RRF_LABEL]
        reranked = by_label[RERANK_LABEL]
        if len(lexical) > RUN_DEPTH:
            raise ValueError(f"{LEXICAL_LABEL}/{query_id}: run exceeds depth {RUN_DEPTH}")
        if len(dense) != RUN_DEPTH:
            raise ValueError(f"{DENSE_LABEL}/{query_id}: run must contain {RUN_DEPTH} ids")
        expected_fused = tuple(reciprocal_rank_fusion([lexical, dense], k=RRF_K, depth=RUN_DEPTH))
        if fused != expected_fused:
            raise ValueError(f"{RRF_LABEL}/{query_id}: run is not the frozen exact RRF")
        if len(reranked) != len(fused) or set(reranked) != set(fused):
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank is not a fused-run permutation")
        if set(reranked[:RERANK_APPLY_DEPTH]) != set(fused[:RERANK_APPLY_DEPTH]):
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank changed the frozen window")
        if reranked[RERANK_APPLY_DEPTH:] != fused[RERANK_APPLY_DEPTH:]:
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank changed the untouched tail")
    return {
        label: {
            "min": min(values),
            "max": max(values),
            "returned_at_least_1": sum(value >= 1 for value in values),
            "returned_at_least_10": sum(value >= EVALUATION_DEPTH for value in values),
        }
        for label, values in depths.items()
    }


def _reconstruct_pool(
    bundle: QrelsBundle,
    runs: TiDBRuns,
) -> tuple[tuple[PooledQuery, ...], dict[str, int], dict[str, int]]:
    """Rebuild the exact pair pool from final qrels and frozen runs."""
    positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
    units: list[PooledQuery] = []
    contribution = {label: 0 for label in RUN_LABELS}
    exclusive = {label: 0 for label in RUN_LABELS}
    for pair in bundle.pairs:
        union_runs: dict[str, tuple[str, ...]] = {}
        for label in RUN_LABELS:
            direct_run = runs.runs[label][positions[pair.direct.query_id]]
            paraphrase_run = runs.runs[label][positions[pair.paraphrase.query_id]]
            candidates, _contributors = build_pool(
                {"direct": direct_run, "paraphrase": paraphrase_run},
                depth=POOL_DEPTH,
            )
            union_runs[label] = candidates
            contribution[label] += len(candidates)
        candidates, contributors = build_pool(
            union_runs,
            depth=2 * POOL_DEPTH + 1,
            required=(pair.pair_id,),
        )
        if set(candidates) != set(pair.direct.judged_doc_ids):
            raise ValueError(f"{pair.pair_id}: final judged set differs from the frozen pool")
        for systems in contributors.values():
            if len(systems) == 1:
                exclusive[systems[0]] += 1
        units.append(
            PooledQuery(
                chunk_id=pair.pair_id,
                query_ids=(pair.direct.query_id, pair.paraphrase.query_id),
                questions=(pair.direct.question, pair.paraphrase.question),
                answer=pair.direct.answer,
                candidates=candidates,
                contributors=contributors,
            )
        )
    return tuple(units), contribution, exclusive


def _pair_aggregates(bundle: QrelsBundle, batch_size: int) -> dict[str, Any]:
    sizes = [len(pair.direct.judged_doc_ids) for pair in bundle.pairs]
    gold_arities = [len(pair.direct.full_doc_ids) for pair in bundle.pairs]
    raw_grades = {"0": 0, "1": 0, "2": 0}
    generating = {"0": 0, "1": 0, "2": 0}
    for pair in bundle.pairs:
        surface = pair.direct
        grade = surface.generating_chunk_grade
        generating[str(grade)] += 1
        raw_two = len(surface.full_doc_ids) - int(grade != 2)
        raw_one = len(surface.partial_doc_ids) + int(grade == 1)
        raw_zero = len(surface.judged_doc_ids) - raw_two - raw_one
        if min(raw_zero, raw_one, raw_two) < 0:
            raise ValueError(f"{pair.pair_id}: final qrels cannot reconstruct raw grades")
        raw_grades["0"] += raw_zero
        raw_grades["1"] += raw_one
        raw_grades["2"] += raw_two
    return {
        "pool_total": sum(sizes),
        "pool_min": min(sizes),
        "pool_mean": sum(sizes) / len(sizes),
        "pool_max": max(sizes),
        "raw_grades": raw_grades,
        "gold_arity": {
            "min": min(gold_arities),
            "mean": sum(gold_arities) / len(gold_arities),
            "max": max(gold_arities),
        },
        "generating_grades": generating,
        "promotions": generating["0"] + generating["1"],
        "batches": sum(math.ceil(size / batch_size) for size in sizes),
    }


def _coverage(
    bundle: QrelsBundle,
    runs: TiDBRuns,
    depths: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int | float | Mapping[str, int]]]:
    positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
    result: dict[str, dict[str, int | float | Mapping[str, int]]] = {}
    for label in RUN_LABELS:
        all_returned_1 = 0
        all_returned_10 = 0
        full_1 = 0
        full_10 = 0
        for query_id, surface in bundle.by_query.items():
            run = runs.runs[label][positions[query_id]]
            judged = set(surface.judged_doc_ids)
            judged_1 = set(run[:1]) <= judged
            judged_10 = set(run[:EVALUATION_DEPTH]) <= judged
            returned_1 = len(run) >= 1
            returned_10 = len(run) >= EVALUATION_DEPTH
            all_returned_1 += int(judged_1)
            all_returned_10 += int(judged_10)
            full_1 += int(returned_1 and judged_1)
            full_10 += int(returned_10 and judged_10)
        returned_1_count = depths[label]["returned_at_least_1"]
        returned_10_count = depths[label]["returned_at_least_10"]
        queries = len(bundle.by_query)
        result[label] = {
            "queries": queries,
            "run_depth": dict(depths[label]),
            "top1_returned_at_least_k": returned_1_count,
            "top1_all_returned_judged": all_returned_1,
            "top1_full_judged": full_1,
            "top10_returned_at_least_k": returned_10_count,
            "top10_all_returned_judged": all_returned_10,
            "top10_full_judged": full_10,
            "top1_all_returned_judged_rate": all_returned_1 / queries,
            "top10_all_returned_judged_rate": all_returned_10 / queries,
        }
        if all_returned_1 != queries or all_returned_10 != queries:
            raise ValueError(f"{label}: an evaluated run prefix contains unjudged documents")
    return result


def _validate_reports(  # noqa: PLR0912, PLR0915
    bundle: QrelsBundle,
    runs: TiDBRuns,
    state: Mapping[str, Any],
    pool_report: Mapping[str, Any],
    qrels_report: Mapping[str, Any],
) -> _ValidationSummary:
    published = published_chunk_ids(state)
    if pool_report.get("schema") != TIDB_RUNS_SCHEMA:
        raise ValueError(f"pool report schema must be {TIDB_RUNS_SCHEMA!r}")
    if qrels_report.get("schema") != QRELS_SCHEMA:
        raise ValueError(f"qrels report schema must be {QRELS_SCHEMA!r}")
    if set(runs.query_ids) != set(bundle.by_query):
        raise ValueError("runs/qrels query coverage differs")
    for surface in bundle.by_query.values():
        referenced = (
            {surface.pair_id}
            | set(surface.full_doc_ids)
            | set(surface.partial_doc_ids)
            | set(surface.judged_doc_ids)
        )
        if not referenced <= published:
            raise ValueError(f"{surface.query_id}: qrels contain unpublished chunks")

    run_depth = _integer(pool_report, "run_depth", "pool report", minimum=1)
    pool_depth = _integer(pool_report, "pool_depth_per_system", "pool report", minimum=1)
    if run_depth != RUN_DEPTH or pool_depth != POOL_DEPTH:
        raise ValueError("pool report does not use the frozen run/pool depths")
    profiles = _mapping(pool_report, "profiles", "pool report")
    expected_profiles = {
        "query_embedding": QUERY_EMBEDDING_PROFILE,
        "rerank": RERANK_PROFILE,
        "rrf_k": RRF_K,
        "rerank_request_depth": RERANK_REQUEST_DEPTH,
        "rerank_apply_depth": RERANK_APPLY_DEPTH,
    }
    for name, expected in expected_profiles.items():
        if profiles.get(name) != expected:
            raise ValueError(f"pool report has unexpected profile field {name!r}")

    depths = _validate_frozen_runs(runs, published=published)
    pool_query_fingerprint = _sha256(
        pool_report.get("query_set_fingerprint"),
        "pool query-set fingerprint",
    )
    qrels_query_fingerprint = _sha256(
        qrels_report.get("query_set_fingerprint_sha256"),
        "qrels query-set fingerprint",
    )
    _same(
        "query-set fingerprint",
        bundle.query_set_fingerprint,
        pool_query_fingerprint,
        qrels_query_fingerprint,
    )
    pool_runs_fingerprint = _sha256(
        pool_report.get("runs_fingerprint"),
        "pool runs fingerprint",
    )
    qrels_runs_fingerprint = _sha256(
        qrels_report.get("runs_fingerprint_sha256"),
        "qrels runs fingerprint",
    )
    _same(
        "runs fingerprint",
        runs.fingerprint,
        pool_runs_fingerprint,
        qrels_runs_fingerprint,
    )
    units, contribution, exclusive = _reconstruct_pool(bundle, runs)
    reconstructed_pool_fingerprint = pool_fingerprint(units)
    pool_fingerprint_value = _sha256(
        pool_report.get("pool_fingerprint"),
        "pool fingerprint",
    )
    _same(
        "pool fingerprint",
        reconstructed_pool_fingerprint,
        pool_fingerprint_value,
        _sha256(
            qrels_report.get("pool_fingerprint_sha256"),
            "qrels pool fingerprint",
        ),
    )
    for field, reconstructed in (
        ("system_candidate_slots", contribution),
        ("system_exclusive_candidates", exclusive),
    ):
        reported = _mapping(pool_report, field, "pool report")
        if reported != dict(sorted(reconstructed.items())):
            raise ValueError(f"pool report {field!r} differs from reconstructed pool")

    queries = len(bundle.by_query)
    pairs = len(bundle.pairs)
    chunks = len(published)
    for report, context in ((pool_report, "pool report"), (qrels_report, "qrels report")):
        _same("query count", queries, _integer(report, "queries", context, minimum=1))
        _same("pair count", pairs, _integer(report, "pairs", context, minimum=1))
        _same("corpus count", chunks, _integer(report, "corpus_chunks", context, minimum=1))

    batch_size = _integer(qrels_report, "batch_size", "qrels report", minimum=1)
    aggregate = _pair_aggregates(bundle, batch_size)
    for name in ("pool_total", "pool_min", "pool_max"):
        report_name = {
            "pool_total": "pool_candidates_total",
            "pool_min": "pool_candidates_min",
            "pool_max": "pool_candidates_max",
        }[name]
        _same(
            report_name,
            aggregate[name],
            _integer(pool_report, report_name, "pool report", minimum=1),
        )
    _same(
        "pool candidate mean",
        aggregate["pool_mean"],
        _number(pool_report, "pool_candidates_mean", "pool report"),
    )

    reported_grades = _mapping(qrels_report, "grades", "qrels report")
    for grade in ("0", "1", "2"):
        _same(
            f"raw grade {grade}",
            aggregate["raw_grades"][grade],
            _integer(reported_grades, grade, "qrels report"),
        )
    reported_generating = _mapping(
        qrels_report,
        "generating_chunk_grades",
        "qrels report",
    )
    for grade in ("0", "1", "2"):
        _same(
            f"generating grade {grade}",
            aggregate["generating_grades"][grade],
            _integer(reported_generating, grade, "qrels report"),
        )
    disagreement = _number(
        qrels_report,
        "generating_chunk_disagreement_rate",
        "qrels report",
    )
    _same("generating disagreement rate", aggregate["promotions"] / pairs, disagreement)
    reported_arity = _mapping(qrels_report, "gold_arity", "qrels report")
    for name in ("min", "mean", "max"):
        _same(
            f"gold arity {name}",
            aggregate["gold_arity"][name],
            _number(reported_arity, name, "qrels report"),
        )
    _same("batch count", aggregate["batches"], _integer(qrels_report, "batches", "qrels report"))
    _same(
        "cache batch count",
        aggregate["batches"],
        _integer(qrels_report, "cache_batches", "qrels report"),
        _integer(qrels_report, "cache_valid_batches", "qrels report"),
    )

    coverage = _coverage(bundle, runs, depths)
    reported_coverage = _mapping(qrels_report, "run_judged_coverage", "qrels report")
    if set(reported_coverage) != set(RUN_LABELS):
        raise ValueError("qrels report coverage labels differ from the frozen systems")
    for label in RUN_LABELS:
        arm = _mapping(reported_coverage, label, "qrels report coverage")
        _same("coverage query count", queries, _integer(arm, "queries", "qrels report"))
        for depth in (1, 10):
            complete = cast(
                int,
                coverage[label][f"top{depth}_all_returned_judged"],
            )
            _same(
                f"{label} top-{depth} coverage",
                complete,
                _integer(arm, f"top{depth}_complete", "qrels report"),
            )
            _same(
                f"{label} top-{depth} coverage rate",
                complete / queries,
                _number(arm, f"top{depth}_rate", "qrels report"),
            )

    return _ValidationSummary(
        chunks=chunks,
        pool_total=int(aggregate["pool_total"]),
        pool_min=int(aggregate["pool_min"]),
        pool_mean=float(aggregate["pool_mean"]),
        pool_max=int(aggregate["pool_max"]),
        raw_grades=dict(aggregate["raw_grades"]),
        gold_arity=dict(aggregate["gold_arity"]),
        generating_grades=dict(aggregate["generating_grades"]),
        promotions=int(aggregate["promotions"]),
        batches=int(aggregate["batches"]),
        run_coverage=coverage,
        pool_fingerprint=pool_fingerprint_value,
        runs_fingerprint=runs.fingerprint,
    )


def score_tidb_pairs(bundle: QrelsBundle, runs: TiDBRuns) -> ScoreCube:
    """Return pair-aligned direct/paraphrase score vectors for every system."""
    if set(runs.query_ids) != set(bundle.by_query):
        raise ValueError("runs/qrels query coverage differs")
    positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
    cube: ScoreCube = {
        label: {task: {metric: () for metric in METRIC_LABELS} for task in VARIANTS}
        for label in RUN_LABELS
    }
    collected: dict[str, dict[str, dict[str, list[float]]]] = {
        label: {task: {metric: [] for metric in METRIC_LABELS} for task in VARIANTS}
        for label in RUN_LABELS
    }
    for pair in bundle.pairs:
        for task in VARIANTS:
            surface = pair.surface(task)
            grades = {
                **{doc_id: 1 for doc_id in surface.partial_doc_ids},
                **{doc_id: 2 for doc_id in surface.full_doc_ids},
            }
            for label in RUN_LABELS:
                run = runs.runs[label][positions[surface.query_id]]
                values = {
                    "full_hit_at_1": hit_at_k(run, surface.full_doc_ids, 1),
                    "full_recall_at_1": recall_at_k(run, surface.full_doc_ids, 1),
                    "full_mrr_at_10": mrr_at_k(run, surface.full_doc_ids, EVALUATION_DEPTH),
                    "full_binary_ndcg_at_10": ndcg_at_k(
                        run,
                        surface.full_doc_ids,
                        EVALUATION_DEPTH,
                    ),
                    "graded_ndcg_at_10": graded_ndcg_at_k(
                        run,
                        grades,
                        EVALUATION_DEPTH,
                    ),
                }
                for metric, value in values.items():
                    collected[label][task][metric].append(value)
    for label in RUN_LABELS:
        for task in VARIANTS:
            for metric in METRIC_LABELS:
                cube[label][task][metric] = tuple(collected[label][task][metric])
    return cube


def _derived_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256()
    for value in ("zhrag-tidb-quality-seed-v1", str(base_seed), namespace):
        _update(digest, value)
    return int.from_bytes(digest.digest()[:8], "big")


def _source_clusters(bundle: QrelsBundle) -> tuple[str, ...]:
    return tuple(pair.direct.source_key for pair in bundle.pairs)


def _ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    namespace: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    result = clustered_bootstrap_ci(
        values,
        clusters,
        confidence=CONFIDENCE,
        resamples=resamples,
        seed=_derived_seed(seed, namespace),
    )
    return {
        "mean": result.mean,
        "low": result.low,
        "high": result.high,
        "n": len(values),
        "clusters": result.n,
    }


def _pair_means(cube: ScoreCube, label: str, metric: str) -> tuple[float, ...]:
    direct = cube[label]["direct"][metric]
    paraphrase = cube[label]["paraphrase"][metric]
    return tuple((left + right) / 2.0 for left, right in zip(direct, paraphrase, strict=True))


def _counts(baseline: Sequence[float], treatment: Sequence[float]) -> dict[str, int]:
    result = win_loss_tie(baseline, treatment)
    return {
        "wins": result.wins,
        "losses": result.losses,
        "ties_nonzero": result.ties_nonzero,
        "ties_zero": result.ties_zero,
    }


def _comparison_rows(
    specifications: Sequence[tuple[str, str, str]],
    vectors: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    clusters: Sequence[str],
    *,
    family: str,
    metric: str,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw: dict[str, float] = {}
    raw_floors: dict[str, bool] = {}
    floor = bootstrap_p_floor(resamples)
    for row_id, comparator, treatment in specifications:
        baseline, treated = vectors[row_id]
        diffs = [right - left for left, right in zip(baseline, treated, strict=True)]
        ci = _ci(
            diffs,
            clusters,
            namespace=f"{family}:{row_id}:delta-ci",
            resamples=resamples,
            seed=seed,
        )
        p_value = clustered_paired_bootstrap_test(
            baseline,
            treated,
            clusters,
            alternative="two-sided",
            resamples=resamples,
            seed=_derived_seed(seed, f"{family}:{row_id}:p"),
        )
        raw[row_id] = p_value
        raw_floors[row_id] = math.isclose(p_value, floor, rel_tol=1e-12, abs_tol=1e-15)
        rows.append(
            {
                "id": row_id,
                "comparator": comparator,
                "treatment": treatment,
                "metric": metric,
                "delta": ci,
                "counts": _counts(baseline, treated),
                "raw_p": p_value,
                "raw_p_at_floor": raw_floors[row_id],
                "adjusted_p": 1.0,
                "adjusted_p_inherits_floor": False,
                "reject": False,
            }
        )
    adjusted = holm_bonferroni(raw, alpha=FAMILYWISE_ALPHA)
    adjusted_floors = holm_floor_flags(raw, raw_floors)
    for row in rows:
        row_id = str(row["id"])
        row["adjusted_p"], row["reject"] = adjusted[row_id]
        row["adjusted_p_inherits_floor"] = adjusted_floors[row_id]
    return rows


def assign_overlap_strata(bundle: QrelsBundle) -> Strata:
    """Assign tie-preserving within-surface mid-CDF tertiles, outcome-blind."""
    result: Strata = {}
    n = len(bundle.pairs)
    for task in VARIANTS:
        grouped: dict[float, list[int]] = {}
        for index, pair in enumerate(bundle.pairs):
            grouped.setdefault(pair.surface(task).bigram_containment, []).append(index)
        assigned: dict[str, list[int]] = {name: [] for name in OVERLAP_STRATA}
        below = 0
        for overlap in sorted(grouped):
            indices = grouped[overlap]
            midpoint_numerator = 2 * below + len(indices)
            if 3 * midpoint_numerator <= 2 * n:
                label = "low"
            elif 3 * midpoint_numerator <= 4 * n:
                label = "middle"
            else:
                label = "high"
            assigned[label].extend(indices)
            below += len(indices)
        if any(not assigned[label] for label in OVERLAP_STRATA):
            raise ValueError(f"{task}: overlap ties leave an empty tertile")
        result[task] = {label: tuple(assigned[label]) for label in OVERLAP_STRATA}
    return result


def _system_metrics(
    bundle: QrelsBundle,
    cube: ScoreCube,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    clusters = _source_clusters(bundle)
    for label in RUN_LABELS:
        views: dict[str, Any] = {}
        for view in (*VARIANTS, "overall"):
            metrics: dict[str, Any] = {}
            for metric in METRIC_LABELS:
                values = (
                    _pair_means(cube, label, metric)
                    if view == "overall"
                    else cube[label][view][metric]
                )
                metrics[metric] = _ci(
                    values,
                    clusters,
                    namespace=f"absolute:{label}:{view}:{metric}",
                    resamples=resamples,
                    seed=seed,
                )
            views[view] = metrics
        result[label] = views
    return result


def _overlap_report(
    bundle: QrelsBundle,
    cube: ScoreCube,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    assigned = assign_overlap_strata(bundle)
    tasks: dict[str, Any] = {}
    for task in VARIANTS:
        strata: dict[str, Any] = {}
        for stratum in OVERLAP_STRATA:
            indices = assigned[task][stratum]
            overlaps = [bundle.pairs[index].surface(task).bigram_containment for index in indices]
            strata[stratum] = {
                "n": len(indices),
                "overlap_min": min(overlaps),
                "overlap_max": max(overlaps),
                "systems": {
                    label: _ci(
                        [cube[label][task][PRIMARY_METRIC][index] for index in indices],
                        [bundle.pairs[index].direct.source_key for index in indices],
                        namespace=f"overlap:{task}:{stratum}:{label}",
                        resamples=resamples,
                        seed=seed,
                    )
                    for label in RUN_LABELS
                },
            }
        tasks[task] = strata
    return {
        "rule": _OVERLAP_RULE,
        "metric": PRIMARY_METRIC,
        "tasks": tasks,
    }


def evaluate_tidb_quality(
    *,
    state: Mapping[str, Any],
    pool_report: Mapping[str, Any],
    qrels_report: Mapping[str, Any],
    run_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate frozen local artifacts and build one aggregate-only report."""
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 63) - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    bundle = parse_qrels(qrel_rows)
    runs = parse_runs(run_rows)
    summary = _validate_reports(bundle, runs, state, pool_report, qrels_report)
    cube = score_tidb_pairs(bundle, runs)

    cluster_ids = _source_clusters(bundle)
    cluster_count = len(set(cluster_ids))
    cluster_sizes: dict[str, int] = {}
    for cluster_id in cluster_ids:
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    primary_vectors = {
        row_id: (
            _pair_means(cube, comparator, PRIMARY_METRIC),
            _pair_means(cube, treatment, PRIMARY_METRIC),
        )
        for row_id, comparator, treatment in PRIMARY_CONTRASTS
    }
    primary = _comparison_rows(
        PRIMARY_CONTRASTS,
        primary_vectors,
        cluster_ids,
        family="primary-system-efficacy",
        metric=PRIMARY_METRIC,
        resamples=resamples,
        seed=seed,
    )
    robustness_specs = tuple((label, "direct", "paraphrase") for label in RUN_LABELS)
    robustness_vectors = {
        label: (
            cube[label]["direct"][PRIMARY_METRIC],
            cube[label]["paraphrase"][PRIMARY_METRIC],
        )
        for label in RUN_LABELS
    }
    robustness = _comparison_rows(
        robustness_specs,
        robustness_vectors,
        cluster_ids,
        family="surface-robustness",
        metric=PRIMARY_METRIC,
        resamples=resamples,
        seed=seed,
    )

    report: dict[str, Any] = {
        "schema": TIDB_QUALITY_SCHEMA,
        "inputs": {
            "state_schema": STATE_SCHEMA,
            "pool_schema": TIDB_RUNS_SCHEMA,
            "qrels_schema": QRELS_SCHEMA,
            "query_set_fingerprint_sha256": bundle.query_set_fingerprint,
            "pool_fingerprint_sha256": summary.pool_fingerprint,
            "runs_fingerprint_sha256": summary.runs_fingerprint,
            "qrels_semantic_fingerprint_schema": QRELS_FINGERPRINT_SCHEMA,
            "qrels_semantic_fingerprint_sha256": bundle.semantic_fingerprint,
        },
        "evaluation_design": {
            "systems": list(RUN_LABELS),
            "queries": len(bundle.by_query),
            "pairs": len(bundle.pairs),
            "views": ["direct", "paraphrase", "overall"],
            "cutoffs": {"hit": 1, "ranking": EVALUATION_DEPTH},
            "metrics": dict(_METRIC_DEFINITIONS),
            "primary_endpoint": PRIMARY_METRIC,
            "inferential_unit": _INFERENTIAL_UNIT,
            "clusters": cluster_count,
            "cluster_key": "gold_source_key",
            "cluster_size": {
                "min": min(cluster_sizes.values()),
                "mean": len(bundle.pairs) / cluster_count,
                "max": max(cluster_sizes.values()),
            },
            "confidence": CONFIDENCE,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "resamples": resamples,
            "base_seed": seed,
            "seed_derivation": SEED_DERIVATION,
            "paired_test": _PAIRED_TEST,
            "correction": _CORRECTION,
            "graded_gains": {"full": 3, "partial": 1, "irrelevant": 0},
            "primary_contrast_order": [row_id for row_id, _left, _right in PRIMARY_CONTRASTS],
            "robustness_order": list(RUN_LABELS),
        },
        "qrels_quality": {
            "corpus_chunks": summary.chunks,
            "queries": len(bundle.by_query),
            "pairs": len(bundle.pairs),
            "pool_candidates": {
                "total": summary.pool_total,
                "min": summary.pool_min,
                "mean": summary.pool_mean,
                "max": summary.pool_max,
            },
            "raw_grades": dict(summary.raw_grades),
            "gold_arity": dict(summary.gold_arity),
            "generating_chunk_grades": dict(summary.generating_grades),
            "promoted_generating_chunks": summary.promotions,
            "judging_batches": summary.batches,
            "run_judged_coverage": {
                label: dict(summary.run_coverage[label]) for label in RUN_LABELS
            },
        },
        "system_metrics": _system_metrics(
            bundle,
            cube,
            resamples=resamples,
            seed=seed,
        ),
        "primary_contrasts": primary,
        "surface_robustness": robustness,
        "overlap_strata": _overlap_report(
            bundle,
            cube,
            resamples=resamples,
            seed=seed,
        ),
        "limitations": list(_LIMITATIONS),
    }
    validate_quality_report(report)
    return report


def _validate_ci(
    value: object,
    *,
    context: str,
    expected_n: int,
    expected_clusters: int | None,
    score: bool,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: confidence interval must be an object")
    _exact_keys(value, {"mean", "low", "high", "n", "clusters"}, context)
    mean = _number(value, "mean", context)
    low = _number(value, "low", context)
    high = _number(value, "high", context)
    n = _integer(value, "n", context, minimum=1)
    clusters = _integer(value, "clusters", context, minimum=1)
    if n != expected_n or low > high:
        raise ValueError(f"{context}: invalid interval bounds or n")
    if clusters > n or (expected_clusters is not None and clusters != expected_clusters):
        raise ValueError(f"{context}: invalid source-cluster count")
    lower, upper = (0.0, 1.0) if score else (-1.0, 1.0)
    if any(number < lower - 1e-12 or number > upper + 1e-12 for number in (mean, low, high)):
        raise ValueError(f"{context}: interval lies outside the metric range")
    return value


def _validate_comparison_family(  # noqa: PLR0912
    rows: object,
    *,
    specifications: Sequence[tuple[str, str, str]],
    metric: str,
    n: int,
    clusters: int,
    resamples: int,
    context: str,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(specifications):
        raise ValueError(f"{context}: expected {len(specifications)} rows")
    expected_keys = {
        "id",
        "comparator",
        "treatment",
        "metric",
        "delta",
        "counts",
        "raw_p",
        "raw_p_at_floor",
        "adjusted_p",
        "adjusted_p_inherits_floor",
        "reject",
    }
    raw: dict[str, float] = {}
    raw_floors: dict[str, bool] = {}
    floor = bootstrap_p_floor(resamples)
    for row, specification in zip(rows, specifications, strict=True):
        if not isinstance(row, dict):
            raise ValueError(f"{context}: comparison row must be an object")
        _exact_keys(row, expected_keys, context)
        row_id, comparator, treatment = specification
        if (
            row.get("id") != row_id
            or row.get("comparator") != comparator
            or row.get("treatment") != treatment
            or row.get("metric") != metric
        ):
            raise ValueError(f"{context}: comparison identity/order drift")
        _validate_ci(
            row.get("delta"),
            context=f"{context}/{row_id}/delta",
            expected_n=n,
            expected_clusters=clusters,
            score=False,
        )
        counts = row.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"{context}/{row_id}: counts must be an object")
        _exact_keys(counts, {"wins", "losses", "ties_nonzero", "ties_zero"}, context)
        if sum(_integer(counts, name, context) for name in counts) != n:
            raise ValueError(f"{context}/{row_id}: direction counts do not sum to n")
        raw_p = _number(row, "raw_p", context)
        adjusted_p = _number(row, "adjusted_p", context)
        if not 0.0 <= raw_p <= 1.0 or not 0.0 <= adjusted_p <= 1.0:
            raise ValueError(f"{context}/{row_id}: p-values must be in [0, 1]")
        for flag in ("raw_p_at_floor", "adjusted_p_inherits_floor", "reject"):
            if not isinstance(row.get(flag), bool):
                raise ValueError(f"{context}/{row_id}: {flag} must be boolean")
        at_floor = math.isclose(raw_p, floor, rel_tol=1e-12, abs_tol=1e-15)
        if row["raw_p_at_floor"] != at_floor:
            raise ValueError(f"{context}/{row_id}: raw floor provenance is inconsistent")
        raw[row_id] = raw_p
        raw_floors[row_id] = at_floor
    adjusted = holm_bonferroni(raw, alpha=FAMILYWISE_ALPHA)
    inherited = holm_floor_flags(raw, raw_floors)
    for row in rows:
        row_id = str(row["id"])
        expected_p, expected_reject = adjusted[row_id]
        if not math.isclose(
            float(row["adjusted_p"]),
            expected_p,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{context}/{row_id}: Holm-adjusted p-value is inconsistent")
        if (
            row["reject"] != expected_reject
            or row["adjusted_p_inherits_floor"] != inherited[row_id]
        ):
            raise ValueError(f"{context}/{row_id}: Holm decision/provenance is inconsistent")


def validate_quality_report(report: Mapping[str, Any]) -> None:  # noqa: PLR0912, PLR0915
    """Strictly validate the aggregate-only quality-report contract."""
    if not isinstance(report, dict):
        raise ValueError("quality report must be an object")
    _exact_keys(
        report,
        {
            "schema",
            "inputs",
            "evaluation_design",
            "qrels_quality",
            "system_metrics",
            "primary_contrasts",
            "surface_robustness",
            "overlap_strata",
            "limitations",
        },
        "quality report",
    )
    if report.get("schema") != TIDB_QUALITY_SCHEMA:
        raise ValueError(f"quality report schema must be {TIDB_QUALITY_SCHEMA!r}")
    inputs = _mapping(report, "inputs", "quality report")
    _exact_keys(
        inputs,
        {
            "state_schema",
            "pool_schema",
            "qrels_schema",
            "query_set_fingerprint_sha256",
            "pool_fingerprint_sha256",
            "runs_fingerprint_sha256",
            "qrels_semantic_fingerprint_schema",
            "qrels_semantic_fingerprint_sha256",
        },
        "quality report inputs",
    )
    expected_schemas = {
        "state_schema": STATE_SCHEMA,
        "pool_schema": TIDB_RUNS_SCHEMA,
        "qrels_schema": QRELS_SCHEMA,
        "qrels_semantic_fingerprint_schema": QRELS_FINGERPRINT_SCHEMA,
    }
    for name, expected in expected_schemas.items():
        if inputs.get(name) != expected:
            raise ValueError(f"quality report inputs: {name} drift")
    for name in (
        "query_set_fingerprint_sha256",
        "pool_fingerprint_sha256",
        "runs_fingerprint_sha256",
        "qrels_semantic_fingerprint_sha256",
    ):
        _sha256(inputs.get(name), name)

    design = _mapping(report, "evaluation_design", "quality report")
    _exact_keys(
        design,
        {
            "systems",
            "queries",
            "pairs",
            "views",
            "cutoffs",
            "metrics",
            "primary_endpoint",
            "inferential_unit",
            "clusters",
            "cluster_key",
            "cluster_size",
            "confidence",
            "familywise_alpha",
            "resamples",
            "base_seed",
            "seed_derivation",
            "paired_test",
            "correction",
            "graded_gains",
            "primary_contrast_order",
            "robustness_order",
        },
        "evaluation design",
    )
    if design.get("systems") != list(RUN_LABELS) or design.get("views") != [
        "direct",
        "paraphrase",
        "overall",
    ]:
        raise ValueError("evaluation design has unexpected systems or views")
    queries = _integer(design, "queries", "evaluation design", minimum=1)
    pairs = _integer(design, "pairs", "evaluation design", minimum=1)
    clusters = _integer(design, "clusters", "evaluation design", minimum=1)
    if queries != 2 * pairs:
        raise ValueError("evaluation design does not contain complete query pairs")
    if clusters > pairs or design.get("cluster_key") != "gold_source_key":
        raise ValueError("evaluation design source clustering drift")
    cluster_size = _mapping(design, "cluster_size", "evaluation design")
    _exact_keys(cluster_size, {"min", "mean", "max"}, "evaluation cluster size")
    cluster_min = _integer(cluster_size, "min", "evaluation cluster size", minimum=1)
    cluster_mean = _number(cluster_size, "mean", "evaluation cluster size")
    cluster_max = _integer(cluster_size, "max", "evaluation cluster size", minimum=1)
    if not cluster_min <= cluster_mean <= cluster_max or not math.isclose(
        cluster_mean * clusters, pairs
    ):
        raise ValueError("evaluation design cluster-size aggregates do not reconcile")
    if design.get("primary_endpoint") != PRIMARY_METRIC:
        raise ValueError("evaluation design primary endpoint drift")
    if _number(design, "confidence", "evaluation design") != CONFIDENCE:
        raise ValueError("evaluation design confidence drift")
    if _number(design, "familywise_alpha", "evaluation design") != FAMILYWISE_ALPHA:
        raise ValueError("evaluation design alpha drift")
    resamples = _integer(design, "resamples", "evaluation design", minimum=1)
    _integer(design, "base_seed", "evaluation design")
    if design.get("seed_derivation") != SEED_DERIVATION:
        raise ValueError("evaluation design seed derivation drift")
    if design.get("primary_contrast_order") != [row[0] for row in PRIMARY_CONTRASTS]:
        raise ValueError("evaluation design primary family drift")
    if design.get("robustness_order") != list(RUN_LABELS):
        raise ValueError("evaluation design robustness family drift")
    if design.get("inferential_unit") != _INFERENTIAL_UNIT:
        raise ValueError("evaluation design inferential unit drift")
    if design.get("paired_test") != _PAIRED_TEST or design.get("correction") != _CORRECTION:
        raise ValueError("evaluation design inferential method drift")
    cutoffs = _mapping(design, "cutoffs", "evaluation design")
    if cutoffs != {"hit": 1, "ranking": EVALUATION_DEPTH}:
        raise ValueError("evaluation design cutoffs drift")
    metrics = _mapping(design, "metrics", "evaluation design")
    if metrics != _METRIC_DEFINITIONS:
        raise ValueError("evaluation design metric definitions drift")
    gains = _mapping(design, "graded_gains", "evaluation design")
    if gains != {"full": 3, "partial": 1, "irrelevant": 0}:
        raise ValueError("evaluation design graded gains drift")

    qrels = _mapping(report, "qrels_quality", "quality report")
    _exact_keys(
        qrels,
        {
            "corpus_chunks",
            "queries",
            "pairs",
            "pool_candidates",
            "raw_grades",
            "gold_arity",
            "generating_chunk_grades",
            "promoted_generating_chunks",
            "judging_batches",
            "run_judged_coverage",
        },
        "qrels quality",
    )
    _same("quality query count", queries, _integer(qrels, "queries", "qrels quality"))
    _same("quality pair count", pairs, _integer(qrels, "pairs", "qrels quality"))
    _integer(qrels, "corpus_chunks", "qrels quality", minimum=1)
    _integer(qrels, "promoted_generating_chunks", "qrels quality")
    _integer(qrels, "judging_batches", "qrels quality", minimum=1)
    pool = _mapping(qrels, "pool_candidates", "qrels quality")
    _exact_keys(pool, {"total", "min", "mean", "max"}, "pool candidates")
    total = _integer(pool, "total", "pool candidates", minimum=1)
    minimum = _integer(pool, "min", "pool candidates", minimum=1)
    maximum = _integer(pool, "max", "pool candidates", minimum=1)
    mean = _number(pool, "mean", "pool candidates")
    if not minimum <= mean <= maximum or not math.isclose(mean * pairs, total):
        raise ValueError("pool candidate aggregates do not reconcile")
    raw_grades = _mapping(qrels, "raw_grades", "qrels quality")
    if (
        set(raw_grades) != {"0", "1", "2"}
        or sum(_integer(raw_grades, grade, "raw grades") for grade in raw_grades) != total
    ):
        raise ValueError("raw grade aggregates do not reconcile")
    generating = _mapping(qrels, "generating_chunk_grades", "qrels quality")
    if (
        set(generating) != {"0", "1", "2"}
        or sum(_integer(generating, grade, "generating grades") for grade in generating) != pairs
    ):
        raise ValueError("generating grade aggregates do not reconcile")
    if _integer(qrels, "promoted_generating_chunks", "qrels quality") != (
        _integer(generating, "0", "generating grades")
        + _integer(generating, "1", "generating grades")
    ):
        raise ValueError("promoted generating count is inconsistent")
    arity = _mapping(qrels, "gold_arity", "qrels quality")
    if set(arity) != {"min", "mean", "max"}:
        raise ValueError("gold arity keys drift")
    arity_min = _number(arity, "min", "gold arity")
    arity_mean = _number(arity, "mean", "gold arity")
    arity_max = _number(arity, "max", "gold arity")
    if not 1 <= arity_min <= arity_mean <= arity_max:
        raise ValueError("gold arity aggregates are invalid")
    coverage = _mapping(qrels, "run_judged_coverage", "qrels quality")
    if set(coverage) != set(RUN_LABELS):
        raise ValueError("quality coverage labels drift")
    coverage_keys = {
        "queries",
        "run_depth",
        "top1_returned_at_least_k",
        "top1_all_returned_judged",
        "top1_full_judged",
        "top10_returned_at_least_k",
        "top10_all_returned_judged",
        "top10_full_judged",
        "top1_all_returned_judged_rate",
        "top10_all_returned_judged_rate",
    }
    for label in RUN_LABELS:
        arm = _mapping(coverage, label, "quality coverage")
        _exact_keys(arm, coverage_keys, f"quality coverage/{label}")
        _same("coverage queries", queries, _integer(arm, "queries", "quality coverage"))
        depth = _mapping(arm, "run_depth", "quality coverage")
        _exact_keys(
            depth,
            {"min", "max", "returned_at_least_1", "returned_at_least_10"},
            "quality run depth",
        )
        for name in ("min", "max", "returned_at_least_1", "returned_at_least_10"):
            _integer(depth, name, "quality run depth")
        for cutoff in (1, 10):
            returned = _integer(arm, f"top{cutoff}_returned_at_least_k", "quality coverage")
            judged = _integer(arm, f"top{cutoff}_all_returned_judged", "quality coverage")
            full = _integer(arm, f"top{cutoff}_full_judged", "quality coverage")
            if full != min(returned, judged) or judged != queries:
                raise ValueError(f"quality coverage/{label}: top-{cutoff} is inconsistent")
            rate = _number(
                arm,
                f"top{cutoff}_all_returned_judged_rate",
                "quality coverage",
            )
            if not math.isclose(rate, judged / queries):
                raise ValueError(f"quality coverage/{label}: top-{cutoff} rate is inconsistent")

    system_metrics = _mapping(report, "system_metrics", "quality report")
    if set(system_metrics) != set(RUN_LABELS):
        raise ValueError("system metric labels drift")
    for label in RUN_LABELS:
        views = _mapping(system_metrics, label, "system metrics")
        if set(views) != {"direct", "paraphrase", "overall"}:
            raise ValueError(f"system metrics/{label}: view drift")
        for view in ("direct", "paraphrase", "overall"):
            metric_rows = _mapping(views, view, "system metrics")
            if set(metric_rows) != set(METRIC_LABELS):
                raise ValueError(f"system metrics/{label}/{view}: metric drift")
            for metric in METRIC_LABELS:
                _validate_ci(
                    metric_rows[metric],
                    context=f"system metrics/{label}/{view}/{metric}",
                    expected_n=pairs,
                    expected_clusters=clusters,
                    score=True,
                )

    _validate_comparison_family(
        report.get("primary_contrasts"),
        specifications=PRIMARY_CONTRASTS,
        metric=PRIMARY_METRIC,
        n=pairs,
        clusters=clusters,
        resamples=resamples,
        context="primary contrasts",
    )
    robustness_specs = tuple((label, "direct", "paraphrase") for label in RUN_LABELS)
    _validate_comparison_family(
        report.get("surface_robustness"),
        specifications=robustness_specs,
        metric=PRIMARY_METRIC,
        n=pairs,
        clusters=clusters,
        resamples=resamples,
        context="surface robustness",
    )

    overlap = _mapping(report, "overlap_strata", "quality report")
    _exact_keys(overlap, {"rule", "metric", "tasks"}, "overlap strata")
    if overlap.get("metric") != PRIMARY_METRIC or overlap.get("rule") != _OVERLAP_RULE:
        raise ValueError("overlap strata method drift")
    tasks = _mapping(overlap, "tasks", "overlap strata")
    if set(tasks) != set(VARIANTS):
        raise ValueError("overlap task labels drift")
    for task in VARIANTS:
        strata = _mapping(tasks, task, "overlap strata")
        if set(strata) != set(OVERLAP_STRATA):
            raise ValueError(f"overlap/{task}: strata drift")
        task_n = 0
        prior_max = -math.inf
        for stratum in OVERLAP_STRATA:
            row = _mapping(strata, stratum, "overlap strata")
            _exact_keys(row, {"n", "overlap_min", "overlap_max", "systems"}, "overlap row")
            stratum_n = _integer(row, "n", "overlap row", minimum=1)
            low = _number(row, "overlap_min", "overlap row")
            high = _number(row, "overlap_max", "overlap row")
            if not 0 <= low <= high <= 1 or low < prior_max:
                raise ValueError(f"overlap/{task}: invalid ordered ranges")
            prior_max = high
            systems = _mapping(row, "systems", "overlap row")
            if set(systems) != set(RUN_LABELS):
                raise ValueError(f"overlap/{task}/{stratum}: system labels drift")
            stratum_clusters: int | None = None
            for label in RUN_LABELS:
                interval = _validate_ci(
                    systems[label],
                    context=f"overlap/{task}/{stratum}/{label}",
                    expected_n=stratum_n,
                    expected_clusters=None,
                    score=True,
                )
                interval_clusters = _integer(
                    interval,
                    "clusters",
                    f"overlap/{task}/{stratum}/{label}",
                    minimum=1,
                )
                if interval_clusters > clusters:
                    raise ValueError(
                        f"overlap/{task}/{stratum}: source-cluster count exceeds total clusters"
                    )
                if interval_clusters > clusters:
                    raise ValueError(
                        f"overlap/{task}/{stratum}: source-cluster count exceeds total clusters"
                    )
                if stratum_clusters is None:
                    stratum_clusters = interval_clusters
                elif interval_clusters != stratum_clusters:
                    raise ValueError(
                        f"overlap/{task}/{stratum}: source-cluster counts differ by system"
                    )
            task_n += stratum_n
        if task_n != pairs:
            raise ValueError(f"overlap/{task}: stratum counts do not sum to pairs")

    if report.get("limitations") != list(_LIMITATIONS):
        raise ValueError("quality report limitations drift")
