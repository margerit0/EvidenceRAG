"""Offline aggregate-only attribution of TiDB retrieval bad cases.

This module consumes the already-frozen runs and final qrels in memory. It
never emits a per-query attribution table: only counts, rates, fingerprints,
and fixed evaluation boundaries are allowed in the resulting report.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from zhrag.eval.pool import PooledQuery, build_pool, pool_fingerprint
from zhrag.eval.tidb_quality import (
    EVALUATION_DEPTH,
    POOL_DEPTH,
    QRELS_SCHEMA,
    QUERY_EMBEDDING_PROFILE,
    RERANK_APPLY_DEPTH,
    RERANK_PROFILE,
    RERANK_REQUEST_DEPTH,
    RRF_K,
    RUN_DEPTH,
    RUN_LABELS,
    QrelsBundle,
    parse_qrels,
    parse_runs,
    published_chunk_ids,
)
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
    "ATTRIBUTION_CATEGORIES",
    "BAD_CASES_SCHEMA",
    "CANDIDATE_DEPTH",
    "EVALUATION_CUTOFF",
    "GENERATION_STATUS",
    "classify_retrieval_outcome",
    "evaluate_tidb_bad_cases",
    "state_fingerprint",
    "validate_bad_case_report",
]

BAD_CASES_SCHEMA = "zhrag-tidb-retrieval-bad-cases-v1"
CANDIDATE_DEPTH = RUN_DEPTH
EVALUATION_CUTOFF = EVALUATION_DEPTH
ATTRIBUTION_CATEGORIES = ("recall_failure", "ranking_failure", "success")
GENERATION_STATUS = "not_evaluated"

_LIMITATIONS = (
    "aggregate-only-no-per-query-identities",
    "retrieval-service-has-no-generation-stage",
    "candidate-window-is-frozen-top-100",
    "final-output-cutoff-is-frozen-top-10",
    "frozen-qrels-and-runs-only",
)
_SENSITIVE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "credential",
        "doc_id",
        "document_id",
        "embedding",
        "exception",
        "passage",
        "password",
        "payload",
        "path",
        "provider",
        "q",
        "query",
        "query_id",
        "question",
        "raw_response",
        "raw_responses",
        "score",
        "scores",
        "secret",
        "source_key",
        "text",
        "token",
        "url",
        "vector",
        "vectors",
    }
)


def _exact_keys(value: object, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} contains a non-string key")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest") from exc
    return value


def _number(
    value: object,
    context: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{context} is outside its allowed range")
    return result


def _walk_forbidden(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key.lower() in _SENSITIVE_KEYS:
                raise ValueError(f"forbidden raw field at {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def _canonical_hash(value: object, schema: str) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("state is not canonically serializable") from exc
    digest = hashlib.sha256()
    encoded_schema = schema.encode("utf-8")
    digest.update(len(encoded_schema).to_bytes(8, "big"))
    digest.update(encoded_schema)
    encoded = rendered.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def state_fingerprint(state: Mapping[str, Any]) -> str:
    """Return an opaque fingerprint for the published state artifact."""
    return _canonical_hash(state, "zhrag-tidb-state-fingerprint-v1")


def classify_retrieval_outcome(
    run: Sequence[str],
    full_gold_doc_ids: Sequence[str],
    *,
    candidate_depth: int = CANDIDATE_DEPTH,
    evaluation_cutoff: int = EVALUATION_CUTOFF,
) -> str:
    """Classify one frozen run without retaining the query or document ids."""
    if isinstance(run, (str, bytes)) or isinstance(full_gold_doc_ids, (str, bytes)):
        raise TypeError("run and full gold ids must be sequences of strings")
    if (
        isinstance(candidate_depth, bool)
        or not isinstance(candidate_depth, int)
        or candidate_depth < 1
        or isinstance(evaluation_cutoff, bool)
        or not isinstance(evaluation_cutoff, int)
        or evaluation_cutoff < 1
        or evaluation_cutoff > candidate_depth
    ):
        raise ValueError("invalid attribution depths")
    gold = {doc_id for doc_id in full_gold_doc_ids if isinstance(doc_id, str) and doc_id}
    if not gold:
        raise ValueError("full gold set must be non-empty")
    candidates = tuple(run[:candidate_depth])
    if not set(candidates) & gold:
        return "recall_failure"
    if not set(candidates[:evaluation_cutoff]) & gold:
        return "ranking_failure"
    return "success"


def _validate_frozen_runs(
    bundle: QrelsBundle,
    runs: TiDBRuns,
    published: frozenset[str],
) -> None:
    if set(runs.query_ids) != set(bundle.by_query):
        raise ValueError("runs/qrels query coverage differs")
    for position, query_id in enumerate(runs.query_ids):
        by_label = {label: runs.runs[label][position] for label in RUN_LABELS}
        for label, run in by_label.items():
            maximum = RUN_DEPTH if label == LEXICAL_LABEL else 2 * RUN_DEPTH
            if len(run) > maximum:
                raise ValueError(f"{label}/{query_id}: run exceeds depth {maximum}")
            if set(run) - published:
                raise ValueError(f"{label}/{query_id}: run contains unpublished chunks")
        dense = by_label[DENSE_LABEL]
        if len(dense) != RUN_DEPTH:
            raise ValueError(f"{DENSE_LABEL}/{query_id}: run must contain {RUN_DEPTH} ids")
        expected_fused = tuple(
            reciprocal_rank_fusion(
                [by_label[LEXICAL_LABEL], dense],
                k=RRF_K,
                depth=RUN_DEPTH,
            )
        )
        if by_label[RRF_LABEL] != expected_fused:
            raise ValueError(f"{RRF_LABEL}/{query_id}: run is not the frozen exact RRF")
        reranked = by_label[RERANK_LABEL]
        fused = by_label[RRF_LABEL]
        if len(reranked) != len(fused) or set(reranked) != set(fused):
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank is not a fused-run permutation")
        if set(reranked[:RERANK_APPLY_DEPTH]) != set(fused[:RERANK_APPLY_DEPTH]):
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank changed the frozen window")
        if reranked[RERANK_APPLY_DEPTH:] != fused[RERANK_APPLY_DEPTH:]:
            raise ValueError(f"{RERANK_LABEL}/{query_id}: rerank changed the untouched tail")


def _reconstruct_pool(
    bundle: QrelsBundle,
    runs: TiDBRuns,
) -> tuple[str, dict[str, int], dict[str, int]]:
    positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
    units: list[PooledQuery] = []
    contribution = {label: 0 for label in RUN_LABELS}
    exclusive = {label: 0 for label in RUN_LABELS}
    for pair in bundle.pairs:
        system_unions: dict[str, tuple[str, ...]] = {}
        for label in RUN_LABELS:
            union, _ = build_pool(
                {
                    "direct": runs.runs[label][positions[pair.direct.query_id]],
                    "paraphrase": runs.runs[label][positions[pair.paraphrase.query_id]],
                },
                depth=POOL_DEPTH,
            )
            system_unions[label] = union
            contribution[label] += len(union)
        candidates, contributors = build_pool(
            system_unions,
            depth=2 * POOL_DEPTH + 1,
            required=(pair.pair_id,),
        )
        if set(candidates) != set(pair.direct.judged_doc_ids):
            raise ValueError(f"{pair.pair_id}: final judged set differs from frozen pool")
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
    return (
        pool_fingerprint(units),
        dict(sorted(contribution.items())),
        dict(sorted(exclusive.items())),
    )


def _validate_input_reports(  # noqa: PLR0912
    state: Mapping[str, Any],
    pool_report: Mapping[str, Any],
    qrels_report: Mapping[str, Any],
    bundle: QrelsBundle,
    runs: TiDBRuns,
) -> dict[str, str]:
    published = published_chunk_ids(state)
    if pool_report.get("schema") != TIDB_RUNS_SCHEMA:
        raise ValueError(f"pool report schema must be {TIDB_RUNS_SCHEMA!r}")
    if qrels_report.get("schema") != QRELS_SCHEMA:
        raise ValueError(f"qrels report schema must be {QRELS_SCHEMA!r}")
    _validate_frozen_runs(bundle, runs, published)
    for surface in bundle.by_query.values():
        referenced = {
            surface.pair_id,
            *surface.full_doc_ids,
            *surface.partial_doc_ids,
            *surface.judged_doc_ids,
        }
        if not referenced <= published:
            raise ValueError(f"{surface.query_id}: qrels contain unpublished chunks")

    queries = len(bundle.by_query)
    pairs = len(bundle.pairs)
    corpus_chunks = len(published)
    for report, context in ((pool_report, "pool report"), (qrels_report, "qrels report")):
        if _integer(report.get("queries"), f"{context}.queries", minimum=1) != queries:
            raise ValueError(f"{context}: query count differs from qrels")
        if _integer(report.get("pairs"), f"{context}.pairs", minimum=1) != pairs:
            raise ValueError(f"{context}: pair count differs from qrels")
        if (
            _integer(report.get("corpus_chunks"), f"{context}.corpus_chunks", minimum=1)
            != corpus_chunks
        ):
            raise ValueError(f"{context}: corpus count differs from published state")

    pool_query = _sha256(pool_report.get("query_set_fingerprint"), "pool query fingerprint")
    qrels_query = _sha256(
        qrels_report.get("query_set_fingerprint_sha256"),
        "qrels query fingerprint",
    )
    if bundle.query_set_fingerprint not in (pool_query, qrels_query) or pool_query != qrels_query:
        raise ValueError("query-set fingerprints disagree")
    runs_fingerprint = runs.fingerprint
    pool_runs = _sha256(pool_report.get("runs_fingerprint"), "pool runs fingerprint")
    qrels_runs = _sha256(qrels_report.get("runs_fingerprint_sha256"), "qrels runs fingerprint")
    if runs_fingerprint not in (pool_runs, qrels_runs) or pool_runs != qrels_runs:
        raise ValueError("runs fingerprints disagree")

    if _integer(pool_report.get("run_depth"), "pool report.run_depth", minimum=1) != RUN_DEPTH:
        raise ValueError("pool report does not use the frozen run depth")
    if (
        _integer(
            pool_report.get("pool_depth_per_system"), "pool report.pool_depth_per_system", minimum=1
        )
        != POOL_DEPTH
    ):
        raise ValueError("pool report does not use the frozen pool depth")
    profiles = pool_report.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("pool report profiles must be an object")
    expected_profiles = {
        "query_embedding": QUERY_EMBEDDING_PROFILE,
        "rerank": RERANK_PROFILE,
        "rrf_k": RRF_K,
        "rerank_request_depth": RERANK_REQUEST_DEPTH,
        "rerank_apply_depth": RERANK_APPLY_DEPTH,
    }
    if dict(profiles) != expected_profiles:
        raise ValueError("pool report profiles differ from the frozen configuration")

    reconstructed_pool, contribution, exclusive = _reconstruct_pool(bundle, runs)
    pool_sha = _sha256(pool_report.get("pool_fingerprint"), "pool fingerprint")
    qrels_pool_sha = _sha256(
        qrels_report.get("pool_fingerprint_sha256"),
        "qrels pool fingerprint",
    )
    if reconstructed_pool not in (pool_sha, qrels_pool_sha) or pool_sha != qrels_pool_sha:
        raise ValueError("pool fingerprints disagree")
    if pool_report.get("system_candidate_slots") != contribution:
        raise ValueError("pool report system candidate slots differ from reconstruction")
    if pool_report.get("system_exclusive_candidates") != exclusive:
        raise ValueError("pool report exclusive candidate counts differ from reconstruction")

    return {
        "state_fingerprint_sha256": state_fingerprint(state),
        "pool_fingerprint_sha256": pool_sha,
        "query_set_fingerprint_sha256": bundle.query_set_fingerprint,
        "runs_fingerprint_sha256": runs_fingerprint,
        "qrels_semantic_fingerprint_sha256": _qrels_semantic_fingerprint(qrels_report, bundle),
    }


def _qrels_semantic_fingerprint(
    qrels_report: Mapping[str, Any],
    bundle: QrelsBundle,
) -> str:
    """Return the qrels semantic digest from the report or final qrels rows."""
    value = qrels_report.get("qrels_semantic_fingerprint_sha256")
    if value is None:
        return bundle.semantic_fingerprint
    digest = _sha256(value, "qrels semantic fingerprint")
    if digest != bundle.semantic_fingerprint:
        raise ValueError("qrels semantic fingerprint differs from final qrels")
    return digest


def evaluate_tidb_bad_cases(
    *,
    state: Mapping[str, Any],
    pool_report: Mapping[str, Any],
    qrels_report: Mapping[str, Any],
    run_rows: Iterable[Mapping[str, Any]],
    qrel_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate frozen TiDB artifacts and build an aggregate-only report."""
    bundle = parse_qrels(qrel_rows)
    runs = parse_runs(run_rows)
    inputs = _validate_input_reports(state, pool_report, qrels_report, bundle, runs)
    positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
    system_categories: dict[str, dict[str, Any]] = {}
    for label in RUN_LABELS:
        counts = {category: 0 for category in ATTRIBUTION_CATEGORIES}
        for query_id, surface in bundle.by_query.items():
            category = classify_retrieval_outcome(
                runs.runs[label][positions[query_id]],
                surface.full_doc_ids,
            )
            counts[category] += 1
        system_categories[label] = {
            "queries": len(bundle.by_query),
            "counts": counts,
            "rates": {
                category: counts[category] / len(bundle.by_query)
                for category in ATTRIBUTION_CATEGORIES
            },
        }

    report: dict[str, Any] = {
        "schema": BAD_CASES_SCHEMA,
        "inputs": inputs,
        "evaluation_design": {
            "systems": list(RUN_LABELS),
            "queries": len(bundle.by_query),
            "pairs": len(bundle.pairs),
            "candidate_depth": CANDIDATE_DEPTH,
            "evaluation_cutoff": EVALUATION_CUTOFF,
            "categories": list(ATTRIBUTION_CATEGORIES),
            "attribution_unit": "one query surface per retrieval system",
        },
        "system_categories": system_categories,
        "generation": {
            "status": GENERATION_STATUS,
            "category": GENERATION_STATUS,
            "reason": "the current HTTP service exposes retrieval only",
        },
        "limitations": list(_LIMITATIONS),
    }
    validate_bad_case_report(report)
    return report


def _validate_category_rows(
    systems: Mapping[str, Any],
    *,
    queries: int,
) -> None:
    for label in RUN_LABELS:
        entry = _exact_keys(
            systems[label],
            {"queries", "counts", "rates"},
            f"system categories/{label}",
        )
        if _integer(entry["queries"], f"system categories/{label}.queries", minimum=1) != queries:
            raise ValueError(f"system categories/{label}: query count differs")
        counts = _exact_keys(
            entry["counts"],
            set(ATTRIBUTION_CATEGORIES),
            f"system categories/{label}.counts",
        )
        rates = _exact_keys(
            entry["rates"],
            set(ATTRIBUTION_CATEGORIES),
            f"system categories/{label}.rates",
        )
        total = sum(
            _integer(counts[category], f"{label}.{category}") for category in ATTRIBUTION_CATEGORIES
        )
        if total != queries:
            raise ValueError(f"system categories/{label}: counts do not sum to queries")
        for category in ATTRIBUTION_CATEGORIES:
            rate = _number(rates[category], f"{label}.{category} rate", maximum=1.0)
            expected = counts[category] / queries
            if not math.isclose(rate, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"system categories/{label}: rate is inconsistent")


def validate_bad_case_report(raw: object) -> None:
    """Validate the exact aggregate-only bad-case report contract."""
    _walk_forbidden(raw)
    root = _exact_keys(
        raw,
        {
            "schema",
            "inputs",
            "evaluation_design",
            "system_categories",
            "generation",
            "limitations",
        },
        "bad-case report",
    )
    if root["schema"] != BAD_CASES_SCHEMA:
        raise ValueError(f"bad-case report schema must be {BAD_CASES_SCHEMA!r}")
    inputs = _exact_keys(
        root["inputs"],
        {
            "state_fingerprint_sha256",
            "pool_fingerprint_sha256",
            "query_set_fingerprint_sha256",
            "runs_fingerprint_sha256",
            "qrels_semantic_fingerprint_sha256",
        },
        "bad-case inputs",
    )
    for name, value in inputs.items():
        _sha256(value, f"bad-case inputs.{name}")

    design = _exact_keys(
        root["evaluation_design"],
        {
            "systems",
            "queries",
            "pairs",
            "candidate_depth",
            "evaluation_cutoff",
            "categories",
            "attribution_unit",
        },
        "bad-case evaluation design",
    )
    if design["systems"] != list(RUN_LABELS):
        raise ValueError("bad-case systems differ from the frozen system labels")
    queries = _integer(design["queries"], "bad-case queries", minimum=1)
    pairs = _integer(design["pairs"], "bad-case pairs", minimum=1)
    if queries != 2 * pairs:
        raise ValueError("bad-case query/pair counts are inconsistent")
    if design["candidate_depth"] != CANDIDATE_DEPTH:
        raise ValueError("bad-case candidate depth drift")
    if design["evaluation_cutoff"] != EVALUATION_CUTOFF:
        raise ValueError("bad-case evaluation cutoff drift")
    if design["categories"] != list(ATTRIBUTION_CATEGORIES):
        raise ValueError("bad-case categories drift")
    if design["attribution_unit"] != "one query surface per retrieval system":
        raise ValueError("bad-case attribution unit drift")

    systems = _exact_keys(root["system_categories"], set(RUN_LABELS), "system categories")
    _validate_category_rows(systems, queries=queries)

    generation = _exact_keys(
        root["generation"],
        {"status", "category", "reason"},
        "generation status",
    )
    if generation["status"] != GENERATION_STATUS or generation["category"] != GENERATION_STATUS:
        raise ValueError("generation status must remain not_evaluated")
    if generation["reason"] != "the current HTTP service exposes retrieval only":
        raise ValueError("generation status reason drift")

    limitations = root["limitations"]
    if (
        not isinstance(limitations, Sequence)
        or isinstance(limitations, (str, bytes))
        or list(limitations) != list(_LIMITATIONS)
    ):
        raise ValueError("bad-case limitations drift")
