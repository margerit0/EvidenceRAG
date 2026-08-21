"""Pure helpers for reproducible rerank experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from zhrag.eval.crud import Query
from zhrag.eval.metrics import (
    BootstrapCI,
    WinLossTie,
    bootstrap_ci,
    bootstrap_p_floor,
    holm_bonferroni,
    holm_floor_flags,
    mcnemar_exact,
    paired_bootstrap_test,
    win_loss_tie,
)

__all__ = [
    "PairwiseComparison",
    "candidate_run_fingerprint",
    "missing_score_queries",
    "paired_metric_family",
    "rerank_input_fingerprint",
    "rerank_prefix",
]


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """One paired contrast after family-wise Holm correction."""

    arity: int
    comparator: str
    treatment: str
    metric: str
    delta: float
    ci: BootstrapCI
    counts: WinLossTie
    raw_p: float
    raw_p_at_floor: bool
    adjusted_p: float
    adjusted_p_inherits_floor: bool
    reject: bool


def paired_metric_family(
    scored: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    *,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[str],
    binary: bool,
    resamples: int,
) -> list[PairwiseComparison]:
    """Evaluate and Holm-correct one predeclared family of paired contrasts."""
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    floor = bootstrap_p_floor(resamples)
    rows: list[PairwiseComparison] = []
    raw: dict[str, float] = {}
    raw_floors: dict[str, bool] = {}
    for arity, per_arm in sorted(scored.items()):
        for comparator, treatment in comparisons:
            for metric in metrics:
                before = per_arm[comparator][metric]
                after = per_arm[treatment][metric]
                diffs = [
                    treatment_score - comparator_score
                    for comparator_score, treatment_score in zip(before, after, strict=True)
                ]
                key = _comparison_key(arity, comparator, treatment, metric)
                raw_p = (
                    mcnemar_exact(before, after)
                    if binary
                    else paired_bootstrap_test(before, after, resamples=resamples)
                )
                raw_at_floor = not binary and raw_p <= floor + 1e-12
                raw[key] = raw_p
                raw_floors[key] = raw_at_floor
                rows.append(
                    PairwiseComparison(
                        arity=arity,
                        comparator=comparator,
                        treatment=treatment,
                        metric=metric,
                        delta=sum(diffs) / len(diffs),
                        ci=bootstrap_ci(diffs, resamples=resamples),
                        counts=win_loss_tie(before, after),
                        raw_p=raw_p,
                        raw_p_at_floor=raw_at_floor,
                        adjusted_p=1.0,
                        adjusted_p_inherits_floor=False,
                        reject=False,
                    )
                )

    adjusted = holm_bonferroni(raw)
    adjusted_floor_flags = holm_floor_flags(raw, raw_floors)
    corrected: list[PairwiseComparison] = []
    for row in rows:
        key = _comparison_key(row.arity, row.comparator, row.treatment, row.metric)
        adjusted_p, reject = adjusted[key]
        corrected.append(
            replace(
                row,
                adjusted_p=adjusted_p,
                adjusted_p_inherits_floor=adjusted_floor_flags[key],
                reject=reject,
            )
        )
    return corrected


def _comparison_key(arity: int, comparator: str, treatment: str, metric: str) -> str:
    return f"{arity}\0{comparator}\0{treatment}\0{metric}"


def rerank_prefix(
    run: Sequence[str],
    scores: Mapping[str, float],
    *,
    depth: int,
) -> list[str]:
    """Rerank exactly the first ``depth`` ids and append the untouched tail.

    Equal reranker scores preserve the fused run's order. That avoids turning a
    numerical tie into arbitrary document-id churn and makes "no preference"
    mean "leave the established retrieval decision alone".
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if len(run) < depth:
        raise ValueError(f"run has only {len(run)} candidates, fewer than depth={depth}")
    if len(set(run)) != len(run):
        raise ValueError("run contains duplicate document ids")

    prefix = list(run[:depth])
    missing = [doc_id for doc_id in prefix if doc_id not in scores]
    if missing:
        raise ValueError(f"missing rerank scores for {len(missing)} candidates")
    for doc_id in prefix:
        if not math.isfinite(scores[doc_id]):
            raise ValueError(f"non-finite rerank score for {doc_id}")

    ranked = sorted(
        enumerate(prefix),
        key=lambda pair: (-scores[pair[1]], pair[0]),
    )
    return [doc_id for _, doc_id in ranked] + list(run[depth:])


def missing_score_queries(
    queries: Sequence[Query],
    candidates: Sequence[Sequence[str]],
    scores: Mapping[tuple[str, str], float],
    *,
    depth: int,
) -> list[str]:
    """Return queries lacking any score in their complete rerank window.

    A partially checkpointed query is deliberately considered wholly missing so
    a resumed run requests the same full batch shape and overwrites every pair.
    """
    if len(queries) != len(candidates):
        raise ValueError(f"query/candidate length mismatch: {len(queries)} vs {len(candidates)}")
    missing: list[str] = []
    for query, run in zip(queries, candidates, strict=True):
        if len(run) < depth:
            raise ValueError(
                f"{query.query_id} has only {len(run)} candidates, fewer than depth={depth}"
            )
        if any((query.query_id, doc_id) not in scores for doc_id in run[:depth]):
            missing.append(query.query_id)
    return missing


def candidate_run_fingerprint(
    queries: Sequence[Query],
    candidates: Sequence[Sequence[str]],
    *,
    depth: int,
) -> str:
    """Hash ordered query ids and ordered candidate ids without corpus text."""
    if len(queries) != len(candidates):
        raise ValueError(f"query/candidate length mismatch: {len(queries)} vs {len(candidates)}")
    digest = hashlib.sha256()
    _update(digest, "zhrag-candidate-run-v1")
    _update(digest, str(depth))
    for query, run in zip(queries, candidates, strict=True):
        if len(run) < depth:
            raise ValueError(
                f"{query.query_id} has only {len(run)} candidates, fewer than depth={depth}"
            )
        _update(digest, query.query_id)
        for doc_id in run[:depth]:
            _update(digest, doc_id)
    return digest.hexdigest()


def rerank_input_fingerprint(
    queries: Sequence[Query],
    candidates: Sequence[Sequence[str]],
    corpus: Mapping[str, str],
    *,
    depth: int,
) -> str:
    """Hash every query/document text sent to the reranker in request order."""
    if len(queries) != len(candidates):
        raise ValueError(f"query/candidate length mismatch: {len(queries)} vs {len(candidates)}")
    digest = hashlib.sha256()
    _update(digest, "zhrag-rerank-input-v1")
    _update(digest, str(depth))
    for query, run in zip(queries, candidates, strict=True):
        if len(run) < depth:
            raise ValueError(
                f"{query.query_id} has only {len(run)} candidates, fewer than depth={depth}"
            )
        _update(digest, query.query_id)
        _update(digest, query.question)
        for doc_id in run[:depth]:
            try:
                text = corpus[doc_id]
            except KeyError as exc:
                raise ValueError(f"candidate {doc_id} is absent from the corpus") from exc
            _update(digest, doc_id)
            _update(digest, text)
    return digest.hexdigest()


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
