"""Offline, review-gated certification and statistics for paired Agent trials.

Hashes bind local records; they are integrity checks, not reviewer authentication.
No provider clients, filesystem writes or automatic semantic judging live here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from zhrag.agent import AGENT_MESSAGES
from zhrag.eval.agent_comparison import COMPARISON_CONTRACT
from zhrag.eval.agent_tasks import METHODS, AgentTask, task_summary, validate_tasks
from zhrag.eval.metrics import (
    bootstrap_p_floor,
    clustered_bootstrap_ci,
    clustered_paired_bootstrap_test,
    holm_bonferroni,
    holm_floor_flags,
)

REVIEW_CONTRACT = "document-investigation-review-v1"
REPORT_CONTRACT = "document-investigation-quality-v1"
PRIMARY_METRIC = "task_success"
STATUSES = frozenset(AGENT_MESSAGES) | {"context_limit"}
TERMINAL_CONTENT = frozenset({"answered", "clarification_needed", "insufficient_evidence"})


def fingerprint(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pending_review() -> dict[str, object]:
    return {
        "reviewed": False,
        "reviewer": "",
        "task_success": None,
        "supported_claims": None,
        "total_claims": None,
        "appropriate_clarification_or_refusal": None,
        "notes": "",
    }


def _object(value: object, *, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected JSON object")
    if keys is not None and set(value) != keys:
        raise ValueError("record schema mismatch")
    return dict(value)


def _sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA-256")
    return value


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer")
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("expected explicit boolean review label")
    return value


def _text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 100_000:
        raise ValueError("expected bounded text")
    value.encode("utf-8")
    if not allow_empty and not value.strip():
        raise ValueError("expected nonempty text")
    return value


def validate_result(value: object) -> dict[str, Any]:
    result = _object(
        value,
        keys={
            "method",
            "status",
            "blocks",
            "sources",
            "clarification",
            "events",
            "model_calls",
            "search_calls",
            "prompt_estimated_tokens",
            "total_seconds",
            "profile_fingerprint",
        },
    )
    if _text(result["method"]) not in METHODS or _text(result["status"]) not in STATUSES:
        raise ValueError("unknown method or status")
    _sha(result["profile_fingerprint"])
    for name in ("model_calls", "search_calls", "prompt_estimated_tokens"):
        _integer(result[name])
    elapsed = result["total_seconds"]
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed time")
    clarification = _text(result["clarification"], allow_empty=True)
    if bool(clarification.strip()) != (result["status"] == "clarification_needed"):
        raise ValueError("clarification/status mismatch")
    if not isinstance(result["events"], list) or not isinstance(result["sources"], list):
        raise ValueError("invalid event or source list")
    ids: set[int] = set()
    for raw in result["sources"]:
        row = _object(
            raw,
            keys={
                "citation_id",
                "title",
                "text",
                "doc_id",
                "source_key",
                "document_sha256",
                "source_url",
            },
        )
        key = _integer(row["citation_id"])
        if key == 0 or key in ids:
            raise ValueError("duplicate or invalid source id")
        ids.add(key)
        for name in ("text", "doc_id", "source_key"):
            _text(row[name])
        _text(row["title"], allow_empty=True)
        _sha(row["document_sha256"])
        if row["source_url"] is not None:
            _text(row["source_url"], allow_empty=True)
    blocks = result["blocks"]
    if not isinstance(blocks, list) or bool(blocks) != (result["status"] == "answered"):
        raise ValueError("blocks/status mismatch")
    for raw in blocks:
        block = _object(raw, keys={"text", "citations"})
        _text(block["text"])
        citations = block["citations"]
        if (
            not isinstance(citations, list)
            or not citations
            or any(type(key) is not int or key not in ids for key in citations)
            or len(set(citations)) != len(citations)
        ):
            raise ValueError("invalid trial citations")
    # Reject nonfinite values and invalid Unicode even in event details.
    fingerprint(result)
    return result


@dataclass(frozen=True, slots=True)
class CertifiedTrial:
    task: AgentTask
    result: Mapping[str, Any]
    sha256: str

    @property
    def method(self) -> str:
        return str(self.result["method"])


@dataclass(frozen=True, slots=True)
class CertifiedRun:
    tasks: tuple[AgentTask, ...]
    methods: tuple[str, ...]
    trials: tuple[CertifiedTrial, ...]
    manifest_sha256: str
    trials_sha256: str
    selection_sha256: str


def certify_run(  # noqa: PLR0912 - fail closed at the artifact boundary
    manifest_value: object,
    raw_trials: Sequence[object],
) -> CertifiedRun:
    manifest = _object(manifest_value)
    if manifest.get("contract") != COMPARISON_CONTRACT or manifest.get("complete") is not True:
        raise ValueError("run is incomplete or uses an unsupported contract")
    if manifest.get("human_review_required") is not True:
        raise ValueError("run must require human review")
    _sha(manifest.get("agent_profile"))
    methods = manifest.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) or method not in METHODS for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("invalid method set")
    profiles = _object(manifest.get("method_profiles"), keys=set(methods))
    for value in profiles.values():
        _sha(value)
    if "document_agent" in profiles and profiles["document_agent"] != manifest["agent_profile"]:
        raise ValueError("agent profile mismatch")
    if fingerprint(raw_trials) != _sha(manifest.get("trials_sha256")):
        raise ValueError("trial content differs from completed manifest")
    by_task: dict[str, AgentTask] = {}
    pairs: set[tuple[str, str]] = set()
    trials: list[CertifiedTrial] = []
    for raw in raw_trials:
        row = _object(raw, keys={"task", "result", "review"})
        if row["review"] != pending_review():
            raise ValueError("original trials are immutable; edit the separate review file")
        task = AgentTask.from_mapping(_object(row["task"]))
        result = validate_result(row["result"])
        method = result["method"]
        if method not in methods or result["profile_fingerprint"] != profiles[method]:
            raise ValueError("mixed method profiles")
        if task.task_id in by_task and by_task[task.task_id] != task:
            raise ValueError("task content changed across methods")
        by_task[task.task_id] = task
        pair = (task.task_id, method)
        if pair in pairs:
            raise ValueError("duplicate trial pair")
        pairs.add(pair)
        trials.append(
            CertifiedTrial(task, result, fingerprint({"task": row["task"], "result": result}))
        )
    tasks = validate_tasks(
        [
            {
                **asdict(task),
                "acceptance_criteria": list(task.acceptance_criteria),
                "reference_sources": list(task.reference_sources),
            }
            for task in by_task.values()
        ]
    )
    expected = {(task.task_id, method) for task in tasks for method in methods}
    if pairs != expected or _integer(manifest.get("trial_count")) != len(pairs):
        raise ValueError("missing trial pairs")
    # Compare full summary, including rubric/question hash and the declared split.
    selection = task_summary(tasks)
    if selection != manifest.get("selection"):
        raise ValueError("selected task set differs from manifest")
    if any(task.split != manifest.get("split") for task in tasks):
        raise ValueError("mixed splits")
    return CertifiedRun(
        tuple(sorted(tasks, key=lambda task: task.task_id)),
        tuple(methods),
        tuple(trials),
        fingerprint(manifest),
        str(manifest["trials_sha256"]),
        str(selection["sha256"]),
    )


def review_template(run: CertifiedRun) -> list[dict[str, object]]:
    return [
        {
            "contract": REVIEW_CONTRACT,
            "trial_sha256": trial.sha256,
            "review": {
                **pending_review(),
                "criteria_met": [None] * len(trial.task.acceptance_criteria),
            },
        }
        for trial in run.trials
    ]


@dataclass(frozen=True, slots=True)
class ReviewedTrial:
    trial: CertifiedTrial
    task_success: bool
    supported_claims: int
    total_claims: int
    appropriate: bool | None


def _review_trial(trial: CertifiedTrial, value: object) -> ReviewedTrial:
    review = _object(value, keys=set(pending_review()) | {"criteria_met"})
    if _bool(review["reviewed"]) is not True or not trial.task.reviewed:
        raise ValueError("both task and output must be reviewed")
    _text(review["reviewer"])
    _text(review["notes"], allow_empty=True)
    success = _bool(review["task_success"])
    criteria = review["criteria_met"]
    if not isinstance(criteria, list) or len(criteria) != len(trial.task.acceptance_criteria):
        raise ValueError("review must cover every acceptance criterion")
    labels = [_bool(value) for value in criteria]
    supported = _integer(review["supported_claims"])
    total = _integer(review["total_claims"])
    if supported > total:
        raise ValueError("supported claims exceed total claims")
    status = trial.result["status"]
    if status == "answered" and total == 0:
        raise ValueError("answered trial requires claim review")
    if status != "answered" and (supported or total):
        raise ValueError("non-answer must have zero answer claims")
    appropriate: bool | None = None
    if status in {"clarification_needed", "insufficient_evidence"}:
        appropriate = _bool(review["appropriate_clarification_or_refusal"])
    elif review["appropriate_clarification_or_refusal"] is not None:
        raise ValueError("clarification/refusal label is not applicable")
    expected_success = (
        all(labels)
        and status == trial.task.expected_status
        and supported == total
        and appropriate is not False
    )
    if success != expected_success:
        raise ValueError("task_success disagrees with rubric, expected action or claim support")
    return ReviewedTrial(trial, success, supported, total, appropriate)


def validate_reviews(run: CertifiedRun, rows: Sequence[object]) -> tuple[ReviewedTrial, ...]:
    by_sha = {trial.sha256: trial for trial in run.trials}
    reviewed: dict[str, ReviewedTrial] = {}
    for raw in rows:
        row = _object(raw, keys={"contract", "trial_sha256", "review"})
        sha = _sha(row["trial_sha256"])
        if row["contract"] != REVIEW_CONTRACT or sha not in by_sha or sha in reviewed:
            raise ValueError("review contract, pairing or duplicate error")
        reviewed[sha] = _review_trial(by_sha[sha], row["review"])
    if set(reviewed) != set(by_sha):
        raise ValueError("missing output reviews; failures cannot be dropped")
    return tuple(reviewed[trial.sha256] for trial in run.trials)


def quality_report(
    run: CertifiedRun,
    raw_reviews: Sequence[object],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    if type(resamples) is not int or not 100 <= resamples <= 100_000 or type(seed) is not int:
        raise ValueError("invalid bootstrap settings")
    reviewed = validate_reviews(run, raw_reviews)
    groups = [task.source_group for task in run.tasks]
    if len(set(groups)) < 2:
        raise ValueError("at least two source groups required for clustered inference")
    paired = {(row.trial.task.task_id, row.trial.method): row for row in reviewed}
    methods: dict[str, object] = {}
    scores: dict[str, list[float]] = {}
    for method in run.methods:
        rows = [paired[(task.task_id, method)] for task in run.tasks]
        scores[method] = [float(row.task_success) for row in rows]
        supported = sum(row.supported_claims for row in rows)
        total = sum(row.total_claims for row in rows)
        applicable = [row.appropriate for row in rows if row.appropriate is not None]
        methods[method] = {
            "task_success": asdict(
                clustered_bootstrap_ci(
                    scores[method],
                    groups,
                    resamples=resamples,
                    seed=seed,
                )
            ),
            "status_counts": dict(Counter(str(row.trial.result["status"]) for row in rows)),
            "supported_claims": supported,
            "total_claims": total,
            "conditional_claim_support": supported / total if total else None,
            "clarification_refusal_applicable": len(applicable),
            "appropriate_clarification_refusal": sum(applicable),
            "mean_model_calls": sum(row.trial.result["model_calls"] for row in rows) / len(rows),
            "mean_search_calls": sum(row.trial.result["search_calls"] for row in rows) / len(rows),
            "mean_prompt_estimated_tokens": sum(
                row.trial.result["prompt_estimated_tokens"] for row in rows
            )
            / len(rows),
            "mean_seconds": sum(row.trial.result["total_seconds"] for row in rows) / len(rows),
            "categories": {
                category: {
                    "count": sum(row.trial.task.category == category for row in rows),
                    "success_count": sum(
                        row.task_success for row in rows if row.trial.task.category == category
                    ),
                }
                for category in sorted({task.category for task in run.tasks})
            },
        }
    comparisons: dict[str, dict[str, object]] = {}
    pvalues: dict[str, float] = {}
    for baseline, treatment in combinations(run.methods, 2):
        key = f"{treatment}-minus-{baseline}"
        differences = [
            right - left for left, right in zip(scores[baseline], scores[treatment], strict=True)
        ]
        pvalues[key] = clustered_paired_bootstrap_test(
            scores[baseline],
            scores[treatment],
            groups,
            alternative="two-sided",
            resamples=resamples,
            seed=seed,
        )
        comparisons[key] = {
            "baseline": baseline,
            "treatment": treatment,
            "difference": asdict(
                clustered_bootstrap_ci(differences, groups, resamples=resamples, seed=seed)
            ),
            "p_value": pvalues[key],
        }
    adjusted = holm_bonferroni(pvalues) if pvalues else {}
    floors = (
        holm_floor_flags(
            pvalues, {key: value == bootstrap_p_floor(resamples) for key, value in pvalues.items()}
        )
        if pvalues
        else {}
    )
    for key, comparison in comparisons.items():
        comparison.update(
            p_holm=adjusted[key][0],
            reject_at_005=adjusted[key][1],
            touches_resolution_floor=floors[key],
        )
    return {
        "contract": REPORT_CONTRACT,
        "primary_metric": PRIMARY_METRIC,
        "provenance": {
            "manifest_sha256": run.manifest_sha256,
            "trials_sha256": run.trials_sha256,
            "selection_sha256": run.selection_sha256,
            "reviews_sha256": fingerprint(raw_reviews),
        },
        "task_count": len(run.tasks),
        "source_group_count": len(set(groups)),
        "split": run.tasks[0].split,
        "inference": {
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "unit": "source_group",
            "point_estimate": "task_weighted_mean",
            "test": "two-sided-centred-paired-cluster-bootstrap",
            "family": "all-method-pairs-task-success",
            "correction": "Holm",
            "p_resolution": bootstrap_p_floor(resamples),
        },
        "methods": methods,
        "comparisons": comparisons,
        "limitations": [
            "reviewer_identity_is_declared_not_authenticated",
            "claim_support_is_conditional_on_answered_trials_not_an_unconditional_quality_score",
            "cost_not_measured_prompt_tokens_are_estimates",
            "small_source_group_counts_and_single_runs_limit_inference",
            "workflow_comparison_does_not_isolate_a_single_causal_factor",
        ],
    }
