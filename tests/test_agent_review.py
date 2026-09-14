from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from zhrag.eval.agent_comparison import COMPARISON_CONTRACT
from zhrag.eval.agent_review import (
    certify_run,
    fingerprint,
    pending_review,
    quality_report,
    review_template,
    validate_reviews,
)
from zhrag.eval.agent_tasks import METHODS, task_summary, validate_tasks


def task(index: int) -> dict[str, Any]:
    return {
        "task_id": f"synthetic-{index}",
        "question": f"合成问题-{index}",
        "category": "simple",
        "split": "dev",
        "source_group": f"group-{index // 2}",
        "expected_status": "answered",
        "acceptance_criteria": ["合成验收项"],
        "reference_sources": [f"synthetic-source-{index}"],
        "reviewed": True,
        "reviewer": "synthetic-author",
        "snapshot": "synthetic-snapshot",
    }


def result(method: str, *, status: str = "answered") -> dict[str, Any]:
    return {
        "method": method,
        "status": status,
        "blocks": [{"text": "合成回答", "citations": [1]}] if status == "answered" else [],
        "sources": [
            {
                "citation_id": 1,
                "title": "合成标题",
                "text": "合成证据",
                "doc_id": "synthetic-id",
                "source_key": "synthetic-source",
                "document_sha256": "d" * 64,
                "source_url": "https://example.invalid",
            }
        ],
        "clarification": "合成追问" if status == "clarification_needed" else "",
        "events": [],
        "model_calls": 2,
        "search_calls": 1,
        "prompt_estimated_tokens": 100,
        "total_seconds": 3.0,
        "profile_fingerprint": str(METHODS.index(method) + 1) * 64,
    }


def bundle(count: int = 6) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = [task(index) for index in range(count)]
    rows = [
        {"task": deepcopy(t), "result": result(method), "review": pending_review()}
        for t in tasks
        for method in METHODS
    ]
    manifest = {
        "contract": COMPARISON_CONTRACT,
        "complete": True,
        "human_review_required": True,
        "agent_profile": "3" * 64,
        "methods": list(METHODS),
        "split": "dev",
        "method_profiles": {method: str(index + 1) * 64 for index, method in enumerate(METHODS)},
        "trials_sha256": fingerprint(rows),
        "trial_count": len(rows),
        "selection": task_summary(validate_tasks(tasks)),
    }
    return manifest, rows


def reviews_for(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run = certify_run(manifest, rows)
    templates = review_template(run)
    for template, trial in zip(templates, run.trials, strict=True):
        index = int(trial.task.task_id.rsplit("-", 1)[1])
        success = trial.method == "document_agent" or index < 2
        template["review"] = {
            "reviewed": True,
            "reviewer": "synthetic-rater",
            "task_success": success,
            "criteria_met": [success],
            "supported_claims": 1,
            "total_claims": 1,
            "appropriate_clarification_or_refusal": None,
            "notes": "合成审核备注",
        }
    return templates


def test_clustered_report_counts_all_tasks_without_leaking_content() -> None:
    manifest, rows = bundle()
    report = quality_report(certify_run(manifest, rows), reviews_for(manifest, rows), resamples=200)
    assert report["task_count"] == 6 and report["source_group_count"] == 3
    assert report["methods"]["document_agent"]["task_success"]["mean"] == 1
    assert report["methods"]["single_rag"]["task_success"]["mean"] == pytest.approx(1 / 3)
    comparison = report["comparisons"]["document_agent-minus-single_rag"]
    assert comparison["difference"]["mean"] == pytest.approx(2 / 3)
    assert comparison["difference"]["n"] == 3  # groups, not six independent questions
    assert comparison["p_holm"] >= comparison["p_value"] > 0
    assert isinstance(comparison["reject_at_005"], bool)
    assert len(report["comparisons"]) == 3
    rendered = json.dumps(report, ensure_ascii=False)
    for private in ("合成", "synthetic-", "example.invalid", "group-"):
        assert private not in rendered
    assert report == quality_report(
        certify_run(manifest, rows), reviews_for(manifest, rows), resamples=200
    )


def test_unreviewed_templates_do_not_publish_quality() -> None:
    manifest, rows = bundle()
    run = certify_run(manifest, rows)
    templates = review_template(run)
    assert all(row["review"]["task_success"] is None for row in templates)
    with pytest.raises(ValueError, match="reviewed"):
        quality_report(run, templates, resamples=100)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "edited", "cross-profile"])
def test_trial_corruption_missing_pairs_and_mixed_profiles_rejected(mutation: str) -> None:
    manifest, rows = bundle()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "extra":
        rows.append(deepcopy(rows[0]))
    elif mutation == "edited":
        rows[0]["result"]["blocks"][0]["text"] = "changed"
    else:
        rows[0]["result"]["profile_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="content"):
        certify_run(manifest, rows)
    if mutation != "edited":
        manifest["trials_sha256"] = fingerprint(rows)
        with pytest.raises(ValueError):
            certify_run(manifest, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_calls", True),
        ("search_calls", -1),
        ("total_seconds", -1),
        ("status", "unknown"),
        ("blocks", []),
        ("clarification", "unexpected"),
        ("sources", []),
        ("profile_fingerprint", "f" * 64),
    ],
)
def test_result_validation_still_applies_with_updated_digest(field: str, value: object) -> None:
    manifest, rows = bundle()
    rows[0]["result"][field] = value
    manifest["trials_sha256"] = fingerprint(rows)
    with pytest.raises(ValueError):
        certify_run(manifest, rows)


@pytest.mark.parametrize(
    "case", ["incomplete", "legacy", "selection", "split", "task-drift", "inline-review"]
)
def test_manifest_and_immutable_task_boundaries(case: str) -> None:
    manifest, rows = bundle()
    if case == "incomplete":
        manifest["complete"] = False
    elif case == "legacy":
        manifest["contract"] = "document-investigation-comparison-v1"
    elif case == "selection":
        manifest["selection"]["sha256"] = "e" * 64
    elif case == "split":
        manifest["split"] = "test"
    elif case == "task-drift":
        rows[0]["task"]["question"] = "edited task"
    else:
        rows[0]["review"]["reviewed"] = True
    manifest["trials_sha256"] = fingerprint(rows)
    with pytest.raises(ValueError):
        certify_run(manifest, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewed", 1),
        ("reviewer", " "),
        ("criteria_met", []),
        ("criteria_met", [1]),
        ("supported_claims", 2),
        ("total_claims", True),
        ("total_claims", 0),
        ("task_success", False),
        ("task_success", 1),
        ("appropriate_clarification_or_refusal", True),
    ],
)
def test_inconsistent_or_missing_review_labels_rejected(field: str, value: object) -> None:
    manifest, rows = bundle()
    reviews = reviews_for(manifest, rows)
    reviews[0]["review"][field] = value
    with pytest.raises(ValueError):
        validate_reviews(certify_run(manifest, rows), reviews)


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong-trial", "unreviewed-task"])
def test_every_task_and_output_must_be_reviewed(case: str) -> None:
    manifest, rows = bundle()
    if case == "unreviewed-task":
        for row in rows:
            row["task"]["reviewed"] = False
        manifest["selection"] = task_summary(validate_tasks([row["task"] for row in rows[::3]]))
        manifest["trials_sha256"] = fingerprint(rows)
    reviews = reviews_for(manifest, rows)
    if case == "missing":
        reviews.pop()
    elif case == "duplicate":
        reviews[-1] = deepcopy(reviews[0])
    elif case == "wrong-trial":
        reviews[0]["trial_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_reviews(certify_run(manifest, rows), reviews)


def test_failures_count_in_denominator_and_cannot_be_successful() -> None:
    manifest, rows = bundle()
    rows[0]["result"] = result("single_rag", status="generation_failed")
    manifest["trials_sha256"] = fingerprint(rows)
    reviews = reviews_for(manifest, rows)
    review = reviews[0]["review"]
    review.update(task_success=False, criteria_met=[False], supported_claims=0, total_claims=0)
    report = quality_report(certify_run(manifest, rows), reviews, resamples=100)
    summary = report["methods"]["single_rag"]
    assert summary["task_success"]["mean"] == pytest.approx(1 / 6)
    assert summary["status_counts"]["generation_failed"] == 1
    assert summary["total_claims"] == 5 and summary["conditional_claim_support"] == 1
    review["task_success"] = True
    with pytest.raises(ValueError):
        quality_report(certify_run(manifest, rows), reviews, resamples=100)


@pytest.mark.parametrize("status", ["clarification_needed", "insufficient_evidence"])
def test_successful_nonanswer_requires_expected_action_and_appropriateness(status: str) -> None:
    manifest, rows = bundle()
    for row in rows[:3]:
        row["task"].update(expected_status=status, reference_sources=[])
        row["result"] = result(row["result"]["method"], status=status)
    manifest["selection"] = task_summary(validate_tasks([row["task"] for row in rows[::3]]))
    manifest["trials_sha256"] = fingerprint(rows)
    reviews = reviews_for(manifest, rows)
    for row in reviews[:3]:
        row["review"].update(
            supported_claims=0, total_claims=0, appropriate_clarification_or_refusal=True
        )
    report = quality_report(certify_run(manifest, rows), reviews, resamples=100)
    assert report["methods"]["document_agent"]["appropriate_clarification_refusal"] == 1
    reviews[0]["review"]["appropriate_clarification_or_refusal"] = False
    with pytest.raises(ValueError):
        quality_report(certify_run(manifest, rows), reviews, resamples=100)


def test_one_source_group_is_not_independent_inference() -> None:
    manifest, rows = bundle(count=2)
    with pytest.raises(ValueError, match="two source groups"):
        quality_report(certify_run(manifest, rows), reviews_for(manifest, rows), resamples=100)
