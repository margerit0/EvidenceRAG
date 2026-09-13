from __future__ import annotations

from copy import deepcopy

import pytest

from zhrag.eval.agent_tasks import draft_tasks, task_summary, validate_tasks


def test_drafts_have_no_human_review_or_frozen_quality_claim() -> None:
    rows = draft_tasks()
    summary = task_summary(validate_tasks(rows))
    assert summary["reviewed_count"] == 0 and summary["evaluation_ready"] is False
    assert set(summary["categories"]) == {
        "simple",
        "multi_document",
        "clarification",
        "unanswerable",
    }
    assert "question" not in summary and "reviewer" not in summary
    with pytest.raises(ValueError, match="unreviewed"):
        validate_tasks(rows, require_reviewed=True)


def test_review_gate_and_fingerprint_bind_rubric_snapshot_and_content() -> None:
    row = draft_tasks()[0]
    row.update(
        reviewed=True,
        reviewer="synthetic-reviewer",
        snapshot="synthetic-snapshot",
        acceptance_criteria=["Synthetic criterion"],
        reference_sources=["synthetic-source"],
    )
    tasks = validate_tasks([row], require_reviewed=True)
    first = task_summary(tasks)
    assert first["evaluation_ready"] is True
    changed = deepcopy(row)
    changed["acceptance_criteria"] = ["Changed synthetic criterion"]
    assert task_summary(validate_tasks([changed]))["sha256"] != first["sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("reviewed", 1),
        ("reviewed", True),
        ("split", "train"),
        ("category", "other"),
        ("task_id", "UPPER"),
        ("question", " "),
        ("reference_sources", "not a list"),
        ("expected_status", "tool_failed"),
        ("unknown", "extra"),
    ],
)
def test_malformed_tasks_and_unsupported_review_claims_fail(key: str, value: object) -> None:
    row = draft_tasks()[0]
    row[key] = value
    with pytest.raises(ValueError):
        validate_tasks([row])


def test_duplicate_question_or_source_group_cannot_cross_splits() -> None:
    rows = draft_tasks()
    duplicate = deepcopy(rows[0])
    duplicate["task_id"] = "different-id"
    with pytest.raises(ValueError, match="duplicate"):
        validate_tasks([rows[0], duplicate])
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="leaks"):
        validate_tasks(rows)
