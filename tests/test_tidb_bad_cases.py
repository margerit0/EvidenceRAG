"""Synthetic tests for offline TiDB retrieval bad-case attribution."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from test_tidb_quality import _Fixture
from zhrag.eval.tidb_bad_cases import (
    ATTRIBUTION_CATEGORIES,
    BAD_CASES_SCHEMA,
    classify_retrieval_outcome,
    evaluate_tidb_bad_cases,
    validate_bad_case_report,
)


def test_classification_uses_candidate_window_then_final_cutoff() -> None:
    assert classify_retrieval_outcome(["x"] * 100, ["gold"]) == "recall_failure"
    run = ["x"] * 10 + ["gold"] + ["x"] * 89
    assert classify_retrieval_outcome(run, ["gold"]) == "ranking_failure"
    run = ["gold"] + ["x"] * 99
    assert classify_retrieval_outcome(run, ["gold"]) == "success"


@pytest.fixture
def fixture() -> _Fixture:
    return _Fixture()


def test_builds_deterministic_aggregate_only_report(fixture: _Fixture) -> None:
    first = evaluate_tidb_bad_cases(
        state=fixture.state,
        pool_report=fixture.pool_report,
        qrels_report=fixture.qrels_report,
        run_rows=fixture.run_rows,
        qrel_rows=fixture.qrel_rows,
    )
    second = evaluate_tidb_bad_cases(
        state=fixture.state,
        pool_report=fixture.pool_report,
        qrels_report=fixture.qrels_report,
        run_rows=fixture.run_rows,
        qrel_rows=fixture.qrel_rows,
    )
    assert first == second
    assert first["schema"] == BAD_CASES_SCHEMA
    assert first["evaluation_design"]["queries"] == 6
    assert first["generation"]["status"] == "not_evaluated"
    assert set(first["system_categories"]) == {
        "bm25-char-bigram",
        "dense-qwen3-4096",
        "rrf-k10-depth100",
        "rerank-qwen3-top50",
    }
    serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "direct:unrelated",
        "paraphrase:other",
        "直接问题",
        "答案 0",
        "chunk-000",
        "source-shared",
    ):
        assert forbidden not in serialized
    for row in first["system_categories"].values():
        assert sum(row["counts"].values()) == row["queries"]
        assert set(row["counts"]) == set(ATTRIBUTION_CATEGORIES)


def test_validator_rejects_raw_fields_and_drift(fixture: _Fixture) -> None:
    report = evaluate_tidb_bad_cases(
        state=fixture.state,
        pool_report=fixture.pool_report,
        qrels_report=fixture.qrels_report,
        run_rows=fixture.run_rows,
        qrel_rows=fixture.qrel_rows,
    )
    raw = copy.deepcopy(report)
    raw["query_id"] = "secret"
    with pytest.raises(ValueError, match="forbidden"):
        validate_bad_case_report(raw)

    drifted = copy.deepcopy(report)
    drifted["system_categories"]["bm25-char-bigram"]["counts"]["success"] += 1
    with pytest.raises(ValueError, match="sum"):
        validate_bad_case_report(drifted)


def test_input_fingerprint_drift_fails_closed(fixture: _Fixture) -> None:
    qrels_report: dict[str, Any] = copy.deepcopy(fixture.qrels_report)
    qrels_report["runs_fingerprint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runs fingerprints"):
        evaluate_tidb_bad_cases(
            state=fixture.state,
            pool_report=fixture.pool_report,
            qrels_report=qrels_report,
            run_rows=fixture.run_rows,
            qrel_rows=fixture.qrel_rows,
        )
