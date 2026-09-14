from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from test_agent_review import bundle, reviews_for
from test_compare_agent_script import load_script
from zhrag.eval.agent_review import fingerprint
from zhrag.io_utils import read_json, read_jsonl, read_text, write_json, write_jsonl, write_text


@pytest.fixture
def local_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, Path]:
    tasks = load_script("agent_tasks")
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    monkeypatch.setattr(tasks, "TASKS_ROOT", tmp_path / "indexes/agent_eval")
    monkeypatch.setitem(sys.modules, "agent_tasks", tasks)
    # Importing a provider or live service is a failure in every offline mode.
    monkeypatch.setitem(sys.modules, "serve", None)
    review = load_script("review_agent")
    run = tasks.TASKS_ROOT / "runs/synthetic"
    manifest, rows = bundle()
    write_json(run / "manifest.json", manifest)
    write_jsonl(run / "trials.jsonl", rows)
    return review, run


def test_prepare_separates_review_and_does_not_overwrite(
    local_review: tuple[ModuleType, Path],
) -> None:
    script, run = local_review
    original = read_text(run / "trials.jsonl")
    assert script.main(["--run-dir", str(run)]) == 0
    assert script.main(["--run-dir", str(run), "--prepare"]) == 0
    reviews = list(read_jsonl(run / "reviews.jsonl"))
    assert len(reviews) == 18 and not any(row["review"]["reviewed"] for row in reviews)
    assert script.main(["--run-dir", str(run), "--prepare"]) == 1
    assert read_text(run / "trials.jsonl") == original
    packet = read_text(run / "review-packet.txt")
    assert "合成证据" in packet and "合成验收项" in packet
    assert all(
        method not in packet for method in ("single_rag", "fixed_workflow", "document_agent")
    )
    assert script.main(["--run-dir", str(run), "--report"]) == 1
    assert not (run / "quality_report.json").exists()


def test_report_recomputation_rejects_stale_labels_and_output(
    local_review: tuple[ModuleType, Path],
) -> None:
    script, run = local_review
    manifest = read_json(run / "manifest.json")
    rows = list(read_jsonl(run / "trials.jsonl"))
    reviews = reviews_for(manifest, rows)
    write_jsonl(run / "reviews.jsonl", reviews)
    args = ["--run-dir", str(run), "--resamples", "100"]
    assert script.main([*args, "--report"]) == 0
    report = read_json(run / "quality_report.json")
    assert report["task_count"] == 6 and report["provenance"]["reviews_sha256"] == fingerprint(
        reviews
    )
    assert script.main([*args, "--check"]) == 0
    reviews[0]["review"]["notes"] = "additional synthetic review note"
    write_jsonl(run / "reviews.jsonl", reviews)
    assert script.main([*args, "--check"]) == 1
    assert script.main([*args, "--report"]) == 0
    report = read_json(run / "quality_report.json")
    report["task_count"] = 100
    write_json(run / "quality_report.json", report)
    assert script.main([*args, "--check"]) == 1


def test_strict_json_prevents_duplicate_keys_and_nonfinite_data(
    local_review: tuple[ModuleType, Path],
) -> None:
    script, run = local_review
    for invalid in ('{"complete":true,"complete":false}', '{"value":NaN}'):
        write_text(run / "manifest.json", invalid)
        assert script.main(["--run-dir", str(run)]) == 1


def test_paths_outside_private_artifacts_rejected(local_review: tuple[ModuleType, Path]) -> None:
    script, run = local_review
    assert script.main(["--run-dir", str(run.parents[4]), "--prepare"]) == 1
    assert not (run.parents[4] / "reviews.jsonl").exists()
