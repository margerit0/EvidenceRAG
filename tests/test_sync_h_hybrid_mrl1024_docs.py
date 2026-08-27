from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from test_hybrid_mrl1024 import _inputs
from zhrag.eval.hybrid_mrl1024 import evaluate_hybrid_mrl1024
from zhrag.io_utils import read_text, write_json, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_h_hybrid_mrl1024_docs.py"
    spec = importlib.util.spec_from_file_location("sync_h_hybrid_mrl1024_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(expanded: Path, *, mutator: object | None = None) -> dict[str, Any]:
    report = evaluate_hybrid_mrl1024(_inputs(), resamples=30, seed=7)
    if mutator is not None:
        assert callable(mutator)
        mutator(report)
    write_json(expanded / "h_hybrid_rrf_mrl1024_report.json", report)
    return report


def _docs(
    root: Path,
    *,
    duplicate_readme_marker: bool = False,
) -> tuple[Path, Path, Path]:
    readme = root / "README.md"
    architecture = root / "docs" / "architecture-decision.md"
    claude = root / "CLAUDE.md"
    readme_region = (
        "<!-- BEGIN H-HYBRID-MRL1024-EVIDENCE -->\nstale\n<!-- END H-HYBRID-MRL1024-EVIDENCE -->"
    )
    if duplicate_readme_marker:
        readme_region = f"{readme_region}\n{readme_region}"
    write_text(readme, f"before\n{readme_region}\nafter\n")
    write_text(
        architecture,
        "\n".join(
            (
                "<!-- BEGIN H-HYBRID-MRL1024-STATUS -->",
                "stale",
                "<!-- END H-HYBRID-MRL1024-STATUS -->",
                "<!-- BEGIN H-HYBRID-MRL1024-M4 -->",
                "stale",
                "<!-- END H-HYBRID-MRL1024-M4 -->",
                "<!-- BEGIN H-HYBRID-MRL1024-CHECKLIST -->",
                "stale",
                "<!-- END H-HYBRID-MRL1024-CHECKLIST -->",
            )
        ),
    )
    write_text(
        claude,
        "<!-- BEGIN H-HYBRID-MRL1024-ARTIFACT -->\nstale\n<!-- END H-HYBRID-MRL1024-ARTIFACT -->",
    )
    return readme, architecture, claude


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> argparse.Namespace:
    argv = [
        "--expanded",
        str(root / "expanded"),
        "--readme",
        str(root / "README.md"),
        "--architecture",
        str(root / "docs" / "architecture-decision.md"),
        "--claude-context",
        str(root / "CLAUDE.md"),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


class TestAuthentication:
    def test_recomputes_the_report_before_returning_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        report = _report(expanded)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())

        assert runner.load_status(expanded) == report

    def test_rejects_a_structurally_valid_forged_statistic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"

        def forge(report: dict[str, Any]) -> None:
            summary = report["metrics"]["headline"]["systems"]["H-hybrid-a-plus-e"]["R@1"]
            summary["mean"] = 0.123456

        _report(expanded, mutator=forge)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())

        with pytest.raises(SystemExit, match="deterministic recomputation"):
            runner.load_status(expanded)

    def test_rejects_malformed_report_before_loading_raw_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        report = _report(expanded)
        report["inputs"]["query_id"] = "leak"
        write_json(expanded / runner.REPORT_NAME, report)
        called = {"raw": False}

        def should_not_load(*_args: object, **_kwargs: object) -> object:
            called["raw"] = True
            raise AssertionError

        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", should_not_load)
        with pytest.raises(SystemExit, match="forbidden raw field"):
            runner.load_status(expanded)
        assert not called["raw"]


class TestSynchronization:
    def test_renders_allowlisted_aggregates_and_is_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _report(expanded)
        readme, architecture, claude = _docs(tmp_path)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())

        assert runner.synchronize(_args(runner, tmp_path)) == (readme, architecture, claude)
        first = tuple(read_text(path) for path in (readme, architecture, claude))
        assert "H：dense-1024 + BM25 的独立 hybrid 基线" in first[0]
        assert "arity" in first[0]
        assert "retention / continuous" in first[0]
        assert "non-inferiority / equivalence" in first[0]
        assert "unit=document" not in first[0]
        assert "完整文档" in first[1]
        assert "retention family 共 " in first[1]
        assert "/12 项" in first[1]
        m4 = (
            first[1]
            .split("<!-- BEGIN H-HYBRID-MRL1024-M4 -->", 1)[1]
            .split("<!-- END H-HYBRID-MRL1024-M4 -->", 1)[0]
            .strip()
        )
        assert m4.startswith("| **M4** |")
        assert m4.count("\n") == 0
        assert m4.count("|") == 6
        assert "h_hybrid_rrf_mrl1024_report.json" in first[2]
        claude_artifact = (
            first[2]
            .split("<!-- BEGIN H-HYBRID-MRL1024-ARTIFACT -->", 1)[1]
            .split("<!-- END H-HYBRID-MRL1024-ARTIFACT -->", 1)[0]
            .strip()
        )
        assert claude_artifact.startswith("- `crud-rag-subset/")
        assert not claude_artifact.startswith("|")
        serialized = "\n".join(first)
        for forbidden in ("主题000是什么", "答案-0", "doc-000", "questanswer_1doc:q0"):
            assert forbidden not in serialized

        assert runner.synchronize(_args(runner, tmp_path)) == ()
        assert tuple(read_text(path) for path in (readme, architecture, claude)) == first
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

    def test_check_reports_stale_docs_without_writing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _report(expanded)
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())

        with pytest.raises(SystemExit, match="documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))
        assert tuple(read_text(path) for path in docs) == before

    def test_docs_lock_contention_blocks_before_artifact_read(self, tmp_path: Path) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        expanded.mkdir()
        _docs(tmp_path)
        lock = tmp_path / runner.DOCS_LOCK
        write_text(lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.synchronize(_args(runner, tmp_path))
        assert read_text(lock) == "pid=123\n"
        assert not (expanded / runner.ARTIFACT_LOCK).exists()

    def test_artifact_lock_contention_cleans_docs_lock(self, tmp_path: Path) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        expanded.mkdir()
        _docs(tmp_path)
        artifact_lock = expanded / runner.ARTIFACT_LOCK
        write_text(artifact_lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.synchronize(_args(runner, tmp_path))
        assert read_text(artifact_lock) == "pid=123\n"
        assert not (tmp_path / runner.DOCS_LOCK).exists()

    def test_publication_failure_rolls_back_every_document(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _report(expanded)
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(source: object, target: object) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("injected replacement failure")
            real_replace(source, target)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "replace", flaky)
        with pytest.raises(OSError, match="injected replacement failure"):
            runner.synchronize(_args(runner, tmp_path))
        monkeypatch.undo()
        assert tuple(read_text(path) for path in docs) == before
        assert not list(tmp_path.glob("*.h-sync.tmp"))

    def test_validates_all_markers_before_writing_anything(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _report(expanded)
        docs = _docs(tmp_path, duplicate_readme_marker=True)
        before = tuple(read_text(path) for path in docs)
        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_a, **_k: _inputs())

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))
        assert tuple(read_text(path) for path in docs) == before
