from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zhrag.io_utils import read_text, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_m9b_docs.py"
    spec = importlib.util.spec_from_file_location("sync_m9b_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _docs(root: Path, *, duplicate_readme_marker: bool = False) -> tuple[Path, Path]:
    evaluation = root / "docs" / "evaluation.md"
    architecture = root / "docs" / "architecture-decision.md"
    evaluation_region = (
        "<!-- BEGIN M9B-GENERATION-STATUS -->\nstale\n<!-- END M9B-GENERATION-STATUS -->"
    )
    if duplicate_readme_marker:
        evaluation_region = f"{evaluation_region}\n{evaluation_region}"
    write_text(evaluation, f"before\n{evaluation_region}\nafter\n")
    write_text(
        architecture,
        "\n".join(
            (
                "before",
                "<!-- BEGIN M9B-GENERATION-STATUS -->",
                "stale",
                "<!-- END M9B-GENERATION-STATUS -->",
                "<!-- BEGIN M9B-GENERATION-ROADMAP -->",
                "stale",
                "<!-- END M9B-GENERATION-ROADMAP -->",
                "after",
                "",
            )
        ),
    )
    return evaluation, architecture


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> argparse.Namespace:
    argv = [
        "--evaluation",
        str(root / "docs" / "evaluation.md"),
        "--architecture",
        str(root / "docs" / "architecture-decision.md"),
        "--repo-root",
        str(root),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


class TestRendering:
    def test_uses_frozen_contract_constants_without_artifacts(self) -> None:
        runner = _runner()
        readme = runner.render_readme_status()
        architecture = runner.render_architecture_status()
        roadmap = runner.render_roadmap()
        rendered = "\n".join((readme, architecture, roadmap))

        assert "m9b1-known-context-v1" in rendered
        assert "event_summary" in rendered
        assert "questanswer_1doc" in rendered
        assert "known-context" in rendered
        assert "retrieval stage" in rendered
        assert "Table 8 reproduction" in rendered
        assert "真实实验待授权" in rendered
        assert "numeric_samples.json" in rendered
        assert "report.json" in rendered
        assert "--allow-paid-provider" in rendered
        assert "--allow-model-download" in rendered
        for forbidden in ("SECRET_QUERY", "SECRET_ANSWER", "provider payload value"):
            assert forbidden not in rendered

    def test_does_not_read_ignored_inputs(self, tmp_path: Path) -> None:
        runner = _runner()
        write_text(
            tmp_path / "indexes" / "crud" / "generation" / "v1" / "report.json",
            "SECRET_REPORT",
        )
        write_text(tmp_path / ".env", "SECRET_KEY=do-not-read")
        readme, architecture = _docs(tmp_path)

        runner.synchronize(_args(runner, tmp_path))
        rendered = read_text(readme) + read_text(architecture)
        assert "SECRET_REPORT" not in rendered
        assert "SECRET_KEY" not in rendered


class TestSynchronization:
    def test_renders_both_documents_and_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner()
        docs = _docs(tmp_path)
        args = _args(runner, tmp_path)

        assert runner.synchronize(args) == docs
        first = tuple(read_text(path) for path in docs)
        assert runner.synchronize(args) == ()
        assert tuple(read_text(path) for path in docs) == first
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()
        assert "<!-- BEGIN M9A-GENERATION-METRICS -->" not in first[0]
        assert "M9b1 编排合同 ✅" in first[1]

    def test_check_rejects_stale_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)

        with pytest.raises(SystemExit, match="generated M9b1 documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))
        assert tuple(read_text(path) for path in docs) == before

    def test_validates_all_markers_before_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        evaluation = tmp_path / "docs" / "evaluation.md"
        write_text(
            evaluation,
            "<!-- BEGIN M9B-GENERATION-STATUS -->\n"
            "one\n<!-- END M9B-GENERATION-STATUS -->\n"
            "<!-- BEGIN M9B-GENERATION-STATUS -->\ntwo\n"
            "<!-- END M9B-GENERATION-STATUS -->\n",
        )
        architecture = tmp_path / "docs" / "architecture-decision.md"
        write_text(
            architecture,
            "<!-- BEGIN M9B-GENERATION-STATUS -->\nstale\n"
            "<!-- END M9B-GENERATION-STATUS -->\n"
            "<!-- BEGIN M9B-GENERATION-ROADMAP -->\nstale\n"
            "<!-- END M9B-GENERATION-ROADMAP -->\n",
        )
        before = (read_text(evaluation), read_text(architecture))

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))
        assert (read_text(evaluation), read_text(architecture)) == before

    def test_publication_failure_rolls_back_both_documents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)
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

        assert tuple(read_text(path) for path in docs) == before
        assert not list(tmp_path.glob("*.m9b-sync.tmp"))


class TestContractIsolation:
    @pytest.mark.parametrize("name", ["read_json", "read_jsonl", "load_evidence"])
    def test_synchronizer_has_no_artifact_reader_api(self, name: str) -> None:
        runner = _runner()
        assert not hasattr(runner, name)

    def test_parser_output_is_not_textual_artifact_data(self) -> None:
        runner = _runner()
        value: Any = runner.render_readme_status()
        assert isinstance(value, str)
        assert "question" in value
        assert "answer" in value
