from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.io_utils import read_text, write_text
from zhrag.quality_gate import JunitCounts, parse_junit, read_junit, require_clean


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_quality_gate_docs.py"
    spec = importlib.util.spec_from_file_location("sync_quality_gate_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _xml(
    *,
    root: str = "testsuites",
    suites: tuple[dict[str, str], ...] = (
        {"tests": "7", "failures": "0", "errors": "0", "skipped": "0"},
    ),
) -> str:
    rendered = []
    for attributes in suites:
        rendered.append(
            "<testsuite "
            + " ".join(f'{name}="{value}"' for name, value in attributes.items())
            + "></testsuite>"
        )
    if root == "testsuite":
        return rendered[0]
    return f"<testsuites>{''.join(rendered)}</testsuites>"


def _write_junit(path: Path, content: str | None = None) -> None:
    write_text(path, _xml() if content is None else content)


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> argparse.Namespace:
    argv = [
        "--junit",
        str(root / "reports" / "pytest.xml"),
        "--readme",
        str(root / "README.md"),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


def _readme(root: Path, *, duplicate: bool = False) -> Path:
    marker = "<!-- BEGIN QUALITY-GATE-STATUS -->\nstale\n<!-- END QUALITY-GATE-STATUS -->"
    if duplicate:
        marker = f"{marker}\n{marker}"
    path = root / "README.md"
    write_text(path, f"before\n{marker}\nafter\n")
    return path


class TestJunitParser:
    def test_accepts_single_testsuite_root(self) -> None:
        assert parse_junit(_xml(root="testsuite")) == JunitCounts(7, 0, 0, 0)

    def test_sums_multiple_testsuites(self) -> None:
        value = parse_junit(
            _xml(
                suites=(
                    {"tests": "2", "failures": "1", "errors": "0", "skipped": "0"},
                    {"tests": "5", "failures": "0", "errors": "2", "skipped": "1"},
                )
            )
        )
        assert value == JunitCounts(7, 1, 2, 1)

    def test_missing_counters_are_zero(self) -> None:
        assert parse_junit('<testsuite tests="3"></testsuite>') == JunitCounts(3, 0, 0, 0)

    @pytest.mark.parametrize(
        "content",
        [
            '<testsuite tests="many"></testsuite>',
            '<testsuite tests="-1"></testsuite>',
            '<testsuites><testsuite tests="1" failures="-1"></testsuite></testsuites>',
        ],
    )
    def test_rejects_non_integer_or_negative_counters(self, content: str) -> None:
        with pytest.raises(ValueError, match="JUnit"):
            parse_junit(content)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("failures", 1), ("errors", 1), ("skipped", 1), ("tests", 0)],
    )
    def test_require_clean_rejects_every_non_clean_state(self, field: str, value: int) -> None:
        counts = JunitCounts(
            tests=value if field == "tests" else 1,
            failures=value if field == "failures" else 0,
            errors=value if field == "errors" else 0,
            skipped=value if field == "skipped" else 0,
        )
        with pytest.raises(ValueError, match="quality gate is not clean"):
            require_clean(counts)

    def test_read_junit_uses_path_and_project_io(self, tmp_path: Path) -> None:
        path = tmp_path / "pytest.xml"
        _write_junit(path)
        assert read_junit(path) == JunitCounts(7, 0, 0, 0)

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "<not-testsuite />",
            "<testsuites />",
            "<testsuites><other /></testsuites>",
            "<testsuite",
        ],
    )
    def test_rejects_malformed_documents(self, content: str) -> None:
        with pytest.raises(ValueError, match="JUnit"):
            parse_junit(content)


class TestQualityGateSynchronization:
    def test_renders_test_count_without_needing_tidb_artifacts(self, tmp_path: Path) -> None:
        runner = _runner()
        junit = tmp_path / "reports" / "pytest.xml"
        _write_junit(junit)
        readme = _readme(tmp_path)

        args = _args(runner, tmp_path)
        assert runner.synchronize(args) == (readme,)
        rendered = read_text(readme)
        assert "tests/                   7 个单元测试" in rendered
        assert "`pytest` 7 passed" in rendered
        assert "ruff check" in rendered
        assert not (tmp_path / "indexes" / "tidb" / "eval" / "pytest.xml").exists()

    def test_is_idempotent_and_check_accepts_fresh_documentation(self, tmp_path: Path) -> None:
        runner = _runner()
        _write_junit(tmp_path / "reports" / "pytest.xml")
        readme = _readme(tmp_path)
        args = _args(runner, tmp_path)

        assert runner.synchronize(args) == (readme,)
        first = read_text(readme)
        assert runner.synchronize(args) == ()
        assert read_text(readme) == first
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

    def test_check_rejects_stale_documentation_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        _write_junit(tmp_path / "reports" / "pytest.xml")
        readme = _readme(tmp_path)
        before = read_text(readme)

        with pytest.raises(SystemExit, match="generated documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))
        assert read_text(readme) == before

    def test_rejects_duplicate_marker_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        _write_junit(tmp_path / "reports" / "pytest.xml")
        readme = _readme(tmp_path, duplicate=True)
        before = read_text(readme)

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))
        assert read_text(readme) == before

    def test_publication_failure_leaves_original_and_cleans_temporary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        _write_junit(tmp_path / "reports" / "pytest.xml")
        readme = _readme(tmp_path)
        before = read_text(readme)
        real_replace = os.replace

        def fail(source: object, target: object) -> None:
            del source, target
            raise OSError("injected replacement failure")

        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="injected replacement failure"):
            runner.synchronize(_args(runner, tmp_path))
        monkeypatch.setattr(os, "replace", real_replace)

        assert read_text(readme) == before
        assert not list(tmp_path.glob("*.sync.tmp"))

    @pytest.mark.parametrize(
        "xml",
        [
            _xml(suites=({"tests": "7", "failures": "1", "errors": "0", "skipped": "0"},)),
            _xml(suites=({"tests": "7", "failures": "0", "errors": "1", "skipped": "0"},)),
            _xml(suites=({"tests": "7", "failures": "0", "errors": "0", "skipped": "1"},)),
            _xml(suites=({"tests": "0", "failures": "0", "errors": "0", "skipped": "0"},)),
        ],
    )
    def test_load_counts_fails_closed(self, tmp_path: Path, xml: str) -> None:
        runner = _runner()
        path = tmp_path / "pytest.xml"
        _write_junit(path, xml)

        with pytest.raises(SystemExit, match="quality gate is not clean"):
            runner.load_counts(path)
