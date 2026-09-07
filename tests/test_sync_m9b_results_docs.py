from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.io_utils import write_text


def _runner(root: Path) -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_m9b_results_docs.py"
    spec = importlib.util.spec_from_file_location("sync_m9b_results_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.__dict__["ROOT"] = root
    return module


def _args(root: Path, *, run_id: str = "trial-a") -> argparse.Namespace:
    return argparse.Namespace(
        artifacts=root / "indexes" / "crud" / "generation" / "v1",
        run_id=run_id,
        canonical=False,
        evaluation=root / "docs" / "evaluation.md",
        architecture=root / "docs" / "architecture-decision.md",
        repo_root=root,
        check=True,
    )


@pytest.mark.parametrize("slug", ["a", "trial-a", "trial_1.v2", "x" * 64])
def test_accepts_canonical_lowercase_slug(tmp_path: Path, slug: str) -> None:
    runner = _runner(tmp_path)
    assert runner._slug(slug, label="slug") == slug


@pytest.mark.parametrize(
    "slug",
    ["", "Trial-a", "trial-a.", "con", "aux.txt", "com1", "a/b", "x" * 65],
)
def test_rejects_windows_aliases_and_noncanonical_slugs(tmp_path: Path, slug: str) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(SystemExit, match="lowercase"):
        runner._slug(slug, label="slug")


def test_rejects_external_artifact_root_before_loading_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    args = _args(tmp_path)
    args.artifacts = tmp_path / "external-artifacts"

    def fail_results(_root: Path) -> object:
        raise AssertionError("results must not load from an external artifact root")

    monkeypatch.setattr(runner, "load_results", fail_results)
    with pytest.raises(SystemExit, match="project generation root"):
        runner.synchronize(args)


def test_rejects_redirected_run_before_loading_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    args = _args(tmp_path)
    redirected = args.artifacts / "runs" / "trials" / args.run_id
    redirected.mkdir(parents=True)
    path_type = type(redirected)
    real_is_junction = path_type.is_junction

    def fake_is_junction(path: Path) -> bool:
        return path == redirected or bool(real_is_junction(path))

    def fail_results(_root: Path) -> object:
        raise AssertionError("results must not load through a redirected run")

    monkeypatch.setattr(path_type, "is_junction", fake_is_junction)
    monkeypatch.setattr(runner, "load_results", fail_results)
    with pytest.raises(SystemExit, match="symlink or junction"):
        runner.synchronize(args)


def test_rejects_redirected_result_file_before_loading_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    args = _args(tmp_path)
    run_root = args.artifacts / "runs" / "trials" / args.run_id
    report = run_root / runner.REPORT
    report.parent.mkdir(parents=True)
    write_text(report, "")
    path_type = type(report)
    real_is_symlink = path_type.is_symlink

    def fake_is_symlink(path: Path) -> bool:
        return path == report or real_is_symlink(path)

    def fail_results(_root: Path) -> object:
        raise AssertionError("results must not load through a redirected result file")

    monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(runner, "load_results", fail_results)
    with pytest.raises(SystemExit, match="artifact tree contains"):
        runner.synchronize(args)
