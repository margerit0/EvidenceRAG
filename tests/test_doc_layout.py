from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.io_utils import read_text

ROOT = Path(__file__).resolve().parent.parent
DETAIL_ONLY = (
    "sync_h_hybrid_mrl1024_docs",
    "sync_tidb_chunk_sweep_docs",
    "sync_m9a_docs",
    "sync_m9b_docs",
    "sync_m9b_results_docs",
)
SYNCHRONIZERS = (
    *DETAIL_ONLY,
    "sync_tidb_eval_docs",
    "sync_m8_docs",
    "sync_quality_gate_docs",
    "sync_ablation_docs",
)


def _runner(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"layout_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _required_args(name: str) -> list[str]:
    return ["--run-id", "synthetic-run"] if name == "sync_m9b_results_docs" else []


@pytest.mark.parametrize("name", SYNCHRONIZERS)
def test_default_doc_targets_and_shared_lock_root(name: str, tmp_path: Path) -> None:
    runner = _runner(name)
    args = runner._parse_args(_required_args(name))
    assert args.repo_root == ROOT
    assert runner.DOCS_LOCK == ".docs.lock"
    if name != "sync_quality_gate_docs":
        assert args.evaluation == ROOT / "docs" / "evaluation.md"
    if name not in DETAIL_ONLY:
        assert args.readme == ROOT / "README.md"
    relocated = runner._parse_args([*_required_args(name), "--repo-root", str(tmp_path)])
    assert relocated.repo_root == tmp_path


@pytest.mark.parametrize("name", DETAIL_ONLY)
def test_legacy_argument_alias_selects_only_the_detailed_doc(name: str, tmp_path: Path) -> None:
    runner = _runner(name)
    output = tmp_path / "nested" / "details.md"
    args = runner._parse_args(
        [*_required_args(name), "--readme", str(output), "--repo-root", str(tmp_path)]
    )
    assert args.evaluation == output
    assert args.repo_root == tmp_path
    assert not hasattr(args, "readme")


def test_readme_keeps_summary_regions_and_two_balanced_diagrams() -> None:
    readme = read_text(ROOT / "README.md")
    evaluation = read_text(ROOT / "docs" / "evaluation.md")
    for marker in ("TIDB-EVAL-SUMMARY", "M8-README-HEADLINE", "QUALITY-GATE-STATUS"):
        assert readme.count(f"<!-- BEGIN {marker} -->") == 1
        assert readme.count(f"<!-- END {marker} -->") == 1
        assert f"<!-- BEGIN {marker} -->" not in evaluation
    for marker in (
        "TIDB-EVAL-STATUS",
        "TIDB-EVAL-EVIDENCE",
        "M8-SERVICE-BENCHMARK",
        "M7-CHUNK-SWEEP",
        "H-HYBRID-MRL1024-EVIDENCE",
        "M9A-GENERATION-METRICS",
        "M9B-GENERATION-STATUS",
    ):
        assert f"<!-- BEGIN {marker} -->" not in readme
        assert evaluation.count(f"<!-- BEGIN {marker} -->") == 1
        assert evaluation.count(f"<!-- END {marker} -->") == 1
    assert readme.count("```mermaid\n") == 2
    assert sum(line.startswith("```") for line in readme.splitlines()) % 2 == 0
    assert "docs/README.legacy.md" not in readme
    for doc in (readme, evaluation):
        assert doc.count("<!-- BEGIN ABLATION-SUMMARY -->") == 1
        assert doc.count("<!-- END ABLATION-SUMMARY -->") == 1


def test_local_snapshot_has_an_explicit_ignore_rule() -> None:
    assert "/docs/README.legacy.md" in read_text(ROOT / ".gitignore").splitlines()


@pytest.mark.parametrize("relative_path", ["README.md", "docs/evaluation.md", "CLAUDE.md"])
def test_public_navigation_does_not_require_the_local_snapshot(relative_path: str) -> None:
    assert "README.legacy.md" not in read_text(ROOT / relative_path)


def test_architecture_retains_generated_performance_contract() -> None:
    architecture = read_text(ROOT / "docs/architecture-decision.md")
    for marker in ("M8-RESUME-EVIDENCE", "M8-RESUME-PERFORMANCE"):
        assert architecture.count(f"<!-- BEGIN {marker} -->") == 1
        assert architecture.count(f"<!-- END {marker} -->") == 1
    assert "客户端 exact RRF" in architecture
    assert "§13" in architecture
