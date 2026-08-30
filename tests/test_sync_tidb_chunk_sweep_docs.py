"""Hermetic tests for M7 report authentication and documentation sync."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zhrag.io_utils import read_text, write_json, write_text

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sync_tidb_chunk_sweep_docs.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_sync_tidb_chunk_sweep_docs_runner", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import M7 documentation synchronizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "zhrag-tidb-chunk-sweep-report-v2",
        "profiles": {
            profile_id: {
                "profile_id": profile_id,
                "documents": 450,
                "chunks": chunks,
                "canonical_exact_reuse": reuse,
                "required_new_document_vectors": new_vectors,
                "split_trigger_exceedance": {"count": 1, "rate": 0.1},
            }
            for profile_id, chunks, reuse, new_vectors in (
                ("tidb-chunk-t256-h384-v1", 2802, 252, 2550),
                ("tidb-chunk-t400-h600-v1", 1832, 1832, 0),
                ("tidb-chunk-t800-h1200-v1", 1019, 190, 829),
            )
        },
        "estimates": {
            profile_id: {
                "rrf-k10-depth100": {
                    "direct": {},
                    "paraphrase": {},
                    "overall": {
                        "origin_mrr_at_10": {
                            "mean": 0.5,
                            "ci_low": 0.4,
                            "ci_high": 0.6,
                        },
                        "origin_hit_at_1": {
                            "mean": 0.5,
                            "ci_low": 0.4,
                            "ci_high": 0.6,
                        },
                        "origin_hit_at_10": {
                            "mean": 0.5,
                            "ci_low": 0.4,
                            "ci_high": 0.6,
                        },
                    },
                }
            }
            for profile_id in (
                "tidb-chunk-t256-h384-v1",
                "tidb-chunk-t400-h600-v1",
                "tidb-chunk-t800-h1200-v1",
            )
        },
        "families": {
            "origin-source-efficacy": [
                {
                    "comparison": "tidb-chunk-t256-h384-v1-vs-tidb-chunk-t400-h600-v1",
                    "delta": 0.1,
                    "ci_low": 0.0,
                    "ci_high": 0.2,
                    "raw_p": 0.01,
                    "adjusted_p": 0.02,
                    "wins": 200,
                    "losses": 100,
                    "ties_nonzero": 100,
                    "ties_zero": 90,
                },
                {
                    "comparison": "tidb-chunk-t800-h1200-v1-vs-tidb-chunk-t400-h600-v1",
                    "delta": -0.1,
                    "ci_low": -0.2,
                    "ci_high": 0.0,
                    "raw_p": 0.2,
                    "adjusted_p": 0.2,
                    "wins": 100,
                    "losses": 200,
                    "ties_nonzero": 100,
                    "ties_zero": 90,
                },
            ],
            "surface-home-field": [
                {
                    "comparison": "tidb-chunk-t256-h384-v1-surface-interaction",
                    "delta": 0.0,
                    "ci_low": -0.1,
                    "ci_high": 0.1,
                    "raw_p": 1.0,
                    "adjusted_p": 1.0,
                    "wins": 0,
                    "losses": 0,
                    "ties_nonzero": 400,
                    "ties_zero": 90,
                },
                {
                    "comparison": "tidb-chunk-t800-h1200-v1-surface-interaction",
                    "delta": 0.0,
                    "ci_low": -0.1,
                    "ci_high": 0.1,
                    "raw_p": 1.0,
                    "adjusted_p": 1.0,
                    "wins": 0,
                    "losses": 0,
                    "ties_nonzero": 400,
                    "ties_zero": 90,
                },
            ],
        },
        "design": {
            "estimand": "400-origin exploratory known-item source retrieval",
            "primary_system": "rrf-k10-depth100",
            "primary_metric": "origin_mrr_at_10",
            "pair_observations": 490,
            "source_clusters": 245,
            "confidence": 0.95,
            "resamples": 2,
            "seed": 0,
            "alternative": "two-sided",
            "correction": "Holm-Bonferroni within each named two-test family",
            "source_collapse": "first occurrence after each final chunk arm; RRF before collapse",
        },
        "inputs": {"fixture_sha256": "a" * 64},
        "samples": {
            "schema": "zhrag-tidb-chunk-sweep-numeric-v1",
            "sha256": "b" * 64,
        },
        "execution_contract": {
            "required_new_document_vectors": 3_379,
            "required_canonical_exact_reuse": 2_274,
            "required_embedding_batches": 212,
            "profile_cache_evidence": "exact-id-complete-matrix",
            "historical_provider_call_evidence": "not-recorded",
            "per_batch_build_receipts": "not-recorded",
            "evaluation_phase": {
                "provider_calls": 0,
                "query_embedding_calls": 0,
                "rerank_calls": 0,
                "chat_calls": 0,
                "milvus_writes": 0,
            },
        },
        "limitations": [
            "400-origin synthetic queries; profile selection on this fixture is exploratory",
            "source known-item retrieval does not certify answer-bearing passage relevance",
            "confirmed-source sensitivity inherits the canonical 400-only judgement pool",
            "unjudged sources are not reliable negatives",
            (
                "embedding provider outputs are cached because repeated calls "
                "are not bitwise deterministic"
            ),
            (
                "split trigger is best-effort; protected blocks and indivisible "
                "paragraphs remain intact"
            ),
        ],
    }
    return report


def _fixture_report_files(root: Path) -> None:
    report = _report()
    # Synchronizer tests replace authentication with a controlled report loader;
    # these files only exercise path and marker plumbing.
    write_json(root / "report.json", report)
    write_json(root / "numeric_samples.json", {"fixture": True})


class TestSynchronizer:
    def test_replace_region_requires_one_marker_pair(self) -> None:
        runner = _runner()
        with pytest.raises(SystemExit, match="exactly once"):
            runner._replace_region("before", "M7", "body", path=Path("README.md"))
        text = "x\n<!-- BEGIN M7 -->\nold\n<!-- END M7 -->\ny"
        assert runner._replace_region(text, "M7", "new", path=Path("README.md")) == (
            "x\n<!-- BEGIN M7 -->\nnew\n<!-- END M7 -->\ny"
        )

    def test_renderers_never_include_raw_identities(self) -> None:
        runner = _runner()
        report = _report()
        for renderer in (runner._readme, runner._architecture, runner._claude):
            rendered = renderer(report)
            assert "query_id" not in rendered
            assert "source_key" not in rendered
            assert "question" not in rendered
            assert "embedding" in rendered or "sweep" in rendered or "分块" in rendered

    def test_renderers_keep_valid_markdown_structure(self) -> None:
        runner = _runner()
        report = _report()
        readme = runner._readme(report)
        architecture = runner._architecture(report)

        assert (
            "> 先在 raw chunk 上分别构建 BM25 / dense-4096，再做 exact RRF "
            "k=10/depth=100，\n> 最终 arm"
        ) in readme
        assert "，> 最终 arm" not in readme
        assert architecture.startswith("| **M7** |")
        assert architecture.count("\n") == 0
        assert architecture.count("|") == 6

    def test_sync_check_detects_stale_marker_without_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _runner()
        report = _report()
        monkeypatch.setattr(runner, "_load_report", lambda _paths, resamples, seed: report)
        readme = tmp_path / "README.md"
        architecture = tmp_path / "architecture.md"
        claude = tmp_path / "CLAUDE.md"
        for path in (readme, architecture, claude):
            write_text(path, "<!-- BEGIN M7-CHUNK-SWEEP -->\nold\n<!-- END M7-CHUNK-SWEEP -->\n")
        args = runner._parse_args(
            [
                "--artifacts",
                str(tmp_path / "artifacts"),
                "--readme",
                str(readme),
                "--architecture",
                str(architecture),
                "--claude-context",
                str(claude),
                "--check",
                "--resamples",
                "2",
            ]
        )
        before = {path: read_text(path) for path in (readme, architecture, claude)}
        with pytest.raises(SystemExit, match="stale"):
            runner.synchronize(args)
        assert {path: read_text(path) for path in (readme, architecture, claude)} == before

    def test_sync_replaces_all_targets_and_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _runner()
        report = _report()
        monkeypatch.setattr(runner, "_load_report", lambda _paths, resamples, seed: report)
        readme = tmp_path / "README.md"
        architecture = tmp_path / "architecture.md"
        claude = tmp_path / "CLAUDE.md"
        marker = "<!-- BEGIN M7-CHUNK-SWEEP -->\nold\n<!-- END M7-CHUNK-SWEEP -->\n"
        for path in (readme, architecture, claude):
            write_text(path, marker)
        args = runner._parse_args(
            [
                "--artifacts",
                str(tmp_path / "artifacts"),
                "--readme",
                str(readme),
                "--architecture",
                str(architecture),
                "--claude-context",
                str(claude),
                "--resamples",
                "2",
            ]
        )
        changed = runner.synchronize(args)
        assert len(changed) == 3
        contents = [read_text(path) for path in (readme, architecture, claude)]
        assert all("old" not in content for content in contents)
        assert all("M7-CHUNK-SWEEP" in content for content in contents)
        assert runner.synchronize(args) == ()

    def test_import_is_provider_free(self) -> None:
        probe = (
            "import json, runpy, sys; "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='offline_import_probe'); "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.startswith('zhrag.providers'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        assert json.loads(completed.stdout) == []
