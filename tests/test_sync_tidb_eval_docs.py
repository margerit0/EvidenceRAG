from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.io_utils import read_text, write_json, write_text

RUN_LABELS = (
    "bm25-char-bigram",
    "dense-qwen3-4096",
    "rrf-k10-depth100",
    "rerank-qwen3-top50",
)


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_tidb_eval_docs.py"
    spec = importlib.util.spec_from_file_location("sync_tidb_eval_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_junit(path: Path, *, tests: int = 7, failures: int = 0, skipped: int = 0) -> None:
    root = ET.Element("testsuites")
    ET.SubElement(
        root,
        "testsuite",
        tests=str(tests),
        failures=str(failures),
        errors="0",
        skipped=str(skipped),
    )
    write_text(path, ET.tostring(root, encoding="unicode"))


def _reports(root: Path, *, qrels_overrides: dict[str, object] | None = None) -> Path:
    artifacts = root / "indexes" / "tidb"
    eval_root = artifacts / "eval"
    documents = {
        "doc-a": {"chunk_ids": ["chunk-a", "chunk-b"]},
        "doc-b": {"chunk_ids": ["chunk-c"]},
    }
    state = {
        "schema": "zhrag-ingest-state-v1",
        "collection_name": "tidb_chunks_v1",
        "documents": documents,
        "chunker_fingerprint": "chunker",
        "embedding_profile": "qwen3-embedding-8b-tidb-doc-4096-v1",
        "scope": "scope",
        "sparse_fingerprint": "sparse",
    }
    qgen = {
        "schema": "zhrag-tidb-qgen-v1",
        "sampled_chunks": 4,
        "pairs_verified": 3,
        "pairs_dropped": 1,
        "complete_pairs": 2,
        "queries": 4,
        "queries_by_variant": {"direct": 2, "paraphrase": 2},
        "published_index": {
            "chunks": 3,
            "collection": "tidb_chunks_v1",
            "chunker": "chunker",
            "embedding_profile": "qwen3-embedding-8b-tidb-doc-4096-v1",
            "scope": "scope",
            "sparse": "sparse",
        },
    }
    pool = {
        "schema": "zhrag-tidb-runs-v1",
        "queries": 4,
        "pairs": 2,
        "corpus_chunks": 3,
        "query_set_fingerprint": "queries",
        "pool_fingerprint": "pool",
        "runs_fingerprint": "runs",
        "pool_candidates_total": 6,
        "pool_candidates_min": 2,
        "pool_candidates_mean": 3.0,
        "pool_candidates_max": 4,
    }
    qrels: dict[str, object] = {
        "schema": "zhrag-tidb-qrels-v1",
        "queries": 4,
        "pairs": 2,
        "corpus_chunks": 3,
        "query_set_fingerprint_sha256": "queries",
        "pool_fingerprint_sha256": "pool",
        "runs_fingerprint_sha256": "runs",
        "batches": 2,
        "cache_batches": 2,
        "cache_valid_batches": 2,
        "grades": {"0": 3, "1": 1, "2": 2},
        "generating_chunk_grades": {"0": 0, "1": 1, "2": 1},
        "generating_chunk_disagreement_rate": 0.5,
        "run_judged_coverage": {
            label: {
                "queries": 4,
                "top1_complete": 4,
                "top1_rate": 1.0,
                "top10_complete": 4,
                "top10_rate": 1.0,
            }
            for label in RUN_LABELS
        },
    }
    if qrels_overrides:
        qrels.update(qrels_overrides)
    write_json(artifacts / "state.json", state)
    write_json(eval_root / "report.json", qgen)
    write_json(eval_root / "pool_report.json", pool)
    write_json(eval_root / "qrels_report.json", qrels)
    _write_junit(eval_root / "pytest.xml")
    return artifacts


def _docs(root: Path, *, duplicate_readme_marker: bool = False) -> tuple[Path, Path, Path]:
    readme = root / "README.md"
    architecture = root / "docs" / "architecture-decision.md"
    claude = root / "CLAUDE.md"
    status_region = "<!-- BEGIN TIDB-EVAL-STATUS -->\nstale\n<!-- END TIDB-EVAL-STATUS -->"
    if duplicate_readme_marker:
        status_region = f"{status_region}\n{status_region}"
    write_text(
        readme,
        "\n".join(
            (
                "before",
                status_region,
                "<!-- BEGIN TIDB-EVAL-EVIDENCE -->",
                "stale",
                "<!-- END TIDB-EVAL-EVIDENCE -->",
                "```",
                "<!-- BEGIN QUALITY-GATE-STATUS -->",
                "stale",
                "<!-- END QUALITY-GATE-STATUS -->",
                "after",
            )
        ),
    )
    write_text(
        architecture,
        "\n".join(
            (
                "<!-- BEGIN TIDB-CORPUS-EVAL-STATUS -->",
                "stale",
                "<!-- END TIDB-CORPUS-EVAL-STATUS -->",
                "<!-- BEGIN M3-TIDB-EVAL-STATUS -->",
                "stale",
                "<!-- END M3-TIDB-EVAL-STATUS -->",
            )
        ),
    )
    write_text(
        claude,
        "<!-- BEGIN TIDB-LOCAL-ARTIFACTS -->\nstale\n<!-- END TIDB-LOCAL-ARTIFACTS -->",
    )
    return readme, architecture, claude


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> object:
    readme, architecture, claude = (
        root / "README.md",
        root / "docs" / "architecture-decision.md",
        root / "CLAUDE.md",
    )
    argv = [
        "--artifacts",
        str(root / "indexes" / "tidb"),
        "--readme",
        str(readme),
        "--architecture",
        str(architecture),
        "--claude-context",
        str(claude),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


class TestReportValidation:
    def test_reconciles_all_report_families_and_junit(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)

        status = runner.load_status(artifacts)

        assert status.documents == 2
        assert status.chunks == 3
        assert status.verified_pairs == 2
        assert status.queries == 4
        assert status.pool_candidates == 6
        assert status.judging_batches == 2
        assert status.tests == 7

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"query_set_fingerprint_sha256": "wrong"}, "query-set fingerprint"),
            ({"cache_valid_batches": 1}, "cache batch count"),
            ({"grades": {"0": 3, "1": 1, "2": 1}}, "grade counts"),
        ],
    )
    def test_fails_closed_on_cross_report_drift(
        self,
        tmp_path: Path,
        override: dict[str, object],
        message: str,
    ) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path, qrels_overrides=override)
        with pytest.raises(SystemExit, match=message):
            runner.load_status(artifacts)

    def test_rejects_incomplete_coverage_or_quality_gate(self, tmp_path: Path) -> None:
        runner = _runner()
        coverage = {
            label: {
                "queries": 4,
                "top1_complete": 4,
                "top1_rate": 1.0,
                "top10_complete": 4,
                "top10_rate": 1.0,
            }
            for label in RUN_LABELS
        }
        coverage["dense-qwen3-4096"]["top10_complete"] = 3
        artifacts = _reports(tmp_path, qrels_overrides={"run_judged_coverage": coverage})
        with pytest.raises(SystemExit, match="coverage is incomplete"):
            runner.load_status(artifacts)

        artifacts = _reports(tmp_path)
        _write_junit(artifacts / "eval" / "pytest.xml", failures=1)
        with pytest.raises(SystemExit, match="quality gate is not clean"):
            runner.load_status(artifacts)


class TestSynchronization:
    def test_renders_allowlisted_aggregates_and_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        readme, architecture, claude = _docs(tmp_path)

        assert runner.synchronize(_args(runner, tmp_path)) == (readme, architecture, claude)
        first = tuple(read_text(path) for path in (readme, architecture, claude))
        assert "2 组 direct/paraphrase、4 条 query" in first[0]
        assert "100% **已判断覆盖**" in first[0]
        assert "`pytest` 7 passed" in first[0]
        assert "无上游人工 gold" in first[1]
        assert "judging_cache.jsonl" in first[2]
        assert runner.synchronize(_args(runner, tmp_path)) == ()
        assert tuple(read_text(path) for path in (readme, architecture, claude)) == first
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

    def test_check_reports_stale_docs_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        readme, architecture, claude = _docs(tmp_path)
        before = tuple(read_text(path) for path in (readme, architecture, claude))

        with pytest.raises(SystemExit, match="documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))

        assert tuple(read_text(path) for path in (readme, architecture, claude)) == before

    def test_validates_every_document_before_writing_any(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        readme, architecture, claude = _docs(tmp_path, duplicate_readme_marker=True)
        before = tuple(read_text(path) for path in (readme, architecture, claude))

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))

        assert tuple(read_text(path) for path in (readme, architecture, claude)) == before
