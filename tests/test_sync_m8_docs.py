from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zhrag.io_utils import read_json, read_text, write_json, write_text
from zhrag.service.bench import (
    STAGE_NAMES,
    BenchmarkSample,
    build_benchmark_report,
    samples_artifact,
    samples_sha256,
)


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_m8_docs.py"
    spec = importlib.util.spec_from_file_location("sync_m8_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stages(value: float) -> dict[str, float]:
    return {name: value for name in STAGE_NAMES}


def _sample(sequence: int, elapsed: float, *, status: int = 200) -> BenchmarkSample:
    return BenchmarkSample(
        sequence=sequence,
        status_code=status,
        elapsed_seconds=elapsed,
        stages=_stages(elapsed / 2.0) if status == 200 else None,
    )


def _artifacts(
    root: Path,
    *,
    report_mutator: object | None = None,
    sample_mutator: object | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifacts = root / "artifacts"
    rows = (
        _sample(0, 0.1),
        _sample(1, 0.2),
        _sample(2, 0.4),
        _sample(3, 0.8, status=503),
    )
    numeric = samples_artifact(rows, measurement_wall_seconds=1.0)
    report = build_benchmark_report(
        samples=rows,
        measurement_wall_seconds=1.0,
        samples_sha256=samples_sha256(numeric),
        profile={
            "name": "cached-local-no-rerank-v1",
            "embedding_profile": "cached-query-embedding-v1",
            "rerank_profile": "disabled-identity-fused-order-v1",
            "rerank_enabled": False,
            "provider_stages_included": False,
        },
        fixture={"sha256": "a" * 64, "count": 3},
        configuration={
            "url_origin": "http://127.0.0.1:8000",
            "warmup": 2,
            "requests": len(rows),
            "concurrency": 1,
            "timeout_seconds": 30.0,
            "seed": 7,
        },
        environment={
            "python": "3.13.7",
            "platform": "Windows-11",
            "machine": "AMD64",
            "processor_count": 8,
        },
        limitations=[
            "This is a local HTTP benchmark, not a production or cloud SLA.",
            "Quality metrics and significance tests are reported separately.",
            "Provider latency is excluded by the cache-backed service profile.",
            "Warm-up requests are excluded from every measured aggregate.",
        ],
    )
    if sample_mutator is not None:
        assert callable(sample_mutator)
        sample_mutator(numeric)
    if report_mutator is not None:
        assert callable(report_mutator)
        report_mutator(report)
    write_json(artifacts / "m8_http_samples.json", numeric)
    write_json(artifacts / "m8_http_report.json", report)
    return artifacts, numeric, report


def _docs(
    root: Path,
    *,
    duplicate_readme_marker: bool = False,
) -> tuple[Path, Path, Path, Path]:
    readme = root / "README.md"
    evaluation = root / "docs" / "evaluation.md"
    architecture = root / "docs" / "architecture-decision.md"
    claude = root / "CLAUDE.md"
    readme_region = "- 服务：<!-- BEGIN M8-README-HEADLINE -->stale<!-- END M8-README-HEADLINE -->"
    if duplicate_readme_marker:
        readme_region = f"{readme_region}\n{readme_region}"
    write_text(readme, f"before\n{readme_region}\nafter\n")
    write_text(
        evaluation,
        "before\n"
        "<!-- BEGIN M8-SERVICE-BENCHMARK -->\n"
        "stale\n"
        "<!-- END M8-SERVICE-BENCHMARK -->\n"
        "after\n",
    )
    write_text(
        architecture,
        "\n".join(
            (
                "<!-- BEGIN M8-STATUS -->",
                "stale",
                "<!-- END M8-STATUS -->",
                "<!-- BEGIN M8-ROADMAP -->",
                "stale",
                "<!-- END M8-ROADMAP -->",
                "<!-- BEGIN M8-RESUME-EVIDENCE -->",
                "stale",
                "<!-- END M8-RESUME-EVIDENCE -->",
                "<!-- BEGIN M8-RESUME-PERFORMANCE -->",
                "stale",
                "<!-- END M8-RESUME-PERFORMANCE -->",
                "<!-- BEGIN M8-CHECKLIST -->",
                "stale",
                "<!-- END M8-CHECKLIST -->",
            )
        ),
    )
    write_text(
        claude,
        "<!-- BEGIN M8-LOCAL-ARTIFACTS -->\nstale\n<!-- END M8-LOCAL-ARTIFACTS -->",
    )
    return readme, evaluation, architecture, claude


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> argparse.Namespace:
    argv = [
        "--artifacts",
        str(root / "artifacts"),
        "--repo-root",
        str(root),
        "--readme",
        str(root / "README.md"),
        "--evaluation",
        str(root / "docs" / "evaluation.md"),
        "--architecture",
        str(root / "docs" / "architecture-decision.md"),
        "--claude-context",
        str(root / "CLAUDE.md"),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


class TestAuthentication:
    def test_rebuilds_report_from_canonical_numeric_samples(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts, _numeric, report = _artifacts(tmp_path)
        assert runner.load_status(artifacts) == report

    def test_rejects_structurally_valid_forged_percentile(self, tmp_path: Path) -> None:
        runner = _runner()

        def forge(report: dict[str, Any]) -> None:
            summary = report["measurement"]["http_latency"]
            summary["p50_seconds"] = 0.21

        artifacts, _numeric, _report = _artifacts(tmp_path, report_mutator=forge)
        with pytest.raises(SystemExit, match="deterministic recomputation"):
            runner.load_status(artifacts)

    def test_rejects_samples_changed_without_report_hash(self, tmp_path: Path) -> None:
        runner = _runner()

        def forge(numeric: dict[str, Any]) -> None:
            numeric["samples"][0]["elapsed_seconds"] = 0.11

        artifacts, _numeric, _report = _artifacts(tmp_path, sample_mutator=forge)
        with pytest.raises(SystemExit, match="SHA-256"):
            runner.load_status(artifacts)

    def test_rejects_samples_and_hash_changed_without_rebuilt_aggregate(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        artifacts, numeric, report = _artifacts(tmp_path)
        rows = numeric["samples"]
        assert isinstance(rows, list)
        rows[1]["elapsed_seconds"] = 0.3
        stages = rows[1]["stages"]
        assert isinstance(stages, dict)
        stages["total_seconds"] = 0.15
        report_ref = report["samples"]
        assert isinstance(report_ref, dict)
        report_ref["sha256"] = samples_sha256(numeric)
        write_json(artifacts / runner.SAMPLES_NAME, numeric)
        write_json(artifacts / runner.REPORT_NAME, report)

        with pytest.raises(SystemExit, match="deterministic recomputation"):
            runner.load_status(artifacts)

    def test_rejects_raw_field_before_rendering(self, tmp_path: Path) -> None:
        runner = _runner()

        def leak(report: dict[str, Any]) -> None:
            profile = report["profile"]
            profile["query"] = "SECRET_QUERY"

        artifacts, _numeric, _report = _artifacts(tmp_path, report_mutator=leak)
        with pytest.raises(SystemExit, match="forbidden"):
            runner.load_status(artifacts)


class TestSynchronization:
    def test_renders_only_aggregates_and_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner()
        _artifacts(tmp_path)
        readme, evaluation, architecture, claude = _docs(tmp_path)

        assert runner.synchronize(_args(runner, tmp_path)) == (
            readme,
            evaluation,
            architecture,
            claude,
        )
        first = tuple(read_text(path) for path in (readme, evaluation, architecture, claude))
        headline = (
            first[0]
            .split("<!-- BEGIN M8-README-HEADLINE -->", 1)[1]
            .split("<!-- END M8-README-HEADLINE -->", 1)[0]
        )
        assert "\n" not in headline
        assert headline.startswith("本机 HTTP 基准（4 次正式请求，并发 1，查询向量走本地缓存")
        assert "p95 380.0 ms" in headline
        assert "吞吐 3.00 QPS" in headline
        assert "成功 3/4" in headline
        assert "未启用重排，不含模型调用耗时" in headline
        assert "- 服务：<!-- BEGIN M8-README-HEADLINE -->本机" in first[0]
        assert "dense encode" not in first[0]
        assert "HTTP 服务与 M8 性能基准" in first[1]
        assert "380.0 ms" in first[1]
        assert "3.00 QPS" in first[1]
        assert "dense encode" in first[1]
        assert "cache-backed" in first[1]
        assert "不调用 chat completion" in first[1]
        roadmap = (
            first[2]
            .split("<!-- BEGIN M8-ROADMAP -->", 1)[1]
            .split("<!-- END M8-ROADMAP -->", 1)[0]
            .strip()
        )
        assert roadmap.startswith("| **M8** |")
        assert roadmap.count("\n") == 0
        assert roadmap.count("|") == 6
        evidence = (
            first[2]
            .split("<!-- BEGIN M8-RESUME-EVIDENCE -->", 1)[1]
            .split("<!-- END M8-RESUME-EVIDENCE -->", 1)[0]
        )
        performance = (
            first[2]
            .split("<!-- BEGIN M8-RESUME-PERFORMANCE -->", 1)[1]
            .split("<!-- END M8-RESUME-PERFORMANCE -->", 1)[0]
        )
        assert "4 次正式请求" in evidence
        assert "p95 380.0 ms / 3.00 QPS" in performance
        assert "不含 provider 墙钟" in performance
        assert "m8_http_samples" in first[3]
        serialized = "\n".join(first)
        for forbidden in ("SECRET_QUERY", "SECRET_DOC", '"query"', '"doc_id"'):
            assert forbidden not in serialized

        assert runner.synchronize(_args(runner, tmp_path)) == ()
        assert (
            tuple(read_text(path) for path in (readme, evaluation, architecture, claude)) == first
        )
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

    def test_check_reports_stale_docs_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        _artifacts(tmp_path)
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)
        with pytest.raises(SystemExit, match="documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))
        assert tuple(read_text(path) for path in docs) == before

    def test_docs_lock_contention_blocks_before_artifact_lock(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts, _numeric, _report = _artifacts(tmp_path)
        _docs(tmp_path)
        docs_lock = tmp_path / runner.DOCS_LOCK
        write_text(docs_lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.synchronize(_args(runner, tmp_path))
        assert read_text(docs_lock) == "pid=123\n"
        assert not (artifacts / runner.ARTIFACT_LOCK).exists()

    def test_artifact_lock_contention_cleans_docs_lock(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts, _numeric, _report = _artifacts(tmp_path)
        _docs(tmp_path)
        artifact_lock = artifacts / runner.ARTIFACT_LOCK
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
        _artifacts(tmp_path)
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
        monkeypatch.undo()
        assert tuple(read_text(path) for path in docs) == before
        assert not list(tmp_path.glob("*.m8-sync.tmp"))

    def test_validates_all_markers_before_writing_anything(self, tmp_path: Path) -> None:
        runner = _runner()
        _artifacts(tmp_path)
        docs = _docs(tmp_path, duplicate_readme_marker=True)
        before = tuple(read_text(path) for path in docs)

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))
        assert tuple(read_text(path) for path in docs) == before


class TestBenchmarkPublicationLock:
    def test_benchmark_publisher_uses_the_same_artifact_lock(self, tmp_path: Path) -> None:
        path = Path(__file__).resolve().parent.parent / "scripts" / "bench.py"
        spec = importlib.util.spec_from_file_location("bench_lock_test_module", path)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)
        samples_path = tmp_path / "samples.json"
        report_path = tmp_path / "report.json"
        write_text(tmp_path / bench.M8_ARTIFACT_LOCK, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            bench._publish(
                samples_path=samples_path,
                report_path=report_path,
                samples={"schema": "numeric"},
                report={"schema": "report"},
            )
        assert not samples_path.exists()
        assert not report_path.exists()

    def test_publisher_rejects_split_artifact_directories(self, tmp_path: Path) -> None:
        path = Path(__file__).resolve().parent.parent / "scripts" / "bench.py"
        spec = importlib.util.spec_from_file_location("bench_split_test_module", path)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench
        spec.loader.exec_module(bench)

        with pytest.raises(ValueError, match="share an artifact directory"):
            bench._publish(
                samples_path=tmp_path / "left" / "samples.json",
                report_path=tmp_path / "right" / "report.json",
                samples={"schema": "numeric"},
                report={"schema": "report"},
            )

    def test_existing_artifacts_are_valid_test_fixtures(self, tmp_path: Path) -> None:
        artifacts, numeric, report = _artifacts(tmp_path)
        assert read_json(artifacts / "m8_http_samples.json") == numeric
        assert read_json(artifacts / "m8_http_report.json") == report
        assert copy.deepcopy(report) == report
