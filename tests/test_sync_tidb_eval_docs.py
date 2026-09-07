from __future__ import annotations

import copy
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from test_tidb_quality import _Fixture
from zhrag.eval.tidb_quality import PRIMARY_METRIC, RUN_LABELS
from zhrag.io_utils import read_json, read_text, write_json, write_jsonl, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_tidb_eval_docs.py"
    spec = importlib.util.spec_from_file_location("sync_tidb_eval_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reports(
    root: Path,
    *,
    qrels_overrides: dict[str, object] | None = None,
    qgen_overrides: dict[str, object] | None = None,
    quality_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    fixture = _Fixture()
    artifacts = root / "indexes" / "tidb"
    eval_root = artifacts / "eval"
    state = copy.deepcopy(fixture.state)
    state["chunker_fingerprint"] = "chunker"
    state["scope"] = "scope"
    state["sparse_fingerprint"] = "sparse"
    qgen = {
        "schema": "zhrag-tidb-qgen-v1",
        "sampled_chunks": 4,
        "generator_model_requested": "same-model",
        "generator_models_served": {"same-model": 4},
        "verifier_model_requested": "same-model",
        "verifier_models_served": {"same-model": 3},
        "verification_independence": "same-requested-model self-agreement",
        "reasoning_effort": "high",
        "pairs_verified": 4,
        "pairs_dropped": 1,
        "complete_pairs": 3,
        "queries": 6,
        "queries_by_variant": {"direct": 3, "paraphrase": 3},
        "published_index": {
            "chunks": len(fixture.doc_ids),
            "collection": "tidb_chunks_v1",
            "chunker": "chunker",
            "embedding_profile": "qwen3-embedding-8b-tidb-doc-4096-v1",
            "scope": "scope",
            "sparse": "sparse",
        },
    }
    if qgen_overrides:
        qgen.update(qgen_overrides)
    pool = copy.deepcopy(fixture.pool_report)
    qrels = copy.deepcopy(fixture.qrels_report)
    qrels["requested_model"] = "same-model"
    qrels["served_models"] = {"same-model": qrels["batches"]}
    qrels["reasoning_effort"] = "high"
    if qrels_overrides:
        qrels.update(qrels_overrides)
    quality = fixture.evaluate(resamples=30, seed=7)
    if quality_mutator is not None:
        quality_mutator(quality)
    write_json(artifacts / "state.json", state)
    write_json(eval_root / "report.json", qgen)
    write_json(eval_root / "pool_report.json", pool)
    write_json(eval_root / "qrels_report.json", qrels)
    write_json(eval_root / "quality_report.json", quality)
    write_jsonl(eval_root / "runs.jsonl", fixture.run_rows)
    write_jsonl(eval_root / "qrels.jsonl", fixture.qrel_rows)
    return artifacts


def _mutate(
    path: tuple[str | int, ...],
    value: object,
) -> Callable[[dict[str, Any]], None]:
    def apply(report: dict[str, Any]) -> None:
        target: Any = report
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


def _read_quality(artifacts: Path) -> dict[str, Any]:
    value = read_json(artifacts / "eval" / "quality_report.json")
    assert isinstance(value, dict)
    return value


def _docs(
    root: Path,
    *,
    duplicate_readme_marker: bool = False,
) -> tuple[Path, Path, Path, Path]:
    readme = root / "README.md"
    evaluation = root / "docs" / "evaluation.md"
    architecture = root / "docs" / "architecture-decision.md"
    claude = root / "CLAUDE.md"
    summary_region = "<!-- BEGIN TIDB-EVAL-SUMMARY -->\nstale\n<!-- END TIDB-EVAL-SUMMARY -->"
    if duplicate_readme_marker:
        summary_region = f"{summary_region}\n{summary_region}"
    write_text(
        readme,
        "\n".join(
            (
                "before",
                summary_region,
                "```",
                "<!-- BEGIN QUALITY-GATE-STATUS -->",
                "stale",
                "<!-- END QUALITY-GATE-STATUS -->",
                "after",
            )
        ),
    )
    write_text(
        evaluation,
        "\n".join(
            (
                "before",
                "<!-- BEGIN TIDB-EVAL-STATUS -->",
                "stale",
                "<!-- END TIDB-EVAL-STATUS -->",
                "<!-- BEGIN TIDB-EVAL-EVIDENCE -->",
                "stale",
                "<!-- END TIDB-EVAL-EVIDENCE -->",
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
    return readme, evaluation, architecture, claude


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> object:
    argv = [
        "--artifacts",
        str(root / "indexes" / "tidb"),
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


class TestReportValidation:
    def test_reconciles_all_report_families(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)

        status = runner.load_status(artifacts)

        fixture = _Fixture()
        assert status.documents == 1
        assert status.chunks == len(fixture.doc_ids)
        assert status.verified_pairs == 3
        assert status.queries == 6
        assert status.pool_candidates == fixture.pool_report["pool_candidates_total"]
        assert status.judging_batches == fixture.qrels_report["batches"]
        assert status.quality["schema"] == "zhrag-tidb-retrieval-quality-v2"
        assert status.source_clusters == 2

    @pytest.mark.parametrize("name", ["runs.jsonl", "qrels.jsonl"])
    def test_requires_raw_artifacts_to_authenticate_the_report(
        self,
        tmp_path: Path,
        name: str,
    ) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        (artifacts / "eval" / name).unlink()

        with pytest.raises(SystemExit, match="raw artifacts are absent"):
            runner.load_status(artifacts)

    def test_rejects_structurally_valid_forged_statistics(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        quality = _read_quality(artifacts)
        interval = quality["system_metrics"][RUN_LABELS[0]]["overall"][PRIMARY_METRIC]
        interval["mean"] = 0.987654
        interval["low"] = 0.9
        interval["high"] = 0.99
        write_json(artifacts / "eval" / "quality_report.json", quality)

        with pytest.raises(SystemExit, match="deterministic recomputation"):
            runner.load_status(artifacts)

    def test_rejects_a_forged_qrels_semantic_digest(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        quality = _read_quality(artifacts)
        quality["inputs"]["qrels_semantic_fingerprint_sha256"] = "a" * 64
        write_json(artifacts / "eval" / "quality_report.json", quality)

        with pytest.raises(SystemExit, match="deterministic recomputation"):
            runner.load_status(artifacts)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"query_set_fingerprint_sha256": "0" * 64}, "query-set fingerprint"),
            ({"cache_valid_batches": 1}, "cache batch count"),
            ({"grades": {"0": 1, "1": 1, "2": 1}}, "grade counts"),
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

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (
                ("inputs", "query_set_fingerprint_sha256"),
                "0" * 64,
                "quality query_set_fingerprint",
            ),
            (("qrels_quality", "raw_grades", "0"), 1, "raw grade aggregates"),
            (
                ("qrels_quality", "generating_chunk_grades", "1"),
                0,
                "generating grade aggregates",
            ),
            (("qrels_quality", "promoted_generating_chunks"), 0, "promoted generating"),
            (("qrels_quality", "judging_batches"), 1, "judging batch"),
            (
                (
                    "qrels_quality",
                    "run_judged_coverage",
                    "dense-qwen3-4096",
                    "top10_all_returned_judged",
                ),
                5,
                "top-10 is inconsistent",
            ),
        ],
    )
    def test_fails_closed_on_quality_report_drift(
        self,
        tmp_path: Path,
        path: tuple[str | int, ...],
        value: object,
        message: str,
    ) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        quality = _read_quality(artifacts)
        _mutate(path, value)(quality)
        write_json(artifacts / "eval" / "quality_report.json", quality)

        with pytest.raises(SystemExit, match=message):
            runner.load_status(artifacts)

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("evaluation_design", "paired_test"), "unpaired", "inferential method"),
            (
                ("system_metrics", RUN_LABELS[0], "overall", PRIMARY_METRIC, "low"),
                2.0,
                "interval bounds",
            ),
            (("primary_contrasts", 0, "adjusted_p"), 0.123456, "Holm-adjusted"),
            (("primary_contrasts", 0, "raw_p_at_floor"), True, "floor provenance"),
        ],
    )
    def test_rejects_malformed_quality_method_or_statistics(
        self,
        tmp_path: Path,
        path: tuple[str | int, ...],
        value: object,
        message: str,
    ) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        quality = _read_quality(artifacts)
        _mutate(path, value)(quality)
        write_json(artifacts / "eval" / "quality_report.json", quality)

        with pytest.raises(SystemExit, match=message):
            runner.load_status(artifacts)

    @pytest.mark.parametrize(
        ("qgen_overrides", "qrels_overrides", "message"),
        [
            (
                {"verifier_model_requested": "other-model"},
                None,
                "synthetic label requested model",
            ),
            (
                {"verification_independence": "unknown"},
                None,
                "unknown verification independence",
            ),
            (
                None,
                {"reasoning_effort": "low"},
                "synthetic label reasoning effort",
            ),
            (
                {"generator_models_served": {"different-served-model": 4}},
                None,
                "served synthetic-label models",
            ),
        ],
    )
    def test_rejects_label_provenance_drift(
        self,
        tmp_path: Path,
        qgen_overrides: dict[str, object] | None,
        qrels_overrides: dict[str, object] | None,
        message: str,
    ) -> None:
        runner = _runner()
        artifacts = _reports(
            tmp_path,
            qgen_overrides=qgen_overrides,
            qrels_overrides=qrels_overrides,
        )

        with pytest.raises(SystemExit, match=message):
            runner.load_status(artifacts)

    def test_requires_quality_report(self, tmp_path: Path) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        (artifacts / "eval" / "quality_report.json").unlink()

        with pytest.raises(FileNotFoundError, match="quality_report"):
            runner.load_status(artifacts)

    def test_rejects_incomplete_coverage_or_quality_gate(self, tmp_path: Path) -> None:
        runner = _runner()
        fixture = _Fixture()
        coverage = copy.deepcopy(fixture.qrels_report["run_judged_coverage"])
        coverage["dense-qwen3-4096"]["top10_complete"] = 5
        artifacts = _reports(tmp_path, qrels_overrides={"run_judged_coverage": coverage})
        with pytest.raises(SystemExit, match="coverage is incomplete"):
            runner.load_status(artifacts)

        artifacts = _reports(tmp_path)
        status = runner.load_status(artifacts)
        assert status.quality["schema"] == "zhrag-tidb-retrieval-quality-v2"
        assert not (artifacts / "eval" / "pytest.xml").exists()


class TestSynchronization:
    def test_renders_allowlisted_aggregates_and_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        readme, evaluation, architecture, claude = _docs(tmp_path)

        assert runner.synchronize(_args(runner, tmp_path)) == (
            readme,
            evaluation,
            architecture,
            claude,
        )
        first = tuple(read_text(path) for path in (readme, evaluation, architecture, claude))
        summary = (
            first[0]
            .split("<!-- BEGIN TIDB-EVAL-SUMMARY -->", 1)[1]
            .split("<!-- END TIDB-EVAL-SUMMARY -->", 1)[0]
        )
        assert summary.startswith("\n在 **1 篇 TiDB 文档、6 条测试问题**上")
        assert "| 检索方案 | 首条命中率 | 95% 置信区间 |" in summary
        assert "| 关键词检索 | " in summary
        assert "| 融合 + 模型重排 | " in summary
        assert summary.count("\n| ") == 5
        assert "Holm 校正 p = " in summary
        assert "本次检验未检测到显著差异" in summary
        assert "不是抽样偶然" not in summary
        assert "不是生成答案的正确率" in summary
        assert "同一模型" in summary
        assert "<details>" in summary and "</details>" in summary
        assert "2 个来源文档簇" in summary
        assert "[评估文档](docs/evaluation.md)" in summary
        assert "预声明主检验族" not in first[0]
        assert "词面重叠分层" not in first[0]
        assert (
            "<!-- BEGIN QUALITY-GATE-STATUS -->\nstale\n<!-- END QUALITY-GATE-STATUS -->"
        ) in first[0]
        assert "`pytest` 7 passed" not in first[0]
        assert "3 组 direct/paraphrase、6 条 query" in first[1]
        assert "2 个 `gold_source_key` 源聚类" in first[1]
        assert "source-cluster bootstrap 95% CI" in first[1]
        assert "独立单位" not in first[1]
        assert "`ruff check` 全通过" not in first[1]
        assert "`mypy --strict` 无告警" not in first[1]
        assert "binary nDCG@10" in first[1]
        assert "预声明主检验族" in first[1]
        assert "direct → paraphrase robustness" in first[1]
        assert "词面重叠分层" in first[1]
        assert "same-model self-agreement" in first[1]
        assert "100% **已判断覆盖**" in first[1]
        assert "无上游人工 gold" in first[2]
        assert "HTTP 延迟/QPS 已由 M8 独立认证" in first[2]
        assert "增量重建仍待测" in first[2]
        assert "延迟/QPS/重建仍待测" not in first[2]
        assert "quality_report.json" in first[3]
        serialized = "\n".join(first)
        for forbidden in (
            "直接问题",
            "改写问题",
            "答案 0",
            "source-0",
            "chunk-000",
            "direct:unrelated",
        ):
            assert forbidden not in serialized
        assert runner.synchronize(_args(runner, tmp_path)) == ()
        assert (
            tuple(read_text(path) for path in (readme, evaluation, architecture, claude)) == first
        )
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

    @pytest.mark.parametrize("delta", [-0.2, 0.2])
    def test_significant_summary_does_not_assume_improvement(
        self, tmp_path: Path, delta: float
    ) -> None:
        runner = _runner()
        status = runner.load_status(_reports(tmp_path))
        contrast = runner._primary_contrast(
            status, treatment=runner.RERANK_LABEL, comparator=runner.RRF_LABEL
        )
        contrast["reject"] = True
        contrast["adjusted_p"] = 0.01
        contrast["adjusted_p_inherits_floor"] = False
        contrast["delta"] = {"mean": delta, "low": delta - 0.01, "high": delta + 0.01}
        summary = runner._readme_summary(status)
        assert "本次检验检测到差异；方向以差值为准" in summary
        assert f"{delta:.4f}" in summary
        assert "不是抽样偶然" not in summary
        assert "提升" not in summary

    def test_check_reports_stale_docs_without_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        docs = _docs(tmp_path)
        before = tuple(read_text(path) for path in docs)

        with pytest.raises(SystemExit, match="documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))

        assert tuple(read_text(path) for path in docs) == before

    def test_publication_failure_rolls_back_every_document(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        _reports(tmp_path)
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
        assert not list(tmp_path.glob("*.sync.tmp"))
        assert not list((tmp_path / "docs").glob("*.sync.tmp"))

    def test_refuses_to_run_while_another_docs_sync_holds_the_shared_lock(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        _reports(tmp_path)
        _docs(tmp_path)
        lock = tmp_path / runner.DOCS_LOCK
        write_text(lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.synchronize(_args(runner, tmp_path))

        assert read_text(lock) == "pid=123\n"
        assert not (tmp_path / "indexes" / "tidb" / "eval" / runner.ARTIFACT_LOCK).exists()

    def test_refuses_to_run_while_a_bundle_writer_holds_the_shared_lock(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        artifacts = _reports(tmp_path)
        _docs(tmp_path)
        lock = artifacts / "eval" / runner.ARTIFACT_LOCK
        write_text(lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.synchronize(_args(runner, tmp_path))

        assert read_text(lock) == "pid=123\n"

    def test_validates_every_document_before_writing_any(self, tmp_path: Path) -> None:
        runner = _runner()
        _reports(tmp_path)
        docs = _docs(tmp_path, duplicate_readme_marker=True)
        before = tuple(read_text(path) for path in docs)

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))

        assert tuple(read_text(path) for path in docs) == before
