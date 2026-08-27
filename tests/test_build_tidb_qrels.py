from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.eval.pool import PooledQuery
from zhrag.io_utils import (
    append_jsonl,
    read_json,
    read_jsonl,
    read_text,
    write_json,
    write_text,
)
from zhrag.providers.chat import ChatConfig


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_tidb_qrels.py"
    spec = importlib.util.spec_from_file_location("build_tidb_qrels_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unit() -> PooledQuery:
    return PooledQuery(
        chunk_id="gold",
        query_ids=("direct:1", "paraphrase:2"),
        questions=("直接问题", "改写问题"),
        answer="答案",
        candidates=("gold", "other"),
        contributors={"gold": (), "other": ("dense",)},
    )


class TestCacheRows:
    def test_last_write_wins_and_rejects_partial_usage(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        append_jsonl(
            cache,
            [
                {"id": "a", "content": "old", "model": "m"},
                {
                    "id": "a",
                    "content": "new",
                    "model": "m",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reasoning_tokens": 3,
                },
            ],
        )
        assert runner._cache_entries(cache)["a"].content == "new"

        append_jsonl(cache, [{"id": "b", "content": "x", "model": "m", "prompt_tokens": 1}])
        with pytest.raises(SystemExit, match="incomplete"):
            runner._cache_entries(cache)


class TestInputValidation:
    def test_rejects_question_or_gold_drift(self) -> None:
        runner = _runner()
        rows = {
            "direct:1": {
                "question": "wrong",
                "answer": "答案",
                "task": "direct",
                "gold_doc_ids": ["gold"],
            },
            "paraphrase:2": {
                "question": "改写问题",
                "answer": "答案",
                "task": "paraphrase",
                "gold_doc_ids": ["gold"],
            },
        }
        with pytest.raises(SystemExit, match="surface"):
            runner._pool_unit(
                {
                    "schema": runner.POOL_SCHEMA,
                    "query_ids": ["direct:1", "paraphrase:2"],
                    "questions": ["直接问题", "改写问题"],
                    "answer": "答案",
                    "generating_chunk_id": "gold",
                    "candidate_doc_ids": ["gold", "other"],
                    "contributors": {"gold": [], "other": ["dense"]},
                    "tasks": ["direct", "paraphrase"],
                },
                row_number=1,
                query_rows=rows,
                corpus={"gold": "第一段", "other": "第二段"},
            )


class TestProvenance:
    def test_finalization_does_not_adopt_absent_sidecar(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [{"id": "a", "content": "{}", "model": "m"}])
        with pytest.raises(SystemExit, match="absent"):
            runner._prepare_or_validate_sidecar(
                cache,
                {"schema": "x"},
                judge=False,
                config=None,
            )
        assert not Path(f"{cache}.meta.json").exists()

    def test_validates_exact_sidecar(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        write_json(Path(f"{cache}.meta.json"), {"schema": "x", "model": "m", "endpoint": "e"})
        assert runner._prepare_or_validate_sidecar(
            cache,
            {"schema": "x"},
            judge=False,
            config=None,
        ) == {"schema": "x", "model": "m", "endpoint": "e"}


class TestCommandModes:
    def _batch(self, runner: ModuleType) -> object:
        unit = _unit()
        return runner._Batch(
            runner.judging_cache_id(unit.questions, unit.answer, unit.candidates),
            unit,
            unit.candidates,
        )

    def _experiment(self, runner: ModuleType) -> object:
        unit = _unit()
        return runner._Experiment(
            corpus={"gold": "第一段", "other": "第二段"},
            query_rows={},
            runs=None,
            units=(unit,),
            pool_report={},
            query_set_fingerprint="queries",
        )

    def _patch_common(
        self,
        runner: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        valid: bool,
    ) -> tuple[object, object]:
        experiment = self._experiment(runner)
        batch = self._batch(runner)
        entry = runner._CacheEntry(
            content='{"grades":{"gold":2,"other":0}}',
            model="model",
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
        )
        parsed = {batch.cache_id: {"gold": 2, "other": 0}}
        monkeypatch.setattr(runner, "_load_experiment", lambda _args: experiment)
        monkeypatch.setattr(runner, "_batches", lambda *_args, **_kwargs: (batch,))
        monkeypatch.setattr(
            runner,
            "_base_provenance",
            lambda *_args, **_kwargs: {"schema": runner.JUDGING_SCHEMA},
        )
        monkeypatch.setattr(
            runner,
            "_prepare_or_validate_sidecar",
            lambda *_args, **_kwargs: {
                "schema": runner.JUDGING_SCHEMA,
                "model": "model",
                "endpoint": "https://example.test/v1/chat/completions",
            },
        )
        monkeypatch.setattr(runner, "_cache_entries", lambda _path: {batch.cache_id: entry})
        monkeypatch.setattr(
            runner,
            "_valid_grades",
            lambda _entries, _batches: (parsed, set() if valid else {batch.cache_id}),
        )
        return experiment, batch

    def test_complete_cache_without_flags_does_not_publish(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_common(runner, monkeypatch, tmp_path, valid=True)
        published: list[object] = []
        monkeypatch.setattr(
            runner, "_publish", lambda *_args, **_kwargs: published.append(object())
        )

        assert runner.main(["--artifacts", str(tmp_path)]) == 0
        assert published == []

    def test_judge_filling_last_batch_does_not_publish(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        experiment, batch = self._patch_common(runner, monkeypatch, tmp_path, valid=False)
        state = {"valid": False}
        parsed = {batch.cache_id: {"gold": 2, "other": 0}}

        def valid_grades(_entries: object, _batches: object) -> tuple[object, set[str]]:
            return parsed, set() if state["valid"] else {batch.cache_id}

        def judge_missing(*_args: object, **_kwargs: object) -> None:
            state["valid"] = True

        config = ChatConfig("secret", "https://example.test/v1", "model")
        monkeypatch.setattr(runner.ChatConfig, "from_env", lambda _env: config)
        monkeypatch.setattr(runner, "load_env", lambda _path: {})
        monkeypatch.setattr(runner, "_valid_grades", valid_grades)
        monkeypatch.setattr(runner, "_judge_missing", judge_missing)
        published: list[object] = []
        monkeypatch.setattr(
            runner, "_publish", lambda *_args, **_kwargs: published.append(object())
        )

        assert runner.main(["--artifacts", str(tmp_path), "--judge"]) == 0
        assert state["valid"]
        assert published == []
        assert experiment.units[0].chunk_id == "gold"

    def test_finalize_complete_cache_publishes_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_common(runner, monkeypatch, tmp_path, valid=True)
        published: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            runner,
            "_publish",
            lambda *args, **kwargs: published.append((*args, kwargs)),
        )

        assert runner.main(["--artifacts", str(tmp_path), "--finalize"]) == 0
        assert len(published) == 1

    def test_incomplete_finalize_preserves_existing_bundle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_common(runner, monkeypatch, tmp_path, valid=False)
        eval_root = tmp_path / "eval"
        qrels = eval_root / runner.QRELS.name
        report = eval_root / runner.QRELS_REPORT.name
        write_text(qrels, "old qrels")
        write_text(report, "old report")
        monkeypatch.setattr(
            runner,
            "_publish",
            lambda *_args, **_kwargs: pytest.fail("incomplete cache must not publish"),
        )

        assert runner.main(["--artifacts", str(tmp_path), "--finalize"]) == 0
        assert read_text(qrels) == "old qrels"
        assert read_text(report) == "old report"

    def test_rejects_combined_judge_and_finalize(self, tmp_path: Path) -> None:
        runner = _runner()
        with pytest.raises(SystemExit, match="mutually exclusive"):
            runner.main(["--artifacts", str(tmp_path), "--judge", "--finalize"])

    def test_default_status_check_does_not_touch_published_bundle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_common(runner, monkeypatch, tmp_path, valid=True)
        eval_root = tmp_path / "eval"
        qrels = eval_root / runner.QRELS.name
        report = eval_root / runner.QRELS_REPORT.name
        write_text(qrels, "old qrels")
        write_text(report, "old report")
        before = (qrels.stat().st_mtime_ns, report.stat().st_mtime_ns)

        assert runner.main(["--artifacts", str(tmp_path)]) == 0
        assert (read_text(qrels), read_text(report)) == ("old qrels", "old report")
        assert (qrels.stat().st_mtime_ns, report.stat().st_mtime_ns) == before


class TestArtifactPublication:
    def test_publishes_qrels_and_report_under_the_shared_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        unit = _unit()
        query_rows = {
            query_id: {
                "query_id": query_id,
                "question": question,
                "answer": unit.answer,
                "gold_doc_ids": [unit.chunk_id],
                "gold_source_key": "source",
                "task": query_id.partition(":")[0],
                "question_type": "config",
                "theme": "sql",
                "bigram_containment": 0.5,
            }
            for query_id, question in zip(unit.query_ids, unit.questions, strict=True)
        }
        labels = (
            runner.LEXICAL_LABEL,
            runner.DENSE_LABEL,
            runner.RRF_LABEL,
            runner.RERANK_LABEL,
        )
        runs = runner.TiDBRuns(
            query_ids=unit.query_ids,
            runs={label: (("gold", "other"), ("gold", "other")) for label in labels},
        )
        experiment = runner._Experiment(
            corpus={"gold": "第一段", "other": "第二段"},
            query_rows=query_rows,
            runs=runs,
            units=(unit,),
            pool_report={},
            query_set_fingerprint="a" * 64,
        )
        judged = runner.JudgedQuery(
            chunk_id=unit.chunk_id,
            query_ids=unit.query_ids,
            grades={"gold": 2, "other": 0},
        )
        batch = runner._Batch("batch", unit, unit.candidates)
        entry = runner._CacheEntry("{}", "model", None, None, None)
        eval_root = tmp_path / "eval"
        shared_lock = eval_root / runner.ARTIFACT_LOCK
        targets: list[str] = []
        real_replace = runner.replace_files

        def replace_while_locked(staged: tuple[tuple[Path, Path], ...]) -> None:
            assert shared_lock.is_file()
            pairs = tuple(staged)
            targets.extend(target.name for _source, target in pairs)
            real_replace(pairs)

        monkeypatch.setattr(runner, "replace_files", replace_while_locked)

        runner._publish(
            runner._parse_args(["--artifacts", str(tmp_path)]),
            experiment,
            [judged],
            {batch.cache_id: entry},
            [batch],
            provenance={
                "batch_size": 8,
                "order_seed": "seed",
                "judging_input_fingerprint_sha256": "b" * 64,
                "model": "model",
                "endpoint": "https://example.test/v1",
            },
        )

        assert targets == ["qrels.jsonl", "qrels_report.json"]
        assert len(list(read_jsonl(eval_root / "qrels.jsonl"))) == 2
        assert read_json(eval_root / "qrels_report.json")["pairs"] == 1
        assert not shared_lock.exists()
