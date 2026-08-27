from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from test_tidb_quality import _Fixture
from zhrag.eval.tidb_quality import TIDB_QUALITY_SCHEMA
from zhrag.io_utils import read_json, read_text, write_json, write_jsonl, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_tidb_retrieval.py"
    spec = importlib.util.spec_from_file_location("evaluate_tidb_retrieval_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifacts(root: Path, fixture: _Fixture) -> Path:
    artifacts = root / "indexes" / "tidb"
    eval_root = artifacts / "eval"
    write_json(artifacts / "state.json", fixture.state)
    write_json(eval_root / "pool_report.json", fixture.pool_report)
    write_json(eval_root / "qrels_report.json", fixture.qrels_report)
    write_jsonl(eval_root / "runs.jsonl", fixture.run_rows)
    write_jsonl(eval_root / "qrels.jsonl", fixture.qrel_rows)
    return artifacts


def _args(runner: ModuleType, artifacts: Path, *extra: str) -> argparse.Namespace:
    return runner._parse_args(
        ["--artifacts", str(artifacts), "--resamples", "30", "--seed", "7", *extra]
    )


class TestArguments:
    def test_cli_has_only_offline_evaluation_controls(self) -> None:
        runner = _runner()
        parser_args = runner._parse_args(["--artifacts", "local", "--resamples", "2"])
        assert vars(parser_args) == {
            "artifacts": Path("local"),
            "resamples": 2,
            "seed": 0,
        }

        with pytest.raises(SystemExit):
            runner._parse_args(["--judge"])
        with pytest.raises(SystemExit):
            runner._parse_args(["--embed"])
        with pytest.raises(SystemExit):
            runner._parse_args(["--rerank"])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--resamples", "0"],
            ["--seed", "-1"],
            ["--seed", str(1 << 63)],
        ],
    )
    def test_rejects_invalid_bootstrap_arguments(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            _runner()._parse_args(argv)


class TestOfflinePublication:
    def test_publishes_one_valid_aggregate_report(self, tmp_path: Path) -> None:
        fixture = _Fixture()
        artifacts = _write_artifacts(tmp_path, fixture)
        runner = _runner()

        assert runner.main(["--artifacts", str(artifacts), "--resamples", "30", "--seed", "7"]) == 0

        report = read_json(artifacts / "eval" / "quality_report.json")
        assert report["schema"] == TIDB_QUALITY_SCHEMA
        assert report["evaluation_design"]["pairs"] == 3
        assert not (artifacts / "eval" / "quality_report.json.tmp").exists()

    def test_clean_import_path_loads_no_provider_modules(self) -> None:
        root = Path(__file__).resolve().parent.parent
        script = root / "scripts" / "evaluate_tidb_retrieval.py"
        probe = (
            "import json, runpy, sys; "
            f"runpy.run_path({str(script)!r}, run_name='offline_import_probe'); "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name == 'zhrag.providers' or name.startswith('zhrag.providers.'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            cwd=root,
            encoding="utf-8",
        )

        assert json.loads(completed.stdout) == []

    def test_identical_inputs_are_byte_reproducible(self, tmp_path: Path) -> None:
        artifacts = _write_artifacts(tmp_path, _Fixture())
        runner = _runner()
        argv = ["--artifacts", str(artifacts), "--resamples", "30", "--seed", "7"]

        runner.main(argv)
        first = read_text(artifacts / "eval" / "quality_report.json")
        runner.main(argv)
        second = read_text(artifacts / "eval" / "quality_report.json")

        assert first == second

    def test_validation_failure_preserves_previous_report(self, tmp_path: Path) -> None:
        fixture = _Fixture()
        artifacts = _write_artifacts(tmp_path, fixture)
        runner = _runner()
        target = artifacts / "eval" / "quality_report.json"
        write_text(target, "previous-good-report\n")
        broken = copy.deepcopy(fixture.qrels_report)
        broken["runs_fingerprint_sha256"] = "0" * 64
        write_json(artifacts / "eval" / "qrels_report.json", broken)

        with pytest.raises(SystemExit, match="runs fingerprint"):
            runner.main(["--artifacts", str(artifacts), "--resamples", "10"])

        assert read_text(target) == "previous-good-report\n"
        assert not (artifacts / "eval" / "quality_report.json.tmp").exists()

    @pytest.mark.parametrize("lock_name", [".pool.lock", ".qrels.lock"])
    def test_input_writer_lock_blocks_reading_without_cleanup(
        self,
        tmp_path: Path,
        lock_name: str,
    ) -> None:
        artifacts = _write_artifacts(tmp_path, _Fixture())
        lock = artifacts / "eval" / lock_name
        write_text(lock, "pid=123\n")

        with pytest.raises(SystemExit, match="input writer lock"):
            _runner().main(["--artifacts", str(artifacts), "--resamples", "10"])

        assert read_text(lock) == "pid=123\n"
        assert not (artifacts / "eval" / "quality_report.json").exists()

    def test_shared_artifact_lock_blocks_before_reading_and_cleans_operation_lock(
        self,
        tmp_path: Path,
    ) -> None:
        artifacts = _write_artifacts(tmp_path, _Fixture())
        eval_root = artifacts / "eval"
        shared_lock = eval_root / ".artifacts.lock"
        write_text(shared_lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            _runner().main(["--artifacts", str(artifacts), "--resamples", "10"])

        assert read_text(shared_lock) == "pid=123\n"
        assert not (eval_root / ".quality.lock").exists()
        assert not (eval_root / "quality_report.json").exists()

    def test_holds_both_locks_through_evaluation_and_cleans_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        artifacts = _write_artifacts(tmp_path, _Fixture())
        eval_root = artifacts / "eval"
        runner = _runner()
        real_evaluate = runner.evaluate_tidb_quality
        observed = {"called": False}

        def guarded_evaluate(**kwargs: object) -> object:
            assert (eval_root / ".quality.lock").is_file()
            assert (eval_root / ".artifacts.lock").is_file()
            observed["called"] = True
            return real_evaluate(**kwargs)

        monkeypatch.setattr(runner, "evaluate_tidb_quality", guarded_evaluate)

        assert runner.main(["--artifacts", str(artifacts), "--resamples", "10"]) == 0
        assert observed["called"]
        assert not (eval_root / ".quality.lock").exists()
        assert not (eval_root / ".artifacts.lock").exists()

    def test_evaluation_exception_cleans_both_locks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        artifacts = _write_artifacts(tmp_path, _Fixture())
        eval_root = artifacts / "eval"
        runner = _runner()

        def fail_while_locked(**_kwargs: object) -> object:
            assert (eval_root / ".quality.lock").is_file()
            assert (eval_root / ".artifacts.lock").is_file()
            raise ValueError("injected evaluation failure")

        monkeypatch.setattr(runner, "evaluate_tidb_quality", fail_while_locked)

        with pytest.raises(SystemExit, match="injected evaluation failure"):
            runner.main(["--artifacts", str(artifacts), "--resamples", "10"])

        assert not (eval_root / ".quality.lock").exists()
        assert not (eval_root / ".artifacts.lock").exists()

    def test_staging_failure_keeps_previous_report_and_cleans_temp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        fixture = _Fixture()
        artifacts = _write_artifacts(tmp_path, fixture)
        runner = _runner()
        target = artifacts / "eval" / "quality_report.json"
        write_text(target, "old\n")
        report = runner._load_and_evaluate(_args(runner, artifacts))
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            runner._publish(_args(runner, artifacts), report)

        assert read_text(target) == "old\n"
        assert not (artifacts / "eval" / "quality_report.json.tmp").exists()

    def test_chinese_artifacts_load_without_pythonutf8(self, tmp_path: Path) -> None:
        fixture = _Fixture()
        fixture.qrel_rows[0]["question"] = "TiDB 中如何配置事务？"
        pair_id = fixture.qrel_rows[0]["generating_chunk_id"]
        twin = next(
            row
            for row in fixture.qrel_rows
            if row["generating_chunk_id"] == pair_id and row is not fixture.qrel_rows[0]
        )
        # Pair questions can differ, while all shared controls still match.
        twin["question"] = "事务设置应该怎样调整？"
        artifacts = _write_artifacts(tmp_path, fixture)
        bundle = runner_bundle = _runner()

        with pytest.raises(SystemExit, match="query-set fingerprint"):
            # The text changed after the fixture reports were frozen, so strict
            # provenance should fail cleanly rather than hit a cp936 decode error.
            runner_bundle.main(["--artifacts", str(artifacts), "--resamples", "10"])

        assert bundle is runner_bundle
