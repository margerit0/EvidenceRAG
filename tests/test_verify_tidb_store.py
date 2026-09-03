from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "verify_tidb_store.py"
    spec = importlib.util.spec_from_file_location("verify_tidb_store_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestLiveGuard:
    def test_parse_requires_explicit_mutation_flag(self) -> None:
        runner = _runner()

        with pytest.raises(SystemExit, match="2"):
            runner._parse_args([])

    def test_guard_rejects_unsafe_suffix_before_loading_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        loaded = False

        def fail_load(_path: Path) -> dict[str, str]:
            nonlocal loaded
            loaded = True
            raise AssertionError("environment must not be loaded")

        monkeypatch.setattr(runner, "load_env", fail_load)
        with pytest.raises(SystemExit, match="2"):
            runner.main(["--allow-live-tidb", "--suffix", "bad-name"])
        assert loaded is False

    def test_main_guard_does_not_connect_without_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        loaded = False

        def fail_load(_path: Path) -> dict[str, str]:
            nonlocal loaded
            loaded = True
            raise AssertionError("environment must not be loaded")

        monkeypatch.setattr(runner, "load_env", fail_load)
        with pytest.raises(SystemExit, match="2"):
            runner.main([])
        assert loaded is False

    def test_explicit_flag_reports_missing_environment_without_connecting(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        connected = False

        def fail_store(_config: object) -> object:
            nonlocal connected
            connected = True
            raise AssertionError("database connection must not be constructed")

        monkeypatch.setattr(runner, "load_env", lambda _path: {})
        monkeypatch.setattr(runner, "TiDBStore", fail_store)
        assert runner.main(["--allow-live-tidb"]) == 1
        assert connected is False


def test_parse_args_returns_namespace_with_opt_in_flag() -> None:
    runner = _runner()
    args = runner._parse_args(["--allow-live-tidb", "--suffix", "ci_01"])
    assert isinstance(args, argparse.Namespace)
    assert args.allow_live_tidb is True
    assert args.suffix == "ci_01"
