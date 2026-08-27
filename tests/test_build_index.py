from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.io_utils import read_json, read_text, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StoreStub:
    def __init__(self, shared_lock: Path) -> None:
        self.shared_lock = shared_lock
        self.aliases: list[str] = []

    def activate_alias(self, alias: str) -> None:
        assert self.shared_lock.is_file()
        self.aliases.append(alias)


class TestIndexPublication:
    def test_publishes_sparse_state_marker_under_the_shared_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        shared_lock = tmp_path / "eval" / runner.ARTIFACT_LOCK
        store = _StoreStub(shared_lock)
        args = argparse.Namespace(artifacts=tmp_path, alias="tidb_chunks")
        sparse = runner.build_sparse_index({"doc": "abc"})
        targets: list[str] = []
        real_replace = runner.replace_files

        def replace_while_locked(staged: tuple[tuple[Path, Path], ...]) -> None:
            assert shared_lock.is_file()
            pairs = tuple(staged)
            targets.extend(target.name for _source, target in pairs)
            real_replace(pairs)

        monkeypatch.setattr(runner, "replace_files", replace_while_locked)

        runner._publish(
            store,
            args,
            planned=(),
            sparse=sparse,
            scope=runner.Scope.evergreen(),
            collection_name="tidb_chunks_v2",
        )

        assert store.aliases == ["tidb_chunks"]
        assert targets == ["sparse_index.json", "state.json"]
        assert read_json(tmp_path / "sparse_index.json")["fingerprint"] == sparse.index.fingerprint
        assert read_json(tmp_path / "state.json")["collection_name"] == "tidb_chunks_v2"
        assert not shared_lock.exists()
        assert not (tmp_path / "sparse_index.json.tmp").exists()
        assert not (tmp_path / "state.json.tmp").exists()

    def test_lock_contention_leaves_alias_and_previous_bundle_untouched(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        shared_lock = tmp_path / "eval" / runner.ARTIFACT_LOCK
        write_text(shared_lock, "pid=123\n")
        write_text(tmp_path / "sparse_index.json", "old sparse\n")
        write_text(tmp_path / "state.json", "old state\n")
        store = _StoreStub(shared_lock)
        args = argparse.Namespace(artifacts=tmp_path, alias="tidb_chunks")

        with pytest.raises(SystemExit, match="another writer"):
            runner._publish(
                store,
                args,
                planned=(),
                sparse=runner.build_sparse_index({"doc": "abc"}),
                scope=runner.Scope.evergreen(),
                collection_name="tidb_chunks_v2",
            )

        assert store.aliases == []
        assert read_text(shared_lock) == "pid=123\n"
        assert read_text(tmp_path / "sparse_index.json") == "old sparse\n"
        assert read_text(tmp_path / "state.json") == "old state\n"
        assert not (tmp_path / "sparse_index.json.tmp").exists()
        assert not (tmp_path / "state.json.tmp").exists()
