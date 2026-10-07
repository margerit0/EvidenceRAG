from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from zhrag.eval.crud import Query
from zhrag.io_utils import read_bytes, read_text, write_text


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_ablation_docs.py"
    spec = importlib.util.spec_from_file_location("ablation_sync_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "EXPANDED", tmp_path / "expanded")
    return module


def _documents(root: Path) -> tuple[Path, Path]:
    paths = (root / "README.md", root / "docs" / "evaluation.md")
    for path in paths:
        write_text(
            path,
            "保留原有正文\n<!-- BEGIN ABLATION-SUMMARY -->\n旧结果\n"
            "<!-- END ABLATION-SUMMARY -->\n保留其他同步区块\n",
        )
    return paths


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> object:
    args = ["--repo-root", str(root)]
    if check:
        args.append("--check")
    return runner._parse_args(args)


def test_sync_and_check_preserve_surrounding_text_and_are_idempotent(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _documents(tmp_path)
    monkeypatch.setattr(runner, "build_summary", lambda _paths: "新聚合结果")
    assert runner.synchronize(_args(runner, tmp_path)) == paths
    snapshots = [read_bytes(path) for path in paths]
    for path in paths:
        text = read_text(path)
        assert "保留原有正文" in text and "保留其他同步区块" in text
        assert "新聚合结果" in text and "旧结果" not in text
    assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()
    assert runner.synchronize(_args(runner, tmp_path)) == ()
    assert [read_bytes(path) for path in paths] == snapshots
    assert not list(tmp_path.rglob("*.lock"))


def test_stale_check_does_not_write_either_document(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _documents(tmp_path)
    snapshots = [read_bytes(path) for path in paths]
    monkeypatch.setattr(runner, "build_summary", lambda _paths: "new aggregate")
    with pytest.raises(SystemExit, match="stale"):
        runner.synchronize(_args(runner, tmp_path, check=True))
    assert [read_bytes(path) for path in paths] == snapshots


@pytest.mark.parametrize("failure", [FileNotFoundError("missing cache"), ValueError("drift")])
def test_missing_or_unauthenticated_inputs_leave_both_documents_intact(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    paths = _documents(tmp_path)
    snapshots = [read_bytes(path) for path in paths]

    def fail(_paths: object) -> str:
        raise failure

    monkeypatch.setattr(runner, "build_summary", fail)
    with pytest.raises(type(failure)):
        runner.synchronize(_args(runner, tmp_path))
    assert [read_bytes(path) for path in paths] == snapshots
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


@pytest.mark.parametrize(
    "malformed",
    [
        "no markers",
        "<!-- BEGIN ABLATION-SUMMARY --><!-- BEGIN ABLATION-SUMMARY -->"
        "<!-- END ABLATION-SUMMARY -->",
        "<!-- END ABLATION-SUMMARY --><!-- BEGIN ABLATION-SUMMARY -->",
    ],
)
def test_bad_second_target_is_rejected_before_recomputation_or_first_write(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str
) -> None:
    readme, evaluation = _documents(tmp_path)
    write_text(evaluation, malformed)
    before = read_bytes(readme)

    def unexpected(_paths: object) -> str:
        raise AssertionError("must validate all markers first")

    monkeypatch.setattr(runner, "build_summary", unexpected)
    with pytest.raises(ValueError, match="ABLATION-SUMMARY"):
        runner.synchronize(_args(runner, tmp_path))
    assert read_bytes(readme) == before
    assert read_text(evaluation) == malformed


def test_paired_interval_keeps_direction_and_floor_label(runner: ModuleType) -> None:
    row = runner._row(
        "narrow vs full",
        "R@1",
        [1.0] * 4,
        [0.0] * 4,
        0.0001,
        method="historical one-sided",
        at_floor=True,
    )
    assert "-100.00pp" in row
    assert "[-100.00, -100.00]pp" in row
    assert "0.0001†" in row
    assert "historical one-sided" in row


@pytest.mark.parametrize("extra", [False, True])
def test_crud_summary_rejects_missing_and_extra_rerank_pairs(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, extra: bool
) -> None:
    queries = (Query("q", "合成问题", "原创答案", ("doc",), "questanswer_1doc"),)
    inputs = SimpleNamespace(
        corpus={"doc": "原创合成段落"},
        queries=queries,
        doc_ids=("doc",),
        doc_matrix=np.ones((1, 2), dtype=np.float32),
        query_matrix=np.ones((1, 2), dtype=np.float32),
    )
    monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(runner, "bm25_runs", lambda *args, **kwargs: [["doc"]])
    monkeypatch.setattr(runner, "dense_runs", lambda *args, **kwargs: [["doc"]])
    cached = {("q", "doc"): 1.0, ("unexpected", "doc"): 0.0} if extra else {}

    def offline_cache(_experiment: object, *, score: bool, analyze: bool) -> tuple:
        assert score is False and analyze is True
        return None, "synthetic", cached

    helper = SimpleNamespace(
        Experiment=lambda *args: None,
        _load_cache_for_mode=offline_cache,
        _expected_pairs=lambda _experiment: {("q", "doc")},
    )
    monkeypatch.setattr(runner, "_script", lambda _name: helper)
    with pytest.raises(ValueError, match="incomplete or contains unexpected"):
        runner._crud_rows()


def test_mrl_rejects_changed_query_population(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_script", lambda _name: None)
    inputs = SimpleNamespace(queries=())
    with pytest.raises(ValueError, match="frozen 800"):
        runner._mrl_rows(inputs)
