from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from test_agent import ABSTAIN, make_agent
from zhrag.eval.agent_tasks import draft_tasks
from zhrag.io_utils import read_json, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, Path]:
    task_script = load_script("agent_tasks")
    monkeypatch.setattr(task_script, "ROOT", tmp_path)
    monkeypatch.setattr(task_script, "TASKS_ROOT", tmp_path / "indexes" / "agent_eval")
    monkeypatch.setitem(sys.modules, "agent_tasks", task_script)
    compare = load_script("compare_agent")
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    path = task_script.TASKS_ROOT / "v1" / "tasks.jsonl"
    write_jsonl(path, draft_tasks())
    return compare, task_script, path


def test_dry_run_never_imports_service_or_providers(
    cli: tuple[ModuleType, ModuleType, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    compare, _, path = cli
    monkeypatch.setitem(sys.modules, "serve", None)
    assert compare.main(["--tasks", str(path), "--allow-drafts"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["provider_calls_enabled"] is False and plan["trial_count"] == 9
    assert plan["selection"]["reviewed_count"] == 0


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--allow-drafts", "--split", "test"],
        ["--allow-drafts", "--limit", "0"],
        ["--allow-drafts", "--methods", "single_rag", "single_rag"],
        ["--allow-drafts", "--run"],
        ["--allow-drafts", "--run", "--run-id", "con"],
    ],
)
def test_invalid_plan_fails_before_service_composition(
    cli: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    compare, _, path = cli
    monkeypatch.setitem(sys.modules, "serve", None)
    assert compare.main(["--tasks", str(path), *options]) == 1


def test_task_initializer_never_overwrites_review_work(
    cli: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, tasks, path = cli
    original = list(read_jsonl(path))
    assert tasks.main(["--initialize", "--tasks", str(path)]) == 1
    assert list(read_jsonl(path)) == original
    assert tasks.main(["--tasks", str(path), "--require-reviewed"]) == 1
    with pytest.raises(ValueError):
        tasks._local_path(tasks.ROOT / "public.jsonl")


def test_private_trials_written_with_pending_review_and_no_overwrite(
    cli: tuple[ModuleType, ModuleType, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compare, _, path = cli
    agent, _ = make_agent(ABSTAIN)
    closed: list[bool] = []
    store = SimpleNamespace(close=lambda: closed.append(True))
    state = SimpleNamespace(collection_name="synthetic", sparse_fingerprint="a" * 64)
    service = SimpleNamespace(
        _parse_args=lambda _argv: SimpleNamespace(
            artifacts=path.parent, uri="synthetic", alias="test"
        ),
        _load_published_artifacts=lambda _path: (state, None),
        MilvusStore=lambda _config: store,
        MilvusConfig=lambda **_kwargs: None,
        DENSE_WIDTH=2,
        _build_retriever=lambda *_args, **_kwargs: (agent.retriever, None),
        _build_agent=lambda *_args, **_kwargs: agent,
    )
    monkeypatch.setitem(sys.modules, "serve", service)
    args = [
        "--tasks",
        str(path),
        "--allow-drafts",
        "--run",
        "--run-id",
        "synthetic-run",
        "--limit",
        "1",
        "--methods",
        "document_agent",
    ]
    assert compare.main(args) == 0
    run = compare.ROOT / "indexes/agent_eval/runs/synthetic-run"
    manifest = read_json(run / "manifest.json")
    assert manifest["complete"] is True and manifest["human_review_required"] is True
    trials = list(read_jsonl(run / "trials.jsonl"))
    assert len(trials) == 1 and trials[0]["review"]["task_success"] is None
    assert not trials[0]["review"]["reviewed"] and closed == [True]
    assert compare.main(args) == 1
    assert len(list(read_jsonl(run / "trials.jsonl"))) == 1
