from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from test_hybrid_mrl1024 import _inputs
from zhrag.embedding_contract import CRUD_EMBEDDING_MODEL, DOCUMENT_PROMPT, QUERY_PROMPT
from zhrag.eval.hybrid_mrl1024 import (
    HYBRID_MRL1024_SCHEMA,
    HybridInputs,
    load_hybrid_mrl1024_inputs,
)
from zhrag.io_utils import read_json, read_text, write_json, write_jsonl, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_h_hybrid_mrl1024.py"
    spec = importlib.util.spec_from_file_location("evaluate_h_hybrid_mrl1024_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _placeholder_inputs(runner: ModuleType, expanded: Path) -> None:
    expanded.mkdir(parents=True)
    for path in runner._required_paths(expanded):
        write_text(path, "placeholder\n")


def _patch_inputs(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    inputs: HybridInputs | None = None,
) -> HybridInputs:
    selected = inputs or _inputs()
    monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", lambda *_args, **_kwargs: selected)
    return selected


def _small_inputs() -> HybridInputs:
    base = _inputs()
    corpus = dict(list(base.corpus.items())[:12])
    unique_gold = {doc_id for query in base.queries for doc_id in query.gold_doc_ids}
    manifest = dict(base.manifest)
    manifest.update(
        {
            "documents": len(corpus),
            "gold_documents": len(unique_gold),
            "distractors": len(corpus) - len(unique_gold),
        }
    )
    return replace(
        base,
        manifest=manifest,
        corpus=corpus,
        doc_ids=tuple(corpus),
        doc_matrix=base.doc_matrix[: len(corpus)].copy(),
    )


def _write_local_inputs(expanded: Path, inputs: HybridInputs, *, drop_query: bool = False) -> None:
    write_json(expanded / "manifest.json", inputs.manifest)
    write_jsonl(
        expanded / "corpus.jsonl",
        ({"doc_id": doc_id, "text": text} for doc_id, text in inputs.corpus.items()),
    )
    write_jsonl(
        expanded / "qrels.jsonl",
        (
            {
                "query_id": query.query_id,
                "question": query.question,
                "answer": query.answer,
                "gold_doc_ids": list(query.gold_doc_ids),
                "task": query.task,
            }
            for query in inputs.queries
        ),
    )
    document_cache = expanded / "emb_cache_4096.jsonl"
    query_cache = expanded / "emb_cache_queries_4096.jsonl"
    write_jsonl(
        document_cache,
        (
            {"doc_id": doc_id, "embedding": row.tolist()}
            for doc_id, row in zip(inputs.doc_ids, inputs.doc_matrix, strict=True)
        ),
    )
    query_pairs = list(zip(inputs.queries, inputs.query_matrix, strict=True))
    if drop_query:
        query_pairs.pop()
    write_jsonl(
        query_cache,
        ({"doc_id": query.query_id, "embedding": row.tolist()} for query, row in query_pairs),
    )
    write_json(
        document_cache.with_suffix(document_cache.suffix + ".meta.json"),
        {"model": CRUD_EMBEDDING_MODEL, "prompt": DOCUMENT_PROMPT},
    )
    write_json(
        query_cache.with_suffix(query_cache.suffix + ".meta.json"),
        {"model": CRUD_EMBEDDING_MODEL, "prompt": QUERY_PROMPT},
    )


class TestArguments:
    def test_cli_has_only_offline_controls(self) -> None:
        runner = _runner()
        parsed = runner._parse_args(["--expanded", "local", "--resamples", "2"])
        assert vars(parsed) == {
            "expanded": Path("local"),
            "resamples": 2,
            "seed": 0,
        }
        for forbidden in ("--embed", "--rerank", "--provider", "--judge"):
            with pytest.raises(SystemExit):
                runner._parse_args([forbidden])

    @pytest.mark.parametrize(
        "argv",
        [["--resamples", "0"], ["--seed", "-1"], ["--seed", str(1 << 63)]],
    )
    def test_rejects_invalid_bootstrap_arguments(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            _runner()._parse_args(argv)


class TestStrictLocalLoading:
    def test_loads_complete_4096_caches_with_chinese_text(self, tmp_path: Path) -> None:
        inputs = _small_inputs()
        expanded = tmp_path / "expanded"
        _write_local_inputs(expanded, inputs)

        loaded = load_hybrid_mrl1024_inputs(expanded, require_frozen=False)

        assert loaded.doc_matrix.shape == inputs.doc_matrix.shape
        assert loaded.query_matrix.shape == inputs.query_matrix.shape
        assert loaded.queries[0].question == inputs.queries[0].question

    def test_rejects_prompt_drift_without_adopting_the_sidecar(self, tmp_path: Path) -> None:
        inputs = _small_inputs()
        expanded = tmp_path / "expanded"
        _write_local_inputs(expanded, inputs)
        sidecar = expanded / "emb_cache_queries_4096.jsonl.meta.json"
        write_json(sidecar, {"model": CRUD_EMBEDDING_MODEL, "prompt": "changed"})

        with pytest.raises(SystemExit, match="different settings"):
            load_hybrid_mrl1024_inputs(expanded, require_frozen=False)

        assert read_json(sidecar)["prompt"] == "changed"

    def test_cache_miss_fails_instead_of_embedding(self, tmp_path: Path) -> None:
        inputs = _small_inputs()
        expanded = tmp_path / "expanded"
        _write_local_inputs(expanded, inputs, drop_query=True)

        with pytest.raises(SystemExit, match="missing 1"):
            load_hybrid_mrl1024_inputs(expanded, require_frozen=False)


class TestOfflinePublication:
    def test_publishes_one_valid_aggregate_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _placeholder_inputs(runner, expanded)
        _patch_inputs(runner, monkeypatch)

        assert runner.main(["--expanded", str(expanded), "--resamples", "30", "--seed", "7"]) == 0

        report = read_json(expanded / runner.REPORT_NAME)
        assert report["schema"] == HYBRID_MRL1024_SCHEMA
        assert report["evaluation_design"]["resamples"] == 30
        assert not (expanded / f"{runner.REPORT_NAME}.tmp").exists()
        assert not (expanded / runner.LOCK_NAME).exists()

    def test_clean_import_path_loads_no_provider_modules(self) -> None:
        root = Path(__file__).resolve().parent.parent
        script = root / "scripts" / "evaluate_h_hybrid_mrl1024.py"
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

    def test_identical_inputs_are_byte_reproducible(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _placeholder_inputs(runner, expanded)
        _patch_inputs(runner, monkeypatch)
        argv = ["--expanded", str(expanded), "--resamples", "30", "--seed", "7"]

        runner.main(argv)
        first = read_text(expanded / runner.REPORT_NAME)
        runner.main(argv)
        assert read_text(expanded / runner.REPORT_NAME) == first

    def test_evaluation_failure_preserves_previous_report_and_cleans_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _placeholder_inputs(runner, expanded)
        target = expanded / runner.REPORT_NAME
        write_text(target, "previous-good-report\n")

        def fail(*_args: object, **_kwargs: object) -> object:
            raise ValueError("injected cache drift")

        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", fail)
        with pytest.raises(SystemExit, match="injected cache drift"):
            runner.main(["--expanded", str(expanded), "--resamples", "10"])

        assert read_text(target) == "previous-good-report\n"
        assert not (expanded / runner.LOCK_NAME).exists()
        assert not (expanded / f"{runner.REPORT_NAME}.tmp").exists()

    def test_lock_contention_blocks_before_reading_without_cleanup(self, tmp_path: Path) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        expanded.mkdir()
        lock = expanded / runner.LOCK_NAME
        write_text(lock, "pid=123\n")

        with pytest.raises(SystemExit, match="another writer"):
            runner.main(["--expanded", str(expanded), "--resamples", "10"])

        assert read_text(lock) == "pid=123\n"
        assert not (expanded / runner.REPORT_NAME).exists()

    def test_holds_the_lock_through_load_evaluate_and_publish(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        _placeholder_inputs(runner, expanded)
        inputs = _inputs()
        observed = {"load": False, "publish": False}

        def guarded_load(*_args: object, **_kwargs: object) -> HybridInputs:
            assert (expanded / runner.LOCK_NAME).is_file()
            observed["load"] = True
            return inputs

        real_publish = runner._publish

        def guarded_publish(args: argparse.Namespace, report: dict[str, object]) -> Path:
            assert (expanded / runner.LOCK_NAME).is_file()
            observed["publish"] = True
            return real_publish(args, report)

        monkeypatch.setattr(runner, "load_hybrid_mrl1024_inputs", guarded_load)
        monkeypatch.setattr(runner, "_publish", guarded_publish)

        assert runner.main(["--expanded", str(expanded), "--resamples", "10"]) == 0
        assert observed == {"load": True, "publish": True}
        assert not (expanded / runner.LOCK_NAME).exists()

    def test_staging_failure_keeps_previous_report_and_cleans_temp(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        expanded = tmp_path / "expanded"
        expanded.mkdir()
        target = expanded / runner.REPORT_NAME
        write_text(target, "old\n")
        args = runner._parse_args(["--expanded", str(expanded), "--resamples", "10"])
        report = runner.evaluate_hybrid_mrl1024(_inputs(), resamples=10)
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            runner._publish(args, report)

        assert read_text(target) == "old\n"
        assert not (expanded / f"{runner.REPORT_NAME}.tmp").exists()
