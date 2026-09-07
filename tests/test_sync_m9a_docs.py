from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zhrag.io_utils import read_text, write_json, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "sync_m9a_docs.py"
    spec = importlib.util.spec_from_file_location("sync_m9a_docs_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evidence() -> dict[str, Any]:
    return {
        "schema": "zhrag-crud-rag-table8-evidence-v1",
        "source": {
            "title": (
                "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented "
                "Generation of Large Language Models"
            ),
            "arxiv_id": "2401.17043",
            "arxiv_version": 3,
            "doi": "10.48550/arXiv.2401.17043",
            "pdf_url": "https://arxiv.org/pdf/2401.17043v3",
            "html_url": "https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8",
            "pdf_sha256": "2e4ae0cb708fdca9d96bcf8d1c0713dae121195a0a31b78e3132e9ef4fa7db8a",
            "pdf_page": 26,
            "table": 8,
        },
        "upstream": {
            "repository": "https://github.com/IAAR-Shanghai/CRUD_RAG",
            "commit": "1aace383994e1f68efa12cf2a8e2dadfb4102ceb",
            "license_file": False,
            "github_license_metadata": None,
            "license_status_date": "2026-08-31",
        },
        "rows": [
            {
                "task": "summarization",
                "model": "Qwen-14B",
                "bleu": 32.51,
                "rouge_l": 33.33,
                "bert_score": 85.62,
                "ragquest_precision": 68.94,
                "ragquest_recall": 40.57,
                "length": 139.1,
            },
            {
                "task": "summarization",
                "model": "GPT-4-0613",
                "bleu": 24.54,
                "rouge_l": 35.91,
                "bert_score": 89.39,
                "ragquest_precision": 71.24,
                "ragquest_recall": 50.53,
                "length": 194.6,
            },
            {
                "task": "question answering 1-document",
                "model": "Qwen-14B",
                "bleu": 37.95,
                "rouge_l": 55.13,
                "bert_score": 83.25,
                "ragquest_precision": 53.03,
                "ragquest_recall": 73.92,
                "length": 73.8,
            },
            {
                "task": "question answering 1-document",
                "model": "GPT-4-0613",
                "bleu": 33.87,
                "rouge_l": 51.42,
                "bert_score": 80.92,
                "ragquest_precision": 53.14,
                "ragquest_recall": 62.39,
                "length": 95.9,
            },
        ],
    }


def _docs(root: Path, *, duplicate_readme_marker: bool = False) -> tuple[Path, Path]:
    evaluation = root / "docs" / "evaluation.md"
    architecture = root / "docs" / "architecture-decision.md"
    marker = "<!-- BEGIN M9A-GENERATION-METRICS -->\nstale\n<!-- END M9A-GENERATION-METRICS -->"
    if duplicate_readme_marker:
        marker = f"{marker}\n{marker}"
    write_text(evaluation, f"before\n{marker}\nafter\n")
    write_text(architecture, f"before\n{marker}\nafter\n")
    return evaluation, architecture


def _args(runner: ModuleType, root: Path, *, check: bool = False) -> object:
    argv = [
        "--evidence",
        str(root / "evidence.json"),
        "--evaluation",
        str(root / "docs" / "evaluation.md"),
        "--architecture",
        str(root / "docs" / "architecture-decision.md"),
        "--repo-root",
        str(root),
    ]
    if check:
        argv.append("--check")
    return runner._parse_args(argv)


class TestEvidenceValidation:
    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("schema",), "wrong", "evidence.schema"),
            (("source", "arxiv_version"), 2, "source.arxiv_version"),
            (("source", "pdf_sha256"), "A" * 64, "pdf_sha256"),
            (("source", "pdf_sha256"), "0" * 64, "audited PDF"),
            (("upstream", "commit"), "0" * 40, "upstream.commit"),
            (("upstream", "license_file"), True, "license_file"),
            (("rows", 0, "bleu"), float("nan"), "bleu"),
            (("rows", 0, "bleu"), 32.50, "audited Table 8 row"),
            (("rows", 0, "length"), -1, "length"),
        ],
    )
    def test_rejects_source_upstream_and_value_drift(
        self,
        path: tuple[str | int, ...],
        value: object,
        message: str,
    ) -> None:
        runner = _runner()
        evidence = _evidence()
        target: Any = evidence
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

        with pytest.raises(ValueError, match=message):
            runner.validate_evidence(evidence)

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value["rows"].pop(),
            lambda value: value["rows"].append(copy.deepcopy(value["rows"][0])),
            lambda value: value["rows"][0].update({"query": "synthetic text"}),
            lambda value: value["rows"][0].update({"answer": "synthetic text"}),
            lambda value: value["rows"][0].update({"extra": "unexpected"}),
            lambda value: value["source"].update({"question": "synthetic text"}),
        ],
    )
    def test_rejects_row_count_duplicates_and_text_leaks(self, mutation: object) -> None:
        runner = _runner()
        evidence = _evidence()
        assert callable(mutation)
        mutation(evidence)

        with pytest.raises(ValueError):
            runner.validate_evidence(evidence)

    def test_load_evidence_fails_closed_for_malformed_json(self, tmp_path: Path) -> None:
        runner = _runner()
        path = tmp_path / "evidence.json"
        write_text(path, "not json")

        with pytest.raises(SystemExit, match="invalid M9a evidence"):
            runner.load_evidence(path)


class TestM9aSynchronization:
    def test_renders_contract_and_aggregate_only_history(self, tmp_path: Path) -> None:
        runner = _runner()
        write_json(tmp_path / "evidence.json", _evidence())
        readme, architecture = _docs(tmp_path)

        assert runner.synchronize(_args(runner, tmp_path)) == (readme, architecture)
        rendered = read_text(readme) + read_text(architecture)
        assert "mean_sentence_bleu4" in rendered
        assert "crud_mean_sentence_bleu4_no_bp" in rendered
        assert "mean_sentence_rouge_l_f1" in rendered
        assert "Table 8" in rendered
        assert "32.51" in rendered
        assert "不是真 BERTScore" in rendered
        assert "不生成答案" in rendered
        assert "**固定合同**：" in rendered
        assert "| **M9a** |" not in rendered
        for forbidden in (
            "synthetic text",
            "SECRET_QUERY",
            "SECRET_ANSWER",
            "provider payload",
            "chunk_id",
            "document_id",
        ):
            assert forbidden not in rendered

    def test_is_idempotent_and_check_detects_stale_docs(self, tmp_path: Path) -> None:
        runner = _runner()
        write_json(tmp_path / "evidence.json", _evidence())
        docs = _docs(tmp_path)
        args = _args(runner, tmp_path)

        assert runner.synchronize(args) == docs
        first = tuple(read_text(path) for path in docs)
        assert runner.synchronize(args) == ()
        assert tuple(read_text(path) for path in docs) == first
        assert runner.synchronize(_args(runner, tmp_path, check=True)) == ()

        write_text(docs[0], read_text(docs[0]).replace("32.51", "32.50"))
        with pytest.raises(SystemExit, match="generated documentation is stale"):
            runner.synchronize(_args(runner, tmp_path, check=True))

    def test_validates_all_markers_before_writing(self, tmp_path: Path) -> None:
        runner = _runner()
        write_json(tmp_path / "evidence.json", _evidence())
        readme, architecture = _docs(tmp_path, duplicate_readme_marker=True)
        before = (read_text(readme), read_text(architecture))

        with pytest.raises(SystemExit, match="must occur exactly once"):
            runner.synchronize(_args(runner, tmp_path))
        assert (read_text(readme), read_text(architecture)) == before

    def test_publication_failure_rolls_back_both_documents(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        write_json(tmp_path / "evidence.json", _evidence())
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

    def test_does_not_read_ignored_or_unrelated_artifacts(self, tmp_path: Path) -> None:
        runner = _runner()
        write_json(tmp_path / "evidence.json", _evidence())
        write_text(tmp_path / "indexes" / "tidb" / "eval" / "query.jsonl", "SECRET_QUERY")
        write_text(tmp_path / "indexes" / "tidb" / "eval" / "answers.jsonl", "SECRET_ANSWER")
        readme, architecture = _docs(tmp_path)

        runner.synchronize(_args(runner, tmp_path))
        serialized = read_text(readme) + read_text(architecture)
        assert "SECRET_QUERY" not in serialized
        assert "SECRET_ANSWER" not in serialized
