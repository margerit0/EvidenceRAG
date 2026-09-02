from __future__ import annotations

import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.eval.crud_generation import (
    CACHE_ROW_SCHEMA,
    GenerationCacheRow,
    build_generation_cases,
    build_input_manifest,
    generation_cache_id,
)
from zhrag.eval.metrics_gen import SemanticProvenance, TokenizerProvenance
from zhrag.io_utils import read_json, read_jsonl, write_json, write_jsonl, write_text

RAW = {
    "event_summary": [
        {
            "ID": "synthetic-summary-1",
            "text": "SOURCE_SUMMARY_ONE 甲 乙 丙 丁",
            "summary": "REFERENCE_SUMMARY_ONE 甲 乙 丙",
            "ignored": "MUST_NOT_ESCAPE",
        },
        {
            "ID": "synthetic-summary-2",
            "text": "SOURCE_SUMMARY_TWO 戊 己 庚 辛",
            "summary": "REFERENCE_SUMMARY_TWO 戊 己 庚",
        },
    ],
    "questanswer_1doc": [
        {
            "ID": "synthetic-qa-1",
            "news1": "SOURCE_QA_ONE 春 夏 秋 冬",
            "questions": "ORIGINAL_QUESTION_ONE 是什么？",
            "answers": "REFERENCE_QA_ONE 春 夏 秋",
        },
        {
            "ID": "synthetic-qa-2",
            "news1": "SOURCE_QA_TWO 风 云 雷 电",
            "questions": "ORIGINAL_QUESTION_TWO 是什么？",
            "answers": "REFERENCE_QA_TWO 风 云 雷",
        },
    ],
}


@dataclass(frozen=True, slots=True)
class FakeReply:
    content: str
    model: str = "synthetic-chat"
    prompt_tokens: int | None = 8
    completion_tokens: int | None = 3
    reasoning_tokens: int | None = 0


class FakeChat:
    def __init__(self, mode: str, *, malformed: bool = False, failing: bool = False) -> None:
        self.mode = mode
        self.malformed = malformed
        self.failing = failing
        self.model = "synthetic-chat"
        self.retries = 7
        self.calls: list[tuple[str, str, bool]] = []

    def _reply(self, content: str) -> FakeReply:
        return FakeReply(content, model=self.model)

    def complete(self, system: str, user: str, *, json_object: bool = True) -> FakeReply:
        self.calls.append((system, user, json_object))
        if self.failing:
            raise RuntimeError("synthetic transport failure")
        if self.malformed:
            return self._reply("not-json")
        if self.mode == "generation":
            if "摘要" in system:
                return self._reply('{"text":"合成摘要 甲 乙 丙"}')
            return self._reply('{"answer":"合成回答 春 夏 秋"}')
        if self.mode == "qg":
            return self._reply('{"questions":["这条内容的核心事实是什么？"]}')
        if self.mode == "qa":
            if "REFERENCE_" in user:
                return self._reply('{"answer":"参考答案 春 夏 秋"}')
            return self._reply('{"answer":"预测答案 春 夏 秋"}')
        raise AssertionError(f"unknown fake mode: {self.mode}")


class SpaceTokenizer:
    @property
    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(name="synthetic-space", package="tests", version="1")

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.split())


class FakeSemanticScorer:
    def __init__(self) -> None:
        self._provenance = SemanticProvenance(
            distribution_version="synthetic-dist",
            module_version="synthetic-module",
            batch_size=2,
        )
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    @property
    def provenance(self) -> SemanticProvenance:
        return self._provenance

    def score(
        self,
        predictions: list[str] | tuple[str, ...],
        references: list[str] | tuple[str, ...],
    ) -> tuple[list[float], list[float], list[float]]:
        self.calls.append((tuple(predictions), tuple(references)))
        count = len(predictions)
        return ([0.6] * count, [0.7] * count, [0.65] * count)


def _runner(root: Path | None = None) -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_crud_generation.py"
    spec = importlib.util.spec_from_file_location("run_crud_generation_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if root is not None:
        module.__dict__["ROOT"] = root
    return module


def _write_input(tmp_path: Path, raw: object = RAW) -> Path:
    path = tmp_path / "synthetic-input.json"
    write_json(path, raw)
    return path


def _base(input_path: Path, artifacts: Path, env: Path) -> list[str]:
    return [
        "--input",
        str(input_path),
        "--artifacts",
        str(artifacts),
        "--env",
        str(env),
    ]


def _profile(runner: ModuleType, client: object, *, model: str) -> object:
    endpoint = "https://synthetic.example/v1/chat/completions"
    effort = "high"
    retries = 7
    if isinstance(client, FakeChat):
        client.model = model
    fingerprint = runner._profile_fingerprint(
        model=model,
        endpoint=endpoint,
        reasoning_effort=effort,
        retries=retries,
        json_object=True,
    )
    return runner.ChatProfile(
        client=client,
        model=model,
        endpoint=endpoint,
        reasoning_effort=effort,
        retries=retries,
        profile_sha256=fingerprint,
    )


def _install_chat_profiles(
    monkeypatch: pytest.MonkeyPatch,
    runner: ModuleType,
    *,
    generation: FakeChat,
    qg: FakeChat,
    qa: FakeChat,
) -> dict[str, object]:
    profiles = {
        "generation": _profile(runner, generation, model="synthetic-generator"),
        "qg": _profile(runner, qg, model="synthetic-qg"),
        "qa": _profile(runner, qa, model="synthetic-judge"),
    }
    monkeypatch.setattr(
        runner,
        "_load_chat_profile",
        lambda _args, stage: profiles[stage],
    )
    return profiles


def _generation_args(
    input_path: Path,
    artifacts: Path,
    env: Path,
    *,
    run_id: str = "trial-a",
    max_calls: int | None = None,
    canonical: bool = False,
) -> list[str]:
    args = [
        *_base(input_path, artifacts, env),
        "--generate",
        "--run-id",
        run_id,
        "--allow-paid-provider",
    ]
    if max_calls is not None:
        args += ["--max-calls", str(max_calls)]
    if canonical:
        args.append("--canonical")
    return args


def _build_question_bank(
    runner: ModuleType,
    input_path: Path,
    artifacts: Path,
    env: Path,
) -> None:
    assert (
        runner.main(
            [
                *_base(input_path, artifacts, env),
                "--generate-questions",
                "--reference-profile",
                "ref-a",
                "--allow-paid-provider",
            ]
        )
        == 0
    )
    assert (
        runner.main(
            [
                *_base(input_path, artifacts, env),
                "--finalize-questions",
                "--reference-profile",
                "ref-a",
            ]
        )
        == 0
    )


def _build_reference_bank(
    runner: ModuleType,
    input_path: Path,
    artifacts: Path,
    env: Path,
) -> None:
    _build_question_bank(runner, input_path, artifacts, env)
    assert (
        runner.main(
            [
                *_base(input_path, artifacts, env),
                "--answer-reference",
                "--reference-profile",
                "ref-a",
                "--allow-paid-provider",
            ]
        )
        == 0
    )
    assert (
        runner.main(
            [
                *_base(input_path, artifacts, env),
                "--finalize-reference-bank",
                "--reference-profile",
                "ref-a",
            ]
        )
        == 0
    )


def _replace_first_served_model(path: Path, value: str) -> None:
    rows = list(read_jsonl(path))
    rows[0]["served_model"] = value
    write_jsonl(path, rows)


class TestStatusAndGuards:
    def test_default_status_is_read_only_and_does_not_read_env_or_create_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        missing_env = tmp_path / "does-not-exist.env"
        imported: list[str] = []
        real_import = runner.importlib.import_module

        def record_import(name: str) -> object:
            imported.append(name)
            return real_import(name)

        monkeypatch.setattr(runner.importlib, "import_module", record_import)

        assert runner.main([*_base(input_path, artifacts, missing_env), "--dry-run"]) == 0
        output = capsys.readouterr().out
        assert "SOURCE_" not in output
        assert "REFERENCE_" not in output
        assert "ORIGINAL_" not in output
        assert "dry-run" in output
        assert not artifacts.exists()
        assert "zhrag.providers.chat" not in imported
        assert "zhrag.providers.embedding" not in imported
        assert "bert_score" not in imported
        assert "jieba" not in imported

    @pytest.mark.parametrize(
        ("action", "message"),
        [
            ("--generate", "allow-paid-provider"),
            ("--generate-questions", "allow-paid-provider"),
            ("--answer-reference", "allow-paid-provider"),
            ("--answer-prediction", "allow-paid-provider"),
            ("--score-semantic", "allow-model-download"),
        ],
    )
    def test_paid_and_model_actions_fail_before_input_or_files_without_guards(
        self,
        action: str,
        message: str,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        missing_input = tmp_path / "missing-input.json"
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        argv = [*_base(missing_input, artifacts, tmp_path / "missing.env"), action]
        if action in {"--generate", "--generate-questions", "--answer-reference"}:
            argv += (
                ["--reference-profile", "ref-a"]
                if action != "--generate"
                else ["--run-id", "run-a"]
            )
        elif action == "--answer-prediction":
            argv += ["--run-id", "run-a", "--reference-profile", "ref-a"]
        else:
            argv += ["--run-id", "run-a"]

        with pytest.raises(SystemExit, match=message):
            runner.main(argv)
        assert not artifacts.exists()

    def test_finalize_is_offline_but_requires_explicit_profile_selectors(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        with pytest.raises(SystemExit, match="requires --reference-profile"):
            runner.main(
                [
                    *_base(input_path, artifacts, tmp_path / "missing.env"),
                    "--finalize",
                    "--run-id",
                    "run-a",
                ]
            )
        assert not artifacts.exists()


class TestLifecycle:
    def test_synthetic_dag_publishes_only_text_free_final_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        generation = FakeChat("generation")
        qg = FakeChat("qg")
        qa = FakeChat("qa")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=generation,
            qg=qg,
            qa=qa,
        )
        scorer = FakeSemanticScorer()
        monkeypatch.setattr(runner, "_load_semantic_adapter", lambda: scorer)
        monkeypatch.setattr(runner, "_new_tokenizer", SpaceTokenizer)

        assert runner.main(_generation_args(input_path, artifacts, env)) == 0
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--generate-questions",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--finalize-questions",
                    "--reference-profile",
                    "ref-a",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-reference",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--finalize-reference-bank",
                    "--reference-profile",
                    "ref-a",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-prediction",
                    "--run-id",
                    "trial-a",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--score-semantic",
                    "--run-id",
                    "trial-a",
                    "--allow-model-download",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--finalize",
                    "--run-id",
                    "trial-a",
                    "--reference-profile",
                    "ref-a",
                    "--resamples",
                    "20",
                ]
            )
            == 0
        )

        reference_root = artifacts / "reference_banks" / "ref-a"
        run_root = artifacts / "runs" / "trials" / "trial-a"
        assert read_json(reference_root / "question_bank.json")["questions"] == 4
        assert read_json(reference_root / "reference_bank.json")["questions"] == 4
        samples = read_json(run_root / "numeric_samples.json")
        report = read_json(run_root / "report.json")
        encoded = json.dumps({"samples": samples, "report": report}, ensure_ascii=False)
        for secret in (
            "SOURCE_",
            "REFERENCE_",
            "ORIGINAL_",
            "MUST_NOT_ESCAPE",
            "合成摘要",
            "参考答案",
            "预测答案",
        ):
            assert secret not in encoded
        assert report["design"]["profile"] == "known-context generation; retrieval excluded"
        assert len(scorer.calls) == 2
        assert len(generation.calls) == 4
        assert len(qg.calls) == 4
        assert len(qa.calls) == 8
        assert all(call[2] is True for call in generation.calls + qg.calls + qa.calls)

        for _system, user, _json_object in generation.calls:
            assert "REFERENCE_" not in user
        for _system, user, _json_object in qg.calls:
            assert "SOURCE_" not in user
            assert "ORIGINAL_" not in user
        for _system, user, _json_object in qa.calls[:4]:
            assert "SOURCE_" not in user
            assert "预测" not in user
        for _system, user, _json_object in qa.calls[4:]:
            assert "REFERENCE_" not in user

    def test_max_calls_pauses_and_resume_does_not_repeat_cached_cases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        first = _generation_args(input_path, artifacts, tmp_path / "missing.env", max_calls=1)
        assert runner.main(first) == 0
        root = artifacts / "runs" / "trials" / "trial-a"
        assert len(tuple(read_jsonl(root / runner.GENERATION_CACHE))) == 1
        assert read_json(root / f"{runner.GENERATION_CACHE}.meta.json")["complete"] is False

        assert runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env")) == 0
        rows = tuple(read_jsonl(root / runner.GENERATION_CACHE))
        assert len(rows) == 4
        assert read_json(root / f"{runner.GENERATION_CACHE}.meta.json")["complete"] is True
        assert len(chat.calls) == 4

    def test_malformed_provider_response_consumes_call_but_appends_zero_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        bad = FakeChat("generation", malformed=True)
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=bad,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )

        with pytest.raises(SystemExit, match="no cache row appended"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        root = artifacts / "runs" / "trials" / "trial-a"
        cache = root / runner.GENERATION_CACHE
        assert not cache.exists() or cache.stat().st_size == 0
        assert read_json(root / f"{runner.GENERATION_CACHE}.meta.json")["complete"] is False
        assert len(bad.calls) == 1

    def test_provider_failure_consumes_call_but_appends_zero_rows(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        failing = FakeChat("generation", failing=True)
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=failing,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )

        with pytest.raises(SystemExit, match="provider call failed"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        root = artifacts / "runs" / "trials" / "trial-a"
        assert not (root / runner.GENERATION_CACHE).exists()
        assert len(failing.calls) == 1


class TestArtifactBoundaries:
    def test_orphan_nonempty_cache_is_not_adopted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        root = artifacts / "runs" / "trials" / "trial-a"
        write_text(root / runner.GENERATION_CACHE, '{"unexpected":"row"}\n')
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )

        with pytest.raises(SystemExit, match="without input manifest"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        assert not (root / runner.INPUT_MANIFEST).exists()
        assert chat.calls == []

    def test_sidecar_drift_is_rejected_before_a_resume_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert (
            runner.main(
                _generation_args(input_path, artifacts, tmp_path / "missing.env", max_calls=0)
            )
            == 0
        )
        meta_path = (
            artifacts / "runs" / "trials" / "trial-a" / f"{runner.GENERATION_CACHE}.meta.json"
        )
        meta = read_json(meta_path)
        meta["prompt_fingerprint"] = "f" * 64
        write_json(meta_path, meta)
        before = len(chat.calls)
        with pytest.raises(SystemExit, match="metadata drift"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        assert len(chat.calls) == before

    def test_trial_and_canonical_runs_have_independent_manifests_and_caches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert (
            runner.main(
                _generation_args(input_path, artifacts, tmp_path / "missing.env", max_calls=0)
            )
            == 0
        )
        assert (
            runner.main(
                _generation_args(
                    input_path,
                    artifacts,
                    tmp_path / "missing.env",
                    max_calls=0,
                    canonical=True,
                )
            )
            == 0
        )
        trial = artifacts / "runs" / "trials" / "trial-a"
        canonical = artifacts / "runs" / "canonical" / "trial-a"
        assert (trial / runner.INPUT_MANIFEST).exists()
        assert (canonical / runner.INPUT_MANIFEST).exists()
        assert trial != canonical
        assert read_json(trial / runner.INPUT_MANIFEST) == read_json(
            canonical / runner.INPUT_MANIFEST
        )
        assert not (trial / runner.GENERATION_CACHE).exists()
        assert not (canonical / runner.GENERATION_CACHE).exists()

    def test_completion_markers_are_immutable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        qg = FakeChat("qg")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=qg,
            qa=FakeChat("qa"),
        )
        args = [
            *_base(input_path, artifacts, tmp_path / "missing.env"),
            "--generate-questions",
            "--reference-profile",
            "ref-a",
            "--allow-paid-provider",
        ]
        assert runner.main(args) == 0
        finalize_args = [
            *_base(input_path, artifacts, tmp_path / "missing.env"),
            "--finalize-questions",
            "--reference-profile",
            "ref-a",
        ]
        assert runner.main(finalize_args) == 0
        marker = artifacts / "reference_banks" / "ref-a" / runner.QUESTION_BANK
        before = read_json(marker)
        assert runner.main(finalize_args) == 0
        assert read_json(marker) == before
        tampered = copy.deepcopy(before)
        tampered["questions"] += 1
        write_json(marker, tampered)
        with pytest.raises(SystemExit, match="marker"):
            runner.main(finalize_args)

    def test_status_marks_tampered_complete_cache_invalid_without_writing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env")) == 0
        root = artifacts / "runs" / "trials" / "trial-a"
        cache = root / runner.GENERATION_CACHE
        write_text(cache, read_jsonl(cache).__iter__().__next__().__repr__())
        # The malformed append above must not be repaired or adopted by status.
        before = read_json(root / f"{runner.GENERATION_CACHE}.meta.json")
        runner.main([*_base(input_path, artifacts, tmp_path / "missing.env"), "--status"])
        output = capsys.readouterr().out
        assert "generation: invalid" in output
        assert read_json(root / f"{runner.GENERATION_CACHE}.meta.json") == before


class TestHardening:
    @pytest.mark.parametrize(
        "slug",
        ["a", "trial-a", "trial_1.v2", "x" * 64],
    )
    def test_accepts_only_canonical_lowercase_slugs(self, slug: str) -> None:
        runner = _runner()
        assert runner._slug(slug, label="slug") == slug

    @pytest.mark.parametrize(
        "slug",
        ["", "Trial-a", "trial-a.", "con", "aux.txt", "com1", "a/b", "x" * 65],
    )
    def test_rejects_windows_aliases_and_noncanonical_slugs(self, slug: str) -> None:
        runner = _runner()
        with pytest.raises(SystemExit, match="lowercase"):
            runner._slug(slug, label="slug")

    def test_rejects_external_artifact_root_before_reading_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)

        def fail_input(_path: Path) -> object:
            raise AssertionError("input must not be read for an invalid artifact root")

        monkeypatch.setattr(runner, "_load_input", fail_input)
        with pytest.raises(SystemExit, match="project generation root"):
            runner.main(
                [
                    *_base(
                        tmp_path / "missing-input.json",
                        tmp_path / "external-artifacts",
                        tmp_path / "missing.env",
                    ),
                    "--status",
                ]
            )

    def test_rejects_artifact_root_junction_before_reading_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        artifacts.mkdir(parents=True)
        path_type = type(artifacts)
        real_is_junction = path_type.is_junction

        def fake_is_junction(path: Path) -> bool:
            return path == artifacts or real_is_junction(path)

        def fail_input(_path: Path) -> object:
            raise AssertionError("input must not be read through a redirected artifact root")

        monkeypatch.setattr(path_type, "is_junction", fake_is_junction)
        monkeypatch.setattr(runner, "_load_input", fail_input)
        with pytest.raises(SystemExit, match="symlink or junction"):
            runner.main(
                [
                    *_base(
                        tmp_path / "missing-input.json",
                        artifacts,
                        tmp_path / "missing.env",
                    ),
                    "--status",
                ]
            )

    def test_status_rejects_a_redirected_discovered_run_before_inspection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        redirected = artifacts / "runs" / "trials" / "trial-a"
        redirected.mkdir(parents=True)
        path_type = type(redirected)
        real_is_symlink = path_type.is_symlink

        def fake_is_symlink(path: Path) -> bool:
            return path == redirected or real_is_symlink(path)

        def fail_status_root(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("redirected run content must not be inspected")

        monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)
        monkeypatch.setattr(runner, "_status_root", fail_status_root)
        with pytest.raises(SystemExit, match="symlink or junction"):
            runner.main([*_base(input_path, artifacts, tmp_path / "missing.env"), "--status"])

    @pytest.mark.parametrize(
        ("action", "selector", "relative"),
        [
            ("--generate", ("--run-id", "trial-a"), ("runs", "trials", "trial-a")),
            (
                "--generate-questions",
                ("--reference-profile", "ref-a"),
                ("reference_banks", "ref-a"),
            ),
        ],
    )
    def test_paid_actions_reject_redirected_selected_roots_before_provider_load(
        self,
        action: str,
        selector: tuple[str, str],
        relative: tuple[str, ...],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        redirected = artifacts.joinpath(*relative)
        redirected.mkdir(parents=True)
        path_type = type(redirected)
        real_is_junction = path_type.is_junction

        def fake_is_junction(path: Path) -> bool:
            return path == redirected or real_is_junction(path)

        def fail_profile(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("provider profile must not load through a redirected root")

        monkeypatch.setattr(path_type, "is_junction", fake_is_junction)
        monkeypatch.setattr(runner, "_load_chat_profile", fail_profile)
        with pytest.raises(SystemExit, match="symlink or junction"):
            runner.main(
                [
                    *_base(input_path, artifacts, tmp_path / "missing.env"),
                    action,
                    *selector,
                    "--allow-paid-provider",
                ]
            )

    def test_status_rejects_a_redirected_cache_file_before_inspection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        root = artifacts / "runs" / "trials" / "trial-a"
        cache = root / runner.GENERATION_CACHE
        write_text(cache, "")
        path_type = type(cache)
        real_is_symlink = path_type.is_symlink

        def fake_is_symlink(path: Path) -> bool:
            return path == cache or real_is_symlink(path)

        def fail_status_root(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("redirected cache content must not be inspected")

        monkeypatch.setattr(path_type, "is_symlink", fake_is_symlink)
        monkeypatch.setattr(runner, "_status_root", fail_status_root)
        with pytest.raises(SystemExit, match="artifact tree contains"):
            runner.main([*_base(input_path, artifacts, tmp_path / "missing.env"), "--status"])

    def test_paid_action_rejects_a_redirected_cache_file_before_provider_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        root = artifacts / "runs" / "trials" / "trial-a"
        cache = root / runner.GENERATION_CACHE
        write_text(cache, "")
        path_type = type(cache)
        real_is_junction = path_type.is_junction

        def fake_is_junction(path: Path) -> bool:
            return path == cache or real_is_junction(path)

        def fail_profile(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("provider profile must not load through a redirected cache")

        monkeypatch.setattr(path_type, "is_junction", fake_is_junction)
        monkeypatch.setattr(runner, "_load_chat_profile", fail_profile)
        with pytest.raises(SystemExit, match="artifact tree contains"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))

    def test_reference_qa_rejects_served_model_drift_before_resume_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        qa = FakeChat("qa")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=FakeChat("generation"),
            qg=FakeChat("qg"),
            qa=qa,
        )
        _build_question_bank(runner, input_path, artifacts, env)
        args = [
            *_base(input_path, artifacts, env),
            "--answer-reference",
            "--reference-profile",
            "ref-a",
            "--allow-paid-provider",
            "--max-calls",
            "1",
        ]
        assert runner.main(args) == 0
        cache = artifacts / "reference_banks" / "ref-a" / runner.REFERENCE_QA_CACHE
        _replace_first_served_model(cache, "unexpected-served-model")
        calls_before = len(qa.calls)

        with pytest.raises(SystemExit, match="served model drift"):
            runner.main(args)
        assert len(qa.calls) == calls_before

    def test_prediction_qa_rejects_served_model_drift_before_resume_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        qa = FakeChat("qa")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=FakeChat("generation"),
            qg=FakeChat("qg"),
            qa=qa,
        )
        assert runner.main(_generation_args(input_path, artifacts, env)) == 0
        _build_reference_bank(runner, input_path, artifacts, env)
        args = [
            *_base(input_path, artifacts, env),
            "--answer-prediction",
            "--run-id",
            "trial-a",
            "--reference-profile",
            "ref-a",
            "--allow-paid-provider",
            "--max-calls",
            "1",
        ]
        assert runner.main(args) == 0
        cache = artifacts / "runs" / "trials" / "trial-a" / runner.PREDICTION_QA_CACHE
        _replace_first_served_model(cache, "unexpected-served-model")
        calls_before = len(qa.calls)

        with pytest.raises(SystemExit, match="served model drift"):
            runner.main(args)
        assert len(qa.calls) == calls_before

    def test_reference_qa_rejects_served_model_drift_before_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        qa = FakeChat("qa")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=FakeChat("generation"),
            qg=FakeChat("qg"),
            qa=qa,
        )
        _build_question_bank(runner, input_path, artifacts, env)
        real_append = runner._append_row
        corrupted = False

        def append_then_corrupt(path: Path, row: object) -> None:
            nonlocal corrupted
            real_append(path, row)
            if path.name == runner.REFERENCE_QA_CACHE and not corrupted:
                _replace_first_served_model(path, "unexpected-served-model")
                corrupted = True

        monkeypatch.setattr(runner, "_append_row", append_then_corrupt)
        with pytest.raises(SystemExit, match="served model drift"):
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-reference",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )

        root = artifacts / "reference_banks" / "ref-a"
        meta = read_json(root / f"{runner.REFERENCE_QA_CACHE}.meta.json")
        assert meta["complete"] is False
        assert len(qa.calls) == 4

    def test_prediction_qa_rejects_served_model_drift_before_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        qa = FakeChat("qa")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=FakeChat("generation"),
            qg=FakeChat("qg"),
            qa=qa,
        )
        assert runner.main(_generation_args(input_path, artifacts, env)) == 0
        _build_reference_bank(runner, input_path, artifacts, env)
        real_append = runner._append_row
        corrupted = False

        def append_then_corrupt(path: Path, row: object) -> None:
            nonlocal corrupted
            real_append(path, row)
            if path.name == runner.PREDICTION_QA_CACHE and not corrupted:
                _replace_first_served_model(path, "unexpected-served-model")
                corrupted = True

        monkeypatch.setattr(runner, "_append_row", append_then_corrupt)
        with pytest.raises(SystemExit, match="served model drift"):
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-prediction",
                    "--run-id",
                    "trial-a",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )

        root = artifacts / "runs" / "trials" / "trial-a"
        meta = read_json(root / f"{runner.PREDICTION_QA_CACHE}.meta.json")
        assert meta["complete"] is False
        assert len(qa.calls) == 8

    def test_legacy_chat_profile_sidecar_is_rejected_before_resume_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        profiles = _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert (
            runner.main(
                _generation_args(
                    input_path,
                    artifacts,
                    tmp_path / "missing.env",
                    max_calls=0,
                )
            )
            == 0
        )
        meta_path = (
            artifacts / "runs" / "trials" / "trial-a" / f"{runner.GENERATION_CACHE}.meta.json"
        )
        profile = profiles["generation"]
        legacy_fingerprint = runner.canonical_fingerprint(
            "zhrag-crud-chat-profile-v1",
            profile.model,
            profile.endpoint,
            profile.reasoning_effort,
            str(profile.retries),
            "True",
        )
        meta = read_json(meta_path)
        meta["model_profile_sha256"] = legacy_fingerprint
        write_json(meta_path, meta)

        with pytest.raises(SystemExit, match="profile drift"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        assert chat.calls == []

    def test_chat_sidecar_profile_fields_are_authenticated_before_resume(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert (
            runner.main(
                _generation_args(
                    input_path,
                    artifacts,
                    tmp_path / "missing.env",
                    max_calls=0,
                )
            )
            == 0
        )
        meta_path = (
            artifacts / "runs" / "trials" / "trial-a" / f"{runner.GENERATION_CACHE}.meta.json"
        )
        meta = read_json(meta_path)
        meta["model"] = "tampered-model"
        write_json(meta_path, meta)
        with pytest.raises(SystemExit, match="profile drift"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        assert chat.calls == []

    def test_prediction_qa_rejects_a_different_profile_before_paid_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        generation = FakeChat("generation")
        qg = FakeChat("qg")
        reference_qa = FakeChat("qa")
        profiles = _install_chat_profiles(
            monkeypatch,
            runner,
            generation=generation,
            qg=qg,
            qa=reference_qa,
        )
        assert runner.main(_generation_args(input_path, artifacts, env)) == 0
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--generate-questions",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--finalize-questions",
                    "--reference-profile",
                    "ref-a",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-reference",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
            == 0
        )
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--finalize-reference-bank",
                    "--reference-profile",
                    "ref-a",
                ]
            )
            == 0
        )

        prediction_qa = FakeChat("qa")
        mismatched = _profile(
            runner,
            prediction_qa,
            model="synthetic-other-judge",
        )
        monkeypatch.setattr(
            runner,
            "_load_chat_profile",
            lambda _args, stage: mismatched if stage == "qa" else profiles[stage],
        )
        with pytest.raises(SystemExit, match="profile differs"):
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--answer-prediction",
                    "--run-id",
                    "trial-a",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                ]
            )
        assert prediction_qa.calls == []
        run_root = artifacts / "runs" / "trials" / "trial-a"
        assert not (run_root / runner.PREDICTION_QA_CACHE).exists()
        assert not (run_root / f"{runner.PREDICTION_QA_CACHE}.meta.json").exists()

    def test_semantic_metadata_drift_does_not_load_the_scorer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        generation = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=generation,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        scorer = FakeSemanticScorer()
        monkeypatch.setattr(runner, "_load_semantic_adapter", lambda: scorer)
        assert runner.main(_generation_args(input_path, artifacts, env)) == 0
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--score-semantic",
                    "--run-id",
                    "trial-a",
                    "--allow-model-download",
                ]
            )
            == 0
        )
        root = artifacts / "runs" / "trials" / "trial-a"
        meta_path = root / f"{runner.SEMANTIC_CACHE}.meta.json"
        meta = read_json(meta_path)
        meta["complete"] = False
        meta["cache_sha256"] = None
        meta["prompt_fingerprint"] = "f" * 64
        write_json(meta_path, meta)
        calls_before = len(scorer.calls)

        def fail_if_loaded() -> object:
            raise AssertionError("semantic scorer must not load after metadata drift")

        monkeypatch.setattr(runner, "_load_semantic_adapter", fail_if_loaded)
        with pytest.raises(SystemExit, match="metadata drift"):
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--score-semantic",
                    "--run-id",
                    "trial-a",
                    "--allow-model-download",
                ]
            )
        assert len(scorer.calls) == calls_before

    def test_initial_manifest_and_sidecar_publication_clean_up_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        state = runner._load_input(input_path)
        root = tmp_path / "run-root"
        root.mkdir()

        real_replace = runner.replace_files

        def fail_replace(_pairs: object) -> None:
            raise OSError("synthetic publication failure")

        monkeypatch.setattr(runner, "replace_files", fail_replace)
        with pytest.raises(OSError, match="publication failure"):
            runner._ensure_manifest(root, state)
        assert not (root / runner.INPUT_MANIFEST).exists()
        assert not list(root.glob("*.initial.tmp"))

        monkeypatch.setattr(runner, "replace_files", real_replace)
        runner._ensure_manifest(root, state)
        profile = _profile(runner, FakeChat("generation"), model="synthetic-generator")
        expected = runner._generation_expected(state, profile)
        definition = runner._cache_definition("generation")
        monkeypatch.setattr(runner, "replace_files", fail_replace)
        with pytest.raises(OSError, match="publication failure"):
            runner._prepare_cache(root, definition, expected)
        assert not (root / runner.GENERATION_CACHE).exists()
        assert not (root / f"{runner.GENERATION_CACHE}.meta.json").exists()
        assert not list(root.glob("*.initial.tmp"))

    def test_status_reports_only_stages_owned_by_each_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        env = tmp_path / "missing.env"
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=FakeChat("generation"),
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        assert runner.main(_generation_args(input_path, artifacts, env, max_calls=0)) == 0
        assert (
            runner.main(
                [
                    *_base(input_path, artifacts, env),
                    "--generate-questions",
                    "--reference-profile",
                    "ref-a",
                    "--allow-paid-provider",
                    "--max-calls",
                    "0",
                ]
            )
            == 0
        )
        assert runner.main([*_base(input_path, artifacts, env), "--status"]) == 0
        output = capsys.readouterr().out
        reference_start = output.index("reference bank:")
        run_start = output.index("\nrun:", reference_start)
        reference_output = output[reference_start:run_start]
        run_output = output[run_start:]
        assert "QG:" in reference_output
        assert "generation:" not in reference_output
        assert "generation:" in run_output
        assert "QG:" not in run_output

    def test_provider_model_mismatch_is_not_cached(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runner = _runner(tmp_path)
        input_path = _write_input(tmp_path)
        artifacts = tmp_path / "indexes" / "crud" / "generation" / "v1"
        chat = FakeChat("generation")
        _install_chat_profiles(
            monkeypatch,
            runner,
            generation=chat,
            qg=FakeChat("qg"),
            qa=FakeChat("qa"),
        )
        chat.model = "unexpected-provider-model"
        with pytest.raises(SystemExit, match="unexpected model"):
            runner.main(_generation_args(input_path, artifacts, tmp_path / "missing.env"))
        root = artifacts / "runs" / "trials" / "trial-a"
        cache = root / runner.GENERATION_CACHE
        assert not cache.exists() or cache.stat().st_size == 0
        assert len(chat.calls) == 1


class TestPureHelpers:
    def test_manifest_and_cache_identity_helpers_remain_provider_free(self) -> None:
        runner = _runner()
        cases = build_generation_cases(RAW)
        manifest = build_input_manifest(cases, dataset_sha256="a" * 64)
        assert manifest.case_count == 4
        row = GenerationCacheRow(
            schema=CACHE_ROW_SCHEMA,
            cache_id=generation_cache_id(cases[0], "b" * 64),
            case_key=cases[0].case_key,
            task=cases[0].task,
            prediction="合成预测",
            served_model="synthetic",
        )
        assert runner._cache_fingerprint((row,))
