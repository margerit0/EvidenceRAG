from __future__ import annotations

import copy
import json
from collections.abc import Sequence

import pytest

from zhrag.eval.crud_generation import (
    CACHE_ROW_SCHEMA,
    GENERATION_REPORT_SCHEMA,
    GENERATION_SAMPLES_SCHEMA,
    MAX_INPUT_ESTIMATED_TOKENS,
    PUBLIC_INPUT_FINGERPRINT_KEYS,
    GeneratedQuestion,
    GenerationCacheRow,
    QACacheRow,
    QGCacheRow,
    SemanticCacheRow,
    build_generation_cases,
    build_generation_report,
    build_input_manifest,
    build_numeric_samples,
    build_quest_answer_pairs,
    build_question_bank,
    build_reference_bank,
    canonical_fingerprint,
    generation_cache_id,
    generation_user_prompt,
    parse_generation_cache_row,
    parse_generation_input_manifest,
    parse_generation_response,
    parse_qa_cache_row,
    parse_qa_response,
    parse_qg_cache_row,
    parse_question_bank,
    parse_questions_response,
    parse_reference_bank,
    parse_semantic_cache_row,
    prediction_qa_cache_id,
    qa_user_prompt,
    qg_cache_id,
    qg_user_prompt,
    question_id,
    reference_qa_cache_id,
    semantic_cache_id,
    validate_generation_report,
    validate_generation_rows,
    validate_numeric_samples,
    validate_prediction_qa_rows,
    validate_prompt_budget,
    validate_qg_rows,
    validate_reference_qa_rows,
    validate_semantic_rows,
)
from zhrag.eval.metrics_gen import SemanticProvenance, TokenizerProvenance


class SpaceTokenizer:
    @property
    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(name="synthetic-space", package="tests", version="1")

    def tokenize(self, text: str) -> Sequence[str]:
        return text.split()


RAW = {
    "event_summary": [
        {
            "ID": "summary-1",
            "text": "SOURCE_SUMMARY 甲 乙 丙 丁 戊",
            "summary": "REFERENCE_SUMMARY 甲 乙 丙 丁",
            "ignored": "UNRELATED_FIELD",
        }
    ],
    "questanswer_1doc": [
        {
            "ID": "qa-1",
            "news1": "SOURCE_QA 春 夏 秋 冬 风",
            "questions": "ORIGINAL_QUESTION 是什么？",
            "answers": "REFERENCE_QA 春 夏 秋 冬",
        }
    ],
}
GENERATION_PROFILE = "a" * 64
QG_PROFILE = "b" * 64
QA_PROFILE = "c" * 64
SEMANTIC_PROFILE = "d" * 64
DATASET_SHA = "e" * 64


def _input_fingerprints() -> dict[str, str]:
    return {
        key: format(index, "064x")
        for index, key in enumerate(sorted(PUBLIC_INPUT_FINGERPRINT_KEYS), start=1)
    }


def _cases():  # type annotation omitted to keep fixture call sites compact
    return build_generation_cases(RAW)


def _questions(cases=None):  # type annotation omitted to keep fixture call sites compact
    rows = _qg_rows(cases)
    return tuple(question for row in rows for question in row.questions)


def _generation_rows(cases=None) -> tuple[GenerationCacheRow, ...]:
    cases = _cases() if cases is None else cases
    predictions = {
        "event_summary": "预测 摘要 甲 乙 丙 丁",
        "questanswer_1doc": "预测 答案 春 夏 秋 冬",
    }
    return tuple(
        GenerationCacheRow(
            schema=CACHE_ROW_SCHEMA,
            cache_id=generation_cache_id(case, GENERATION_PROFILE),
            case_key=case.case_key,
            task=case.task,
            prediction=predictions[case.task],
            served_model="synthetic-generator",
            prompt_tokens=10,
            completion_tokens=4,
            reasoning_tokens=0,
        )
        for case in cases
    )


def _qg_rows(cases=None) -> tuple[QGCacheRow, ...]:
    cases = _cases() if cases is None else cases
    result = []
    for case in cases:
        texts = (f"{case.task} 的第一项关键信息是什么？",)
        questions = tuple(
            GeneratedQuestion(
                question_id=question_id(case, ordinal, text),
                case_key=case.case_key,
                ordinal=ordinal,
                text=text,
            )
            for ordinal, text in enumerate(texts)
        )
        result.append(
            QGCacheRow(
                schema=CACHE_ROW_SCHEMA,
                cache_id=qg_cache_id(case, QG_PROFILE),
                case_key=case.case_key,
                questions=questions,
                served_model="synthetic-qg",
            )
        )
    return tuple(result)


def _qa_rows(cases=None, *, lane: str) -> tuple[QACacheRow, ...]:
    cases = _cases() if cases is None else cases
    questions = _questions(cases)
    generation = {row.case_key: row.prediction for row in _generation_rows(cases)}
    case_by_key = {case.case_key: case for case in cases}
    result = []
    for question in questions:
        case = case_by_key[question.case_key]
        if lane == "reference":
            cache_id = reference_qa_cache_id(case, question, QA_PROFILE)
            answer = "蓝 色 纸 张" if case.task == "event_summary" else "春 夏 秋 冬"
        else:
            cache_id = prediction_qa_cache_id(
                case,
                question,
                generation[case.case_key],
                QA_PROFILE,
            )
            answer = "蓝 色 纸 张" if case.task == "event_summary" else "无法推断"
        result.append(
            QACacheRow(
                schema=CACHE_ROW_SCHEMA,
                cache_id=cache_id,
                case_key=case.case_key,
                question_id=question.question_id,
                lane=lane,  # type: ignore[arg-type]
                answer=answer,
                served_model="synthetic-judge",
            )
        )
    return tuple(result)


def _semantic_rows(cases=None) -> tuple[SemanticCacheRow, ...]:
    cases = _cases() if cases is None else cases
    predictions = {row.case_key: row.prediction for row in _generation_rows(cases)}
    return tuple(
        SemanticCacheRow(
            schema=CACHE_ROW_SCHEMA,
            cache_id=semantic_cache_id(
                case,
                predictions[case.case_key],
                SEMANTIC_PROFILE,
            ),
            case_key=case.case_key,
            precision=0.1 + index / 10,
            recall=0.2 + index / 10,
            f1=0.15 + index / 10,
        )
        for index, case in enumerate(cases)
    )


class TestCasesAndIdentity:
    def test_loads_only_the_two_frozen_shapes_in_task_order(self) -> None:
        summary, qa = _cases()

        assert (summary.task, qa.task) == ("event_summary", "questanswer_1doc")
        assert summary.generation_context == "SOURCE_SUMMARY 甲 乙 丙 丁 戊"
        assert summary.reference == "REFERENCE_SUMMARY 甲 乙 丙 丁"
        assert "ORIGINAL_QUESTION" in qa.user_request
        assert "UNRELATED_FIELD" not in repr(summary)

    @pytest.mark.parametrize(
        "raw",
        [
            {**RAW, "event_summary": []},
            {**RAW, "questanswer_1doc": []},
            {**RAW, "event_summary": [{"ID": "x", "text": "正文"}]},
            {
                **RAW,
                "event_summary": [
                    RAW["event_summary"][0],
                    RAW["event_summary"][0],
                ],
            },
        ],
    )
    def test_rejects_empty_missing_or_duplicate_cases(self, raw: object) -> None:
        with pytest.raises(ValueError, match=r"non-empty|summary|unique"):
            build_generation_cases(raw)  # type: ignore[arg-type]

    def test_manifest_separates_generation_and_reference_universes(self) -> None:
        base = build_input_manifest(_cases(), dataset_sha256=DATASET_SHA)
        source_changed = copy.deepcopy(RAW)
        source_changed["event_summary"][0]["text"] = "CHANGED_SOURCE 甲 乙 丙 丁"
        changed_source_manifest = build_input_manifest(
            build_generation_cases(source_changed),
            dataset_sha256=DATASET_SHA,
        )
        reference_changed = copy.deepcopy(RAW)
        reference_changed["event_summary"][0]["summary"] = "CHANGED_REFERENCE 甲乙"
        changed_reference_manifest = build_input_manifest(
            build_generation_cases(reference_changed),
            dataset_sha256=DATASET_SHA,
        )

        assert (
            base.generation_universe_fingerprint
            != changed_source_manifest.generation_universe_fingerprint
        )
        assert (
            base.reference_universe_fingerprint
            == changed_source_manifest.reference_universe_fingerprint
        )
        assert (
            base.generation_universe_fingerprint
            == changed_reference_manifest.generation_universe_fingerprint
        )
        assert (
            base.reference_universe_fingerprint
            != changed_reference_manifest.reference_universe_fingerprint
        )

    def test_manifest_parser_round_trips_and_rejects_key_drift(self) -> None:
        manifest = build_input_manifest(_cases(), dataset_sha256=DATASET_SHA)

        assert parse_generation_input_manifest(manifest.as_json()) == manifest

        extra = manifest.as_json()
        extra["extra"] = "x"
        with pytest.raises(ValueError, match="keys drift"):
            parse_generation_input_manifest(extra)

        missing = manifest.as_json()
        del missing["contract"]
        with pytest.raises(ValueError, match="keys drift"):
            parse_generation_input_manifest(missing)

    def test_cache_keys_have_stage_specific_dependencies(self) -> None:
        case = _cases()[0]
        source_changed_raw = copy.deepcopy(RAW)
        source_changed_raw["event_summary"][0]["text"] = "CHANGED_SOURCE"
        source_changed = build_generation_cases(source_changed_raw)[0]
        reference_changed_raw = copy.deepcopy(RAW)
        reference_changed_raw["event_summary"][0]["summary"] = "CHANGED_REFERENCE"
        reference_changed = build_generation_cases(reference_changed_raw)[0]

        assert generation_cache_id(case, GENERATION_PROFILE) != generation_cache_id(
            source_changed,
            GENERATION_PROFILE,
        )
        assert qg_cache_id(case, QG_PROFILE) == qg_cache_id(source_changed, QG_PROFILE)
        assert generation_cache_id(case, GENERATION_PROFILE) == generation_cache_id(
            reference_changed,
            GENERATION_PROFILE,
        )
        assert qg_cache_id(case, QG_PROFILE) != qg_cache_id(reference_changed, QG_PROFILE)

    def test_length_framing_prevents_concatenation_collision(self) -> None:
        assert canonical_fingerprint("schema", "ab", "c") != canonical_fingerprint(
            "schema", "a", "bc"
        )


class TestPromptsAndResponses:
    def test_qg_prompt_contains_only_reference(self) -> None:
        case = _cases()[1]
        prompt = qg_user_prompt(case)

        assert "REFERENCE_QA" in prompt
        assert "SOURCE_QA" not in prompt
        assert "ORIGINAL_QUESTION" not in prompt
        assert "预测" not in prompt

    def test_qa_prompt_contains_only_explicit_question_and_context(self) -> None:
        prompt = qa_user_prompt("QUESTION_ONLY？", "CONTEXT_ONLY")

        assert "QUESTION_ONLY" in prompt
        assert "CONTEXT_ONLY" in prompt
        assert "SOURCE_QA" not in prompt
        assert "REFERENCE_QA" not in prompt

    def test_generation_prompt_uses_context_and_request_but_not_reference(self) -> None:
        case = _cases()[1]
        prompt = generation_user_prompt(case)

        assert "SOURCE_QA" in prompt
        assert "ORIGINAL_QUESTION" in prompt
        assert "REFERENCE_QA" not in prompt

    def test_prompt_budget_refuses_instead_of_truncating(self) -> None:
        with pytest.raises(ValueError, match="refusing to truncate"):
            validate_prompt_budget("系统", "中" * 100, maximum=2)
        assert validate_prompt_budget("系统", "用户", maximum=MAX_INPUT_ESTIMATED_TOKENS) > 0

    def test_parses_strict_stage_responses(self) -> None:
        assert parse_generation_response("event_summary", '{"text":"摘要"}') == "摘要"
        assert parse_generation_response("questanswer_1doc", '{"answer":"答案"}') == "答案"
        assert parse_questions_response('{"questions":["问题一？","问题二？"]}') == (
            "问题一？",
            "问题二？",
        )
        assert parse_qa_response('{"answer":"无法推断。"}') == "无法推断。"

    @pytest.mark.parametrize(
        ("call", "raw", "match"),
        [
            (lambda raw: parse_generation_response("event_summary", raw), "{}", "keys drift"),
            (
                parse_questions_response,
                '{"questions":["重复？"," 重复？ "]}',
                "unique",
            ),
            (parse_questions_response, '{"questions":[]}', "1..8"),
            (parse_qa_response, '{"answer":" "}', "non-blank"),
            (parse_qa_response, "not-json", "malformed"),
        ],
    )
    def test_rejects_malformed_or_lossy_responses(self, call, raw: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            call(raw)


class TestCacheRows:
    def test_round_trips_every_strict_row_schema(self) -> None:
        generation = _generation_rows()[0]
        qg = _qg_rows()[0]
        reference_qa = _qa_rows(lane="reference")[0]
        semantic = _semantic_rows()[0]

        assert parse_generation_cache_row(generation.as_json()) == generation
        assert parse_qg_cache_row(qg.as_json()) == qg
        assert parse_qa_cache_row(reference_qa.as_json()) == reference_qa
        assert parse_semantic_cache_row(semantic.as_json()) == semantic

    def test_rejects_extra_keys_and_partial_usage(self) -> None:
        row = _generation_rows()[0].as_json()
        row["extra"] = "x"
        with pytest.raises(ValueError, match="extra"):
            parse_generation_cache_row(row)

        row = _generation_rows()[0].as_json()
        row["completion_tokens"] = None
        with pytest.raises(ValueError, match="complete or entirely unknown"):
            parse_generation_cache_row(row)

    def test_partial_cache_is_valid_but_complete_mode_rejects_it(self) -> None:
        cases = _cases()
        rows = _generation_rows(cases)[:1]

        assert (
            len(
                validate_generation_rows(
                    cases,
                    rows,
                    model_profile_fingerprint=GENERATION_PROFILE,
                )
            )
            == 1
        )
        with pytest.raises(ValueError, match="incomplete"):
            validate_generation_rows(
                cases,
                rows,
                model_profile_fingerprint=GENERATION_PROFILE,
                require_complete=True,
            )

    def test_rejects_duplicate_and_unknown_cache_ids(self) -> None:
        cases = _cases()
        row = _generation_rows(cases)[0]
        with pytest.raises(ValueError, match="duplicate"):
            validate_generation_rows(
                cases,
                (row, row),
                model_profile_fingerprint=GENERATION_PROFILE,
            )
        unknown = GenerationCacheRow(
            schema=CACHE_ROW_SCHEMA,
            cache_id="f" * 64,
            case_key=row.case_key,
            task=row.task,
            prediction=row.prediction,
            served_model=row.served_model,
        )
        with pytest.raises(ValueError, match="unknown"):
            validate_generation_rows(
                cases,
                (unknown,),
                model_profile_fingerprint=GENERATION_PROFILE,
            )

    def test_qg_validator_recomputes_question_identity(self) -> None:
        cases = _cases()
        row = _qg_rows(cases)[0]
        bad_question = GeneratedQuestion(
            question_id="f" * 64,
            case_key=row.case_key,
            ordinal=0,
            text=row.questions[0].text,
        )
        bad = QGCacheRow(
            schema=row.schema,
            cache_id=row.cache_id,
            case_key=row.case_key,
            questions=(bad_question,),
            served_model=row.served_model,
        )
        with pytest.raises(ValueError, match="question identity"):
            validate_qg_rows(
                cases,
                (bad,),
                model_profile_fingerprint=QG_PROFILE,
            )

    def test_qa_and_semantic_keys_bind_the_exact_prediction(self) -> None:
        cases = _cases()
        case = cases[0]
        question = _questions(cases)[0]
        assert prediction_qa_cache_id(case, question, "PREDICTION_A", QA_PROFILE) != (
            prediction_qa_cache_id(case, question, "PREDICTION_B", QA_PROFILE)
        )
        assert semantic_cache_id(case, "PREDICTION_A", SEMANTIC_PROFILE) != semantic_cache_id(
            case,
            "PREDICTION_B",
            SEMANTIC_PROFILE,
        )
        assert reference_qa_cache_id(case, question, QA_PROFILE) == reference_qa_cache_id(
            case,
            question,
            QA_PROFILE,
        )

    def test_all_stage_validators_accept_complete_exact_universes(self) -> None:
        cases = _cases()
        qg = _qg_rows(cases)
        questions = _questions(cases)
        generation = _generation_rows(cases)
        predictions = {row.case_key: row.prediction for row in generation}

        validate_generation_rows(
            cases,
            generation,
            model_profile_fingerprint=GENERATION_PROFILE,
            require_complete=True,
        )
        validate_qg_rows(
            cases,
            qg,
            model_profile_fingerprint=QG_PROFILE,
            require_complete=True,
        )
        validate_reference_qa_rows(
            cases,
            questions,
            _qa_rows(cases, lane="reference"),
            model_profile_fingerprint=QA_PROFILE,
            require_complete=True,
        )
        validate_prediction_qa_rows(
            cases,
            questions,
            _qa_rows(cases, lane="prediction"),
            model_profile_fingerprint=QA_PROFILE,
            predictions=predictions,
            require_complete=True,
        )
        validate_semantic_rows(
            cases,
            _semantic_rows(cases),
            predictions=predictions,
            semantic_profile_fingerprint=SEMANTIC_PROFILE,
            require_complete=True,
        )


class TestBanksAndScoring:
    def test_question_and_reference_banks_are_text_free_markers(self) -> None:
        cases = _cases()
        manifest = build_input_manifest(cases, dataset_sha256=DATASET_SHA)
        qg_rows = _qg_rows(cases)
        questions = _questions(cases)
        question_bank = build_question_bank(
            cases,
            qg_rows,
            reference_universe_fingerprint=manifest.reference_universe_fingerprint,
            qg_model_profile_fingerprint=QG_PROFILE,
        )
        reference_bank = build_reference_bank(
            cases,
            questions,
            _qa_rows(cases, lane="reference"),
            question_universe_fingerprint=question_bank.question_universe_fingerprint,
            qa_model_profile_fingerprint=QA_PROFILE,
        )

        encoded = json.dumps(
            {"question_bank": question_bank.as_json(), "reference_bank": reference_bank.as_json()}
        )
        assert question_bank.questions == 2
        assert reference_bank.questions == 2
        for secret in ("SOURCE_", "REFERENCE_", "问题", "答案"):
            assert secret not in encoded

    def test_bank_parsers_round_trip_and_reject_key_drift(self) -> None:
        cases = _cases()
        manifest = build_input_manifest(cases, dataset_sha256=DATASET_SHA)
        questions = _questions(cases)
        question_bank = build_question_bank(
            cases,
            _qg_rows(cases),
            reference_universe_fingerprint=manifest.reference_universe_fingerprint,
            qg_model_profile_fingerprint=QG_PROFILE,
        )
        reference_bank = build_reference_bank(
            cases,
            questions,
            _qa_rows(cases, lane="reference"),
            question_universe_fingerprint=question_bank.question_universe_fingerprint,
            qa_model_profile_fingerprint=QA_PROFILE,
        )

        assert parse_question_bank(question_bank.as_json()) == question_bank
        assert parse_reference_bank(reference_bank.as_json()) == reference_bank

        for parser, marker in (
            (parse_question_bank, question_bank),
            (parse_reference_bank, reference_bank),
        ):
            extra = marker.as_json()
            extra["extra"] = "x"
            with pytest.raises(ValueError, match="keys drift"):
                parser(extra)

            missing = marker.as_json()
            del missing["schema"]
            with pytest.raises(ValueError, match="keys drift"):
                parser(missing)

    def test_quest_pairs_use_reference_qa_answers_not_task_references(self) -> None:
        cases = _cases()
        questions = _questions(cases)
        pairs = build_quest_answer_pairs(
            questions,
            _qa_rows(cases, lane="reference"),
            _qa_rows(cases, lane="prediction"),
        )

        assert pairs[0].reference_answer == "蓝 色 纸 张"
        assert pairs[0].reference_answer != cases[0].reference
        assert pairs[1].generated_answer == "无法推断"

    def test_builds_text_free_samples_and_task_separated_report(self) -> None:
        cases = _cases()
        samples = build_numeric_samples(
            cases,
            generation_rows=_generation_rows(cases),
            semantic_rows=_semantic_rows(cases),
            questions=_questions(cases),
            reference_qa_rows=_qa_rows(cases, lane="reference"),
            prediction_qa_rows=_qa_rows(cases, lane="prediction"),
            tokenizer=SpaceTokenizer(),
            semantic_provenance=SemanticProvenance(
                distribution_version="synthetic-dist",
                module_version="synthetic-module",
            ),
            input_fingerprints=_input_fingerprints(),
        )
        report = build_generation_report(samples, resamples=20, seed=7)

        validate_numeric_samples(samples)
        validate_generation_report(report)
        assert samples["schema"] == GENERATION_SAMPLES_SCHEMA
        assert report["schema"] == GENERATION_REPORT_SCHEMA
        tasks = report["tasks"]
        assert tasks["event_summary"]["case_count"] == 1
        assert tasks["questanswer_1doc"]["case_count"] == 1
        assert (
            tasks["questanswer_1doc"]["ragquest"]["code_precision_generated_answerable"]["mean"]
            is None
        )
        assert (
            tasks["questanswer_1doc"]["ragquest"]["code_precision_generated_answerable"][
                "denominator"
            ]
            == 0
        )

        encoded = json.dumps({"samples": samples, "report": report}, ensure_ascii=False)
        for secret in (
            "SOURCE_SUMMARY",
            "SOURCE_QA",
            "REFERENCE_SUMMARY",
            "REFERENCE_QA",
            "ORIGINAL_QUESTION",
            "预测 摘要",
            "蓝 色 纸 张",
            "无法推断",
        ):
            assert secret not in encoded

    def test_public_validators_reject_raw_field_injection(self) -> None:
        cases = _cases()
        samples = build_numeric_samples(
            cases,
            generation_rows=_generation_rows(cases),
            semantic_rows=_semantic_rows(cases),
            questions=_questions(cases),
            reference_qa_rows=_qa_rows(cases, lane="reference"),
            prediction_qa_rows=_qa_rows(cases, lane="prediction"),
            tokenizer=SpaceTokenizer(),
            semantic_provenance=SemanticProvenance(),
            input_fingerprints=_input_fingerprints(),
        )
        report = build_generation_report(samples, resamples=5)
        tampered = copy.deepcopy(report)
        tampered["tasks"]["event_summary"]["question"] = "SECRET"

        with pytest.raises(ValueError, match=r"keys drift|forbidden"):
            validate_generation_report(tampered)

    def test_tampered_numeric_structure_cannot_be_republished(self) -> None:
        cases = _cases()
        samples = build_numeric_samples(
            cases,
            generation_rows=_generation_rows(cases),
            semantic_rows=_semantic_rows(cases),
            questions=_questions(cases),
            reference_qa_rows=_qa_rows(cases, lane="reference"),
            prediction_qa_rows=_qa_rows(cases, lane="prediction"),
            tokenizer=SpaceTokenizer(),
            semantic_provenance=SemanticProvenance(),
            input_fingerprints=_input_fingerprints(),
        )
        tampered = copy.deepcopy(samples)
        tampered["case_rows"][0]["sequence"] = 1

        with pytest.raises(ValueError, match="dense"):
            validate_numeric_samples(tampered)
