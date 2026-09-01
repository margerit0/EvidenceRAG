from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from zhrag.eval.metrics_gen import TokenizerProvenance
from zhrag.eval.quest_eval import (
    QUEST_EVAL_SCHEMA,
    UNANSWERABLE_SENTINEL,
    QuestAnswerPair,
    TokenOverlapScore,
    classify_unanswerable,
    evaluate_quest_answers,
    token_overlap_f1,
)


class SpaceTokenizer:
    @property
    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(name="synthetic-space", package="tests", version="1")

    def tokenize(self, text: str) -> Sequence[str]:
        return text.split()


class TestSentinel:
    @pytest.mark.parametrize(
        ("answer", "match", "unanswerable"),
        [
            (UNANSWERABLE_SENTINEL, "exact", True),
            ("无法 推断。", "normalized", True),
            ("“无法推断！”", "normalized", True),
            ("无法推断", "exact", True),
            ("根据材料无法推断具体日期", "near", False),
            ("答案是无法推断的例外", "near", False),
            ("系统会返回成功", "none", False),
        ],
    )
    def test_normalizes_only_wrappers_and_marks_near_misses(
        self,
        answer: str,
        match: str,
        unanswerable: bool,
    ) -> None:
        result = classify_unanswerable(answer)
        assert result.match == match
        assert result.is_unanswerable is unanswerable
        assert result.is_near_sentinel is (match == "near")

    @pytest.mark.parametrize("answer", ["", " ", "\n\t"])
    def test_rejects_blank_answers(self, answer: str) -> None:
        with pytest.raises(ValueError, match="non-blank"):
            classify_unanswerable(answer)


def test_token_overlap_uses_multiset_counts() -> None:
    score = token_overlap_f1(
        ("甲", "甲", "乙"),
        ("甲", "乙", "乙", "丙"),
    )

    assert score.overlap_tokens == 2
    assert score.generated_tokens == 3
    assert score.reference_tokens == 4
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(4 / 7)


def test_report_keeps_paper_and_code_denominators_distinct() -> None:
    pairs = (
        QuestAnswerPair("q1", "蓝色 纸张", "蓝色 纸张"),
        QuestAnswerPair("q2", "无法推断", "无法推断。"),
        QuestAnswerPair("q3", "绿色 纸张", "无法推断"),
        QuestAnswerPair("q4", "黄色 纸张", "根据材料无法推断具体日期"),
    )

    report = evaluate_quest_answers(pairs, SpaceTokenizer())

    assert report.schema == QUEST_EVAL_SCHEMA
    assert report.total_questions == 4
    assert report.reference_unanswerable_count == 1
    assert report.generated_unanswerable_count == 2
    assert report.generated_unanswerable_after_reference_filter_count == 1
    assert report.reference_exact_sentinel_count == 1
    assert report.reference_normalized_sentinel_count == 0
    assert report.generated_exact_sentinel_count == 1
    assert report.generated_normalized_sentinel_count == 1
    assert report.generated_near_sentinel_count == 1
    assert report.paper_recall_denominator_all_questions == 4
    assert report.paper_precision_denominator_all_questions == 4
    assert report.code_recall_denominator_reference_answerable == 3
    assert report.code_precision_denominator_generated_answerable == 2
    assert report.paper_recall_all_questions == pytest.approx(0.5)
    assert report.paper_precision_all_questions == pytest.approx(0.25)
    assert report.code_recall_reference_answerable == pytest.approx(2 / 3)
    assert report.code_precision_generated_answerable == pytest.approx(0.5)
    assert tuple(row.question_id for row in report.scores) == ("q1", "q2", "q3", "q4")
    assert report.scores[1].token_overlap is None
    assert report.scores[3].token_overlap is not None


def test_near_sentinel_is_scored_as_an_answerable_string() -> None:
    report = evaluate_quest_answers(
        (QuestAnswerPair("q1", "材料 没有 日期", "根据材料无法推断具体日期"),),
        SpaceTokenizer(),
    )

    assert report.reference_unanswerable_count == 0
    assert report.generated_unanswerable_count == 0
    assert report.generated_near_sentinel_count == 1
    assert report.code_precision_generated_answerable == 0.0
    assert report.paper_precision_all_questions == 0.0


def test_all_generated_unanswerable_makes_code_precision_none() -> None:
    pairs = (
        QuestAnswerPair("q1", "甲答案", "无法推断"),
        QuestAnswerPair("q2", "乙答案", "无法推断。"),
    )

    report = evaluate_quest_answers(pairs, SpaceTokenizer())

    assert report.code_recall_reference_answerable == 0.0
    assert report.code_precision_generated_answerable is None
    assert report.code_precision_denominator_generated_answerable == 0


def test_all_reference_unanswerable_makes_code_recall_none() -> None:
    pairs = (
        QuestAnswerPair("q1", "无法推断", "甲答案"),
        QuestAnswerPair("q2", "无法推断。", "乙答案"),
    )

    report = evaluate_quest_answers(pairs, SpaceTokenizer())

    assert report.code_recall_reference_answerable is None
    assert report.code_precision_generated_answerable is None
    assert report.code_recall_denominator_reference_answerable == 0
    assert report.paper_recall_all_questions == 1.0
    assert report.paper_precision_all_questions == 0.0


@pytest.mark.parametrize(
    "pairs",
    [
        (),
        (QuestAnswerPair("same", "甲", "乙"), QuestAnswerPair("same", "丙", "丁")),
    ],
)
def test_rejects_empty_or_duplicate_question_sets(
    pairs: tuple[QuestAnswerPair, ...],
) -> None:
    with pytest.raises(ValueError, match=r"non-empty|unique"):
        evaluate_quest_answers(pairs, SpaceTokenizer())


def test_tokenizer_failure_is_not_converted_to_zero() -> None:
    class BrokenTokenizer(SpaceTokenizer):
        def tokenize(self, text: str) -> Sequence[str]:
            raise RuntimeError("synthetic tokenizer failure")

    with pytest.raises(RuntimeError, match="synthetic tokenizer failure"):
        evaluate_quest_answers((QuestAnswerPair("q1", "甲答案", "乙答案"),), BrokenTokenizer())


def test_bad_token_stream_is_rejected() -> None:
    class BadTokenizer(SpaceTokenizer):
        def tokenize(self, text: str) -> Sequence[str]:
            return ("甲", " ")

    with pytest.raises(ValueError, match="non-blank"):
        evaluate_quest_answers((QuestAnswerPair("q1", "甲答案", "乙答案"),), BadTokenizer())


def test_token_overlap_rejects_nonfinite_or_boolean_derived_values() -> None:
    # The public constructor remains fail-closed even when called independently.
    with pytest.raises(ValueError, match=r"finite|real"):
        TokenOverlapScore(math.nan, 0.5, 0.5, 1, 2, 2)
    with pytest.raises(ValueError, match="real"):
        TokenOverlapScore(True, 0.5, 0.5, 1, 2, 2)
