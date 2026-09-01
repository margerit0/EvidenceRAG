"""Pure answer scoring for the two documented RAGQuestEval semantics.

This module neither generates questions nor calls a QA model.  It scores answer
pairs that already exist, while keeping the paper's all-question denominators
separate from the conditional filtering observed in the historical CRUD-RAG
implementation.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from zhrag.eval.metrics_gen import Tokenizer, TokenizerProvenance, tokenize_text

__all__ = [
    "QUEST_EVAL_SCHEMA",
    "UNANSWERABLE_SENTINEL",
    "AnswerSentinelClassification",
    "QuestAnswerPair",
    "QuestAnswerScore",
    "QuestEvalReport",
    "SentinelMatch",
    "TokenOverlapScore",
    "classify_unanswerable",
    "evaluate_quest_answers",
    "token_overlap_f1",
]

QUEST_EVAL_SCHEMA = "zhrag-rag-quest-eval-v1"
UNANSWERABLE_SENTINEL = "无法推断"

type SentinelMatch = Literal["none", "exact", "normalized", "near"]
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
    ("《", "》"),
)
_TERMINAL_PUNCTUATION = "。.!！?？;；,，…"


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _count(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _unit_interval(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _normalise_for_sentinel(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value)
    normalised = "".join(normalised.split())
    while normalised:
        previous = normalised
        normalised = normalised.rstrip(_TERMINAL_PUNCTUATION)
        for opening, closing in _QUOTE_PAIRS:
            if normalised.startswith(opening) and normalised.endswith(closing):
                normalised = normalised[len(opening) : -len(closing)]
                break
        if normalised == previous:
            break
    return normalised


@dataclass(frozen=True, slots=True)
class AnswerSentinelClassification:
    """Classification only; normalized answer text is deliberately not retained."""

    match: SentinelMatch

    def __post_init__(self) -> None:
        if self.match not in ("none", "exact", "normalized", "near"):
            raise ValueError(f"unknown sentinel classification {self.match!r}")

    @property
    def is_unanswerable(self) -> bool:
        return self.match in ("exact", "normalized")

    @property
    def is_near_sentinel(self) -> bool:
        return self.match == "near"


def classify_unanswerable(answer: str) -> AnswerSentinelClassification:
    """Recognize the exact sentinel after a finite wrapper normalization.

    Unicode compatibility forms, whitespace, balanced outer quotation marks and
    terminal Chinese/ASCII punctuation are normalized.  A sentence that merely
    contains the sentinel remains answerable and is classified as ``near`` so
    the ambiguity is visible in the report.
    """

    _require_text(answer, label="answer")
    if answer == UNANSWERABLE_SENTINEL:
        return AnswerSentinelClassification("exact")
    normalised = _normalise_for_sentinel(answer)
    if normalised == UNANSWERABLE_SENTINEL:
        return AnswerSentinelClassification("normalized")
    if UNANSWERABLE_SENTINEL in normalised:
        return AnswerSentinelClassification("near")
    return AnswerSentinelClassification("none")


@dataclass(frozen=True, slots=True)
class QuestAnswerPair:
    """One already-produced reference/generated answer pair."""

    question_id: str
    reference_answer: str
    generated_answer: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, label="question_id")
        _require_text(self.reference_answer, label=f"{self.question_id}: reference_answer")
        _require_text(self.generated_answer, label=f"{self.question_id}: generated_answer")


@dataclass(frozen=True, slots=True)
class TokenOverlapScore:
    """Multiset token-overlap precision, recall and beta=1 F1."""

    precision: float
    recall: float
    f1: float
    overlap_tokens: int
    generated_tokens: int
    reference_tokens: int

    def __post_init__(self) -> None:
        for name in ("precision", "recall", "f1"):
            _unit_interval(getattr(self, name), label=f"token overlap {name}")
        for name in ("overlap_tokens", "generated_tokens", "reference_tokens"):
            _count(getattr(self, name), label=name)
        if self.generated_tokens < 1 or self.reference_tokens < 1:
            raise ValueError("token overlap input lengths must be positive")
        if self.overlap_tokens > min(self.generated_tokens, self.reference_tokens):
            raise ValueError("token overlap exceeds an input length")


def token_overlap_f1(
    generated_tokens: Sequence[str],
    reference_tokens: Sequence[str],
) -> TokenOverlapScore:
    """Compute bag-of-token overlap without discarding repeated tokens."""

    generated = _validated_tokens(generated_tokens, label="generated tokens")
    reference = _validated_tokens(reference_tokens, label="reference tokens")
    overlap = sum((Counter(generated) & Counter(reference)).values())
    precision = overlap / len(generated)
    recall = overlap / len(reference)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return TokenOverlapScore(
        precision=precision,
        recall=recall,
        f1=f1,
        overlap_tokens=overlap,
        generated_tokens=len(generated),
        reference_tokens=len(reference),
    )


def _validated_tokens(tokens: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise ValueError(f"{label} must be a non-empty sequence")
    checked: list[str] = []
    for index, token in enumerate(tokens):
        if type(token) is not str or not token or token.isspace():
            raise ValueError(f"{label}[{index}] must be a non-blank string")
        checked.append(token)
    if not checked:
        raise ValueError(f"{label} must be non-empty")
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class QuestAnswerScore:
    """One classified pair and its conditional answer-similarity score."""

    question_id: str
    reference_sentinel: AnswerSentinelClassification
    generated_sentinel: AnswerSentinelClassification
    token_overlap: TokenOverlapScore | None

    def __post_init__(self) -> None:
        _require_text(self.question_id, label="question score question_id")
        both_answerable = not (
            self.reference_sentinel.is_unanswerable or self.generated_sentinel.is_unanswerable
        )
        if both_answerable != (self.token_overlap is not None):
            raise ValueError("token overlap exists exactly when both answers are answerable")


@dataclass(frozen=True, slots=True)
class QuestEvalReport:
    """Both paper and historical-code semantics with explicit denominators."""

    schema: str
    tokenizer: TokenizerProvenance
    scores: tuple[QuestAnswerScore, ...]
    total_questions: int
    reference_unanswerable_count: int
    generated_unanswerable_count: int
    generated_unanswerable_after_reference_filter_count: int
    reference_exact_sentinel_count: int
    reference_normalized_sentinel_count: int
    reference_near_sentinel_count: int
    generated_exact_sentinel_count: int
    generated_normalized_sentinel_count: int
    generated_near_sentinel_count: int
    paper_recall_denominator_all_questions: int
    paper_precision_denominator_all_questions: int
    code_recall_denominator_reference_answerable: int
    code_precision_denominator_generated_answerable: int
    paper_recall_all_questions: float
    paper_precision_all_questions: float
    code_recall_reference_answerable: float | None
    code_precision_generated_answerable: float | None

    def __post_init__(self) -> None:
        if self.schema != QUEST_EVAL_SCHEMA:
            raise ValueError(f"unsupported RAGQuestEval schema {self.schema!r}")
        if type(self.total_questions) is not int or self.total_questions < 1:
            raise ValueError("total_questions must be a positive integer")
        if len(self.scores) != self.total_questions:
            raise ValueError("score count does not match total_questions")
        ids = [score.question_id for score in self.scores]
        if len(set(ids)) != len(ids):
            raise ValueError("question score ids must be unique")

        count_names = (
            "reference_unanswerable_count",
            "generated_unanswerable_count",
            "generated_unanswerable_after_reference_filter_count",
            "reference_exact_sentinel_count",
            "reference_normalized_sentinel_count",
            "reference_near_sentinel_count",
            "generated_exact_sentinel_count",
            "generated_normalized_sentinel_count",
            "generated_near_sentinel_count",
            "paper_recall_denominator_all_questions",
            "paper_precision_denominator_all_questions",
            "code_recall_denominator_reference_answerable",
            "code_precision_denominator_generated_answerable",
        )
        for name in count_names:
            _count(getattr(self, name), label=name)

        expected = _report_counts(self.scores)
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match classified score rows")

        paper_recall = _unit_interval(
            self.paper_recall_all_questions,
            label="paper_recall_all_questions",
        )
        paper_precision = _unit_interval(
            self.paper_precision_all_questions,
            label="paper_precision_all_questions",
        )
        expected_paper_recall = (
            self.total_questions - self.generated_unanswerable_count
        ) / self.total_questions
        overlap_sum = math.fsum(
            score.token_overlap.f1 for score in self.scores if score.token_overlap is not None
        )
        expected_paper_precision = overlap_sum / self.total_questions
        if not math.isclose(paper_recall, expected_paper_recall, abs_tol=1e-12):
            raise ValueError("paper_recall_all_questions does not match score rows")
        if not math.isclose(paper_precision, expected_paper_precision, abs_tol=1e-12):
            raise ValueError("paper_precision_all_questions does not match score rows")

        self._validate_conditional(
            "code_recall_reference_answerable",
            self.code_recall_reference_answerable,
            numerator=self.code_precision_denominator_generated_answerable,
            denominator=self.code_recall_denominator_reference_answerable,
        )
        self._validate_conditional(
            "code_precision_generated_answerable",
            self.code_precision_generated_answerable,
            numerator=overlap_sum,
            denominator=self.code_precision_denominator_generated_answerable,
        )

    @staticmethod
    def _validate_conditional(
        label: str,
        value: float | None,
        *,
        numerator: float,
        denominator: int,
    ) -> None:
        if denominator == 0:
            if value is not None:
                raise ValueError(f"{label} must be None for an empty conditional set")
            return
        if value is None:
            raise ValueError(f"{label} is required for a non-empty conditional set")
        checked = _unit_interval(value, label=label)
        if not math.isclose(checked, numerator / denominator, abs_tol=1e-12):
            raise ValueError(f"{label} does not match score rows")


def _report_counts(scores: Sequence[QuestAnswerScore]) -> dict[str, int]:
    reference_unanswerable = sum(score.reference_sentinel.is_unanswerable for score in scores)
    generated_unanswerable = sum(score.generated_sentinel.is_unanswerable for score in scores)
    generated_removed_after_reference = sum(
        not score.reference_sentinel.is_unanswerable and score.generated_sentinel.is_unanswerable
        for score in scores
    )
    reference_answerable = len(scores) - reference_unanswerable
    both_answerable = reference_answerable - generated_removed_after_reference
    return {
        "reference_unanswerable_count": reference_unanswerable,
        "generated_unanswerable_count": generated_unanswerable,
        "generated_unanswerable_after_reference_filter_count": generated_removed_after_reference,
        "reference_exact_sentinel_count": sum(
            score.reference_sentinel.match == "exact" for score in scores
        ),
        "reference_normalized_sentinel_count": sum(
            score.reference_sentinel.match == "normalized" for score in scores
        ),
        "reference_near_sentinel_count": sum(
            score.reference_sentinel.match == "near" for score in scores
        ),
        "generated_exact_sentinel_count": sum(
            score.generated_sentinel.match == "exact" for score in scores
        ),
        "generated_normalized_sentinel_count": sum(
            score.generated_sentinel.match == "normalized" for score in scores
        ),
        "generated_near_sentinel_count": sum(
            score.generated_sentinel.match == "near" for score in scores
        ),
        "paper_recall_denominator_all_questions": len(scores),
        "paper_precision_denominator_all_questions": len(scores),
        "code_recall_denominator_reference_answerable": reference_answerable,
        "code_precision_denominator_generated_answerable": both_answerable,
    }


def evaluate_quest_answers(
    pairs: Sequence[QuestAnswerPair],
    tokenizer: Tokenizer,
) -> QuestEvalReport:
    """Score all pairs, preserving both denominator definitions or fail closed."""

    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
        raise ValueError("RAGQuestEval requires a non-empty sequence of answer pairs")
    checked_pairs: list[QuestAnswerPair] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, QuestAnswerPair):
            raise ValueError(f"pairs[{index}] must be a QuestAnswerPair")
        checked_pairs.append(pair)
    if len({pair.question_id for pair in checked_pairs}) != len(checked_pairs):
        raise ValueError("question ids must be unique")

    scores: list[QuestAnswerScore] = []
    for pair in checked_pairs:
        reference_sentinel = classify_unanswerable(pair.reference_answer)
        generated_sentinel = classify_unanswerable(pair.generated_answer)
        overlap = None
        if not reference_sentinel.is_unanswerable and not generated_sentinel.is_unanswerable:
            generated_tokens = tokenize_text(
                pair.generated_answer,
                tokenizer,
                label=f"{pair.question_id}: generated_answer",
            )
            reference_tokens = tokenize_text(
                pair.reference_answer,
                tokenizer,
                label=f"{pair.question_id}: reference_answer",
            )
            overlap = token_overlap_f1(generated_tokens, reference_tokens)
        scores.append(
            QuestAnswerScore(
                question_id=pair.question_id,
                reference_sentinel=reference_sentinel,
                generated_sentinel=generated_sentinel,
                token_overlap=overlap,
            )
        )

    score_rows = tuple(scores)
    counts = _report_counts(score_rows)
    total = len(score_rows)
    overlap_sum = math.fsum(
        score.token_overlap.f1 for score in score_rows if score.token_overlap is not None
    )
    code_recall_denominator = counts["code_recall_denominator_reference_answerable"]
    code_precision_denominator = counts["code_precision_denominator_generated_answerable"]
    return QuestEvalReport(
        schema=QUEST_EVAL_SCHEMA,
        tokenizer=tokenizer.provenance,
        scores=score_rows,
        total_questions=total,
        **counts,
        paper_recall_all_questions=(total - counts["generated_unanswerable_count"]) / total,
        paper_precision_all_questions=overlap_sum / total,
        code_recall_reference_answerable=(
            None
            if code_recall_denominator == 0
            else code_precision_denominator / code_recall_denominator
        ),
        code_precision_generated_answerable=(
            None if code_precision_denominator == 0 else overlap_sum / code_precision_denominator
        ),
    )
