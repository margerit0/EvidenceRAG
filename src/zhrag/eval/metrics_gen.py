"""Independent, reproducible metrics for generated Chinese text.

The implementation is intentionally small and provider-free.  BLEU and
ROUGE-L are computed over tokens supplied by an explicit tokenizer, one
prediction/reference pair at a time, and then averaged.  The no-brevity-penalty
BLEU value is exposed under a separate compatibility name; it is not standard
BLEU.

Optional packages are imported only inside their adapters.  Importing this
module therefore does not import ``jieba``, ``bert_score``, ``torch`` or
``transformers``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from types import ModuleType
from typing import Protocol, cast

__all__ = [
    "BERTSCORE_BATCH_SIZE",
    "BERTSCORE_IDF",
    "BERTSCORE_LANG",
    "BERTSCORE_METRIC",
    "BERTSCORE_MODEL",
    "BERTSCORE_NUM_LAYERS",
    "BERTSCORE_RESCALE_WITH_BASELINE",
    "BERTSCORE_USE_FAST_TOKENIZER",
    "CRUD_BLEU_METRIC",
    "GENERATION_METRICS_SCHEMA",
    "ROUGE_L_METRIC",
    "STANDARD_BLEU_METRIC",
    "BertScoreAdapter",
    "BleuScore",
    "GenerationExample",
    "GenerationMetricReport",
    "JiebaTokenizer",
    "LexicalExampleScore",
    "RougeLScore",
    "SemanticExampleScore",
    "SemanticProvenance",
    "SemanticScore",
    "SemanticScorer",
    "Tokenizer",
    "TokenizerProvenance",
    "evaluate_generation",
    "rouge_l",
    "sentence_bleu4",
    "tokenize_text",
]

GENERATION_METRICS_SCHEMA = "zhrag-generation-metrics-v1"
STANDARD_BLEU_METRIC = "mean_sentence_bleu4"
CRUD_BLEU_METRIC = "crud_mean_sentence_bleu4_no_bp"
ROUGE_L_METRIC = "mean_sentence_rouge_l_f1"
BERTSCORE_METRIC = "mean_bertscore_f1_zh_rescaled"

BERTSCORE_LANG = "zh"
BERTSCORE_MODEL = "bert-base-chinese"
BERTSCORE_NUM_LAYERS = 8
BERTSCORE_RESCALE_WITH_BASELINE = True
BERTSCORE_IDF = False
BERTSCORE_BATCH_SIZE = 64
BERTSCORE_USE_FAST_TOKENIZER = False


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _finite_number(value: object, *, label: str, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {number!r}")
    if unit_interval and not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {number!r}")
    return number


def _validate_tokens(tokens: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise ValueError(f"{label} must be a non-empty sequence of strings")
    result: list[str] = []
    for index, token in enumerate(tokens):
        if type(token) is not str or not token or token.isspace():
            raise ValueError(f"{label}[{index}] must be a non-blank string")
        result.append(token)
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GenerationExample:
    """One generated answer and its single reference answer."""

    sample_id: str
    prediction: str
    reference: str

    def __post_init__(self) -> None:
        _require_text(self.sample_id, label="sample_id")
        _require_text(self.prediction, label=f"{self.sample_id}: prediction")
        _require_text(self.reference, label=f"{self.sample_id}: reference")


@dataclass(frozen=True, slots=True)
class TokenizerProvenance:
    """The segmentation identity recorded with every lexical report."""

    name: str
    package: str
    version: str
    mode: str = "custom"
    hmm: bool = False
    user_dictionary: bool = False

    def __post_init__(self) -> None:
        for field_name in ("name", "package", "version", "mode"):
            _require_text(getattr(self, field_name), label=f"tokenizer {field_name}")
        if type(self.hmm) is not bool or type(self.user_dictionary) is not bool:
            raise ValueError("tokenizer boolean provenance fields must be bool")


class Tokenizer(Protocol):
    """Narrow tokenization port used by the pure metric functions."""

    @property
    def provenance(self) -> TokenizerProvenance: ...

    def tokenize(self, text: str) -> Sequence[str]: ...


class _JiebaBackend(Protocol):
    def lcut(self, text: str, *, cut_all: bool, HMM: bool) -> Sequence[str]: ...


class JiebaTokenizer:
    """Jieba precise-mode tokenizer, loaded only when it is first used."""

    __slots__ = ("_tokenizer",)

    def __init__(self) -> None:
        self._tokenizer: _JiebaBackend | None = None

    @staticmethod
    def _load_module() -> ModuleType:
        try:
            return importlib.import_module("jieba")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "jieba is required for the frozen Chinese generation metrics; "
                "install it with `uv sync --extra verify`"
            ) from exc

    @property
    def provenance(self) -> TokenizerProvenance:
        try:
            version = importlib.metadata.version("jieba")
        except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "jieba is required for the frozen Chinese generation metrics; "
                "install it with `uv sync --extra verify`"
            ) from exc
        return TokenizerProvenance(
            name="jieba-precise",
            package="jieba",
            version=version,
            mode="precise",
            hmm=True,
            user_dictionary=False,
        )

    def tokenize(self, text: str) -> Sequence[str]:
        _require_text(text, label="text")
        module = self._load_module()
        if self._tokenizer is None:
            factory = cast(Callable[[], _JiebaBackend], module.Tokenizer)
            self._tokenizer = factory()
        return tuple(
            token
            for token in self._tokenizer.lcut(text, cut_all=False, HMM=True)
            if token and not token.isspace()
        )


def tokenize_text(text: str, tokenizer: Tokenizer, *, label: str = "text") -> tuple[str, ...]:
    """Tokenize and validate one string before any aggregate is published."""

    _require_text(text, label=label)
    return _validate_tokens(tokenizer.tokenize(text), label=f"{label} tokens")


@dataclass(frozen=True, slots=True)
class BleuScore:
    """Unsmoothed single-reference sentence BLEU-4 and its no-BP companion."""

    score: float
    score_without_brevity_penalty: float
    precisions: tuple[float, float, float, float]
    brevity_penalty: float
    candidate_length: int
    reference_length: int

    def __post_init__(self) -> None:
        _finite_number(self.score, label="BLEU-4 score", unit_interval=True)
        _finite_number(
            self.score_without_brevity_penalty,
            label="BLEU-4 score without brevity penalty",
            unit_interval=True,
        )
        if len(self.precisions) != 4:
            raise ValueError("BLEU precisions must contain four orders")
        for order, precision in enumerate(self.precisions, start=1):
            _finite_number(precision, label=f"BLEU-{order} precision", unit_interval=True)
        _finite_number(self.brevity_penalty, label="BLEU brevity penalty", unit_interval=True)
        if type(self.candidate_length) is not int or self.candidate_length < 1:
            raise ValueError("BLEU candidate_length must be a positive integer")
        if type(self.reference_length) is not int or self.reference_length < 1:
            raise ValueError("BLEU reference_length must be a positive integer")

    @property
    def bleu4(self) -> float:
        """Alias for callers that prefer the explicit order in the field name."""

        return self.score

    @property
    def bleu4_without_brevity_penalty(self) -> float:
        return self.score_without_brevity_penalty


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def sentence_bleu4(candidate: Sequence[str], reference: Sequence[str]) -> BleuScore:
    """Compute unsmoothed, one-reference sentence BLEU-4.

    There is no smoothing or effective-order fallback.  A missing order has
    zero precision and therefore makes the geometric mean zero.
    """

    candidate_tokens = _validate_tokens(candidate, label="candidate tokens")
    reference_tokens = _validate_tokens(reference, label="reference tokens")
    precisions: list[float] = []
    for order in range(1, 5):
        denominator = len(candidate_tokens) - order + 1
        if denominator <= 0:
            precisions.append(0.0)
            continue
        candidate_counts = _ngrams(candidate_tokens, order)
        reference_counts = _ngrams(reference_tokens, order)
        clipped = sum(
            min(count, reference_counts[ngram]) for ngram, count in candidate_counts.items()
        )
        precisions.append(clipped / denominator)

    if any(value == 0.0 for value in precisions):
        geometric_mean = 0.0
    else:
        geometric_mean = math.exp(sum(math.log(value) for value in precisions) / 4.0)
    candidate_length = len(candidate_tokens)
    reference_length = len(reference_tokens)
    brevity_penalty = (
        1.0
        if candidate_length >= reference_length
        else math.exp(1.0 - reference_length / candidate_length)
    )
    precision_tuple = cast(tuple[float, float, float, float], tuple(precisions))
    return BleuScore(
        score=brevity_penalty * geometric_mean,
        score_without_brevity_penalty=geometric_mean,
        precisions=precision_tuple,
        brevity_penalty=brevity_penalty,
        candidate_length=candidate_length,
        reference_length=reference_length,
    )


@dataclass(frozen=True, slots=True)
class RougeLScore:
    """Sentence-level ROUGE-L with beta equal to one."""

    precision: float
    recall: float
    f1: float
    lcs_length: int
    candidate_length: int
    reference_length: int

    def __post_init__(self) -> None:
        for name in ("precision", "recall", "f1"):
            _finite_number(getattr(self, name), label=f"ROUGE-L {name}", unit_interval=True)
        if type(self.lcs_length) is not int or self.lcs_length < 0:
            raise ValueError("ROUGE-L lcs_length must be non-negative")
        if type(self.candidate_length) is not int or self.candidate_length < 1:
            raise ValueError("ROUGE-L candidate_length must be positive")
        if type(self.reference_length) is not int or self.reference_length < 1:
            raise ValueError("ROUGE-L reference_length must be positive")
        if self.lcs_length > min(self.candidate_length, self.reference_length):
            raise ValueError("ROUGE-L lcs_length exceeds an input length")


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    # Keep the dynamic-programming row proportional to the shorter sequence.
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> RougeLScore:
    """Compute token LCS precision, recall and F1 for one example."""

    candidate_tokens = _validate_tokens(candidate, label="candidate tokens")
    reference_tokens = _validate_tokens(reference, label="reference tokens")
    lcs_length = _lcs_length(candidate_tokens, reference_tokens)
    precision = lcs_length / len(candidate_tokens)
    recall = lcs_length / len(reference_tokens)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return RougeLScore(
        precision=precision,
        recall=recall,
        f1=f1,
        lcs_length=lcs_length,
        candidate_length=len(candidate_tokens),
        reference_length=len(reference_tokens),
    )


@dataclass(frozen=True, slots=True)
class LexicalExampleScore:
    """Deterministic lexical scores for one generation example."""

    sample_id: str
    bleu: BleuScore
    rouge_l: RougeLScore

    def __post_init__(self) -> None:
        _require_text(self.sample_id, label="lexical score sample_id")

    @property
    def bleu4(self) -> float:
        return self.bleu.score

    @property
    def rouge_l_f1(self) -> float:
        return self.rouge_l.f1


@dataclass(frozen=True, slots=True)
class SemanticExampleScore:
    """Precision, recall and F1 for one semantic-scoring example."""

    sample_id: str
    precision: float
    recall: float
    f1: float

    def __post_init__(self) -> None:
        _require_text(self.sample_id, label="semantic score sample_id")
        for name in ("precision", "recall", "f1"):
            _finite_number(getattr(self, name), label=f"semantic {name}")


# The shorter name is part of the public contract; retain the descriptive alias
# as well because it is useful at call sites and in generated type signatures.
SemanticScore = SemanticExampleScore


@dataclass(frozen=True, slots=True)
class SemanticProvenance:
    """Exact semantic metric/model settings recorded in a report."""

    metric: str = BERTSCORE_METRIC
    package: str = "bert-score"
    distribution_version: str = "injected"
    module_version: str = "injected"
    model: str = BERTSCORE_MODEL
    lang: str = BERTSCORE_LANG
    num_layers: int = BERTSCORE_NUM_LAYERS
    rescale_with_baseline: bool = BERTSCORE_RESCALE_WITH_BASELINE
    idf: bool = BERTSCORE_IDF
    batch_size: int = BERTSCORE_BATCH_SIZE
    use_fast_tokenizer: bool = BERTSCORE_USE_FAST_TOKENIZER

    def __post_init__(self) -> None:
        for name in (
            "metric",
            "package",
            "distribution_version",
            "module_version",
            "model",
            "lang",
        ):
            _require_text(getattr(self, name), label=f"semantic provenance {name}")
        if type(self.num_layers) is not int or self.num_layers < 1:
            raise ValueError("semantic num_layers must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("semantic batch_size must be a positive integer")
        for name in ("rescale_with_baseline", "idf", "use_fast_tokenizer"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"semantic provenance {name} must be bool")


class SemanticScorer(Protocol):
    """Batch semantic scorer; each returned column follows input order."""

    @property
    def provenance(self) -> SemanticProvenance: ...

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> tuple[Sequence[object], Sequence[object], Sequence[object]]: ...


class _BertScoreBackend(Protocol):
    def score(
        self,
        cands: Sequence[str],
        refs: Sequence[str],
        *,
        verbose: bool,
        batch_size: int,
        return_hash: bool,
    ) -> object: ...


class BertScoreAdapter:
    """Adapter whose model construction is possible only through ``load_default``."""

    __slots__ = ("_backend", "_provenance")

    def __init__(
        self,
        backend: _BertScoreBackend,
        *,
        distribution_version: str = "injected",
        module_version: str = "injected",
    ) -> None:
        self._backend = backend
        self._provenance = SemanticProvenance(
            distribution_version=distribution_version,
            module_version=module_version,
        )

    @classmethod
    def load_default(cls) -> BertScoreAdapter:
        """Load and instantiate the optional model-backed BERTScore implementation."""

        try:
            module = importlib.import_module("bert_score")
            distribution_version = importlib.metadata.version("bert-score")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:  # pragma: no cover
            raise ImportError(
                "bert-score is required for semantic generation metrics; "
                "install it with `uv sync --extra verify`"
            ) from exc
        scorer_class = cast(Callable[..., _BertScoreBackend], module.BERTScorer)
        backend = scorer_class(
            model_type=BERTSCORE_MODEL,
            num_layers=BERTSCORE_NUM_LAYERS,
            lang=BERTSCORE_LANG,
            batch_size=BERTSCORE_BATCH_SIZE,
            idf=BERTSCORE_IDF,
            rescale_with_baseline=BERTSCORE_RESCALE_WITH_BASELINE,
            use_fast_tokenizer=BERTSCORE_USE_FAST_TOKENIZER,
        )
        return cls(
            backend,
            distribution_version=distribution_version,
            module_version=str(getattr(module, "__version__", "unknown")),
        )

    @property
    def provenance(self) -> SemanticProvenance:
        return self._provenance

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> tuple[Sequence[object], Sequence[object], Sequence[object]]:
        raw = self._backend.score(
            list(predictions),
            list(references),
            verbose=False,
            batch_size=BERTSCORE_BATCH_SIZE,
            return_hash=False,
        )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise ValueError("BERTScore backend must return precision, recall and F1")
        columns = tuple(
            _as_sequence(part, label=label)
            for part, label in zip(raw, ("precision", "recall", "F1"), strict=True)
        )
        return cast(tuple[Sequence[object], Sequence[object], Sequence[object]], columns)


def _as_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, (str, bytes, Sequence)):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = cast(Callable[[], object], tolist)()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"semantic {label} output must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class GenerationMetricReport:
    """Validated lexical report and optional semantic report."""

    schema: str
    sample_count: int
    tokenizer: TokenizerProvenance
    lexical_scores: tuple[LexicalExampleScore, ...]
    mean_sentence_bleu4: float
    crud_mean_sentence_bleu4_no_bp: float
    mean_sentence_rouge_l_precision: float
    mean_sentence_rouge_l_recall: float
    mean_sentence_rouge_l_f1: float
    semantic_provenance: SemanticProvenance | None = None
    semantic_scores: tuple[SemanticScore, ...] = ()
    mean_bertscore_precision_zh_rescaled: float | None = None
    mean_bertscore_recall_zh_rescaled: float | None = None
    mean_bertscore_f1_zh_rescaled: float | None = None

    def __post_init__(self) -> None:
        if self.schema != GENERATION_METRICS_SCHEMA:
            raise ValueError(f"unsupported generation metric schema {self.schema!r}")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        if len(self.lexical_scores) != self.sample_count:
            raise ValueError("lexical score count does not match sample_count")
        lexical_ids = tuple(row.sample_id for row in self.lexical_scores)
        if len(set(lexical_ids)) != self.sample_count:
            raise ValueError("lexical score sample ids must be unique")
        for name in (
            "mean_sentence_bleu4",
            "crud_mean_sentence_bleu4_no_bp",
            "mean_sentence_rouge_l_precision",
            "mean_sentence_rouge_l_recall",
            "mean_sentence_rouge_l_f1",
        ):
            _finite_number(getattr(self, name), label=name, unit_interval=True)

        semantic_means = (
            self.mean_bertscore_precision_zh_rescaled,
            self.mean_bertscore_recall_zh_rescaled,
            self.mean_bertscore_f1_zh_rescaled,
        )
        if self.semantic_provenance is None:
            if self.semantic_scores or any(value is not None for value in semantic_means):
                raise ValueError("semantic values require semantic provenance")
            return
        if len(self.semantic_scores) != self.sample_count:
            raise ValueError("semantic score count does not match sample_count")
        if tuple(row.sample_id for row in self.semantic_scores) != lexical_ids:
            raise ValueError("semantic score order must match lexical score order")
        if any(value is None for value in semantic_means):
            raise ValueError("semantic reports require all three aggregate means")
        for name, value in zip(
            (
                "mean_bertscore_precision_zh_rescaled",
                "mean_bertscore_recall_zh_rescaled",
                "mean_bertscore_f1_zh_rescaled",
            ),
            semantic_means,
            strict=True,
        ):
            _finite_number(value, label=name)

    @property
    def mean_bleu4(self) -> float:
        return self.mean_sentence_bleu4

    @property
    def mean_rouge_l_f1(self) -> float:
        return self.mean_sentence_rouge_l_f1


def _semantic_rows(
    examples: Sequence[GenerationExample],
    scorer: SemanticScorer,
) -> tuple[SemanticScore, ...]:
    raw = scorer.score(
        [example.prediction for example in examples],
        [example.reference for example in examples],
    )
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise ValueError("semantic scorer must return precision, recall and F1")
    columns = tuple(
        _as_sequence(values, label=label)
        for values, label in zip(raw, ("precision", "recall", "F1"), strict=True)
    )
    for label, values in zip(("precision", "recall", "F1"), columns, strict=True):
        if len(values) != len(examples):
            raise ValueError(
                f"semantic {label} output has {len(values)} rows; expected {len(examples)}"
            )
    rows: list[SemanticScore] = []
    for index, example in enumerate(examples):
        rows.append(
            SemanticScore(
                sample_id=example.sample_id,
                precision=_finite_number(
                    columns[0][index],
                    label=f"{example.sample_id}: semantic precision",
                ),
                recall=_finite_number(
                    columns[1][index],
                    label=f"{example.sample_id}: semantic recall",
                ),
                f1=_finite_number(
                    columns[2][index],
                    label=f"{example.sample_id}: semantic F1",
                ),
            )
        )
    return tuple(rows)


def evaluate_generation(
    examples: Sequence[GenerationExample],
    tokenizer: Tokenizer,
    *,
    semantic_scorer: SemanticScorer | None = None,
) -> GenerationMetricReport:
    """Evaluate every row, rejecting the complete report on any invalid row."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("generation evaluation requires a non-empty sequence of examples")
    checked_examples: list[GenerationExample] = []
    for index, example in enumerate(examples):
        if not isinstance(example, GenerationExample):
            raise ValueError(f"examples[{index}] must be a GenerationExample")
        checked_examples.append(example)
    if len({example.sample_id for example in checked_examples}) != len(checked_examples):
        raise ValueError("generation example sample ids must be unique")

    lexical_rows: list[LexicalExampleScore] = []
    for example in checked_examples:
        candidate = tokenize_text(
            example.prediction,
            tokenizer,
            label=f"{example.sample_id}: prediction",
        )
        reference = tokenize_text(
            example.reference,
            tokenizer,
            label=f"{example.sample_id}: reference",
        )
        lexical_rows.append(
            LexicalExampleScore(
                sample_id=example.sample_id,
                bleu=sentence_bleu4(candidate, reference),
                rouge_l=rouge_l(candidate, reference),
            )
        )

    lexical = tuple(lexical_rows)
    semantic = () if semantic_scorer is None else _semantic_rows(checked_examples, semantic_scorer)
    return GenerationMetricReport(
        schema=GENERATION_METRICS_SCHEMA,
        sample_count=len(checked_examples),
        tokenizer=tokenizer.provenance,
        lexical_scores=lexical,
        mean_sentence_bleu4=math.fsum(row.bleu.score for row in lexical) / len(lexical),
        crud_mean_sentence_bleu4_no_bp=(
            math.fsum(row.bleu.score_without_brevity_penalty for row in lexical) / len(lexical)
        ),
        mean_sentence_rouge_l_precision=(
            math.fsum(row.rouge_l.precision for row in lexical) / len(lexical)
        ),
        mean_sentence_rouge_l_recall=(
            math.fsum(row.rouge_l.recall for row in lexical) / len(lexical)
        ),
        mean_sentence_rouge_l_f1=math.fsum(row.rouge_l.f1 for row in lexical) / len(lexical),
        semantic_provenance=None if semantic_scorer is None else semantic_scorer.provenance,
        semantic_scores=semantic,
        mean_bertscore_precision_zh_rescaled=(
            None if not semantic else math.fsum(row.precision for row in semantic) / len(semantic)
        ),
        mean_bertscore_recall_zh_rescaled=(
            None if not semantic else math.fsum(row.recall for row in semantic) / len(semantic)
        ),
        mean_bertscore_f1_zh_rescaled=(
            None if not semantic else math.fsum(row.f1 for row in semantic) / len(semantic)
        ),
    )
