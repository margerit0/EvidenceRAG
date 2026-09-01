from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import zhrag.eval.metrics_gen as metrics_gen_module
from zhrag.eval.metrics_gen import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_IDF,
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_RESCALE_WITH_BASELINE,
    BERTSCORE_USE_FAST_TOKENIZER,
    GENERATION_METRICS_SCHEMA,
    BertScoreAdapter,
    BleuScore,
    GenerationExample,
    SemanticProvenance,
    TokenizerProvenance,
    evaluate_generation,
    rouge_l,
    sentence_bleu4,
)


class SpaceTokenizer:
    @property
    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(
            name="synthetic-space",
            package="tests",
            version="1",
        )

    def tokenize(self, text: str) -> Sequence[str]:
        return text.split()


class FixedSemanticScorer:
    def __init__(
        self,
        scores: tuple[Sequence[object], Sequence[object], Sequence[object]],
    ) -> None:
        self._scores = scores
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    @property
    def provenance(self) -> SemanticProvenance:
        return SemanticProvenance(
            distribution_version="fake-dist",
            module_version="fake-module",
        )

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> tuple[Sequence[object], Sequence[object], Sequence[object]]:
        self.calls.append((tuple(predictions), tuple(references)))
        return self._scores


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeBertBackend:
    def __init__(self) -> None:
        self.call: dict[str, object] | None = None

    def score(
        self,
        cands: Sequence[str],
        refs: Sequence[str],
        *,
        verbose: bool,
        batch_size: int,
        return_hash: bool,
    ) -> object:
        self.call = {
            "cands": tuple(cands),
            "refs": tuple(refs),
            "verbose": verbose,
            "batch_size": batch_size,
            "return_hash": return_hash,
        }
        return FakeTensor([-0.2, 0.7]), FakeTensor([-0.1, 0.5]), FakeTensor([-0.15, 0.6])


EXAMPLES = (
    GenerationExample("s1", "春 风 吹 过 湖 面", "春 风 吹 过 湖 面"),
    GenerationExample("s2", "夜 色 照亮 石 桥", "夜 色 照亮 长 桥"),
)


class TestSentenceBleu4:
    def test_exact_and_disjoint_sequences(self) -> None:
        exact = sentence_bleu4(("甲", "乙", "丙", "丁"), ("甲", "乙", "丙", "丁"))
        disjoint = sentence_bleu4(("甲", "乙", "丙", "丁"), ("戊", "己", "庚", "辛"))

        assert exact.score == 1.0
        assert exact.score_without_brevity_penalty == 1.0
        assert exact.precisions == (1.0, 1.0, 1.0, 1.0)
        assert disjoint.score == 0.0
        assert disjoint.precisions == (0.0, 0.0, 0.0, 0.0)

    def test_clips_repeated_ngram_counts(self) -> None:
        score = sentence_bleu4(
            ("甲", "甲", "甲", "甲", "乙", "丙", "丁"),
            ("甲", "甲", "甲", "乙", "丙", "丁", "戊"),
        )

        assert score.precisions == pytest.approx((6 / 7, 5 / 6, 4 / 5, 3 / 4))
        assert score.score == pytest.approx(math.prod(score.precisions) ** 0.25)

    def test_standard_score_keeps_brevity_penalty(self) -> None:
        score = sentence_bleu4(
            ("甲", "乙", "丙", "丁"),
            ("甲", "乙", "丙", "丁", "戊"),
        )

        assert score.precisions == (1.0, 1.0, 1.0, 1.0)
        assert score.brevity_penalty == pytest.approx(math.exp(-0.25))
        assert score.score == pytest.approx(math.exp(-0.25))
        assert score.score_without_brevity_penalty == 1.0

    def test_short_candidate_has_no_smoothing_or_effective_order(self) -> None:
        score = sentence_bleu4(("甲", "乙", "丙"), ("甲", "乙", "丙"))

        assert score.precisions == (1.0, 1.0, 1.0, 0.0)
        assert score.score == 0.0
        assert score.score_without_brevity_penalty == 0.0

    @pytest.mark.parametrize("tokens", [(), ("甲", ""), ("甲", " ")])
    def test_rejects_invalid_token_sequences(self, tokens: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="token"):
            sentence_bleu4(tokens, ("甲", "乙", "丙", "丁"))


class TestRougeL:
    def test_is_order_sensitive(self) -> None:
        score = rouge_l(("甲", "乙", "丙"), ("甲", "丙", "乙"))

        assert score.lcs_length == 2
        assert score.precision == pytest.approx(2 / 3)
        assert score.recall == pytest.approx(2 / 3)
        assert score.f1 == pytest.approx(2 / 3)

    def test_reports_precision_recall_and_beta_one_f1(self) -> None:
        score = rouge_l(("甲", "乙"), ("甲", "乙", "丙", "丁"))

        assert score.precision == 1.0
        assert score.recall == 0.5
        assert score.f1 == pytest.approx(2 / 3)

    def test_no_common_subsequence_is_zero(self) -> None:
        assert rouge_l(("甲",), ("乙",)).f1 == 0.0


class TestGenerationReport:
    def test_preserves_order_and_uses_arithmetic_sentence_means(self) -> None:
        report = evaluate_generation(EXAMPLES, SpaceTokenizer())
        expected_bleu = [
            sentence_bleu4(example.prediction.split(), example.reference.split()).score
            for example in EXAMPLES
        ]
        expected_rouge = [
            rouge_l(example.prediction.split(), example.reference.split()).f1
            for example in EXAMPLES
        ]

        assert report.schema == GENERATION_METRICS_SCHEMA
        assert report.sample_count == 2
        assert tuple(row.sample_id for row in report.lexical_scores) == ("s1", "s2")
        assert report.mean_sentence_bleu4 == pytest.approx(sum(expected_bleu) / 2)
        assert report.mean_sentence_rouge_l_f1 == pytest.approx(sum(expected_rouge) / 2)
        assert report.semantic_provenance is None
        assert report.semantic_scores == ()

    def test_semantic_scores_are_batched_and_negative_rescaled_values_are_valid(self) -> None:
        scorer = FixedSemanticScorer(([-0.2, 0.8], [-0.1, 0.6], [-0.15, 0.7]))
        report = evaluate_generation(EXAMPLES, SpaceTokenizer(), semantic_scorer=scorer)

        assert scorer.calls == [
            (
                tuple(example.prediction for example in EXAMPLES),
                tuple(example.reference for example in EXAMPLES),
            )
        ]
        assert tuple(row.sample_id for row in report.semantic_scores) == ("s1", "s2")
        assert report.mean_bertscore_precision_zh_rescaled == pytest.approx(0.3)
        assert report.mean_bertscore_recall_zh_rescaled == pytest.approx(0.25)
        assert report.mean_bertscore_f1_zh_rescaled == pytest.approx(0.275)

    @pytest.mark.parametrize(
        "values",
        [
            ([0.1], [0.1, 0.2], [0.1, 0.2]),
            ([0.1, 0.2, 0.3], [0.1, 0.2], [0.1, 0.2]),
        ],
    )
    def test_rejects_semantic_row_count_mismatch(
        self,
        values: tuple[list[float], list[float], list[float]],
    ) -> None:
        scorer = FixedSemanticScorer(values)
        with pytest.raises(ValueError, match=r"rows; expected 2"):
            evaluate_generation(EXAMPLES, SpaceTokenizer(), semantic_scorer=scorer)

    @pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_numeric_or_non_finite_semantic_values(self, bad: object) -> None:
        scorer = FixedSemanticScorer(([bad, 0.2], [0.1, 0.2], [0.1, 0.2]))
        with pytest.raises(ValueError, match=r"real number|finite"):
            evaluate_generation(EXAMPLES, SpaceTokenizer(), semantic_scorer=scorer)

    def test_rejects_duplicate_ids(self) -> None:
        duplicate = (
            GenerationExample("same", "甲 乙 丙 丁", "甲 乙 丙 丁"),
            GenerationExample("same", "戊 己 庚 辛", "戊 己 庚 辛"),
        )
        with pytest.raises(ValueError, match="unique"):
            evaluate_generation(duplicate, SpaceTokenizer())

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_generation((), SpaceTokenizer())

    @pytest.mark.parametrize(
        ("sample_id", "prediction", "reference"),
        [("", "甲", "乙"), ("s", " ", "乙"), ("s", "甲", "\t")],
    )
    def test_rejects_blank_example_fields(
        self,
        sample_id: str,
        prediction: str,
        reference: str,
    ) -> None:
        with pytest.raises(ValueError, match="non-blank"):
            GenerationExample(sample_id, prediction, reference)

    def test_tokenizer_failure_aborts_the_report(self) -> None:
        class BrokenTokenizer(SpaceTokenizer):
            def tokenize(self, text: str) -> Sequence[str]:
                raise RuntimeError("synthetic tokenizer failure")

        with pytest.raises(RuntimeError, match="synthetic tokenizer failure"):
            evaluate_generation(EXAMPLES, BrokenTokenizer())

    def test_blank_token_aborts_the_report(self) -> None:
        class BlankTokenizer(SpaceTokenizer):
            def tokenize(self, text: str) -> Sequence[str]:
                return ("甲", " ")

        with pytest.raises(ValueError, match="non-blank"):
            evaluate_generation(EXAMPLES, BlankTokenizer())

    def test_lexical_result_objects_reject_out_of_range_values(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            BleuScore(1.1, 1.0, (1.0, 1.0, 1.0, 1.0), 1.0, 4, 4)


class TestBertScoreAdapter:
    def test_injected_backend_preserves_order_and_fixed_runtime_settings(self) -> None:
        backend = FakeBertBackend()
        adapter = BertScoreAdapter(
            backend,
            distribution_version="0.3.13",
            module_version="0.3.12",
        )

        precision, recall, f1 = adapter.score(
            ["春风吹过湖面", "夜色照亮石桥"],
            ["春风掠过湖面", "灯火照亮石桥"],
        )

        assert backend.call == {
            "cands": ("春风吹过湖面", "夜色照亮石桥"),
            "refs": ("春风掠过湖面", "灯火照亮石桥"),
            "verbose": False,
            "batch_size": 64,
            "return_hash": False,
        }
        assert tuple(precision) == (-0.2, 0.7)
        assert tuple(recall) == (-0.1, 0.5)
        assert tuple(f1) == (-0.15, 0.6)
        assert adapter.provenance.distribution_version == "0.3.13"
        assert adapter.provenance.module_version == "0.3.12"
        assert adapter.provenance.model == BERTSCORE_MODEL
        assert adapter.provenance.num_layers == BERTSCORE_NUM_LAYERS

    def test_load_default_constructs_only_the_frozen_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        constructed: dict[str, object] = {}
        backend = FakeBertBackend()

        def fake_scorer(**kwargs: object) -> FakeBertBackend:
            constructed.update(kwargs)
            return backend

        module = SimpleNamespace(BERTScorer=fake_scorer, __version__="0.3.12")
        monkeypatch.setattr(
            metrics_gen_module.importlib,
            "import_module",
            lambda name: module if name == "bert_score" else pytest.fail(name),
        )
        monkeypatch.setattr(
            metrics_gen_module.importlib.metadata,
            "version",
            lambda name: "0.3.13" if name == "bert-score" else pytest.fail(name),
        )

        adapter = BertScoreAdapter.load_default()

        assert constructed == {
            "model_type": BERTSCORE_MODEL,
            "num_layers": BERTSCORE_NUM_LAYERS,
            "lang": BERTSCORE_LANG,
            "batch_size": BERTSCORE_BATCH_SIZE,
            "idf": BERTSCORE_IDF,
            "rescale_with_baseline": BERTSCORE_RESCALE_WITH_BASELINE,
            "use_fast_tokenizer": BERTSCORE_USE_FAST_TOKENIZER,
        }
        assert adapter.provenance.distribution_version == "0.3.13"
        assert adapter.provenance.module_version == "0.3.12"

    def test_rejects_malformed_backend_result(self) -> None:
        class BadBackend(FakeBertBackend):
            def score(
                self,
                cands: Sequence[str],
                refs: Sequence[str],
                *,
                verbose: bool,
                batch_size: int,
                return_hash: bool,
            ) -> object:
                return [0.1], [0.2]

        with pytest.raises(ValueError, match="precision, recall and F1"):
            BertScoreAdapter(BadBackend()).score(["甲"], ["乙"])


def test_eval_package_import_does_not_load_optional_model_packages() -> None:
    script = """
import sys
import zhrag.eval
for name in ('jieba', 'bert_score', 'torch', 'transformers'):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
