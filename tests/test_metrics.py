"""Tests for the IR metrics.

These are the numbers the whole project is judged on, so the multi-gold cases --
where the eval libraries disagree with each other -- are pinned explicitly.
"""

from __future__ import annotations

import math

import pytest

from zhrag.eval import (
    all_gold_at_k,
    bootstrap_ci,
    evaluate,
    hit_at_k,
    holm_bonferroni,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_test,
    recall_at_k,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]


class TestRecall:
    def test_single_gold_hit_at_rank_1(self) -> None:
        assert recall_at_k(RANKED, ["d1"], 1) == 1.0

    def test_single_gold_miss_outside_k(self) -> None:
        assert recall_at_k(RANKED, ["d4"], 3) == 0.0

    def test_multi_gold_gives_partial_credit(self) -> None:
        # This is the case the libraries disagree on: 2 of 3 gold inside top-3.
        assert recall_at_k(RANKED, ["d1", "d2", "d9"], 3) == pytest.approx(2 / 3)

    def test_gold_absent_from_ranking_entirely(self) -> None:
        assert recall_at_k(RANKED, ["missing"], 5) == 0.0

    def test_empty_gold_is_an_error_not_a_zero(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            recall_at_k(RANKED, [], 5)

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_nonpositive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            recall_at_k(RANKED, ["d1"], k)


class TestAllGoldAndHit:
    def test_all_gold_requires_every_document(self) -> None:
        assert all_gold_at_k(RANKED, ["d1", "d2"], 2) == 1.0
        assert all_gold_at_k(RANKED, ["d1", "d3"], 2) == 0.0

    def test_hit_needs_only_one(self) -> None:
        assert hit_at_k(RANKED, ["d1", "d9"], 2) == 1.0
        assert hit_at_k(RANKED, ["d9"], 5) == 0.0

    def test_partial_recall_overstates_multidoc_success(self) -> None:
        """The reason all_gold_at_k exists: recall looks like success, it is not."""
        gold = ["d1", "d2", "d9"]
        assert recall_at_k(RANKED, gold, 3) > 0.5
        assert all_gold_at_k(RANKED, gold, 3) == 0.0


class TestMRR:
    @pytest.mark.parametrize(
        ("gold", "expected"),
        [(["d1"], 1.0), (["d2"], 0.5), (["d3"], 1 / 3), (["d9"], 0.0)],
    )
    def test_reciprocal_of_first_gold_rank(self, gold: list[str], expected: float) -> None:
        assert mrr_at_k(RANKED, gold, 10) == pytest.approx(expected)

    def test_uses_only_the_first_gold_hit(self) -> None:
        assert mrr_at_k(RANKED, ["d2", "d3"], 10) == pytest.approx(0.5)

    def test_truncates_at_k(self) -> None:
        assert mrr_at_k(RANKED, ["d3"], 2) == 0.0


class TestNDCG:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(RANKED, ["d1"], 10) == pytest.approx(1.0)
        assert ndcg_at_k(RANKED, ["d1", "d2"], 10) == pytest.approx(1.0)

    def test_demoted_gold_discounts_by_log2(self) -> None:
        # gold at rank 2 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        assert ndcg_at_k(RANKED, ["d2"], 10) == pytest.approx(1 / math.log2(3))

    def test_ideal_accounts_for_multiple_gold(self) -> None:
        # 2 gold at ranks 1 and 3 against an ideal of ranks 1 and 2
        dcg = 1 + 1 / math.log2(4)
        idcg = 1 + 1 / math.log2(3)
        assert ndcg_at_k(RANKED, ["d1", "d3"], 10) == pytest.approx(dcg / idcg)

    def test_bounded_to_unit_interval(self) -> None:
        assert 0.0 <= ndcg_at_k(RANKED, ["d5"], 5) <= 1.0


class TestBootstrap:
    def test_interval_brackets_the_mean(self) -> None:
        ci = bootstrap_ci([1.0] * 60 + [0.0] * 40, resamples=2000)
        assert ci.mean == pytest.approx(0.6)
        assert ci.low < ci.mean < ci.high
        assert ci.n == 100

    def test_is_deterministic_under_a_fixed_seed(self) -> None:
        scores = [0.0, 1.0] * 50
        assert bootstrap_ci(scores, resamples=500) == bootstrap_ci(scores, resamples=500)

    def test_zero_variance_collapses_the_interval(self) -> None:
        ci = bootstrap_ci([1.0] * 30, resamples=500)
        assert ci.low == ci.high == pytest.approx(1.0)

    def test_more_data_narrows_the_interval(self) -> None:
        narrow = bootstrap_ci([1.0, 0.0] * 500, resamples=2000)
        wide = bootstrap_ci([1.0, 0.0] * 25, resamples=2000)
        assert (narrow.high - narrow.low) < (wide.high - wide.low)


class TestPairedBootstrap:
    def test_consistent_improvement_is_significant(self) -> None:
        baseline = [0.0] * 100
        treatment = [1.0] * 100
        assert paired_bootstrap_test(baseline, treatment, resamples=2000) < 0.01

    def test_no_improvement_returns_one(self) -> None:
        scores = [0.5] * 50
        assert paired_bootstrap_test(scores, scores, resamples=500) == 1.0

    def test_regression_returns_one(self) -> None:
        assert paired_bootstrap_test([1.0] * 50, [0.0] * 50, resamples=500) == 1.0

    def test_bidirectional_noise_swamps_a_small_net_gain(self) -> None:
        """A +1pp net gain is not significant once queries move in both directions.

        This is the realistic ablation shape: a new reranker fixes some queries
        and breaks others. Contrast with
        :meth:`test_small_but_perfectly_consistent_gain_is_significant` -- the
        same +1pp *is* detectable when nothing regresses, which is exactly why
        the paired test is the right one to use here.
        """
        baseline = [float(i % 2) for i in range(500)]
        treatment = list(baseline)
        for i in range(0, 120, 4):  # 30 queries improve
            treatment[i] = 1.0
        for i in range(201, 300, 4):  # 25 queries regress -> net +1pp
            treatment[i] = 0.0
        assert paired_bootstrap_test(baseline, treatment, resamples=4000) > 0.05

    def test_small_but_perfectly_consistent_gain_is_significant(self) -> None:
        baseline = [float(i % 2) for i in range(500)]
        treatment = list(baseline)
        for i in range(0, 500, 100):  # 5 queries improve, none regress
            treatment[i] = 1.0
        assert paired_bootstrap_test(baseline, treatment, resamples=4000) < 0.05

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            paired_bootstrap_test([1.0], [1.0, 0.0])


class TestHolmBonferroni:
    def test_uncorrected_significance_can_vanish(self) -> None:
        """The whole point: two arms look significant until the family is corrected."""
        raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
        assert sum(1 for p in raw.values() if p <= 0.05) == 3
        out = holm_bonferroni(raw)
        assert sum(1 for _, rejected in out.values() if rejected) == 1
        assert out["a"] == (pytest.approx(0.004), True)

    def test_adjusted_p_is_monotone_non_decreasing(self) -> None:
        out = holm_bonferroni({f"m{i}": p for i, p in enumerate([0.01, 0.011, 0.012, 0.9])})
        adjusted = [out[f"m{i}"][0] for i in range(4)]
        assert adjusted == sorted(adjusted)

    def test_ordering_is_by_pvalue_not_insertion(self) -> None:
        """Holm sorts ascending, so a tiny p-value inserted last is still tested early."""
        out = holm_bonferroni({"a": 0.001, "big": 0.99, "c": 0.0011})
        assert out["a"][1] is True
        assert out["c"][1] is True  # tested 2nd despite being inserted last
        assert out["big"][1] is False

    def test_step_down_halts_the_whole_family(self) -> None:
        """If the smallest p-value fails its threshold, nothing after it can pass.

        All four p-values clear a naive 0.05 individually; none survive as a
        family. This is the false-discovery case the correction exists for.
        """
        raw = {"a": 0.02, "b": 0.03, "c": 0.04, "d": 0.045}
        assert all(p <= 0.05 for p in raw.values())
        out = holm_bonferroni(raw)
        assert not any(rejected for _, rejected in out.values())

    def test_single_comparison_is_unchanged(self) -> None:
        assert holm_bonferroni({"only": 0.03}) == {"only": (pytest.approx(0.03), True)}

    def test_adjusted_p_is_capped_at_one(self) -> None:
        out = holm_bonferroni({f"m{i}": 0.5 for i in range(10)})
        assert all(p <= 1.0 for p, _ in out.values())

    def test_more_comparisons_penalise_harder(self) -> None:
        few = holm_bonferroni({"a": 0.02, "b": 0.9})
        many = holm_bonferroni({"a": 0.02, **{f"x{i}": 0.9 for i in range(20)}})
        assert few["a"][0] < many["a"][0]

    def test_empty_family(self) -> None:
        assert holm_bonferroni({}) == {}

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_rejects_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            holm_bonferroni({"a": 0.01}, alpha=alpha)

    def test_integrates_with_paired_bootstrap(self) -> None:
        base = [float(i % 2) for i in range(200)]
        strong = [1.0] * 200
        null = list(base)
        raw = {
            "strong": paired_bootstrap_test(base, strong, resamples=1000),
            "null": paired_bootstrap_test(base, null, resamples=1000),
        }
        out = holm_bonferroni(raw)
        assert out["strong"][1] is True
        assert out["null"][1] is False


class TestEvaluate:
    def test_scores_every_default_metric(self) -> None:
        out = evaluate({"q1": RANKED}, {"q1": ["d1"]})
        assert set(out) == {"R@1", "MRR@10", "nDCG@10"}
        assert out["R@1"].mean == pytest.approx(1.0)

    def test_missing_query_scores_zero_rather_than_being_dropped(self) -> None:
        out = evaluate({"q1": RANKED}, {"q1": ["d1"], "q2": ["d1"]})
        assert out["R@1"].mean == pytest.approx(0.5)
        assert out["R@1"].n == 2
