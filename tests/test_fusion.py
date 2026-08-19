"""Tests for reciprocal rank fusion.

RRF is about to be used to decide whether M4 (hybrid retrieval) is worth
building, so its behaviour under the cases that decide that -- one arm confident
and alone, both arms agreeing, one arm's tail -- is pinned here rather than
inferred from the ablation numbers it produces.
"""

from __future__ import annotations

import pytest

from zhrag.retrieval import reciprocal_rank_fusion


class TestBasics:
    def test_a_single_run_is_returned_in_order(self) -> None:
        assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_unanimous_runs_keep_their_order(self) -> None:
        run = ["a", "b", "c"]
        assert reciprocal_rank_fusion([run, run]) == run

    def test_union_of_both_runs_is_returned(self) -> None:
        assert set(reciprocal_rank_fusion([["a", "b"], ["c", "d"]])) == {"a", "b", "c", "d"}

    def test_agreement_beats_one_strong_vote(self) -> None:
        """The property the whole method rests on.

        'x' is ranked 1st by one system and unseen by the other; 'y' is 3rd in
        both. At k=60 the two third places outweigh the single first place,
        which is what makes RRF a consensus rule rather than a max.

        Third and not second on purpose: ``1/(k+1) > 2/(k+2)`` has no solution
        for non-negative k, so a document agreed at rank 2 wins at *every*
        setting and the example would demonstrate nothing about k.
        """
        fused = reciprocal_rank_fusion([["x", "a", "y"], ["z", "b", "y"]], k=60)
        assert fused[0] == "y"

    def test_small_k_lets_a_confident_single_vote_win(self) -> None:
        """Same inputs, k=0: one first place now outweighs two third places."""
        fused = reciprocal_rank_fusion([["x", "a", "y"], ["z", "b", "y"]], k=0)
        assert fused[0] in {"x", "z"}
        assert fused.index("y") > 0

    def test_ties_break_on_document_id_not_dict_order(self) -> None:
        assert reciprocal_rank_fusion([["b", "a"], ["a", "b"]]) == ["a", "b"]

    @pytest.mark.parametrize("k", [0, 1, 5, 60])
    def test_mathematically_tied_documents_break_on_id_at_any_k(self, k: int) -> None:
        """Three runs, each document holding ranks 1, 2 and 3 exactly once.

        Every fused score is the same real number, so the documented doc_id
        tie-break must decide the order. A naive running ``+=`` fails this:
        float addition is not associative, the three sums land one ULP apart,
        and the ordering silently becomes a function of rounding. Two runs
        cannot expose it -- a third arm, which is what adding a reranker gives,
        can.
        """
        runs = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
        assert reciprocal_rank_fusion(runs, k=k) == ["a", "b", "c"]


class TestWeightsAndDepth:
    def test_weights_can_override_a_rank_advantage(self) -> None:
        runs = [["a", "b"], ["b", "a"]]
        assert reciprocal_rank_fusion(runs, weights=[3.0, 1.0])[0] == "a"
        assert reciprocal_rank_fusion(runs, weights=[1.0, 3.0])[0] == "b"

    def test_only_weight_ratios_matter(self) -> None:
        runs = [["a", "c"], ["b", "c"]]
        assert reciprocal_rank_fusion(runs, weights=[2.0, 6.0]) == reciprocal_rank_fusion(
            runs, weights=[1.0, 3.0]
        )

    def test_depth_truncates_every_run_before_fusing(self) -> None:
        runs = [["a", "b", "c"], ["d", "e", "f"]]
        assert set(reciprocal_rank_fusion(runs, depth=1)) == {"a", "d"}

    def test_depth_can_change_the_winner(self) -> None:
        """A document deep in both runs beats a shallow one only if depth reaches it."""
        runs = [["x", "y", "z"], ["w", "v", "z"]]
        assert reciprocal_rank_fusion(runs, k=1, depth=2)[0] in {"x", "w"}
        assert "z" not in reciprocal_rank_fusion(runs, depth=2)

    def test_a_repeated_id_counts_only_at_its_best_rank(self) -> None:
        """Otherwise a duplicated id would accumulate score and float to the top."""
        assert reciprocal_rank_fusion([["a", "a", "a", "b"]], k=0) == ["a", "b"]


class TestValidation:
    def test_rejects_no_runs(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            reciprocal_rank_fusion([])

    def test_rejects_negative_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            reciprocal_rank_fusion([["a"]], k=-1)

    def test_rejects_nonpositive_depth(self) -> None:
        with pytest.raises(ValueError, match="depth must be"):
            reciprocal_rank_fusion([["a"]], depth=0)

    def test_rejects_weight_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_empty_runs_fuse_to_nothing(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []
