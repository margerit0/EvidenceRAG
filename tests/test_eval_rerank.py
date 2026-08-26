"""Tests for rerank-window semantics and cache provenance fingerprints."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.eval.crud import Query
from zhrag.eval.rerank import (
    candidate_run_fingerprint,
    missing_score_queries,
    paired_metric_family,
    rerank_input_fingerprint,
    rerank_prefix,
)
from zhrag.io_utils import read_text


def _query(query_id: str = "q", question: str = "question") -> Query:
    return Query(query_id, question, "answer", ("d0",), "task")


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_rerank.py"
    spec = importlib.util.spec_from_file_location("evaluate_rerank_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPairedMetricFamily:
    def test_builds_every_declared_contrast_in_treatment_direction(self) -> None:
        scored = {
            arity: {
                "baseline": {
                    "hit@1": [0.0, 1.0],
                    "ALL@10": [0.0, 1.0],
                },
                "top50": {
                    "hit@1": [1.0, 1.0],
                    "ALL@10": [1.0, 1.0],
                },
                "top100": {
                    "hit@1": [1.0, 0.0],
                    "ALL@10": [1.0, 0.0],
                },
            }
            for arity in (1, 2, 3)
        }
        rows = paired_metric_family(
            scored,
            comparisons=(("baseline", "top50"), ("baseline", "top100")),
            metrics=("hit@1", "ALL@10"),
            binary=True,
            resamples=20,
        )

        assert len(rows) == 12
        top50 = next(
            row
            for row in rows
            if row.arity == 1 and row.treatment == "top50" and row.metric == "hit@1"
        )
        top100 = next(
            row
            for row in rows
            if row.arity == 1 and row.treatment == "top100" and row.metric == "hit@1"
        )
        assert (top50.delta, top50.counts.wins, top50.counts.losses) == (0.5, 1, 0)
        assert (top100.delta, top100.counts.wins, top100.counts.losses) == (0.0, 1, 1)

    def test_marks_raw_and_holm_adjusted_bootstrap_floor_provenance(self) -> None:
        scored = {
            1: {
                "baseline": {"MRR@10": [0.0, 0.0], "nDCG@10": [0.0, 0.0]},
                "treatment": {"MRR@10": [1.0, 1.0], "nDCG@10": [1.0, 1.0]},
            }
        }
        rows = paired_metric_family(
            scored,
            comparisons=(("baseline", "treatment"),),
            metrics=("MRR@10", "nDCG@10"),
            binary=False,
            resamples=99,
        )

        assert len(rows) == 2
        assert [row.raw_p for row in rows] == pytest.approx([0.01, 0.01])
        assert [row.adjusted_p for row in rows] == pytest.approx([0.02, 0.02])
        assert all(row.raw_p_at_floor for row in rows)
        assert all(row.adjusted_p_inherits_floor for row in rows)

    def test_holm_ceiling_does_not_inherit_floor_provenance(self) -> None:
        scored = {
            1: {
                "baseline": {"MRR@10": [0.0]},
                "first": {"MRR@10": [1.0]},
                "second": {"MRR@10": [1.0]},
            }
        }
        rows = paired_metric_family(
            scored,
            comparisons=(("baseline", "first"), ("baseline", "second")),
            metrics=("MRR@10",),
            binary=False,
            resamples=1,
        )

        assert all(row.raw_p_at_floor for row in rows)
        assert all(row.adjusted_p == 1.0 for row in rows)
        assert not any(row.adjusted_p_inherits_floor for row in rows)

    def test_runner_formats_floor_as_an_estimate_marker_not_an_upper_bound(self) -> None:
        runner = _runner()

        assert runner._fmt_p(0.01, at_floor=True) == "0.0100†"
        assert "<" not in runner._fmt_p(0.01, at_floor=True)
        assert runner._fmt_p(0.01, at_floor=False) == "0.0100"

    def test_floor_marker_is_not_claimed_for_a_nonfloor_bootstrap_value(self) -> None:
        scored = {
            1: {
                "baseline": {"MRR@10": [0.0, 1.0]},
                "treatment": {"MRR@10": [1.0, 0.0]},
            }
        }
        rows = paired_metric_family(
            scored,
            comparisons=(("baseline", "treatment"),),
            metrics=("MRR@10",),
            binary=False,
            resamples=99,
        )

        assert rows[0].raw_p > 0.01
        assert not rows[0].raw_p_at_floor
        assert not rows[0].adjusted_p_inherits_floor

    def test_rejects_nonpositive_resample_count(self) -> None:
        with pytest.raises(ValueError, match="resamples must be >= 1"):
            paired_metric_family(
                {},
                comparisons=(),
                metrics=(),
                binary=False,
                resamples=0,
            )

    def test_runner_rejects_missing_or_unexpected_arities(self) -> None:
        runner = _runner()
        with pytest.raises(ValueError, match="exactly arities 1, 2, and 3"):
            runner._families({}, resamples=20)

    def test_runner_declares_four_complete_multiplicity_families(self) -> None:
        runner = _runner()
        baseline_label = runner.BASELINE_LABEL
        rerank_labels = runner.RERANK_LABELS

        metrics: Mapping[str, Sequence[float]] = {
            "hit@1": [0.0, 1.0],
            "ALL@10": [0.0, 1.0],
            "MRR@10": [0.0, 1.0],
            "nDCG@10": [0.0, 1.0],
        }
        scored = {
            arity: {
                baseline_label: metrics,
                rerank_labels[50]: metrics,
                rerank_labels[100]: metrics,
            }
            for arity in (1, 2, 3)
        }
        families = runner._families(scored, resamples=20)

        assert [(name, binary, len(rows)) for name, binary, rows in families] == [
            ("Efficacy binary", True, 12),
            ("Efficacy graded", False, 12),
            ("Depth binary", True, 6),
            ("Depth graded", False, 6),
        ]
        depth_rows = [row for name, _, rows in families if name.startswith("Depth") for row in rows]
        assert depth_rows
        assert all(row.comparator == rerank_labels[50] for row in depth_rows)
        assert all(row.treatment == rerank_labels[100] for row in depth_rows)
        graded_metrics = {
            row.metric for name, _, rows in families if name.endswith("graded") for row in rows
        }
        assert graded_metrics == {"MRR@10", "nDCG@10"}

    def test_analyze_cache_path_is_credential_free_and_read_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from zhrag.io_utils import write_json  # noqa: PLC0415 - isolated script test

        runner = _runner()
        score_cache = tmp_path / "scores.jsonl"
        sidecar = Path(f"{score_cache}.meta.json")
        provenance = {"model": "cached-model", "endpoint": "https://relay/v1/rerank"}
        write_json(sidecar, provenance)
        before = read_text(sidecar)
        monkeypatch.setattr(runner, "SCORE_CACHE", score_cache)
        monkeypatch.setattr(
            runner,
            "load_env",
            lambda _path: (_ for _ in ()).throw(AssertionError("analyze read .env")),
        )
        experiment = runner.Experiment(corpus={}, queries=[], candidates=[])
        monkeypatch.setattr(runner, "_provenance", lambda *_args, **_kwargs: provenance)

        config, model, scores = runner._load_cache_for_mode(
            experiment,
            score=False,
            analyze=True,
        )

        assert config is None
        assert model == "cached-model"
        assert scores == {}
        assert read_text(sidecar) == before

    def test_analysis_rejects_unexpected_pairs_even_when_coverage_is_complete(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner()
        query = _query()
        experiment = runner.Experiment(
            corpus={"d0": "document"},
            queries=[query],
            candidates=[["d0"]],
        )
        scores = {("q", "d0"): 0.5, ("foreign", "d0"): 0.7}
        runner.SCORE_DEPTH = 1

        with pytest.raises(SystemExit, match="unexpected query/document pairs"):
            runner._analyse(experiment, scores, resamples=10)
        assert capsys.readouterr().out == ""


class TestRerankPrefix:
    def test_reorders_only_the_requested_prefix(self) -> None:
        run = ["a", "b", "c", "d", "e"]
        scores = {"a": 0.1, "b": 0.9, "c": 0.5}
        assert rerank_prefix(run, scores, depth=3) == ["b", "c", "a", "d", "e"]

    def test_ties_preserve_the_fused_order(self) -> None:
        run = ["z", "a", "m"]
        scores = {doc_id: 0.5 for doc_id in run}
        assert rerank_prefix(run, scores, depth=3) == run

    def test_scores_outside_the_window_do_not_move_the_tail(self) -> None:
        run = ["a", "b", "c"]
        scores = {"a": 0.1, "b": 1.0, "c": 999.0}
        assert rerank_prefix(run, scores, depth=2) == ["b", "a", "c"]

    def test_rejects_incomplete_or_nonfinite_scores(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            rerank_prefix(["a", "b"], {"a": 1.0}, depth=2)
        with pytest.raises(ValueError, match="non-finite"):
            rerank_prefix(["a"], {"a": math.nan}, depth=1)

    def test_rejects_shallow_or_duplicate_runs(self) -> None:
        with pytest.raises(ValueError, match="fewer"):
            rerank_prefix(["a"], {"a": 1.0}, depth=2)
        with pytest.raises(ValueError, match="duplicate"):
            rerank_prefix(["a", "a"], {"a": 1.0}, depth=2)


class TestCoverage:
    def test_partial_query_is_retried_as_a_whole(self) -> None:
        queries = [_query("q1"), _query("q2")]
        runs = [["a", "b"], ["c", "d"]]
        scores = {("q1", "a"): 0.8, ("q1", "b"): 0.4, ("q2", "c"): 0.7}
        assert missing_score_queries(queries, runs, scores, depth=2) == ["q2"]

    def test_complete_queries_are_not_missing(self) -> None:
        query = _query()
        run = ["a", "b"]
        scores = {("q", "a"): 0.8, ("q", "b"): 0.4}
        assert missing_score_queries([query], [run], scores, depth=2) == []


class TestFingerprints:
    def test_candidate_hash_is_stable_but_order_sensitive(self) -> None:
        query = _query()
        first = candidate_run_fingerprint([query], [["a", "b"]], depth=2)
        assert first == candidate_run_fingerprint([query], [["a", "b"]], depth=2)
        assert first != candidate_run_fingerprint([query], [["b", "a"]], depth=2)

    def test_candidate_hash_ignores_the_unscored_tail(self) -> None:
        query = _query()
        first = candidate_run_fingerprint([query], [["a", "b", "c"]], depth=2)
        second = candidate_run_fingerprint([query], [["a", "b", "changed"]], depth=2)
        assert first == second

    def test_input_hash_detects_question_or_document_text_drift(self) -> None:
        query = _query()
        run = [["a"]]
        original = rerank_input_fingerprint([query], run, {"a": "document"}, depth=1)
        question_changed = rerank_input_fingerprint(
            [_query(question="changed")], run, {"a": "document"}, depth=1
        )
        document_changed = rerank_input_fingerprint([query], run, {"a": "changed"}, depth=1)
        assert original != question_changed
        assert original != document_changed

    def test_input_hash_rejects_a_missing_candidate_document(self) -> None:
        with pytest.raises(ValueError, match="absent"):
            rerank_input_fingerprint([_query()], [["missing"]], {}, depth=1)

    def test_fingerprints_reject_duplicate_candidate_prefixes(self) -> None:
        query = _query()
        with pytest.raises(ValueError, match="duplicate"):
            candidate_run_fingerprint([query], [["a", "a"]], depth=2)
        with pytest.raises(ValueError, match="duplicate"):
            rerank_input_fingerprint([query], [["a", "a"]], {"a": "text"}, depth=2)

    @pytest.mark.parametrize("depth", [0, -1])
    def test_fingerprints_reject_non_positive_depth(self, depth: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            candidate_run_fingerprint([_query()], [["a"]], depth=depth)
        with pytest.raises(ValueError, match="positive"):
            rerank_input_fingerprint([_query()], [["a"]], {"a": "text"}, depth=depth)
