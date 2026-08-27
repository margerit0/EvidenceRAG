from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from zhrag.eval.pool import PooledQuery, build_pool, pool_fingerprint
from zhrag.eval.tidb_quality import (
    METRIC_LABELS,
    PRIMARY_CONTRASTS,
    PRIMARY_METRIC,
    RUN_LABELS,
    TIDB_QUALITY_SCHEMA,
    assign_overlap_strata,
    evaluate_tidb_quality,
    parse_qrels,
    parse_runs,
    published_chunk_ids,
    score_tidb_pairs,
    validate_quality_report,
)
from zhrag.eval.tidb_runs import DENSE_LABEL, LEXICAL_LABEL, RERANK_LABEL, RRF_LABEL
from zhrag.retrieval.fusion import reciprocal_rank_fusion


class _Fixture:
    def __init__(self) -> None:
        self.doc_ids = tuple(f"chunk-{index:03d}" for index in range(120))
        self.pair_ids = self.doc_ids[:3]
        self.run_rows = self._run_rows()
        self.qrel_rows, units = self._qrel_rows()
        bundle = parse_qrels(self.qrel_rows)
        runs = parse_runs(self.run_rows)
        sizes = [len(pair.direct.judged_doc_ids) for pair in bundle.pairs]
        grades = self._raw_grades(bundle)
        generating = {
            str(grade): sum(pair.direct.generating_chunk_grade == grade for pair in bundle.pairs)
            for grade in (0, 1, 2)
        }
        gold_arities = [len(pair.direct.full_doc_ids) for pair in bundle.pairs]
        contributions, exclusive = self._pool_contributions()
        query_fingerprint = bundle.query_set_fingerprint
        self.state: dict[str, Any] = {
            "schema": "zhrag-ingest-state-v1",
            "collection_name": "tidb_chunks_v1",
            "embedding_profile": "qwen3-embedding-8b-tidb-doc-4096-v1",
            "documents": {
                "doc": {
                    "chunk_ids": list(self.doc_ids),
                    "document_sha256": "unused",
                    "metadata_fingerprint": "unused",
                }
            },
        }
        pool_sha = pool_fingerprint(units)
        self.pool_report: dict[str, Any] = {
            "schema": "zhrag-tidb-runs-v1",
            "queries": 6,
            "pairs": 3,
            "corpus_chunks": len(self.doc_ids),
            "query_set_fingerprint": query_fingerprint,
            "runs_fingerprint": runs.fingerprint,
            "pool_fingerprint": pool_sha,
            "run_depth": 100,
            "pool_depth_per_system": 20,
            "pool_candidates_total": sum(sizes),
            "pool_candidates_mean": sum(sizes) / len(sizes),
            "pool_candidates_min": min(sizes),
            "pool_candidates_max": max(sizes),
            "profiles": {
                "query_embedding": "qwen3-embedding-8b-tidb-query-4096-v1",
                "rerank": "qwen3-reranker-8b-tidb-v1",
                "rrf_k": 10,
                "rerank_request_depth": 100,
                "rerank_apply_depth": 50,
            },
            "system_candidate_slots": contributions,
            "system_exclusive_candidates": exclusive,
        }
        batch_size = 8
        batches = sum(math.ceil(size / batch_size) for size in sizes)
        coverage = {
            label: {
                "queries": 6,
                "top1_complete": 6,
                "top1_rate": 1.0,
                "top10_complete": 6,
                "top10_rate": 1.0,
            }
            for label in RUN_LABELS
        }
        self.qrels_report: dict[str, Any] = {
            "schema": "zhrag-tidb-qrels-v1",
            "queries": 6,
            "pairs": 3,
            "corpus_chunks": len(self.doc_ids),
            "query_set_fingerprint_sha256": query_fingerprint,
            "runs_fingerprint_sha256": runs.fingerprint,
            "pool_fingerprint_sha256": pool_sha,
            "batch_size": batch_size,
            "batches": batches,
            "cache_batches": batches,
            "cache_valid_batches": batches,
            "grades": grades,
            "generating_chunk_grades": generating,
            "generating_chunk_disagreement_rate": (generating["0"] + generating["1"]) / 3,
            "gold_arity": {
                "min": min(gold_arities),
                "mean": sum(gold_arities) / len(gold_arities),
                "max": max(gold_arities),
            },
            "run_judged_coverage": coverage,
        }

    def _per_query_runs(self, query_number: int) -> dict[str, list[str]]:
        rotation = (query_number * 7) % len(self.doc_ids)
        rotated = self.doc_ids[rotation:] + self.doc_ids[:rotation]
        lexical = list(rotated[:100])
        dense = list(reversed(rotated[:100]))
        fused = reciprocal_rank_fusion([lexical, dense], k=10, depth=100)
        reranked = fused[1:50] + fused[:1] + fused[50:]
        return {
            LEXICAL_LABEL: lexical,
            DENSE_LABEL: dense,
            RRF_LABEL: fused,
            RERANK_LABEL: reranked,
        }

    def _run_rows(self) -> list[dict[str, Any]]:
        query_ids = [f"direct:unrelated-{index}" for index in range(3)] + [
            f"paraphrase:other-{index}" for index in range(3)
        ]
        return [
            {"query_id": query_id, "runs": self._per_query_runs(index)}
            for index, query_id in enumerate(query_ids)
        ]

    def _pool_units(self) -> tuple[PooledQuery, ...]:
        runs = parse_runs(self.run_rows)
        positions = {query_id: index for index, query_id in enumerate(runs.query_ids)}
        units: list[PooledQuery] = []
        for index, pair_id in enumerate(self.pair_ids):
            direct_id = f"direct:unrelated-{index}"
            paraphrase_id = f"paraphrase:other-{index}"
            system_unions: dict[str, tuple[str, ...]] = {}
            for label in RUN_LABELS:
                union, _ = build_pool(
                    {
                        "direct": runs.runs[label][positions[direct_id]],
                        "paraphrase": runs.runs[label][positions[paraphrase_id]],
                    },
                    depth=20,
                )
                system_unions[label] = union
            candidates, contributors = build_pool(
                system_unions,
                depth=41,
                required=(pair_id,),
            )
            units.append(
                PooledQuery(
                    chunk_id=pair_id,
                    query_ids=(direct_id, paraphrase_id),
                    questions=(f"直接问题 {index}", f"改写问题 {index}"),
                    answer=f"答案 {index}",
                    candidates=candidates,
                    contributors=contributors,
                )
            )
        return tuple(units)

    def _qrel_rows(self) -> tuple[list[dict[str, Any]], tuple[PooledQuery, ...]]:
        units = self._pool_units()
        rows: list[dict[str, Any]] = []
        direct_overlaps = (0.1, 0.5, 0.9)
        paraphrase_overlaps = (0.2, 0.6, 1.0)
        for index, unit in enumerate(units):
            generator_grade = 1 if index == 0 else 2
            alternative = next(doc_id for doc_id in unit.candidates if doc_id != unit.chunk_id)
            partial = next(
                doc_id for doc_id in unit.candidates if doc_id not in {unit.chunk_id, alternative}
            )
            full = sorted((unit.chunk_id, alternative))
            for task, query_id, question, overlap in zip(
                ("direct", "paraphrase"),
                unit.query_ids,
                unit.questions,
                (direct_overlaps[index], paraphrase_overlaps[index]),
                strict=True,
            ):
                rows.append(
                    {
                        "query_id": query_id,
                        "question": question,
                        "answer": unit.answer,
                        "gold_doc_ids": full,
                        "gold_source_key": "source-shared" if index < 2 else "source-unique",
                        "task": task,
                        "question_type": "factoid",
                        "theme": "sql",
                        "bigram_containment": overlap,
                        "partial_doc_ids": [partial],
                        "judged_doc_ids": sorted(unit.candidates),
                        "generating_chunk_id": unit.chunk_id,
                        "generating_chunk_grade": generator_grade,
                    }
                )
        return list(reversed(rows)), units

    def _raw_grades(self, bundle: object) -> dict[str, int]:
        pairs = bundle.pairs  # type: ignore[attr-defined]
        grades = {"0": 0, "1": 0, "2": 0}
        for pair in pairs:
            surface = pair.direct
            raw_two = len(surface.full_doc_ids) - int(surface.generating_chunk_grade != 2)
            raw_one = len(surface.partial_doc_ids) + int(surface.generating_chunk_grade == 1)
            grades["2"] += raw_two
            grades["1"] += raw_one
            grades["0"] += len(surface.judged_doc_ids) - raw_two - raw_one
        return grades

    def _pool_contributions(self) -> tuple[dict[str, int], dict[str, int]]:
        contribution = {label: 0 for label in RUN_LABELS}
        exclusive = {label: 0 for label in RUN_LABELS}
        for unit in self._pool_units():
            for systems in unit.contributors.values():
                for label in systems:
                    contribution[label] += 1
                if len(systems) == 1:
                    exclusive[systems[0]] += 1
        return dict(sorted(contribution.items())), dict(sorted(exclusive.items()))

    def evaluate(self, *, resamples: int = 100, seed: int = 0) -> dict[str, Any]:
        return evaluate_tidb_quality(
            state=self.state,
            pool_report=self.pool_report,
            qrels_report=self.qrels_report,
            run_rows=self.run_rows,
            qrel_rows=self.qrel_rows,
            resamples=resamples,
            seed=seed,
        )


class TestQrelsAndRunContracts:
    def test_pairs_by_generating_chunk_under_shuffled_unrelated_ids(self) -> None:
        fixture = _Fixture()
        bundle = parse_qrels(fixture.qrel_rows)

        assert [pair.pair_id for pair in bundle.pairs] == sorted(fixture.pair_ids)
        assert bundle.pairs[0].direct.query_id.startswith("direct:unrelated")
        assert bundle.pairs[0].paraphrase.query_id.startswith("paraphrase:other")
        reversed_bundle = parse_qrels(reversed(fixture.qrel_rows))
        assert bundle.query_set_fingerprint == reversed_bundle.query_set_fingerprint
        assert (
            bundle.semantic_fingerprint
            == parse_qrels(reversed(fixture.qrel_rows)).semantic_fingerprint
        )

    def test_rejects_incomplete_pairs_and_shared_control_drift(self) -> None:
        fixture = _Fixture()
        with pytest.raises(ValueError, match="direct and paraphrase"):
            parse_qrels(fixture.qrel_rows[:-1])

        drifted = copy.deepcopy(fixture.qrel_rows)
        drifted[0]["answer"] = "另一个答案"
        with pytest.raises(ValueError, match="controls drift"):
            parse_qrels(drifted)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("gold_doc_ids", [], "non-empty"),
            (
                "partial_doc_ids",
                ["chunk-000"],
                r"controls drift|overlap|operational gold",
            ),
            ("judged_doc_ids", ["chunk-119"], "must be judged|operational gold"),
            ("generating_chunk_grade", 3, "must be 0, 1, or 2"),
            ("bigram_containment", math.nan, "finite"),
        ],
    )
    def test_rejects_invalid_final_qrel_semantics(
        self,
        field: str,
        value: object,
        message: str,
    ) -> None:
        rows = copy.deepcopy(_Fixture().qrel_rows)
        rows[0][field] = value
        with pytest.raises(ValueError, match=message):
            parse_qrels(rows)

    def test_runs_preserve_order_and_enforce_frozen_labels(self) -> None:
        fixture = _Fixture()
        runs = parse_runs(fixture.run_rows)
        assert runs.query_ids == tuple(row["query_id"] for row in fixture.run_rows)

        bad = copy.deepcopy(fixture.run_rows)
        bad[0]["runs"].pop(LEXICAL_LABEL)
        with pytest.raises(ValueError, match="labels differ"):
            parse_runs(bad)

    def test_published_chunks_are_global_unique(self) -> None:
        fixture = _Fixture()
        assert len(published_chunk_ids(fixture.state)) == len(fixture.doc_ids)
        bad = copy.deepcopy(fixture.state)
        bad["documents"]["other"] = {"chunk_ids": [fixture.doc_ids[0]]}
        with pytest.raises(ValueError, match="globally unique"):
            published_chunk_ids(bad)


class TestPairAwareEvaluation:
    def test_scores_all_metrics_in_pair_order(self) -> None:
        fixture = _Fixture()
        bundle = parse_qrels(fixture.qrel_rows)
        cube = score_tidb_pairs(bundle, parse_runs(fixture.run_rows))

        assert set(cube) == set(RUN_LABELS)
        assert set(cube[LEXICAL_LABEL]) == {"direct", "paraphrase"}
        assert set(cube[LEXICAL_LABEL]["direct"]) == set(METRIC_LABELS)
        assert len(cube[LEXICAL_LABEL]["direct"][PRIMARY_METRIC]) == 3

    def test_overall_mean_is_surface_mean_with_source_cluster_ci(self) -> None:
        fixture = _Fixture()
        report = fixture.evaluate()
        cube = score_tidb_pairs(parse_qrels(fixture.qrel_rows), parse_runs(fixture.run_rows))
        direct = cube[DENSE_LABEL]["direct"][PRIMARY_METRIC]
        paraphrase = cube[DENSE_LABEL]["paraphrase"][PRIMARY_METRIC]
        expected = (sum(direct) + sum(paraphrase)) / (2 * len(direct))
        overall = report["system_metrics"][DENSE_LABEL]["overall"][PRIMARY_METRIC]

        assert overall["mean"] == pytest.approx(expected)
        assert overall["n"] == 3
        assert overall["clusters"] == 2
        assert report["evaluation_design"]["queries"] == 6
        assert report["evaluation_design"]["pairs"] == 3
        assert report["evaluation_design"]["clusters"] == 2
        assert report["evaluation_design"]["cluster_key"] == "gold_source_key"
        inferential_unit = report["evaluation_design"]["inferential_unit"]
        assert "direct/paraphrase use one surface score" in inferential_unit
        assert "overall averages both" in inferential_unit
        assert "whole gold_source_key clusters resampled" in inferential_unit
        assert report["evaluation_design"]["cluster_size"] == {
            "min": 1,
            "mean": 1.5,
            "max": 2,
        }

    def test_source_reassignment_changes_uncertainty_not_point_estimates(self) -> None:
        fixture = _Fixture()
        clustered = fixture.evaluate(resamples=1000, seed=19)
        reassigned_rows = copy.deepcopy(fixture.qrel_rows)
        for row in reassigned_rows:
            row["gold_source_key"] = f"unique:{row['generating_chunk_id']}"
        independent = evaluate_tidb_quality(
            state=fixture.state,
            pool_report=fixture.pool_report,
            qrels_report=fixture.qrels_report,
            run_rows=fixture.run_rows,
            qrel_rows=reassigned_rows,
            resamples=1000,
            seed=19,
        )

        clustered_intervals: list[tuple[float, float]] = []
        independent_intervals: list[tuple[float, float]] = []
        for label in RUN_LABELS:
            for view in ("direct", "paraphrase", "overall"):
                for metric in METRIC_LABELS:
                    left = clustered["system_metrics"][label][view][metric]
                    right = independent["system_metrics"][label][view][metric]
                    assert left["mean"] == right["mean"]
                    assert left["n"] == right["n"] == 3
                    assert left["clusters"] == 2
                    assert right["clusters"] == 3
                    clustered_intervals.append((left["low"], left["high"]))
                    independent_intervals.append((right["low"], right["high"]))
        assert clustered_intervals != independent_intervals

    def test_has_exact_predeclared_families_and_pair_counts(self) -> None:
        report = _Fixture().evaluate()

        assert [row["id"] for row in report["primary_contrasts"]] == [
            row[0] for row in PRIMARY_CONTRASTS
        ]
        assert [row["id"] for row in report["surface_robustness"]] == list(RUN_LABELS)
        for family in (report["primary_contrasts"], report["surface_robustness"]):
            for row in family:
                assert row["delta"]["n"] == 3
                assert row["delta"]["clusters"] == 2
                assert sum(row["counts"].values()) == 3

    def test_overlap_tertiles_are_within_surface_stable_and_complete(self) -> None:
        fixture = _Fixture()
        bundle = parse_qrels(fixture.qrel_rows)
        strata = assign_overlap_strata(bundle)
        shuffled = parse_qrels(reversed(fixture.qrel_rows))

        assert strata == assign_overlap_strata(shuffled)
        assert strata["direct"] == {"low": (0,), "middle": (1,), "high": (2,)}
        assert strata["paraphrase"] == {"low": (0,), "middle": (1,), "high": (2,)}

    def test_fails_before_scoring_on_unjudged_or_foreign_candidates(self) -> None:
        fixture = _Fixture()
        unjudged = copy.deepcopy(fixture.qrel_rows)
        target = unjudged[0]
        removed = target["judged_doc_ids"].pop()
        twin = next(
            row
            for row in unjudged
            if row["generating_chunk_id"] == target["generating_chunk_id"] and row is not target
        )
        twin["judged_doc_ids"].remove(removed)
        with pytest.raises(ValueError, match=r"judged set differs|unjudged"):
            evaluate_tidb_quality(
                state=fixture.state,
                pool_report=fixture.pool_report,
                qrels_report=fixture.qrels_report,
                run_rows=fixture.run_rows,
                qrel_rows=unjudged,
                resamples=10,
            )

        foreign = copy.deepcopy(fixture.run_rows)
        foreign[0]["runs"][LEXICAL_LABEL][0] = "foreign"
        with pytest.raises(ValueError, match=r"unpublished|exact RRF"):
            evaluate_tidb_quality(
                state=fixture.state,
                pool_report=fixture.pool_report,
                qrels_report=fixture.qrels_report,
                run_rows=foreign,
                qrel_rows=fixture.qrel_rows,
                resamples=10,
            )

    def test_rejects_run_and_aggregate_fingerprint_drift(self) -> None:
        fixture = _Fixture()
        broken_run = copy.deepcopy(fixture.run_rows)
        broken_run[0]["runs"][RRF_LABEL][0:2] = reversed(broken_run[0]["runs"][RRF_LABEL][0:2])
        with pytest.raises(ValueError, match="exact RRF"):
            evaluate_tidb_quality(
                state=fixture.state,
                pool_report=fixture.pool_report,
                qrels_report=fixture.qrels_report,
                run_rows=broken_run,
                qrel_rows=fixture.qrel_rows,
                resamples=10,
            )

        qrels_report = copy.deepcopy(fixture.qrels_report)
        qrels_report["query_set_fingerprint_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="query-set fingerprint"):
            evaluate_tidb_quality(
                state=fixture.state,
                pool_report=fixture.pool_report,
                qrels_report=qrels_report,
                run_rows=fixture.run_rows,
                qrel_rows=fixture.qrel_rows,
                resamples=10,
            )


class TestAggregateReportContract:
    def test_report_is_deterministic_and_contains_no_raw_content(self) -> None:
        fixture = _Fixture()
        first = fixture.evaluate(resamples=50, seed=7)
        second = fixture.evaluate(resamples=50, seed=7)
        serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)

        assert first == second
        assert first["schema"] == TIDB_QUALITY_SCHEMA
        for forbidden in (
            "直接问题",
            "改写问题",
            "答案 0",
            "source-0",
            "chunk-000",
            "direct:unrelated",
            '"endpoint"',
            "requested_model",
            "prompt_tokens",
        ):
            assert forbidden not in serialized

    def test_validator_rejects_unexpected_fields_and_holm_drift(self) -> None:
        report = _Fixture().evaluate()
        unexpected = copy.deepcopy(report)
        unexpected["raw_queries"] = []
        with pytest.raises(ValueError, match="keys differ"):
            validate_quality_report(unexpected)

        drifted = copy.deepcopy(report)
        drifted["primary_contrasts"][0]["adjusted_p"] = 0.123456
        with pytest.raises(ValueError, match="Holm-adjusted"):
            validate_quality_report(drifted)

    def test_validator_rejects_cluster_metadata_drift(self) -> None:
        report = _Fixture().evaluate()
        wrong_design = copy.deepcopy(report)
        wrong_design["evaluation_design"]["clusters"] = 3
        with pytest.raises(ValueError, match="cluster-size aggregates"):
            validate_quality_report(wrong_design)

        wrong_interval = copy.deepcopy(report)
        interval = wrong_interval["system_metrics"][DENSE_LABEL]["overall"][PRIMARY_METRIC]
        interval["clusters"] = 3
        with pytest.raises(ValueError, match="source-cluster count"):
            validate_quality_report(wrong_interval)

        wrong_stratum = copy.deepcopy(report)
        row = wrong_stratum["overlap_strata"]["tasks"]["direct"]["low"]
        row["n"] = 2
        systems = row["systems"]
        for interval in systems.values():
            interval["n"] = 2
        systems[DENSE_LABEL]["clusters"] = 2
        with pytest.raises(ValueError, match="source-cluster counts differ"):
            validate_quality_report(wrong_stratum)

    @pytest.mark.parametrize(("resamples", "seed"), [(0, 0), (10, -1), (10, 1 << 63)])
    def test_rejects_invalid_bootstrap_configuration(self, resamples: int, seed: int) -> None:
        fixture = _Fixture()
        with pytest.raises(ValueError, match=r"resamples|seed"):
            fixture.evaluate(resamples=resamples, seed=seed)
