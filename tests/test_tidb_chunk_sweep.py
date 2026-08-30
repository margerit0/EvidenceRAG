"""Hermetic contracts for the TiDB source-level chunk-size sweep."""

from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from test_tidb_quality import _Fixture
from zhrag.eval.tidb_chunk_sweep import (
    CHUNK_SWEEP_REPORT_SCHEMA,
    CHUNK_SWEEP_SAMPLES_SCHEMA,
    PROFILE_IDS,
    PROFILES,
    RUN_LABELS,
    ChunkSweepProfile,
    EmbeddingCache,
    SourceQrels,
    SweepRuns,
    build_numeric_samples,
    build_report,
    build_source_qrels,
    build_sweep_runs,
    collapse_chunk_run_to_sources,
    materialize_profile,
    numeric_samples_sha256,
    parse_sweep_runs,
    profile_for_id,
    strict_embedding_cache,
    sweep_runs_rows,
    validate_numeric_samples,
    validate_report,
    validated_embedding_rows,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    authenticate_qrels_quality_anchor,
    build_query_cache_provenance,
    query_text_fingerprint,
    validate_query_cache_provenance,
    verify_query_cache_dense_runs,
)
from zhrag.eval.tidb_quality import (
    QrelPair,
    QrelsBundle,
    QrelSurface,
    parse_qrels,
)
from zhrag.ingest import ChunkPlan, DocumentPlan
from zhrag.io_utils import read_json, write_json, write_jsonl
from zhrag.retrieval.fusion import reciprocal_rank_fusion
from zhrag.tokens import estimate_tokens


class TestProfiles:
    def test_registry_freezes_order_and_cost_guards(self) -> None:
        assert PROFILE_IDS == (
            "tidb-chunk-t256-h384-v1",
            "tidb-chunk-t400-h600-v1",
            "tidb-chunk-t800-h1200-v1",
        )
        assert [PROFILES[item].expected_chunks for item in PROFILE_IDS] == [2_802, 1_832, 1_019]
        assert [PROFILES[item].expected_new_vectors for item in PROFILE_IDS] == [2_550, 0, 829]
        assert [PROFILES[item].expected_paid_batches for item in PROFILE_IDS] == [160, 0, 52]

    def test_profile_fingerprint_binds_chunker_settings(self) -> None:
        left = ChunkSweepProfile("test-a", 8, 12, 2, 0, 2, 1)
        right = ChunkSweepProfile("test-a", 9, 12, 2, 0, 2, 1)
        assert left.chunker_fingerprint != right.chunker_fingerprint
        assert left.fingerprint != right.fingerprint

    def test_rejects_incoherent_cost_guard(self) -> None:
        with pytest.raises(ValueError, match="batch size"):
            ChunkSweepProfile("bad", 8, 12, 17, 0, 17, 1)

    def test_unknown_profile_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown frozen"):
            profile_for_id("tidb-chunk-custom")

    def test_materializes_ledger_and_versioned_fingerprints(self) -> None:
        profile = ChunkSweepProfile("test", 8, 12, 2, 0, 2, 1)
        long_body = "```text\n" + "protected block content " * 8 + "\n```"
        document = DocumentPlan(
            key="source-a",
            path="doc.md",
            document_sha256="a" * 64,
            metadata_fingerprint="b" * 64,
            metadata=MappingProxyType({}),
            chunks=(
                ChunkPlan("chunk-a", 0, "H\n\nshort", ("H",), estimate_tokens("short")),
                ChunkPlan(
                    "chunk-b",
                    1,
                    f"H\n\n{long_body}",
                    ("H",),
                    estimate_tokens(long_body),
                ),
            ),
        )

        planned = materialize_profile(profile, (document,), enforce_document_count=False)

        assert planned.chunk_ids == ("chunk-a", "chunk-b")
        assert planned.source_by_chunk == {"chunk-a": "source-a", "chunk-b": "source-a"}
        assert len(planned.exceedance_rows) == 1
        assert planned.exceedance_rows[0]["reason"] == "protected_fence"
        exceedance = planned.profile_summary["split_trigger_exceedance"]
        assert isinstance(exceedance, dict)
        assert exceedance["count"] == 1
        assert exceedance["reasons"]["unexplained"] == 0
        assert len(planned.corpus_fingerprint) == 64
        assert len(planned.chunk_ids_fingerprint) == 64
        assert len(planned.source_mapping_fingerprint) == 64


class TestStrictEmbeddingCache:
    def test_aligns_rows_to_expected_order_and_fingerprints(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        write_jsonl(
            cache,
            [
                {"doc_id": "b", "embedding": [0.0, 2.0]},
                {"doc_id": "a", "embedding": [1.0, 0.0]},
            ],
        )

        loaded = strict_embedding_cache(cache, ("a", "b"), width=2, allow_partial=False)

        assert loaded.missing_ids == ()
        assert loaded.matrix(("a", "b")).tolist() == [[1.0, 0.0], [0.0, 2.0]]
        assert len(loaded.fingerprint) == 64

    def test_missing_file_is_an_empty_partial_checkpoint(self, tmp_path: Path) -> None:
        loaded = strict_embedding_cache(
            tmp_path / "missing.jsonl",
            ("a", "b"),
            width=2,
            allow_partial=True,
        )
        assert loaded.missing_ids == ("a", "b")
        assert not loaded.vectors

    @pytest.mark.parametrize(
        ("rows", "message"),
        (
            (
                [
                    {"doc_id": "a", "embedding": [1.0, 0.0]},
                    {"doc_id": "a", "embedding": [0.0, 1.0]},
                ],
                "duplicate",
            ),
            ([{"doc_id": "extra", "embedding": [1.0, 0.0]}], "unexpected"),
            ([{"doc_id": "a", "embedding": [True, 0.0]}], "not numeric"),
            ([{"doc_id": "a", "embedding": [float("nan"), 0.0]}], "non-finite"),
            ([{"doc_id": "a", "embedding": [0.0, 0.0]}], "zero/non-finite"),
            ([{"doc_id": "a", "embedding": [1.0]}], "width"),
        ),
    )
    def test_rejects_ambiguous_or_invalid_rows(
        self,
        tmp_path: Path,
        rows: list[dict[str, object]],
        message: str,
    ) -> None:
        cache = tmp_path / "cache.jsonl"
        write_jsonl(cache, rows)
        with pytest.raises(ValueError, match=message):
            strict_embedding_cache(cache, ("a",), width=2, allow_partial=True)

    def test_complete_mode_rejects_missing_rows(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        write_jsonl(cache, [{"doc_id": "a", "embedding": [1.0, 0.0]}])
        with pytest.raises(ValueError, match="missing 1"):
            strict_embedding_cache(cache, ("a", "b"), width=2, allow_partial=False)

    def test_validates_whole_provider_batch_before_returning_rows(self) -> None:
        rows = validated_embedding_rows(
            ("a", "b"),
            ([1.0, 0.0], [0.0, 2.0]),
            width=2,
        )
        assert rows == (
            {"doc_id": "a", "embedding": [1.0, 0.0]},
            {"doc_id": "b", "embedding": [0.0, 2.0]},
        )
        with pytest.raises(ValueError, match="not numeric"):
            validated_embedding_rows(
                ("a", "b"),
                ([1.0, 0.0], [True, 2.0]),
                width=2,
            )


class TestSourceCollapse:
    def test_collapses_by_first_chunk_occurrence(self) -> None:
        assert collapse_chunk_run_to_sources(
            ("a-2", "b-1", "a-1", "c-1"),
            {"a-1": "A", "a-2": "A", "b-1": "B", "c-1": "C"},
        ) == ("A", "B", "C")

    def test_rejects_unknown_and_duplicate_chunks(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            collapse_chunk_run_to_sources(("unknown",), {})
        with pytest.raises(ValueError, match="duplicate"):
            collapse_chunk_run_to_sources(("a", "a"), {"a": "A"})

    def test_rrf_must_precede_source_collapse(self) -> None:
        source_by_chunk = {
            "a1": "A",
            "a2": "A",
            "b": "B",
            "x": "X",
            "y": "Y",
        }
        lexical = ("a1", "b", "x", "a2")
        dense = ("a2", "b", "y", "a1")
        chunk_fused = reciprocal_rank_fusion((lexical, dense), k=10, depth=10)
        correct = collapse_chunk_run_to_sources(chunk_fused, source_by_chunk)
        premature = reciprocal_rank_fusion(
            (
                collapse_chunk_run_to_sources(lexical, source_by_chunk),
                collapse_chunk_run_to_sources(dense, source_by_chunk),
            ),
            k=10,
            depth=10,
        )
        assert correct[:2] == ("B", "A")
        assert premature[:2] == ["A", "B"]

    def test_exact_rrf_keeps_the_union_of_top_100_input_arms(self) -> None:
        lexical = tuple(f"lexical-{index:03d}" for index in range(100))
        dense = tuple(f"dense-{index:03d}" for index in range(100))
        source_by_chunk = {chunk_id: f"source-{chunk_id}" for chunk_id in (*lexical, *dense)}

        runs = build_sweep_runs(
            ("query-000",),
            (lexical,),
            (dense,),
            source_by_chunk,
            enforce_query_count=False,
        )

        assert len(runs.chunk_runs[RUN_LABELS[0]][0]) == 100
        assert len(runs.chunk_runs[RUN_LABELS[1]][0]) == 100
        assert len(runs.chunk_runs[RUN_LABELS[2]][0]) == 200
        assert runs.chunk_runs[RUN_LABELS[2]][0] == tuple(
            reciprocal_rank_fusion((lexical, dense), k=10, depth=100)
        )

    def test_builds_and_round_trips_authenticated_three_arm_runs(self) -> None:
        chunk_ids = tuple(f"chunk-{index:03d}" for index in range(100))
        source_by_chunk = {
            chunk_id: f"source-{index // 2:03d}" for index, chunk_id in enumerate(chunk_ids)
        }
        runs = build_sweep_runs(
            ("query-000",),
            (tuple(reversed(chunk_ids[:5])),),
            (chunk_ids,),
            source_by_chunk,
            enforce_query_count=False,
        )

        rows = sweep_runs_rows(runs)
        parsed = parse_sweep_runs(
            rows,
            source_by_chunk,
            enforce_query_count=False,
        )
        assert parsed.fingerprint == runs.fingerprint
        assert parsed.source_runs[RUN_LABELS[0]][0] == ("source-002", "source-001", "source-000")

        tampered = cast(list[dict[str, Any]], copy.deepcopy(list(rows)))
        tampered[0]["source_runs"][RUN_LABELS[2]] = ["forged-source"]
        with pytest.raises(ValueError, match="post-arm collapse"):
            parse_sweep_runs(
                tampered,
                source_by_chunk,
                enforce_query_count=False,
            )


def _surface(
    *,
    query_id: str,
    task: str,
    pair_id: str,
    source: str,
    full: tuple[str, ...],
    partial: tuple[str, ...],
    judged: tuple[str, ...],
    grade: int,
) -> QrelSurface:
    return QrelSurface(
        query_id=query_id,
        task=task,
        pair_id=pair_id,
        question=f"question-{query_id}",
        answer=f"answer-{pair_id}",
        full_doc_ids=full,
        partial_doc_ids=partial,
        judged_doc_ids=judged,
        generating_chunk_grade=grade,
        bigram_containment=0.5,
        theme="theme",
        question_type="factoid",
        source_key=source,
    )


def _canonical_bundle() -> tuple[QrelsBundle, dict[str, str]]:
    pairs: list[QrelPair] = []
    by_query: dict[str, QrelSurface] = {}
    source_by_chunk: dict[str, str] = {}
    for pair_index in range(490):
        cluster = pair_index // 2
        source = f"source-{cluster:03d}"
        pair_id = f"generating-{pair_index:03d}"
        support = f"support-{pair_index:03d}"
        other = f"other-{pair_index:03d}"
        source_by_chunk[pair_id] = source
        source_by_chunk[support] = source
        source_by_chunk[other] = f"other-source-{pair_index:03d}"
        grade = 1 if pair_index == 0 else 2
        full = tuple(sorted((pair_id, support, other)))
        partial: tuple[str, ...] = ()
        judged = full
        direct = _surface(
            query_id=f"direct:{pair_index:03d}",
            task="direct",
            pair_id=pair_id,
            source=source,
            full=full,
            partial=partial,
            judged=judged,
            grade=grade,
        )
        paraphrase = _surface(
            query_id=f"paraphrase:{pair_index:03d}",
            task="paraphrase",
            pair_id=pair_id,
            source=source,
            full=full,
            partial=partial,
            judged=judged,
            grade=grade,
        )
        pairs.append(QrelPair(pair_id, direct, paraphrase))
        by_query[direct.query_id] = direct
        by_query[paraphrase.query_id] = paraphrase
    return (
        QrelsBundle(tuple(pairs), by_query, "a" * 64, "b" * 64),
        source_by_chunk,
    )


def _sweep_runs(
    source_qrels: SourceQrels,
    *,
    profile_index: int,
) -> SweepRuns:
    by_query = source_qrels.by_query
    query_ids = tuple(sorted(by_query))
    source_rows: list[tuple[str, ...]] = []
    chunk_rows: list[tuple[str, ...]] = []
    for position, query_id in enumerate(query_ids):
        surface = by_query[query_id]
        origin = surface.origin_source
        task = surface.task
        if profile_index == 0 and task == "direct":
            source_run = (origin, "unjudged-noise")
        elif profile_index in (0, 1):
            source_run = ("unjudged-noise", origin)
        else:
            source_run = ("unjudged-noise", "other-noise")
        source_rows.append(source_run)
        chunk_rows.append((f"chunk-{position:03d}-0", f"chunk-{position:03d}-1"))
    source_mapping = {label: tuple(source_rows) for label in RUN_LABELS}
    chunk_mapping = {label: tuple(chunk_rows) for label in RUN_LABELS}
    return SweepRuns(query_ids, chunk_mapping, source_mapping)


@pytest.fixture(scope="module")
def source_fixture() -> SourceQrels:
    bundle, source_by_chunk = _canonical_bundle()
    return build_source_qrels(bundle, source_by_chunk)


@pytest.fixture(scope="module")
def samples(source_fixture: SourceQrels) -> dict[str, object]:
    runs = {
        profile_id: _sweep_runs(source_fixture, profile_index=index)
        for index, profile_id in enumerate(PROFILE_IDS)
    }
    return build_numeric_samples(
        source_fixture,
        runs,
        input_fingerprints={"qrels_sha256": "c" * 64},
    )


class TestQrelsProvenance:
    def test_quality_report_certifies_exact_qrels_semantics(self) -> None:
        fixture = _Fixture()
        bundle = parse_qrels(fixture.qrel_rows)
        report = fixture.evaluate(resamples=2)

        authenticate_qrels_quality_anchor(
            bundle,
            report,
            expected_queries=6,
            expected_pairs=3,
        )

    def test_rejects_relevance_mutation_with_unchanged_query_set(self) -> None:
        fixture = _Fixture()
        report = fixture.evaluate(resamples=2)
        original = parse_qrels(fixture.qrel_rows)
        changed_rows = copy.deepcopy(fixture.qrel_rows)
        pair_id = changed_rows[0]["generating_chunk_id"]
        pair_rows = [row for row in changed_rows if row["generating_chunk_id"] == pair_id]
        for row in pair_rows:
            replacement = row["partial_doc_ids"][0]
            demoted = next(
                doc_id for doc_id in row["gold_doc_ids"] if doc_id != row["generating_chunk_id"]
            )
            row["gold_doc_ids"] = sorted(
                replacement if doc_id == demoted else doc_id for doc_id in row["gold_doc_ids"]
            )
            row["partial_doc_ids"] = [demoted]
        changed = parse_qrels(changed_rows)

        assert changed.query_set_fingerprint == original.query_set_fingerprint
        assert changed.semantic_fingerprint != original.semantic_fingerprint
        with pytest.raises(ValueError, match="does not certify the final qrels semantics"):
            authenticate_qrels_quality_anchor(
                changed,
                report,
                expected_queries=6,
                expected_pairs=3,
            )

    def test_query_cache_adoption_binds_text_matrix_and_dense_runs(self) -> None:
        direct = _surface(
            query_id="direct:q",
            task="direct",
            pair_id="doc-a",
            source="source",
            full=("doc-a",),
            partial=(),
            judged=("doc-a",),
            grade=2,
        )
        paraphrase = _surface(
            query_id="paraphrase:q",
            task="paraphrase",
            pair_id="doc-a",
            source="source",
            full=("doc-a",),
            partial=(),
            judged=("doc-a",),
            grade=2,
        )
        qrels = QrelsBundle(
            (QrelPair("doc-a", direct, paraphrase),),
            {direct.query_id: direct, paraphrase.query_id: paraphrase},
            "a" * 64,
            "b" * 64,
        )
        query_vectors = {
            direct.query_id: np.asarray([1.0, 0.0], dtype=np.float32),
            paraphrase.query_id: np.asarray([0.0, 1.0], dtype=np.float32),
        }
        document_vectors = {
            "doc-a": np.asarray([1.0, 0.0], dtype=np.float32),
            "doc-b": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        query_cache = EmbeddingCache(MappingProxyType(query_vectors), (), "c" * 64, 2)
        document_cache = EmbeddingCache(MappingProxyType(document_vectors), (), "d" * 64, 2)
        runs = (
            {
                "query_id": direct.query_id,
                "runs": {
                    "bm25-char-bigram": ["doc-a", "doc-b"],
                    "dense-qwen3-4096": ["doc-a", "doc-b"],
                    "rrf-k10-depth100": ["doc-a", "doc-b"],
                    "rerank-qwen3-top50": ["doc-a", "doc-b"],
                },
            },
            {
                "query_id": paraphrase.query_id,
                "runs": {
                    "bm25-char-bigram": ["doc-b", "doc-a"],
                    "dense-qwen3-4096": ["doc-b", "doc-a"],
                    "rrf-k10-depth100": ["doc-b", "doc-a"],
                    "rerank-qwen3-top50": ["doc-b", "doc-a"],
                },
            },
        )

        dense_fingerprint = verify_query_cache_dense_runs(
            query_cache,
            document_cache,
            ("doc-a", "doc-b"),
            qrels,
            runs,
        )
        certificate = build_query_cache_provenance(
            query_cache=query_cache,
            qrels=qrels,
            dense_runs_fingerprint=dense_fingerprint,
            query_cache_file_sha256="e" * 64,
            legacy_sidecar_file_sha256="f" * 64,
            quality_report_file_sha256="1" * 64,
        )
        validate_query_cache_provenance(
            certificate,
            query_cache=query_cache,
            qrels=qrels,
            dense_runs_fingerprint=dense_fingerprint,
            query_cache_file_sha256="e" * 64,
            legacy_sidecar_file_sha256="f" * 64,
            quality_report_file_sha256="1" * 64,
        )
        assert certificate["historical_request_text_binding"] == "not-observed"

        changed_surface = QrelSurface(
            direct.query_id,
            direct.task,
            direct.pair_id,
            "changed question",
            direct.answer,
            direct.full_doc_ids,
            direct.partial_doc_ids,
            direct.judged_doc_ids,
            direct.generating_chunk_grade,
            direct.bigram_containment,
            direct.theme,
            direct.question_type,
            direct.source_key,
        )
        changed_qrels = QrelsBundle(
            qrels.pairs,
            {direct.query_id: changed_surface, paraphrase.query_id: paraphrase},
            qrels.query_set_fingerprint,
            qrels.semantic_fingerprint,
        )
        assert query_text_fingerprint(changed_qrels) != query_text_fingerprint(qrels)
        with pytest.raises(ValueError, match="provenance drift"):
            validate_query_cache_provenance(
                certificate,
                query_cache=query_cache,
                qrels=changed_qrels,
                dense_runs_fingerprint=dense_fingerprint,
                query_cache_file_sha256="e" * 64,
                legacy_sidecar_file_sha256="f" * 64,
                quality_report_file_sha256="1" * 64,
            )

        swapped = EmbeddingCache(
            MappingProxyType(
                {
                    direct.query_id: query_vectors[paraphrase.query_id],
                    paraphrase.query_id: query_vectors[direct.query_id],
                }
            ),
            (),
            "2" * 64,
            2,
        )
        with pytest.raises(ValueError, match="does not reproduce"):
            verify_query_cache_dense_runs(
                swapped,
                document_cache,
                ("doc-a", "doc-b"),
                qrels,
                runs,
            )


class TestSourceQrels:
    def test_recovers_245_origin_clusters_and_raw_full_sources(
        self, source_fixture: SourceQrels
    ) -> None:
        assert len(source_fixture.pairs) == 490
        assert len(source_fixture.by_query) == 980
        first = source_fixture.pairs[0].direct
        assert first.origin_source in first.confirmed_sources
        assert first.origin_source not in first.partial_sources
        assert len(first.confirmed_sources) == 2
        assert len(source_fixture.fingerprint) == 64

    def test_rejects_operational_promotion_without_raw_origin_support(self) -> None:
        bundle, source_by_chunk = _canonical_bundle()
        first = bundle.pairs[0]
        promoted_only = (first.pair_id, "other-000")
        direct = _surface(
            query_id=first.direct.query_id,
            task="direct",
            pair_id=first.pair_id,
            source=first.direct.source_key,
            full=promoted_only,
            partial=(),
            judged=promoted_only,
            grade=1,
        )
        paraphrase = _surface(
            query_id=first.paraphrase.query_id,
            task="paraphrase",
            pair_id=first.pair_id,
            source=first.direct.source_key,
            full=promoted_only,
            partial=(),
            judged=promoted_only,
            grade=1,
        )
        pairs = (QrelPair(first.pair_id, direct, paraphrase), *bundle.pairs[1:])
        by_query = dict(bundle.by_query)
        by_query[direct.query_id] = direct
        by_query[paraphrase.query_id] = paraphrase
        malformed = QrelsBundle(pairs, by_query, "a" * 64, "b" * 64)
        with pytest.raises(ValueError, match="lacks raw grade-2"):
            build_source_qrels(malformed, source_by_chunk)


class TestSamplesAndReport:
    def test_samples_are_identity_free_and_deterministic(self, samples: dict[str, object]) -> None:
        assert samples["schema"] == CHUNK_SWEEP_SAMPLES_SCHEMA
        assert samples["queries"] == 980
        assert samples["pairs"] == 490
        assert samples["clusters"] == 245
        validate_numeric_samples(samples)
        assert numeric_samples_sha256(samples) == numeric_samples_sha256(copy.deepcopy(samples))
        rendered = str(samples)
        assert "question-direct" not in rendered
        assert "source-000" not in rendered

    def test_samples_validator_rejects_tamper_and_raw_identity(
        self, samples: dict[str, object]
    ) -> None:
        tampered = cast(dict[str, Any], copy.deepcopy(samples))
        tampered["rows"][0]["cluster"] = 245
        with pytest.raises(ValueError, match="cluster index"):
            validate_numeric_samples(tampered)

        leaked = copy.deepcopy(samples)
        leaked["query_id"] = "direct:000"
        with pytest.raises(ValueError, match="keys differ"):
            validate_numeric_samples(leaked)

    def test_report_rebuilds_both_holm_families(self, samples: dict[str, object]) -> None:
        report = build_report(
            samples,
            profile_summaries={
                profile_id: {
                    "profile_id": profile_id,
                    "chunks": PROFILES[profile_id].expected_chunks,
                }
                for profile_id in PROFILE_IDS
            },
            provenance={"qrels_sha256": "c" * 64},
            resamples=30,
            seed=0,
        )

        assert report["schema"] == CHUNK_SWEEP_REPORT_SCHEMA
        validate_report(report)
        typed = cast(dict[str, Any], report)
        efficacy = typed["families"]["origin-source-efficacy"]
        assert efficacy[0]["delta"] == pytest.approx(0.25)
        assert efficacy[1]["delta"] == pytest.approx(-0.5)
        assert efficacy[0]["raw_p_at_floor"] is True
        interactions = typed["families"]["surface-home-field"]
        assert interactions[0]["delta"] == pytest.approx(-0.5)
        assert interactions[1]["delta"] == pytest.approx(0.0)
        assert interactions[1]["raw_p"] == pytest.approx(1.0)

    def test_samples_and_report_validate_after_sorted_json_round_trip(
        self,
        samples: dict[str, object],
        tmp_path: Path,
    ) -> None:
        report = build_report(
            samples,
            profile_summaries={
                profile_id: {
                    "profile_id": profile_id,
                    "chunks": PROFILES[profile_id].expected_chunks,
                }
                for profile_id in PROFILE_IDS
            },
            provenance={"qrels_sha256": "c" * 64},
            resamples=2,
            seed=0,
        )
        samples_path = tmp_path / "numeric_samples.json"
        report_path = tmp_path / "report.json"
        write_json(samples_path, samples)
        write_json(report_path, report)

        reloaded_samples = read_json(samples_path)
        reloaded_report = read_json(report_path)
        validate_numeric_samples(reloaded_samples)
        validate_report(reloaded_report)
        assert numeric_samples_sha256(cast(dict[str, object], reloaded_samples)) == (
            numeric_samples_sha256(samples)
        )
        assert reloaded_report == report

    def test_report_validator_rejects_build_history_claim(self, samples: dict[str, object]) -> None:
        report = build_report(
            samples,
            profile_summaries={
                profile_id: {"profile_id": profile_id} for profile_id in PROFILE_IDS
            },
            provenance={"qrels_sha256": "c" * 64},
            resamples=2,
        )
        typed = cast(dict[str, Any], report)
        typed["execution_contract"]["document_vectors_created"] = 3_379
        with pytest.raises(ValueError, match="forbidden raw field"):
            validate_report(report)
