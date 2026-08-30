"""Hermetic tests for the offline M7 chunk-sweep evaluator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from zhrag.eval.tidb_chunk_sweep import (
    CHUNK_SWEEP_REPORT_SCHEMA,
    CHUNK_SWEEP_SAMPLES_SCHEMA,
    PROFILE_IDS,
    PROFILES,
    RUN_LABELS,
    EmbeddingCache,
    PlannedProfile,
    SourcePair,
    SourceQrels,
    SourceSurface,
    SweepRuns,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    CanonicalSnapshot,
    ChunkSweepPaths,
    FinalizedProfile,
)
from zhrag.eval.tidb_chunk_sweep_evaluation import (
    ChunkSweepEvaluation,
    evaluate_chunk_sweep,
)
from zhrag.eval.tidb_quality import QrelsBundle, QrelSurface
from zhrag.eval.tidb_runs import RRF_LABEL
from zhrag.io_utils import read_json, replace_files, write_json, write_text

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluate_tidb_chunk_sweep.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_evaluate_tidb_chunk_sweep_runner", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import evaluate_tidb_chunk_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _surface(query_id: str, task: str, pair_id: str, source: str) -> QrelSurface:
    return QrelSurface(
        query_id=query_id,
        task=task,
        pair_id=pair_id,
        question=f"question {query_id}",
        answer="answer",
        full_doc_ids=(pair_id,),
        partial_doc_ids=(),
        judged_doc_ids=(pair_id,),
        generating_chunk_grade=2,
        bigram_containment=0.5,
        theme="theme",
        question_type="factoid",
        source_key=source,
    )


def _evaluation_fixture() -> tuple[
    CanonicalSnapshot,
    dict[str, FinalizedProfile],
]:
    source_pairs: list[SourcePair] = []
    source_by_query: dict[str, SourceSurface] = {}
    qrels_by_query: dict[str, QrelSurface] = {}
    for pair_index in range(490):
        source = f"source-{pair_index // 2:03d}"
        pair_id = f"pair-{pair_index:03d}"
        direct_id = f"direct:{pair_index:03d}"
        paraphrase_id = f"paraphrase:{pair_index:03d}"
        direct = SourceSurface(
            direct_id,
            "direct",
            pair_id,
            source,
            (source,),
            (),
            (source,),
        )
        paraphrase = SourceSurface(
            paraphrase_id,
            "paraphrase",
            pair_id,
            source,
            (source,),
            (),
            (source,),
        )
        source_pairs.append(SourcePair(direct, paraphrase))
        source_by_query[direct_id] = direct
        source_by_query[paraphrase_id] = paraphrase
        qrels_by_query[direct_id] = _surface(direct_id, "direct", pair_id, source)
        qrels_by_query[paraphrase_id] = _surface(
            paraphrase_id,
            "paraphrase",
            pair_id,
            source,
        )
    source_qrels = SourceQrels(tuple(source_pairs), source_by_query, "1" * 64)
    qrels = QrelsBundle((), qrels_by_query, "2" * 64, "3" * 64)
    query_ids = tuple(sorted(qrels_by_query))
    query_vectors = {query_id: np.asarray([1.0, 0.0], dtype=np.float32) for query_id in query_ids}
    query_cache = EmbeddingCache(
        MappingProxyType(query_vectors),
        (),
        "4" * 64,
        2,
    )

    profiles: dict[str, PlannedProfile] = {}
    finalized: dict[str, FinalizedProfile] = {}
    for profile_index, profile_id in enumerate(PROFILE_IDS):
        chunk_ids = tuple(f"{profile_id}-chunk-{index:03d}" for index in range(100))
        corpus = {chunk_id: f"common term {index}" for index, chunk_id in enumerate(chunk_ids)}
        source_by_chunk = {
            chunk_id: f"source-{index:03d}" for index, chunk_id in enumerate(chunk_ids)
        }
        profile = PROFILES[profile_id]
        summary: dict[str, object] = {
            "profile_id": profile_id,
            "target_tokens": profile.target_tokens,
            "split_trigger_tokens": profile.split_trigger_tokens,
            "documents": 450,
            "chunks": profile.expected_chunks,
            "tokens": {
                "p10": 1.0,
                "p50": 1.0,
                "p90": 1.0,
                "p99": 1.0,
                "max": 1,
                "under_100": 1,
                "total": 1,
            },
            "split_trigger_exceedance": {
                "count": 0,
                "rate": 0.0,
                "reasons": {
                    "protected_fence": 0,
                    "protected_table": 0,
                    "protected_fence_and_table": 0,
                    "indivisible_paragraph": 0,
                    "unexplained": 0,
                },
            },
            "embedding_input_estimate": {
                "limit": 32_768,
                "max": 1,
                "exceedances": 0,
            },
            "chunker_fingerprint_sha256": profile.chunker_fingerprint,
            "profile_fingerprint_sha256": profile.fingerprint,
            "document_set_fingerprint_sha256": "5" * 64,
            "canonical_exact_reuse": profile.expected_canonical_reuse,
            "required_new_document_vectors": profile.expected_new_vectors,
            "required_embedding_batches": profile.expected_paid_batches,
            "corpus_fingerprint_sha256": "6" * 64,
            "chunk_ids_fingerprint_sha256": "7" * 64,
            "source_mapping_fingerprint_sha256": "8" * 64,
            "embedding_matrix_fingerprint_sha256": str(profile_index + 1) * 64,
        }
        planned = cast(
            PlannedProfile,
            SimpleNamespace(
                profile=profile,
                document_count=450,
                chunk_ids=chunk_ids,
                corpus=corpus,
                source_by_chunk=source_by_chunk,
                profile_summary=summary,
            ),
        )
        vectors = {
            chunk_id: np.asarray(
                [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0],
                dtype=np.float32,
            )
            for index, chunk_id in enumerate(chunk_ids)
        }
        embeddings = EmbeddingCache(
            MappingProxyType(vectors),
            (),
            str(profile_index + 1) * 64,
            2,
        )
        profiles[profile_id] = planned
        finalized[profile_id] = FinalizedProfile(
            profile_id,
            planned,
            embeddings,
            MappingProxyType({"summary": summary}),
            MappingProxyType({"profile_artifact_sha256": str(profile_index + 6) * 64}),
        )
    snapshot = CanonicalSnapshot(
        MappingProxyType(profiles),
        finalized[PROFILE_IDS[1]].embeddings,
        query_cache,
        qrels,
        source_qrels,
        MappingProxyType({profile_id: () for profile_id in PROFILE_IDS}),
        MappingProxyType({profile_id: () for profile_id in PROFILE_IDS}),
        MappingProxyType({"fixture_sha256": "9" * 64}),
    )
    return snapshot, finalized


@pytest.fixture(scope="module")
def computed_evaluation() -> ChunkSweepEvaluation:
    snapshot, finalized = _evaluation_fixture()
    return evaluate_chunk_sweep(snapshot, finalized, resamples=2, seed=0)


class TestPureEvaluation:
    def test_builds_three_profile_runs_samples_and_report(
        self,
        computed_evaluation: ChunkSweepEvaluation,
    ) -> None:
        assert tuple(computed_evaluation.runs) == PROFILE_IDS
        assert computed_evaluation.numeric_samples["schema"] == CHUNK_SWEEP_SAMPLES_SCHEMA
        assert computed_evaluation.report["schema"] == CHUNK_SWEEP_REPORT_SCHEMA
        contract = cast(dict[str, Any], computed_evaluation.report["execution_contract"])
        assert contract == {
            "required_new_document_vectors": 3_379,
            "required_canonical_exact_reuse": 2_274,
            "required_embedding_batches": 212,
            "profile_cache_evidence": "exact-id-complete-matrix",
            "historical_provider_call_evidence": "not-recorded",
            "per_batch_build_receipts": "not-recorded",
            "evaluation_phase": {
                "provider_calls": 0,
                "query_embedding_calls": 0,
                "rerank_calls": 0,
                "chat_calls": 0,
                "milvus_writes": 0,
            },
        }
        families = cast(dict[str, Any], computed_evaluation.report["families"])
        assert families["origin-source-efficacy"][0]["delta"] == pytest.approx(0.0)
        assert families["origin-source-efficacy"][0]["raw_p"] == pytest.approx(1.0)
        assert len(computed_evaluation.runs[PROFILE_IDS[0]].chunk_runs[RRF_LABEL]) == 980

    def test_rejects_profile_from_another_snapshot(self) -> None:
        snapshot, finalized = _evaluation_fixture()
        wrong = dict(finalized)
        original = wrong[PROFILE_IDS[0]]
        wrong[PROFILE_IDS[0]] = FinalizedProfile(
            original.profile_id,
            cast(PlannedProfile, object()),
            original.embeddings,
            original.artifact,
            original.report,
        )
        with pytest.raises(ValueError, match="not from this snapshot"):
            evaluate_chunk_sweep(snapshot, wrong, resamples=1, seed=0)


def _fake_evaluation() -> ChunkSweepEvaluation:
    query_ids = ("query",)
    rows = (("chunk",),)
    sources = (("source",),)
    runs = {
        profile_id: SweepRuns(
            query_ids,
            {label: rows for label in RUN_LABELS},
            {label: sources for label in RUN_LABELS},
        )
        for profile_id in PROFILE_IDS
    }
    return ChunkSweepEvaluation(
        MappingProxyType(runs),
        MappingProxyType({profile_id: "a" * 64 for profile_id in PROFILE_IDS}),
        MappingProxyType({"schema": "numeric"}),
        MappingProxyType({"schema": "report"}),
    )


class TestScript:
    def test_parse_args_and_rejects_invalid_statistics(self) -> None:
        runner = _runner()
        args = runner._parse_args(
            [
                "--artifacts",
                "a",
                "--curated",
                "c",
                "--resamples",
                "7",
                "--seed",
                "9",
            ]
        )
        assert vars(args) == {
            "artifacts": Path("a"),
            "curated": Path("c"),
            "resamples": 7,
            "seed": 9,
        }
        for argv in (["--resamples", "0"], ["--seed", "-1"], ["--env", ".env"]):
            with pytest.raises(SystemExit):
                runner._parse_args(argv)

    def test_import_is_provider_free(self) -> None:
        probe = (
            "import json, runpy, sys; "
            f"runpy.run_path({str(SCRIPT)!r}, run_name='offline_import_probe'); "
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name.startswith('zhrag.providers'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        assert json.loads(completed.stdout) == []

    def test_publish_orders_report_last_and_cleans_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = ChunkSweepPaths(tmp_path / "artifacts", tmp_path / "curated")
        published: list[str] = []

        def capture(pairs: Any) -> None:
            materialized = tuple(pairs)
            published.extend(Path(target).name for _source, target in materialized)
            replace_files(materialized)

        monkeypatch.setattr(runner, "replace_files", capture)
        target = runner._publish(paths, _fake_evaluation())

        assert target.is_file()
        assert published[-1] == "report.json"
        assert read_json(target) == {"schema": "report"}
        assert list(paths.sweep_root.rglob("*.tmp")) == []

    def test_publication_failure_preserves_old_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = ChunkSweepPaths(tmp_path / "artifacts", tmp_path / "curated")
        report = paths.sweep_root / "report.json"
        write_json(report, {"schema": "old"})
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: (_ for _ in ()).throw(RuntimeError("injected")),
        )

        with pytest.raises(RuntimeError, match="injected"):
            runner._publish(paths, _fake_evaluation())

        assert read_json(report) == {"schema": "old"}
        assert list(paths.sweep_root.rglob("*.tmp")) == []

    def test_main_holds_all_locks_through_load_and_publish(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        artifacts = tmp_path / "artifacts"
        curated = tmp_path / "curated"
        paths = ChunkSweepPaths(artifacts, curated)

        def assert_locked(_paths: ChunkSweepPaths, *, resamples: int, seed: int) -> Any:
            assert (resamples, seed) == (1, 0)
            expected = [
                paths.sweep_root / ".evaluate.lock",
                paths.sweep_root / ".bundle.lock",
                artifacts / ".index.lock",
                paths.eval_root / ".artifacts.lock",
                *(paths.profile_root(profile_id) / ".profile.lock" for profile_id in PROFILE_IDS),
            ]
            assert all(path.exists() for path in expected)
            return _fake_evaluation()

        def publish_locked(_paths: ChunkSweepPaths, _evaluation: Any) -> Path:
            assert (paths.sweep_root / ".bundle.lock").exists()
            assert (paths.eval_root / ".artifacts.lock").exists()
            return paths.sweep_root / "report.json"

        monkeypatch.setattr(runner, "_load_and_evaluate", assert_locked)
        monkeypatch.setattr(runner, "_publish", publish_locked)

        assert (
            runner.main(
                [
                    "--artifacts",
                    str(artifacts),
                    "--curated",
                    str(curated),
                    "--resamples",
                    "1",
                ]
            )
            == 0
        )
        assert not any(path.name.endswith(".lock") for path in artifacts.rglob("*.lock"))

    def test_profile_lock_contention_cleans_outer_operation_lock(self, tmp_path: Path) -> None:
        runner = _runner()
        paths = ChunkSweepPaths(tmp_path / "artifacts", tmp_path / "curated")
        profile_lock = paths.profile_root(PROFILE_IDS[0]) / ".profile.lock"
        write_text(profile_lock, "existing")

        with pytest.raises(SystemExit, match="another writer"):
            runner.main(
                [
                    "--artifacts",
                    str(paths.artifacts),
                    "--curated",
                    str(paths.curated),
                    "--resamples",
                    "1",
                ]
            )

        assert profile_lock.exists()
        assert not (paths.sweep_root / ".evaluate.lock").exists()
