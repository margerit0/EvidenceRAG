"""Hermetic tests for the strict M7 chunk-sweep cache builder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from zhrag.eval.tidb_chunk_sweep import (
    PROFILE_IDS,
    PROFILES,
    VECTOR_WIDTH,
    EmbeddingCache,
    PlannedProfile,
    SourceQrels,
    embedding_cache_fingerprint,
    strict_embedding_cache,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import load_finalized_profile
from zhrag.eval.tidb_quality import QrelsBundle
from zhrag.io_utils import (
    read_json,
    read_jsonl,
    replace_files,
    write_json,
    write_jsonl,
    write_text,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_tidb_chunk_sweep.py"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("test_build_tidb_chunk_sweep_runner", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import build_tidb_chunk_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vector(index: int) -> NDArray[np.float32]:
    vector = np.zeros(VECTOR_WIDTH, dtype=np.float32)
    vector[index % VECTOR_WIDTH] = 1.0
    return vector


def _cache(ids: Sequence[str]) -> EmbeddingCache:
    vectors = {item_id: _vector(index) for index, item_id in enumerate(ids)}
    return EmbeddingCache(
        MappingProxyType(vectors),
        (),
        embedding_cache_fingerprint(ids, vectors, width=VECTOR_WIDTH),
        VECTOR_WIDTH,
    )


def _snapshot(
    runner: ModuleType,
    *,
    profile_id: str,
    reused: tuple[str, ...],
    fresh: tuple[str, ...],
) -> Any:
    ids = (*reused, *fresh)
    profile = PROFILES[profile_id]
    planned = cast(
        PlannedProfile,
        SimpleNamespace(
            profile=profile,
            document_count=450,
            chunk_ids=ids,
            corpus={item_id: f"text-{item_id}" for item_id in ids},
            source_by_chunk={item_id: "source" for item_id in ids},
            corpus_fingerprint="1" * 64,
            chunk_ids_fingerprint="2" * 64,
            source_mapping_fingerprint="3" * 64,
            profile_summary={
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
                "document_set_fingerprint_sha256": "4" * 64,
            },
            exceedance_rows=(),
        ),
    )
    canonical = _cache(reused)
    return runner.CanonicalSnapshot(
        profiles=MappingProxyType({profile_id: planned}),
        canonical_cache=canonical,
        query_cache=canonical,
        qrels=cast(QrelsBundle, object()),
        source_qrels=cast(SourceQrels, object()),
        reusable_ids=MappingProxyType({profile_id: reused}),
        new_ids=MappingProxyType({profile_id: fresh}),
        input_fingerprints=MappingProxyType({"fixture_sha256": "5" * 64}),
    )


class _FakeEmbedder:
    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.calls: list[tuple[tuple[str, ...], int, str]] = []

    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]:
        self.calls.append((tuple(texts), batch, label))
        output: list[list[float]] = []
        for offset in range(0, len(texts), batch):
            count = len(texts[offset : offset + batch])
            vectors = [_vector(offset + index).tolist() for index in range(count)]
            if self.malformed and vectors:
                vectors[-1][0] = cast(Any, True)
            if on_batch is not None:
                on_batch(offset, vectors)
            output.extend(vectors)
        return output


class TestCLI:
    def test_parse_args_freezes_actions(self) -> None:
        runner = _runner()
        args = runner._parse_args(
            [
                "--artifacts",
                "a",
                "--curated",
                "c",
                "--env",
                "e",
                "--profile",
                PROFILE_IDS[0],
                "--embed",
                "--max-batches",
                "1",
            ]
        )
        assert vars(args) == {
            "artifacts": Path("a"),
            "curated": Path("c"),
            "env": Path("e"),
            "profile": PROFILE_IDS[0],
            "embed": True,
            "finalize": False,
            "migrate_metadata": False,
            "adopt_query_cache": False,
            "max_batches": 1,
        }

    @pytest.mark.parametrize(
        "argv",
        (
            ["--embed"],
            ["--finalize"],
            ["--profile", PROFILE_IDS[0]],
            ["--max-batches", "1"],
            ["--profile", PROFILE_IDS[0], "--embed", "--max-batches", "0"],
            ["--target", "256"],
            ["--model", "other"],
        ),
    )
    def test_rejects_incomplete_or_semantic_overrides(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            _runner()._parse_args(argv)

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

    def test_default_main_has_no_filesystem_or_provider_side_effects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner()
        sentinel = object()
        monkeypatch.setattr(runner, "_load_snapshot", lambda _paths: sentinel)
        monkeypatch.setattr(runner, "_status", lambda value: "ok" if value is sentinel else "bad")
        monkeypatch.setattr(
            runner,
            "_create_client",
            lambda _path: pytest.fail("read-only status created a provider client"),
        )
        artifacts = tmp_path / "absent-artifacts"
        curated = tmp_path / "absent-curated"

        assert runner.main(["--artifacts", str(artifacts), "--curated", str(curated)]) == 0

        assert not artifacts.exists()
        assert not curated.exists()
        assert "provider calls=0, files written=0" in capsys.readouterr().out

    def test_read_only_status_refuses_active_writer_without_creating_files(
        self, tmp_path: Path
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        lock = paths.artifacts / ".index.lock"
        lock.parent.mkdir(parents=True)
        write_text(lock, "existing")
        before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

        with pytest.raises(SystemExit, match="locks exist"):
            runner._check_read_only_locks(paths)

        after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
        assert after == before

    def test_adopt_query_cache_is_an_exclusive_offline_action(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner()
        certificate = tmp_path / "query_cache_provenance.json"
        monkeypatch.setattr(runner, "_adopt_query_cache", lambda _paths: certificate)
        monkeypatch.setattr(
            runner,
            "_create_client",
            lambda _path: pytest.fail("offline adoption created a provider client"),
        )

        assert runner.main(["--artifacts", str(tmp_path), "--adopt-query-cache"]) == 0

        output = capsys.readouterr().out
        assert "provider calls=0" in output
        assert "historical request-text binding remains not-observed" in output

    @pytest.mark.parametrize(
        "argv",
        (
            ["--adopt-query-cache", "--embed", "--profile", PROFILE_IDS[0]],
            ["--adopt-query-cache", "--finalize", "--profile", PROFILE_IDS[1]],
            ["--adopt-query-cache", "--profile", PROFILE_IDS[0]],
            ["--adopt-query-cache", "--max-batches", "1"],
            ["--migrate-metadata", "--profile", PROFILE_IDS[0]],
            ["--migrate-metadata", "--max-batches", "1"],
            ["--migrate-metadata", "--env", "custom.env"],
        ),
    )
    def test_rejects_adoption_action_overrides(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            _runner()._parse_args(argv)


class TestQueryCacheAdoption:
    @staticmethod
    def _stub_inputs(
        runner: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        *,
        certificate: dict[str, object],
    ) -> Any:
        snapshot = SimpleNamespace(
            query_cache=object(),
            canonical_cache=object(),
            profiles={PROFILE_IDS[1]: SimpleNamespace(chunk_ids=("chunk",))},
            qrels=object(),
        )
        monkeypatch.setattr(
            runner,
            "_load_snapshot",
            lambda _paths, *, require_query_cache_provenance=True: snapshot,
        )
        monkeypatch.setattr(runner, "read_jsonl", lambda _path: ({"fixture": True},))
        monkeypatch.setattr(
            runner,
            "verify_query_cache_dense_runs",
            lambda *_args: "d" * 64,
        )
        monkeypatch.setattr(
            runner,
            "build_query_cache_provenance",
            lambda **_kwargs: certificate,
        )
        monkeypatch.setattr(runner, "file_sha256", lambda _path: "f" * 64)
        return snapshot

    def test_adopts_under_all_locks_and_is_idempotent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        certificate = {"schema": "fixture", "historical_request_text_binding": "not-observed"}
        snapshot = self._stub_inputs(runner, monkeypatch, certificate=certificate)
        loads = 0
        publications = 0

        def load(_paths: Any, *, require_query_cache_provenance: bool = True) -> Any:
            nonlocal loads
            loads += 1
            assert require_query_cache_provenance is False
            assert all(
                path.exists()
                for path in (
                    paths.sweep_root / ".query-cache-adopt.lock",
                    paths.artifacts / ".index.lock",
                    paths.eval_root / ".artifacts.lock",
                )
            )
            return snapshot

        def publish(staged: Any) -> None:
            nonlocal publications
            publications += 1
            assert all(
                path.exists()
                for path in (
                    paths.sweep_root / ".query-cache-adopt.lock",
                    paths.artifacts / ".index.lock",
                    paths.eval_root / ".artifacts.lock",
                )
            )
            replace_files(staged)

        monkeypatch.setattr(runner, "_load_snapshot", load)
        monkeypatch.setattr(runner, "replace_files", publish)

        assert runner._adopt_query_cache(paths) == paths.query_cache_provenance
        assert read_json(paths.query_cache_provenance) == certificate
        assert runner._adopt_query_cache(paths) == paths.query_cache_provenance

        assert loads == 2
        assert publications == 1
        assert not (
            paths.query_cache_provenance.with_name("query_cache_provenance.json.tmp")
        ).exists()
        assert not (paths.sweep_root / ".query-cache-adopt.lock").exists()
        assert not (paths.artifacts / ".index.lock").exists()
        assert not (paths.eval_root / ".artifacts.lock").exists()

    def test_rejects_existing_certificate_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        expected = {"schema": "expected"}
        forged = {"schema": "forged"}
        self._stub_inputs(runner, monkeypatch, certificate=expected)
        write_json(paths.query_cache_provenance, forged)
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: pytest.fail("drifted certificate was replaced"),
        )

        with pytest.raises(ValueError, match="provenance drift"):
            runner._adopt_query_cache(paths)

        assert read_json(paths.query_cache_provenance) == forged
        assert not paths.query_cache_provenance.with_name(
            "query_cache_provenance.json.tmp"
        ).exists()

    def test_publication_failure_leaves_no_certificate_or_staging_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        self._stub_inputs(runner, monkeypatch, certificate={"schema": "fixture"})
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: (_ for _ in ()).throw(RuntimeError("injected")),
        )

        with pytest.raises(RuntimeError, match="injected"):
            runner._adopt_query_cache(paths)

        assert not paths.query_cache_provenance.exists()
        assert not paths.query_cache_provenance.with_name(
            "query_cache_provenance.json.tmp"
        ).exists()
        assert not (paths.sweep_root / ".query-cache-adopt.lock").exists()
        assert not (paths.artifacts / ".index.lock").exists()
        assert not (paths.eval_root / ".artifacts.lock").exists()


class TestCheckpointing:
    def test_fixed_batch_and_max_batches(self, tmp_path: Path) -> None:
        runner = _runner()
        ids = tuple(f"id-{index:02d}" for index in range(20))
        client = _FakeEmbedder()
        cache = tmp_path / "cache.jsonl"

        completed = runner._append_provider_batches(
            client=client,
            cache_path=cache,
            ids=ids,
            texts=tuple(f"text-{index}" for index in range(20)),
            max_batches=1,
            label="fixture",
        )

        assert completed == 1
        assert len(client.calls) == 1
        sent, batch, label = client.calls[0]
        assert len(sent) == 16
        assert batch == 16
        assert label == "fixture"
        assert [row["doc_id"] for row in read_jsonl(cache)] == list(ids[:16])

    def test_bad_row_appends_none_of_its_batch(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        with pytest.raises(ValueError, match="not numeric"):
            runner._append_provider_batches(
                client=_FakeEmbedder(malformed=True),
                cache_path=cache,
                ids=("a", "b"),
                texts=("A", "B"),
                max_batches=None,
                label="fixture",
            )
        assert not cache.exists()

    def test_profile_embed_seeds_exact_reuse_then_resumes(self, tmp_path: Path) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[0]
        reused = ("canonical",)
        fresh = tuple(f"new-{index:02d}" for index in range(16))
        snapshot = _snapshot(runner, profile_id=profile_id, reused=reused, fresh=fresh)
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        client = _FakeEmbedder()

        completed, batches = runner._embed_profile(
            paths,
            snapshot,
            profile_id,
            client=client,
            endpoint_url="https://provider.example/v1/embeddings",
            max_batches=1,
            snapshot_loader=lambda _paths: snapshot,
        )

        assert (completed, batches) == (16, 1)
        files = runner._profile_files(paths, profile_id)
        loaded = strict_embedding_cache(
            files.cache,
            (*reused, *fresh),
            width=VECTOR_WIDTH,
            allow_partial=False,
        )
        assert loaded.missing_ids == ()
        assert [row["doc_id"] for row in read_jsonl(files.cache)] == [*reused, *fresh]
        assert not (files.root / ".profile.lock").exists()

        resumed = _FakeEmbedder()
        assert runner._embed_profile(
            paths,
            snapshot,
            profile_id,
            client=resumed,
            endpoint_url="https://provider.example/v1/embeddings",
            max_batches=None,
            snapshot_loader=lambda _paths: snapshot,
        ) == (16, 0)
        assert resumed.calls == [((), 16, profile_id)]

    def test_sidecar_drift_stops_before_provider(self, tmp_path: Path) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[0]
        snapshot = _snapshot(
            runner,
            profile_id=profile_id,
            reused=("canonical",),
            fresh=("new",),
        )
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        files = runner._profile_files(paths, profile_id)
        files.root.mkdir(parents=True)
        write_json(files.sidecar, {"schema": "forged"})
        client = _FakeEmbedder()

        with pytest.raises(ValueError, match="sidecar drift"):
            runner._embed_profile(
                paths,
                snapshot,
                profile_id,
                client=client,
                endpoint_url="https://provider.example/v1/embeddings",
                max_batches=1,
                snapshot_loader=lambda _paths: snapshot,
            )
        assert client.calls == []
        assert not files.cache.exists()
        assert not (files.root / ".profile.lock").exists()

    def test_snapshot_drift_after_provider_batch_stops_publication(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[0]
        reused = ("canonical",)
        fresh = tuple(f"new-{index:02d}" for index in range(16))
        snapshot = _snapshot(runner, profile_id=profile_id, reused=reused, fresh=fresh)
        changed = _snapshot(runner, profile_id=profile_id, reused=reused, fresh=fresh)
        changed = runner.CanonicalSnapshot(
            profiles=changed.profiles,
            canonical_cache=changed.canonical_cache,
            query_cache=changed.query_cache,
            qrels=changed.qrels,
            source_qrels=changed.source_qrels,
            reusable_ids=changed.reusable_ids,
            new_ids=changed.new_ids,
            input_fingerprints=MappingProxyType({"fixture_sha256": "6" * 64}),
        )
        loads = iter((snapshot, changed))
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")

        with pytest.raises(ValueError, match="canonical M7 inputs changed"):
            runner._embed_profile(
                paths,
                snapshot,
                profile_id,
                client=_FakeEmbedder(),
                endpoint_url="https://provider.example/v1/embeddings",
                max_batches=1,
                snapshot_loader=lambda _paths: next(loads),
            )

        files = runner._profile_files(paths, profile_id)
        assert not files.report.exists()
        assert len(tuple(read_jsonl(files.cache))) == len(reused) + len(fresh)
        assert not (files.root / ".profile.lock").exists()
        assert not (paths.artifacts / ".index.lock").exists()
        assert not (paths.eval_root / ".artifacts.lock").exists()

    def test_snapshot_drift_before_provider_stops_without_calling_it(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[0]
        snapshot = _snapshot(
            runner,
            profile_id=profile_id,
            reused=("canonical",),
            fresh=("new",),
        )
        changed = runner.CanonicalSnapshot(
            profiles=snapshot.profiles,
            canonical_cache=snapshot.canonical_cache,
            query_cache=snapshot.query_cache,
            qrels=snapshot.qrels,
            source_qrels=snapshot.source_qrels,
            reusable_ids=snapshot.reusable_ids,
            new_ids=snapshot.new_ids,
            input_fingerprints=MappingProxyType({"fixture_sha256": "6" * 64}),
        )
        client = _FakeEmbedder()

        with pytest.raises(ValueError, match="canonical M7 inputs changed"):
            runner._embed_profile(
                runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated"),
                snapshot,
                profile_id,
                client=client,
                endpoint_url="https://provider.example/v1/embeddings",
                max_batches=1,
                snapshot_loader=lambda _paths: changed,
            )

        assert client.calls == []

    def test_rejects_non_prefix_checkpoint(self) -> None:
        runner = _runner()
        cache = _cache(("second",))
        with pytest.raises(ValueError, match="stable prefix"):
            runner._check_new_prefix(cache, ("first", "second"))


class TestFinalize:
    def test_400_reference_publishes_marker_last_and_is_idempotent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[1]
        snapshot = _snapshot(
            runner,
            profile_id=profile_id,
            reused=("canonical",),
            fresh=(),
        )
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        paths.artifacts.mkdir(parents=True)
        write_jsonl(
            paths.canonical_cache,
            [{"doc_id": "canonical", "embedding": _vector(0).tolist()}],
        )
        published: list[str] = []

        def capture(staged: Any) -> None:
            pairs = tuple(staged)
            published.extend(Path(target).name for _source, target in pairs)
            replace_files(pairs)

        monkeypatch.setattr(runner, "replace_files", capture)

        report = runner._finalize_profile(
            paths,
            snapshot,
            profile_id,
            snapshot_loader=lambda _paths: snapshot,
        )

        assert report.is_file()
        assert published[-1] == "cache_report.json"
        files = runner._profile_files(paths, profile_id)
        marker = read_json(files.report)
        assert marker["complete"] is True
        assert marker["cache_mode"] == "canonical-reference-no-copy"
        assert files.profile.is_file()
        assert files.ledger.is_file()
        assert files.sidecar.is_file()
        assert not files.cache.exists()
        loaded = load_finalized_profile(paths, snapshot, profile_id)
        assert loaded.embeddings.fingerprint == snapshot.canonical_cache.fingerprint
        assert loaded.public_summary["profile_id"] == profile_id
        first = read_json(files.report)

        published.clear()
        assert (
            runner._finalize_profile(
                paths,
                snapshot,
                profile_id,
                snapshot_loader=lambda _paths: snapshot,
            )
            == files.report
        )
        assert read_json(files.report) == first
        assert published == []

    def test_finalize_rejects_stale_snapshot_before_writing(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[1]
        snapshot = _snapshot(
            runner,
            profile_id=profile_id,
            reused=("canonical",),
            fresh=(),
        )
        changed = runner.CanonicalSnapshot(
            profiles=snapshot.profiles,
            canonical_cache=snapshot.canonical_cache,
            query_cache=snapshot.query_cache,
            qrels=snapshot.qrels,
            source_qrels=snapshot.source_qrels,
            reusable_ids=snapshot.reusable_ids,
            new_ids=snapshot.new_ids,
            input_fingerprints=MappingProxyType({"fixture_sha256": "6" * 64}),
        )
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        paths.artifacts.mkdir(parents=True)
        write_jsonl(
            paths.canonical_cache,
            [{"doc_id": "canonical", "embedding": _vector(0).tolist()}],
        )

        with pytest.raises(ValueError, match="canonical M7 inputs changed"):
            runner._finalize_profile(
                paths,
                snapshot,
                profile_id,
                snapshot_loader=lambda _paths: changed,
            )

        files = runner._profile_files(paths, profile_id)
        assert not files.report.exists()
        assert not files.profile.exists()
        assert not files.ledger.exists()
        assert not files.sidecar.exists()

    def test_finalize_holds_profile_bundle_index_and_artifact_locks(
        self,
        tmp_path: Path,
    ) -> None:
        runner = _runner()
        profile_id = PROFILE_IDS[1]
        snapshot = _snapshot(
            runner,
            profile_id=profile_id,
            reused=("canonical",),
            fresh=(),
        )
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        paths.artifacts.mkdir(parents=True)
        write_jsonl(
            paths.canonical_cache,
            [{"doc_id": "canonical", "embedding": _vector(0).tolist()}],
        )

        def assert_locked(_paths: Any) -> Any:
            expected = (
                paths.profile_root(profile_id) / ".profile.lock",
                paths.sweep_root / ".bundle.lock",
                paths.artifacts / ".index.lock",
                paths.eval_root / ".artifacts.lock",
            )
            assert all(path.exists() for path in expected)
            return snapshot

        runner._finalize_profile(
            paths,
            snapshot,
            profile_id,
            snapshot_loader=assert_locked,
        )

    def test_migration_action_is_provider_free_and_holds_all_locks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        observed: list[Path] = []

        def migrate(_paths: Any) -> tuple[Path, ...]:
            expected = (
                paths.sweep_root / ".migrate.lock",
                *(paths.profile_root(profile_id) / ".profile.lock" for profile_id in PROFILE_IDS),
                paths.eval_root / ".pool.lock",
                paths.eval_root / ".qrels.lock",
                paths.eval_root / ".quality.lock",
                paths.sweep_root / ".bundle.lock",
                paths.artifacts / ".index.lock",
                paths.eval_root / ".artifacts.lock",
            )
            observed.extend(expected)
            assert all(path.is_file() for path in expected)
            return ()

        monkeypatch.setattr(runner, "_migrate_metadata", migrate)
        monkeypatch.setattr(
            runner,
            "_load_snapshot",
            lambda _paths: pytest.fail("migration should not use provider-backed snapshot loader"),
        )
        assert runner._run_migration(paths) == ()
        assert all(not path.exists() for path in observed)

    def test_migration_v2_noop_authenticates_and_never_replaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        monkeypatch.setattr(runner, "classify_metadata_generation", lambda _paths: "v2")
        monkeypatch.setattr(runner, "_authenticate_v2_bundle", lambda _paths: None)
        monkeypatch.setattr(
            runner,
            "replace_files",
            lambda _pairs: pytest.fail("authenticated v2 no-op must not replace files"),
        )
        assert runner._migrate_metadata(paths) == ()

    @pytest.mark.parametrize("generation", ["mixed", "unknown"])
    def test_migration_rejects_mixed_or_unknown_generation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        generation: str,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        monkeypatch.setattr(runner, "classify_metadata_generation", lambda _paths: generation)
        with pytest.raises(ValueError, match=f"generation '{generation}'"):
            runner._migrate_metadata(paths)

    def test_migration_cleans_registered_staging_after_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _runner()
        paths = runner.BuildPaths(tmp_path / "artifacts", tmp_path / "curated")
        monkeypatch.setattr(runner, "classify_metadata_generation", lambda _paths: "v1")
        monkeypatch.setattr(runner, "_load_snapshot", lambda _paths: object())
        monkeypatch.setattr(
            runner,
            "load_legacy_finalized_profile",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("before staging")),
        )
        with pytest.raises(RuntimeError, match="before staging"):
            runner._migrate_metadata(paths)
        assert not list(paths.sweep_root.rglob("*migrate.tmp"))
