"""Build strict local caches for the frozen TiDB chunk-size sweep.

The default status action is read-only: it re-plans all three profiles and
certifies canonical state, qrels, and embedding caches without creating a
sidecar or provider client. Explicit ``--adopt-query-cache``,
``--migrate-metadata``, and ``--finalize`` actions may publish offline metadata;
only ``--embed`` may call the document embedding provider. Finalization and
metadata migration publish their bundle marker last.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Protocol, cast

from zhrag.eval.tidb_chunk_sweep import (
    BATCH_SIZE,
    DOCUMENT_EMBEDDING_MODEL,
    DOCUMENT_PROMPT,
    PROFILE_IDS,
    PROFILES,
    VECTOR_WIDTH,
    EmbeddingCache,
    SweepRuns,
    numeric_samples_sha256,
    parse_sweep_runs,
    strict_embedding_cache,
    validate_exact_embedding_reuse,
    validate_numeric_samples,
    validate_report,
    validated_embedding_rows,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    CanonicalSnapshot,
    ChunkSweepPaths,
    FinalizedProfile,
    ProfileFiles,
    build_query_cache_provenance,
    classify_metadata_generation,
    file_sha256,
    load_canonical_snapshot,
    load_finalized_profile,
    load_legacy_finalized_profile,
    migrate_numeric_samples_v1,
    migrate_report_v1,
    profile_artifact,
    profile_report,
    profile_sidecar,
    validate_profile_sidecar,
    verify_query_cache_dense_runs,
)
from zhrag.io_utils import (
    append_jsonl,
    exclusive_lock,
    read_json,
    read_jsonl,
    replace_files,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
CURATED = ROOT / "tidb-rag-curated"
ENV_FILE = ROOT / ".env"
BuildPaths = ChunkSweepPaths


class _BatchEmbedder(Protocol):
    def embed_all(
        self,
        texts: Sequence[str],
        *,
        batch: int = 16,
        label: str = "embed",
        on_batch: Callable[[int, list[list[float]]], None] | None = None,
    ) -> list[list[float]]: ...


# The import block and module constants are deliberately kept provider-free.
# ``_create_client`` remains the only path that imports the paid transport.
MIGRATION_LOCK = ".migrate.lock"
MIGRATION_STAGE_SUFFIX = ".migrate.tmp"


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--curated", type=Path, default=CURATED)
    parser.add_argument("--env", type=Path, default=ENV_FILE)
    parser.add_argument("--profile", choices=PROFILE_IDS)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--embed", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument(
        "--migrate-metadata",
        action="store_true",
        help="offline-authenticate and migrate the M7 v1 metadata bundle to v2",
    )
    action.add_argument(
        "--adopt-query-cache",
        action="store_true",
        help="offline-authenticate the legacy query cache against frozen dense runs",
    )
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    if (args.embed or args.finalize) and args.profile is None:
        parser.error("--embed/--finalize requires --profile")
    if (not args.embed and not args.finalize) and args.profile is not None:
        parser.error("--profile is only valid with --embed or --finalize")
    if args.max_batches is not None:
        if not args.embed:
            parser.error("--max-batches requires --embed")
        if args.max_batches < 1:
            parser.error("--max-batches must be positive")
    if args.migrate_metadata and args.env != ENV_FILE:
        parser.error("--env is not valid with --migrate-metadata")
    return args


def _load_snapshot(
    paths: BuildPaths,
    *,
    require_query_cache_provenance: bool = True,
) -> CanonicalSnapshot:
    return load_canonical_snapshot(
        paths,
        require_query_cache_provenance=require_query_cache_provenance,
    )


def _check_read_only_locks(paths: BuildPaths) -> None:
    candidates = (
        paths.artifacts / ".index.lock",
        paths.eval_root / ".artifacts.lock",
        paths.eval_root / ".pool.lock",
        paths.eval_root / ".qrels.lock",
        paths.sweep_root / ".bundle.lock",
        *(paths.profile_root(profile_id) / ".profile.lock" for profile_id in PROFILE_IDS),
    )
    active = [str(path) for path in candidates if path.exists()]
    if active:
        raise SystemExit(
            f"! refusing a non-atomic preflight while locks exist: {', '.join(active)}"
        )


def _snapshot_with_locks(
    paths: BuildPaths,
    *,
    loader: Callable[[BuildPaths], CanonicalSnapshot] | None = None,
) -> CanonicalSnapshot:
    with (
        exclusive_lock(paths.artifacts / ".index.lock"),
        exclusive_lock(paths.eval_root / ".artifacts.lock"),
    ):
        return (loader or _load_snapshot)(paths)


def _snapshot_contract(
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> tuple[object, ...]:
    planned = snapshot.profiles[profile_id]
    return (
        tuple(sorted(snapshot.input_fingerprints.items())),
        planned.profile.fingerprint,
        planned.corpus_fingerprint,
        planned.chunk_ids_fingerprint,
        planned.source_mapping_fingerprint,
        planned.chunk_ids,
        snapshot.reusable_ids[profile_id],
        snapshot.new_ids[profile_id],
    )


def _require_same_snapshot(
    frozen: CanonicalSnapshot,
    current: CanonicalSnapshot,
    profile_id: str,
) -> None:
    if _snapshot_contract(frozen, profile_id) != _snapshot_contract(current, profile_id):
        raise ValueError(
            f"{profile_id}: canonical M7 inputs changed during this operation; "
            "refusing to reuse or publish a stale cache"
        )


def _profile_files(paths: BuildPaths, profile_id: str) -> ProfileFiles:
    return paths.profile_files(profile_id)


def _sidecar(
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    endpoint_url: str | None,
    endpoint_provenance: str,
) -> dict[str, object]:
    return profile_sidecar(
        snapshot,
        profile_id,
        endpoint_url=endpoint_url,
        endpoint_provenance=endpoint_provenance,
    )


def _read_sidecar_for_finalize(
    files: ProfileFiles,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> tuple[dict[str, object], str]:
    return validate_profile_sidecar(read_json(files.sidecar), snapshot, profile_id)


def _check_new_prefix(cache: EmbeddingCache, new_ids: Sequence[str]) -> int:
    present = tuple(item_id for item_id in new_ids if item_id in cache.vectors)
    if present != tuple(new_ids[: len(present)]):
        raise ValueError("paid cache rows are not a stable prefix of the frozen new-ID order")
    if len(present) < len(new_ids) and len(present) % BATCH_SIZE != 0:
        raise ValueError("partial paid cache ends inside a frozen provider batch")
    return len(present)


def _seed_canonical(
    files: ProfileFiles,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> EmbeddingCache:
    planned = snapshot.profiles[profile_id]
    cache = strict_embedding_cache(
        files.cache,
        planned.chunk_ids,
        width=VECTOR_WIDTH,
        allow_partial=True,
    )
    missing_reuse = [
        item_id for item_id in snapshot.reusable_ids[profile_id] if item_id not in cache.vectors
    ]
    present_reuse = tuple(
        item_id for item_id in snapshot.reusable_ids[profile_id] if item_id in cache.vectors
    )
    validate_exact_embedding_reuse(cache, snapshot.canonical_cache, present_reuse)
    if missing_reuse:
        append_jsonl(
            files.cache,
            (
                {
                    "doc_id": item_id,
                    "embedding": snapshot.canonical_cache.vectors[item_id].tolist(),
                }
                for item_id in missing_reuse
            ),
        )
        cache = strict_embedding_cache(
            files.cache,
            planned.chunk_ids,
            width=VECTOR_WIDTH,
            allow_partial=True,
        )
        validate_exact_embedding_reuse(
            cache,
            snapshot.canonical_cache,
            snapshot.reusable_ids[profile_id],
        )
    return cache


def _append_provider_batches(
    *,
    client: _BatchEmbedder,
    cache_path: Path,
    ids: Sequence[str],
    texts: Sequence[str],
    max_batches: int | None,
    label: str,
) -> int:
    if len(ids) != len(texts):
        raise ValueError("provider IDs and texts are not aligned")
    selected_count = len(ids)
    if max_batches is not None:
        selected_count = min(selected_count, max_batches * BATCH_SIZE)
    selected_ids = tuple(ids[:selected_count])
    selected_texts = tuple(texts[:selected_count])
    completed = 0

    def checkpoint(offset: int, vectors: list[list[float]]) -> None:
        nonlocal completed
        batch_ids = selected_ids[offset : offset + len(vectors)]
        rows = validated_embedding_rows(batch_ids, vectors, width=VECTOR_WIDTH)
        append_jsonl(cache_path, rows)
        completed += 1

    client.embed_all(
        [DOCUMENT_PROMPT + text for text in selected_texts],
        batch=BATCH_SIZE,
        label=label,
        on_batch=checkpoint,
    )
    return completed


def _create_client(env_path: Path) -> tuple[_BatchEmbedder, str]:
    # Deliberately lazy: preflight/finalize must not import provider modules,
    # read .env, or construct a transport.
    from zhrag.providers.embedding import (  # noqa: PLC0415
        EmbeddingClient,
        EmbeddingConfig,
        load_env,
    )

    if not env_path.is_file():
        raise ValueError(".env is missing")
    config = EmbeddingConfig.from_env(load_env(env_path))
    if config.model != DOCUMENT_EMBEDDING_MODEL:
        raise ValueError("Embedding_MODEL_NAME differs from the frozen M7 model")
    return EmbeddingClient(config), config.url


def _embed_profile(
    paths: BuildPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    client: _BatchEmbedder,
    endpoint_url: str,
    max_batches: int | None,
    snapshot_loader: Callable[[BuildPaths], CanonicalSnapshot] | None = None,
) -> tuple[int, int]:
    profile = PROFILES[profile_id]
    if profile.expected_new_vectors == 0:
        raise ValueError("the 400/600 profile is a canonical reference and cannot be embedded")
    files = _profile_files(paths, profile_id)
    with exclusive_lock(files.root / ".profile.lock"):
        current = _snapshot_with_locks(paths, loader=snapshot_loader)
        _require_same_snapshot(snapshot, current, profile_id)
        snapshot = current
        expected_sidecar = _sidecar(
            snapshot,
            profile_id,
            endpoint_url=endpoint_url,
            endpoint_provenance="recorded-live-endpoint",
        )
        if files.report.exists():
            raise ValueError(f"{profile_id}: finalized profile is immutable")
        if files.sidecar.exists():
            if read_json(files.sidecar) != expected_sidecar:
                raise ValueError(f"{profile_id}: immutable cache sidecar drift")
        else:
            if files.cache.exists() and files.cache.stat().st_size > 0:
                raise ValueError(f"{profile_id}: non-empty cache has no M7 sidecar")
            write_json(files.sidecar, expected_sidecar)

        cache = _seed_canonical(files, snapshot, profile_id)
        new_ids = snapshot.new_ids[profile_id]
        done = _check_new_prefix(cache, new_ids)
        pending = tuple(new_ids[done:])
        texts = tuple(snapshot.profiles[profile_id].corpus[item_id] for item_id in pending)
        try:
            batches = _append_provider_batches(
                client=client,
                cache_path=files.cache,
                ids=pending,
                texts=texts,
                max_batches=max_batches,
                label=profile_id,
            )
        finally:
            current = _snapshot_with_locks(paths, loader=snapshot_loader)
            _require_same_snapshot(snapshot, current, profile_id)
        checked = strict_embedding_cache(
            files.cache,
            snapshot.profiles[profile_id].chunk_ids,
            width=VECTOR_WIDTH,
            allow_partial=True,
        )
        completed_vectors = _check_new_prefix(checked, new_ids)
        return completed_vectors, batches


def _stage(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _finalize_profile(
    paths: BuildPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    snapshot_loader: Callable[[BuildPaths], CanonicalSnapshot] | None = None,
) -> Path:
    files = _profile_files(paths, profile_id)
    staged_ledger = _stage(files.ledger)
    staged_profile = _stage(files.profile)
    staged_report = _stage(files.report)
    staged_sidecar = _stage(files.sidecar)
    staged = (staged_ledger, staged_profile, staged_report, staged_sidecar)
    with (
        exclusive_lock(files.root / ".profile.lock"),
        exclusive_lock(paths.sweep_root / ".bundle.lock"),
        exclusive_lock(paths.artifacts / ".index.lock"),
        exclusive_lock(paths.eval_root / ".artifacts.lock"),
    ):
        current = (snapshot_loader or _load_snapshot)(paths)
        _require_same_snapshot(snapshot, current, profile_id)
        snapshot = current
        try:
            if profile_id == PROFILE_IDS[1]:
                cache = snapshot.canonical_cache
                cache_path = paths.canonical_cache
                cache_mode = "canonical-reference-no-copy"
                endpoint_provenance = "legacy-unknown"
                sidecar = _sidecar(
                    snapshot,
                    profile_id,
                    endpoint_url=None,
                    endpoint_provenance=endpoint_provenance,
                )
                write_json(staged_sidecar, sidecar)
                sidecar_path = staged_sidecar
            else:
                _sidecar_value, endpoint_provenance = _read_sidecar_for_finalize(
                    files,
                    snapshot,
                    profile_id,
                )
                cache = strict_embedding_cache(
                    files.cache,
                    snapshot.profiles[profile_id].chunk_ids,
                    width=VECTOR_WIDTH,
                    allow_partial=False,
                )
                if _check_new_prefix(cache, snapshot.new_ids[profile_id]) != len(
                    snapshot.new_ids[profile_id]
                ):
                    raise ValueError(f"{profile_id}: cache is incomplete")
                cache_path = files.cache
                cache_mode = "profile-local-complete"
                sidecar_path = files.sidecar

            write_jsonl(
                staged_ledger,
                (dict(row) for row in snapshot.profiles[profile_id].exceedance_rows),
            )
            artifact = profile_artifact(
                snapshot,
                profile_id,
                matrix_fingerprint=cache.fingerprint,
                cache_mode=cache_mode,
            )
            write_json(staged_profile, artifact)
            report = profile_report(
                profile_id=profile_id,
                matrix_fingerprint=cache.fingerprint,
                cache_mode=cache_mode,
                endpoint_provenance=endpoint_provenance,
                profile_sha256=file_sha256(staged_profile),
                ledger_sha256=file_sha256(staged_ledger),
                cache_sha256=file_sha256(cache_path),
                sidecar_sha256=file_sha256(sidecar_path),
            )
            write_json(staged_report, report)

            if files.report.exists():
                if read_json(files.report) != report:
                    raise ValueError(f"{profile_id}: finalized profile marker drift")
                if (
                    not files.profile.is_file()
                    or not files.ledger.is_file()
                    or not files.sidecar.is_file()
                    or file_sha256(files.profile) != report["profile_artifact_sha256"]
                    or file_sha256(files.ledger) != report["split_exceedance_ledger_sha256"]
                    or file_sha256(files.sidecar) != report["cache_sidecar_sha256"]
                    or file_sha256(cache_path) != report["cache_file_sha256"]
                ):
                    raise ValueError(f"{profile_id}: finalized profile data drift")
                return files.report

            replacements: list[tuple[Path, Path]] = []
            if profile_id == PROFILE_IDS[1]:
                replacements.append((staged_sidecar, files.sidecar))
            replacements.extend(
                (
                    (staged_ledger, files.ledger),
                    (staged_profile, files.profile),
                    (staged_report, files.report),
                )
            )
            replace_files(replacements)
            return files.report
        finally:
            for path in staged:
                path.unlink(missing_ok=True)


def _adopt_query_cache(paths: BuildPaths) -> Path:
    operation_lock = paths.sweep_root / ".query-cache-adopt.lock"
    with (
        exclusive_lock(operation_lock),
        exclusive_lock(paths.artifacts / ".index.lock"),
        exclusive_lock(paths.eval_root / ".artifacts.lock"),
    ):
        snapshot = _load_snapshot(paths, require_query_cache_provenance=False)
        dense_runs_fingerprint = verify_query_cache_dense_runs(
            snapshot.query_cache,
            snapshot.canonical_cache,
            snapshot.profiles[PROFILE_IDS[1]].chunk_ids,
            snapshot.qrels,
            tuple(read_jsonl(paths.runs)),
        )
        query_sidecar = paths.query_cache.with_name(f"{paths.query_cache.name}.meta.json")
        provenance = build_query_cache_provenance(
            query_cache=snapshot.query_cache,
            qrels=snapshot.qrels,
            dense_runs_fingerprint=dense_runs_fingerprint,
            query_cache_file_sha256=file_sha256(paths.query_cache),
            legacy_sidecar_file_sha256=file_sha256(query_sidecar),
            quality_report_file_sha256=file_sha256(paths.quality_report),
        )
        target = paths.query_cache_provenance
        if target.exists():
            if read_json(target) != provenance:
                raise ValueError("immutable M7 query-cache provenance drift")
            return target
        staged = target.with_name(f"{target.name}.tmp")
        try:
            write_json(staged, provenance)
            replace_files(((staged, target),))
        finally:
            staged.unlink(missing_ok=True)
        return target


def _migration_stage(path: Path) -> Path:
    return path.with_name(f"{path.name}{MIGRATION_STAGE_SUFFIX}")


def _statistics_design(report: Mapping[str, object]) -> tuple[int, int]:
    design = report.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("chunk-sweep report design is malformed")
    resamples = design.get("resamples")
    seed = design.get("seed")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (1 << 63) - 1
    ):
        raise ValueError("chunk-sweep report design resamples/seed are invalid")
    return resamples, seed


def _load_profile_runs(
    paths: BuildPaths,
    profiles: Mapping[str, FinalizedProfile],
) -> dict[str, SweepRuns]:
    runs: dict[str, SweepRuns] = {}
    for profile_id in PROFILE_IDS:
        run_path = paths.sweep_root / "runs" / f"{profile_id}.jsonl"
        runs[profile_id] = parse_sweep_runs(
            read_jsonl(run_path),
            profiles[profile_id].planned.source_by_chunk,
        )
    return runs


def _authenticate_statistics(
    paths: BuildPaths,
    snapshot: CanonicalSnapshot,
    profiles: Mapping[str, FinalizedProfile],
    samples: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    """Bind samples and report to the authenticated frozen raw run bundle."""
    # Imported lazily to keep the builder's module import provider-free without
    # introducing an artifacts/evaluation module cycle.
    from zhrag.eval.tidb_chunk_sweep_evaluation import (  # noqa: PLC0415
        evaluate_chunk_sweep,
    )

    resamples, seed = _statistics_design(report)
    evaluation = evaluate_chunk_sweep(
        snapshot,
        profiles,
        resamples=resamples,
        seed=seed,
    )
    stored_runs = _load_profile_runs(paths, profiles)
    for profile_id in PROFILE_IDS:
        if stored_runs[profile_id] != evaluation.runs[profile_id]:
            raise ValueError(f"{profile_id}: stored runs differ from deterministic rebuild")
    if dict(evaluation.numeric_samples) != dict(samples):
        raise ValueError("numeric samples differ from authenticated raw runs")
    if dict(evaluation.report) != dict(report):
        raise ValueError("report differs from authenticated deterministic rebuild")


def _authenticate_v2_bundle(paths: BuildPaths) -> None:
    snapshot = _load_snapshot(paths)
    profiles = {
        profile_id: load_finalized_profile(paths, snapshot, profile_id)
        for profile_id in PROFILE_IDS
    }
    samples = read_json(paths.sweep_root / "numeric_samples.json")
    report = read_json(paths.sweep_root / "report.json")
    if not isinstance(samples, Mapping) or not isinstance(report, Mapping):
        raise ValueError("v2 samples and report must be objects")
    validate_numeric_samples(samples)
    validate_report(report)
    sample_ref = report.get("samples")
    if not isinstance(sample_ref, Mapping) or sample_ref.get("sha256") != numeric_samples_sha256(
        samples
    ):
        raise ValueError("v2 report sample reference drift")
    _authenticate_statistics(paths, snapshot, profiles, samples, report)


def _migration_staging(paths: BuildPaths) -> tuple[dict[str, Path], dict[str, Path], Path, Path]:
    profile_paths = {
        profile_id: _migration_stage(paths.profile_files(profile_id).profile)
        for profile_id in PROFILE_IDS
    }
    marker_paths = {
        profile_id: _migration_stage(paths.profile_files(profile_id).report)
        for profile_id in PROFILE_IDS
    }
    return (
        profile_paths,
        marker_paths,
        _migration_stage(paths.sweep_root / "numeric_samples.json"),
        _migration_stage(paths.sweep_root / "report.json"),
    )


def _cleanup_migration_staging(paths: BuildPaths, staged_paths: Sequence[Path]) -> None:
    for staged in staged_paths:
        staged.unlink(missing_ok=True)
    roots = (
        paths.sweep_root,
        *(paths.profile_root(profile_id) for profile_id in PROFILE_IDS),
    )
    for root in roots:
        if root.is_dir():
            for staged in root.rglob(f"*{MIGRATION_STAGE_SUFFIX}"):
                staged.unlink(missing_ok=True)


def _migrate_v1_bundle(paths: BuildPaths) -> tuple[Path, ...]:
    snapshot = _load_snapshot(paths)
    legacy_profiles = {
        profile_id: load_legacy_finalized_profile(paths, snapshot, profile_id)
        for profile_id in PROFILE_IDS
    }
    migrated_profiles = {
        profile_id: profile_artifact(
            snapshot,
            profile_id,
            matrix_fingerprint=legacy_profiles[profile_id].embeddings.fingerprint,
            cache_mode=str(legacy_profiles[profile_id].report["cache_mode"]),
        )
        for profile_id in PROFILE_IDS
    }
    legacy_hashes = {
        profile_id: str(legacy_profiles[profile_id].report["profile_artifact_sha256"])
        for profile_id in PROFILE_IDS
    }
    staged_profiles, staged_markers, staged_samples, staged_report = _migration_staging(paths)
    staged_paths = (
        *(staged_profiles[profile_id] for profile_id in PROFILE_IDS),
        *(staged_markers[profile_id] for profile_id in PROFILE_IDS),
        staged_samples,
        staged_report,
    )
    try:
        migrated_hashes: dict[str, str] = {}
        for profile_id in PROFILE_IDS:
            write_json(staged_profiles[profile_id], migrated_profiles[profile_id])
            migrated_hashes[profile_id] = file_sha256(staged_profiles[profile_id])

        old_samples = read_json(paths.sweep_root / "numeric_samples.json")
        old_report = read_json(paths.sweep_root / "report.json")
        if not isinstance(old_samples, Mapping) or not isinstance(old_report, Mapping):
            raise ValueError("legacy samples and report must be objects")
        new_samples = migrate_numeric_samples_v1(
            old_samples,
            legacy_profile_artifact_sha256=legacy_hashes,
            migrated_profile_artifact_sha256=migrated_hashes,
        )
        profile_summaries = {
            profile_id: cast(Mapping[str, object], migrated_profiles[profile_id]["summary"])
            for profile_id in PROFILE_IDS
        }
        new_report = migrate_report_v1(
            old_report,
            old_samples=old_samples,
            new_samples=new_samples,
            profile_summaries=profile_summaries,
            provenance=cast(Mapping[str, object], new_samples["input_fingerprints"]),
        )
        migrated_finalized: dict[str, FinalizedProfile] = {}
        for profile_id in PROFILE_IDS:
            legacy = legacy_profiles[profile_id]
            marker = profile_report(
                profile_id=profile_id,
                matrix_fingerprint=legacy.embeddings.fingerprint,
                cache_mode=str(legacy.report["cache_mode"]),
                endpoint_provenance=str(legacy.report["endpoint_provenance"]),
                profile_sha256=migrated_hashes[profile_id],
                ledger_sha256=str(legacy.report["split_exceedance_ledger_sha256"]),
                cache_sha256=str(legacy.report["cache_file_sha256"]),
                sidecar_sha256=str(legacy.report["cache_sidecar_sha256"]),
            )
            write_json(staged_markers[profile_id], marker)
            migrated_finalized[profile_id] = FinalizedProfile(
                profile_id,
                legacy.planned,
                legacy.embeddings,
                cast(Mapping[str, object], migrated_profiles[profile_id]),
                marker,
            )
        _authenticate_statistics(
            paths,
            snapshot,
            migrated_finalized,
            new_samples,
            new_report,
        )
        write_json(staged_samples, new_samples)
        write_json(staged_report, new_report)

        pairs = [
            *(
                (staged_profiles[profile_id], paths.profile_files(profile_id).profile)
                for profile_id in PROFILE_IDS
            ),
            *(
                (staged_markers[profile_id], paths.profile_files(profile_id).report)
                for profile_id in PROFILE_IDS
            ),
            (staged_samples, paths.sweep_root / "numeric_samples.json"),
            (staged_report, paths.sweep_root / "report.json"),
        ]
        replace_files(pairs)
        return tuple(target for _source, target in pairs)
    finally:
        _cleanup_migration_staging(paths, staged_paths)


def _migrate_metadata(paths: BuildPaths) -> tuple[Path, ...]:
    """Authenticate a complete v1 bundle, then publish only its metadata."""
    generation = classify_metadata_generation(paths)
    if generation == "v2":
        _authenticate_v2_bundle(paths)
        return ()
    if generation != "v1":
        raise ValueError(f"cannot migrate metadata generation {generation!r}")
    return _migrate_v1_bundle(paths)


def _run_migration(paths: BuildPaths) -> tuple[Path, ...]:
    with (
        exclusive_lock(paths.sweep_root / MIGRATION_LOCK),
        ExitStack() as stack,
    ):
        for profile_id in PROFILE_IDS:
            stack.enter_context(exclusive_lock(paths.profile_root(profile_id) / ".profile.lock"))
        # Producer operation locks protect the inputs while they are loaded and
        # replayed; the shared artifact lock alone only protects their final swap.
        for lock_name in (".pool.lock", ".qrels.lock", ".quality.lock"):
            stack.enter_context(exclusive_lock(paths.eval_root / lock_name))
        stack.enter_context(exclusive_lock(paths.sweep_root / ".bundle.lock"))
        stack.enter_context(exclusive_lock(paths.artifacts / ".index.lock"))
        stack.enter_context(exclusive_lock(paths.eval_root / ".artifacts.lock"))
        return _migrate_metadata(paths)


def _status(snapshot: CanonicalSnapshot) -> str:
    parts = []
    for profile_id in PROFILE_IDS:
        profile = PROFILES[profile_id]
        planned = snapshot.profiles[profile_id]
        exceedance = planned.profile_summary["split_trigger_exceedance"]
        if not isinstance(exceedance, Mapping):
            raise AssertionError("validated profile lost split-trigger summary")
        reasons = exceedance["reasons"]
        if not isinstance(reasons, Mapping):
            raise AssertionError("validated profile lost split-trigger reasons")
        parts.append(
            f"{profile_id}: docs={planned.document_count}, chunks={len(planned.chunk_ids)}, "
            f"reuse={len(snapshot.reusable_ids[profile_id])}, "
            f"new={len(snapshot.new_ids[profile_id])}, "
            f"batches={profile.expected_paid_batches}, "
            f"split_exceedances={exceedance['count']}, unexplained={reasons['unexplained']}"
        )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    paths = BuildPaths(args.artifacts, args.curated)
    try:
        if args.migrate_metadata:
            changed = _run_migration(paths)
            if changed:
                print(f"migrated M7 metadata to v2: {len(changed)} files")
            else:
                print("M7 metadata already authenticated as v2; no files replaced")
            print("provider calls=0; .env not read")
            return 0

        if args.adopt_query_cache:
            certificate = _adopt_query_cache(paths)
            print(f"adopted legacy query cache offline: {certificate}")
            print("provider calls=0; historical request-text binding remains not-observed")
            return 0

        if not args.embed and not args.finalize:
            _check_read_only_locks(paths)
            snapshot = _load_snapshot(paths)
            print(_status(snapshot))
            print("preflight only: provider calls=0, files written=0")
            return 0

        snapshot = _snapshot_with_locks(paths)
        if args.embed:
            client, endpoint_url = _create_client(args.env)
            completed, batches = _embed_profile(
                paths,
                snapshot,
                args.profile,
                client=client,
                endpoint_url=endpoint_url,
                max_batches=args.max_batches,
            )
            total = len(snapshot.new_ids[args.profile])
            print(
                f"{args.profile}: cached {completed:,}/{total:,} new vectors; "
                f"provider batches this invocation={batches}"
            )
            return 0

        report = _finalize_profile(paths, snapshot, args.profile)
        print(f"finalized {args.profile}: {report}")
        return 0
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! M7 chunk-sweep build failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
