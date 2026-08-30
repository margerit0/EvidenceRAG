"""Provider-free retrieval computation for the TiDB chunk-size sweep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zhrag.eval.tidb_chunk_sweep import (
    PROFILE_IDS,
    RETRIEVAL_DEPTH,
    SweepRuns,
    build_numeric_samples,
    build_report,
    build_sweep_runs,
    validate_numeric_samples,
    validate_report,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    CanonicalSnapshot,
    FinalizedProfile,
)
from zhrag.eval.tidb_runs import build_dense_runs, build_lexical_runs
from zhrag.lexical import build_sparse_index

__all__ = ["ChunkSweepEvaluation", "evaluate_chunk_sweep"]


@dataclass(frozen=True, slots=True)
class ChunkSweepEvaluation:
    runs: Mapping[str, SweepRuns]
    sparse_fingerprints: Mapping[str, str]
    numeric_samples: Mapping[str, object]
    report: Mapping[str, object]


def _fingerprint(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def evaluate_chunk_sweep(
    snapshot: CanonicalSnapshot,
    finalized: Mapping[str, FinalizedProfile],
    *,
    resamples: int,
    seed: int,
) -> ChunkSweepEvaluation:
    """Recompute all raw runs, identity-free samples, and aggregate statistics."""
    if tuple(finalized) != PROFILE_IDS:
        raise ValueError("finalized profiles must use frozen registry order")
    query_ids = tuple(sorted(snapshot.source_qrels.by_query))
    if tuple(sorted(snapshot.qrels.by_query)) != query_ids:
        raise ValueError("source qrels and final qrels query IDs differ")
    questions = tuple(snapshot.qrels.by_query[query_id].question for query_id in query_ids)
    query_matrix = snapshot.query_cache.matrix(query_ids)

    runs: dict[str, SweepRuns] = {}
    sparse_fingerprints: dict[str, str] = {}
    input_fingerprints = dict(snapshot.input_fingerprints)
    profile_summaries: dict[str, Mapping[str, object]] = {}
    for profile_id in PROFILE_IDS:
        profile = finalized[profile_id]
        if profile.profile_id != profile_id or profile.planned is not snapshot.profiles[profile_id]:
            raise ValueError(f"{profile_id}: finalized profile is not from this snapshot")
        sparse = build_sparse_index(profile.planned.corpus)
        lexical = build_lexical_runs(
            sparse.index,
            profile.planned.corpus,
            questions,
            depth=RETRIEVAL_DEPTH,
        )
        document_matrix = profile.embeddings.matrix(profile.planned.chunk_ids)
        dense = build_dense_runs(
            query_matrix,
            document_matrix,
            profile.planned.chunk_ids,
            depth=RETRIEVAL_DEPTH,
        )
        profile_runs = build_sweep_runs(
            query_ids,
            lexical,
            dense,
            profile.planned.source_by_chunk,
        )
        runs[profile_id] = profile_runs
        sparse_fingerprints[profile_id] = sparse.index.fingerprint
        input_fingerprints[f"{profile_id}-profile-artifact-sha256"] = _fingerprint(
            profile.report["profile_artifact_sha256"],
            context=f"{profile_id} profile artifact",
        )
        input_fingerprints[f"{profile_id}-matrix-sha256"] = profile.embeddings.fingerprint
        input_fingerprints[f"{profile_id}-sparse-sha256"] = sparse.index.fingerprint
        input_fingerprints[f"{profile_id}-runs-sha256"] = profile_runs.fingerprint
        profile_summaries[profile_id] = profile.public_summary

    samples = build_numeric_samples(
        snapshot.source_qrels,
        runs,
        input_fingerprints=input_fingerprints,
    )
    report = build_report(
        samples,
        profile_summaries=profile_summaries,
        provenance=input_fingerprints,
        resamples=resamples,
        seed=seed,
    )
    validate_numeric_samples(samples)
    validate_report(report)
    return ChunkSweepEvaluation(
        MappingProxyType(runs),
        MappingProxyType(sparse_fingerprints),
        MappingProxyType(samples),
        MappingProxyType(report),
    )
