"""Source-level TiDB chunk-size sweep contracts and deterministic statistics.

M3's final qrels and four-arm quality report are frozen to the published 400/600
chunk universe. M7 deliberately uses a separate schema: raw retrieval still ranks
chunks, exact RRF still fuses chunks, and only the final arm is collapsed to the
stable source identity for a cross-profile known-item sensitivity analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zhrag.chunking import split_trigger_exceedance_reason
from zhrag.eval.metrics import (
    bootstrap_p_floor,
    clustered_bootstrap_ci,
    clustered_paired_bootstrap_test,
    hit_at_k,
    holm_bonferroni,
    holm_floor_flags,
    mrr_at_k,
    win_loss_tie,
)
from zhrag.eval.tidb_quality import QrelsBundle, QrelSurface
from zhrag.eval.tidb_runs import (
    DENSE_LABEL,
    LEXICAL_LABEL,
    RRF_LABEL,
    build_rrf_runs,
)
from zhrag.ingest import ChunkPlan, DocumentPlan, chunker_fingerprint
from zhrag.io_utils import read_jsonl
from zhrag.tokens import estimate_tokens

__all__ = [
    "BATCH_SIZE",
    "CACHE_META_SCHEMA",
    "CHUNK_SWEEP_RELATIVE",
    "CHUNK_SWEEP_REPORT_SCHEMA",
    "CHUNK_SWEEP_SAMPLES_SCHEMA",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "DOCUMENT_EMBEDDING_MODEL",
    "DOCUMENT_EMBEDDING_PROFILE",
    "DOCUMENT_PROMPT",
    "EXPECTED_CLUSTERS",
    "EXPECTED_DOCUMENTS",
    "EXPECTED_PAIRS",
    "EXPECTED_QUERIES",
    "LEGACY_CHUNK_SWEEP_REPORT_SCHEMA",
    "LEGACY_LIMITATIONS",
    "LEGACY_PROFILE_ARTIFACT_SCHEMA",
    "LEGACY_PROFILE_REPORT_SCHEMA",
    "LEGACY_PROVIDER_USAGE",
    "METRIC_LABELS",
    "PROFILES",
    "PROFILE_ARTIFACT_SCHEMA",
    "PROFILE_IDS",
    "PROFILE_REPORT_SCHEMA",
    "PROVIDER_INPUT_ESTIMATE_LIMIT",
    "QUERY_EMBEDDING_PROFILE",
    "QUERY_PROMPT",
    "RETRIEVAL_DEPTH",
    "RRF_K",
    "RUNS_SCHEMA",
    "RUN_LABELS",
    "VECTOR_POLICY",
    "VECTOR_WIDTH",
    "ChunkSweepProfile",
    "EmbeddingCache",
    "PlannedProfile",
    "SourcePair",
    "SourceQrels",
    "SourceSurface",
    "SweepRuns",
    "build_execution_contract",
    "build_numeric_samples",
    "build_report",
    "build_source_qrels",
    "build_sweep_runs",
    "collapse_chunk_run_to_sources",
    "embedding_cache_fingerprint",
    "materialize_profile",
    "numeric_samples_sha256",
    "parse_sweep_runs",
    "profile_for_id",
    "public_profile_summary",
    "strict_embedding_cache",
    "sweep_runs_fingerprint",
    "sweep_runs_rows",
    "validate_exact_embedding_reuse",
    "validate_frozen_sweep_runs",
    "validate_numeric_samples",
    "validate_report",
    "validated_embedding_rows",
]

DOCUMENT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DOCUMENT_PROMPT = ""
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
QUERY_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-query-4096-v1"
QUERY_PROMPT = (
    "Instruct: Given a Chinese question about TiDB, retrieve the documentation "
    "passage that answers it\nQuery:"
)
VECTOR_WIDTH = 4096
BATCH_SIZE = 16
RETRIEVAL_DEPTH = 100
RRF_K = 10
PROVIDER_INPUT_ESTIMATE_LIMIT = 32_768
EXPECTED_DOCUMENTS = 450
EXPECTED_QUERIES = 980
EXPECTED_PAIRS = 490
EXPECTED_CLUSTERS = 245
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0
RUN_LABELS = (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL)
METRIC_LABELS = (
    "origin_mrr_at_10",
    "origin_hit_at_1",
    "origin_hit_at_10",
    "confirmed_mrr_at_10",
    "confirmed_hit_at_1",
    "confirmed_hit_at_10",
    "judged_source_at_1",
    "judged_source_at_10",
)
SURFACES = ("direct", "paraphrase", "overall")

CACHE_META_SCHEMA = "zhrag-tidb-chunk-sweep-embedding-cache-v1"
LEGACY_PROFILE_ARTIFACT_SCHEMA = "zhrag-tidb-chunk-sweep-profile-artifact-v1"
LEGACY_PROFILE_REPORT_SCHEMA = "zhrag-tidb-chunk-sweep-profile-report-v1"
LEGACY_CHUNK_SWEEP_REPORT_SCHEMA = "zhrag-tidb-chunk-sweep-report-v1"
PROFILE_ARTIFACT_SCHEMA = "zhrag-tidb-chunk-sweep-profile-artifact-v2"
PROFILE_REPORT_SCHEMA = "zhrag-tidb-chunk-sweep-profile-report-v2"
RUNS_SCHEMA = "zhrag-tidb-chunk-sweep-runs-v1"
CHUNK_SWEEP_SAMPLES_SCHEMA = "zhrag-tidb-chunk-sweep-numeric-v1"
CHUNK_SWEEP_REPORT_SCHEMA = "zhrag-tidb-chunk-sweep-report-v2"
CHUNK_SWEEP_RELATIVE = Path("eval") / "chunk_sweep" / "v1"
VECTOR_POLICY = "float32-finite-nonzero-no-renormalization"
_PROFILE_FINGERPRINT_SCHEMA = "zhrag-tidb-chunk-sweep-profile-v1"
_CORPUS_FINGERPRINT_SCHEMA = "zhrag-tidb-chunk-sweep-corpus-v1"
_CHUNK_IDS_FINGERPRINT_SCHEMA = "zhrag-tidb-chunk-sweep-chunk-ids-v1"
_CACHE_FINGERPRINT_SCHEMA = "zhrag-tidb-chunk-sweep-cache-matrix-v1"
_SOURCE_MAPPING_FINGERPRINT_SCHEMA = "zhrag-tidb-chunk-sweep-source-map-v1"
_SAMPLES_HASH_SCHEMA = "zhrag-tidb-chunk-sweep-samples-sha256-v1"
_DESIGN_KEYS = frozenset(
    {
        "estimand",
        "primary_system",
        "primary_metric",
        "pair_observations",
        "source_clusters",
        "confidence",
        "resamples",
        "seed",
        "alternative",
        "correction",
        "source_collapse",
    }
)
LEGACY_PROVIDER_USAGE = {
    "chat_calls": 0,
    "document_vectors_created": 3_379,
    "document_vectors_reused": 2_274,
    "evaluation_provider_calls": 0,
    "milvus_writes": 0,
    "query_embedding_calls": 0,
    "rerank_calls": 0,
    "successful_embedding_batches": 212,
}
LEGACY_LIMITATIONS = (
    "400-origin synthetic queries; profile selection on this fixture is exploratory",
    "source known-item retrieval does not certify answer-bearing passage relevance",
    "confirmed-source sensitivity inherits the canonical 400-only judgement pool",
    "unjudged sources are not reliable negatives",
    ("embedding provider outputs are cached because repeated calls are not bitwise deterministic"),
    ("split trigger is best-effort; protected blocks and indivisible paragraphs remain intact"),
)
_LIMITATIONS = (
    "400-origin synthetic queries; profile selection on this fixture is exploratory",
    "source known-item retrieval does not certify answer-bearing passage relevance",
    "confirmed-source sensitivity inherits the canonical 400-only judgement pool",
    "unjudged sources are not reliable negatives",
    (
        "profile cache completeness authenticates required vector coverage, not historical "
        "provider calls; per-batch build receipts were not recorded"
    ),
    ("embedding provider outputs are cached because repeated calls are not bitwise deterministic"),
    ("split trigger is best-effort; protected blocks and indivisible paragraphs remain intact"),
)


_PROFILE_SUMMARY_LEGACY_FIELDS = frozenset({"new_document_vectors", "successful_embedding_batches"})
_PROFILE_SUMMARY_V2_FIELDS = frozenset(
    {"required_new_document_vectors", "required_embedding_batches"}
)
_FORBIDDEN_LEGACY_PROFILE_FIELDS = _PROFILE_SUMMARY_V2_FIELDS
_FORBIDDEN_V2_PROFILE_FIELDS = _PROFILE_SUMMARY_LEGACY_FIELDS


_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "chunk_id",
        "doc_id",
        "document_vectors_created",
        "document_vectors_reused",
        "embedding",
        "embeddings",
        "new_document_vectors",
        "passage",
        "passages",
        "path",
        "paths",
        "provider_payload",
        "query",
        "query_id",
        "question",
        "raw_runs",
        "source_key",
        "successful_embedding_batches",
        "text",
        "texts",
        "vector",
        "vectors",
    }
)


@dataclass(frozen=True, slots=True)
class ChunkSweepProfile:
    """One immutable M7 chunk profile and its preflight drift guards."""

    profile_id: str
    target_tokens: int
    split_trigger_tokens: int
    expected_chunks: int
    expected_canonical_reuse: int
    expected_new_vectors: int
    expected_paid_batches: int

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")
        if not 1 <= self.target_tokens <= self.split_trigger_tokens:
            raise ValueError("profile requires 1 <= target <= split trigger")
        for name in (
            "expected_chunks",
            "expected_canonical_reuse",
            "expected_new_vectors",
            "expected_paid_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.expected_chunks != (self.expected_canonical_reuse + self.expected_new_vectors):
            raise ValueError("reuse and new-vector counts must cover the profile")
        expected_batches = math.ceil(self.expected_new_vectors / BATCH_SIZE)
        if self.expected_paid_batches != expected_batches:
            raise ValueError("expected_paid_batches does not match fixed batch size")

    @property
    def chunker_fingerprint(self) -> str:
        return chunker_fingerprint(
            target_tokens=self.target_tokens,
            hard_max_tokens=self.split_trigger_tokens,
        )

    @property
    def fingerprint(self) -> str:
        return _semantic_digest(
            _PROFILE_FINGERPRINT_SCHEMA,
            (
                self.profile_id,
                str(self.target_tokens),
                str(self.split_trigger_tokens),
                str(self.expected_chunks),
                str(self.expected_canonical_reuse),
                str(self.expected_new_vectors),
                str(self.expected_paid_batches),
                self.chunker_fingerprint,
            ),
        )


_PROFILE_SEQUENCE = (
    ChunkSweepProfile("tidb-chunk-t256-h384-v1", 256, 384, 2_802, 252, 2_550, 160),
    ChunkSweepProfile("tidb-chunk-t400-h600-v1", 400, 600, 1_832, 1_832, 0, 0),
    ChunkSweepProfile("tidb-chunk-t800-h1200-v1", 800, 1_200, 1_019, 190, 829, 52),
)
PROFILE_IDS = tuple(profile.profile_id for profile in _PROFILE_SEQUENCE)
PROFILES: Mapping[str, ChunkSweepProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILE_SEQUENCE}
)


def profile_for_id(profile_id: str) -> ChunkSweepProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown frozen chunk-sweep profile {profile_id!r}") from exc


def build_execution_contract() -> dict[str, object]:
    """State only required coverage and phase-local calls the artifacts can support."""
    return {
        "required_new_document_vectors": sum(
            profile.expected_new_vectors for profile in _PROFILE_SEQUENCE
        ),
        "required_canonical_exact_reuse": sum(
            profile.expected_canonical_reuse for profile in _PROFILE_SEQUENCE
        ),
        "required_embedding_batches": sum(
            profile.expected_paid_batches for profile in _PROFILE_SEQUENCE
        ),
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


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _semantic_digest(schema: str, values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    _update(digest, schema)
    for value in values:
        _update(digest, value)
    return digest.hexdigest()


def _body_text(chunk: ChunkPlan) -> str:
    if not chunk.heading_path:
        return chunk.contextual_text
    prefix = f"{' > '.join(chunk.heading_path)}\n\n"
    if not chunk.contextual_text.startswith(prefix):
        raise ValueError(f"{chunk.chunk_id}: contextual heading prefix drift")
    return chunk.contextual_text[len(prefix) :]


@dataclass(frozen=True, slots=True)
class PlannedProfile:
    """One verified in-memory profile without retaining mutable plan mappings."""

    profile: ChunkSweepProfile
    document_count: int
    chunk_ids: tuple[str, ...]
    corpus: Mapping[str, str]
    source_by_chunk: Mapping[str, str]
    corpus_fingerprint: str
    chunk_ids_fingerprint: str
    source_mapping_fingerprint: str
    profile_summary: Mapping[str, object]
    exceedance_rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if self.document_count < 1:
            raise ValueError("planned profile must contain documents")
        if len(self.chunk_ids) != self.profile.expected_chunks:
            raise ValueError(
                f"{self.profile.profile_id}: planned {len(self.chunk_ids):,} chunks, "
                f"expected {self.profile.expected_chunks:,}"
            )
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("planned chunk ids must be unique")
        if set(self.corpus) != set(self.chunk_ids):
            raise ValueError("planned corpus keys differ from chunk ids")
        if set(self.source_by_chunk) != set(self.chunk_ids):
            raise ValueError("planned source mapping keys differ from chunk ids")


def materialize_profile(
    profile: ChunkSweepProfile,
    documents: Sequence[DocumentPlan],
    *,
    enforce_document_count: bool = True,
) -> PlannedProfile:
    """Flatten verified plans, classify split-trigger exceedances, and fingerprint."""
    if not documents:
        raise ValueError("profile planning produced no documents")
    if enforce_document_count and len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"{profile.profile_id}: planned {len(documents)} documents, "
            f"expected {EXPECTED_DOCUMENTS}"
        )
    corpus: dict[str, str] = {}
    source_by_chunk: dict[str, str] = {}
    chunk_ids: list[str] = []
    approx_tokens: list[int] = []
    embedding_tokens: list[int] = []
    exceedance_rows: list[Mapping[str, object]] = []
    document_values: list[str] = []
    corpus_values: list[str] = []
    source_values: list[str] = []

    for document in documents:
        document_values.extend(
            (
                document.key,
                document.path,
                document.document_sha256,
                document.metadata_fingerprint,
            )
        )
        for chunk in document.chunks:
            if chunk.chunk_id in corpus:
                raise ValueError(f"duplicate planned chunk id {chunk.chunk_id!r}")
            chunk_ids.append(chunk.chunk_id)
            corpus[chunk.chunk_id] = chunk.contextual_text
            source_by_chunk[chunk.chunk_id] = document.key
            approx_tokens.append(chunk.approx_tokens)
            contextual_tokens = estimate_tokens(chunk.contextual_text)
            embedding_tokens.append(contextual_tokens)
            corpus_values.extend(
                (
                    document.key,
                    str(chunk.ordinal),
                    chunk.chunk_id,
                    chunk.contextual_text,
                    str(chunk.approx_tokens),
                )
            )
            source_values.extend((chunk.chunk_id, document.key))
            if chunk.approx_tokens <= profile.split_trigger_tokens:
                continue
            body = _body_text(chunk)
            reason = split_trigger_exceedance_reason(
                body,
                split_trigger_tokens=profile.split_trigger_tokens,
            )
            if reason is None:
                raise ValueError(f"{chunk.chunk_id}: exceedance has no diagnostic reason")
            paragraphs = [part for part in body.split("\n\n") if part.strip()]
            max_paragraph = max((estimate_tokens(part) for part in paragraphs), default=0)
            exceedance_rows.append(
                MappingProxyType(
                    {
                        "chunk_id": chunk.chunk_id,
                        "approx_tokens": chunk.approx_tokens,
                        "max_paragraph_tokens": max_paragraph,
                        "reason": reason,
                    }
                )
            )

    ordered_ids = tuple(chunk_ids)
    if len(ordered_ids) != profile.expected_chunks:
        raise ValueError(
            f"{profile.profile_id}: planned {len(ordered_ids):,} chunks, "
            f"expected {profile.expected_chunks:,}"
        )
    if max(embedding_tokens) > PROVIDER_INPUT_ESTIMATE_LIMIT:
        raise ValueError(
            f"{profile.profile_id}: contextual input estimate exceeds "
            f"{PROVIDER_INPUT_ESTIMATE_LIMIT:,} tokens"
        )
    reasons = {
        name: sum(row["reason"] == name for row in exceedance_rows)
        for name in (
            "protected_fence",
            "protected_table",
            "protected_fence_and_table",
            "indivisible_paragraph",
            "unexplained",
        )
    }
    if sum(reasons.values()) != len(exceedance_rows) or reasons["unexplained"] != 0:
        raise ValueError(f"{profile.profile_id}: split-trigger ledger does not reconcile")

    token_array = np.asarray(approx_tokens, dtype=np.float64)
    summary: dict[str, object] = {
        "profile_id": profile.profile_id,
        "target_tokens": profile.target_tokens,
        "split_trigger_tokens": profile.split_trigger_tokens,
        "documents": len(documents),
        "chunks": len(ordered_ids),
        "tokens": {
            "p10": float(np.percentile(token_array, 10, method="linear")),
            "p50": float(np.percentile(token_array, 50, method="linear")),
            "p90": float(np.percentile(token_array, 90, method="linear")),
            "p99": float(np.percentile(token_array, 99, method="linear")),
            "max": int(token_array.max()),
            "under_100": sum(value < 100 for value in approx_tokens),
            "total": sum(approx_tokens),
        },
        "split_trigger_exceedance": {
            "count": len(exceedance_rows),
            "rate": len(exceedance_rows) / len(ordered_ids),
            "reasons": reasons,
        },
        "embedding_input_estimate": {
            "limit": PROVIDER_INPUT_ESTIMATE_LIMIT,
            "max": max(embedding_tokens),
            "exceedances": 0,
        },
        "chunker_fingerprint_sha256": profile.chunker_fingerprint,
        "profile_fingerprint_sha256": profile.fingerprint,
        "document_set_fingerprint_sha256": _semantic_digest(
            "zhrag-tidb-chunk-sweep-documents-v1", document_values
        ),
    }
    return PlannedProfile(
        profile=profile,
        document_count=len(documents),
        chunk_ids=ordered_ids,
        corpus=MappingProxyType(corpus),
        source_by_chunk=MappingProxyType(source_by_chunk),
        corpus_fingerprint=_semantic_digest(_CORPUS_FINGERPRINT_SCHEMA, corpus_values),
        chunk_ids_fingerprint=_semantic_digest(_CHUNK_IDS_FINGERPRINT_SCHEMA, ordered_ids),
        source_mapping_fingerprint=_semantic_digest(
            _SOURCE_MAPPING_FINGERPRINT_SCHEMA, source_values
        ),
        profile_summary=MappingProxyType(summary),
        exceedance_rows=tuple(exceedance_rows),
    )


def public_profile_summary(
    planned: PlannedProfile,
    *,
    embedding_matrix_fingerprint: str,
) -> dict[str, object]:
    """Return the aggregate-only profile record allowed into the public report."""
    if not _is_sha256(embedding_matrix_fingerprint):
        raise ValueError("embedding matrix fingerprint must be a lowercase SHA-256")
    summary = dict(planned.profile_summary)
    summary.update(
        {
            "canonical_exact_reuse": planned.profile.expected_canonical_reuse,
            "required_new_document_vectors": planned.profile.expected_new_vectors,
            "required_embedding_batches": planned.profile.expected_paid_batches,
            "corpus_fingerprint_sha256": planned.corpus_fingerprint,
            "chunk_ids_fingerprint_sha256": planned.chunk_ids_fingerprint,
            "source_mapping_fingerprint_sha256": planned.source_mapping_fingerprint,
            "embedding_matrix_fingerprint_sha256": embedding_matrix_fingerprint,
        }
    )
    return summary


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    """A strict cache snapshot aligned to one exact expected ID universe."""

    vectors: Mapping[str, NDArray[np.float32]]
    missing_ids: tuple[str, ...]
    fingerprint: str
    width: int

    def matrix(self, ids: Sequence[str]) -> NDArray[np.float32]:
        missing = [item_id for item_id in ids if item_id not in self.vectors]
        if missing:
            raise ValueError(f"cache matrix request is missing {len(missing)} ids")
        return np.asarray([self.vectors[item_id] for item_id in ids], dtype=np.float32)


def embedding_cache_fingerprint(
    ids: Sequence[str], vectors: Mapping[str, NDArray[np.float32]], *, width: int
) -> str:
    digest = hashlib.sha256()
    _update(digest, _CACHE_FINGERPRINT_SCHEMA)
    _update(digest, str(width))
    for item_id in ids:
        vector = vectors[item_id]
        _update(digest, item_id)
        little = np.asarray(vector, dtype="<f4")
        digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


def _float32_embedding(raw: object, *, width: int, context: str) -> NDArray[np.float32]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != width:
        raise ValueError(f"{context}: invalid embedding width")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise ValueError(f"{context}: embedding is not numeric")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: invalid float32 vector") from exc
    if not bool(np.isfinite(vector).all()):
        raise ValueError(f"{context}: non-finite embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{context}: zero/non-finite embedding norm")
    return vector


def validated_embedding_rows(
    ids: Sequence[str],
    raw_vectors: Sequence[Sequence[object]],
    *,
    width: int = VECTOR_WIDTH,
) -> tuple[dict[str, object], ...]:
    """Validate an entire provider batch before exposing any appendable row."""
    if len(ids) != len(raw_vectors):
        raise ValueError("embedding response count differs from requested batch")
    if len(set(ids)) != len(ids) or any(not item_id for item_id in ids):
        raise ValueError("embedding batch ids must be unique non-empty strings")
    rows: list[dict[str, object]] = []
    for position, (item_id, raw) in enumerate(zip(ids, raw_vectors, strict=True)):
        vector = _float32_embedding(raw, width=width, context=f"embedding row {position}")
        rows.append({"doc_id": item_id, "embedding": vector.tolist()})
    return tuple(rows)


def strict_embedding_cache(
    cache: Path,
    expected_ids: Sequence[str],
    *,
    width: int = VECTOR_WIDTH,
    allow_partial: bool,
) -> EmbeddingCache:
    """Reject malformed/duplicate/extra rows before a cache can be reused."""
    if len(set(expected_ids)) != len(expected_ids) or any(
        not isinstance(item_id, str) or not item_id for item_id in expected_ids
    ):
        raise ValueError("expected embedding ids must be unique non-empty strings")
    expected = set(expected_ids)
    vectors: dict[str, NDArray[np.float32]] = {}
    if cache.is_file():
        for line_number, row in enumerate(read_jsonl(cache), 1):
            if set(row) != {"doc_id", "embedding"}:
                raise ValueError(f"{cache.name}:{line_number}: cache row keys differ")
            item_id = row["doc_id"]
            if not isinstance(item_id, str) or not item_id:
                raise ValueError(f"{cache.name}:{line_number}: invalid doc_id")
            if item_id not in expected:
                raise ValueError(f"{cache.name}:{line_number}: unexpected doc_id")
            if item_id in vectors:
                raise ValueError(f"{cache.name}:{line_number}: duplicate doc_id")
            raw = row["embedding"]
            vector = _float32_embedding(
                raw,
                width=width,
                context=f"{cache.name}:{line_number}",
            )
            vectors[item_id] = vector
    missing = tuple(item_id for item_id in expected_ids if item_id not in vectors)
    if missing and not allow_partial:
        raise ValueError(
            f"{cache.name} is missing {len(missing):,} of {len(expected_ids):,} vectors"
        )
    present = tuple(item_id for item_id in expected_ids if item_id in vectors)
    fingerprint = embedding_cache_fingerprint(present, vectors, width=width)
    return EmbeddingCache(MappingProxyType(vectors), missing, fingerprint, width)


def validate_exact_embedding_reuse(
    cache: EmbeddingCache,
    canonical: EmbeddingCache,
    ids: Sequence[str],
) -> None:
    """Require reused IDs to carry the exact canonical float32 vectors."""
    if cache.width != canonical.width:
        raise ValueError("canonical reuse matrix width drift")
    for item_id in ids:
        if item_id not in cache.vectors or item_id not in canonical.vectors:
            raise ValueError(f"{item_id}: canonical reuse vector is missing")
        if not np.array_equal(cache.vectors[item_id], canonical.vectors[item_id]):
            raise ValueError(f"{item_id}: canonical reuse vector drift")
    expected = embedding_cache_fingerprint(ids, canonical.vectors, width=canonical.width)
    actual = embedding_cache_fingerprint(ids, cache.vectors, width=cache.width)
    if actual != expected:
        raise ValueError("canonical reuse matrix fingerprint drift")


@dataclass(frozen=True, slots=True)
class SweepRuns:
    """Aligned raw chunk runs and their exact first-source collapses."""

    query_ids: tuple[str, ...]
    chunk_runs: Mapping[str, tuple[tuple[str, ...], ...]]
    source_runs: Mapping[str, tuple[tuple[str, ...], ...]]

    def __post_init__(self) -> None:
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query ids must be non-empty and unique")
        required = set(RUN_LABELS)
        if set(self.chunk_runs) != required or set(self.source_runs) != required:
            raise ValueError("M7 run labels differ from the frozen three-arm set")
        for kind, runs in (("chunk", self.chunk_runs), ("source", self.source_runs)):
            for label, rows in runs.items():
                if len(rows) != len(self.query_ids):
                    raise ValueError(f"{kind}/{label}: run count does not match queries")
                for query_id, run in zip(self.query_ids, rows, strict=True):
                    if len(set(run)) != len(run):
                        raise ValueError(f"{kind}/{label}/{query_id}: duplicate run ids")

    @property
    def fingerprint(self) -> str:
        return sweep_runs_fingerprint(self)


def collapse_chunk_run_to_sources(
    run: Sequence[str], source_by_chunk: Mapping[str, str]
) -> tuple[str, ...]:
    """Keep the first occurrence of each source in one ordered chunk run."""
    if len(set(run)) != len(run):
        raise ValueError("raw chunk run contains duplicate chunk ids")
    seen: set[str] = set()
    sources: list[str] = []
    for chunk_id in run:
        try:
            source = source_by_chunk[chunk_id]
        except KeyError as exc:
            raise ValueError(f"run contains unknown chunk id {chunk_id!r}") from exc
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return tuple(sources)


def sweep_runs_fingerprint(runs: SweepRuns) -> str:
    values: list[str] = []
    for kind, by_label in (("chunk", runs.chunk_runs), ("source", runs.source_runs)):
        values.append(kind)
        for label in RUN_LABELS:
            values.append(label)
            for query_id, run in zip(runs.query_ids, by_label[label], strict=True):
                values.extend((query_id, str(len(run)), *run))
    return _semantic_digest(RUNS_SCHEMA, values)


def validate_frozen_sweep_runs(
    runs: SweepRuns,
    source_by_chunk: Mapping[str, str],
    *,
    enforce_query_count: bool = True,
) -> None:
    """Authenticate depth, exact raw RRF, and post-arm source collapse."""
    if enforce_query_count and len(runs.query_ids) != EXPECTED_QUERIES:
        raise ValueError("M7 runs do not cover the frozen 980-query fixture")
    if runs.query_ids != tuple(sorted(runs.query_ids)):
        raise ValueError("M7 run query ids must use canonical sorted order")
    for position, query_id in enumerate(runs.query_ids):
        lexical = runs.chunk_runs[LEXICAL_LABEL][position]
        dense = runs.chunk_runs[DENSE_LABEL][position]
        fused = runs.chunk_runs[RRF_LABEL][position]
        if len(lexical) > RETRIEVAL_DEPTH:
            raise ValueError(f"{query_id}: lexical run exceeds frozen depth")
        if len(dense) != RETRIEVAL_DEPTH:
            raise ValueError(f"{query_id}: dense run does not have frozen depth")
        expected_fused = build_rrf_runs(
            (lexical,),
            (dense,),
            depth=RETRIEVAL_DEPTH,
            rrf_k=RRF_K,
        )[0]
        if fused != expected_fused:
            raise ValueError(f"{query_id}: fused run is not exact frozen chunk RRF")
        for label in RUN_LABELS:
            raw = runs.chunk_runs[label][position]
            unknown = set(raw) - set(source_by_chunk)
            if unknown:
                raise ValueError(f"{label}/{query_id}: raw run contains unknown chunks")
            expected_sources = collapse_chunk_run_to_sources(raw, source_by_chunk)
            if runs.source_runs[label][position] != expected_sources:
                raise ValueError(f"{label}/{query_id}: source run is not post-arm collapse")


def build_sweep_runs(
    query_ids: Sequence[str],
    lexical: Sequence[Sequence[str]],
    dense: Sequence[Sequence[str]],
    source_by_chunk: Mapping[str, str],
    *,
    enforce_query_count: bool = True,
) -> SweepRuns:
    if len(lexical) != len(query_ids) or len(dense) != len(query_ids):
        raise ValueError("M7 raw runs do not align to query ids")
    chunk_runs = {
        LEXICAL_LABEL: tuple(tuple(run) for run in lexical),
        DENSE_LABEL: tuple(tuple(run) for run in dense),
        RRF_LABEL: build_rrf_runs(
            lexical,
            dense,
            depth=RETRIEVAL_DEPTH,
            rrf_k=RRF_K,
        ),
    }
    source_runs = {
        label: tuple(
            collapse_chunk_run_to_sources(run, source_by_chunk) for run in chunk_runs[label]
        )
        for label in RUN_LABELS
    }
    runs = SweepRuns(
        tuple(query_ids),
        MappingProxyType(chunk_runs),
        MappingProxyType(source_runs),
    )
    validate_frozen_sweep_runs(
        runs,
        source_by_chunk,
        enforce_query_count=enforce_query_count,
    )
    return runs


def sweep_runs_rows(runs: SweepRuns) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema": RUNS_SCHEMA,
            "query_id": query_id,
            "chunk_runs": {label: list(runs.chunk_runs[label][position]) for label in RUN_LABELS},
            "source_runs": {label: list(runs.source_runs[label][position]) for label in RUN_LABELS},
        }
        for position, query_id in enumerate(runs.query_ids)
    )


def _run_ids(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item_id, str) or not item_id for item_id in value
    ):
        raise ValueError(f"{context}: run must be a list of non-empty ids")
    return tuple(value)


def parse_sweep_runs(
    rows: Iterable[Mapping[str, Any]],
    source_by_chunk: Mapping[str, str],
    *,
    enforce_query_count: bool = True,
) -> SweepRuns:
    query_ids: list[str] = []
    chunk_by_label: dict[str, list[tuple[str, ...]]] = {label: [] for label in RUN_LABELS}
    source_by_label: dict[str, list[tuple[str, ...]]] = {label: [] for label in RUN_LABELS}
    for row_number, row in enumerate(rows, 1):
        context = f"M7 runs row {row_number}"
        if set(row) != {"schema", "query_id", "chunk_runs", "source_runs"}:
            raise ValueError(f"{context}: keys differ from frozen schema")
        if row["schema"] != RUNS_SCHEMA:
            raise ValueError(f"{context}: schema drift")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{context}: query_id must be non-empty")
        query_ids.append(query_id)
        for kind, raw_mapping, destination in (
            ("chunk", row["chunk_runs"], chunk_by_label),
            ("source", row["source_runs"], source_by_label),
        ):
            if not isinstance(raw_mapping, Mapping) or set(raw_mapping) != set(RUN_LABELS):
                raise ValueError(f"{context}: {kind} run labels drift")
            for label in RUN_LABELS:
                destination[label].append(
                    _run_ids(raw_mapping[label], context=f"{context}/{kind}/{label}")
                )
    runs = SweepRuns(
        tuple(query_ids),
        MappingProxyType({label: tuple(chunk_by_label[label]) for label in RUN_LABELS}),
        MappingProxyType({label: tuple(source_by_label[label]) for label in RUN_LABELS}),
    )
    validate_frozen_sweep_runs(
        runs,
        source_by_chunk,
        enforce_query_count=enforce_query_count,
    )
    return runs


@dataclass(frozen=True, slots=True)
class SourceSurface:
    query_id: str
    task: str
    pair_id: str
    origin_source: str
    confirmed_sources: tuple[str, ...]
    partial_sources: tuple[str, ...]
    judged_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourcePair:
    direct: SourceSurface
    paraphrase: SourceSurface

    @property
    def origin_source(self) -> str:
        if self.direct.origin_source != self.paraphrase.origin_source:
            raise ValueError("pair origin source drift")
        return self.direct.origin_source


@dataclass(frozen=True, slots=True)
class SourceQrels:
    pairs: tuple[SourcePair, ...]
    by_query: Mapping[str, SourceSurface]
    fingerprint: str


def _source_set(
    ids: Sequence[str], source_by_chunk: Mapping[str, str], *, context: str
) -> set[str]:
    sources: set[str] = set()
    for chunk_id in ids:
        try:
            sources.add(source_by_chunk[chunk_id])
        except KeyError as exc:
            raise ValueError(f"{context}: qrels contain an unknown canonical chunk") from exc
    return sources


def _source_surface(surface: QrelSurface, source_by_chunk: Mapping[str, str]) -> SourceSurface:
    try:
        generating_source = source_by_chunk[surface.pair_id]
    except KeyError as exc:
        raise ValueError(f"{surface.query_id}: generating chunk is absent") from exc
    if generating_source != surface.source_key:
        raise ValueError(f"{surface.query_id}: gold_source_key does not match generating chunk")
    raw_full_ids = [
        chunk_id
        for chunk_id in surface.full_doc_ids
        if chunk_id != surface.pair_id or surface.generating_chunk_grade == 2
    ]
    confirmed = _source_set(raw_full_ids, source_by_chunk, context=surface.query_id)
    partial = _source_set(surface.partial_doc_ids, source_by_chunk, context=surface.query_id)
    if surface.generating_chunk_grade == 1:
        partial.add(generating_source)
    judged = _source_set(surface.judged_doc_ids, source_by_chunk, context=surface.query_id)
    if generating_source not in confirmed:
        raise ValueError(f"{surface.query_id}: origin source lacks raw grade-2 support")
    return SourceSurface(
        query_id=surface.query_id,
        task=surface.task,
        pair_id=surface.pair_id,
        origin_source=generating_source,
        confirmed_sources=tuple(sorted(confirmed)),
        partial_sources=tuple(sorted(partial - confirmed)),
        judged_sources=tuple(sorted(judged)),
    )


def build_source_qrels(bundle: QrelsBundle, source_by_chunk: Mapping[str, str]) -> SourceQrels:
    pairs: list[SourcePair] = []
    by_query: dict[str, SourceSurface] = {}
    values: list[str] = []
    for pair in bundle.pairs:
        direct = _source_surface(pair.direct, source_by_chunk)
        paraphrase = _source_surface(pair.paraphrase, source_by_chunk)
        shared = (
            direct.pair_id == paraphrase.pair_id
            and direct.origin_source == paraphrase.origin_source
            and direct.confirmed_sources == paraphrase.confirmed_sources
            and direct.partial_sources == paraphrase.partial_sources
            and direct.judged_sources == paraphrase.judged_sources
        )
        if not shared:
            raise ValueError(f"{pair.pair_id}: source-level pair controls drift")
        source_pair = SourcePair(direct, paraphrase)
        pairs.append(source_pair)
        for surface in (direct, paraphrase):
            if surface.query_id in by_query:
                raise ValueError(f"duplicate source qrels query {surface.query_id!r}")
            by_query[surface.query_id] = surface
            values.extend(
                (
                    surface.query_id,
                    surface.task,
                    surface.pair_id,
                    surface.origin_source,
                    *surface.confirmed_sources,
                    "partial",
                    *surface.partial_sources,
                    "judged",
                    *surface.judged_sources,
                )
            )
    if len(pairs) != EXPECTED_PAIRS or len(by_query) != EXPECTED_QUERIES:
        raise ValueError("source qrels do not match the frozen 490-pair/980-query design")
    clusters = {pair.origin_source for pair in pairs}
    if len(clusters) != EXPECTED_CLUSTERS:
        raise ValueError("source qrels do not match the frozen 245-cluster design")
    return SourceQrels(
        pairs=tuple(pairs),
        by_query=MappingProxyType(by_query),
        fingerprint=_semantic_digest("zhrag-tidb-chunk-sweep-source-qrels-v1", values),
    )


def _surface_metrics(run: Sequence[str], surface: SourceSurface) -> dict[str, float]:
    confirmed = surface.confirmed_sources
    judged = set(surface.judged_sources)
    return {
        "origin_mrr_at_10": mrr_at_k(run, (surface.origin_source,), 10),
        "origin_hit_at_1": hit_at_k(run, (surface.origin_source,), 1),
        "origin_hit_at_10": hit_at_k(run, (surface.origin_source,), 10),
        "confirmed_mrr_at_10": mrr_at_k(run, confirmed, 10),
        "confirmed_hit_at_1": hit_at_k(run, confirmed, 1),
        "confirmed_hit_at_10": hit_at_k(run, confirmed, 10),
        "judged_source_at_1": sum(source in judged for source in run[:1]),
        "judged_source_at_10": sum(source in judged for source in run[:10]) / 10.0,
    }


def build_numeric_samples(
    source_qrels: SourceQrels,
    profile_runs: Mapping[str, SweepRuns],
    *,
    input_fingerprints: Mapping[str, str],
) -> dict[str, object]:
    """Create identity-free pair samples sufficient for every published aggregate."""
    if tuple(profile_runs) != PROFILE_IDS:
        raise ValueError("profile runs must use frozen profile order")
    query_ids = tuple(sorted(source_qrels.by_query))
    for profile_id, runs in profile_runs.items():
        if runs.query_ids != query_ids:
            raise ValueError(f"{profile_id}: run query order differs from source qrels")
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    cluster_names = sorted({pair.origin_source for pair in source_qrels.pairs})
    cluster_indices = {source: index for index, source in enumerate(cluster_names)}
    rows: list[dict[str, object]] = []
    for sequence, pair in enumerate(source_qrels.pairs):
        surface_rows: dict[str, object] = {}
        for task, surface in (("direct", pair.direct), ("paraphrase", pair.paraphrase)):
            position = query_positions[surface.query_id]
            profile_values: dict[str, object] = {}
            for profile_id in PROFILE_IDS:
                system_values: dict[str, object] = {}
                runs = profile_runs[profile_id]
                for label in RUN_LABELS:
                    system_values[label] = _surface_metrics(
                        runs.source_runs[label][position], surface
                    )
                profile_values[profile_id] = system_values
            surface_rows[task] = profile_values
        rows.append(
            {
                "sequence": sequence,
                "cluster": cluster_indices[pair.origin_source],
                "direct": surface_rows["direct"],
                "paraphrase": surface_rows["paraphrase"],
            }
        )
    artifact: dict[str, object] = {
        "schema": CHUNK_SWEEP_SAMPLES_SCHEMA,
        "profiles": list(PROFILE_IDS),
        "systems": list(RUN_LABELS),
        "metrics": list(METRIC_LABELS),
        "queries": len(query_ids),
        "pairs": len(rows),
        "clusters": len(cluster_names),
        "input_fingerprints": dict(sorted(input_fingerprints.items())),
        "rows": rows,
    }
    validate_numeric_samples(artifact)
    return artifact


def _exact_mapping(value: object, *, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if set(value) != keys:
        raise ValueError(f"{context} keys differ from the frozen schema")
    return value


def _finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def validate_numeric_samples(raw: object) -> None:  # noqa: PLR0912
    root = _exact_mapping(
        raw,
        keys={
            "schema",
            "profiles",
            "systems",
            "metrics",
            "queries",
            "pairs",
            "clusters",
            "input_fingerprints",
            "rows",
        },
        context="numeric samples",
    )
    if root["schema"] != CHUNK_SWEEP_SAMPLES_SCHEMA:
        raise ValueError("numeric samples schema drift")
    if root["profiles"] != list(PROFILE_IDS) or root["systems"] != list(RUN_LABELS):
        raise ValueError("numeric samples profile/system order drift")
    if root["metrics"] != list(METRIC_LABELS):
        raise ValueError("numeric samples metric order drift")
    if (root["queries"], root["pairs"], root["clusters"]) != (
        EXPECTED_QUERIES,
        EXPECTED_PAIRS,
        EXPECTED_CLUSTERS,
    ):
        raise ValueError("numeric samples frozen counts drift")
    fingerprints = root["input_fingerprints"]
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError("numeric samples need input fingerprints")
    for name, digest in fingerprints.items():
        if not isinstance(name, str) or not name or not _is_sha256(digest):
            raise ValueError("numeric samples contain an invalid input fingerprint")
    rows = root["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_PAIRS:
        raise ValueError("numeric samples rows do not cover every pair")
    observed_clusters: set[int] = set()
    metric_keys = set(METRIC_LABELS)
    for index, raw_row in enumerate(rows):
        row = _exact_mapping(
            raw_row,
            keys={"sequence", "cluster", "direct", "paraphrase"},
            context=f"numeric row {index}",
        )
        if row["sequence"] != index:
            raise ValueError("numeric sample sequences must be contiguous")
        cluster = row["cluster"]
        if (
            isinstance(cluster, bool)
            or not isinstance(cluster, int)
            or not 0 <= cluster < EXPECTED_CLUSTERS
        ):
            raise ValueError("numeric sample cluster index is invalid")
        observed_clusters.add(cluster)
        for task in ("direct", "paraphrase"):
            profile_map = row[task]
            if not isinstance(profile_map, Mapping) or set(profile_map) != set(PROFILE_IDS):
                raise ValueError("numeric sample profile keys drift")
            for profile_id in PROFILE_IDS:
                systems = profile_map[profile_id]
                if not isinstance(systems, Mapping) or set(systems) != set(RUN_LABELS):
                    raise ValueError("numeric sample system keys drift")
                for label in RUN_LABELS:
                    metrics = systems[label]
                    if not isinstance(metrics, Mapping) or set(metrics) != metric_keys:
                        raise ValueError("numeric sample metric keys drift")
                    for metric, value in metrics.items():
                        score = _finite(value, context=f"{task}/{profile_id}/{label}/{metric}")
                        if not 0.0 <= score <= 1.0:
                            raise ValueError("numeric sample metric must be in [0, 1]")
    if observed_clusters != set(range(EXPECTED_CLUSTERS)):
        raise ValueError("numeric samples cluster indices are not dense")
    _walk_forbidden(raw)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def numeric_samples_sha256(samples: Mapping[str, object]) -> str:
    validate_numeric_samples(samples)
    digest = hashlib.sha256()
    _update(digest, _SAMPLES_HASH_SCHEMA)
    digest.update(_canonical_json(samples))
    return digest.hexdigest()


def _seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256()
    _update(digest, "zhrag-tidb-chunk-sweep-bootstrap-seed-v1")
    _update(digest, str(base_seed))
    _update(digest, namespace)
    return int.from_bytes(digest.digest()[:8], "big")


def _metric_values(
    samples: Mapping[str, Any],
    profile_id: str,
    label: str,
    metric: str,
    surface: str,
) -> list[float]:
    values: list[float] = []
    for row in samples["rows"]:
        direct = float(row["direct"][profile_id][label][metric])
        paraphrase = float(row["paraphrase"][profile_id][label][metric])
        if surface == "direct":
            values.append(direct)
        elif surface == "paraphrase":
            values.append(paraphrase)
        elif surface == "overall":
            values.append((direct + paraphrase) / 2.0)
        else:  # pragma: no cover - internal frozen caller
            raise AssertionError(surface)
    return values


def _ci_object(
    values: Sequence[float], clusters: Sequence[int], *, resamples: int, seed: int
) -> dict[str, object]:
    interval = clustered_bootstrap_ci(values, clusters, resamples=resamples, seed=seed)
    return {
        "mean": interval.mean,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "observations": len(values),
        "clusters": interval.n,
    }


def _contrast_object(
    baseline: Sequence[float],
    treatment: Sequence[float],
    clusters: Sequence[int],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, object], float, bool]:
    differences = [right - left for left, right in zip(baseline, treatment, strict=True)]
    interval = clustered_bootstrap_ci(
        differences,
        clusters,
        resamples=resamples,
        seed=_seed(seed, "difference-ci"),
    )
    p_value = clustered_paired_bootstrap_test(
        baseline,
        treatment,
        clusters,
        alternative="two-sided",
        resamples=resamples,
        seed=_seed(seed, "paired-test"),
    )
    counts = win_loss_tie(baseline, treatment)
    floor = bootstrap_p_floor(resamples)
    at_floor = math.isclose(p_value, floor, rel_tol=1e-12, abs_tol=1e-15)
    return (
        {
            "delta": interval.mean,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "raw_p": p_value,
            "raw_p_at_floor": at_floor,
            "wins": counts.wins,
            "losses": counts.losses,
            "ties_nonzero": counts.ties_nonzero,
            "ties_zero": counts.ties_zero,
            "observations": counts.n,
            "clusters": interval.n,
        },
        p_value,
        at_floor,
    )


def _apply_holm(
    rows: list[dict[str, object]], raw: Mapping[str, float], floors: Mapping[str, bool]
) -> None:
    adjusted = holm_bonferroni(dict(raw))
    adjusted_floors = holm_floor_flags(raw, floors)
    for row in rows:
        key = str(row["comparison"])
        row["adjusted_p"] = adjusted[key][0]
        row["reject"] = adjusted[key][1]
        row["adjusted_p_inherits_floor"] = adjusted_floors[key]


def build_report(  # noqa: PLR0912, PLR0915 - frozen statistical publication
    samples: Mapping[str, object],
    *,
    profile_summaries: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Rebuild every M7 aggregate solely from authenticated numeric samples."""
    validate_numeric_samples(samples)
    sample_inputs = samples["input_fingerprints"]
    if not isinstance(sample_inputs, Mapping):
        raise AssertionError("validated samples lost their input fingerprints")
    if dict(provenance) != dict(sample_inputs):
        raise ValueError("report provenance differs from numeric sample inputs")
    if tuple(profile_summaries) != PROFILE_IDS:
        raise ValueError("profile summaries must use frozen profile order")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (1 << 63) - 1:
        raise ValueError("seed must be in [0, 2**63 - 1]")
    raw_rows = samples["rows"]
    if not isinstance(raw_rows, list):  # already checked above; narrows for mypy
        raise AssertionError("validated samples lost their rows")
    clusters: list[int] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise AssertionError("validated sample row is not an object")
        clusters.append(int(row["cluster"]))

    estimates: dict[str, object] = {}
    for profile_id in PROFILE_IDS:
        systems: dict[str, object] = {}
        for label in RUN_LABELS:
            surfaces: dict[str, object] = {}
            for surface in SURFACES:
                metrics: dict[str, object] = {}
                for metric in METRIC_LABELS:
                    values = _metric_values(samples, profile_id, label, metric, surface)
                    metrics[metric] = _ci_object(
                        values,
                        clusters,
                        resamples=resamples,
                        seed=_seed(seed, f"estimate/{profile_id}/{label}/{surface}/{metric}"),
                    )
                surfaces[surface] = metrics
            systems[label] = surfaces
        estimates[profile_id] = systems

    baseline_id = PROFILE_IDS[1]
    efficacy_rows: list[dict[str, object]] = []
    efficacy_raw: dict[str, float] = {}
    efficacy_floors: dict[str, bool] = {}
    for treatment_id in (PROFILE_IDS[0], PROFILE_IDS[2]):
        comparison = f"{treatment_id}-vs-{baseline_id}"
        baseline = _metric_values(samples, baseline_id, RRF_LABEL, "origin_mrr_at_10", "overall")
        treatment = _metric_values(samples, treatment_id, RRF_LABEL, "origin_mrr_at_10", "overall")
        row, raw_p, at_floor = _contrast_object(
            baseline,
            treatment,
            clusters,
            resamples=resamples,
            seed=_seed(seed, f"efficacy/{comparison}"),
        )
        row.update(
            {
                "comparison": comparison,
                "baseline": baseline_id,
                "treatment": treatment_id,
                "system": RRF_LABEL,
                "metric": "origin_mrr_at_10",
                "surface": "overall",
            }
        )
        efficacy_rows.append(row)
        efficacy_raw[comparison] = raw_p
        efficacy_floors[comparison] = at_floor
    _apply_holm(efficacy_rows, efficacy_raw, efficacy_floors)

    interaction_rows: list[dict[str, object]] = []
    interaction_raw: dict[str, float] = {}
    interaction_floors: dict[str, bool] = {}
    zeros = [0.0] * EXPECTED_PAIRS
    for treatment_id in (PROFILE_IDS[0], PROFILE_IDS[2]):
        comparison = f"{treatment_id}-surface-interaction"
        alt_direct = _metric_values(samples, treatment_id, RRF_LABEL, "origin_mrr_at_10", "direct")
        alt_para = _metric_values(
            samples, treatment_id, RRF_LABEL, "origin_mrr_at_10", "paraphrase"
        )
        base_direct = _metric_values(samples, baseline_id, RRF_LABEL, "origin_mrr_at_10", "direct")
        base_para = _metric_values(
            samples, baseline_id, RRF_LABEL, "origin_mrr_at_10", "paraphrase"
        )
        interactions = [
            (ap - bp) - (ad - bd)
            for ap, bp, ad, bd in zip(alt_para, base_para, alt_direct, base_direct, strict=True)
        ]
        row, raw_p, at_floor = _contrast_object(
            zeros,
            interactions,
            clusters,
            resamples=resamples,
            seed=_seed(seed, f"interaction/{comparison}"),
        )
        row.update(
            {
                "comparison": comparison,
                "baseline": baseline_id,
                "treatment": treatment_id,
                "system": RRF_LABEL,
                "metric": "origin_mrr_at_10",
                "contrast": "(alt-400)_paraphrase-(alt-400)_direct",
            }
        )
        interaction_rows.append(row)
        interaction_raw[comparison] = raw_p
        interaction_floors[comparison] = at_floor
    _apply_holm(interaction_rows, interaction_raw, interaction_floors)

    report: dict[str, object] = {
        "schema": CHUNK_SWEEP_REPORT_SCHEMA,
        "design": {
            "estimand": "400-origin exploratory known-item source retrieval",
            "primary_system": RRF_LABEL,
            "primary_metric": "origin_mrr_at_10",
            "pair_observations": EXPECTED_PAIRS,
            "source_clusters": EXPECTED_CLUSTERS,
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "alternative": "two-sided",
            "correction": "Holm-Bonferroni within each named two-test family",
            "source_collapse": "first occurrence after each final chunk arm; RRF before collapse",
        },
        "inputs": dict(provenance),
        "profiles": {profile_id: dict(profile_summaries[profile_id]) for profile_id in PROFILE_IDS},
        "samples": {
            "schema": CHUNK_SWEEP_SAMPLES_SCHEMA,
            "sha256": numeric_samples_sha256(samples),
        },
        "estimates": estimates,
        "families": {
            "origin-source-efficacy": efficacy_rows,
            "surface-home-field": interaction_rows,
        },
        "execution_contract": build_execution_contract(),
        "limitations": list(_LIMITATIONS),
    }
    validate_report(report)
    return report


def _walk_forbidden(value: object, *, context: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context}: public object has a non-string key")
            if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{context}.{key}: forbidden raw field")
            _walk_forbidden(child, context=f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk_forbidden(child, context=f"{context}[{index}]")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _non_negative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _validate_report_profiles(raw: object) -> None:
    if not isinstance(raw, Mapping) or set(raw) != set(PROFILE_IDS):
        raise ValueError("chunk-sweep report profile keys drift")
    for profile_id in PROFILE_IDS:
        summary = raw[profile_id]
        if not isinstance(summary, Mapping) or summary.get("profile_id") != profile_id:
            raise ValueError(f"{profile_id}: report profile summary drift")
        chunks = summary.get("chunks")
        if chunks is not None and chunks != PROFILES[profile_id].expected_chunks:
            raise ValueError(f"{profile_id}: report profile chunk count drift")
        for key, value in summary.items():
            if isinstance(key, str) and key.endswith("_sha256") and not _is_sha256(value):
                raise ValueError(f"{profile_id}: invalid {key}")


def _validate_report_estimates(raw: object) -> None:  # noqa: PLR0912
    if not isinstance(raw, Mapping) or set(raw) != set(PROFILE_IDS):
        raise ValueError("report estimates profile keys drift")
    ci_keys = {"mean", "ci_low", "ci_high", "observations", "clusters"}
    for profile_id in PROFILE_IDS:
        systems = raw[profile_id]
        if not isinstance(systems, Mapping) or set(systems) != set(RUN_LABELS):
            raise ValueError(f"{profile_id}: estimate system keys drift")
        for label in RUN_LABELS:
            surfaces = systems[label]
            if not isinstance(surfaces, Mapping) or set(surfaces) != set(SURFACES):
                raise ValueError(f"{profile_id}/{label}: estimate surface keys drift")
            for surface in SURFACES:
                metrics = surfaces[surface]
                if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_LABELS):
                    raise ValueError(f"{profile_id}/{label}/{surface}: metric keys drift")
                for metric in METRIC_LABELS:
                    interval = _exact_mapping(
                        metrics[metric],
                        keys=ci_keys,
                        context=f"estimate/{profile_id}/{label}/{surface}/{metric}",
                    )
                    for key in ("mean", "ci_low", "ci_high"):
                        value = _finite(
                            interval[key],
                            context=f"estimate/{profile_id}/{label}/{surface}/{metric}/{key}",
                        )
                        if not 0.0 <= value <= 1.0:
                            raise ValueError("retrieval estimate must be in [0, 1]")
                    if _finite(interval["ci_low"], context="ci_low") > _finite(
                        interval["ci_high"], context="ci_high"
                    ):
                        raise ValueError("retrieval estimate CI is reversed")
                    if interval["observations"] != EXPECTED_PAIRS:
                        raise ValueError("retrieval estimate observation count drift")
                    if interval["clusters"] != EXPECTED_CLUSTERS:
                        raise ValueError("retrieval estimate cluster count drift")


def _validate_family(  # noqa: PLR0912
    name: str,
    raw_rows: object,
    *,
    resamples: int,
) -> None:
    if not isinstance(raw_rows, list) or len(raw_rows) != 2:
        raise ValueError(f"{name}: expected exactly two tests")
    baseline_id = PROFILE_IDS[1]
    treatments = (PROFILE_IDS[0], PROFILE_IDS[2])
    raw_pvalues: dict[str, float] = {}
    raw_floors: dict[str, bool] = {}
    rows_by_comparison: dict[str, Mapping[str, Any]] = {}
    common = {
        "comparison",
        "baseline",
        "treatment",
        "system",
        "metric",
        "delta",
        "ci_low",
        "ci_high",
        "raw_p",
        "raw_p_at_floor",
        "wins",
        "losses",
        "ties_nonzero",
        "ties_zero",
        "observations",
        "clusters",
        "adjusted_p",
        "reject",
        "adjusted_p_inherits_floor",
    }
    discriminator = "surface" if name == "origin-source-efficacy" else "contrast"
    for treatment_id, raw_row in zip(treatments, raw_rows, strict=True):
        row = _exact_mapping(
            raw_row,
            keys={*common, discriminator},
            context=f"{name}/{treatment_id}",
        )
        expected_comparison = (
            f"{treatment_id}-vs-{baseline_id}"
            if name == "origin-source-efficacy"
            else f"{treatment_id}-surface-interaction"
        )
        if (
            row["comparison"] != expected_comparison
            or row["baseline"] != baseline_id
            or row["treatment"] != treatment_id
            or row["system"] != RRF_LABEL
            or row["metric"] != "origin_mrr_at_10"
        ):
            raise ValueError(f"{name}: comparison identity drift")
        if name == "origin-source-efficacy" and row["surface"] != "overall":
            raise ValueError(f"{name}: primary surface drift")
        if name == "surface-home-field" and row["contrast"] != (
            "(alt-400)_paraphrase-(alt-400)_direct"
        ):
            raise ValueError(f"{name}: interaction definition drift")
        for key in ("delta", "ci_low", "ci_high"):
            value = _finite(row[key], context=f"{name}/{expected_comparison}/{key}")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name}: contrast estimate must be in [-1, 1]")
        if _finite(row["ci_low"], context="ci_low") > _finite(row["ci_high"], context="ci_high"):
            raise ValueError(f"{name}: contrast CI is reversed")
        raw_p = _finite(row["raw_p"], context=f"{name}/{expected_comparison}/raw_p")
        adjusted_p = _finite(row["adjusted_p"], context=f"{name}/{expected_comparison}/adjusted_p")
        if not 0.0 <= raw_p <= 1.0 or not 0.0 <= adjusted_p <= 1.0:
            raise ValueError(f"{name}: p-values must be in [0, 1]")
        at_floor = math.isclose(
            raw_p,
            bootstrap_p_floor(resamples),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        if row["raw_p_at_floor"] is not at_floor:
            raise ValueError(f"{name}: raw p-value floor provenance drift")
        if not isinstance(row["reject"], bool) or not isinstance(
            row["adjusted_p_inherits_floor"], bool
        ):
            raise ValueError(f"{name}: Holm flags must be boolean")
        counts = [
            _non_negative_integer(row[key], context=f"{name}/{key}")
            for key in ("wins", "losses", "ties_nonzero", "ties_zero")
        ]
        if sum(counts) != EXPECTED_PAIRS:
            raise ValueError(f"{name}: W/L/T counts do not cover pairs")
        if row["observations"] != EXPECTED_PAIRS or row["clusters"] != EXPECTED_CLUSTERS:
            raise ValueError(f"{name}: inferential unit counts drift")
        raw_pvalues[expected_comparison] = raw_p
        raw_floors[expected_comparison] = at_floor
        rows_by_comparison[expected_comparison] = row

    adjusted = holm_bonferroni(raw_pvalues)
    adjusted_floors = holm_floor_flags(raw_pvalues, raw_floors)
    for comparison, row in rows_by_comparison.items():
        expected_adjusted, expected_reject = adjusted[comparison]
        if not math.isclose(
            float(row["adjusted_p"]),
            expected_adjusted,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{name}: Holm-adjusted p-value drift")
        if row["reject"] is not expected_reject:
            raise ValueError(f"{name}: Holm decision drift")
        if row["adjusted_p_inherits_floor"] is not adjusted_floors[comparison]:
            raise ValueError(f"{name}: Holm floor provenance drift")


def validate_report(raw: object) -> None:
    root = _exact_mapping(
        raw,
        keys={
            "schema",
            "design",
            "inputs",
            "profiles",
            "samples",
            "estimates",
            "families",
            "execution_contract",
            "limitations",
        },
        context="chunk-sweep report",
    )
    if root["schema"] != CHUNK_SWEEP_REPORT_SCHEMA:
        raise ValueError("chunk-sweep report schema drift")
    _walk_forbidden(raw)
    design = _exact_mapping(
        root["design"],
        keys=set(_DESIGN_KEYS),
        context="chunk-sweep design",
    )
    expected_design: Mapping[str, object] = {
        "estimand": "400-origin exploratory known-item source retrieval",
        "primary_system": RRF_LABEL,
        "primary_metric": "origin_mrr_at_10",
        "pair_observations": EXPECTED_PAIRS,
        "source_clusters": EXPECTED_CLUSTERS,
        "confidence": 0.95,
        "alternative": "two-sided",
        "correction": "Holm-Bonferroni within each named two-test family",
        "source_collapse": "first occurrence after each final chunk arm; RRF before collapse",
    }
    if any(design[key] != value for key, value in expected_design.items()):
        raise ValueError("chunk-sweep report design drift")
    resamples = _non_negative_integer(design["resamples"], context="design.resamples")
    if resamples < 1:
        raise ValueError("design.resamples must be positive")
    seed = _non_negative_integer(design["seed"], context="design.seed")
    if seed > (1 << 63) - 1:
        raise ValueError("design.seed exceeds the frozen range")

    inputs = root["inputs"]
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("chunk-sweep report inputs must be a non-empty object")
    if any(
        not isinstance(key, str) or not key or not _is_sha256(value)
        for key, value in inputs.items()
    ):
        raise ValueError("chunk-sweep report input fingerprints are invalid")
    _validate_report_profiles(root["profiles"])
    sample_ref = _exact_mapping(
        root["samples"],
        keys={"schema", "sha256"},
        context="chunk-sweep sample reference",
    )
    if sample_ref["schema"] != CHUNK_SWEEP_SAMPLES_SCHEMA or not _is_sha256(sample_ref["sha256"]):
        raise ValueError("chunk-sweep sample reference drift")
    _validate_report_estimates(root["estimates"])

    families = root["families"]
    family_names = ("origin-source-efficacy", "surface-home-field")
    if not isinstance(families, Mapping) or set(families) != set(family_names):
        raise ValueError("chunk-sweep report family keys drift")
    for family in family_names:
        _validate_family(family, families[family], resamples=resamples)

    execution_contract = root["execution_contract"]
    if execution_contract != build_execution_contract():
        raise ValueError("chunk-sweep execution contract drift")
    if root["limitations"] != list(_LIMITATIONS):
        raise ValueError("chunk-sweep report limitations drift")
    _walk_forbidden(raw)
