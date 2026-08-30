"""Offline artifact authentication shared by the TiDB chunk-sweep commands.

The builder, evaluator, and documentation synchronizer must certify the same
canonical state, query cache, qrels, profile sidecars, and profile markers. This
module is provider-free and centralises that I/O contract so those three commands
cannot silently disagree about what a complete M7 profile means.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from zhrag.eval.tidb_chunk_sweep import (
    BATCH_SIZE,
    CACHE_META_SCHEMA,
    CHUNK_SWEEP_RELATIVE,
    CHUNK_SWEEP_REPORT_SCHEMA,
    CHUNK_SWEEP_SAMPLES_SCHEMA,
    DOCUMENT_EMBEDDING_MODEL,
    DOCUMENT_EMBEDDING_PROFILE,
    DOCUMENT_PROMPT,
    EXPECTED_DOCUMENTS,
    EXPECTED_PAIRS,
    EXPECTED_QUERIES,
    LEGACY_CHUNK_SWEEP_REPORT_SCHEMA,
    LEGACY_LIMITATIONS,
    LEGACY_PROFILE_ARTIFACT_SCHEMA,
    LEGACY_PROFILE_REPORT_SCHEMA,
    LEGACY_PROVIDER_USAGE,
    PROFILE_ARTIFACT_SCHEMA,
    PROFILE_IDS,
    PROFILE_REPORT_SCHEMA,
    PROFILES,
    QUERY_EMBEDDING_PROFILE,
    QUERY_PROMPT,
    RETRIEVAL_DEPTH,
    VECTOR_POLICY,
    VECTOR_WIDTH,
    EmbeddingCache,
    PlannedProfile,
    SourceQrels,
    build_report,
    build_source_qrels,
    embedding_cache_fingerprint,
    materialize_profile,
    numeric_samples_sha256,
    public_profile_summary,
    strict_embedding_cache,
    validate_exact_embedding_reuse,
    validate_numeric_samples,
    validate_report,
)
from zhrag.eval.tidb_quality import (
    QRELS_FINGERPRINT_SCHEMA,
    QrelsBundle,
    parse_qrels,
    parse_runs,
    validate_quality_report,
)
from zhrag.eval.tidb_runs import DENSE_LABEL, build_dense_runs
from zhrag.ingest import (
    DocumentPlan,
    Scope,
    chunker_fingerprint,
    document_loader,
    load_manifest,
    plan_documents,
    read_state,
    scope_fingerprint,
)
from zhrag.io_utils import read_bytes, read_json, read_jsonl

__all__ = [
    "QUERY_CACHE_PROVENANCE_SCHEMA",
    "CanonicalSnapshot",
    "ChunkSweepPaths",
    "FinalizedProfile",
    "ProfileFiles",
    "authenticate_qrels_quality_anchor",
    "build_query_cache_provenance",
    "classify_metadata_generation",
    "file_sha256",
    "ids_fingerprint",
    "legacy_profile_artifact",
    "legacy_profile_report",
    "load_canonical_snapshot",
    "load_finalized_profile",
    "load_legacy_finalized_profile",
    "migrate_numeric_samples_v1",
    "migrate_report_v1",
    "profile_artifact",
    "profile_report",
    "profile_sidecar",
    "query_text_fingerprint",
    "validate_legacy_profile_artifact",
    "validate_legacy_profile_report",
    "validate_profile_sidecar",
    "validate_query_cache_provenance",
    "verify_query_cache_dense_runs",
]


_METADATA_GENERATION_V1 = "v1"
_METADATA_GENERATION_V2 = "v2"
_METADATA_GENERATION_MIXED = "mixed"
_METADATA_GENERATION_UNKNOWN = "unknown"
_LEGACY_PROFILE_ARTIFACT_KEYS = frozenset({"cache_contract", "inputs", "schema", "summary"})
_LEGACY_PROFILE_REPORT_KEYS = frozenset(
    {
        "cache_file_sha256",
        "cache_mode",
        "cache_sidecar_sha256",
        "complete",
        "embedding_matrix_fingerprint_sha256",
        "endpoint_provenance",
        "profile_artifact_sha256",
        "profile_id",
        "schema",
        "split_exceedance_ledger_sha256",
    }
)
_PROFILE_SUMMARY_COMMON_KEYS = frozenset(
    {
        "chunk_ids_fingerprint_sha256",
        "chunker_fingerprint_sha256",
        "chunks",
        "corpus_fingerprint_sha256",
        "document_set_fingerprint_sha256",
        "documents",
        "embedding_input_estimate",
        "embedding_matrix_fingerprint_sha256",
        "profile_fingerprint_sha256",
        "profile_id",
        "source_mapping_fingerprint_sha256",
        "split_trigger_exceedance",
        "split_trigger_tokens",
        "target_tokens",
        "tokens",
    }
)
_LEGACY_PROFILE_SUMMARY_KEYS = _PROFILE_SUMMARY_COMMON_KEYS | {
    "canonical_exact_reuse",
    "new_document_vectors",
    "successful_embedding_batches",
}
_V2_PROFILE_SUMMARY_KEYS = _PROFILE_SUMMARY_COMMON_KEYS | {
    "canonical_exact_reuse",
    "required_embedding_batches",
    "required_new_document_vectors",
}
_LEGACY_METADATA_RELATIVE_FILES = (
    *(Path("profiles") / profile_id / "profile.json" for profile_id in PROFILE_IDS),
    *(Path("profiles") / profile_id / "cache_report.json" for profile_id in PROFILE_IDS),
    Path("numeric_samples.json"),
    Path("report.json"),
)


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without allowing bool/int equality surprises."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (bool, int, float)) or isinstance(right, (bool, int, float)):
        return type(left) is type(right) and left == right
    return type(left) is type(right) and left == right


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    assert isinstance(value, str)
    return value


def _mapping_with_keys(value: object, keys: set[str], *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if set(value) != keys:
        raise ValueError(f"{context} keys differ from the frozen schema")
    return value


def _legacy_summary_from_v2(summary: Mapping[str, object], *, profile_id: str) -> dict[str, object]:
    if set(summary) != _V2_PROFILE_SUMMARY_KEYS:
        raise ValueError(f"{profile_id}: v2 profile summary keys differ")
    if summary.get("profile_id") != profile_id:
        raise ValueError(f"{profile_id}: v2 profile summary identity drift")
    converted = copy.deepcopy(dict(summary))
    converted["new_document_vectors"] = converted.pop("required_new_document_vectors")
    converted["successful_embedding_batches"] = converted.pop("required_embedding_batches")
    return converted


def _v2_summary_from_legacy(summary: Mapping[str, object], *, profile_id: str) -> dict[str, object]:
    if set(summary) != _LEGACY_PROFILE_SUMMARY_KEYS:
        raise ValueError(f"{profile_id}: legacy profile summary keys differ")
    if summary.get("profile_id") != profile_id:
        raise ValueError(f"{profile_id}: legacy profile summary identity drift")
    converted = copy.deepcopy(dict(summary))
    converted["required_new_document_vectors"] = converted.pop("new_document_vectors")
    converted["required_embedding_batches"] = converted.pop("successful_embedding_batches")
    return converted


def validate_legacy_profile_artifact(raw: object) -> None:
    """Validate the exact JSON shape of one v1 profile artifact."""
    root = _mapping_with_keys(
        raw,
        set(_LEGACY_PROFILE_ARTIFACT_KEYS),
        context="legacy profile artifact",
    )
    if root["schema"] != LEGACY_PROFILE_ARTIFACT_SCHEMA:
        raise ValueError("legacy profile artifact schema drift")
    summary = root["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("legacy profile artifact summary must be an object")
    if set(summary) != _LEGACY_PROFILE_SUMMARY_KEYS:
        raise ValueError("legacy profile artifact summary keys differ")
    profile_id = summary.get("profile_id")
    if not isinstance(profile_id, str) or profile_id not in PROFILE_IDS:
        raise ValueError("legacy profile artifact profile identity drift")
    inputs = root["inputs"]
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("legacy profile artifact inputs are malformed")
    if any(not isinstance(key, str) or not _is_sha256(value) for key, value in inputs.items()):
        raise ValueError("legacy profile artifact input fingerprints are invalid")
    contract = root["cache_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "mode",
        "model",
        "prompt",
        "width",
        "sidecar_schema",
        "vector_policy",
    }:
        raise ValueError("legacy profile artifact cache contract is malformed")
    _require_sha256(summary["chunk_ids_fingerprint_sha256"], context="legacy chunk IDs")
    _require_sha256(summary["chunker_fingerprint_sha256"], context="legacy chunker")
    _require_sha256(summary["corpus_fingerprint_sha256"], context="legacy corpus")
    _require_sha256(summary["document_set_fingerprint_sha256"], context="legacy document set")
    _require_sha256(summary["embedding_matrix_fingerprint_sha256"], context="legacy matrix")
    _require_sha256(summary["profile_fingerprint_sha256"], context="legacy profile")
    _require_sha256(summary["source_mapping_fingerprint_sha256"], context="legacy source map")


def validate_legacy_profile_report(raw: object) -> None:
    """Validate the exact JSON shape of one v1 profile marker."""
    root = _mapping_with_keys(
        raw,
        set(_LEGACY_PROFILE_REPORT_KEYS),
        context="legacy profile report",
    )
    if root["schema"] != LEGACY_PROFILE_REPORT_SCHEMA:
        raise ValueError("legacy profile report schema drift")
    profile_id = root["profile_id"]
    if not isinstance(profile_id, str) or profile_id not in PROFILE_IDS:
        raise ValueError("legacy profile report profile identity drift")
    if root["complete"] is not True:
        raise ValueError("legacy profile report is not complete")
    if root["cache_mode"] not in {"profile-local-complete", "canonical-reference-no-copy"}:
        raise ValueError("legacy profile report cache mode drift")
    if root["endpoint_provenance"] not in {"recorded-live-endpoint", "legacy-unknown"}:
        raise ValueError("legacy profile report endpoint provenance drift")
    for key in (
        "cache_file_sha256",
        "cache_sidecar_sha256",
        "embedding_matrix_fingerprint_sha256",
        "profile_artifact_sha256",
        "split_exceedance_ledger_sha256",
    ):
        _require_sha256(root[key], context=f"legacy profile report {key}")


def _metadata_schema(path: Path) -> str | None:
    if not path.is_file():
        return None
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        return None
    schema = raw.get("schema")
    return schema if isinstance(schema, str) else None


def classify_metadata_generation(paths: ChunkSweepPaths) -> str:
    """Classify the complete metadata bundle without trusting its contents yet."""
    schemas: list[str | None] = []
    for relative in _LEGACY_METADATA_RELATIVE_FILES:
        schemas.append(_metadata_schema(paths.sweep_root / relative))
    v1_schemas = {
        LEGACY_PROFILE_ARTIFACT_SCHEMA,
        LEGACY_PROFILE_REPORT_SCHEMA,
        LEGACY_CHUNK_SWEEP_REPORT_SCHEMA,
    }
    v2_schemas = {
        PROFILE_ARTIFACT_SCHEMA,
        PROFILE_REPORT_SCHEMA,
        CHUNK_SWEEP_REPORT_SCHEMA,
    }
    generations: set[str] = set()
    unknown = False
    for schema in schemas:
        if schema in v1_schemas:
            generations.add(_METADATA_GENERATION_V1)
        elif schema in v2_schemas:
            generations.add(_METADATA_GENERATION_V2)
        elif schema == CHUNK_SWEEP_SAMPLES_SCHEMA:
            continue
        else:
            unknown = True
    if len(generations) > 1:
        return _METADATA_GENERATION_MIXED
    if unknown or not generations:
        return _METADATA_GENERATION_UNKNOWN
    return next(iter(generations))


QUERY_CACHE_PROVENANCE_SCHEMA = "zhrag-tidb-m7-query-cache-adoption-v1"
_QUERY_TEXT_FINGERPRINT_SCHEMA = "zhrag-tidb-m7-query-text-v1"
_QUERY_CACHE_ADOPTION_LIMITATION = (
    "post-hoc adoption: historical provider request texts were not recorded; exact "
    "reproduction of the frozen dense top-100 run binds this matrix to the published "
    "benchmark behaviour, not to an observed request payload"
)


@dataclass(frozen=True, slots=True)
class ChunkSweepPaths:
    artifacts: Path
    curated: Path

    @property
    def eval_root(self) -> Path:
        return self.artifacts / "eval"

    @property
    def sweep_root(self) -> Path:
        return self.artifacts / CHUNK_SWEEP_RELATIVE

    @property
    def manifest(self) -> Path:
        return self.curated / "selected_manifest.json"

    @property
    def documents(self) -> Path:
        return self.curated / "documents"

    @property
    def state(self) -> Path:
        return self.artifacts / "state.json"

    @property
    def canonical_cache(self) -> Path:
        return self.artifacts / "dense_cache.jsonl"

    @property
    def qrels(self) -> Path:
        return self.eval_root / "qrels.jsonl"

    @property
    def qrels_report(self) -> Path:
        return self.eval_root / "qrels_report.json"

    @property
    def quality_report(self) -> Path:
        return self.eval_root / "quality_report.json"

    @property
    def queries(self) -> Path:
        return self.eval_root / "queries.jsonl"

    @property
    def query_cache(self) -> Path:
        return self.eval_root / "query_embeddings_4096.jsonl"

    @property
    def runs(self) -> Path:
        return self.eval_root / "runs.jsonl"

    @property
    def query_cache_provenance(self) -> Path:
        return self.sweep_root / "query_cache_provenance.json"

    def profile_root(self, profile_id: str) -> Path:
        return self.sweep_root / "profiles" / profile_id

    def profile_files(self, profile_id: str) -> ProfileFiles:
        return ProfileFiles(self.profile_root(profile_id))


@dataclass(frozen=True, slots=True)
class ProfileFiles:
    root: Path

    @property
    def cache(self) -> Path:
        return self.root / "document_embeddings_4096.jsonl"

    @property
    def sidecar(self) -> Path:
        return self.root / "document_embeddings_4096.jsonl.meta.json"

    @property
    def ledger(self) -> Path:
        return self.root / "split_exceedance_ledger.jsonl"

    @property
    def profile(self) -> Path:
        return self.root / "profile.json"

    @property
    def report(self) -> Path:
        return self.root / "cache_report.json"


@dataclass(frozen=True, slots=True)
class CanonicalSnapshot:
    profiles: Mapping[str, PlannedProfile]
    canonical_cache: EmbeddingCache
    query_cache: EmbeddingCache
    qrels: QrelsBundle
    source_qrels: SourceQrels
    reusable_ids: Mapping[str, tuple[str, ...]]
    new_ids: Mapping[str, tuple[str, ...]]
    input_fingerprints: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FinalizedProfile:
    profile_id: str
    planned: PlannedProfile
    embeddings: EmbeddingCache
    artifact: Mapping[str, object]
    report: Mapping[str, object]

    @property
    def public_summary(self) -> Mapping[str, object]:
        raw = self.artifact["summary"]
        if not isinstance(raw, Mapping):  # pragma: no cover - constructor authority
            raise AssertionError("validated profile artifact lost its summary")
        return raw


def file_sha256(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def ids_fingerprint(schema: str, ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    _update(digest, schema)
    for item_id in ids:
        _update(digest, item_id)
    return digest.hexdigest()


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing required M7 input files: {', '.join(missing)}")


def _legacy_sidecar(cache: Path, *, model: str, prompt: str) -> None:
    sidecar = cache.with_name(f"{cache.name}.meta.json")
    if read_json(sidecar) != {"model": model, "prompt": prompt}:
        raise ValueError(f"{sidecar.name}: legacy embedding provenance drift")


def _validate_state(
    planned: PlannedProfile,
    documents: Sequence[DocumentPlan],
    state_path: Path,
) -> None:
    state = read_state(state_path)
    if state is None:
        raise ValueError("canonical TiDB state is absent")
    if state.scope != scope_fingerprint(Scope.evergreen()):
        raise ValueError("canonical TiDB state has the wrong evergreen scope")
    if state.chunker_fingerprint != chunker_fingerprint(
        target_tokens=400,
        hard_max_tokens=600,
    ):
        raise ValueError("canonical TiDB state has the wrong 400/600 chunker")
    if state.embedding_profile != DOCUMENT_EMBEDDING_PROFILE:
        raise ValueError("canonical TiDB state has the wrong embedding profile")
    if len(state.documents) != EXPECTED_DOCUMENTS:
        raise ValueError("canonical TiDB state does not contain 450 documents")
    planned_by_key = {document.key: document for document in documents}
    if set(state.documents) != set(planned_by_key):
        raise ValueError("canonical state source keys differ from the re-plan")
    for source, document in planned_by_key.items():
        record = state.documents[source]
        raw_ids = record.get("chunk_ids")
        if not isinstance(raw_ids, list) or tuple(raw_ids) != document.chunk_ids:
            raise ValueError(f"canonical state chunk order drift for {source}")
        if record.get("document_sha256") != document.document_sha256:
            raise ValueError(f"canonical state document digest drift for {source}")
        if record.get("metadata_fingerprint") != document.metadata_fingerprint:
            raise ValueError(f"canonical state metadata digest drift for {source}")
    if state.chunk_ids() != frozenset(planned.chunk_ids):
        raise ValueError("canonical state chunk universe differs from the re-plan")


def _validate_query_fixture(path: Path, qrels: QrelsBundle) -> None:
    seen: dict[str, str] = {}
    for row_number, row in enumerate(read_jsonl(path), 1):
        query_id = row.get("query_id")
        question = row.get("question")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"queries row {row_number}: invalid query_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"queries row {row_number}: invalid question")
        if query_id in seen:
            raise ValueError(f"queries row {row_number}: duplicate query_id")
        seen[query_id] = question
    if set(seen) != set(qrels.by_query):
        raise ValueError("query fixture IDs differ from final qrels")
    for query_id, surface in qrels.by_query.items():
        if seen[query_id] != surface.question:
            raise ValueError(f"query fixture text drift for {query_id}")


def query_text_fingerprint(qrels: QrelsBundle) -> str:
    digest = hashlib.sha256()
    _update(digest, _QUERY_TEXT_FINGERPRINT_SCHEMA)
    for query_id in sorted(qrels.by_query):
        _update(digest, query_id)
        _update(digest, qrels.by_query[query_id].question)
    return digest.hexdigest()


def verify_query_cache_dense_runs(
    query_cache: EmbeddingCache,
    canonical_cache: EmbeddingCache,
    canonical_chunk_ids: Sequence[str],
    qrels: QrelsBundle,
    frozen_run_rows: Sequence[Mapping[str, object]],
) -> str:
    """Require this matrix to reproduce every frozen dense top-100 row exactly."""
    runs = parse_runs(frozen_run_rows)
    query_ids = tuple(sorted(qrels.by_query))
    if set(runs.query_ids) != set(query_ids):
        raise ValueError("frozen runs and query cache use different query IDs")
    positions = {query_id: position for position, query_id in enumerate(runs.query_ids)}
    expected = tuple(runs.runs[DENSE_LABEL][positions[query_id]] for query_id in query_ids)
    rebuilt = build_dense_runs(
        query_cache.matrix(query_ids),
        canonical_cache.matrix(canonical_chunk_ids),
        canonical_chunk_ids,
        depth=RETRIEVAL_DEPTH,
    )
    if rebuilt != expected:
        mismatches = sum(left != right for left, right in zip(rebuilt, expected, strict=True))
        raise ValueError(
            f"query cache does not reproduce the frozen dense top-100 runs "
            f"({mismatches:,}/{len(query_ids):,} rows differ)"
        )
    return runs.fingerprint


def build_query_cache_provenance(
    *,
    query_cache: EmbeddingCache,
    qrels: QrelsBundle,
    dense_runs_fingerprint: str,
    query_cache_file_sha256: str,
    legacy_sidecar_file_sha256: str,
    quality_report_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema": QUERY_CACHE_PROVENANCE_SCHEMA,
        "profile": QUERY_EMBEDDING_PROFILE,
        "model": DOCUMENT_EMBEDDING_MODEL,
        "prompt": QUERY_PROMPT,
        "width": VECTOR_WIDTH,
        "query_count": len(qrels.by_query),
        "query_set_fingerprint_sha256": qrels.query_set_fingerprint,
        "query_text_fingerprint_sha256": query_text_fingerprint(qrels),
        "query_matrix_fingerprint_sha256": query_cache.fingerprint,
        "query_cache_file_sha256": query_cache_file_sha256,
        "legacy_sidecar_file_sha256": legacy_sidecar_file_sha256,
        "dense_runs_fingerprint_sha256": dense_runs_fingerprint,
        "dense_run_reproduction": {
            "label": DENSE_LABEL,
            "depth": RETRIEVAL_DEPTH,
            "rows": len(qrels.by_query),
            "mismatches": 0,
        },
        "quality_report_file_sha256": quality_report_file_sha256,
        "historical_request_text_binding": "not-observed",
        "provenance": "post-hoc-offline-adoption",
        "limitation": _QUERY_CACHE_ADOPTION_LIMITATION,
    }


def validate_query_cache_provenance(
    raw: object,
    *,
    query_cache: EmbeddingCache,
    qrels: QrelsBundle,
    dense_runs_fingerprint: str,
    query_cache_file_sha256: str,
    legacy_sidecar_file_sha256: str,
    quality_report_file_sha256: str,
) -> None:
    expected = build_query_cache_provenance(
        query_cache=query_cache,
        qrels=qrels,
        dense_runs_fingerprint=dense_runs_fingerprint,
        query_cache_file_sha256=query_cache_file_sha256,
        legacy_sidecar_file_sha256=legacy_sidecar_file_sha256,
        quality_report_file_sha256=quality_report_file_sha256,
    )
    if raw != expected:
        raise ValueError("immutable M7 query-cache provenance drift")


def authenticate_qrels_quality_anchor(
    qrels: QrelsBundle,
    quality_report: object,
    *,
    expected_queries: int,
    expected_pairs: int,
) -> Mapping[str, object]:
    """Require a validated quality report to certify these exact qrels semantics."""
    if not isinstance(quality_report, dict):
        raise ValueError("quality report must be an object")
    validate_quality_report(quality_report)
    quality_inputs = quality_report.get("inputs")
    quality_design = quality_report.get("evaluation_design")
    if not isinstance(quality_inputs, Mapping) or not isinstance(quality_design, Mapping):
        raise ValueError("quality report lost its validated provenance sections")
    if (
        quality_design.get("queries") != expected_queries
        or quality_design.get("pairs") != expected_pairs
        or quality_inputs.get("query_set_fingerprint_sha256") != qrels.query_set_fingerprint
        or quality_inputs.get("qrels_semantic_fingerprint_schema") != QRELS_FINGERPRINT_SCHEMA
        or quality_inputs.get("qrels_semantic_fingerprint_sha256") != qrels.semantic_fingerprint
    ):
        raise ValueError("quality report does not certify the final qrels semantics")
    return quality_inputs


def _load_query_cache(
    paths: ChunkSweepPaths,
    qrels: QrelsBundle,
    quality_inputs: Mapping[str, object],
    *,
    canonical_cache: EmbeddingCache,
    canonical_chunk_ids: Sequence[str],
    query_sidecar: Path,
    require_provenance: bool,
) -> tuple[EmbeddingCache, str]:
    _legacy_sidecar(
        paths.query_cache,
        model=DOCUMENT_EMBEDDING_MODEL,
        prompt=QUERY_PROMPT,
    )
    query_cache = strict_embedding_cache(
        paths.query_cache,
        tuple(sorted(qrels.by_query)),
        width=VECTOR_WIDTH,
        allow_partial=False,
    )
    dense_runs_fingerprint = verify_query_cache_dense_runs(
        query_cache,
        canonical_cache,
        canonical_chunk_ids,
        qrels,
        tuple(read_jsonl(paths.runs)),
    )
    if quality_inputs.get("runs_fingerprint_sha256") != dense_runs_fingerprint:
        raise ValueError("quality report does not certify the frozen runs")
    if require_provenance:
        validate_query_cache_provenance(
            read_json(paths.query_cache_provenance),
            query_cache=query_cache,
            qrels=qrels,
            dense_runs_fingerprint=dense_runs_fingerprint,
            query_cache_file_sha256=file_sha256(paths.query_cache),
            legacy_sidecar_file_sha256=file_sha256(query_sidecar),
            quality_report_file_sha256=file_sha256(paths.quality_report),
        )
    return query_cache, dense_runs_fingerprint


def load_canonical_snapshot(
    paths: ChunkSweepPaths,
    *,
    require_query_cache_provenance: bool = True,
) -> CanonicalSnapshot:
    canonical_sidecar = paths.canonical_cache.with_name(f"{paths.canonical_cache.name}.meta.json")
    query_sidecar = paths.query_cache.with_name(f"{paths.query_cache.name}.meta.json")
    required = [
        paths.manifest,
        paths.state,
        paths.canonical_cache,
        canonical_sidecar,
        paths.queries,
        paths.query_cache,
        query_sidecar,
        paths.runs,
        paths.qrels,
        paths.qrels_report,
        paths.quality_report,
    ]
    if require_query_cache_provenance:
        required.append(paths.query_cache_provenance)
    _require_files(required)
    manifest = load_manifest(paths.manifest)
    scope = Scope.evergreen()
    loader = document_loader(paths.documents)
    profiles: dict[str, PlannedProfile] = {}
    canonical_documents: tuple[DocumentPlan, ...] | None = None
    for profile_id in PROFILE_IDS:
        profile = PROFILES[profile_id]
        documents = plan_documents(
            manifest,
            scope=scope,
            load=loader,
            target_tokens=profile.target_tokens,
            hard_max_tokens=profile.split_trigger_tokens,
        )
        profiles[profile_id] = materialize_profile(profile, documents)
        if profile_id == PROFILE_IDS[1]:
            canonical_documents = documents
    if canonical_documents is None:  # pragma: no cover - frozen registry invariant
        raise AssertionError("canonical profile is absent from registry")

    canonical = profiles[PROFILE_IDS[1]]
    _validate_state(canonical, canonical_documents, paths.state)
    _legacy_sidecar(
        paths.canonical_cache,
        model=DOCUMENT_EMBEDDING_MODEL,
        prompt=DOCUMENT_PROMPT,
    )
    canonical_cache = strict_embedding_cache(
        paths.canonical_cache,
        canonical.chunk_ids,
        width=VECTOR_WIDTH,
        allow_partial=False,
    )

    qrels = parse_qrels(read_jsonl(paths.qrels))
    qrels_report = read_json(paths.qrels_report)
    if not isinstance(qrels_report, Mapping):
        raise ValueError("qrels report must be an object")
    if (
        qrels_report.get("queries") != EXPECTED_QUERIES
        or qrels_report.get("pairs") != EXPECTED_PAIRS
        or qrels_report.get("query_set_fingerprint_sha256") != qrels.query_set_fingerprint
    ):
        raise ValueError("qrels report does not certify the final query set")

    quality_report = read_json(paths.quality_report)
    quality_inputs = authenticate_qrels_quality_anchor(
        qrels,
        quality_report,
        expected_queries=EXPECTED_QUERIES,
        expected_pairs=EXPECTED_PAIRS,
    )
    _validate_query_fixture(paths.queries, qrels)
    source_qrels = build_source_qrels(qrels, canonical.source_by_chunk)

    query_cache, dense_runs_fingerprint = _load_query_cache(
        paths,
        qrels,
        quality_inputs,
        canonical_cache=canonical_cache,
        canonical_chunk_ids=canonical.chunk_ids,
        query_sidecar=query_sidecar,
        require_provenance=require_query_cache_provenance,
    )

    canonical_ids = set(canonical.chunk_ids)
    reusable_ids: dict[str, tuple[str, ...]] = {}
    new_ids: dict[str, tuple[str, ...]] = {}
    for profile_id in PROFILE_IDS:
        profile = PROFILES[profile_id]
        ordered = profiles[profile_id].chunk_ids
        reused = tuple(item_id for item_id in ordered if item_id in canonical_ids)
        fresh = tuple(item_id for item_id in ordered if item_id not in canonical_ids)
        if len(reused) != profile.expected_canonical_reuse:
            raise ValueError(f"{profile_id}: canonical exact-ID reuse count drift")
        if len(fresh) != profile.expected_new_vectors:
            raise ValueError(f"{profile_id}: new-vector count drift")
        if math.ceil(len(fresh) / BATCH_SIZE) != profile.expected_paid_batches:
            raise ValueError(f"{profile_id}: paid batch count drift")
        reusable_ids[profile_id] = reused
        new_ids[profile_id] = fresh

    fingerprints = {
        "manifest_file_sha256": file_sha256(paths.manifest),
        "canonical_state_file_sha256": file_sha256(paths.state),
        "canonical_document_matrix_sha256": canonical_cache.fingerprint,
        "query_fixture_file_sha256": file_sha256(paths.queries),
        "query_cache_file_sha256": file_sha256(paths.query_cache),
        "query_cache_sidecar_file_sha256": file_sha256(query_sidecar),
        "query_matrix_sha256": query_cache.fingerprint,
        "query_text_sha256": query_text_fingerprint(qrels),
        "frozen_runs_file_sha256": file_sha256(paths.runs),
        "frozen_runs_semantic_sha256": dense_runs_fingerprint,
        "query_cache_provenance_file_sha256": (
            file_sha256(paths.query_cache_provenance)
            if require_query_cache_provenance
            else "0" * 64
        ),
        "query_set_sha256": qrels.query_set_fingerprint,
        "qrels_semantic_sha256": qrels.semantic_fingerprint,
        "qrels_file_sha256": file_sha256(paths.qrels),
        "quality_report_file_sha256": file_sha256(paths.quality_report),
        "source_qrels_sha256": source_qrels.fingerprint,
    }
    return CanonicalSnapshot(
        profiles=MappingProxyType(profiles),
        canonical_cache=canonical_cache,
        query_cache=query_cache,
        qrels=qrels,
        source_qrels=source_qrels,
        reusable_ids=MappingProxyType(reusable_ids),
        new_ids=MappingProxyType(new_ids),
        input_fingerprints=MappingProxyType(fingerprints),
    )


def profile_sidecar(
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    endpoint_url: str | None,
    endpoint_provenance: str,
) -> dict[str, object]:
    planned = snapshot.profiles[profile_id]
    reused = snapshot.reusable_ids[profile_id]
    fresh = snapshot.new_ids[profile_id]
    seed_fingerprint = embedding_cache_fingerprint(
        reused,
        snapshot.canonical_cache.vectors,
        width=VECTOR_WIDTH,
    )
    return {
        "schema": CACHE_META_SCHEMA,
        "profile_id": profile_id,
        "scope_fingerprint_sha256": scope_fingerprint(Scope.evergreen()),
        "profile_fingerprint_sha256": planned.profile.fingerprint,
        "chunker_fingerprint_sha256": planned.profile.chunker_fingerprint,
        "corpus_fingerprint_sha256": planned.corpus_fingerprint,
        "ordered_chunk_ids_sha256": planned.chunk_ids_fingerprint,
        "source_mapping_fingerprint_sha256": planned.source_mapping_fingerprint,
        "model": DOCUMENT_EMBEDDING_MODEL,
        "prompt": DOCUMENT_PROMPT,
        "width": VECTOR_WIDTH,
        "vector_policy": VECTOR_POLICY,
        "canonical_matrix_fingerprint_sha256": snapshot.canonical_cache.fingerprint,
        "canonical_seed_matrix_fingerprint_sha256": seed_fingerprint,
        "canonical_reuse_ids_sha256": ids_fingerprint("zhrag-m7-canonical-reuse-ids-v1", reused),
        "new_ids_sha256": ids_fingerprint("zhrag-m7-new-ids-v1", fresh),
        "canonical_reuse_count": len(reused),
        "new_vector_count": len(fresh),
        "batch_size": BATCH_SIZE,
        "endpoint": {"provenance": endpoint_provenance, "url": endpoint_url},
        "input_fingerprints": dict(snapshot.input_fingerprints),
    }


def validate_profile_sidecar(
    raw: object,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> tuple[dict[str, object], str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{profile_id}: cache sidecar must be an object")
    endpoint = raw.get("endpoint")
    if not isinstance(endpoint, dict) or set(endpoint) != {"provenance", "url"}:
        raise ValueError(f"{profile_id}: endpoint provenance is malformed")
    provenance = endpoint["provenance"]
    url = endpoint["url"]
    if profile_id == PROFILE_IDS[1]:
        if provenance != "legacy-unknown" or url is not None:
            raise ValueError(f"{profile_id}: canonical endpoint provenance drift")
    elif provenance != "recorded-live-endpoint" or not isinstance(url, str) or not url:
        raise ValueError(f"{profile_id}: paid cache endpoint provenance is missing")
    expected = profile_sidecar(
        snapshot,
        profile_id,
        endpoint_url=cast_url(url),
        endpoint_provenance=str(provenance),
    )
    if raw != expected:
        raise ValueError(f"{profile_id}: immutable cache sidecar drift")
    return expected, str(provenance)


def cast_url(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("endpoint URL must be a string or null")


def profile_artifact(
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    matrix_fingerprint: str,
    cache_mode: str,
) -> dict[str, object]:
    return {
        "schema": PROFILE_ARTIFACT_SCHEMA,
        "summary": public_profile_summary(
            snapshot.profiles[profile_id],
            embedding_matrix_fingerprint=matrix_fingerprint,
        ),
        "inputs": dict(snapshot.input_fingerprints),
        "cache_contract": {
            "mode": cache_mode,
            "model": DOCUMENT_EMBEDDING_MODEL,
            "prompt": DOCUMENT_PROMPT,
            "width": VECTOR_WIDTH,
            "sidecar_schema": CACHE_META_SCHEMA,
            "vector_policy": VECTOR_POLICY,
        },
    }


def profile_report(
    *,
    profile_id: str,
    matrix_fingerprint: str,
    cache_mode: str,
    endpoint_provenance: str,
    profile_sha256: str,
    ledger_sha256: str,
    cache_sha256: str,
    sidecar_sha256: str,
) -> dict[str, object]:
    return {
        "schema": PROFILE_REPORT_SCHEMA,
        "profile_id": profile_id,
        "complete": True,
        "cache_mode": cache_mode,
        "endpoint_provenance": endpoint_provenance,
        "embedding_matrix_fingerprint_sha256": matrix_fingerprint,
        "profile_artifact_sha256": profile_sha256,
        "split_exceedance_ledger_sha256": ledger_sha256,
        "cache_file_sha256": cache_sha256,
        "cache_sidecar_sha256": sidecar_sha256,
    }


def legacy_profile_artifact(
    snapshot: CanonicalSnapshot,
    profile_id: str,
    *,
    matrix_fingerprint: str,
    cache_mode: str,
) -> dict[str, object]:
    """Reconstruct the exact v1 profile artifact shape for migration input."""
    current = profile_artifact(
        snapshot,
        profile_id,
        matrix_fingerprint=matrix_fingerprint,
        cache_mode=cache_mode,
    )
    summary = current["summary"]
    if not isinstance(summary, Mapping):  # pragma: no cover - builder invariant
        raise AssertionError("profile artifact lost its summary")
    legacy_summary = _legacy_summary_from_v2(summary, profile_id=profile_id)
    return {
        "schema": LEGACY_PROFILE_ARTIFACT_SCHEMA,
        "summary": legacy_summary,
        "inputs": copy.deepcopy(current["inputs"]),
        "cache_contract": copy.deepcopy(current["cache_contract"]),
    }


def legacy_profile_report(
    *,
    profile_id: str,
    matrix_fingerprint: str,
    cache_mode: str,
    endpoint_provenance: str,
    profile_sha256: str,
    ledger_sha256: str,
    cache_sha256: str,
    sidecar_sha256: str,
) -> dict[str, object]:
    """Reconstruct the exact v1 profile marker for migration authentication."""
    current = profile_report(
        profile_id=profile_id,
        matrix_fingerprint=matrix_fingerprint,
        cache_mode=cache_mode,
        endpoint_provenance=endpoint_provenance,
        profile_sha256=profile_sha256,
        ledger_sha256=ledger_sha256,
        cache_sha256=cache_sha256,
        sidecar_sha256=sidecar_sha256,
    )
    current["schema"] = LEGACY_PROFILE_REPORT_SCHEMA
    return current


def _profile_cache_for_generation(
    paths: ChunkSweepPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> tuple[EmbeddingCache, Path, str]:
    files = paths.profile_files(profile_id)
    if profile_id == PROFILE_IDS[1]:
        return (
            snapshot.canonical_cache,
            paths.canonical_cache,
            "canonical-reference-no-copy",
        )
    _require_files((files.cache,))
    embeddings = strict_embedding_cache(
        files.cache,
        snapshot.profiles[profile_id].chunk_ids,
        width=VECTOR_WIDTH,
        allow_partial=False,
    )
    validate_exact_embedding_reuse(
        embeddings,
        snapshot.canonical_cache,
        snapshot.reusable_ids[profile_id],
    )
    return embeddings, files.cache, "profile-local-complete"


def _validate_legacy_profile_files(
    paths: ChunkSweepPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> tuple[EmbeddingCache, dict[str, object], dict[str, object]]:
    if profile_id not in PROFILE_IDS:
        raise ValueError(f"unknown frozen chunk-sweep profile {profile_id!r}")
    files = paths.profile_files(profile_id)
    _require_files((files.sidecar, files.ledger, files.profile, files.report))
    embeddings, cache_path, cache_mode = _profile_cache_for_generation(
        paths,
        snapshot,
        profile_id,
    )
    _sidecar, endpoint_provenance = validate_profile_sidecar(
        read_json(files.sidecar),
        snapshot,
        profile_id,
    )
    ledger = tuple(read_jsonl(files.ledger))
    expected_ledger = tuple(dict(row) for row in snapshot.profiles[profile_id].exceedance_rows)
    if ledger != expected_ledger:
        raise ValueError(f"{profile_id}: split-trigger ledger drift")

    expected_artifact = legacy_profile_artifact(
        snapshot,
        profile_id,
        matrix_fingerprint=embeddings.fingerprint,
        cache_mode=cache_mode,
    )
    raw_artifact = read_json(files.profile)
    validate_legacy_profile_artifact(raw_artifact)
    if raw_artifact != expected_artifact:
        raise ValueError(f"{profile_id}: legacy profile artifact drift")

    expected_report = legacy_profile_report(
        profile_id=profile_id,
        matrix_fingerprint=embeddings.fingerprint,
        cache_mode=cache_mode,
        endpoint_provenance=endpoint_provenance,
        profile_sha256=file_sha256(files.profile),
        ledger_sha256=file_sha256(files.ledger),
        cache_sha256=file_sha256(cache_path),
        sidecar_sha256=file_sha256(files.sidecar),
    )
    raw_report = read_json(files.report)
    validate_legacy_profile_report(raw_report)
    if raw_report != expected_report:
        raise ValueError(f"{profile_id}: legacy profile report drift")
    return embeddings, expected_artifact, expected_report


def load_legacy_finalized_profile(
    paths: ChunkSweepPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> FinalizedProfile:
    """Authenticate one complete v1 profile before metadata migration."""
    embeddings, artifact, report = _validate_legacy_profile_files(
        paths,
        snapshot,
        profile_id,
    )
    return FinalizedProfile(
        profile_id,
        snapshot.profiles[profile_id],
        embeddings,
        MappingProxyType(artifact),
        MappingProxyType(report),
    )


def migrate_numeric_samples_v1(
    samples: Mapping[str, object],
    *,
    legacy_profile_artifact_sha256: Mapping[str, str],
    migrated_profile_artifact_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Replace only authenticated profile-artifact hashes in numeric samples."""
    validate_numeric_samples(samples)
    expected_keys = set(PROFILE_IDS)
    if set(legacy_profile_artifact_sha256) != expected_keys:
        raise ValueError("legacy profile artifact fingerprints must cover frozen profiles")
    if set(migrated_profile_artifact_sha256) != expected_keys:
        raise ValueError("migrated profile artifact fingerprints must cover frozen profiles")
    for profile_id in PROFILE_IDS:
        _require_sha256(
            legacy_profile_artifact_sha256[profile_id],
            context=f"{profile_id} legacy profile artifact fingerprint",
        )
        _require_sha256(
            migrated_profile_artifact_sha256[profile_id],
            context=f"{profile_id} migrated profile artifact fingerprint",
        )
    raw_inputs = samples["input_fingerprints"]
    if not isinstance(raw_inputs, Mapping):  # pragma: no cover - validator invariant
        raise AssertionError("validated samples lost input fingerprints")
    migrated = copy.deepcopy(dict(samples))
    migrated_inputs = migrated["input_fingerprints"]
    if not isinstance(migrated_inputs, dict):  # pragma: no cover - deepcopy invariant
        raise AssertionError("copied samples lost input fingerprints")
    target_keys = {f"{profile_id}-profile-artifact-sha256" for profile_id in PROFILE_IDS}
    for key, value in samples.items():
        if key != "input_fingerprints" and not _json_equal(value, migrated[key]):
            raise ValueError("migrated numeric samples changed non-fingerprint content")
    for key, value in raw_inputs.items():
        if key not in target_keys and not _json_equal(value, migrated_inputs[key]):
            raise ValueError("migrated numeric samples changed a non-profile fingerprint")
    for profile_id in PROFILE_IDS:
        key = f"{profile_id}-profile-artifact-sha256"
        old_value = raw_inputs.get(key)
        if old_value != legacy_profile_artifact_sha256[profile_id]:
            raise ValueError(f"{profile_id}: numeric sample legacy artifact fingerprint drift")
        migrated_inputs[key] = migrated_profile_artifact_sha256[profile_id]
    validate_numeric_samples(migrated)
    return migrated


def _legacy_report_expected_profile_summaries(
    raw: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(raw, Mapping) or set(raw) != set(PROFILE_IDS):
        raise ValueError("legacy report profile keys drift")
    converted: dict[str, dict[str, object]] = {}
    for profile_id in PROFILE_IDS:
        summary = raw[profile_id]
        if not isinstance(summary, Mapping):
            raise ValueError(f"{profile_id}: legacy report summary is malformed")
        converted[profile_id] = _v2_summary_from_legacy(summary, profile_id=profile_id)
    return converted


def migrate_report_v1(  # noqa: PLR0912 - strict migration authentication
    report: Mapping[str, object],
    *,
    old_samples: Mapping[str, object],
    new_samples: Mapping[str, object],
    profile_summaries: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild a v2 report only after authenticating every v1 aggregate."""
    root = _mapping_with_keys(
        report,
        {
            "design",
            "estimates",
            "families",
            "inputs",
            "limitations",
            "profiles",
            "provider_usage",
            "samples",
            "schema",
        },
        context="legacy chunk-sweep report",
    )
    if root["schema"] != LEGACY_CHUNK_SWEEP_REPORT_SCHEMA:
        raise ValueError("legacy chunk-sweep report schema drift")
    if not _json_equal(root["provider_usage"], LEGACY_PROVIDER_USAGE):
        raise ValueError("legacy report provider usage drift")
    if root["limitations"] != list(LEGACY_LIMITATIONS):
        raise ValueError("legacy report limitations drift")
    if "execution_contract" in root:
        raise ValueError("legacy report unexpectedly contains execution contract")

    validate_numeric_samples(old_samples)
    validate_numeric_samples(new_samples)
    old_inputs = old_samples["input_fingerprints"]
    new_inputs = new_samples["input_fingerprints"]
    if not isinstance(old_inputs, Mapping) or not isinstance(new_inputs, Mapping):
        raise AssertionError("validated samples lost input fingerprints")
    if not _json_equal(root["inputs"], old_inputs):
        raise ValueError("legacy report inputs differ from numeric samples")
    if not _json_equal(provenance, new_inputs):
        raise ValueError("migrated report provenance differs from numeric samples")

    old_sample_ref = _mapping_with_keys(
        root["samples"],
        {"schema", "sha256"},
        context="legacy chunk-sweep sample reference",
    )
    if old_sample_ref["schema"] != CHUNK_SWEEP_SAMPLES_SCHEMA or old_sample_ref[
        "sha256"
    ] != numeric_samples_sha256(old_samples):
        raise ValueError("legacy report sample reference drift")

    old_summaries = _legacy_report_expected_profile_summaries(root["profiles"])
    if tuple(profile_summaries) != PROFILE_IDS:
        raise ValueError("migrated profile summaries must use frozen profile order")
    for profile_id in PROFILE_IDS:
        summary = profile_summaries[profile_id]
        if set(summary) != _V2_PROFILE_SUMMARY_KEYS:
            raise ValueError(f"{profile_id}: migrated profile summary keys differ")
        if not _json_equal(old_summaries[profile_id], summary):
            raise ValueError(f"{profile_id}: legacy and migrated profile summaries differ")

    design = root["design"]
    if not isinstance(design, Mapping):
        raise ValueError("legacy report design is malformed")
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
        raise ValueError("legacy report design resamples/seed are invalid")

    rebuilt = build_report(
        new_samples,
        profile_summaries=profile_summaries,
        provenance=provenance,
        resamples=resamples,
        seed=seed,
    )
    for key in ("design", "estimates", "families"):
        if not _json_equal(root[key], rebuilt[key]):
            raise ValueError(f"legacy report {key} differ from deterministic rebuild")

    migrated = copy.deepcopy(rebuilt)
    validate_report(migrated)
    return migrated


def load_finalized_profile(
    paths: ChunkSweepPaths,
    snapshot: CanonicalSnapshot,
    profile_id: str,
) -> FinalizedProfile:
    files = paths.profile_files(profile_id)
    _require_files((files.sidecar, files.ledger, files.profile, files.report))
    if profile_id == PROFILE_IDS[1]:
        embeddings = snapshot.canonical_cache
        cache_path = paths.canonical_cache
        cache_mode = "canonical-reference-no-copy"
    else:
        _require_files((files.cache,))
        embeddings = strict_embedding_cache(
            files.cache,
            snapshot.profiles[profile_id].chunk_ids,
            width=VECTOR_WIDTH,
            allow_partial=False,
        )
        validate_exact_embedding_reuse(
            embeddings,
            snapshot.canonical_cache,
            snapshot.reusable_ids[profile_id],
        )
        cache_path = files.cache
        cache_mode = "profile-local-complete"

    _sidecar, endpoint_provenance = validate_profile_sidecar(
        read_json(files.sidecar),
        snapshot,
        profile_id,
    )
    ledger = tuple(read_jsonl(files.ledger))
    expected_ledger = tuple(dict(row) for row in snapshot.profiles[profile_id].exceedance_rows)
    if ledger != expected_ledger:
        raise ValueError(f"{profile_id}: split-trigger ledger drift")

    expected_artifact = profile_artifact(
        snapshot,
        profile_id,
        matrix_fingerprint=embeddings.fingerprint,
        cache_mode=cache_mode,
    )
    raw_artifact = read_json(files.profile)
    if raw_artifact != expected_artifact:
        raise ValueError(f"{profile_id}: profile artifact drift")
    expected_report = profile_report(
        profile_id=profile_id,
        matrix_fingerprint=embeddings.fingerprint,
        cache_mode=cache_mode,
        endpoint_provenance=endpoint_provenance,
        profile_sha256=file_sha256(files.profile),
        ledger_sha256=file_sha256(files.ledger),
        cache_sha256=file_sha256(cache_path),
        sidecar_sha256=file_sha256(files.sidecar),
    )
    raw_report = read_json(files.report)
    if raw_report != expected_report:
        raise ValueError(f"{profile_id}: finalized profile report drift")
    return FinalizedProfile(
        profile_id,
        snapshot.profiles[profile_id],
        embeddings,
        MappingProxyType(expected_artifact),
        MappingProxyType(expected_report),
    )
