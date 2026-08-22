"""Manifest-driven ingestion identity, scope, and delta computation.

Everything here is deterministic and offline: it turns a checked-in manifest plus
raw document bytes into stable identities and a document-level delta. Embedding,
sparse construction, and any database write live in ``scripts/build_index.py``,
so the parts that decide *what changed* can be tested without a corpus, a
provider key, or Milvus.

Two identity rules are load-bearing:

* The stable source key is ``pingcap/docs-cn:<upstream path>``. ``collection`` is
  routing metadata that a re-curation can change; folding it into identity would
  turn a re-labelled document into a delete plus an add and discard its vectors.
* A chunk id hashes the source key, the ordinal, and the exact contextual text.
  Chunk ids are therefore stable while a document's text is stable, and change
  the moment the indexed text does -- which is what makes reusing a cached dense
  vector safe.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from zhrag.chunking import Chunk, chunk_markdown
from zhrag.io_utils import read_bytes, read_json, write_json

__all__ = [
    "DEFAULT_NAMESPACE",
    "STATE_SCHEMA",
    "ChunkPlan",
    "DocumentDelta",
    "DocumentPlan",
    "IngestState",
    "ManifestDocument",
    "Scope",
    "chunk_id",
    "chunker_fingerprint",
    "diff_documents",
    "document_loader",
    "document_sha256",
    "git_blob_sha1",
    "load_manifest",
    "plan_document",
    "plan_documents",
    "read_state",
    "reusable_chunk_ids",
    "scope_fingerprint",
    "source_key",
    "write_state",
]

DEFAULT_NAMESPACE = "pingcap/docs-cn"
STATE_SCHEMA = "zhrag-ingest-state-v1"
_CHUNK_ID_SCHEMA = "zhrag-chunk-id-v1"
_CHUNKER_SCHEMA = "zhrag-chunker-v1"
_METADATA_SCHEMA = "zhrag-chunk-metadata-v1"
_SCOPE_SCHEMA = "zhrag-ingest-scope-v1"
_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def git_blob_sha1(data: bytes) -> str:
    """Recompute the Git blob SHA-1 that the manifest recorded from the API."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def document_sha256(data: bytes) -> str:
    """Hash raw document bytes; this is the version key, not the Git blob id."""
    return hashlib.sha256(data).hexdigest()


def source_key(path: str, *, namespace: str = DEFAULT_NAMESPACE) -> str:
    """Return the namespaced, routing-independent identity of one document."""
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"manifest path must be a relative POSIX path, got {path!r}")
    if not namespace:
        raise ValueError("namespace must be non-empty")
    return f"{namespace}:{path}"


def chunk_id(key: str, ordinal: int, contextual_text: str) -> str:
    """Hash the exact indexed text under a versioned, length-framed schema.

    Length framing matters: concatenating the fields directly would let a source
    key ending in a digit and a shifted ordinal produce the same byte string as a
    different document, silently merging two chunks under one primary key.
    """
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    digest = hashlib.sha256()
    _update(digest, _CHUNK_ID_SCHEMA)
    _update(digest, key)
    _update(digest, str(ordinal))
    _update(digest, contextual_text)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """One validated manifest row. ``blob_sha1``/``size`` are integrity checks."""

    path: str
    blob_sha1: str
    size: int
    collection: str
    theme: str
    category: str
    source_url: str

    def __post_init__(self) -> None:
        if not _HEX40.match(self.blob_sha1):
            raise ValueError(f"{self.path}: sha must be 40 lowercase hexadecimal characters")
        if self.size < 0:
            raise ValueError(f"{self.path}: size must be non-negative")
        for name in ("collection", "theme", "category"):
            if not getattr(self, name):
                raise ValueError(f"{self.path}: {name} must be non-empty")

    @property
    def key(self) -> str:
        return source_key(self.path)

    def relative_path(self) -> Path:
        """Return the on-disk location under the curated documents root."""
        return Path(self.collection) / self.path


@dataclass(frozen=True, slots=True)
class Scope:
    """Which manifest rows belong in one physical collection."""

    include_collections: frozenset[str] | None = None
    exclude_collections: frozenset[str] = frozenset({"temporal_releases"})
    include_themes: frozenset[str] | None = None

    def selects(self, document: ManifestDocument) -> bool:
        if document.collection in self.exclude_collections:
            return False
        if (
            self.include_collections is not None
            and document.collection not in self.include_collections
        ):
            return False
        return self.include_themes is None or document.theme in self.include_themes

    @classmethod
    def evergreen(cls) -> Scope:
        """Exclude ``temporal_releases``: release notes date, evergreen docs do not."""
        return cls()


def scope_fingerprint(scope: Scope) -> str:
    """Hash a scope so a narrowed selection cannot silently look like deletions."""
    digest = hashlib.sha256()
    _update(digest, _SCOPE_SCHEMA)
    for name in ("include_collections", "exclude_collections", "include_themes"):
        value = getattr(scope, name)
        _update(digest, name)
        _update(digest, "*" if value is None else ",".join(sorted(value)))
    return digest.hexdigest()


def chunker_fingerprint(
    *,
    target_tokens: int,
    hard_max_tokens: int,
    strip_links: bool = True,
) -> str:
    """Hash the chunker settings that decide the indexed text of every chunk."""
    digest = hashlib.sha256()
    _update(digest, _CHUNKER_SCHEMA)
    _update(digest, str(target_tokens))
    _update(digest, str(hard_max_tokens))
    _update(digest, str(strip_links))
    return digest.hexdigest()


def load_manifest(path: str | Path) -> tuple[ManifestDocument, ...]:
    """Read and validate the curated manifest, rejecting duplicate paths."""
    raw = read_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ValueError(f"{path}: manifest has no documents list")
    documents: list[ManifestDocument] = []
    seen: set[str] = set()
    for row in raw["documents"]:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: manifest row is not an object")
        missing = [
            name
            for name in ("path", "sha", "size", "collection", "theme", "category")
            if name not in row
        ]
        if missing:
            raise ValueError(f"{path}: manifest row is missing {', '.join(missing)}")
        document = ManifestDocument(
            path=str(row["path"]),
            blob_sha1=str(row["sha"]),
            size=int(row["size"]),
            collection=str(row["collection"]),
            theme=str(row["theme"]),
            category=str(row["category"]),
            source_url=str(row.get("source_url", "")),
        )
        if document.path in seen:
            raise ValueError(f"{path}: duplicate manifest path {document.path!r}")
        seen.add(document.path)
        documents.append(document)
    if not documents:
        raise ValueError(f"{path}: manifest selects no documents")
    return tuple(sorted(documents, key=lambda item: item.path))


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One chunk with its permanent id and the exact text that will be indexed."""

    chunk_id: str
    ordinal: int
    contextual_text: str
    heading_path: tuple[str, ...]
    approx_tokens: int


@dataclass(frozen=True, slots=True)
class DocumentPlan:
    """One in-scope document, verified against the manifest and chunked."""

    key: str
    path: str
    document_sha256: str
    metadata_fingerprint: str
    metadata: Mapping[str, str]
    chunks: tuple[ChunkPlan, ...]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self.chunks)


@dataclass(frozen=True, slots=True)
class DocumentDelta:
    """What one build changes, relative to the previously published state."""

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    metadata_only: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "added": len(self.added),
                "updated": len(self.updated),
                "metadata_only": len(self.metadata_only),
                "unchanged": len(self.unchanged),
                "deleted": len(self.deleted),
                "out_of_scope": len(self.out_of_scope),
            }
        )


def _metadata_fingerprint(metadata: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    _update(digest, _METADATA_SCHEMA)
    for name in sorted(metadata):
        _update(digest, name)
        _update(digest, metadata[name])
    return digest.hexdigest()


def plan_document(
    document: ManifestDocument,
    data: bytes,
    *,
    target_tokens: int = 400,
    hard_max_tokens: int = 600,
    verify_blob_sha1: bool = True,
) -> DocumentPlan:
    """Verify one document's bytes against the manifest, then chunk it."""
    if len(data) != document.size:
        raise ValueError(
            f"{document.path}: manifest records {document.size} bytes, file has {len(data)}"
        )
    if verify_blob_sha1:
        actual = git_blob_sha1(data)
        if actual != document.blob_sha1:
            raise ValueError(
                f"{document.path}: manifest sha {document.blob_sha1} != file sha {actual}"
            )
    metadata: dict[str, str] = {
        "collection": document.collection,
        "theme": document.theme,
        "category": document.category,
        "path": document.path,
        "source_url": document.source_url,
    }
    chunks = chunk_markdown(
        data.decode("utf-8"),
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )
    key = document.key
    planned = tuple(_plan_chunk(key, chunk) for chunk in chunks)
    ids = [chunk.chunk_id for chunk in planned]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{document.path}: chunking produced duplicate chunk ids")
    return DocumentPlan(
        key=key,
        path=document.path,
        document_sha256=document_sha256(data),
        metadata_fingerprint=_metadata_fingerprint(metadata),
        metadata=MappingProxyType(metadata),
        chunks=planned,
    )


def _plan_chunk(key: str, chunk: Chunk) -> ChunkPlan:
    text = chunk.contextual_text
    return ChunkPlan(
        chunk_id=chunk_id(key, chunk.ordinal, text),
        ordinal=chunk.ordinal,
        contextual_text=text,
        heading_path=chunk.heading_path,
        approx_tokens=chunk.approx_tokens,
    )


def plan_documents(
    manifest: Sequence[ManifestDocument],
    *,
    scope: Scope,
    load: Callable[[ManifestDocument], bytes],
    target_tokens: int = 400,
    hard_max_tokens: int = 600,
    verify_blob_sha1: bool = True,
) -> tuple[DocumentPlan, ...]:
    """Plan every in-scope document in stable manifest order."""
    return tuple(
        plan_document(
            document,
            load(document),
            target_tokens=target_tokens,
            hard_max_tokens=hard_max_tokens,
            verify_blob_sha1=verify_blob_sha1,
        )
        for document in manifest
        if scope.selects(document)
    )


def document_loader(root: str | Path) -> Callable[[ManifestDocument], bytes]:
    """Return a loader reading curated documents under ``root``."""
    base = Path(root)

    def load(document: ManifestDocument) -> bytes:
        path = base / document.relative_path()
        if not path.is_file():
            raise FileNotFoundError(f"{document.path}: missing curated file at {path}")
        return read_bytes(path)

    return load


@dataclass(frozen=True, slots=True)
class IngestState:
    """The last successfully published build, keyed by stable source key."""

    scope: str
    chunker_fingerprint: str
    embedding_profile: str
    sparse_fingerprint: str
    collection_name: str
    documents: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def chunk_ids(self) -> frozenset[str]:
        ids: set[str] = set()
        for record in self.documents.values():
            raw = record.get("chunk_ids")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                ids.update(str(value) for value in raw)
        return frozenset(ids)


def read_state(path: str | Path) -> IngestState | None:
    """Load a published state, or ``None`` when this is the first build."""
    if not Path(path).is_file():
        return None
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != STATE_SCHEMA:
        raise ValueError(f"{path}: not a {STATE_SCHEMA} state file")
    documents = raw.get("documents")
    if not isinstance(documents, dict):
        raise ValueError(f"{path}: state has no documents mapping")
    for key, record in documents.items():
        if not isinstance(record, dict) or not _HEX64.match(str(record.get("document_sha256"))):
            raise ValueError(f"{path}: state entry {key!r} has no document_sha256")
    return IngestState(
        scope=str(raw.get("scope", "")),
        chunker_fingerprint=str(raw.get("chunker_fingerprint", "")),
        embedding_profile=str(raw.get("embedding_profile", "")),
        sparse_fingerprint=str(raw.get("sparse_fingerprint", "")),
        collection_name=str(raw.get("collection_name", "")),
        documents=MappingProxyType({str(key): dict(value) for key, value in documents.items()}),
    )


def write_state(path: str | Path, state: IngestState) -> None:
    """Write a state file. Callers must only do this after a verified build."""
    write_json(
        path,
        {
            "schema": STATE_SCHEMA,
            "scope": state.scope,
            "chunker_fingerprint": state.chunker_fingerprint,
            "embedding_profile": state.embedding_profile,
            "sparse_fingerprint": state.sparse_fingerprint,
            "collection_name": state.collection_name,
            "documents": {key: dict(value) for key, value in state.documents.items()},
        },
    )


def diff_documents(
    planned: Iterable[DocumentPlan],
    previous: IngestState | None,
    *,
    scope: Scope,
    manifest: Sequence[ManifestDocument],
) -> DocumentDelta:
    """Classify one build against the previous state without guessing deletions.

    A previously indexed document that the current manifest still lists but this
    scope no longer selects is reported as ``out_of_scope``; one the manifest no
    longer lists at all is ``deleted``. Both remove rows, but only the second
    means the upstream document disappeared, and conflating them would make a
    narrowed scope read as upstream churn in the build report.
    """
    desired = {plan.key: plan for plan in planned}
    manifest_keys = {document.key for document in manifest}
    in_scope_keys = {document.key for document in manifest if scope.selects(document)}
    missing_plans = sorted(in_scope_keys - set(desired))
    if missing_plans:
        # Trusting a partial plan here would delete rows the manifest still
        # selects, so an incomplete planning pass must stop the build instead.
        raise ValueError(
            f"{len(missing_plans)} in-scope documents were not planned: "
            f"{', '.join(missing_plans[:3])}"
        )
    recorded = dict(previous.documents) if previous is not None else {}

    added: list[str] = []
    updated: list[str] = []
    metadata_only: list[str] = []
    unchanged: list[str] = []
    for key, plan in sorted(desired.items()):
        record = recorded.get(key)
        if record is None:
            added.append(key)
        elif str(record.get("document_sha256")) != plan.document_sha256:
            updated.append(key)
        elif str(record.get("metadata_fingerprint", "")) != plan.metadata_fingerprint:
            metadata_only.append(key)
        else:
            unchanged.append(key)

    dropped = [key for key in recorded if key not in desired]
    deleted = sorted(key for key in dropped if key not in manifest_keys)
    out_of_scope = sorted(key for key in dropped if key in manifest_keys)
    return DocumentDelta(
        added=tuple(added),
        updated=tuple(updated),
        deleted=tuple(deleted),
        unchanged=tuple(unchanged),
        metadata_only=tuple(metadata_only),
        out_of_scope=tuple(out_of_scope),
    )


def reusable_chunk_ids(
    planned: Iterable[DocumentPlan],
    previous: IngestState | None,
    *,
    embedding_profile: str,
) -> frozenset[str]:
    """Return chunk ids whose cached dense vectors this build may reuse.

    A chunk id already binds the source key, ordinal, and exact indexed text, so
    the only remaining way a cached vector can be wrong is a different embedding
    profile. When the profile changed, nothing is reusable -- reusing a vector
    from another model or prompt convention would leave the index silently mixed.
    """
    if previous is None or previous.embedding_profile != embedding_profile:
        return frozenset()
    published = previous.chunk_ids()
    return frozenset(
        chunk.chunk_id for plan in planned for chunk in plan.chunks if chunk.chunk_id in published
    )
