"""Vendor-neutral vector-store contracts used by ingestion and retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from zhrag.lexical.sparse import SparseVector

__all__ = [
    "ArmHit",
    "ChunkRecord",
    "DenseVector",
    "MetadataValue",
    "Passage",
    "VectorStore",
]

type DenseVector = tuple[float, ...]
type MetadataValue = str | int | float | bool | None


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _freeze_metadata(
    metadata: Mapping[str, MetadataValue],
) -> Mapping[str, MetadataValue]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen = dict(metadata)
    for key, value in frozen.items():
        _validate_identifier(key, "metadata key")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"metadata value for {key!r} is not a supported scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One complete row written to the vector store."""

    doc_id: str
    text: str
    dense: DenseVector
    sparse: SparseVector
    source_key: str
    document_sha256: str
    ordinal: int
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.doc_id, "doc_id")
        _validate_identifier(self.source_key, "source_key")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("text must be a non-empty string")
        if len(self.document_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.document_sha256
        ):
            raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.dense or any(not math.isfinite(value) for value in self.dense):
            raise ValueError("dense vector must be non-empty and finite")
        previous = -1
        for index, value in self.sparse:
            if index <= previous or index < 0:
                raise ValueError("sparse indices must be unique, non-negative, and sorted")
            if not math.isfinite(value) or value == 0:
                raise ValueError("sparse values must be finite and non-zero")
            previous = index
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ArmHit:
    """One normalized result from either dense or sparse retrieval."""

    doc_id: str
    score: float

    def __post_init__(self) -> None:
        _validate_identifier(self.doc_id, "doc_id")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class Passage:
    """Text and provenance fetched for one candidate id."""

    doc_id: str
    text: str
    source_key: str
    document_sha256: str
    ordinal: int
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.doc_id, "doc_id")
        _validate_identifier(self.source_key, "source_key")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if len(self.document_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.document_sha256
        ):
            raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


class VectorStore(Protocol):
    """The online and ingestion operations independent of a database vendor."""

    @property
    def collection_name(self) -> str: ...

    @property
    def dense_dimensions(self) -> int: ...

    def ensure_collection(self) -> None: ...

    def upsert(self, records: Sequence[ChunkRecord]) -> int: ...

    def delete(self, doc_ids: Sequence[str]) -> int: ...

    def search_dense(self, vector: DenseVector, *, limit: int) -> Sequence[ArmHit]: ...

    def search_sparse(self, vector: SparseVector, *, limit: int) -> Sequence[ArmHit]: ...

    def fetch(self, doc_ids: Sequence[str]) -> Sequence[Passage]: ...

    def count(self) -> int: ...

    def activate_alias(self, alias: str) -> None: ...

    def alias_target(self, alias: str) -> str | None: ...

    def close(self) -> None: ...
