"""Client-side character-bigram BM25 as Milvus sparse vectors.

Milvus' built-in Chinese analyzer is jieba search mode, which loses to the
character-bigram baseline measured by this project. The database therefore gets
precomputed BM25 document contributions in a ``SPARSE_FLOAT_VECTOR`` and a
binary vector of unique query terms. Their inner product is the same score as
:class:`zhrag.lexical.bm25.BM25`, without asking Milvus to tokenize or estimate
segment-local IDF statistics.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from zhrag.io_utils import read_json, write_json
from zhrag.lexical.analyzers import char_ngram
from zhrag.lexical.bm25 import BM25Params

__all__ = [
    "SPARSE_INDEX_SCHEMA",
    "SparseBuild",
    "SparseIndex",
    "SparseVector",
    "build_sparse_index",
    "read_sparse_index",
    "sparse_dot",
    "write_sparse_index",
]

type SparseVector = tuple[tuple[int, float], ...]

SPARSE_INDEX_SCHEMA = "zhrag-sparse-index-v1"
_ANALYZE = char_ngram(2)
_DEFAULT_PARAMS = BM25Params()
_FINGERPRINT_SCHEMA = "zhrag-char-bigram-bm25-v1"


def _update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _term_index(terms: tuple[str, ...], term: str) -> int | None:
    position = bisect_left(terms, term)
    return position if position < len(terms) and terms[position] == term else None


@dataclass(frozen=True, slots=True)
class SparseIndex:
    """Immutable vocabulary and corpus statistics for one physical collection."""

    terms: tuple[str, ...]
    inverse_document_frequency: tuple[float, ...]
    document_count: int
    average_document_length: float
    params: BM25Params
    fingerprint: str

    def __post_init__(self) -> None:
        if self.document_count < 1:
            raise ValueError("document_count must be positive")
        if len(self.terms) != len(self.inverse_document_frequency):
            raise ValueError("terms and inverse_document_frequency differ in length")
        if tuple(sorted(set(self.terms))) != self.terms:
            raise ValueError("terms must be unique and sorted")
        if not math.isfinite(self.average_document_length) or self.average_document_length <= 0:
            raise ValueError("average_document_length must be finite and positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.inverse_document_frequency):
            raise ValueError("inverse document frequencies must be finite and positive")

    def encode_query(self, text: str) -> SparseVector:
        """Encode unique in-vocabulary bigrams with binary query weights."""
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        indices = {
            index
            for term in set(_ANALYZE(text))
            if (index := _term_index(self.terms, term)) is not None
        }
        return tuple((index, 1.0) for index in sorted(indices))

    def encode_document(self, text: str) -> SparseVector:
        """Encode one document using this index's frozen BM25 statistics."""
        if not isinstance(text, str):
            raise TypeError("document text must be a string")
        frequencies = Counter(_ANALYZE(text))
        length = sum(frequencies.values())
        if not length:
            return ()

        k1, b = self.params.k1, self.params.b
        norm = 1 - b + b * length / self.average_document_length
        values: list[tuple[int, float]] = []
        for term, frequency in frequencies.items():
            index = _term_index(self.terms, term)
            if index is None:
                continue
            weight = (
                self.inverse_document_frequency[index]
                * (frequency * (k1 + 1))
                / (frequency + k1 * norm)
            )
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"non-finite or non-positive sparse weight for term {term!r}")
            values.append((index, weight))
        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class SparseBuild:
    """A sparse index and one aligned document vector per sorted document id."""

    index: SparseIndex
    document_ids: tuple[str, ...]
    document_vectors: tuple[SparseVector, ...]

    def __post_init__(self) -> None:
        if len(self.document_ids) != len(self.document_vectors):
            raise ValueError("document ids and vectors differ in length")
        if tuple(sorted(set(self.document_ids))) != self.document_ids:
            raise ValueError("document ids must be unique and sorted")
        if len(self.document_ids) != self.index.document_count:
            raise ValueError("document vectors do not match the index document count")

    def vector_for(self, document_id: str) -> SparseVector:
        """Return a document vector by id without exposing a mutable mapping."""
        position = bisect_left(self.document_ids, document_id)
        if position >= len(self.document_ids) or self.document_ids[position] != document_id:
            raise KeyError(document_id)
        return self.document_vectors[position]


def _fingerprint(corpus: Mapping[str, str], params: BM25Params) -> str:
    digest = hashlib.sha256()
    _update(digest, _FINGERPRINT_SCHEMA)
    _update(digest, params.k1.hex())
    _update(digest, params.b.hex())
    for document_id in sorted(corpus):
        _update(digest, document_id)
        _update(digest, corpus[document_id])
    return digest.hexdigest()


def build_sparse_index(
    corpus: Mapping[str, str],
    *,
    params: BM25Params = _DEFAULT_PARAMS,
) -> SparseBuild:
    """Build deterministic BM25 vectors for a complete physical collection.

    BM25 statistics are corpus-global: adding or deleting one chunk can change
    IDF and average length for every row. Callers must therefore rebuild this
    object whenever the materialized lexical corpus changes rather than upserting
    only the changed document's sparse vector.
    """
    if not corpus:
        raise ValueError("cannot build a sparse index over an empty corpus")
    if not math.isfinite(params.k1) or params.k1 < 0:
        raise ValueError("k1 must be finite and non-negative")
    if not math.isfinite(params.b) or not 0 <= params.b <= 1:
        raise ValueError("b must be finite and between zero and one")

    document_ids = tuple(sorted(corpus))
    if any(not isinstance(document_id, str) or not document_id for document_id in document_ids):
        raise ValueError("document ids must be non-empty strings")
    if any(not isinstance(corpus[document_id], str) for document_id in document_ids):
        raise TypeError("document texts must be strings")

    frequencies = [Counter(_ANALYZE(corpus[document_id])) for document_id in document_ids]
    lengths = [sum(row.values()) for row in frequencies]
    average_length = sum(lengths) / len(lengths) or 1.0

    document_frequency: Counter[str] = Counter()
    for row in frequencies:
        document_frequency.update(row.keys())
    terms = tuple(sorted(document_frequency))
    count = len(document_ids)
    idf = tuple(
        math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in terms
    )
    index = SparseIndex(
        terms=terms,
        inverse_document_frequency=idf,
        document_count=count,
        average_document_length=average_length,
        params=params,
        fingerprint=_fingerprint(corpus, params),
    )
    vectors = tuple(index.encode_document(corpus[document_id]) for document_id in document_ids)
    return SparseBuild(index=index, document_ids=document_ids, document_vectors=vectors)


def sparse_dot(left: SparseVector, right: SparseVector) -> float:
    """Compute a sparse inner product for parity tests and local diagnostics."""
    left_position = 0
    right_position = 0
    products: list[float] = []
    while left_position < len(left) and right_position < len(right):
        left_index, left_value = left[left_position]
        right_index, right_value = right[right_position]
        if left_index == right_index:
            products.append(left_value * right_value)
            left_position += 1
            right_position += 1
        elif left_index < right_index:
            left_position += 1
        else:
            right_position += 1
    return math.fsum(products)


def write_sparse_index(path: str | Path, index: SparseIndex) -> None:
    """Persist the vocabulary and corpus statistics beside the collection.

    Without this, querying a published collection would require the original
    documents plus an identical re-chunking pass, because a query vector's term
    indexes are only meaningful against the vocabulary the documents were encoded
    with. The file is derived data and stays gitignored.
    """
    write_json(
        path,
        {
            "schema": SPARSE_INDEX_SCHEMA,
            "fingerprint": index.fingerprint,
            "document_count": index.document_count,
            "average_document_length": index.average_document_length,
            "k1": index.params.k1,
            "b": index.params.b,
            "terms": list(index.terms),
            "inverse_document_frequency": list(index.inverse_document_frequency),
        },
        indent=None,
    )


def read_sparse_index(path: str | Path) -> SparseIndex:
    """Load a persisted vocabulary, rejecting anything written by other code."""
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != SPARSE_INDEX_SCHEMA:
        raise ValueError(f"{path}: not a {SPARSE_INDEX_SCHEMA} file")
    missing = [
        name
        for name in (
            "fingerprint",
            "document_count",
            "average_document_length",
            "k1",
            "b",
            "terms",
            "inverse_document_frequency",
        )
        if name not in raw
    ]
    if missing:
        raise ValueError(f"{path}: sparse index is missing {', '.join(missing)}")
    return SparseIndex(
        terms=tuple(str(term) for term in raw["terms"]),
        inverse_document_frequency=tuple(
            float(value) for value in raw["inverse_document_frequency"]
        ),
        document_count=int(raw["document_count"]),
        average_document_length=float(raw["average_document_length"]),
        params=BM25Params(k1=float(raw["k1"]), b=float(raw["b"])),
        fingerprint=str(raw["fingerprint"]),
    )
