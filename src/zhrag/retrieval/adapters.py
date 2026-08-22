"""Adapters binding provider clients to the online retrieval Protocols.

:mod:`zhrag.retrieval.online` deliberately knows nothing about HTTP, ``.env``, or
Milvus. These adapters are the only place where a provider client meets an online
port, and each one carries its instruction explicitly rather than defaulting to a
module constant: the frozen benchmark prompts say "news passage", so silently
reusing them for TiDB documentation would relabel one corpus's evidence as
another's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from zhrag.lexical.sparse import SparseIndex
from zhrag.providers.embedding import BatchEmbedder
from zhrag.providers.rerank import RerankClient

__all__ = ["DenseQueryAdapter", "RerankAdapter", "SparseQueryAdapter"]


@dataclass(frozen=True, slots=True)
class DenseQueryAdapter:
    """Embed one query under an explicit asymmetric instruction."""

    client: BatchEmbedder
    prompt: str
    dimensions: int

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")

    def encode(self, query: str) -> Sequence[float]:
        vectors = self.client.embed_all([self.prompt + query], batch=1, label="query")
        if len(vectors) != 1:
            raise RuntimeError(f"embedding provider returned {len(vectors)} vectors for one query")
        vector = vectors[0]
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"embedding provider returned {len(vector)} dimensions, expected {self.dimensions}"
            )
        if any(not math.isfinite(value) for value in vector):
            raise RuntimeError("embedding provider returned a non-finite value")
        return tuple(float(value) for value in vector)


@dataclass(frozen=True, slots=True)
class SparseQueryAdapter:
    """Encode one query with the physical collection's frozen vocabulary.

    The index must be the one the collection was built from. A vocabulary from a
    different build assigns different term indexes, and the inner product would
    then score against whatever terms happen to occupy those positions -- a
    silently wrong ranking rather than an error.
    """

    index: SparseIndex

    def encode_query(self, query: str) -> tuple[tuple[int, float], ...]:
        return self.index.encode_query(query)


@dataclass(frozen=True, slots=True)
class RerankAdapter:
    """Score passages with an explicit instruction, in request order."""

    client: RerankClient
    instruction: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return self.client.score(query, documents, instruction=self.instruction).scores
