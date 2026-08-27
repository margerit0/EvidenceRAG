"""Retrieval interfaces with lazy optional provider adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zhrag.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from zhrag.retrieval.adapters import DenseQueryAdapter, RerankAdapter, SparseQueryAdapter
    from zhrag.retrieval.online import (
        DenseQueryEncoder,
        OnlineRetrievalResult,
        OnlineRetriever,
        OnlineSettings,
        PassageReranker,
        RankedPassage,
        SparseQueryEncoder,
        StageTimings,
    )

__all__ = [
    "DenseQueryAdapter",
    "DenseQueryEncoder",
    "OnlineRetrievalResult",
    "OnlineRetriever",
    "OnlineSettings",
    "PassageReranker",
    "RankedPassage",
    "RerankAdapter",
    "SparseQueryAdapter",
    "SparseQueryEncoder",
    "StageTimings",
    "reciprocal_rank_fusion",
]

_ADAPTER_EXPORTS = frozenset(
    {
        "DenseQueryAdapter",
        "RerankAdapter",
        "SparseQueryAdapter",
    }
)
_ONLINE_EXPORTS = frozenset(
    {
        "DenseQueryEncoder",
        "OnlineRetrievalResult",
        "OnlineRetriever",
        "OnlineSettings",
        "PassageReranker",
        "RankedPassage",
        "SparseQueryEncoder",
        "StageTimings",
    }
)


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_EXPORTS:
        from zhrag.retrieval import adapters  # noqa: PLC0415 - lazy provider boundary

        return getattr(adapters, name)
    if name in _ONLINE_EXPORTS:
        from zhrag.retrieval import online  # noqa: PLC0415 - lazy provider boundary

        return getattr(online, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
