"""Vendor-neutral storage contracts and optional database implementations."""

from zhrag.store.base import (
    ArmHit,
    ChunkRecord,
    DenseVector,
    MetadataValue,
    Passage,
    VectorStore,
)
from zhrag.store.milvus import MilvusConfig, MilvusStore, MilvusUnavailableError
from zhrag.store.tidb import TiDBConfig, TiDBStore, TiDBUnavailableError

__all__ = [
    "ArmHit",
    "ChunkRecord",
    "DenseVector",
    "MetadataValue",
    "MilvusConfig",
    "MilvusStore",
    "MilvusUnavailableError",
    "Passage",
    "TiDBConfig",
    "TiDBStore",
    "TiDBUnavailableError",
    "VectorStore",
]
