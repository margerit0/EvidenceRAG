"""Opt-in integration check for :class:`zhrag.store.MilvusStore`.

Run with the isolated environment documented in ``smoke_milvus_lite.py``:

    PYTHONPATH=src .venv-verify-milvus/Scripts/python.exe scripts/verify_milvus_store.py

The parent uses a worker process because Windows may keep Milvus Lite's LOCK
file open briefly after ``MilvusClient.close()``. All rows are synthetic.

The worker runs with loopback added to ``no_proxy``/``NO_PROXY``. Milvus Lite
spawns a local gRPC server on 127.0.0.1, and gRPC honours
``HTTP_PROXY``/``HTTPS_PROXY``: with a proxy exported, the loopback handshake is
routed away and fails as ``code=2, illegal connection params or server
unavailable`` -- which reads like a dead server rather than like a proxied
client. ``GRPC_ENABLE_HTTP_PROXY=0`` alone was *not* sufficient with
pymilvus 3.0.1 on this machine; the bypass list is what takes effect.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from zhrag.store import ChunkRecord, MilvusConfig, MilvusStore

DB_PATH = Path("milvus-store-verify.db")
COLLECTION = "verify_store"
ALIAS = "verify_active"
SHA256 = "a" * 64


def _remove_db(*, retries: int = 20) -> None:
    for attempt in range(retries):
        if not DB_PATH.exists():
            return
        try:
            shutil.rmtree(DB_PATH)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.1)


def _record(
    doc_id: str,
    text: str,
    dense: tuple[float, ...],
    sparse: tuple[tuple[int, float], ...],
) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        text=text,
        dense=dense,
        sparse=sparse,
        source_key=f"synthetic:{doc_id}",
        document_sha256=SHA256,
        ordinal=0,
        metadata={"fixture": True},
    )


def _worker_environment() -> dict[str, str]:
    """Copy the environment with loopback excluded from any HTTP proxy."""
    environment = dict(os.environ)
    bypass = "127.0.0.1,localhost"
    for name in ("no_proxy", "NO_PROXY"):
        existing = environment.get(name, "")
        environment[name] = f"{existing},{bypass}" if existing else bypass
    environment["GRPC_ENABLE_HTTP_PROXY"] = "0"
    return environment


def _run_parent() -> None:
    _remove_db()
    subprocess.run(
        [sys.executable, __file__, "--worker"],
        check=True,
        env=_worker_environment(),
    )
    _remove_db()
    assert not DB_PATH.exists()


def _run_worker() -> None:
    store = MilvusStore(
        MilvusConfig(
            uri=str(DB_PATH),
            collection_name=COLLECTION,
            dense_dimensions=3,
        )
    )
    try:
        store.ensure_collection()
        store.ensure_collection()
        rows = [
            _record("a", "甲", (1.0, 0.0, 0.0), ((0, 2.0),)),
            _record("b", "乙", (0.8, 0.2, 0.0), ((1, 3.0),)),
            _record("c", "丙", (0.0, 1.0, 0.0), ((0, 1.0),)),
        ]
        assert store.upsert(rows) == 3
        assert store.count() == 3
        assert [hit.doc_id for hit in store.search_dense((1.0, 0.0, 0.0), limit=3)] == [
            "a",
            "b",
            "c",
        ]
        assert [hit.doc_id for hit in store.search_sparse(((0, 1.0),), limit=2)] == [
            "a",
            "c",
        ]
        assert [row.doc_id for row in store.fetch(["c", "missing", "a"])] == ["c", "a"]
        assert store.alias_target(ALIAS) is None
        store.activate_alias(ALIAS)
        assert store.alias_target(ALIAS) == COLLECTION
        store.activate_alias(ALIAS)
        assert store.upsert([_record("a", "甲更新", (1.0, 0.0, 0.0), ((0, 4.0),))]) == 1
        [updated] = store.fetch(["a"])
        assert updated.text == "甲更新"
        assert store.delete(["b", "missing"]) == 2
        assert store.fetch(["b"]) == ()
    finally:
        store.close()

    reopened = MilvusStore(
        MilvusConfig(
            uri=str(DB_PATH),
            collection_name=COLLECTION,
            dense_dimensions=3,
        )
    )
    try:
        reopened.ensure_collection()
        assert reopened.count() == 2
        assert reopened.alias_target(ALIAS) == COLLECTION
    finally:
        reopened.close()
    print("PASS: MilvusStore schema, arms, rows, alias, and reopen")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _run_worker()
    else:
        _run_parent()


if __name__ == "__main__":
    main()
