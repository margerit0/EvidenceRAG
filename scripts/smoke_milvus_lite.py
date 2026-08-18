"""Smoke-test Milvus Lite on Windows + Python 3.13.

Run with an isolated environment, not the project environment:

    uv venv --python 3.13.5 .venv-verify-milvus
    # pymilvus 3.0.1's optional-dependency marker explicitly excludes win32,
    # so `[milvus-lite]` alone installs only pymilvus on Windows. Install the
    # two distributions explicitly until upstream changes that marker.
    uv pip install --python .venv-verify-milvus/Scripts/python.exe \
        "pymilvus==3.0.1" "milvus-lite==3.2.0"
    .venv-verify-milvus/Scripts/python.exe scripts/smoke_milvus_lite.py

Verified on 2026-08-18 with Python 3.13.5, pymilvus 3.0.1, and milvus-lite
3.2.0: 4096-dim dense search and dense+sparse RRFRanker hybrid search both
pass on Windows 11. The worker subprocess is intentional: Windows can keep the
Lite LOCK file open briefly after ``MilvusClient.close()``, but the lock releases
when the interpreter exits.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

DB_PATH = Path("milvus-lite-smoke.db")
COLLECTION = "smoke4096"
DIMENSION = 4096


def _remove_db(*, retries: int = 20) -> None:
    """Remove the Lite data directory after its native file lock is released."""
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


def _run_parent() -> None:
    """Run the DB test in a subprocess, then clean up after its process exits.

    On Windows, ``MilvusClient.close()`` can return while a native background
    thread still owns ``milvus-lite-smoke.db/LOCK``. The lock reliably releases
    at interpreter exit, so process isolation is the correct cleanup boundary.
    """
    _remove_db()
    subprocess.run([sys.executable, __file__, "--worker"], check=True)
    _remove_db()
    assert not DB_PATH.exists()


def _run_worker() -> None:
    client = MilvusClient(str(DB_PATH))
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=128)

        indexes = client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        client.create_collection(collection_name=COLLECTION, schema=schema, index_params=indexes)

        query_vector = [0.0] * DIMENSION
        query_vector[0] = 1.0
        client.insert(
            COLLECTION,
            [
                {
                    "id": "a",
                    "dense": query_vector,
                    "sparse": {0: 1.0},
                    "text": "中文向量检索",
                }
            ],
        )

        dense = client.search(
            COLLECTION,
            [query_vector],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {}},
            limit=1,
            output_fields=["text"],
        )
        dense_id = dense[0][0]["id"]

        dense_request = AnnSearchRequest(
            [query_vector],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {}},
            limit=1,
        )
        sparse_request = AnnSearchRequest(
            [{0: 1.0}],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=1,
        )
        hybrid = client.hybrid_search(
            COLLECTION,
            [dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=1,
            output_fields=["text"],
        )
        hybrid_id = hybrid[0][0]["id"]

        assert dense_id == "a"
        assert hybrid_id == "a"
        print("PASS: Milvus Lite + Python 3.13 + 4096-dim dense+sparse hybrid")
    finally:
        try:
            client.drop_collection(COLLECTION)
        finally:
            client.close()


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
