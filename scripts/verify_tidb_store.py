"""Opt-in synthetic live verification for :class:`zhrag.store.TiDBStore`.

This script requires a disposable TiDB Cloud Starter database. It creates two
physical test collections and one alias-registry row, uses only synthetic text,
then removes all three tables. Nothing from the project corpus is read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhrag.providers.embedding import load_env
from zhrag.store import ChunkRecord, TiDBConfig, TiDBStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = ROOT / ".env"
SHA256 = "a" * 64
REQUIRED_ENV = (
    "TIDB_HOST",
    "TIDB_USER",
    "TIDB_PASSWORD",
    "TIDB_DATABASE",
)


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--suffix", default="manual")
    parser.add_argument("--allow-live-tidb", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_live_tidb:
        parser.error("live database mutation requires --allow-live-tidb")
    if not args.suffix.replace("_", "").isalnum() or not args.suffix.isascii():
        parser.error("--suffix must contain only ASCII letters, digits, or underscores")
    return args


def _config(
    env: dict[str, str],
    *,
    collection_name: str,
    read_alias: str | None = None,
) -> TiDBConfig:
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        raise ValueError(f".env is missing or empty for: {', '.join(missing)}")
    port = int(env.get("TIDB_PORT", "4000"))
    return TiDBConfig(
        host=env["TIDB_HOST"],
        port=port,
        user=env["TIDB_USER"],
        password=env["TIDB_PASSWORD"],
        database=env["TIDB_DATABASE"],
        collection_name=collection_name,
        dense_dimensions=3,
        read_alias=read_alias,
    )


def _record(
    doc_id: str,
    dense: tuple[float, ...],
    sparse: tuple[tuple[int, float], ...],
) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        text=f"synthetic TiDB adapter row {doc_id}",
        dense=dense,
        sparse=sparse,
        source_key=f"synthetic/{doc_id}",
        document_sha256=SHA256,
        ordinal=0,
        metadata={"fixture": True},
    )


def _drop_test_artifacts(config: TiDBConfig, alias: str, tables: tuple[str, ...]) -> None:
    store = TiDBStore(config)
    connection = store._get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "DELETE FROM `_zhrag_vector_aliases` WHERE `alias_name` = %s",
            (alias,),
        )
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS `{table}__sparse`")
            cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        store.close()


def _run(env: dict[str, str], suffix: str) -> None:
    first_name = f"zhrag_verify_a_{suffix}"
    second_name = f"zhrag_verify_b_{suffix}"
    alias = f"zhrag_verify_active_{suffix}"
    tables = (first_name, second_name)
    cleanup_config = _config(env, collection_name=first_name)
    try:
        first = TiDBStore(_config(env, collection_name=first_name))
        second = TiDBStore(_config(env, collection_name=second_name))
        try:
            for store in (first, second):
                store.ensure_collection()
            rows = [
                _record("a", (1.0, 0.0, 0.0), ((0, 2.0),)),
                _record("b", (0.8, 0.2, 0.0), ((1, 3.0),)),
                _record("c", (0.0, 1.0, 0.0), ((0, 1.0),)),
            ]
            assert first.upsert(rows) == 3
            assert first.count() == 3
            first.verify_dense_index((1.0, 0.0, 0.0), limit=3)
            assert [hit.doc_id for hit in first.search_dense((1.0, 0.0, 0.0), limit=3)] == [
                "a",
                "b",
                "c",
            ]
            assert [hit.doc_id for hit in first.search_sparse(((0, 1.0),), limit=2)] == [
                "a",
                "c",
            ]
            assert [row.doc_id for row in first.fetch(["c", "missing", "a"])] == ["c", "a"]
            assert first.alias_target(alias) is None
            first.activate_alias(alias)
            assert first.alias_target(alias) == first_name
            assert second.upsert([rows[2]]) == 1
            second.activate_alias(alias)
            assert second.alias_target(alias) == second_name
        finally:
            first.close()
            second.close()

        reader = TiDBStore(_config(env, collection_name=second_name, read_alias=alias))
        try:
            assert reader.count() == 1
            assert [row.doc_id for row in reader.fetch(["c"])] == ["c"]
        finally:
            reader.close()

        reopened = TiDBStore(_config(env, collection_name=second_name))
        try:
            assert reopened.delete(["c"]) == 1
            assert reopened.count() == 0
        finally:
            reopened.close()
    finally:
        _drop_test_artifacts(cleanup_config, alias, tables)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    try:
        args = _parse_args(argv)
        _run(load_env(args.env), args.suffix)
    except (AssertionError, OSError, RuntimeError, ValueError) as exc:
        print(f"! TiDBStore live verification failed ({type(exc).__name__}): {exc}")
        return 1
    print("PASS: TiDBStore HNSW plan, arms, rows, alias publication, and reconnect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
