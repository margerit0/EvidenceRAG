"""Hermetic contract tests for the optional TiDB Vector adapter."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from zhrag.store import ChunkRecord, TiDBConfig, TiDBStore

SHA256 = "a" * 64


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.rows: list[Mapping[str, object]] = []
        self.rowcount = 0
        self.closed = False

    def execute(self, operation: str, parameters: Sequence[object] = ()) -> int:
        params = tuple(parameters)
        self.connection.executions.append((operation, params))
        if self.connection.fail_on is not None and self.connection.fail_on in operation:
            raise RuntimeError("injected database failure")
        normalized = operation.lstrip().upper()
        if normalized.startswith("SELECT") or normalized.startswith("EXPLAIN SELECT"):
            if not self.connection.results:
                raise AssertionError(f"no fake result queued for SQL: {operation}")
            self.rows = self.connection.results.pop(0)
        self.rowcount = len(self.rows)
        return self.rowcount

    def executemany(
        self,
        operation: str,
        parameters: Sequence[Sequence[object]],
    ) -> int:
        rows = tuple(tuple(row) for row in parameters)
        self.connection.executemany_calls.append((operation, rows))
        if self.connection.fail_on is not None and self.connection.fail_on in operation:
            raise RuntimeError("injected database failure")
        self.rowcount = len(rows)
        return self.rowcount

    def fetchone(self) -> Mapping[str, object] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> Sequence[Mapping[str, object]]:
        return self.rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self) -> None:
        self.results: list[list[Mapping[str, object]]] = []
        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.executemany_calls: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
        self.cursors: list[FakeCursor] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0
        self.fail_on: str | None = None

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed += 1


def make_store(
    connection: FakeConnection,
    *,
    collection_name: str = "chunks_v1",
    read_alias: str | None = None,
    dimensions: int = 3,
) -> TiDBStore:
    def factory(config: TiDBConfig) -> Any:
        assert config.host == "gateway.example"
        return connection

    return TiDBStore(
        TiDBConfig(
            host="gateway.example",
            port=4000,
            user="zhrag",
            password="secret",
            database="test_db",
            collection_name=collection_name,
            dense_dimensions=dimensions,
            read_alias=read_alias,
        ),
        _connection_factory=factory,
    )


def record(doc_id: str = "a") -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        text="synthetic passage",
        dense=(1.0, 0.0, -1.0),
        sparse=((1, 0.5), (4, 2.0)),
        source_key="synthetic/source.md",
        document_sha256=SHA256,
        ordinal=2,
        metadata={"scope": "test", "published": True},
    )


def column_rows(
    fields: Sequence[tuple[str, str, str]],
) -> list[Mapping[str, object]]:
    return [
        {
            "column_name": name,
            "column_type": column_type,
            "is_nullable": nullable,
        }
        for name, column_type, nullable in fields
    ]


def index_rows(
    fields: Sequence[tuple[str, str, int, int]],
) -> list[Mapping[str, object]]:
    return [
        {
            "index_name": name,
            "column_name": column,
            "seq_in_index": position,
            "non_unique": non_unique,
        }
        for name, column, position, non_unique in fields
    ]


def valid_schema_results(dimensions: int = 3) -> list[list[Mapping[str, object]]]:
    return [
        column_rows(
            [
                ("doc_id", "varchar(64)", "NO"),
                ("text", "longtext", "NO"),
                ("dense", f"vector({dimensions})", "NO"),
                ("source_key", "varchar(2048)", "NO"),
                ("document_sha256", "char(64)", "NO"),
                ("ordinal", "bigint unsigned", "NO"),
                ("metadata", "json", "NO"),
            ]
        ),
        column_rows(
            [
                ("doc_id", "varchar(64)", "NO"),
                ("term_index", "bigint unsigned", "NO"),
                ("weight", "double", "NO"),
            ]
        ),
        column_rows(
            [
                ("alias_name", "varchar(64)", "NO"),
                ("collection_name", "varchar(64)", "NO"),
            ]
        ),
        index_rows([("PRIMARY", "doc_id", 1, 0)]),
        index_rows(
            [
                ("PRIMARY", "doc_id", 1, 0),
                ("PRIMARY", "term_index", 2, 0),
                ("term_index", "term_index", 1, 1),
            ]
        ),
        index_rows([("PRIMARY", "alias_name", 1, 0)]),
        [
            {
                "index_name": "dense_cosine",
                "column_name": "dense",
                "index_kind": "HNSW",
            }
        ],
    ]


class TestConfigAndSchema:
    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"collection_name": "chunks;DROP"}, "SQL identifier"),
            ({"database": "db-name"}, "SQL identifier"),
            ({"dense_dimensions": 16_384}, "16383"),
            ({"read_alias": "chunks_v1"}, "must differ"),
            ({"port": True}, "port"),
        ],
    )
    def test_config_rejects_unsafe_or_unsupported_values(
        self,
        changes: Mapping[str, object],
        message: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "host": "gateway.example",
            "user": "user",
            "password": "secret",
            "database": "test_db",
            "collection_name": "chunks_v1",
            "dense_dimensions": 3,
        }
        kwargs.update(changes)
        with pytest.raises(ValueError, match=message):
            TiDBConfig(**kwargs)  # type: ignore[arg-type]

    def test_config_repr_redacts_password(self) -> None:
        config = make_store(FakeConnection()).config
        assert "secret" not in repr(config)

    def test_ensure_collection_uses_fixed_native_and_sparse_schema(self) -> None:
        connection = FakeConnection()
        connection.results.extend(valid_schema_results())
        store = make_store(connection)

        store.ensure_collection()

        ddl = [sql for sql, _params in connection.executions[:3]]
        assert "VECTOR(3) NOT NULL" in ddl[0]
        assert "VECTOR INDEX `dense_cosine`" in ddl[0]
        assert "VEC_COSINE_DISTANCE(`dense`)" in ddl[0]
        assert "PRIMARY KEY (`doc_id`, `term_index`)" in ddl[1]
        assert "CREATE TABLE IF NOT EXISTS `_zhrag_vector_aliases`" in ddl[2]
        assert connection.commits == 1
        assert connection.rollbacks == 0
        assert all(cursor.closed for cursor in connection.cursors)

    def test_incompatible_existing_schema_fails_closed(self) -> None:
        connection = FakeConnection()
        results = valid_schema_results()
        results[0][2] = {
            "column_name": "dense",
            "column_type": "vector(2)",
            "is_nullable": "NO",
        }
        connection.results.extend(results)

        with pytest.raises(RuntimeError, match="incompatible schema"):
            make_store(connection).ensure_collection()

    def test_missing_vector_index_fails_closed(self) -> None:
        connection = FakeConnection()
        results = valid_schema_results()
        results[-1] = []
        connection.results.extend(results)

        with pytest.raises(RuntimeError, match="HNSW"):
            make_store(connection).ensure_collection()

    def test_normal_import_does_not_import_pymysql(self) -> None:
        before = "pymysql" in sys.modules
        make_store(FakeConnection()).close()
        assert ("pymysql" in sys.modules) is before


class TestMutations:
    def test_upsert_writes_complete_rows_and_sparse_postings_atomically(self) -> None:
        connection = FakeConnection()
        store = make_store(connection)

        assert store.upsert([record()]) == 1

        main_sql, main_rows = connection.executemany_calls[0]
        sparse_sql, sparse_rows = connection.executemany_calls[1]
        assert "VEC_FROM_TEXT(%s)" in main_sql
        assert main_rows == (
            (
                "a",
                "synthetic passage",
                "[1.0,0.0,-1.0]",
                "synthetic/source.md",
                SHA256,
                2,
                '{"published":true,"scope":"test"}',
            ),
        )
        assert "`chunks_v1__sparse`" in sparse_sql
        assert sparse_rows == (("a", 1, 0.5), ("a", 4, 2.0))
        delete_sql, delete_params = connection.executions[0]
        assert "DELETE FROM `chunks_v1__sparse`" in delete_sql
        assert delete_params == ("a",)
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_upsert_rolls_back_both_arms_when_postings_write_fails(self) -> None:
        connection = FakeConnection()
        connection.fail_on = "INSERT INTO `chunks_v1__sparse`"

        with pytest.raises(RuntimeError, match="injected"):
            make_store(connection).upsert([record()])

        assert connection.commits == 0
        assert connection.rollbacks == 1

    def test_empty_mutations_are_no_ops_and_input_is_validated(self) -> None:
        connection = FakeConnection()
        store = make_store(connection)
        assert store.upsert([]) == 0
        assert store.delete([]) == 0
        with pytest.raises(ValueError, match="duplicate"):
            store.upsert([record(), record()])
        with pytest.raises(ValueError, match="dimensions"):
            store.upsert([replace(record(), dense=(1.0,))])
        with pytest.raises(ValueError, match="unique"):
            store.delete(["a", "a"])
        assert connection.executions == []
        assert connection.executemany_calls == []

    def test_delete_removes_postings_before_main_rows_in_one_transaction(self) -> None:
        connection = FakeConnection()

        assert make_store(connection).delete(["a", "b"]) == 2

        assert len(connection.executions) == 2
        assert "DELETE FROM `chunks_v1__sparse`" in connection.executions[0][0]
        assert "DELETE FROM `chunks_v1`" in connection.executions[1][0]
        assert connection.executions[0][1] == ("a", "b")
        assert connection.commits == 1


class TestSearchFetchAndAlias:
    def test_dense_and_sparse_search_use_exact_parameterized_contracts(self) -> None:
        connection = FakeConnection()
        connection.results.extend(
            [
                [
                    {"doc_id": "a", "distance": 0.1},
                    {"doc_id": "b", "distance": 0.8},
                ],
                [{"doc_id": "b", "score": 3.5}],
            ]
        )
        store = make_store(connection)

        dense = store.search_dense((1.0, 0.0, 0.0), limit=2)
        sparse = store.search_sparse(((2, 1.0), (5, 0.5)), limit=1)

        assert [(hit.doc_id, hit.score) for hit in dense] == [
            ("a", 0.9),
            ("b", pytest.approx(0.2)),
        ]
        assert [(hit.doc_id, hit.score) for hit in sparse] == [("b", 3.5)]
        dense_sql, dense_params = connection.executions[0]
        sparse_sql, sparse_params = connection.executions[1]
        assert "VEC_COSINE_DISTANCE(`dense`, VEC_FROM_TEXT(%s))" in dense_sql
        assert "ORDER BY VEC_COSINE_DISTANCE(`dense`, VEC_FROM_TEXT(%s)) ASC\n" in dense_sql
        assert "ORDER BY distance ASC" not in dense_sql
        assert "distance ASC," not in dense_sql
        assert dense_params == ("[1.0,0.0,0.0]", "[1.0,0.0,0.0]", 2)
        assert "FROM `chunks_v1__sparse` AS s" in sparse_sql
        assert "LEFT JOIN" not in sparse_sql
        assert sparse_params == (2, 1.0, 5, 0.5, 2, 5, 1)

    def test_dense_index_verification_uses_exact_query_shape(self) -> None:
        connection = FakeConnection()
        connection.results.append(
            [
                {
                    "id": "TableFullScan_7",
                    "operator info": "annIndex:dense_cosine, limit:1",
                }
            ]
        )
        store = make_store(connection)

        store.verify_dense_index((1.0, 0.0, 0.0))

        sql, params = connection.executions[0]
        assert sql.startswith("EXPLAIN SELECT")
        assert "distance ASC," not in sql
        assert params == ("[1.0,0.0,0.0]", "[1.0,0.0,0.0]", 1)

        missing = FakeConnection()
        missing.results.append([{"operator info": "keep order:false"}])
        with pytest.raises(RuntimeError, match="vector index"):
            make_store(missing).verify_dense_index((1.0, 0.0, 0.0))

    def test_empty_sparse_query_short_circuits(self) -> None:
        connection = FakeConnection()
        assert make_store(connection).search_sparse((), limit=10) == ()
        assert connection.executions == []

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_search_scores_fail_closed(self, value: float) -> None:
        connection = FakeConnection()
        connection.results.append([{"doc_id": "a", "distance": value}])
        with pytest.raises(RuntimeError, match="non-finite"):
            make_store(connection).search_dense((1.0, 0.0, 0.0), limit=1)

    def test_fetch_restores_request_order_and_omits_vectors(self) -> None:
        connection = FakeConnection()
        connection.results.append(
            [
                {
                    "doc_id": "b",
                    "text": "second",
                    "source_key": "source/b",
                    "document_sha256": "b" * 64,
                    "ordinal": 1,
                    "metadata": '{"x":2}',
                },
                {
                    "doc_id": "a",
                    "text": "first",
                    "source_key": "source/a",
                    "document_sha256": SHA256,
                    "ordinal": 0,
                    "metadata": {"x": 1},
                },
            ]
        )

        got = make_store(connection).fetch(["a", "missing", "b"])

        assert [passage.doc_id for passage in got] == ["a", "b"]
        assert got[0].metadata == {"x": 1}
        sql, params = connection.executions[0]
        assert "`dense`" not in sql
        assert params == ("a", "missing", "b")

    def test_count_alias_switch_and_close_are_explicit(self) -> None:
        connection = FakeConnection()
        connection.results.extend([[{"row_count": 7}], [], [{"collection_name": "chunks_v1"}]])
        store = make_store(connection)

        assert store.count() == 7
        assert store.alias_target("active") is None
        store.activate_alias("active")
        assert store.alias_target("active") == "chunks_v1"
        store.close()
        store.close()

        alias_write = next(sql for sql, _params in connection.executions if "INSERT INTO" in sql)
        assert "ON DUPLICATE KEY UPDATE" in alias_write
        assert connection.commits == 1
        assert connection.closed == 1

    def test_read_alias_resolves_to_physical_tables_and_missing_alias_fails(self) -> None:
        connection = FakeConnection()
        connection.results.extend(
            [
                [{"collection_name": "chunks_v2"}],
                [{"doc_id": "a", "distance": 0.25}],
            ]
        )
        store = make_store(connection, collection_name="reader", read_alias="active")

        assert store.search_dense((1.0, 0.0, 0.0), limit=1)[0].score == 0.75
        assert "FROM `chunks_v2`" in connection.executions[1][0]
        with pytest.raises(RuntimeError, match="read-alias"):
            store.upsert([record()])
        with pytest.raises(RuntimeError, match="read-alias"):
            store.delete(["a"])
        with pytest.raises(RuntimeError, match="read-alias"):
            store.activate_alias("other")

        missing = FakeConnection()
        missing.results.append([])
        with pytest.raises(RuntimeError, match="has not been published"):
            make_store(missing, collection_name="reader", read_alias="active").count()

    def test_alias_target_and_search_reject_malformed_vendor_rows(self) -> None:
        connection = FakeConnection()
        connection.results.append([{"collection_name": "bad-name"}])
        with pytest.raises(ValueError, match="SQL identifier"):
            make_store(connection).alias_target("active")

        duplicate = FakeConnection()
        duplicate.results.append(
            [
                {"doc_id": "a", "distance": 0.1},
                {"doc_id": "a", "distance": 0.2},
            ]
        )
        with pytest.raises(RuntimeError, match="duplicate"):
            make_store(duplicate).search_dense((1.0, 0.0, 0.0), limit=2)
