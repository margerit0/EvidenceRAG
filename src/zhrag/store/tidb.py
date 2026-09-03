"""TiDB Vector adapter with a lazy MySQL-protocol dependency.

Dense retrieval uses TiDB's native ``VECTOR`` type and HNSW index. The sparse
arm remains the repository's client-computed char-bigram BM25 inner product;
it is stored in a companion postings table rather than being replaced by
TiDB Full-Text Search, whose tokenizer and deployment boundary are different.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from types import ModuleType
from typing import Any, Protocol, cast

from zhrag.lexical.sparse import SparseVector
from zhrag.store.base import ArmHit, ChunkRecord, DenseVector, Passage

__all__ = ["TiDBConfig", "TiDBStore", "TiDBUnavailableError"]

_SPARSE_SUFFIX = "__sparse"
_ALIAS_TABLE = "_zhrag_vector_aliases"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_COLLECTION_LENGTH = 64 - len(_SPARSE_SUFFIX)


class TiDBUnavailableError(RuntimeError):
    """Raised when the optional MySQL-protocol dependency is not installed."""


class _Cursor(Protocol):
    rowcount: int

    def execute(self, operation: str, parameters: Sequence[object] = ()) -> int: ...

    def executemany(
        self,
        operation: str,
        parameters: Sequence[Sequence[object]],
    ) -> int: ...

    def fetchone(self) -> Mapping[str, object] | None: ...

    def fetchall(self) -> Sequence[Mapping[str, object]]: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


type _ConnectionFactory = Callable[["TiDBConfig"], _Connection]


@dataclass(frozen=True, slots=True)
class TiDBConfig:
    """Connection settings and one physical TiDB collection table."""

    host: str
    user: str
    password: str = field(repr=False)
    database: str
    collection_name: str
    dense_dimensions: int
    port: int = 4000
    connect_timeout: float = 10.0
    verify_tls: bool = True
    read_alias: str | None = None

    def __post_init__(self) -> None:
        for name in ("host", "user", "database"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        _validated_collection_name(self.collection_name)
        _validated_identifier(self.database, "database", maximum=64)
        if self.read_alias is not None:
            _validated_identifier(self.read_alias, "read_alias", maximum=64)
            if self.read_alias == self.collection_name:
                raise ValueError("read_alias must differ from the physical collection name")
        if not 1 <= self.dense_dimensions <= 16_383:
            raise ValueError("dense_dimensions must be between 1 and 16383")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65_535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        if not math.isfinite(self.connect_timeout) or self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be finite and positive")


def _validated_identifier(value: str, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be an ASCII SQL identifier of at most {maximum} characters")
    return value


def _validated_collection_name(value: str) -> str:
    return _validated_identifier(value, "collection_name", maximum=_MAX_COLLECTION_LENGTH)


def _quoted(value: str) -> str:
    return f"`{value}`"


def _load_pymysql() -> ModuleType:
    try:
        return importlib.import_module("pymysql")
    except ImportError as exc:
        raise TiDBUnavailableError(
            "TiDB support is optional; install zhrag[tidb] or pymysql>=1.1.2."
        ) from exc


def _default_connection_factory(config: TiDBConfig) -> _Connection:
    module = _load_pymysql()
    kwargs: dict[str, object] = {
        "host": config.host,
        "port": config.port,
        "user": config.user,
        "password": config.password,
        "database": config.database,
        "charset": "utf8mb4",
        "autocommit": False,
        "connect_timeout": math.ceil(config.connect_timeout),
        "cursorclass": module.cursors.DictCursor,
    }
    if config.verify_tls:
        kwargs.update(ssl_verify_cert=True, ssl_verify_identity=True)
    factory = cast(Callable[..., object], module.connect)
    return cast(_Connection, factory(**kwargs))


def _require_positive_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")


def _validated_dense(vector: DenseVector, dimensions: int) -> tuple[float, ...]:
    if len(vector) != dimensions:
        raise ValueError(f"dense vector has {len(vector)} dimensions, expected {dimensions}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("dense vector must contain only finite values")
    return tuple(vector)


def _validated_sparse(vector: SparseVector) -> tuple[tuple[int, float], ...]:
    values: list[tuple[int, float]] = []
    previous = -1
    for index, value in vector:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index <= previous:
            raise ValueError("sparse indices must be unique, non-negative, and sorted")
        if not math.isfinite(value) or value == 0:
            raise ValueError("sparse values must be finite and non-zero")
        values.append((index, value))
        previous = index
    return tuple(values)


def _vector_literal(vector: Sequence[float]) -> str:
    return json.dumps(vector, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _placeholders(size: int) -> str:
    return ", ".join("%s" for _ in range(size))


def _row_string(row: Mapping[str, object], name: str, context: str) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise RuntimeError(f"TiDB {context} row has no string {name}")
    return value


def _row_integer(row: Mapping[str, object], name: str, context: str) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"TiDB {context} row has no integer {name}")
    return value


def _row_float(row: Mapping[str, object], name: str, context: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise RuntimeError(f"TiDB {context} row has no numeric {name}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"TiDB {context} row has a non-finite {name}")
    return result


def _metadata(value: object) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError("TiDB fetch row has malformed metadata JSON") from exc
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError("TiDB fetch row has no metadata object")
    return cast(Mapping[str, Any], value)


def _close_cursor(cursor: _Cursor) -> None:
    try:
        cursor.close()
    except Exception:
        return


@dataclass(slots=True)
class TiDBStore:
    """A fail-closed TiDB implementation of the vendor-neutral store contract."""

    config: TiDBConfig
    _connection_factory: _ConnectionFactory = _default_connection_factory
    _connection: _Connection | None = None
    _resolved_read_target: str | None = None

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    @property
    def dense_dimensions(self) -> int:
        return self.config.dense_dimensions

    def _get_connection(self) -> _Connection:
        if self._connection is None:
            self._connection = self._connection_factory(self.config)
        return self._connection

    def _execute(self, sql: str, parameters: Sequence[object] = ()) -> list[Mapping[str, object]]:
        cursor = self._get_connection().cursor()
        try:
            cursor.execute(sql, parameters)
            return list(cursor.fetchall())
        finally:
            _close_cursor(cursor)

    def _transaction(self, action: Callable[[_Cursor], None]) -> None:
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            action(cursor)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            _close_cursor(cursor)

    @property
    def _physical_table(self) -> str:
        return _quoted(self.collection_name)

    @property
    def _physical_sparse_table(self) -> str:
        return _quoted(f"{self.collection_name}{_SPARSE_SUFFIX}")

    def ensure_collection(self) -> None:
        self._require_physical_writer()
        main = self._physical_table
        sparse = self._physical_sparse_table
        aliases = _quoted(_ALIAS_TABLE)
        dimensions = self.dense_dimensions

        def create(cursor: _Cursor) -> None:
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {main} (
                    `doc_id` VARCHAR(64) NOT NULL PRIMARY KEY,
                    `text` LONGTEXT NOT NULL,
                    `dense` VECTOR({dimensions}) NOT NULL,
                    `source_key` VARCHAR(2048) NOT NULL,
                    `document_sha256` CHAR(64) NOT NULL,
                    `ordinal` BIGINT UNSIGNED NOT NULL,
                    `metadata` JSON NOT NULL,
                    VECTOR INDEX `dense_cosine` ((VEC_COSINE_DISTANCE(`dense`))) USING HNSW
                )"""
            )
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {sparse} (
                    `doc_id` VARCHAR(64) NOT NULL,
                    `term_index` BIGINT UNSIGNED NOT NULL,
                    `weight` DOUBLE NOT NULL,
                    PRIMARY KEY (`doc_id`, `term_index`),
                    INDEX `term_index` (`term_index`)
                )"""
            )
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {aliases} (
                    `alias_name` VARCHAR(64) NOT NULL PRIMARY KEY,
                    `collection_name` VARCHAR(64) NOT NULL
                )"""
            )

        self._transaction(create)
        self._verify_schema()

    def _verify_schema(self) -> None:
        expected = {
            self.collection_name: (
                ("doc_id", "varchar(64)", "NO"),
                ("text", "longtext", "NO"),
                ("dense", f"vector({self.dense_dimensions})", "NO"),
                ("source_key", "varchar(2048)", "NO"),
                ("document_sha256", "char(64)", "NO"),
                ("ordinal", "bigint unsigned", "NO"),
                ("metadata", "json", "NO"),
            ),
            f"{self.collection_name}{_SPARSE_SUFFIX}": (
                ("doc_id", "varchar(64)", "NO"),
                ("term_index", "bigint unsigned", "NO"),
                ("weight", "double", "NO"),
            ),
            _ALIAS_TABLE: (
                ("alias_name", "varchar(64)", "NO"),
                ("collection_name", "varchar(64)", "NO"),
            ),
        }
        for table_name, expected_columns in expected.items():
            rows = self._execute(
                """SELECT COLUMN_NAME AS column_name, COLUMN_TYPE AS column_type,
                          IS_NULLABLE AS is_nullable
                   FROM INFORMATION_SCHEMA.COLUMNS
                   WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                   ORDER BY ORDINAL_POSITION""",
                (self.config.database, table_name),
            )
            actual = tuple(
                (
                    _row_string(row, "column_name", "schema"),
                    _row_string(row, "column_type", "schema").lower(),
                    _row_string(row, "is_nullable", "schema"),
                )
                for row in rows
            )
            if actual != expected_columns:
                raise RuntimeError(f"TiDB table {table_name!r} has an incompatible schema")

        required_indexes = {
            self.collection_name: {("PRIMARY", "doc_id", 1, 0)},
            f"{self.collection_name}{_SPARSE_SUFFIX}": {
                ("PRIMARY", "doc_id", 1, 0),
                ("PRIMARY", "term_index", 2, 0),
                ("term_index", "term_index", 1, 1),
            },
            _ALIAS_TABLE: {("PRIMARY", "alias_name", 1, 0)},
        }
        for table_name, required in required_indexes.items():
            index_rows = self._execute(
                """SELECT INDEX_NAME AS index_name, COLUMN_NAME AS column_name,
                          SEQ_IN_INDEX AS seq_in_index, NON_UNIQUE AS non_unique
                   FROM INFORMATION_SCHEMA.STATISTICS
                   WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s""",
                (self.config.database, table_name),
            )
            signatures = {
                (
                    _row_string(row, "index_name", "index"),
                    _row_string(row, "column_name", "index"),
                    _row_integer(row, "seq_in_index", "index"),
                    _row_integer(row, "non_unique", "index"),
                )
                for row in index_rows
            }
            if not required <= signatures:
                raise RuntimeError(f"TiDB table {table_name!r} has incompatible indexes")

        vector_rows = self._execute(
            """SELECT INDEX_NAME AS index_name, COLUMN_NAME AS column_name,
                      INDEX_KIND AS index_kind
               FROM INFORMATION_SCHEMA.TIFLASH_INDEXES
               WHERE TIDB_DATABASE = %s AND TIDB_TABLE = %s""",
            (self.config.database, self.collection_name),
        )
        vector_signatures = {
            (
                _row_string(row, "index_name", "vector index"),
                _row_string(row, "column_name", "vector index"),
                _row_string(row, "index_kind", "vector index").upper(),
            )
            for row in vector_rows
        }
        if ("dense_cosine", "dense", "HNSW") not in vector_signatures:
            raise RuntimeError("TiDB collection has no compatible dense HNSW index")

    def _resolved_collection_name(self) -> str:
        alias = self.config.read_alias
        if alias is None:
            return self.collection_name
        if self._resolved_read_target is not None:
            return self._resolved_read_target
        rows = self._execute(
            f"SELECT `collection_name` FROM `{_ALIAS_TABLE}` WHERE `alias_name` = %s",
            (alias,),
        )
        if len(rows) > 1:
            raise RuntimeError("TiDB alias registry returned duplicate rows")
        if not rows:
            raise RuntimeError(f"TiDB read alias {alias!r} has not been published")
        target = _validated_collection_name(
            _row_string(rows[0], "collection_name", "alias registry")
        )
        self._resolved_read_target = target
        return target

    def _require_physical_writer(self) -> None:
        if self.config.read_alias is not None:
            raise RuntimeError("mutations are not allowed on a read-alias store")

    def upsert(self, records: Sequence[ChunkRecord]) -> int:
        self._require_physical_writer()

        rows = list(records)
        if not rows:
            return 0
        if len({record.doc_id for record in rows}) != len(rows):
            raise ValueError("upsert batch contains duplicate doc_ids")
        serialized: list[tuple[object, ...]] = []
        postings: list[tuple[object, ...]] = []
        for record in rows:
            if len(record.doc_id) > 64:
                raise ValueError("doc_id exceeds the TiDB schema character limit")
            if len(record.source_key) > 2048:
                raise ValueError("source_key exceeds the TiDB schema character limit")
            dense = _validated_dense(record.dense, self.dense_dimensions)
            sparse = _validated_sparse(record.sparse)
            serialized.append(
                (
                    record.doc_id,
                    record.text,
                    _vector_literal(dense),
                    record.source_key,
                    record.document_sha256,
                    record.ordinal,
                    json.dumps(
                        dict(record.metadata),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
            )
            postings.extend((record.doc_id, index, weight) for index, weight in sparse)
        ids = [record.doc_id for record in rows]
        main = self._physical_table
        sparse_table = self._physical_sparse_table

        def write(cursor: _Cursor) -> None:
            cursor.executemany(
                f"""INSERT INTO {main}
                    (`doc_id`, `text`, `dense`, `source_key`,
                     `document_sha256`, `ordinal`, `metadata`)
                    VALUES (%s, %s, VEC_FROM_TEXT(%s), %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      `text` = VALUES(`text`),
                      `dense` = VALUES(`dense`),
                      `source_key` = VALUES(`source_key`),
                      `document_sha256` = VALUES(`document_sha256`),
                      `ordinal` = VALUES(`ordinal`),
                      `metadata` = VALUES(`metadata`)""",
                serialized,
            )
            cursor.execute(
                f"DELETE FROM {sparse_table} WHERE `doc_id` IN ({_placeholders(len(ids))})",
                ids,
            )
            if postings:
                cursor.executemany(
                    f"""INSERT INTO {sparse_table}
                        (`doc_id`, `term_index`, `weight`)
                        VALUES (%s, %s, %s)""",
                    postings,
                )

        self._transaction(write)
        return len(rows)

    def delete(self, doc_ids: Sequence[str]) -> int:
        self._require_physical_writer()
        ids = list(doc_ids)

        if not ids:
            return 0
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("delete ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("delete ids must be unique")
        main = self._physical_table
        sparse = self._physical_sparse_table

        def remove(cursor: _Cursor) -> None:
            placeholders = _placeholders(len(ids))
            cursor.execute(f"DELETE FROM {sparse} WHERE `doc_id` IN ({placeholders})", ids)
            cursor.execute(f"DELETE FROM {main} WHERE `doc_id` IN ({placeholders})", ids)

        self._transaction(remove)
        return len(ids)

    @staticmethod
    def _dense_statement(table: str, *, explain: bool) -> str:
        prefix = "EXPLAIN " if explain else ""
        return f"""{prefix}SELECT `doc_id`,
                       VEC_COSINE_DISTANCE(`dense`, VEC_FROM_TEXT(%s)) AS distance
                FROM {table}
                ORDER BY VEC_COSINE_DISTANCE(`dense`, VEC_FROM_TEXT(%s)) ASC
                LIMIT %s"""

    def search_dense(self, vector: DenseVector, *, limit: int) -> Sequence[ArmHit]:
        _require_positive_limit(limit)
        dense = _validated_dense(vector, self.dense_dimensions)
        table = _quoted(self._resolved_collection_name())
        literal = _vector_literal(dense)
        rows = self._execute(
            self._dense_statement(table, explain=False),
            (literal, literal, limit),
        )
        return self._search_hits(rows, limit=limit, distance=True)

    def verify_dense_index(self, vector: DenseVector, *, limit: int = 1) -> None:
        """Fail unless TiDB plans the exact dense query with the HNSW index."""
        _require_positive_limit(limit)
        dense = _validated_dense(vector, self.dense_dimensions)
        table = _quoted(self._resolved_collection_name())
        literal = _vector_literal(dense)
        rows = self._execute(
            self._dense_statement(table, explain=True),
            (literal, literal, limit),
        )
        plan = "\n".join(str(value) for row in rows for value in row.values())
        if "annIndex:" not in plan:
            raise RuntimeError("TiDB did not plan the dense query with a vector index")

    def search_sparse(self, vector: SparseVector, *, limit: int) -> Sequence[ArmHit]:
        _require_positive_limit(limit)
        values = _validated_sparse(vector)
        if not values:
            return ()
        collection = self._resolved_collection_name()
        sparse = _quoted(f"{collection}{_SPARSE_SUFFIX}")
        cases = " ".join("WHEN %s THEN %s" for _ in values)
        indices = [index for index, _value in values]
        case_params = [item for pair in values for item in pair]
        rows = self._execute(
            f"""SELECT s.`doc_id`, SUM(s.`weight` * CASE s.`term_index`
                           {cases} ELSE 0 END) AS score
                FROM {sparse} AS s
                WHERE s.`term_index` IN ({_placeholders(len(indices))})
                GROUP BY s.`doc_id`
                ORDER BY score DESC, s.`doc_id` ASC
                LIMIT %s""",
            (*case_params, *indices, limit),
        )
        return self._search_hits(rows, limit=limit, distance=False)

    @staticmethod
    def _search_hits(
        rows: Sequence[Mapping[str, object]],
        *,
        limit: int,
        distance: bool,
    ) -> tuple[ArmHit, ...]:
        if len(rows) > limit:
            raise RuntimeError("TiDB search returned more hits than requested")
        hits: list[ArmHit] = []
        seen: set[str] = set()
        for row in rows:
            doc_id = _row_string(row, "doc_id", "search")
            if doc_id in seen:
                raise RuntimeError("TiDB search response contains duplicate doc_ids")
            seen.add(doc_id)
            value = _row_float(row, "distance" if distance else "score", "search")
            hits.append(ArmHit(doc_id=doc_id, score=1.0 - value if distance else value))
        return tuple(hits)

    def fetch(self, doc_ids: Sequence[str]) -> Sequence[Passage]:
        ids = list(doc_ids)
        if not ids:
            return ()
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("fetch ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("fetch ids must be unique")
        table = _quoted(self._resolved_collection_name())
        fetched = self._execute(
            f"""SELECT `doc_id`, `text`, `source_key`,
                       `document_sha256`, `ordinal`, `metadata`
                FROM {table}
                WHERE `doc_id` IN ({_placeholders(len(ids))})""",
            ids,
        )
        rows: dict[str, Mapping[str, object]] = {}
        for row in fetched:
            doc_id = _row_string(row, "doc_id", "fetch")
            if doc_id not in ids:
                raise RuntimeError("TiDB fetch response contains an unrequested doc_id")
            if doc_id in rows:
                raise RuntimeError("TiDB fetch response contains duplicate doc_ids")
            rows[doc_id] = row
        return tuple(
            Passage(
                doc_id=doc_id,
                text=_row_string(rows[doc_id], "text", "fetch"),
                source_key=_row_string(rows[doc_id], "source_key", "fetch"),
                document_sha256=_row_string(rows[doc_id], "document_sha256", "fetch"),
                ordinal=_row_integer(rows[doc_id], "ordinal", "fetch"),
                metadata=_metadata(rows[doc_id].get("metadata")),
            )
            for doc_id in ids
            if doc_id in rows
        )

    def count(self) -> int:
        table = _quoted(self._resolved_collection_name())
        rows = self._execute(f"SELECT COUNT(*) AS row_count FROM {table}")
        if len(rows) != 1:
            raise RuntimeError("TiDB count query did not return exactly one row")
        value = _row_integer(rows[0], "row_count", "count")
        if value < 0:
            raise RuntimeError("TiDB row count must be non-negative")
        return value

    def alias_target(self, alias: str) -> str | None:
        _validated_identifier(alias, "alias", maximum=64)
        rows = self._execute(
            f"SELECT `collection_name` FROM `{_ALIAS_TABLE}` WHERE `alias_name` = %s",
            (alias,),
        )
        if len(rows) > 1:
            raise RuntimeError("TiDB alias registry returned duplicate rows")
        if not rows:
            return None
        return _validated_collection_name(_row_string(rows[0], "collection_name", "alias"))

    def activate_alias(self, alias: str) -> None:
        self._require_physical_writer()
        _validated_identifier(alias, "alias", maximum=64)
        if alias == self.collection_name:
            raise ValueError("alias must differ from the physical collection name")

        def activate(cursor: _Cursor) -> None:
            cursor.execute(
                f"""INSERT INTO `{_ALIAS_TABLE}` (`alias_name`, `collection_name`)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE `collection_name` = VALUES(`collection_name`)""",
                (alias, self.collection_name),
            )

        self._transaction(activate)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._resolved_read_target = None

    def __enter__(self) -> TiDBStore:
        self._get_connection()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
