"""Milvus adapter with no import-time dependency on :mod:`pymilvus`.

The adapter stores client-computed BM25 contributions in a sparse field and
searches it with inner product. It deliberately does not attach a Milvus BM25
function: doing so would tokenize again and estimate a second set of IDF values.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

from zhrag.lexical.sparse import SparseVector
from zhrag.store.base import ArmHit, ChunkRecord, DenseVector, Passage

__all__ = ["MilvusConfig", "MilvusStore", "MilvusUnavailableError"]

_DOC_ID = "doc_id"
_TEXT = "text"
_DENSE = "dense"
_SPARSE = "sparse"
_SOURCE_KEY = "source_key"
_DOCUMENT_SHA256 = "document_sha256"
_ORDINAL = "ordinal"
_METADATA = "metadata"
_DENSE_INDEX = "dense"
_SPARSE_INDEX = "sparse"
_DOC_ID_MAX_LENGTH = 64
_TEXT_MAX_LENGTH = 65_535
_SOURCE_KEY_MAX_LENGTH = 2_048
_SHA256_LENGTH = 64


class MilvusUnavailableError(RuntimeError):
    """Raised when the optional Milvus dependency is not installed."""


class _VendorSchema(Protocol):
    def add_field(self, **kwargs: object) -> None: ...


class _VendorIndexes(Protocol):
    def add_index(self, **kwargs: object) -> None: ...


class _VendorClient(Protocol):
    def has_collection(self, collection_name: str) -> bool: ...

    def describe_collection(self, collection_name: str) -> Mapping[str, object]: ...

    def create_schema(self, **kwargs: object) -> _VendorSchema: ...

    def prepare_index_params(self) -> _VendorIndexes: ...

    def create_collection(self, **kwargs: object) -> object: ...

    def list_indexes(self, collection_name: str) -> Sequence[str]: ...

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
    ) -> Mapping[str, object]: ...

    def load_collection(self, collection_name: str) -> object: ...

    def upsert(
        self,
        collection_name: str,
        data: list[dict[str, object]],
        **kwargs: object,
    ) -> object: ...

    def delete(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> object: ...

    def search(self, collection_name: str, **kwargs: object) -> object: ...

    def get(
        self,
        collection_name: str,
        *,
        ids: list[str],
        output_fields: list[str],
    ) -> object: ...

    def get_collection_stats(self, collection_name: str) -> Mapping[str, object]: ...

    def describe_alias(self, alias: str) -> Mapping[str, object]: ...

    def create_alias(self, collection_name: str, alias: str) -> object: ...

    def alter_alias(self, collection_name: str, alias: str) -> object: ...

    def close(self) -> None: ...


type _ClientFactory = Callable[[str, str | None, str, float | None], _VendorClient]
type _DataTypeResolver = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class MilvusConfig:
    """Connection and physical collection settings."""

    uri: str
    collection_name: str
    dense_dimensions: int
    token: str | None = None
    database: str = "default"
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("uri must be non-empty")
        if not self.collection_name:
            raise ValueError("collection_name must be non-empty")
        if self.dense_dimensions < 1:
            raise ValueError("dense_dimensions must be positive")
        if not self.database:
            raise ValueError("database must be non-empty")
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be finite and positive")


def _load_pymilvus() -> ModuleType:
    try:
        return importlib.import_module("pymilvus")
    except ImportError as exc:
        raise MilvusUnavailableError(
            "Milvus support is optional; install zhrag[milvus] or pymilvus==3.0.1. "
            "On Windows Milvus Lite also requires an explicit milvus-lite==3.2.0 install."
        ) from exc


def _default_client_factory(
    uri: str,
    token: str | None,
    database: str,
    timeout: float | None,
) -> _VendorClient:
    module = _load_pymilvus()
    kwargs: dict[str, object] = {"uri": uri, "db_name": database}
    if token is not None:
        kwargs["token"] = token
    if timeout is not None:
        kwargs["timeout"] = timeout
    client_type = cast(type[object], module.MilvusClient)
    return cast(_VendorClient, client_type(**kwargs))


def _datatype(name: str) -> object:
    data_type = _load_pymilvus().DataType
    return getattr(data_type, name)


def _require_positive_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")


def _sparse_mapping(vector: SparseVector) -> dict[int, float]:
    values: dict[int, float] = {}
    previous = -1
    for index, value in vector:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index <= previous:
            raise ValueError("sparse indices must be unique, non-negative, and sorted")
        if not math.isfinite(value) or value == 0:
            raise ValueError("sparse values must be finite and non-zero")
        values[index] = value
        previous = index
    return values


def _validated_dense(vector: DenseVector, dimensions: int) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError(f"dense vector has {len(vector)} dimensions, expected {dimensions}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("dense vector must contain only finite values")
    return list(vector)


def _strict_count(result: object, key: str, expected: int) -> int:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"Milvus {key} response is not a mapping")
    raw = result.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(f"Milvus {key} response has no integer {key}")
    if raw != expected:
        raise RuntimeError(f"Milvus acknowledged {raw} rows, expected {expected}")
    return raw


def _field_map(description: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_fields = description.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise RuntimeError("Milvus collection description has no field list")
    fields: dict[str, Mapping[str, object]] = {}
    for raw in raw_fields:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise RuntimeError("Milvus collection description contains a malformed field")
        fields[raw["name"]] = raw
    return fields


def _type_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _field_param(field: Mapping[str, object], name: str) -> int | None:
    raw_params = field.get("params")
    if not isinstance(raw_params, Mapping):
        return None
    value = raw_params.get(name)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _index_signature(description: Mapping[str, object]) -> tuple[str, str, str]:
    field = description.get("field_name")
    index_type = description.get("index_type")
    metric = description.get("metric_type")
    if not all(isinstance(value, str) for value in (field, index_type, metric)):
        raise RuntimeError("Milvus index description is malformed")
    return cast(tuple[str, str, str], (field, index_type, metric))


def _verify_field(
    fields: Mapping[str, Mapping[str, object]],
    name: str,
    expected_type: str,
) -> None:
    if _type_name(fields[name].get("type")) != expected_type:
        raise RuntimeError(f"Milvus field {name!r} must have type {expected_type}")


def _verify_field_param(
    fields: Mapping[str, Mapping[str, object]],
    name: str,
    param: str,
    expected: int,
) -> None:
    if _field_param(fields[name], param) != expected:
        label = "dimension" if param == "dim" else param
        raise RuntimeError(f"Milvus {name} field has the wrong {label}")


def _row_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Milvus {context} row is not a mapping")
    return value


def _row_string(row: Mapping[str, object], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"Milvus {context} row has no string {field}")
    return value


def _row_int(row: Mapping[str, object], field: str, *, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Milvus {context} row has no integer {field}")
    return value


def _query_rows(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise RuntimeError("Milvus get response is not a row sequence")
    return [_row_mapping(value, context="get") for value in raw]


def _search_rows(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 1:
        raise RuntimeError("Milvus search response must contain exactly one query result")
    first = raw[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        raise RuntimeError("Milvus search hits are not a sequence")
    return [_row_mapping(value, context="search") for value in first]


def _metadata(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get(_METADATA)
    if not isinstance(value, Mapping):
        raise RuntimeError("Milvus get row has no metadata object")
    return value


@dataclass(slots=True)
class MilvusStore:
    """A fail-closed adapter for complete rows and separate retrieval arms."""

    config: MilvusConfig
    _client_factory: _ClientFactory = _default_client_factory
    _data_type: _DataTypeResolver = _datatype
    _client: _VendorClient | None = None

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    @property
    def dense_dimensions(self) -> int:
        return self.config.dense_dimensions

    def _get_client(self) -> _VendorClient:
        if self._client is None:
            self._client = self._client_factory(
                self.config.uri,
                self.config.token,
                self.config.database,
                self.config.timeout,
            )
        return self._client

    def ensure_collection(self) -> None:
        client = self._get_client()
        if client.has_collection(self.collection_name):
            self._verify_collection(client)
            client.load_collection(self.collection_name)
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=_DOC_ID,
            datatype=self._data_type("VARCHAR"),
            is_primary=True,
            max_length=_DOC_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name=_TEXT,
            datatype=self._data_type("VARCHAR"),
            max_length=_TEXT_MAX_LENGTH,
        )
        schema.add_field(
            field_name=_DENSE,
            datatype=self._data_type("FLOAT_VECTOR"),
            dim=self.dense_dimensions,
        )
        schema.add_field(field_name=_SPARSE, datatype=self._data_type("SPARSE_FLOAT_VECTOR"))
        schema.add_field(
            field_name=_SOURCE_KEY,
            datatype=self._data_type("VARCHAR"),
            max_length=_SOURCE_KEY_MAX_LENGTH,
        )
        schema.add_field(
            field_name=_DOCUMENT_SHA256,
            datatype=self._data_type("VARCHAR"),
            max_length=_SHA256_LENGTH,
        )
        schema.add_field(field_name=_ORDINAL, datatype=self._data_type("INT64"))
        schema.add_field(field_name=_METADATA, datatype=self._data_type("JSON"))

        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name=_DENSE,
            index_name=_DENSE_INDEX,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        indexes.add_index(
            field_name=_SPARSE,
            index_name=_SPARSE_INDEX,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )

    def _verify_collection(self, client: _VendorClient) -> None:
        description = client.describe_collection(self.collection_name)
        if description.get("auto_id") is not False:
            raise RuntimeError("Milvus collection must disable auto_id")
        if description.get("enable_dynamic_field") is not False:
            raise RuntimeError("Milvus collection must disable dynamic fields")
        fields = _field_map(description)
        expected_types = {
            _DOC_ID: "VARCHAR",
            _TEXT: "VARCHAR",
            _DENSE: "FLOAT_VECTOR",
            _SPARSE: "SPARSE_FLOAT_VECTOR",
            _SOURCE_KEY: "VARCHAR",
            _DOCUMENT_SHA256: "VARCHAR",
            _ORDINAL: "INT64",
            _METADATA: "JSON",
        }
        if set(fields) != set(expected_types):
            raise RuntimeError("Milvus collection fields do not match the required schema")
        for name, expected in expected_types.items():
            _verify_field(fields, name, expected)
        if fields[_DOC_ID].get("is_primary") is not True:
            raise RuntimeError("Milvus doc_id field must be the primary key")
        _verify_field_param(fields, _DOC_ID, "max_length", _DOC_ID_MAX_LENGTH)
        _verify_field_param(fields, _TEXT, "max_length", _TEXT_MAX_LENGTH)
        _verify_field_param(fields, _SOURCE_KEY, "max_length", _SOURCE_KEY_MAX_LENGTH)
        _verify_field_param(
            fields,
            _DOCUMENT_SHA256,
            "max_length",
            _SHA256_LENGTH,
        )
        _verify_field_param(fields, _DENSE, "dim", self.dense_dimensions)

        index_names = set(client.list_indexes(self.collection_name))
        if index_names != {_DENSE_INDEX, _SPARSE_INDEX}:
            raise RuntimeError("Milvus collection indexes do not match the required schema")
        signatures = {
            _index_signature(client.describe_index(self.collection_name, index_name))
            for index_name in sorted(index_names)
        }
        expected_indexes = {
            (_DENSE, "AUTOINDEX", "COSINE"),
            (_SPARSE, "SPARSE_INVERTED_INDEX", "IP"),
        }
        if signatures != expected_indexes:
            raise RuntimeError("Milvus collection indexes use incompatible metrics or types")

    def upsert(self, records: Sequence[ChunkRecord]) -> int:
        rows = list(records)
        if not rows:
            return 0
        if len({record.doc_id for record in rows}) != len(rows):
            raise ValueError("upsert batch contains duplicate doc_ids")
        data: list[dict[str, object]] = []
        for record in rows:
            if len(record.doc_id.encode("utf-8")) > _DOC_ID_MAX_LENGTH:
                raise ValueError("doc_id exceeds the Milvus schema byte limit")
            if len(record.text.encode("utf-8")) > _TEXT_MAX_LENGTH:
                raise ValueError("text exceeds the Milvus schema byte limit")
            if len(record.source_key.encode("utf-8")) > _SOURCE_KEY_MAX_LENGTH:
                raise ValueError("source_key exceeds the Milvus schema byte limit")
            data.append(
                {
                    _DOC_ID: record.doc_id,
                    _TEXT: record.text,
                    _DENSE: _validated_dense(record.dense, self.dense_dimensions),
                    _SPARSE: _sparse_mapping(record.sparse),
                    _SOURCE_KEY: record.source_key,
                    _DOCUMENT_SHA256: record.document_sha256,
                    _ORDINAL: record.ordinal,
                    _METADATA: dict(record.metadata),
                }
            )
        result = self._get_client().upsert(
            self.collection_name,
            data,
            partial_update=False,
        )
        return _strict_count(result, "upsert_count", len(rows))

    def delete(self, doc_ids: Sequence[str]) -> int:
        ids = list(doc_ids)
        if not ids:
            return 0
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("delete ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("delete ids must be unique")
        result = self._get_client().delete(self.collection_name, ids=ids)
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            returned_ids = list(result)
            if any(not isinstance(doc_id, str) for doc_id in returned_ids):
                raise RuntimeError("Milvus delete response returned malformed ids")
            if not set(returned_ids) <= set(ids):
                raise RuntimeError("Milvus delete response returned unexpected ids")
            return len(ids)
        if not isinstance(result, Mapping):
            raise RuntimeError("Milvus delete response is not a mapping")
        raw_count = result.get("delete_count")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise RuntimeError("Milvus delete response has no non-negative delete_count")
        return len(ids)

    def search_dense(self, vector: DenseVector, *, limit: int) -> Sequence[ArmHit]:
        _require_positive_limit(limit)
        data = _validated_dense(vector, self.dense_dimensions)
        return self._search(
            data=data,
            anns_field=_DENSE,
            metric_type="COSINE",
            limit=limit,
        )

    def search_sparse(self, vector: SparseVector, *, limit: int) -> Sequence[ArmHit]:
        _require_positive_limit(limit)
        data = _sparse_mapping(vector)
        if not data:
            return ()
        return self._search(
            data=data,
            anns_field=_SPARSE,
            metric_type="IP",
            limit=limit,
        )

    def _search(
        self,
        *,
        data: list[float] | dict[int, float],
        anns_field: str,
        metric_type: str,
        limit: int,
    ) -> tuple[ArmHit, ...]:
        raw = self._get_client().search(
            self.collection_name,
            data=[data],
            anns_field=anns_field,
            search_params={"metric_type": metric_type, "params": {}},
            limit=limit,
            output_fields=[],
        )
        hits: list[ArmHit] = []
        seen: set[str] = set()
        for row in _search_rows(raw):
            doc_id = _row_string(row, _DOC_ID, context="search")
            raw_score = row.get("distance")
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise RuntimeError("Milvus search row has no numeric distance")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("Milvus search row has a non-finite distance")
            if doc_id in seen:
                raise RuntimeError("Milvus search response contains duplicate doc_ids")
            seen.add(doc_id)
            hits.append(ArmHit(doc_id=doc_id, score=score))
        if len(hits) > limit:
            raise RuntimeError("Milvus search returned more hits than requested")
        return tuple(hits)

    def fetch(self, doc_ids: Sequence[str]) -> Sequence[Passage]:
        ids = list(doc_ids)
        if not ids:
            return ()
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("fetch ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("fetch ids must be unique")
        raw = self._get_client().get(
            self.collection_name,
            ids=ids,
            output_fields=[
                _DOC_ID,
                _TEXT,
                _SOURCE_KEY,
                _DOCUMENT_SHA256,
                _ORDINAL,
                _METADATA,
            ],
        )
        rows: dict[str, Mapping[str, object]] = {}
        for row in _query_rows(raw):
            doc_id = _row_string(row, _DOC_ID, context="get")
            if doc_id not in ids:
                raise RuntimeError("Milvus get response contains an unrequested doc_id")
            if doc_id in rows:
                raise RuntimeError("Milvus get response contains duplicate doc_ids")
            rows[doc_id] = row
        return tuple(
            Passage(
                doc_id=doc_id,
                text=_row_string(rows[doc_id], _TEXT, context="get"),
                source_key=_row_string(rows[doc_id], _SOURCE_KEY, context="get"),
                document_sha256=_row_string(
                    rows[doc_id],
                    _DOCUMENT_SHA256,
                    context="get",
                ),
                ordinal=_row_int(rows[doc_id], _ORDINAL, context="get"),
                metadata=cast(Mapping[str, Any], _metadata(rows[doc_id])),
            )
            for doc_id in ids
            if doc_id in rows
        )

    def count(self) -> int:
        raw = self._get_client().get_collection_stats(self.collection_name).get("row_count")
        if isinstance(raw, bool):
            raise RuntimeError("Milvus row_count is not an integer")
        if isinstance(raw, str) and raw.isdecimal():
            raw = int(raw)
        if not isinstance(raw, int) or raw < 0:
            raise RuntimeError("Milvus row_count is not a non-negative integer")
        return raw

    def alias_target(self, alias: str) -> str | None:
        if not alias:
            raise ValueError("alias must be non-empty")
        try:
            description = self._get_client().describe_alias(alias)
        except Exception as exc:
            if _is_missing_alias(exc):
                return None
            raise
        value = description.get("collection")
        if value is None:
            value = description.get("collection_name")
        if not isinstance(value, str) or not value:
            raise RuntimeError("Milvus alias description has no collection name")
        return value

    def activate_alias(self, alias: str) -> None:
        target = self.alias_target(alias)
        if target == self.collection_name:
            return
        client = self._get_client()
        if target is None:
            client.create_alias(self.collection_name, alias)
        else:
            client.alter_alias(self.collection_name, alias)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> MilvusStore:
        self._get_client()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _is_missing_alias(exc: Exception) -> bool:
    """Recognise "this alias has not been published yet" and nothing else.

    Milvus Lite 3.2.0 raises ``code=100, alias 'x' does not exist``. Code 100 is
    reused for other absent objects, so the code alone must not be treated as a
    negative answer: a transport failure that happened to carry it would then be
    reported as "no alias yet" and an ingestion run would publish over a
    collection it never verified. Only wording that states absence qualifies.
    """
    message = str(exc).lower()
    return any(phrase in message for phrase in ("not exist", "doesn't exist", "not found"))
