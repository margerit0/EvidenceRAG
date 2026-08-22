"""Hermetic contract tests for the optional Milvus adapter."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from zhrag.store import ChunkRecord, MilvusConfig, MilvusStore

SHA256 = "a" * 64


class MilvusLikeError(RuntimeError):
    """Mimic ``pymilvus.MilvusException``: a numeric code plus a message."""

    def __init__(self, message: str, *, code: int) -> None:
        super().__init__(message)
        self.code = code


class FakeSchema:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = kwargs
        self.fields: list[dict[str, object]] = []

    def add_field(self, **kwargs: object) -> None:
        self.fields.append(kwargs)


class FakeIndexes:
    def __init__(self) -> None:
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **kwargs: object) -> None:
        self.indexes.append(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.exists = False
        self.factory_calls: list[tuple[str, str | None, str, float | None]] = []
        self.schema_description: dict[str, object] = {}
        self.index_descriptions: dict[str, dict[str, object]] = {}
        self.created_schema: FakeSchema | None = None
        self.created_indexes: FakeIndexes | None = None
        self.create_kwargs: dict[str, object] | None = None
        self.loaded: list[str] = []
        self.upserts: list[tuple[str, list[dict[str, object]], dict[str, object]]] = []
        self.delete_result: object = {"delete_count": 0}
        self.deletes: list[list[str]] = []
        self.search_results: list[object] = []
        self.searches: list[tuple[str, dict[str, object]]] = []
        self.get_rows: list[dict[str, object]] = []
        self.get_calls: list[tuple[list[str], list[str]]] = []
        self.row_count: object = 0
        self.aliases: dict[str, str] = {}
        self.describe_alias_error: Exception | None = None
        self.closed = 0

    def has_collection(self, collection_name: str) -> bool:
        assert collection_name == "chunks_v1"
        return self.exists

    def describe_collection(self, collection_name: str) -> Mapping[str, object]:
        assert collection_name == "chunks_v1"
        return self.schema_description

    def create_schema(self, **kwargs: object) -> FakeSchema:
        self.created_schema = FakeSchema(kwargs)
        return self.created_schema

    def prepare_index_params(self) -> FakeIndexes:
        self.created_indexes = FakeIndexes()
        return self.created_indexes

    def create_collection(self, **kwargs: object) -> None:
        self.create_kwargs = kwargs
        self.exists = True

    def list_indexes(self, collection_name: str) -> list[str]:
        assert collection_name == "chunks_v1"
        return list(self.index_descriptions)

    def describe_index(self, collection_name: str, index_name: str) -> Mapping[str, object]:
        assert collection_name == "chunks_v1"
        return self.index_descriptions[index_name]

    def load_collection(self, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def upsert(
        self,
        collection_name: str,
        data: list[dict[str, object]],
        **kwargs: object,
    ) -> object:
        self.upserts.append((collection_name, data, kwargs))
        return {"upsert_count": len(data)}

    def delete(self, collection_name: str, *, ids: list[str]) -> object:
        assert collection_name == "chunks_v1"
        self.deletes.append(ids)
        return self.delete_result

    def search(self, collection_name: str, **kwargs: object) -> object:
        self.searches.append((collection_name, kwargs))
        return self.search_results.pop(0)

    def get(
        self,
        collection_name: str,
        *,
        ids: list[str],
        output_fields: list[str],
    ) -> object:
        assert collection_name == "chunks_v1"
        self.get_calls.append((ids, output_fields))
        return self.get_rows

    def get_collection_stats(self, collection_name: str) -> Mapping[str, object]:
        assert collection_name == "chunks_v1"
        return {"row_count": self.row_count}

    def describe_alias(self, alias: str) -> Mapping[str, object]:
        if self.describe_alias_error is not None:
            raise self.describe_alias_error
        if alias not in self.aliases:
            raise MilvusLikeError(f"alias '{alias}' does not exist", code=100)
        return {"collection": self.aliases[alias]}

    def create_alias(self, collection_name: str, alias: str) -> None:
        self.aliases[alias] = collection_name

    def alter_alias(self, collection_name: str, alias: str) -> None:
        self.aliases[alias] = collection_name

    def close(self) -> None:
        self.closed += 1


def make_store(
    client: FakeClient,
    *,
    dimensions: int = 3,
) -> MilvusStore:
    def factory(
        uri: str,
        token: str | None,
        database: str,
        timeout: float | None,
    ) -> Any:
        client.factory_calls.append((uri, token, database, timeout))
        return client

    store = MilvusStore(
        MilvusConfig(
            uri="local.db",
            collection_name="chunks_v1",
            dense_dimensions=dimensions,
            token="secret",
            database="zhrag",
            timeout=9.0,
        ),
        _client_factory=factory,
        _data_type=lambda name: name,
    )
    return store


def valid_description(
    *, dimensions: int = 3
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    fields: list[dict[str, object]] = [
        {
            "name": "doc_id",
            "type": "VARCHAR",
            "params": {"max_length": 64},
            "is_primary": True,
        },
        {"name": "text", "type": "VARCHAR", "params": {"max_length": 65_535}},
        {"name": "dense", "type": "FLOAT_VECTOR", "params": {"dim": dimensions}},
        {"name": "sparse", "type": "SPARSE_FLOAT_VECTOR", "params": {}},
        {
            "name": "source_key",
            "type": "VARCHAR",
            "params": {"max_length": 2_048},
        },
        {
            "name": "document_sha256",
            "type": "VARCHAR",
            "params": {"max_length": 64},
        },
        {"name": "ordinal", "type": "INT64", "params": {}},
        {"name": "metadata", "type": "JSON", "params": {}},
    ]
    indexes = {
        "dense": {
            "field_name": "dense",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        },
        "sparse": {
            "field_name": "sparse",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        },
    }
    return (
        {
            "auto_id": False,
            "enable_dynamic_field": False,
            "fields": fields,
        },
        indexes,
    )


def record(doc_id: str = "a") -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        text="中文段落",
        dense=(1.0, 0.0, -1.0),
        sparse=((1, 0.5), (4, 2.0)),
        source_key="pingcap/docs-cn:doc.md",
        document_sha256=SHA256,
        ordinal=2,
        metadata={"scope": "evergreen", "published": True},
    )


class TestCollection:
    def test_create_uses_fixed_schema_and_indexes(self) -> None:
        client = FakeClient()
        store = make_store(client)

        store.ensure_collection()

        assert client.created_schema is not None
        assert client.created_schema.kwargs == {
            "auto_id": False,
            "enable_dynamic_field": False,
        }
        assert [field["field_name"] for field in client.created_schema.fields] == [
            "doc_id",
            "text",
            "dense",
            "sparse",
            "source_key",
            "document_sha256",
            "ordinal",
            "metadata",
        ]
        dense = client.created_schema.fields[2]
        assert dense["datatype"] == "FLOAT_VECTOR"
        assert dense["dim"] == 3
        assert client.created_indexes is not None
        assert client.created_indexes.indexes == [
            {
                "field_name": "dense",
                "index_name": "dense",
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE",
            },
            {
                "field_name": "sparse",
                "index_name": "sparse",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        ]
        assert client.create_kwargs is not None
        assert client.create_kwargs["consistency_level"] == "Strong"
        assert client.factory_calls == [("local.db", "secret", "zhrag", 9.0)]

    def test_existing_compatible_collection_is_loaded(self) -> None:
        client = FakeClient()
        client.exists = True
        client.schema_description, client.index_descriptions = valid_description()
        store = make_store(client)

        store.ensure_collection()
        store.ensure_collection()

        assert client.create_kwargs is None
        assert client.loaded == ["chunks_v1", "chunks_v1"]

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (
                lambda schema, _indexes: schema["fields"][2]["params"].update(dim=2),
                "wrong dimension",
            ),
            (
                lambda schema, _indexes: schema.update(enable_dynamic_field=True),
                "disable dynamic",
            ),
            (
                lambda _schema, indexes: indexes["dense"].update(metric_type="L2"),
                "incompatible metrics",
            ),
        ],
    )
    def test_incompatible_existing_collection_fails_closed(self, mutate: Any, message: str) -> None:
        client = FakeClient()
        client.exists = True
        client.schema_description, client.index_descriptions = valid_description()
        mutate(client.schema_description, client.index_descriptions)

        with pytest.raises(RuntimeError, match=message):
            make_store(client).ensure_collection()
        assert client.create_kwargs is None

    def test_normal_import_does_not_import_pymilvus(self) -> None:
        before = "pymilvus" in sys.modules
        client = FakeClient()
        make_store(client).count()
        assert ("pymilvus" in sys.modules) is before


class TestRows:
    def test_upsert_sends_one_complete_serialized_row(self) -> None:
        client = FakeClient()
        store = make_store(client)

        assert store.upsert([record()]) == 1

        collection, rows, kwargs = client.upserts[0]
        assert collection == "chunks_v1"
        assert kwargs == {"partial_update": False}
        assert rows == [
            {
                "doc_id": "a",
                "text": "中文段落",
                "dense": [1.0, 0.0, -1.0],
                "sparse": {1: 0.5, 4: 2.0},
                "source_key": "pingcap/docs-cn:doc.md",
                "document_sha256": SHA256,
                "ordinal": 2,
                "metadata": {"scope": "evergreen", "published": True},
            }
        ]

    def test_upsert_rejects_duplicate_ids_wrong_width_and_utf8_byte_overflow(self) -> None:
        client = FakeClient()
        store = make_store(client)
        with pytest.raises(ValueError, match="duplicate"):
            store.upsert([record(), record()])
        with pytest.raises(ValueError, match="dimensions"):
            store.upsert([replace(record(), dense=(1.0,))])
        with pytest.raises(ValueError, match="byte limit"):
            store.upsert([replace(record(), text="甲" * 30_000)])
        assert client.upserts == []

    def test_empty_mutations_are_no_ops_and_delete_ack_is_not_business_count(self) -> None:
        client = FakeClient()
        store = make_store(client)
        assert store.upsert([]) == 0
        assert store.delete([]) == 0
        client.delete_result = {"delete_count": 0}
        assert store.delete(["missing"]) == 1
        assert client.deletes == [["missing"]]

    def test_fetch_restores_request_order_and_omits_vectors(self) -> None:
        client = FakeClient()
        client.get_rows = [
            {
                "doc_id": "b",
                "text": "乙",
                "source_key": "source:b",
                "document_sha256": "b" * 64,
                "ordinal": 1,
                "metadata": {"x": 2},
            },
            {
                "doc_id": "a",
                "text": "甲",
                "source_key": "source:a",
                "document_sha256": SHA256,
                "ordinal": 0,
                "metadata": {"x": 1},
            },
        ]
        store = make_store(client)

        got = store.fetch(["a", "missing", "b"])

        assert [passage.doc_id for passage in got] == ["a", "b"]
        assert got[0].metadata == {"x": 1}
        ids, fields = client.get_calls[0]
        assert ids == ["a", "missing", "b"]
        assert "dense" not in fields
        assert "sparse" not in fields


class TestSearchAndLifecycle:
    def test_dense_and_sparse_search_map_positive_scores(self) -> None:
        client = FakeClient()
        client.search_results = [
            [[{"doc_id": "a", "distance": 0.9}, {"doc_id": "b", "distance": 0.2}]],
            [[{"doc_id": "b", "distance": 3.5}]],
        ]
        store = make_store(client)

        dense = store.search_dense((1.0, 0.0, 0.0), limit=2)
        sparse = store.search_sparse(((2, 1.0),), limit=1)

        assert [(hit.doc_id, hit.score) for hit in dense] == [("a", 0.9), ("b", 0.2)]
        assert [(hit.doc_id, hit.score) for hit in sparse] == [("b", 3.5)]
        _, dense_call = client.searches[0]
        _, sparse_call = client.searches[1]
        assert dense_call["data"] == [[1.0, 0.0, 0.0]]
        assert dense_call["anns_field"] == "dense"
        assert dense_call["search_params"] == {"metric_type": "COSINE", "params": {}}
        assert sparse_call["data"] == [{2: 1.0}]
        assert sparse_call["anns_field"] == "sparse"
        assert sparse_call["search_params"] == {"metric_type": "IP", "params": {}}

    def test_empty_sparse_query_short_circuits_without_vendor_search(self) -> None:
        client = FakeClient()
        assert make_store(client).search_sparse((), limit=10) == ()
        assert client.searches == []

    @pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
    def test_non_finite_vendor_scores_fail_closed(self, score: float) -> None:
        client = FakeClient()
        client.search_results = [[[{"doc_id": "a", "distance": score}]]]
        with pytest.raises(RuntimeError, match="non-finite"):
            make_store(client).search_dense((1.0, 0.0, 0.0), limit=1)

    def test_count_alias_switch_and_close_are_idempotent(self) -> None:
        client = FakeClient()
        client.row_count = "7"
        store = make_store(client)

        assert store.count() == 7
        assert store.alias_target("active") is None
        store.activate_alias("active")
        assert store.alias_target("active") == "chunks_v1"
        store.activate_alias("active")
        store.close()
        store.close()
        assert client.aliases == {"active": "chunks_v1"}
        assert client.closed == 1

    def test_alias_absence_is_recognised_only_from_wording(self) -> None:
        # Milvus Lite 3.2.0 raises exactly this for an unpublished alias.
        client = FakeClient()
        client.describe_alias_error = MilvusLikeError(
            "<MilvusException: (code=100, message=alias 'active' does not exist)>",
            code=100,
        )
        assert make_store(client).alias_target("active") is None

    def test_transport_failure_is_not_reported_as_a_missing_alias(self) -> None:
        client = FakeClient()
        client.describe_alias_error = MilvusLikeError(
            "<MilvusException: (code=100, message=Fail connecting to server on "
            "127.0.0.1:19530, illegal connection params or server unavailable)>",
            code=100,
        )
        store = make_store(client)
        with pytest.raises(MilvusLikeError):
            store.alias_target("active")
        with pytest.raises(MilvusLikeError):
            store.activate_alias("active")
        assert client.aliases == {}
