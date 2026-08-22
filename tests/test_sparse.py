"""Parity tests for client-side BM25 sparse vectors."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from zhrag.io_utils import write_json
from zhrag.lexical import (
    BM25,
    SPARSE_INDEX_SCHEMA,
    BM25Params,
    build_sparse_index,
    char_ngram,
    read_sparse_index,
    sparse_dot,
    write_sparse_index,
)

CORPUS = {
    "a": "TiDB 向量检索支持余弦距离。",
    "b": "TiKV 集群使用 Raft 协议。",
    "c": "向量索引可以加速检索，检索结果按距离排序。",
}


class TestParity:
    @pytest.mark.parametrize(
        "query",
        ["向量检索", "检索 检索", "TiDB 距离", "不存在词", ""],
    )
    def test_sparse_inner_product_matches_local_bm25(self, query: str) -> None:
        local = BM25(analyzer=char_ngram(2)).index(list(CORPUS), list(CORPUS.values()))
        expected = dict(local.search(query, k=len(CORPUS)))
        build = build_sparse_index(CORPUS)
        query_vector = build.index.encode_query(query)

        for document_id in CORPUS:
            got = sparse_dot(query_vector, build.vector_for(document_id))
            assert got == pytest.approx(expected.get(document_id, 0.0), abs=1e-12)

    def test_repeated_query_terms_are_binary(self) -> None:
        build = build_sparse_index(CORPUS)
        assert build.index.encode_query("向量检索") == build.index.encode_query("向量检索向量检索")

    def test_ranking_matches_local_bm25(self) -> None:
        local = BM25(analyzer=char_ngram(2)).index(list(CORPUS), list(CORPUS.values()))
        expected = [document_id for document_id, _ in local.search("向量检索", k=3)]
        build = build_sparse_index(CORPUS)
        query = build.index.encode_query("向量检索")
        got = sorted(
            CORPUS,
            key=lambda document_id: (
                -sparse_dot(query, build.vector_for(document_id)),
                document_id,
            ),
        )
        assert got[: len(expected)] == expected


class TestDeterminism:
    def test_input_mapping_order_does_not_change_the_build(self) -> None:
        forward = build_sparse_index(CORPUS)
        reverse = build_sparse_index(dict(reversed(CORPUS.items())))
        assert forward == reverse

    def test_fingerprint_binds_id_text_and_params(self) -> None:
        original = build_sparse_index(CORPUS)
        edited = build_sparse_index({**CORPUS, "a": CORPUS["a"] + "更新"})
        tuned = build_sparse_index(CORPUS, params=BM25Params(k1=1.2))
        renamed = build_sparse_index(
            {"z" if key == "a" else key: value for key, value in CORPUS.items()}
        )
        fingerprints = {
            original.index.fingerprint,
            edited.index.fingerprint,
            tuned.index.fingerprint,
            renamed.index.fingerprint,
        }
        assert len(fingerprints) == 4

    def test_vectors_and_vocabulary_are_sorted(self) -> None:
        build = build_sparse_index(CORPUS)
        assert build.index.terms == tuple(sorted(build.index.terms))
        assert build.document_ids == tuple(sorted(CORPUS))
        assert all(vector == tuple(sorted(vector)) for vector in build.document_vectors)


class TestBoundaries:
    def test_out_of_vocabulary_and_empty_queries_encode_empty(self) -> None:
        index = build_sparse_index({"a": "甲乙"}).index
        assert index.encode_query("丙丁") == ()
        assert index.encode_query("") == ()

    def test_short_nonempty_text_uses_one_term(self) -> None:
        build = build_sparse_index({"a": "甲", "b": "乙"})
        assert build.index.terms == ("乙", "甲")
        assert len(build.vector_for("a")) == 1

    def test_empty_document_has_an_empty_vector(self) -> None:
        build = build_sparse_index({"a": "", "b": "甲乙"})
        assert build.vector_for("a") == ()

    def test_unknown_document_id_raises(self) -> None:
        with pytest.raises(KeyError, match="missing"):
            build_sparse_index(CORPUS).vector_for("missing")

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            build_sparse_index({})

    @pytest.mark.parametrize(
        "params",
        [BM25Params(k1=-1), BM25Params(k1=math.inf), BM25Params(b=-0.1), BM25Params(b=1.1)],
    )
    def test_rejects_invalid_parameters(self, params: BM25Params) -> None:
        with pytest.raises(ValueError):
            build_sparse_index(CORPUS, params=params)

    def test_rejects_non_string_text(self) -> None:
        with pytest.raises(TypeError, match="texts"):
            build_sparse_index({"a": 1})  # type: ignore[dict-item]


class TestPersistence:
    def test_round_trip_preserves_query_encoding_exactly(self, tmp_path: Path) -> None:
        build = build_sparse_index(CORPUS)
        path = tmp_path / "sparse_index.json"
        write_sparse_index(path, build.index)
        loaded = read_sparse_index(path)

        assert loaded == build.index
        for query in ("向量检索", "TiDB 距离", "不存在词", ""):
            assert loaded.encode_query(query) == build.index.encode_query(query)

    def test_a_reloaded_index_still_reproduces_local_bm25(self, tmp_path: Path) -> None:
        build = build_sparse_index(CORPUS)
        path = tmp_path / "sparse_index.json"
        write_sparse_index(path, build.index)
        loaded = read_sparse_index(path)

        local = BM25(analyzer=char_ngram(2)).index(list(CORPUS), list(CORPUS.values()))
        expected = dict(local.search("向量检索", k=len(CORPUS)))
        query = loaded.encode_query("向量检索")
        for document_id in CORPUS:
            got = sparse_dot(query, build.vector_for(document_id))
            assert got == pytest.approx(expected.get(document_id, 0.0), abs=1e-12)

    def test_rejects_foreign_and_incomplete_files(self, tmp_path: Path) -> None:
        write_json(tmp_path / "other.json", {"schema": "something-else"})
        with pytest.raises(ValueError, match=SPARSE_INDEX_SCHEMA):
            read_sparse_index(tmp_path / "other.json")

        write_json(
            tmp_path / "partial.json",
            {"schema": SPARSE_INDEX_SCHEMA, "terms": [], "fingerprint": "x"},
        )
        with pytest.raises(ValueError, match="missing"):
            read_sparse_index(tmp_path / "partial.json")
