"""Tests for the char-n-gram BM25 baseline."""

from __future__ import annotations

import pytest

from zhrag.lexical import BM25, BM25Params, char_ngram, union

DOCS = {
    "d_vector": "TiDB 向量搜索功能支持 HNSW 索引，用于近似最近邻检索。",
    "d_backup": "使用 BR 工具对 TiDB 集群进行全量备份与增量恢复。",
    "d_deploy": "通过 tiup cluster deploy 命令部署一个新的 TiDB 集群。",
    "d_sql": "EXPLAIN 语句用于查看 SQL 的执行计划。",
}


@pytest.fixture
def index() -> BM25:
    return BM25().index(list(DOCS), list(DOCS.values()))


class TestAnalyzer:
    def test_bigrams_strip_whitespace(self) -> None:
        assert char_ngram(2)("向 量") == ["向量"]

    def test_unigram_and_trigram_widths(self) -> None:
        assert char_ngram(1)("向量搜索") == ["向", "量", "搜", "索"]
        assert char_ngram(3)("向量搜索") == ["向量搜", "量搜索"]

    def test_text_shorter_than_n_is_kept_whole(self) -> None:
        assert char_ngram(3)("向量") == ["向量"]

    def test_empty_input_yields_no_terms(self) -> None:
        assert char_ngram(2)("") == []

    def test_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            char_ngram(0)

    def test_union_concatenates(self) -> None:
        assert union(char_ngram(1), char_ngram(2))("向量") == ["向", "量", "向量"]


class TestSearch:
    def test_ranks_the_topical_document_first(self, index: BM25) -> None:
        assert index.search("向量搜索索引", k=1)[0][0] == "d_vector"

    def test_matches_latin_identifiers(self, index: BM25) -> None:
        """Exact identifier matching is why the lexical arm exists at all."""
        assert index.search("tiup cluster deploy", k=1)[0][0] == "d_deploy"

    def test_returns_at_most_k(self, index: BM25) -> None:
        assert len(index.search("TiDB", k=2)) == 2

    def test_scores_are_descending(self, index: BM25) -> None:
        scores = [s for _, s in index.search("TiDB 集群备份", k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_unmatched_query_returns_nothing(self, index: BM25) -> None:
        assert index.search("ZZZZZZ", k=5) == []

    def test_repeated_query_terms_do_not_double_count(self, index: BM25) -> None:
        once = dict(index.search("向量", k=4))
        twice = dict(index.search("向量 向量", k=4))
        assert once == twice


class TestIndexLifecycle:
    def test_search_before_index_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="index\\(\\) must be called"):
            BM25().search("x")

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            BM25().index([], [])

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            BM25().index(["a"], ["x", "y"])

    def test_reports_size_and_vocabulary(self, index: BM25) -> None:
        assert len(index) == 4
        assert index.vocabulary_size > 0

    def test_reindexing_replaces_previous_content(self, index: BM25) -> None:
        index.index(["only"], ["全新的内容"])
        assert len(index) == 1
        assert index.search("全新", k=1)[0][0] == "only"


class TestParameters:
    def test_b_shrinks_the_long_documents_advantage(self) -> None:
        """Length normalization damps, but does not reverse, a raw-frequency win.

        With k1=1.5 the tf saturation curve is not steep enough to make a 40x
        shorter document outrank a 40x more frequent one, so both settings still
        rank ``long`` first. What ``b`` actually controls is the size of that
        gap -- which is the property worth pinning.
        """
        ids, docs = ["short", "long"], ["向量" * 5, "向量" * 200]

        def ratio(b: float) -> float:
            scores = dict(BM25(params=BM25Params(b=b)).index(ids, docs).search("向量", k=2))
            return scores["long"] / scores["short"]

        assert ratio(0.75) < ratio(0.0)

    def test_k1_zero_ignores_term_frequency(self) -> None:
        """At k1=0 the tf component collapses to a constant, leaving pure IDF."""
        ids, docs = ["once", "many"], ["向量 检索", "向量 向量 向量 向量 检索"]
        scores = dict(BM25(params=BM25Params(k1=0.0, b=0.0)).index(ids, docs).search("向量", k=2))
        assert scores["once"] == pytest.approx(scores["many"])
