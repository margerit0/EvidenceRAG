from __future__ import annotations

import numpy as np
import pytest

from zhrag.eval.tidb_runs import (
    DENSE_LABEL,
    LEXICAL_LABEL,
    RERANK_LABEL,
    RRF_LABEL,
    TiDBRuns,
    build_dense_runs,
    build_lexical_runs,
    build_reranked_runs,
    build_rrf_runs,
    runs_fingerprint,
)
from zhrag.lexical import build_sparse_index


class TestLexicalRuns:
    def test_uses_persisted_sparse_bm25_scores_with_doc_id_ties(self) -> None:
        corpus = {"b": "事务隔离级别", "a": "事务隔离级别", "c": "备份恢复"}
        index = build_sparse_index(corpus).index
        runs = build_lexical_runs(index, corpus, ["事务隔离"], depth=3)
        assert runs == (("a", "b"),)

    def test_oov_query_returns_no_candidates_instead_of_zero_score_padding(self) -> None:
        corpus = {"b": "事务隔离", "a": "备份恢复"}
        index = build_sparse_index(corpus).index
        assert build_lexical_runs(index, corpus, ["𠮷"], depth=2) == ((),)

    def test_depth_is_capped_by_corpus_size(self) -> None:
        corpus = {"a": "事务隔离"}
        index = build_sparse_index(corpus).index
        assert build_lexical_runs(index, corpus, ["事务"], depth=100) == (("a",),)


class TestDenseRuns:
    def test_delegates_cosine_ranking_and_preserves_alignment(self) -> None:
        docs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert build_dense_runs(queries, docs, ["a", "b"], depth=2) == (
            ("a", "b"),
            ("b", "a"),
        )

    def test_rejects_duplicate_document_ids(self) -> None:
        matrix = np.asarray([[1.0], [1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="unique"):
            build_dense_runs(matrix[:1], matrix, ["a", "a"], depth=2)

    def test_dense_ties_use_document_id_order_at_the_cutoff(self) -> None:
        queries = np.asarray([[0.0, 0.0]], dtype=np.float32)
        docs = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        assert build_dense_runs(queries, docs, ["c", "a", "b"], depth=2) == (("a", "b"),)

    def test_rejects_matrix_shape_mismatches(self) -> None:
        with pytest.raises(ValueError, match="row count"):
            build_dense_runs(
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                ["a", "b"],
                depth=1,
            )
        with pytest.raises(ValueError, match="widths"):
            build_dense_runs(
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
                ["a"],
                depth=1,
            )


class TestRrfRuns:
    def test_fuses_every_aligned_query(self) -> None:
        rows = build_rrf_runs(
            (("a", "b"), ("c", "d")),
            (("b", "a"), ("d", "c")),
            depth=2,
            rrf_k=10,
        )
        assert rows == (("a", "b"), ("c", "d"))

    def test_rejects_misaligned_query_counts(self) -> None:
        with pytest.raises(ValueError, match="counts differ"):
            build_rrf_runs((("a",),), (), depth=1, rrf_k=10)


class TestRerankedRuns:
    def test_reranks_only_the_apply_prefix(self) -> None:
        fused = (("a", "b", "c", "d"),)
        scores = {("q", "a"): 0.1, ("q", "b"): 0.9, ("q", "c"): 1.0}
        rows = build_reranked_runs(
            fused,
            ("q",),
            scores,
            request_depth=3,
            apply_depth=2,
        )
        assert rows == (("b", "a", "c", "d"),)

    def test_requires_scores_for_the_full_request_not_just_apply_prefix(self) -> None:
        with pytest.raises(ValueError, match="missing 1"):
            build_reranked_runs(
                (("a", "b", "c"),),
                ("q",),
                {("q", "a"): 1.0, ("q", "b"): 0.5},
                request_depth=3,
                apply_depth=2,
            )

    def test_rejects_apply_depth_above_request_depth(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            build_reranked_runs(
                (("a", "b"),),
                ("q",),
                {},
                request_depth=1,
                apply_depth=2,
            )


class TestTiDBRuns:
    def _runs(self) -> TiDBRuns:
        return TiDBRuns(
            query_ids=("q1", "q2"),
            runs={
                LEXICAL_LABEL: (("a", "b"), ("c", "d")),
                DENSE_LABEL: (("b", "a"), ("d", "c")),
                RRF_LABEL: (("a", "b"), ("c", "d")),
                RERANK_LABEL: (("b", "a"), ("d", "c")),
            },
        )

    def test_selects_aligned_runs_by_query_id(self) -> None:
        assert self._runs().for_query("q2") == {
            LEXICAL_LABEL: ("c", "d"),
            DENSE_LABEL: ("d", "c"),
            RRF_LABEL: ("c", "d"),
            RERANK_LABEL: ("d", "c"),
        }

    def test_requires_the_frozen_four_systems(self) -> None:
        with pytest.raises(ValueError, match="labels differ"):
            TiDBRuns(query_ids=("q",), runs={LEXICAL_LABEL: (("a",),)})

    def test_allows_an_empty_lexical_run_for_an_oov_query(self) -> None:
        rows = {
            LEXICAL_LABEL: ((),),
            DENSE_LABEL: (("a",),),
            RRF_LABEL: (("a",),),
            RERANK_LABEL: (("a",),),
        }
        assert TiDBRuns(query_ids=("q",), runs=rows).runs[LEXICAL_LABEL] == ((),)

    def test_rejects_duplicate_ids_within_a_run(self) -> None:
        rows = {
            LEXICAL_LABEL: (("a", "a"),),
            DENSE_LABEL: (("a",),),
            RRF_LABEL: (("a",),),
            RERANK_LABEL: (("a",),),
        }
        with pytest.raises(ValueError, match="unique"):
            TiDBRuns(query_ids=("q",), runs=rows)

    def test_fingerprint_is_stable_and_order_sensitive(self) -> None:
        rows = self._runs()
        assert len(rows.fingerprint) == 64
        assert rows.fingerprint == runs_fingerprint(rows.query_ids, rows.runs)
        changed = dict(rows.runs)
        changed[LEXICAL_LABEL] = (("b", "a"), ("c", "d"))
        assert rows.fingerprint != runs_fingerprint(rows.query_ids, changed)

    def test_unknown_query_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="missing"):
            self._runs().for_query("missing")


@pytest.mark.parametrize("depth", [0, -1])
def test_builders_reject_non_positive_depth(depth: int) -> None:
    corpus = {"a": "事务隔离"}
    index = build_sparse_index(corpus).index
    with pytest.raises(ValueError, match="positive"):
        build_lexical_runs(index, corpus, ["事务"], depth=depth)
