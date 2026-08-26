from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zhrag.eval.retrieval import dense_runs, load_embedding_matrix
from zhrag.io_utils import append_jsonl


class TestLoadEmbeddingMatrix:
    def test_loads_last_write_wins_in_requested_order(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        append_jsonl(
            cache,
            [
                {"doc_id": "b", "embedding": [0.0, 1.0]},
                {"doc_id": "a", "embedding": [1.0, 0.0]},
                {"doc_id": "a", "embedding": [0.5, 0.5]},
            ],
        )
        matrix, missing = load_embedding_matrix(cache, ["a", "b"], width=2)
        assert not missing
        assert matrix.dtype == np.float32
        assert matrix.tolist() == [[0.5, 0.5], [0.0, 1.0]]

    def test_optional_missing_rows_remain_explicit(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [{"doc_id": "a", "embedding": [1.0]}])
        matrix, missing = load_embedding_matrix(cache, ["a", "b"], width=1, require_all=False)
        assert missing == ["b"]
        assert matrix.tolist() == [[1.0], [0.0]]

    @pytest.mark.parametrize(
        "row",
        [
            {"embedding": [1.0]},
            {"doc_id": "a", "embedding": "bad"},
            {"doc_id": "a", "embedding": ["bad"]},
            {"doc_id": "a", "embedding": [float("nan")]},
            {"doc_id": "a", "embedding": [float("inf")]},
        ],
    )
    def test_rejects_malformed_cached_vectors(self, tmp_path: Path, row: dict[str, object]) -> None:
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [row])
        with pytest.raises(SystemExit, match=r"doc_id|array|numeric|non-finite"):
            load_embedding_matrix(cache, ["a"], width=1)

    def test_rejects_duplicate_requested_ids(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [{"doc_id": "a", "embedding": [1.0]}])
        with pytest.raises(ValueError, match="unique"):
            load_embedding_matrix(cache, ["a", "a"], width=1)


class TestDenseRuns:
    def test_ties_break_on_document_id_not_input_order(self) -> None:
        query = np.zeros((1, 2), dtype=np.float32)
        docs = np.ones((3, 2), dtype=np.float32)
        assert dense_runs(query, docs, ["c", "a", "b"], depth=2) == [["a", "b"]]

    def test_rejects_matrix_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="row count"):
            dense_runs(
                np.ones((1, 2), dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                ["a", "b"],
                depth=1,
            )

    def test_rejects_non_finite_matrices_and_bad_depth(self) -> None:
        query = np.asarray([[float("nan")]], dtype=np.float32)
        docs = np.ones((1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            dense_runs(query, docs, ["a"], depth=1)
        with pytest.raises(ValueError, match="positive"):
            dense_runs(np.ones((1, 1), dtype=np.float32), docs, ["a"], depth=0)
