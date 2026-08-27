from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from numpy.typing import NDArray

import zhrag.eval.hybrid_mrl1024 as hybrid
from zhrag.embedding_contract import CRUD_EMBEDDING_MODEL, DOCUMENT_PROMPT, QUERY_PROMPT
from zhrag.eval.crud import Query
from zhrag.eval.hybrid_mrl1024 import (
    A_LABEL,
    ARM_LABELS,
    E_LABEL,
    G_LABEL,
    H_LABEL,
    HybridInputs,
    build_hybrid_mrl1024_runs,
    evaluate_hybrid_mrl1024,
    validate_hybrid_mrl1024_report,
)


def _inputs() -> HybridInputs:
    corpus = {
        f"doc-{index:03d}": f"主题{index:03d}的唯一新闻正文，包含足够的字符供二元分词。"
        for index in range(100)
    }
    specs = [
        ("questanswer_1doc", (0,)),
        ("questanswer_1doc", (1,)),
        ("questanswer_2docs", (2, 3)),
        ("questanswer_2docs", (4, 5)),
        ("questanswer_3docs", (6, 7, 8)),
        ("questanswer_3docs", (9, 10, 11)),
    ]
    queries = tuple(
        Query(
            query_id=f"{task}:q{index}",
            question=f"主题{gold[0]:03d}是什么？",
            answer=f"答案-{index}",
            gold_doc_ids=tuple(f"doc-{doc:03d}" for doc in gold),
            task=task,
        )
        for index, (task, gold) in enumerate(specs)
    )
    doc_matrix = np.zeros((len(corpus), hybrid.SOURCE_WIDTH), dtype=np.float32)
    for index in range(len(corpus)):
        doc_matrix[index, index] = 0.8
        doc_matrix[index, hybrid.EFFECTIVE_WIDTH + index] = 0.6
    query_matrix = np.zeros((len(queries), hybrid.SOURCE_WIDTH), dtype=np.float32)
    for index, (_task, gold) in enumerate(specs):
        query_matrix[index, gold[0]] = 0.6
        query_matrix[index, hybrid.EFFECTIVE_WIDTH + ((gold[0] + 1) % len(corpus))] = 0.8

    unique_gold = {doc_id for query in queries for doc_id in query.gold_doc_ids}
    manifest: dict[str, object] = {
        "distractors": len(corpus) - len(unique_gold),
        "documents": len(corpus),
        "gold_documents": len(unique_gold),
        "queries": len(queries),
        "queries_per_gold_arity": {"1": 2, "2": 2, "3": 2},
        "queries_per_task": {
            "questanswer_1doc": 2,
            "questanswer_2docs": 2,
            "questanswer_3docs": 2,
        },
        "records_per_task": {
            "questanswer_1doc": 2,
            "questanswer_2docs": 2,
            "questanswer_3docs": 2,
        },
        "source_file": "fixture.json",
        "source_sha256": "a" * 64,
    }
    return HybridInputs(
        manifest=manifest,
        corpus=corpus,
        queries=queries,
        doc_ids=tuple(corpus),
        doc_matrix=doc_matrix,
        query_matrix=query_matrix,
        document_provenance={"model": CRUD_EMBEDDING_MODEL, "prompt": DOCUMENT_PROMPT},
        query_provenance={"model": CRUD_EMBEDDING_MODEL, "prompt": QUERY_PROMPT},
    )


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return evaluate_hybrid_mrl1024(_inputs(), resamples=99, seed=7)


class TestFrozenRuns:
    def test_builds_only_the_named_a_e_h_g_arms(self) -> None:
        runs = build_hybrid_mrl1024_runs(_inputs())
        assert tuple(runs) == ARM_LABELS
        assert all(len(runs[arm]) == 6 for arm in ARM_LABELS)
        assert all(len(run) <= 100 for run in runs[A_LABEL])
        assert all(len(run) == 100 for run in runs[E_LABEL])
        assert all(len(run) >= 100 for run in runs[H_LABEL])
        assert all(len(run) >= 100 for run in runs[G_LABEL])

    def test_applies_the_mrl_transform_to_query_and_document_matrices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = _inputs()
        real = hybrid.prefix_l2_normalize
        seen: list[tuple[int, int]] = []

        def recording(
            matrix: NDArray[np.float32],
            width: int,
        ) -> NDArray[np.float32]:
            seen.append(matrix.shape)
            return real(matrix, width)

        monkeypatch.setattr(hybrid, "prefix_l2_normalize", recording)
        build_hybrid_mrl1024_runs(inputs)
        assert seen == [inputs.query_matrix.shape, inputs.doc_matrix.shape]

    def test_h_and_g_are_distinct_frozen_references(self) -> None:
        runs = build_hybrid_mrl1024_runs(_inputs())
        assert runs[H_LABEL] != runs[G_LABEL]


class TestAggregateReport:
    def test_has_the_two_scopes_and_predeclared_family_sizes(
        self, report: dict[str, object]
    ) -> None:
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["headline"]["n"] == 2  # type: ignore[index]
        assert [metrics["by_arity"][str(arity)]["n"] for arity in (1, 2, 3)] == [2, 2, 2]  # type: ignore[index]

        contrasts = report["contrasts"]
        assert isinstance(contrasts, dict)
        assert {name: len(rows) for name, rows in contrasts.items()} == {
            "efficacy-binary": 12,
            "efficacy-continuous": 12,
            "retention-binary": 6,
            "retention-continuous": 6,
        }

    def test_every_contrast_is_paired_with_h_as_treatment(self, report: dict[str, object]) -> None:
        contrasts = report["contrasts"]
        assert isinstance(contrasts, dict)
        rows = [row for family in contrasts.values() for row in family]
        assert all(row["treatment"] == H_LABEL for row in rows)
        assert all(
            row["wins"] + row["losses"] + row["ties_nonzero"] + row["ties_zero"] == 2
            for row in rows
        )

    def test_is_byte_reproducible_for_the_same_seed(self, report: dict[str, object]) -> None:
        rerun = evaluate_hybrid_mrl1024(_inputs(), resamples=99, seed=7)
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rerendered = json.dumps(
            rerun,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert rerendered == rendered

    def test_contains_no_raw_or_per_query_content(self, report: dict[str, object]) -> None:
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)
        for sentinel in ("主题000是什么", "答案-0", "doc-000", "questanswer_1doc:q0"):
            assert sentinel not in rendered
        for forbidden in (
            '"answer"',
            '"doc_id"',
            '"embedding"',
            '"per_query"',
            '"query_id"',
            '"question"',
            '"raw_rows"',
            '"runs"',
            '"scores"',
            '"text"',
            '"vectors"',
        ):
            assert forbidden not in rendered


class TestReportValidation:
    @pytest.mark.parametrize(
        "mutation",
        [
            lambda value: value.update({"unexpected": 1}),
            lambda value: value["inputs"].update({"query_id": "leak"}),
            lambda value: value["contrasts"]["efficacy-binary"][0].update({"adjusted_p": 0.123456}),
            lambda value: value["contrasts"]["retention-continuous"][0].update({"wins": 999}),
        ],
        ids=["top-level", "raw-field", "holm", "wlt"],
    )
    def test_rejects_structural_or_inferential_drift(
        self,
        report: dict[str, object],
        mutation: object,
    ) -> None:
        broken = copy.deepcopy(report)
        assert callable(mutation)
        mutation(broken)
        with pytest.raises(ValueError):
            validate_hybrid_mrl1024_report(broken)

    def test_rejects_invalid_resamples_and_seed(self) -> None:
        for kwargs in ({"resamples": 0}, {"seed": -1}, {"seed": True}):
            with pytest.raises(ValueError):
                evaluate_hybrid_mrl1024(_inputs(), **kwargs)
