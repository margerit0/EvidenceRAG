"""Tests for CRUD-RAG loading.

The corpus this builds is what every ablation number is measured against, so the
cases pinned here are the ones that would silently corrupt a benchmark: letting
fabricated text into the pool, losing a gold document, or producing a corpus
whose size does not match what the ablation table claims.
"""

from __future__ import annotations

import pytest

from zhrag.eval.crud import (
    MIN_DOCUMENT_CHARS,
    Query,
    build_queries,
    document_id,
    harvest_documents,
    sample_corpus,
)

LONG = "新闻正文内容。" * 40  # comfortably over MIN_DOCUMENT_CHARS


def body(tag: str) -> str:
    return f"{tag}：{LONG}"


RAW = {
    "questanswer_1doc": [
        {"ID": "q1", "questions": "问题一？", "answers": "答案一", "news1": body("A")},
    ],
    "questanswer_2docs": [
        {
            "ID": "q2",
            "questions": "问题二？",
            "answers": "答案二",
            "news1": body("A"),
            "news2": body("B"),
        },
    ],
    "questanswer_3docs": [
        {
            "ID": "q3",
            "questions": "问题三？",
            "answers": "答案三",
            "news1": body("B"),
            "news2": body("C"),
            "news3": body("D"),
        },
    ],
    "event_summary": [{"ID": "e1", "text": body("E"), "summary": "摘要"}],
    "hallu_modified": [
        {
            "ID": "h1",
            "newsBeginning": "太短了。",
            "newsRemainder": body("F"),
            "hallucinatedContinuation": body("FABRICATED-1"),
            "hallucinatedMod": body("FABRICATED-2"),
        }
    ],
    "continuing_writing": [{"ID": "c1", "beginning": body("G"), "continuing": body("H")}],
}


class TestDocumentId:
    def test_is_stable(self) -> None:
        assert document_id("文本") == document_id("文本")

    def test_ignores_surrounding_whitespace(self) -> None:
        assert document_id("  文本\n") == document_id("文本")

    def test_distinguishes_different_text(self) -> None:
        assert document_id("甲") != document_id("乙")

    def test_is_short_enough_to_read(self) -> None:
        assert len(document_id("文本")) == 16


class TestHarvest:
    def test_collects_bodies_from_every_task(self) -> None:
        pool = harvest_documents(RAW)
        texts = set(pool.values())
        for tag in ("A", "B", "C", "D", "E", "F"):
            assert body(tag) in texts

    def test_excludes_fabricated_text(self) -> None:
        """The load-bearing case: hallucinated prose must never be retrievable."""
        pool = harvest_documents(RAW)
        assert not any("FABRICATED" in t for t in pool.values())

    def test_excludes_continuing_writing_halves(self) -> None:
        pool = harvest_documents(RAW)
        texts = set(pool.values())
        assert body("G") not in texts
        assert body("H") not in texts

    def test_drops_fragments_below_the_length_floor(self) -> None:
        pool = harvest_documents(RAW)
        assert all(len(t) >= MIN_DOCUMENT_CHARS for t in pool.values())
        assert "太短了。" not in pool.values()

    def test_deduplicates_across_tasks(self) -> None:
        """Document A is evidence in both 1doc and 2docs; it is one document."""
        pool = harvest_documents(RAW)
        assert list(pool.values()).count(body("A")) == 1

    def test_ignores_non_string_values(self) -> None:
        assert harvest_documents({"t": [{"news1": None, "text": 123}]}) == {}

    def test_empty_input(self) -> None:
        assert harvest_documents({}) == {}


class TestBuildQueries:
    def test_reads_arity_from_the_record(self) -> None:
        arities = {q.query_id: len(q.gold_doc_ids) for q in build_queries(RAW)}
        assert arities == {
            "questanswer_1doc:q1": 1,
            "questanswer_2docs:q2": 2,
            "questanswer_3docs:q3": 3,
        }

    def test_query_ids_are_task_qualified(self) -> None:
        ids = {q.query_id for q in build_queries(RAW)}
        assert "questanswer_1doc:q1" in ids

    def test_gold_ids_resolve_into_the_pool(self) -> None:
        pool = harvest_documents(RAW)
        for q in build_queries(RAW, pool=pool):
            assert all(doc_id in pool for doc_id in q.gold_doc_ids)

    def test_duplicate_evidence_collapses(self) -> None:
        """news1 == news2 is one document, so the query has one gold, not two."""
        raw = {
            "questanswer_2docs": [
                {
                    "ID": "d",
                    "questions": "问？",
                    "answers": "答",
                    "news1": body("X"),
                    "news2": body("X"),
                }
            ]
        }
        assert len(build_queries(raw)[0].gold_doc_ids) == 1

    def test_skips_queries_whose_evidence_is_missing_from_the_pool(self) -> None:
        """Keeping them would depress every arm's recall by a constant."""
        pool = {document_id(body("A")): body("A")}
        kept = {q.query_id for q in build_queries(RAW, pool=pool)}
        assert kept == {"questanswer_1doc:q1"}

    def test_skips_records_without_a_usable_question(self) -> None:
        raw = {
            "questanswer_1doc": [
                {"ID": "a", "questions": "   ", "news1": body("A")},
                {"ID": "b", "news1": body("A")},
                {"questions": "无 ID？", "news1": body("A")},
            ]
        }
        assert build_queries(raw) == []

    def test_skips_records_with_no_evidence_over_the_floor(self) -> None:
        raw = {"questanswer_1doc": [{"ID": "a", "questions": "问？", "news1": "太短"}]}
        assert build_queries(raw) == []

    def test_missing_answer_becomes_empty_string(self) -> None:
        raw = {"questanswer_1doc": [{"ID": "a", "questions": "问？", "news1": body("A")}]}
        assert build_queries(raw)[0].answer == ""

    def test_task_selection_is_respected(self) -> None:
        tasks = {q.task for q in build_queries(RAW, tasks=["questanswer_3docs"])}
        assert tasks == {"questanswer_3docs"}

    def test_rejects_a_query_with_no_gold(self) -> None:
        with pytest.raises(ValueError, match="at least one gold document"):
            Query("q", "问？", "答", (), "questanswer_1doc")


class TestSampleCorpus:
    @pytest.fixture
    def pool_and_queries(self) -> tuple[dict[str, str], list[Query]]:
        pool = harvest_documents(RAW)
        return pool, build_queries(RAW, pool=pool)

    def test_none_returns_the_whole_pool(
        self, pool_and_queries: tuple[dict[str, str], list[Query]]
    ) -> None:
        pool, queries = pool_and_queries
        assert sample_corpus(pool, queries) == pool

    def test_always_retains_every_gold_document(
        self, pool_and_queries: tuple[dict[str, str], list[Query]]
    ) -> None:
        pool, queries = pool_and_queries
        gold = {d for q in queries for d in q.gold_doc_ids}
        corpus = sample_corpus(pool, queries, size=len(gold))
        assert gold <= set(corpus)

    def test_honours_the_requested_size(
        self, pool_and_queries: tuple[dict[str, str], list[Query]]
    ) -> None:
        pool, queries = pool_and_queries
        gold = {d for q in queries for d in q.gold_doc_ids}
        assert len(sample_corpus(pool, queries, size=len(gold) + 1)) == len(gold) + 1

    def test_is_deterministic_under_a_fixed_seed(
        self, pool_and_queries: tuple[dict[str, str], list[Query]]
    ) -> None:
        pool, queries = pool_and_queries
        n = len({d for q in queries for d in q.gold_doc_ids}) + 1
        assert sample_corpus(pool, queries, size=n) == sample_corpus(pool, queries, size=n)

    def test_different_seeds_can_pick_different_distractors(self) -> None:
        pool = {document_id(body(t)): body(t) for t in "ABCDEFGHIJ"}
        q = [Query("q", "问？", "答", (document_id(body("A")),), "questanswer_1doc")]
        picks = {tuple(sorted(sample_corpus(pool, q, size=3, seed=s))) for s in range(12)}
        assert len(picks) > 1

    def test_rejects_a_size_below_the_gold_floor(
        self, pool_and_queries: tuple[dict[str, str], list[Query]]
    ) -> None:
        pool, queries = pool_and_queries
        gold = {d for q in queries for d in q.gold_doc_ids}
        with pytest.raises(ValueError, match="below the"):
            sample_corpus(pool, queries, size=len(gold) - 1)

    def test_rejects_gold_absent_from_the_pool(self) -> None:
        q = [Query("q", "问？", "答", ("nonexistent",), "questanswer_1doc")]
        with pytest.raises(ValueError, match="absent from the pool"):
            sample_corpus({document_id(body("A")): body("A")}, q)
