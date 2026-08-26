from __future__ import annotations

import json

import pytest

from zhrag.eval.pool import (
    GRADE_LABELS,
    JudgedQuery,
    PooledQuery,
    batched,
    build_pool,
    judging_cache_id,
    judging_input_fingerprint,
    judging_instructions_fingerprint,
    judging_order,
    judging_prompt,
    parse_judgements,
    pool_contribution,
    pool_fingerprint,
    qrels_rows,
)


def _reply(*grades: int) -> str:
    return json.dumps(
        {"judgements": [{"id": i, "grade": g} for i, g in enumerate(grades, start=1)]},
        ensure_ascii=False,
    )


class TestBuildPool:
    def test_unions_runs_and_records_every_contributor(self) -> None:
        candidates, contributors = build_pool(
            {"dense": ["a", "b", "c"], "sparse": ["c", "d"]},
            depth=2,
        )

        assert set(candidates) == {"a", "b", "c", "d"}
        assert contributors["c"] == ("sparse",)
        assert contributors["d"] == ("sparse",)
        assert "c" not in contributors or "c" in candidates

    def test_depth_truncates_each_run_independently(self) -> None:
        candidates, _ = build_pool({"dense": ["a", "b", "c"]}, depth=2)
        assert candidates == ("a", "b")

    def test_required_ids_join_the_pool_with_no_contributor(self) -> None:
        candidates, contributors = build_pool(
            {"dense": ["a"]},
            depth=5,
            required=["gold"],
        )
        assert set(candidates) == {"a", "gold"}
        assert contributors["gold"] == ()

    def test_order_is_deterministic_and_rank_major(self) -> None:
        runs = {"dense": ["z", "y"], "sparse": ["y", "x"]}
        first = build_pool(runs, depth=2)[0]
        assert first == build_pool(runs, depth=2)[0]
        # "y" is rank 1 in sparse, so it precedes "z" which is only ever rank 1
        # in dense -- ties fall back to the id.
        assert first[0] == "y"

    @pytest.mark.parametrize("depth", [0, -1])
    def test_rejects_a_non_positive_depth(self, depth: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_pool({"dense": ["a"]}, depth=depth)

    def test_rejects_an_empty_run_set(self) -> None:
        with pytest.raises(ValueError, match="at least one run"):
            build_pool({}, depth=5)

    def test_reports_single_system_pools(self) -> None:
        _, contributors = build_pool({"dense": ["a", "b"], "sparse": ["a"]}, depth=5)
        assert pool_contribution(contributors) == {
            "dense": 2,
            "exclusive_to_one_system": 1,
            "sparse": 1,
            "unique_candidates": 2,
        }


class TestJudgingOrder:
    def test_is_stable_seed_dependent_and_a_permutation(self) -> None:
        candidates = ("a", "b", "c", "d", "e")
        first = judging_order(candidates, seed="q1")
        assert first == judging_order(candidates, seed="q1")
        assert sorted(first) == sorted(candidates)
        assert first != judging_order(candidates, seed="q2")

    def test_decorrelates_position_from_retrieval_rank(self) -> None:
        """A pool judged in rank order would feed leniency drift into top ranks."""
        ranked = tuple(f"doc-{i:02d}" for i in range(20))
        assert judging_order(ranked, seed="q1") != ranked

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            judging_order(("a", "a"), seed="q1")


class TestBatching:
    def test_splits_without_dropping_or_reordering(self) -> None:
        assert batched(("a", "b", "c", "d", "e"), 2) == [("a", "b"), ("c", "d"), ("e",)]

    @pytest.mark.parametrize("size", [0, -3])
    def test_rejects_a_non_positive_size(self, size: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            batched(("a",), size)


class TestCacheIdentity:
    def test_binds_both_questions_answer_and_batch_contents(self) -> None:
        pair = ("直接问题", "改写问题")
        base = judging_cache_id(pair, "答案", ("a", "b"))
        assert base == judging_cache_id(pair, "答案", ("a", "b"))
        assert base != judging_cache_id(("另一个问题", "改写问题"), "答案", ("a", "b"))
        assert base != judging_cache_id(("直接问题", "另一个改写"), "答案", ("a", "b"))
        assert base != judging_cache_id(pair, "另一个答案", ("a", "b"))
        assert base != judging_cache_id(pair, "答案", ("a", "c"))
        assert base != judging_cache_id(pair, "答案", ("b", "a"))

    def test_rejects_empty_questions_or_an_invalid_batch(self) -> None:
        with pytest.raises(ValueError, match="questions"):
            judging_cache_id((), "答案", ("a",))
        with pytest.raises(ValueError, match="non-empty"):
            judging_cache_id(("问题",), "答案", ())
        with pytest.raises(ValueError, match="repeat"):
            judging_cache_id(("问题",), "答案", ("a", "a"))

    def test_instruction_fingerprint_is_a_full_digest(self) -> None:
        assert len(judging_instructions_fingerprint()) == 64


class TestJudgingInputFingerprint:
    def test_binds_exact_passage_text_and_batch_contract(self) -> None:
        unit = PooledQuery(
            chunk_id="gold",
            query_ids=("direct:1", "paraphrase:1"),
            questions=("如何备份？", "怎样进行备份？"),
            answer="使用 BR。",
            candidates=("gold", "other"),
            contributors={"gold": (), "other": ("dense",)},
        )
        base = judging_input_fingerprint(
            [unit], corpus={"gold": "第一段", "other": "第二段"}, seed="s", batch_size=2
        )
        assert base == judging_input_fingerprint(
            [unit], corpus={"gold": "第一段", "other": "第二段"}, seed="s", batch_size=2
        )
        assert base != judging_input_fingerprint(
            [unit], corpus={"gold": "改过的第一段", "other": "第二段"}, seed="s", batch_size=2
        )
        assert base != judging_input_fingerprint(
            [unit], corpus={"gold": "第一段", "other": "第二段"}, seed="other", batch_size=2
        )


class TestPrompt:
    def test_numbers_every_passage_and_hides_chunk_ids(self) -> None:
        prompt = judging_prompt(
            ("如何备份？", "怎样进行备份？"),
            "使用 BR。",
            ("chunk-a", "chunk-b"),
            {"chunk-a": "第一段", "chunk-b": "第二段"},
        )
        assert "问题1：如何备份？" in prompt
        assert "问题2：怎样进行备份？" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "第一段" in prompt
        assert "chunk-a" not in prompt

    def test_fails_when_a_pooled_candidate_has_no_text(self) -> None:
        with pytest.raises(KeyError, match="chunk-b"):
            judging_prompt(("问题？",), "答案。", ("chunk-b",), {})


class TestParseJudgements:
    def test_maps_numbers_back_onto_chunk_ids(self) -> None:
        assert parse_judgements(_reply(0, 2), ("a", "b")) == {"a": 0, "b": 2}

    def test_tolerates_one_json_code_fence(self) -> None:
        assert parse_judgements(f"```json\n{_reply(1)}\n```", ("a",)) == {"a": 1}

    def test_rejects_prose_around_the_object(self) -> None:
        with pytest.raises(ValueError, match="malformed JSON"):
            parse_judgements(f"好的\n{_reply(1)}\n", ("a",))

    @pytest.mark.parametrize(
        ("reply", "reason"),
        [
            ("not json", "malformed JSON"),
            (json.dumps({"nope": []}), "no judgements array"),
            (json.dumps({"judgements": [{"id": 1, "grade": 3}]}), "outside"),
            (json.dumps({"judgements": [{"id": 1, "grade": True}]}), "must be an integer"),
            (json.dumps({"judgements": [{"id": "1", "grade": 0}]}), "must be an integer"),
            (
                json.dumps({"judgements": [{"id": 1, "grade": 0}, {"id": 1, "grade": 2}]}),
                "repeated",
            ),
        ],
    )
    def test_rejects_malformed_replies(self, reply: str, reason: str) -> None:
        with pytest.raises(ValueError, match=reason):
            parse_judgements(reply, ("a", "b"))

    def test_rejects_a_short_reply_rather_than_implying_zeroes(self) -> None:
        """Silently zero-filling a skipped passage would depress every system."""
        with pytest.raises(ValueError, match=r"missing=\[2, 3\]"):
            parse_judgements(_reply(2), ("a", "b", "c"))

    def test_rejects_grades_for_passages_that_were_not_sent(self) -> None:
        with pytest.raises(ValueError, match=r"extra=\[3\]"):
            parse_judgements(
                json.dumps({"judgements": [{"id": 1, "grade": 0}, {"id": 3, "grade": 2}]}),
                ("a", "b"),
            )


class TestPooledQuery:
    def _pooled(self, **overrides: object) -> PooledQuery:
        kwargs: dict[str, object] = {
            "chunk_id": "gold",
            "query_ids": ("direct:1", "paraphrase:1"),
            "questions": ("如何备份？", "怎样进行备份？"),
            "answer": "使用 BR。",
            "candidates": ("gold", "other"),
            "contributors": {"gold": (), "other": ("dense",)},
        }
        kwargs.update(overrides)
        return PooledQuery(**kwargs)  # type: ignore[arg-type]

    def test_accepts_a_well_formed_unit(self) -> None:
        assert self._pooled().candidates == ("gold", "other")

    def test_requires_the_generating_chunk_in_the_pool(self) -> None:
        with pytest.raises(ValueError, match="must be pooled"):
            self._pooled(candidates=("other",), contributors={"other": ("dense",)})

    def test_requires_contributor_keys_for_every_candidate(self) -> None:
        with pytest.raises(ValueError, match="keys differ"):
            self._pooled(contributors={"gold": ()})

    def test_freezes_the_contributor_mapping(self) -> None:
        contributors = {"gold": (), "other": ("dense",)}
        pooled = self._pooled(contributors=contributors)
        contributors["other"] = ("sparse",)
        assert pooled.contributors["other"] == ("dense",)

    def test_pool_fingerprint_binds_questions_candidates_and_contributors(self) -> None:
        base = self._pooled()
        changed_question = self._pooled(questions=("另一个问题", "怎样进行备份？"))
        changed_candidate = self._pooled(
            candidates=("gold", "third"),
            contributors={"gold": (), "third": ("dense",)},
        )
        assert pool_fingerprint([base]) == pool_fingerprint([base])
        assert pool_fingerprint([base]) != pool_fingerprint([changed_question])
        assert pool_fingerprint([base]) != pool_fingerprint([changed_candidate])

    def test_rejects_duplicate_candidates_and_query_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate pooled"):
            self._pooled(candidates=("gold", "other", "other"))
        with pytest.raises(ValueError, match="duplicate query ids"):
            self._pooled(query_ids=("direct:1", "direct:1"))

    def test_requires_one_question_per_query_id(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            self._pooled(questions=("只有一个问题",))

    def test_requires_exactly_one_direct_paraphrase_pair(self) -> None:
        with pytest.raises(ValueError, match="requires one"):
            self._pooled(
                query_ids=("direct:1",),
                questions=("只有一个问题",),
            )
        with pytest.raises(ValueError, match="not a direct/paraphrase"):
            self._pooled(query_ids=("direct:1", "direct:2"))


class TestJudgedQuery:
    def _judged(self, grades: dict[str, int]) -> JudgedQuery:
        return JudgedQuery(
            chunk_id="gold",
            query_ids=("direct:1", "paraphrase:2"),
            grades=grades,
        )

    def test_splits_full_partial_and_judged_sets(self) -> None:
        judged = self._judged({"gold": 2, "a": 2, "b": 1, "c": 0})
        assert judged.gold_doc_ids == ("a", "gold")
        assert judged.partial_doc_ids == ("b",)
        assert judged.judged_doc_ids == ("a", "b", "c", "gold")

    def test_retains_the_verified_generating_chunk_but_records_disagreement(self) -> None:
        judged = self._judged({"gold": 1, "a": 2})
        assert "gold" in judged.gold_doc_ids
        assert judged.generator_chunk_grade == 1
        assert "gold" not in judged.partial_doc_ids

    def test_requires_the_generating_chunk_to_be_judged(self) -> None:
        with pytest.raises(ValueError, match="was not judged"):
            self._judged({"a": 2})

    @pytest.mark.parametrize("grade", [3, -1])
    def test_rejects_out_of_scale_grades(self, grade: int) -> None:
        with pytest.raises(ValueError, match="outside"):
            self._judged({"gold": grade})

    def test_grade_labels_cover_the_whole_scale(self) -> None:
        assert sorted(GRADE_LABELS) == [0, 1, 2]

    def test_freezes_the_grade_mapping(self) -> None:
        grades = {"gold": 2}
        judged = self._judged(grades)
        grades["gold"] = 0
        assert judged.generator_chunk_grade == 2


class TestQrelsRows:
    def test_both_variants_inherit_one_judgement_set(self) -> None:
        judged = JudgedQuery(
            chunk_id="gold",
            query_ids=("direct:1", "paraphrase:1"),
            grades={"gold": 2, "a": 2, "b": 1},
        )
        queries = {
            "direct:1": {"query_id": "direct:1", "question": "直接问题", "task": "direct"},
            "paraphrase:1": {
                "query_id": "paraphrase:1",
                "question": "改写问题",
                "task": "paraphrase",
            },
        }

        rows = qrels_rows([judged], queries=queries)

        assert [row["query_id"] for row in rows] == ["direct:1", "paraphrase:1"]
        assert all(row["gold_doc_ids"] == ["a", "gold"] for row in rows)
        assert all(row["partial_doc_ids"] == ["b"] for row in rows)
        assert all(row["generating_chunk_id"] == "gold" for row in rows)
        assert [row["question"] for row in rows] == ["直接问题", "改写问题"]

    def test_does_not_mutate_the_source_query(self) -> None:
        source = {"query_id": "direct:1", "gold_doc_ids": ["gold"]}
        queries = {
            "direct:1": source,
            "paraphrase:2": {"query_id": "paraphrase:2", "gold_doc_ids": ["gold"]},
        }
        judged = JudgedQuery(
            chunk_id="gold",
            query_ids=("direct:1", "paraphrase:2"),
            grades={"gold": 2, "a": 2},
        )

        qrels_rows([judged], queries=queries)

        assert source["gold_doc_ids"] == ["gold"]

    def test_fails_on_an_unknown_query_id(self) -> None:
        judged = JudgedQuery(
            chunk_id="gold",
            query_ids=("direct:missing", "paraphrase:missing"),
            grades={"gold": 2},
        )
        with pytest.raises(KeyError, match="direct:missing"):
            qrels_rows([judged], queries={})
