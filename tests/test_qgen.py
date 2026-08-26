from __future__ import annotations

import json
from typing import Any

import pytest

from zhrag.eval.qgen import (
    EvalChunk,
    GeneratedQuery,
    bigram_containment,
    dedupe_questions,
    generation_instructions_fingerprint,
    instructions_fingerprint,
    parse_generation,
    parse_verdict,
    query_id,
    retain_complete_pairs,
    stratified_sample,
    theme_counts,
    verification_cache_id,
    verification_instructions_fingerprint,
    verification_prompt,
)


def _chunk(chunk_id: str, theme: str = "sql") -> EvalChunk:
    return EvalChunk(
        chunk_id=chunk_id,
        source_key=f"pingcap/docs-cn:{chunk_id}.md",
        ordinal=0,
        collection="manual",
        theme=theme,
        text="TiDB 的事务隔离级别可以通过配置进行调整。",
        approx_tokens=20,
    )


def _reply(
    *,
    question: str = "如何调整 TiDB 的事务隔离级别？",
    paraphrase: str = "TiDB 事务隔离配置应怎样修改？",
    answer: str = "通过相应配置调整。",
    question_type: str = "config",
    usable: bool = True,
) -> str:
    return json.dumps(
        {
            "usable": usable,
            "reason": "",
            "question": question,
            "paraphrase": paraphrase,
            "answer": answer,
            "question_type": question_type,
        },
        ensure_ascii=False,
    )


class TestQueryIdentity:
    def test_is_stable_and_variant_specific(self) -> None:
        assert query_id("chunk-a", "direct") == query_id("chunk-a", "direct")
        assert query_id("chunk-a", "direct") != query_id("chunk-a", "paraphrase")
        with pytest.raises(ValueError, match="unknown variant"):
            query_id("chunk-a", "other")

    def test_instruction_fingerprints_are_stage_specific(self) -> None:
        fingerprints = {
            instructions_fingerprint(),
            generation_instructions_fingerprint(),
            verification_instructions_fingerprint(),
        }
        assert len(fingerprints) == 3
        assert all(len(value) == 64 for value in fingerprints)

    def test_verification_cache_identity_binds_pair_content(self) -> None:
        direct, paraphrase = parse_generation(_chunk("a"), _reply()).queries
        changed_paraphrase = GeneratedQuery(
            paraphrase.query_id,
            paraphrase.chunk_id,
            paraphrase.variant,
            "TiDB 的事务隔离设置位于哪里？",
            paraphrase.answer,
            paraphrase.question_type,
        )
        changed_answer = GeneratedQuery(
            direct.query_id,
            direct.chunk_id,
            direct.variant,
            direct.question,
            "使用系统变量调整。",
            direct.question_type,
        )
        assert verification_cache_id(direct, paraphrase) == verification_cache_id(
            direct,
            paraphrase,
        )
        assert verification_cache_id(direct, paraphrase) != verification_cache_id(
            direct,
            changed_paraphrase,
        )
        with pytest.raises(ValueError, match="share one answer"):
            verification_cache_id(changed_answer, paraphrase)


class TestSampling:
    def test_is_deterministic_and_balances_themes(self) -> None:
        chunks = tuple(
            [_chunk(f"sql-{i}", "sql") for i in range(8)]
            + [_chunk(f"br-{i}", "br") for i in range(2)]
        )
        first = stratified_sample(chunks, size=5, seed="seed")
        assert first == stratified_sample(chunks, size=5, seed="seed")
        assert theme_counts(first) == {"br": 1, "sql": 4}
        assert [row.source_key for row in first] == sorted(row.source_key for row in first)

    @pytest.mark.parametrize("size", [0, -1, 11])
    def test_rejects_invalid_sizes(self, size: int) -> None:
        with pytest.raises(ValueError, match=r"sample|positive"):
            stratified_sample((_chunk("a"),), size=size, seed="seed")


class TestGenerationParsing:
    def test_accepts_a_pair_and_derives_ids(self) -> None:
        outcome = parse_generation(_chunk("a"), _reply())
        assert outcome.rejection is None
        assert [query.variant for query in outcome.queries] == ["direct", "paraphrase"]
        assert all(query.query_id == query_id("a", query.variant) for query in outcome.queries)

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ("not json", "malformed JSON"),
            (json.dumps({"usable": False, "reason": "目录"}), "declined:目录"),
            (_reply(question="本文如何配置事务？"), "self-reference"),
            (_reply(paraphrase="如何调整 TiDB 的事务隔离级别？"), "repeats"),
            (_reply(question_type="other"), "unknown question_type"),
        ],
    )
    def test_turns_bad_replies_into_reportable_rejections(self, payload: str, reason: str) -> None:
        outcome = parse_generation(_chunk("a"), payload)
        assert not outcome.queries
        assert outcome.rejection is not None
        assert reason in outcome.rejection

    def test_rejects_non_object_json(self) -> None:
        outcome = parse_generation(_chunk("a"), "[]")
        assert outcome.rejection == "invalid:reply is not a JSON object"


class TestVerificationParsing:
    def test_accepts_all_boolean_axes(self) -> None:
        verdict = parse_verdict(
            json.dumps(
                {
                    "answerable": True,
                    "self_contained": True,
                    "specific": True,
                    "leaks_answer": False,
                    "answer_supported": True,
                    "same_intent": True,
                    "note": "ok",
                }
            )
        )
        assert verdict.keep
        assert verdict.note == "ok"

    @pytest.mark.parametrize("value", ["true", 1, None])
    def test_rejects_non_boolean_axes(self, value: Any) -> None:
        payload = {
            "answerable": True,
            "self_contained": True,
            "specific": True,
            "leaks_answer": False,
            "answer_supported": True,
            "same_intent": True,
        }
        payload["specific"] = value
        with pytest.raises(ValueError, match="specific"):
            parse_verdict(json.dumps(payload))

    def test_keep_is_false_when_pair_has_different_intent(self) -> None:
        verdict = parse_verdict(
            json.dumps(
                {
                    "answerable": True,
                    "self_contained": True,
                    "specific": True,
                    "leaks_answer": False,
                    "answer_supported": True,
                    "same_intent": False,
                }
            )
        )
        assert not verdict.keep
        assert "same_intent" in verdict.failures()

    def test_keep_is_false_when_answer_is_not_supported(self) -> None:
        verdict = parse_verdict(
            json.dumps(
                {
                    "answerable": True,
                    "self_contained": True,
                    "specific": True,
                    "leaks_answer": False,
                    "answer_supported": False,
                    "same_intent": True,
                }
            )
        )
        assert not verdict.keep
        assert "answer_supported" in verdict.failures()


class TestOverlapAndDedupe:
    def test_containment_uses_query_as_the_denominator(self) -> None:
        assert bigram_containment("事务隔离", "TiDB 的事务隔离级别") == 1.0
        assert bigram_containment("事务隔离", "完全不同") == 0.0

    def test_drops_a_collision_with_its_paraphrase_twin(self) -> None:
        def make_query(chunk_id: str, variant: str, question: str) -> GeneratedQuery:
            return GeneratedQuery(
                query_id=chunk_id + variant,
                chunk_id=chunk_id,
                variant=variant,
                question=question,
                answer="答案",
                question_type="concept",
            )

        rows = [
            make_query("a", "direct", "如何配置事务隔离级别"),
            make_query("a", "paraphrase", "事务隔离级别怎样配置"),
            make_query("b", "direct", "如何配置事务隔离级别"),
            make_query("b", "paraphrase", "事务隔离级别怎样配置"),
        ]
        kept, dropped = dedupe_questions(rows, threshold=0.8)
        assert {row.chunk_id for row in kept} == {"a"}
        assert dropped == (("b", "a"),)

    def test_requires_a_direct_question(self) -> None:
        row = GeneratedQuery("q", "a", "paraphrase", "一个改写问题", "答案", "concept")
        with pytest.raises(ValueError, match="complete query pair"):
            dedupe_questions([row])

    def test_verification_prompt_contains_both_questions(self) -> None:
        outcome = parse_generation(_chunk("a"), _reply())
        direct, paraphrase = outcome.queries
        prompt = verification_prompt(_chunk("a"), direct, paraphrase)
        assert direct.question in prompt
        assert paraphrase.question in prompt
        assert direct.answer in prompt

    def test_retains_only_complete_pairs(self) -> None:
        direct = GeneratedQuery("d", "a", "direct", "一个直接问题", "答案", "concept")
        twin = GeneratedQuery("p", "a", "paraphrase", "一个改写问题", "答案", "concept")
        orphan = GeneratedQuery("o", "b", "direct", "另一个直接问题", "答案", "concept")
        kept, dropped = retain_complete_pairs(
            [direct, twin, orphan],
            required_chunk_ids={"a", "b", "c"},
        )
        assert kept == (direct, twin)
        assert dropped == ("b", "c")
