from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from test_agent import ABSTAIN, ANSWER, READ, SEARCH, make_agent
from test_answering import ranked
from zhrag.agent import AgentSettings
from zhrag.agent_answer_review import REVIEW_PROMPT, parse_review
from zhrag.answering import AnswerBlock, Evidence, GenerationError


def feedback(
    *, covered: bool = True, supported: bool = True, scope: bool = True, ids: tuple[int, ...] = (1,)
) -> dict[str, Any]:
    return {
        "coverage": [
            {"requirement": "合成问题的原因与操作", "covered": covered, "evidence_ids": list(ids)}
        ],
        "blocks": [
            {
                "block": 1,
                "supported": supported,
                "condition_scope_supported": scope,
                "evidence_ids": list(ids),
                "finding": "" if supported and scope else "合成缺陷：不能排除手动启动这一例外。",
            }
        ],
        "suggested_queries": [] if covered else ["合成引擎 为什么可以启动"],
    }


def test_draft_needs_review_before_publication_and_review_has_no_unread_candidates() -> None:
    agent, generator = make_agent(
        SEARCH, READ, ANSWER, feedback(), settings=AgentSettings(review_answers=True)
    )
    outcome = agent.run("合成问题")
    assert outcome.status == "answered" and outcome.model_calls == 4
    review_input = generator.calls[-1]
    assert set(review_input) == {"question", "draft_blocks", "evidence"}
    assert review_input["draft_blocks"][0]["text"] == "合成回答"
    assert review_input["evidence"][0]["id"] == 1
    assert ("answer", "drafted") in [(e.action, e.outcome) for e in outcome.events]
    assert ("review_answer", "accepted") in [(e.action, e.outcome) for e in outcome.events]


def test_missing_mechanism_feedback_drives_read_and_second_review() -> None:
    repaired = {
        "action": "answer",
        "answer": {"answerable": True, "blocks": [{"text": "合成原因与操作", "citations": [1, 2]}]},
    }
    agent, generator = make_agent(
        SEARCH,
        READ,
        ANSWER,
        feedback(covered=False),
        {"action": "search_docs", "query": "合成原因"},
        {"action": "read_passage", "evidence_id": 2},
        repaired,
        feedback(ids=(1, 2)),
        settings=AgentSettings(review_answers=True),
    )
    outcome = agent.run("合成装置为什么能够启动，应该怎样操作？")
    assert outcome.status == "answered" and outcome.blocks[0].text == "合成原因与操作"
    assert outcome.model_calls == 8 and outcome.search_calls == outcome.read_calls == 2
    assert generator.calls[4]["answer_review"]["coverage"][0]["covered"] is False
    assert len(generator.calls[-1]["evidence"]) == 2
    assert sum(e.outcome == "revise" for e in outcome.events) == 1


def test_scope_failure_requires_rewriting_even_when_claim_flag_is_true() -> None:
    narrow = {
        "action": "answer",
        "answer": {
            "answerable": True,
            "blocks": [{"text": "合成：只有蓝灯亮才能启动。", "citations": [1]}],
        },
    }
    repaired = {
        "action": "answer",
        "answer": {
            "answerable": True,
            "blocks": [{"text": "合成：蓝灯亮时可启动，手动启动也可用。", "citations": [1]}],
        },
    }
    agent, generator = make_agent(
        SEARCH,
        READ,
        narrow,
        feedback(scope=False),
        repaired,
        feedback(),
        settings=AgentSettings(review_answers=True),
    )
    outcome = agent.run("合成装置有哪些启动方式？")
    assert outcome.status == "answered" and "手动" in outcome.blocks[0].text
    assert generator.calls[4]["answer_review"]["blocks"][0]["condition_scope_supported"] is False
    assert "手动启动" not in json.dumps([e.outcome for e in outcome.events], ensure_ascii=False)


def test_second_failed_review_stops_without_a_third_repair() -> None:
    agent, generator = make_agent(
        SEARCH,
        READ,
        ANSWER,
        feedback(scope=False),
        ANSWER,
        feedback(covered=False),
        ANSWER,
        settings=AgentSettings(review_answers=True),
    )
    outcome = agent.run("合成问题")
    assert outcome.status == "invalid_answer" and not outcome.blocks
    assert outcome.model_calls == 6 and len(generator.actions) == 1
    assert any(e.outcome == "rejected" for e in outcome.events)


@pytest.mark.parametrize(
    "action,status",
    [
        (ABSTAIN, "insufficient_evidence"),
        (
            {"action": "answer", "answer": {"answerable": False, "blocks": []}},
            "insufficient_evidence",
        ),
        ({"action": "clarify", "question": "合成装置现在是哪种模式？"}, "clarification_needed"),
    ],
)
def test_nonanswers_do_not_require_review(action: object, status: str) -> None:
    agent, _ = make_agent(action, settings=AgentSettings(review_answers=True))
    outcome = agent.run("合成问题")
    assert outcome.status == status and outcome.model_calls == 1


def test_feedback_suggested_queries_are_data_not_automatically_executed() -> None:
    rejected = feedback(covered=False)
    agent, generator = make_agent(
        SEARCH,
        READ,
        ANSWER,
        rejected,
        {"action": "clarify", "question": "请说明合成模式。"},
        settings=AgentSettings(review_answers=True),
    )
    outcome = agent.run("合成问题")
    assert outcome.status == "clarification_needed" and outcome.search_calls == 1
    assert (
        generator.calls[-1]["answer_review"]["suggested_queries"] == rejected["suggested_queries"]
    )


@pytest.mark.parametrize(
    "failure,status",
    [
        (GenerationError("generation_timeout"), "generation_timeout"),
        (RuntimeError("private-review-payload"), "generation_failed"),
        ("{}", "invalid_answer"),
    ],
)
def test_review_failures_never_release_draft(failure: object, status: str) -> None:
    agent, _ = make_agent(
        SEARCH, READ, ANSWER, failure, settings=AgentSettings(review_answers=True)
    )
    outcome = agent.run("合成问题")
    assert outcome.status == status and not outcome.blocks and outcome.evidence
    assert "private-review-payload" not in repr(outcome)


def test_review_consumes_decision_and_cumulative_token_budgets() -> None:
    agent, generator = make_agent(
        SEARCH, READ, ANSWER, feedback(), settings=AgentSettings(review_answers=True, max_steps=3)
    )
    result = agent.run("合成问题")
    assert result.status == "budget_exhausted" and result.model_calls == 3
    assert not result.blocks and len(generator.actions) == 1
    reference, _ = make_agent(
        SEARCH, READ, ANSWER, feedback(), settings=AgentSettings(review_answers=True)
    )
    # Elapsed floats enter observations; pin the clock for an exact token boundary.
    total = replace(reference, clock=lambda: 0.0).run("合成问题").prompt_estimated_tokens
    agent, generator = make_agent(
        SEARCH,
        READ,
        ANSWER,
        feedback(),
        settings=AgentSettings(review_answers=True, max_total_prompt_tokens=total - 1),
    )
    result = replace(agent, clock=lambda: 0.0).run("合成问题")
    assert result.status == "budget_exhausted" and result.model_calls == 3
    assert result.prompt_estimated_tokens < total and not result.blocks


def test_review_prompt_size_is_checked_before_its_call() -> None:
    draft = {
        "action": "answer",
        "answer": {"answerable": True, "blocks": [{"text": "合成" * 2000, "citations": [1]}]},
    }
    agent, generator = make_agent(
        SEARCH,
        READ,
        draft,
        feedback(),
        settings=AgentSettings(review_answers=True, max_prompt_chars=5000),
    )
    result = agent.run("合成问题")
    assert result.status == "budget_exhausted" and result.model_calls == 3
    assert len(generator.actions) == 1 and not result.blocks


@pytest.mark.parametrize("cancel", [False, True])
def test_time_or_cancellation_during_review_prevents_publication(cancel: bool) -> None:
    now = [0.0]
    stop = threading.Event()
    agent, generator = make_agent(
        SEARCH, READ, ANSWER, feedback(), settings=AgentSettings(review_answers=True)
    )
    original = generator.generate

    def interrupt(system: str, user: str) -> str:
        reply = original(system, user)
        if system == REVIEW_PROMPT:
            stop.set() if cancel else now.__setitem__(0, 181.0)
        return reply

    generator.generate = interrupt
    result = replace(agent, clock=lambda: now[0]).run("合成问题", cancelled=stop.is_set)
    assert result.status == ("cancelled" if cancel else "budget_exhausted")
    assert result.model_calls == 4 and not result.blocks


def test_repair_state_is_request_local_and_new_contract_changes_legacy_profile() -> None:
    agent, generator = make_agent(
        SEARCH,
        READ,
        ANSWER,
        feedback(scope=False),
        ANSWER,
        feedback(),
        ABSTAIN,
        settings=AgentSettings(review_answers=True),
    )
    assert agent.run("合成问题").status == "answered"
    assert agent.run("另一问题").status == "insufficient_evidence"
    assert "answer_review" not in generator.calls[-1]
    original, _ = make_agent(SEARCH, READ, ANSWER)
    # The v3 wire guidance and diagnostics intentionally differ from 3503741.
    assert (
        original.profile_fingerprint
        != "2a38d4d76071f6f767e4927b6c5d9642ba4f1fecca3133bee2815070ff92329e"
    )
    assert (
        replace(agent, settings=replace(agent.settings, review_answers=False)).profile_fingerprint
        == original.profile_fingerprint
    )
    assert original.run("合成问题").model_calls == 3
    assert agent.profile_fingerprint != original.profile_fingerprint


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_review_switch_must_be_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="review_answers"):
        AgentSettings(review_answers=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_coverage",
        "empty_blocks",
        "duplicate_block",
        "missing_block",
        "extra_key",
        "boolean_id",
        "uncited_id",
        "unread_id",
        "empty_ids",
        "string_label",
        "empty_finding",
        "duplicate_coverage",
        "duplicate_ids",
        "too_many_queries",
        "surrogate",
        "uncited_coverage",
    ],
)
def test_review_schema_rejects_false_certification(mutation: str) -> None:
    obj = deepcopy(feedback())
    blocks = (AnswerBlock("合成内容", (1,)),)
    evidence = (Evidence(1, ranked("合成证据"), ""), Evidence(2, ranked("未引用证据"), ""))
    if mutation == "empty_coverage":
        obj["coverage"] = []
    elif mutation == "empty_blocks":
        obj["blocks"] = []
    elif mutation == "duplicate_block":
        obj["blocks"] *= 2
    elif mutation == "missing_block":
        blocks += (AnswerBlock("另一段", (2,)),)
    elif mutation == "extra_key":
        obj["accept"] = True
    elif mutation in {"boolean_id", "uncited_id", "unread_id", "empty_ids", "duplicate_ids"}:
        obj["blocks"][0]["evidence_ids"] = {
            "boolean_id": [True],
            "uncited_id": [2],
            "unread_id": [99],
            "empty_ids": [],
            "duplicate_ids": [1, 1],
        }[mutation]
    elif mutation == "string_label":
        obj["blocks"][0]["supported"] = "true"
    elif mutation == "empty_finding":
        obj["blocks"][0]["condition_scope_supported"] = False
    elif mutation == "duplicate_coverage":
        obj["coverage"] *= 2
    elif mutation == "too_many_queries":
        obj["suggested_queries"] = ["a", "b", "c"]
    elif mutation == "surrogate":
        obj["coverage"][0]["requirement"] = "\ud800"
    elif mutation == "uncited_coverage":
        obj["coverage"][0]["evidence_ids"] = [2]
    with pytest.raises(ValueError, match="invalid_answer_review"):
        parse_review(json.dumps(obj), blocks, evidence)


@pytest.mark.parametrize(
    "raw",
    [
        '{"coverage":[],"coverage":[],"blocks":[],"suggested_queries":[]}',
        '{"coverage":NaN,"blocks":[],"suggested_queries":[]}',
        "[" * 2000,
        " " * 24001,
    ],
)
def test_malformed_review_json_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid_answer_review"):
        parse_review(raw, (AnswerBlock("合成内容", (1,)),), (Evidence(1, ranked("合成证据"), ""),))
