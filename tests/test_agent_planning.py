"""Synthetic planning, grounded coverage, routing and shared-budget regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from test_agent import ABSTAIN, READ, SEARCH, InvestigationRetriever, make_agent
from test_agent_answer_review import feedback
from test_answering import ranked
from zhrag.agent import AgentSettings, PlanningEvent
from zhrag.agent_planning import (
    PLANNING_ERROR_CODES,
    PlanningValidationError,
    answer_gate,
    parse_assessment,
    parse_plan,
)
from zhrag.answering import AnswerBlock, Evidence
from zhrag.retrieval.online import OnlineRetrievalResult
from zhrag.service.app import create_app

QUESTION = "合成设备如何启动，为什么要锁定飞轮，有什么风险？当前使用自动模式。"
PLAN = {
    "requirements": [
        {"id": 1, "kind": "procedure", "question_quote": "如何启动", "description": "启动操作"},
        {
            "id": 2,
            "kind": "mechanism",
            "question_quote": "为什么要锁定飞轮",
            "description": "锁定机制",
        },
        {"id": 3, "kind": "risk", "question_quote": "有什么风险", "description": "未锁定的后果"},
    ],
    "facts": [{"id": 1, "quote": "当前使用自动模式。"}],
}
TEXTS = (
    "有电池时可自动启动，也可用手柄启动。",
    "锁定飞轮能阻止回转。未锁定就检修可能受伤。选择启动配额前必须确认控制模式。",
)
QUOTES = (
    (1, "有电池时可自动启动，也可用手柄启动。"),
    (2, "锁定飞轮能阻止回转。"),
    (2, "未锁定就检修可能受伤。"),
)
ANSWER = {
    "action": "answer",
    "answer": {
        "answerable": True,
        "blocks": [
            {"text": "有电池时可以自动启动，也可用手柄启动。", "citations": [1]},
            {"text": "锁定飞轮会阻止回转。", "citations": [2]},
            {"text": "未锁定就检修可能受伤。", "citations": [2]},
        ],
    },
}


class PlanningRetriever(InvestigationRetriever):
    def retrieve(self, query: str) -> OnlineRetrievalResult:
        return replace(
            super().retrieve(query),
            passages=tuple(ranked(text, rank=i, doc_id=str(i)) for i, text in enumerate(TEXTS, 1)),
        )


def decision(
    action: object, covered: tuple[int, ...] = (), dependencies: list[object] | None = None
) -> dict[str, Any]:
    answering = isinstance(action, dict) and action.get("action") == "answer"
    return {
        "coverage": [
            {
                "requirement_id": i,
                "covered": i in covered,
                "evidence": [{"evidence_id": QUOTES[i - 1][0], "quote": QUOTES[i - 1][1]}]
                if i in covered
                else [],
                "blocks": [i] if answering and i in covered else [],
            }
            for i in range(1, 4)
        ],
        "dependencies": dependencies or [],
        "action": action,
    }


def evidence() -> tuple[Evidence, ...]:
    return tuple(
        Evidence(i, ranked(text, doc_id=str(i)), "合成") for i, text in enumerate(TEXTS, 1)
    )


def test_incomplete_answer_is_blocked_until_mechanism_and_risk_are_read() -> None:
    early = {
        "action": "answer",
        "answer": {"answerable": True, "blocks": ANSWER["answer"]["blocks"][:1]},
    }
    agent, generator = make_agent(
        PLAN,
        decision(SEARCH),
        decision(READ),
        decision(early, (1,)),
        decision({"action": "read_passage", "evidence_id": 2}, (1,)),
        decision(ANSWER, (1, 2, 3)),
        settings=AgentSettings(plan_investigation=True),
        fake=PlanningRetriever(),
    )
    result = agent.run(QUESTION)
    assert result.status == "answered" and result.model_calls == 6
    assert result.search_calls == 1 and result.read_calls == 2
    assert [block.citations for block in result.blocks] == [(1,), (2,), (2,)]
    assert any(e.outcome == "coverage_incomplete" for e in result.events)
    assert generator.calls[0] == {"question": QUESTION}
    assert generator.calls[4]["planning_feedback"] == "coverage_incomplete"
    assert generator.calls[4]["plan"] == generator.calls[-1]["plan"]
    assert len(generator.calls[-1]["evidence"]) == 2
    assert all(
        not {"assessment", "plan", "error_code"} & e.keys()
        for e in generator.calls[-1]["observations"]
    )
    audits = [e for e in result.events if isinstance(e, PlanningEvent)]
    assert audits[0].plan == parse_plan(json.dumps(PLAN), QUESTION)
    assert audits[-1].assessment.coverage[-1].evidence[0].quote == QUOTES[-1][1]


@pytest.mark.parametrize(
    "ignored_action", [ANSWER, {"action": "search_docs", "query": "unnecessary"}]
)
def test_declared_missing_environment_routes_to_clarify_before_action(
    ignored_action: object,
) -> None:
    dependency = {
        "name": "实际控制模式",
        "requirement_ids": [1],
        "evidence": [{"evidence_id": 2, "quote": "选择启动配额前必须确认控制模式。"}],
        "user_fact_ids": [],
        "question": "实际使用自动模式还是人工模式？",
    }
    plan = {**PLAN, "facts": []}
    question = QUESTION.removesuffix("当前使用自动模式。")
    agent, generator = make_agent(
        plan,
        decision(SEARCH),
        decision({"action": "read_passage", "evidence_id": 2}),
        decision(ignored_action, (2, 3), [dependency]),
        settings=AgentSettings(plan_investigation=True, review_answers=True),
        fake=PlanningRetriever(),
    )
    result = agent.run(question)
    assert (
        result.status == "clarification_needed" and result.clarification == dependency["question"]
    )
    assert not result.blocks and result.search_calls == 1 and len(generator.calls) == 4
    assert not any(e.action == "review_answer" for e in result.events)
    assert [e for e in result.events if isinstance(e, PlanningEvent)][
        -1
    ].proposed_action == ignored_action["action"]


def test_known_dependency_uses_anchored_fact_and_does_not_ask_again() -> None:
    dependency = {
        "name": "实际控制模式",
        "requirement_ids": [1],
        "evidence": [{"evidence_id": 2, "quote": "选择启动配额前必须确认控制模式。"}],
        "user_fact_ids": [1],
        "question": "",
    }
    agent, _ = make_agent(
        PLAN,
        decision(SEARCH),
        decision(READ),
        decision({"action": "read_passage", "evidence_id": 2}, (1,)),
        decision(ANSWER, (1, 2, 3), [dependency]),
        settings=AgentSettings(plan_investigation=True),
        fake=PlanningRetriever(),
    )
    result = agent.run(QUESTION)
    assert result.status == "answered" and not result.clarification
    assert [e for e in result.events if isinstance(e, PlanningEvent)][-1].assessment.dependencies[
        0
    ].user_fact_ids == (1,)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_quote",
        "unknown_fact",
        "boolean_id",
        "missing_requirement",
        "duplicate_requirement",
        "extra_field",
    ],
)
def test_plan_rejects_unanchored_or_incomplete_schema(mutation: str) -> None:
    plan = deepcopy(PLAN)
    if mutation == "wrong_quote":
        plan["requirements"][0]["question_quote"] = "PRIVATE_INVENTED_FACT"
    elif mutation == "unknown_fact":
        plan["facts"][0]["quote"] = "PRIVATE_INVENTED_FACT"
    elif mutation == "boolean_id":
        plan["facts"][0]["id"] = True
    elif mutation == "missing_requirement":
        plan["requirements"] = []
    elif mutation == "duplicate_requirement":
        plan["requirements"].append(plan["requirements"][0])
    else:
        plan["PRIVATE_EXTRA_KEY"] = "PRIVATE_INVENTED_FACT"
    with pytest.raises(ValueError, match=r"^invalid_investigation_plan$"):
        parse_plan(json.dumps(plan), QUESTION)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_requirement",
        "duplicate_requirement",
        "unknown_requirement",
        "unread_quote",
        "altered_quote",
        "no_quote",
        "boolean_label",
        "false_known_fact",
        "question_for_known",
        "missing_question",
        "extra_field",
    ],
)
def test_assessment_rejects_laundered_evidence_and_dependencies(mutation: str) -> None:
    row = decision(ANSWER, (1, 2, 3))
    if mutation == "missing_requirement":
        row["coverage"].pop()
    elif mutation == "duplicate_requirement":
        row["coverage"][-1] = row["coverage"][0]
    elif mutation == "unknown_requirement":
        row["coverage"][0]["requirement_id"] = 9
    elif mutation == "unread_quote":
        row["coverage"][0]["evidence"][0]["evidence_id"] = 9
    elif mutation == "altered_quote":
        row["coverage"][0]["evidence"][0]["quote"] = "只有电池才能启动"
    elif mutation == "no_quote":
        row["coverage"][0]["evidence"] = []
    elif mutation == "boolean_label":
        row["coverage"][0]["covered"] = "true"
    elif mutation == "extra_field":
        row["PRIVATE_EXTRA_KEY"] = "PRIVATE"
    else:
        row["dependencies"] = [
            {
                "name": "模式",
                "requirement_ids": [1],
                "evidence": [{"evidence_id": 2, "quote": "选择启动配额前必须确认控制模式。"}],
                "user_fact_ids": [9]
                if mutation == "false_known_fact"
                else ([1] if mutation == "question_for_known" else []),
                "question": "哪种模式？" if mutation == "question_for_known" else "",
            }
        ]
    with pytest.raises(ValueError, match=r"^invalid_investigation_assessment$"):
        parse_assessment(json.dumps(row), parse_plan(json.dumps(PLAN), QUESTION), evidence())


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing", "coverage_incomplete"),
        ("bad_block", "invalid_block_mapping"),
        ("uncited", "uncited_coverage"),
        ("unmapped", "unmapped_block"),
        ("extra_citation", "ungrounded_citation"),
    ],
)
def test_final_gate_binds_coverage_to_actual_blocks(mutation: str, code: str) -> None:
    row = decision(ANSWER, (1, 2, 3))
    blocks = tuple(
        AnswerBlock(b["text"], tuple(b["citations"])) for b in ANSWER["answer"]["blocks"]
    )
    if mutation == "missing":
        row["coverage"][2]["covered"] = False
    elif mutation == "bad_block":
        row["coverage"][2]["blocks"] = [12]
    elif mutation == "uncited":
        row["coverage"][2]["blocks"] = [1]
    elif mutation == "unmapped":
        blocks += (AnswerBlock("合成额外说明", (1,)),)
    else:
        blocks = (replace(blocks[0], citations=(1, 2)), *blocks[1:])
    assessment, _ = parse_assessment(
        json.dumps(row), parse_plan(json.dumps(PLAN), QUESTION), evidence()
    )
    assert answer_gate(assessment, blocks) == code


def test_plan_call_shares_step_budget_and_never_leaks_across_requests() -> None:
    agent, generator = make_agent(
        PLAN, PLAN, settings=AgentSettings(plan_investigation=True, max_steps=1)
    )
    for _ in range(2):
        result = agent.run(QUESTION)
        assert (
            result.status == "budget_exhausted" and result.model_calls == 1 and not result.evidence
        )
    assert generator.calls == [{"question": QUESTION}, {"question": QUESTION}]


@pytest.mark.parametrize("cancel", [False, True])
def test_initial_plan_cannot_bypass_time_or_cancellation_boundary(cancel: bool) -> None:
    now, stop = [0.0], [False]
    agent, generator = make_agent(
        PLAN, decision(SEARCH), settings=AgentSettings(plan_investigation=True)
    )
    generate = generator.generate

    def slow(system: str, user: str) -> str:
        raw = generate(system, user)
        stop[0] = cancel
        now[0] = 181.0 if not cancel else 0.0
        return raw

    generator.generate = slow
    result = replace(agent, clock=lambda: now[0]).run(QUESTION, cancelled=lambda: stop[0])
    assert result.status == ("cancelled" if cancel else "budget_exhausted")
    assert result.model_calls == 1 and result.search_calls == 0
    assert not any(isinstance(e, PlanningEvent) for e in result.events)


def test_http_result_carries_the_validated_plan_for_local_audit() -> None:
    fake = PlanningRetriever()
    agent, _ = make_agent(
        PLAN, decision(ABSTAIN), settings=AgentSettings(plan_investigation=True), fake=fake
    )
    response = TestClient(create_app(fake, agent=agent)).post(
        "/api/investigate", json={"query": QUESTION}
    )
    assert response.status_code == 200
    events = response.json()["events"]
    plan_event = next(e for e in events if e.get("plan"))
    assert plan_event["plan"]["facts"] == PLAN["facts"]
    assert plan_event["plan"]["requirements"] == PLAN["requirements"]


def test_incomplete_coverage_cannot_add_calls_beyond_original_budget() -> None:
    early = {
        "action": "answer",
        "answer": {"answerable": True, "blocks": ANSWER["answer"]["blocks"][:1]},
    }
    agent, generator = make_agent(
        PLAN,
        decision(SEARCH),
        decision(READ),
        decision(early, (1,)),
        decision(ABSTAIN),
        settings=AgentSettings(plan_investigation=True, max_steps=4),
        fake=PlanningRetriever(),
    )
    result = agent.run(QUESTION)
    assert result.status == "budget_exhausted" and not result.blocks
    assert result.model_calls == 4 and len(generator.actions) == 1


def test_disabled_planning_preserves_5814e5f_profile_and_original_path() -> None:
    agent, generator = make_agent(
        SEARCH,
        READ,
        {
            "action": "answer",
            "answer": {"answerable": True, "blocks": [{"text": "合成", "citations": [1]}]},
        },
    )
    assert (
        agent.profile_fingerprint
        == "57acea1d6a51814cce0008cf5ea05132fc1ff5826cf14f955fd8951ce2b3964c"
    )
    result = agent.run("q")
    assert result.status == "answered" and result.model_calls == 3
    assert "plan" not in generator.calls[0]
    assert not any(isinstance(e, PlanningEvent) for e in result.events)


def test_planning_and_answer_review_both_apply_without_hidden_calls() -> None:
    plan = {"requirements": [PLAN["requirements"][0]], "facts": []}

    def one(action: object, covered: bool = False) -> dict[str, Any]:
        row = decision(action, (1,) if covered else ())
        row["coverage"] = row["coverage"][:1]
        return row

    answer = {
        "action": "answer",
        "answer": {"answerable": True, "blocks": [ANSWER["answer"]["blocks"][0]]},
    }
    agent, generator = make_agent(
        plan,
        one(SEARCH),
        one(READ),
        one(answer, True),
        feedback(),
        settings=AgentSettings(plan_investigation=True, review_answers=True),
        fake=PlanningRetriever(),
    )
    result = agent.run(QUESTION)
    assert result.status == "answered" and len(generator.calls) == 5
    assert set(generator.calls[-1]) == {"question", "draft_blocks", "evidence"}
    assert any(e.outcome == "accepted" for e in result.events)


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_planning_switch_requires_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="plan_investigation"):
        AgentSettings(plan_investigation=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw,code",
    [
        ('{"PRIVATE":', "json_syntax"),
        ('{"requirements":[],"facts":[],"PRIVATE":"secret"}', "extra_fields"),
        ('{"requirements":[],"facts":[]}', "list_size"),
        ('{"requirements":[],"requirements":[],"facts":[]}', "duplicate_key"),
        ('{"requirements":NaN,"facts":[]}', "nonfinite_JSON"),
        ('{"requirements":[],"facts":"\\ud800"}', "list_size"),
        ("\ud800", "invalid_unicode"),
    ],
)
def test_parser_diagnostic_is_fixed_and_contains_no_rejected_content(raw: str, code: str) -> None:
    with pytest.raises(PlanningValidationError) as caught:
        parse_plan(raw, QUESTION)
    assert caught.value.reason == code and code in PLANNING_ERROR_CODES
    assert str(caught.value) == "invalid_investigation_plan"
    assert "PRIVATE" not in repr(caught.value) and "secret" not in repr(caught.value)


def test_fact_extra_field_fails_closed_and_http_exposes_only_safe_code() -> None:
    plan = deepcopy(PLAN)
    plan["facts"][0]["PRIVATE_FIELD"] = "PRIVATE_VALUE"
    fake = PlanningRetriever()
    agent, _ = make_agent(plan, settings=AgentSettings(plan_investigation=True), fake=fake)
    response = TestClient(create_app(fake, agent=agent)).post(
        "/api/investigate", json={"query": QUESTION}
    )
    assert response.status_code == 503
    result = response.json()
    assert result["status"] == "invalid_action"
    event = next(e for e in result["events"] if e.get("error_code"))
    assert event["outcome"] == "invalid_plan" and event["error_code"] == "extra_fields"
    assert event["plan"] is None and event["assessment"] is None
    assert "PRIVATE_FIELD" not in response.text and "PRIVATE_VALUE" not in response.text


def test_json_recursion_failure_has_a_safe_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RecursionError("PRIVATE_DETAIL")

    monkeypatch.setattr(json, "loads", fail)
    with pytest.raises(PlanningValidationError) as caught:
        parse_plan("{}", QUESTION)
    assert caught.value.reason == "json_depth"
    assert "PRIVATE_DETAIL" not in repr(caught.value)


@pytest.mark.parametrize("bad_action", [False, True])
def test_failed_assessment_does_not_republish_previous_valid_assessment(bad_action: bool) -> None:
    invalid = decision({"action": "PRIVATE_ACTION"}) if bad_action else decision(READ, (1,))
    agent, _ = make_agent(
        PLAN,
        decision(SEARCH),
        invalid,
        settings=AgentSettings(plan_investigation=True),
        fake=PlanningRetriever(),
    )
    result = agent.run(QUESTION)
    assert result.status == "invalid_action"
    events = [e for e in result.events if isinstance(e, PlanningEvent)]
    assert events[-2].assessment is not None
    assert events[-1].error_code == ("action_schema" if bad_action else "evidence_id")
    assert events[-1].assessment is None and events[-1].plan is None
    assert events[-1].proposed_action is None
    assert "PRIVATE_ACTION" not in repr(result)


@pytest.mark.parametrize("stage,code", [("PRIVATE", "extra_fields"), ("plan", "PRIVATE")])
def test_diagnostic_constructor_rejects_unknown_codes(stage: str, code: str) -> None:
    with pytest.raises(ValueError, match=r"^invalid planning diagnostic code$"):
        PlanningValidationError(stage, code)
