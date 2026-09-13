from __future__ import annotations

import json
from dataclasses import replace

import pytest

from test_agent import ANSWER, READ, SEARCH, make_agent
from test_service import FakeRetriever
from zhrag.agent import AgentSettings
from zhrag.answering import GenerationError
from zhrag.eval.agent_comparison import parse_plan, run_method


def test_single_rag_retrieves_original_question_once_without_planner() -> None:
    agent, generator = make_agent(ANSWER["answer"])
    trial = run_method(agent, "original", "single_rag")
    assert trial["status"] == "answered"
    assert trial["search_calls"] == trial["model_calls"] == 1
    assert generator.calls[0]["question"] == "original"


def test_fixed_workflow_plans_once_then_interleaves_both_queries() -> None:
    answer = {"answerable": True, "blocks": [{"text": "合成综合", "citations": [1, 2]}]}
    agent, generator = make_agent({"queries": ["first", "second"]}, answer)
    trial = run_method(agent, "original", "fixed_workflow")
    assert trial["status"] == "answered"
    assert trial["search_calls"] == trial["model_calls"] == 2
    assert len(generator.calls[1]["evidence"]) == 2
    assert trial["clarification"] == ""


def test_agent_comparison_retains_public_execution_events() -> None:
    agent, _ = make_agent(SEARCH, READ, ANSWER)
    trial = run_method(agent, "original", "document_agent")
    assert trial["status"] == "answered" and trial["events"]
    assert trial["profile_fingerprint"] == agent.profile_fingerprint


@pytest.mark.parametrize(
    "plan",
    [
        '{"queries":["a"],"queries":["b"]}',
        '{"queries":["a"," A "]}',
        '{"queries":[]}',
        '{"queries":[1]}',
        '{"queries":["a"],"extra":true}',
        json.dumps({"queries": ["\ud800"]}),
    ],
)
def test_invalid_plan_cannot_dispatch_retrieval(plan: str) -> None:
    with pytest.raises(GenerationError):
        parse_plan(plan, max_queries=3)
    agent, _ = make_agent(plan)
    trial = run_method(agent, "q", "fixed_workflow")
    assert trial["status"] == "invalid_answer" and trial["search_calls"] == 0


def test_baseline_applies_shared_decision_budget() -> None:
    agent, generator = make_agent({"queries": ["first"]}, ANSWER["answer"])
    agent = replace(agent, settings=AgentSettings(max_steps=1))
    trial = run_method(agent, "q", "fixed_workflow")
    assert trial["status"] == "budget_exhausted" and not trial["blocks"]
    assert len(generator.calls) == 1 and trial["search_calls"] == 1


def test_baseline_retrieval_failure_does_not_generate_or_leak_error() -> None:
    agent, generator = make_agent(fake=FakeRetriever(error=RuntimeError("private-error")))
    trial = run_method(agent, "q", "single_rag")
    assert trial["status"] == "retrieval_failed" and not generator.calls
    assert "private-error" not in json.dumps(trial)
