from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from test_answering import ranked
from test_service import FakeRetriever
from zhrag.agent import AgentSettings, DocumentAgent, parse_action
from zhrag.answering import GenerationError
from zhrag.retrieval.online import OnlineRetrievalResult


class ScriptedGenerator:
    profile_fingerprint = "a" * 64

    def __init__(self, *actions: object) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, object]] = []

    def generate(self, system: str, user: str) -> str:
        assert "untrusted" in system
        self.calls.append(json.loads(user))
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)


SEARCH = {"action": "search_docs", "query": "first"}
READ = {"action": "read_passage", "evidence_id": 1}
ANSWER = {
    "action": "answer",
    "answer": {"answerable": True, "blocks": [{"text": "合成回答", "citations": [1]}]},
}
ABSTAIN = {"action": "abstain"}


class InvestigationRetriever(FakeRetriever):
    def retrieve(self, query: str) -> OnlineRetrievalResult:
        base = super().retrieve(query)
        return replace(base, passages=(ranked("合成证据 " + query, doc_id=query),))


def make_agent(
    *actions: object,
    settings: AgentSettings | None = None,
    fake: FakeRetriever | None = None,
) -> tuple[DocumentAgent, ScriptedGenerator]:
    generator = ScriptedGenerator(*actions)
    return DocumentAgent(
        fake or InvestigationRetriever(), generator, "test-index", settings or AgentSettings()
    ), generator


def test_adaptive_search_accumulates_evidence_and_keeps_stable_citations() -> None:
    answer = {
        "action": "answer",
        "answer": {
            "answerable": True,
            "blocks": [{"text": "合成综合结论", "citations": [1, 2]}],
        },
    }
    agent, generator = make_agent(
        SEARCH,
        READ,
        {"action": "search_docs", "query": "second"},
        {"action": "read_passage", "evidence_id": 2},
        answer,
    )
    outcome = agent.run("合成跨文档问题")
    assert outcome.status == "answered"
    assert outcome.blocks[0].citations == (1, 2)
    assert [row.ranked.passage.doc_id for row in outcome.evidence] == ["first", "second"]
    assert outcome.search_calls == 2 and outcome.read_calls == 2 and outcome.model_calls == 5
    assert generator.calls[1]["evidence"] == []  # search preview is not opened evidence
    assert generator.calls[-1]["searched_queries"] == ["first", "second"]
    assert len(generator.calls[-1]["evidence"]) == 2
    assert outcome.prompt_estimated_tokens > 0
    assert "合成" not in json.dumps([event.outcome for event in outcome.events])


def test_can_read_second_candidate_without_renumbering() -> None:
    agent, _ = make_agent(
        SEARCH,
        {"action": "read_passage", "evidence_id": 2},
        {
            "action": "answer",
            "answer": {
                "answerable": True,
                "blocks": [
                    {"text": "合成回答", "citations": [2]},
                ],
            },
        },
        fake=FakeRetriever(),
    )
    outcome = agent.run("q")
    assert outcome.status == "answered" and outcome.evidence[0].citation_id == 2


@pytest.mark.parametrize(
    "actions,status",
    [
        (({"action": "clarify", "question": "请补充版本和执行计划。"},), "clarification_needed"),
        ((ABSTAIN,), "insufficient_evidence"),
        ((ANSWER,), "invalid_answer"),
        ((SEARCH, ANSWER), "invalid_answer"),
    ],
)
def test_terminal_decisions_and_unread_evidence(actions: tuple[object, ...], status: str) -> None:
    agent, _ = make_agent(*actions)
    outcome = agent.run("q")
    assert outcome.status == status and not outcome.blocks
    if status == "clarification_needed":
        assert outcome.clarification == "请补充版本和执行计划。"
        assert outcome.search_calls == 0


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "null",
        "[]",
        '{"action":"abstain","action":"search_docs"}',
        '{"action":"abstain","extra":1}',
        '{"action":"read_passage","evidence_id":true}',
        '{"action":"read_passage","evidence_id":1.0}',
        '{"action":"search_docs","query":" "}',
        '{"action":"execute_sql","query":"DROP TABLE x"}',
        '{"action":"read_passage","evidence_id":NaN}',
        json.dumps({"action": "clarify", "question": "\ud800"}),
        "[" * 2000,
    ],
)
def test_invalid_actions_do_not_dispatch(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid_action"):
        parse_action(raw)
    agent, _ = make_agent(raw)
    outcome = agent.run("q")
    assert outcome.status == "invalid_action" and outcome.search_calls == 0


def test_nested_nonfinite_answer_has_safe_diagnostic_instead_of_escaping() -> None:
    raw = (
        '{"action":"answer","answer":{"answerable":true,'
        '"blocks":[{"text":"private","citations":[1e999]}]}}'
    )
    agent, _ = make_agent(SEARCH, READ, raw)
    outcome = agent.run("q")
    assert outcome.status == "invalid_answer" and not outcome.blocks
    assert outcome.events[-1].validation_error == "json_value"
    assert "private" not in repr(outcome)


def test_duplicate_queries_reads_and_unknown_ids_do_not_consume_tool_budget() -> None:
    agent, _ = make_agent(
        SEARCH,
        {"action": "search_docs", "query": " FIRST "},
        READ,
        READ,
        {"action": "read_passage", "evidence_id": 99},
        ANSWER,
    )
    outcome = agent.run("q")
    assert outcome.search_calls == 1 and outcome.read_calls == 1
    assert {"duplicate_query", "already_read", "unknown_evidence"} <= {
        event.outcome for event in outcome.events
    }


def test_search_and_read_limits_are_independent_of_decision_limit() -> None:
    agent, _ = make_agent(
        SEARCH,
        {"action": "search_docs", "query": "second"},
        READ,
        {"action": "read_passage", "evidence_id": 2},
        ANSWER,
        settings=AgentSettings(max_searches=1, max_reads=1),
        fake=FakeRetriever(),
    )
    outcome = agent.run("q")
    assert outcome.status == "answered"
    assert outcome.search_calls == outcome.read_calls == 1
    assert {"search_limit", "read_limit"} <= {event.outcome for event in outcome.events}


def test_step_budget_never_publishes_partial_answer_or_makes_extra_call() -> None:
    agent, generator = make_agent(SEARCH, READ, ANSWER, settings=AgentSettings(max_steps=2))
    outcome = agent.run("q")
    assert outcome.status == "budget_exhausted" and not outcome.blocks
    assert len(outcome.evidence) == 1 and len(generator.calls) == 2


@pytest.mark.parametrize(
    "settings",
    [
        AgentSettings(max_prompt_tokens=1),
        AgentSettings(max_total_prompt_tokens=1),
        AgentSettings(max_prompt_bytes=1),
        AgentSettings(max_prompt_chars=1),
    ],
)
def test_prompt_budgets_checked_before_paid_call(settings: AgentSettings) -> None:
    agent, generator = make_agent(ANSWER, settings=settings)
    assert agent.run("问题").status == "budget_exhausted"
    assert not generator.calls


def test_complete_oversized_passage_is_not_opened() -> None:
    agent, generator = make_agent(
        SEARCH, READ, ABSTAIN, fake=FakeRetriever(), settings=AgentSettings(max_prompt_chars=5000)
    )
    outcome = agent.run("q")
    assert not outcome.evidence and outcome.status == "insufficient_evidence"
    assert any(event.outcome == "context_limit" for event in outcome.events)
    assert generator.calls[-1]["evidence"] == []


@pytest.mark.parametrize(
    "error,status",
    [
        (GenerationError("generation_timeout"), "generation_timeout"),
        (SystemExit("private-provider-payload"), "generation_failed"),
    ],
)
def test_generation_error_keeps_open_evidence_but_hides_provider_error(
    error: BaseException,
    status: str,
) -> None:
    agent, _ = make_agent(SEARCH, READ, error)
    outcome = agent.run("q")
    assert outcome.status == status and outcome.evidence and not outcome.blocks
    assert "private-provider" not in repr(outcome)


def test_tool_failure_is_an_observation_and_can_trigger_different_search() -> None:
    agent, generator = make_agent(SEARCH, ABSTAIN, fake=FakeRetriever(error=RuntimeError("secret")))
    outcome = agent.run("q")
    assert outcome.status == "retrieval_failed"
    assert outcome.search_calls == 1
    assert "tool_failed" in json.dumps(generator.calls[-1])
    assert "secret" not in repr(outcome)


@pytest.mark.parametrize(
    "refusal",
    [
        ABSTAIN,
        {
            "action": "answer",
            "answer": {
                "answerable": False,
                "blocks": [],
            },
        },
    ],
)
def test_both_refusal_forms_preserve_retrieval_failure(refusal: object) -> None:
    agent, _ = make_agent(SEARCH, refusal, fake=FakeRetriever(error=RuntimeError("synthetic")))
    assert agent.run("q").status == "retrieval_failed"


def test_time_budget_checked_after_model_before_tool_dispatch() -> None:
    now = [0.0]
    agent, generator = make_agent(SEARCH)
    original = generator.generate

    def slow(system: str, user: str) -> str:
        reply = original(system, user)
        now[0] = 181.0
        return reply

    generator.generate = slow
    agent = replace(agent, clock=lambda: now[0])
    outcome = agent.run("q")
    assert outcome.status == "budget_exhausted" and outcome.search_calls == 0


def test_cancellation_before_or_during_model_prevents_next_tool() -> None:
    stop = threading.Event()
    agent, generator = make_agent(SEARCH)
    stop.set()
    assert agent.run("q", cancelled=stop.is_set).status == "cancelled"
    assert not generator.calls
    stop.clear()
    original = generator.generate

    def interrupt(system: str, user: str) -> str:
        stop.set()
        return original(system, user)

    generator.generate = interrupt
    outcome = agent.run("q", cancelled=stop.is_set)
    assert outcome.status == "cancelled" and outcome.search_calls == 0


def test_request_state_does_not_leak_and_fingerprint_binds_settings_and_index() -> None:
    agent, generator = make_agent(SEARCH, READ, ANSWER, ABSTAIN)
    first = agent.run("first task")
    second = agent.run("second task")
    assert first.evidence and not second.evidence
    assert generator.calls[-1]["candidates"] == []
    assert (
        first.profile_fingerprint != replace(agent, retrieval_identity="other").profile_fingerprint
    )
    assert (
        first.profile_fingerprint
        != replace(
            agent,
            settings=AgentSettings(max_steps=9),
        ).profile_fingerprint
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": True},
        {"max_searches": 0},
        {"max_reads": 13},
        {"max_seconds": float("nan")},
        {"max_seconds": True},
    ],
)
def test_invalid_settings_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AgentSettings(**kwargs)
