"""Decision details are request-local UI data, never model history or raw replies."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from test_agent import ABSTAIN, ANSWER, READ, SEARCH, make_agent
from test_agent_service import app_for
from test_agent_stream import parse_frames
from zhrag.agent_progress import AgentProgress
from zhrag.answering import GenerationError


def test_decisions_include_validated_parameters_without_changing_model_history() -> None:
    search = {"action": "search_docs", "query": "  合成搜索词  "}
    agent, generator = make_agent(search, READ, ANSWER)
    progress: list[AgentProgress] = []
    outcome = agent.run("合成问题", observe=progress.append)
    decisions = [
        asdict(event).get("decision")
        for event in progress
        if event.action == "decide" and event.phase == "completed"
    ]
    assert decisions == [
        {"action": "search_docs", "query": "合成搜索词", "evidence_id": None, "question": None},
        {"action": "read_passage", "query": None, "evidence_id": 1, "question": None},
        {"action": "answer", "query": None, "evidence_id": None, "question": None},
    ]
    assert all("decision" not in asdict(event) for event in outcome.events)
    for prompt in generator.calls:
        observations = prompt["observations"]
        assert isinstance(observations, list)
        assert all("decision" not in event for event in observations)
    assert "合成回答" not in repr(progress)
    assert all(
        asdict(event).get("decision") is None for event in progress if event.phase == "started"
    )


@pytest.mark.parametrize(
    "action",
    [ABSTAIN, {"action": "clarify", "question": "请补充合成设备的版本。"}],
)
def test_terminal_choices_have_decision_content(action: dict[str, str]) -> None:
    agent, _ = make_agent(action)
    progress: list[AgentProgress] = []
    agent.run("合成问题", observe=progress.append)
    detail = asdict(progress[1]).get("decision")
    assert detail is not None
    assert detail["action"] == action["action"]
    assert detail["question"] == action.get("question")


@pytest.mark.parametrize(
    "reply",
    [
        GenerationError("generation_timeout"),
        {"action": "search_docs", "query": "PRIVATE_QUERY", "reasoning": "PRIVATE_RAW_REPLY"},
        {"action": "read_passage", "evidence_id": "PRIVATE_INVALID_ID"},
    ],
)
def test_failed_decisions_never_publish_unvalidated_reply_content(reply: object) -> None:
    agent, _ = make_agent(reply)
    progress: list[AgentProgress] = []
    outcome = agent.run("合成问题", observe=progress.append)
    assert outcome.status in {"generation_timeout", "invalid_action"}
    assert all(asdict(event).get("decision") is None for event in progress)
    assert progress[1].outcome == outcome.status
    assert "PRIVATE_" not in repr(progress)


def test_cancelled_generation_does_not_publish_an_undispatched_decision() -> None:
    agent, generator = make_agent(SEARCH)
    progress: list[AgentProgress] = []
    outcome = agent.run(
        "合成问题", observe=progress.append, cancelled=lambda: bool(generator.calls)
    )
    assert outcome.status == "cancelled"
    assert all(asdict(event).get("decision") is None for event in progress)


@pytest.mark.parametrize("stream", [False, True])
def test_http_decisions_survive_streaming_and_final_response(stream: bool) -> None:
    client = TestClient(app_for(SEARCH, READ, ANSWER))
    response = client.post(
        "/api/investigate/stream" if stream else "/api/investigate",
        json={"query": "合成问题"},
    )
    body = parse_frames(response.text)[-1][1]["response"] if stream else response.json()
    decisions = [event.get("decision") for event in body["events"] if event["action"] == "decide"]
    assert [decision and decision["action"] for decision in decisions] == [
        "search_docs",
        "read_passage",
        "answer",
    ]
    assert decisions[0]["query"] == "first"
    assert decisions[1]["evidence_id"] == 1
    assert "合成回答" not in json.dumps(decisions, ensure_ascii=False)
    if stream:
        streamed = [
            payload["decision"]
            for event, payload in parse_frames(response.text)
            if event == "progress"
            and payload["action"] == "decide"
            and payload["phase"] == "completed"
        ]
        assert streamed == decisions


def test_batch_response_keeps_invalid_decision_status_and_no_raw_reply() -> None:
    client = TestClient(app_for({"action": "PRIVATE_INVALID_ACTION"}))
    response = client.post("/api/investigate", json={"query": "合成问题"})
    decision = response.json()["events"][0]
    assert decision["action"] == "decide" and decision["outcome"] == "invalid_action"
    assert decision.get("decision") is None
    assert "PRIVATE_" not in response.text
