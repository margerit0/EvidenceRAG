from __future__ import annotations

from dataclasses import replace

from test_agent import ANSWER, READ, SEARCH, make_agent
from zhrag.agent_progress import AgentProgress
from zhrag.answering import GenerationError


def test_observations_preserve_prompts_outcome_and_profile() -> None:
    baseline, baseline_generator = make_agent(SEARCH, READ, ANSWER)
    observed, observed_generator = make_agent(SEARCH, READ, ANSWER)
    baseline = replace(baseline, clock=lambda: 1.0)
    observed = replace(observed, clock=lambda: 1.0)
    progress: list[AgentProgress] = []
    expected = baseline.run("合成问题")
    actual = observed.run("合成问题", observe=progress.append)
    assert actual == expected
    assert baseline_generator.calls == observed_generator.calls
    assert observed.profile_fingerprint == baseline.profile_fingerprint
    assert [event.seq for event in progress] == list(range(1, len(progress) + 1))
    starts = [event for event in progress if event.phase == "started"]
    assert [event.action for event in starts] == [
        "decide",
        "search_docs",
        "decide",
        "read_passage",
        "decide",
        "validate_answer",
    ]
    assert len({event.invocation_id for event in starts}) == len(starts)
    for event in starts:
        assert (
            len(
                [
                    end
                    for end in progress
                    if end.invocation_id == event.invocation_id and end.phase == "completed"
                ]
            )
            == 1
        )
    assert all("合成" not in str(event) for event in progress)


def test_failed_dependency_closes_its_active_invocation() -> None:
    agent, _ = make_agent(GenerationError("generation_timeout"))
    progress: list[AgentProgress] = []
    outcome = agent.run("合成问题", observe=progress.append)
    assert outcome.status == "generation_timeout"
    assert progress[0].phase == "started"
    assert progress[1].invocation_id == progress[0].invocation_id
    assert progress[1].outcome == "generation_timeout"
    assert progress[-1].action == "finish"


def test_broken_ui_observer_cannot_change_agent_result() -> None:
    def broken(_event: AgentProgress) -> None:
        raise RuntimeError("display failure")

    agent, _ = make_agent(SEARCH, READ, ANSWER)
    assert agent.run("合成问题", observe=broken).status == "answered"
