"""Original synthetic evidence only. No provider calls or evaluation datasets."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from test_agent import ANSWER, READ, SEARCH, make_agent
from zhrag.agent import AgentSettings
from zhrag.agent_checkpoint import new_checkpoint, unpack
from zhrag.investigation_executor import BoundaryExecution, InvestigationExecutor
from zhrag.investigation_store import InvestigationStore, LeaseLost, RunConflict
from zhrag.service.app import _AdmissionGate, create_app
from zhrag.service.investigations import public_run

CLARIFY = {"action": "clarify", "question": "合成环境使用哪个版本？"}


class Clock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


class Crash(BaseException):
    """Simulate process death: skip every normal exception handler."""


def setup(tmp_path: Path, *actions: object, settings: AgentSettings | None = None):
    clock = Clock()
    store = InvestigationStore(tmp_path / "runs.db", clock=clock)
    agent, generator = make_agent(*actions, settings=settings)
    agent = replace(agent, clock=clock)
    executor = InvestigationExecutor(
        store, {"fake": agent}, "synthetic-snapshot-v1", _AdmissionGate(1)
    )
    run_id = executor.create("create-key", "合成问题", "fake")
    return store, agent, generator, executor, run_id, clock


def run(store, agent, run_id, *, fault=lambda _: None, owner="worker"):
    epoch = store.claim(run_id, owner, agent.profile_fingerprint, "synthetic-snapshot-v1")
    assert epoch is not None
    execution = BoundaryExecution(
        store, agent, run_id, owner=owner, epoch=epoch, shutdown=threading.Event(), fault=fault
    )
    execution.execute()
    return epoch


def command(store, run_id, action, key="command-key", **kwargs):
    store.command(run_id, key, action, version=store.get(run_id)["version"], **kwargs)


def test_normal_result_and_attempts_are_durable(tmp_path):
    store, agent, generator, executor, run_id, _ = setup(tmp_path, SEARCH, READ, ANSWER)
    run(store, agent, run_id)
    view = public_run(executor, run_id)
    assert view["status"] == "completed"
    assert view["response"]["blocks"][0]["citations"] == [1]
    assert view["response"]["sources"][0]["text"] == "合成证据 first"
    assert len(generator.calls) == 3
    assert [a["status"] for a in view["attempts"]] == ["completed"] * 5
    assert view["response"]["usage"]["model_calls"] == 3
    assert view["progress"][1]["decision"]["action"] == "search_docs"
    assert InvestigationStore(store.path).get(run_id)["response"] == store.get(run_id)["response"]


def test_clarify_supplement_preserves_evidence_ids_and_budget(tmp_path):
    store, agent, generator, _, run_id, clock = setup(tmp_path, SEARCH, READ, CLARIFY, ANSWER)
    run(store, agent, run_id)
    before = store.get(run_id)
    assert before["status"] == "waiting_input"
    clock.now += 10_000  # waiting is free, including across a process restart
    command(store, run_id, "supplement", wait_id=before["wait_id"], text="合成版本甲")
    run(store, agent, run_id)
    after = store.get(run_id)
    assert after["status"] == "completed"
    assert after["response"]["usage"]["model_calls"] == 4
    assert after["response"]["usage"]["search_calls"] == 1
    assert after["response"]["usage"]["read_calls"] == 1
    assert generator.calls[-1]["evidence"][0]["id"] == 1
    assert "合成版本甲" in generator.calls[-1]["question"]
    assert after["active_seconds"] == 0


def test_pause_saves_returned_model_result_then_resumes_without_repeating(tmp_path):
    store, agent, generator, _, run_id, clock = setup(tmp_path, SEARCH, READ, ANSWER)
    original = generator.generate

    def pause_in_call(system, user):
        result = original(system, user)
        command(store, run_id, "pause")
        return result

    generator.generate = pause_in_call
    run(store, agent, run_id)
    row = store.get(run_id)
    assert row["status"] == "paused" and row["checkpoint"]["phase"] == "model_result"
    assert len(generator.calls) == 1
    clock.now += 10000
    generator.generate = original
    command(store, run_id, "resume", key="resume-key")
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "completed" and len(generator.calls) == 3


@pytest.mark.parametrize(
    "boundary", ["before_model_result", "before_dispatch", "before_search_result"]
)
def test_crash_after_result_commit_new_executor_reuses_saved_result(tmp_path, boundary):
    store, agent, generator, _, run_id, clock = setup(tmp_path, SEARCH, READ, ANSWER)

    def crash(point):
        if point == boundary:
            raise Crash()

    with pytest.raises(Crash):
        run(store, agent, run_id, fault=crash)
    assert len(generator.calls) == 1
    clock.now += 16
    new_store = InvestigationStore(store.path, clock=clock)
    new_store.recover()
    assert new_store.get(run_id)["status"] == "interrupted"
    agent2, generator2 = make_agent(READ, ANSWER)
    agent2 = replace(agent2, clock=clock)
    command(new_store, run_id, "resume")
    run(new_store, agent2, run_id, owner="replacement")
    assert new_store.get(run_id)["status"] == "completed"
    assert len(generator2.calls) == 2
    assert new_store.get(run_id)["response"]["usage"]["model_calls"] == 3


@pytest.mark.parametrize("point", ["model_returned_before_commit", "search_returned_before_commit"])
def test_unknown_external_result_is_not_retried_after_restart(tmp_path, point):
    store, agent, _, _, run_id, clock = setup(tmp_path, SEARCH, READ, ANSWER)

    def crash(value):
        if value == point:
            raise Crash()

    with pytest.raises(Crash):
        run(store, agent, run_id, fault=crash)
    clock.now += 16
    replacement = InvestigationStore(store.path, clock=clock)
    replacement.recover()
    assert replacement.get(run_id)["reason"] == "call_outcome_unknown"
    assert replacement.get(run_id)["status"] == "unrecoverable"
    assert "unknown" in [a["status"] for a in replacement.attempts(run_id)]
    with pytest.raises(RunConflict, match="run_is_terminal"):
        command(replacement, run_id, "resume")


def test_create_and_command_idempotency_and_parallel_claim(tmp_path):
    store, agent, _, executor, run_id, _ = setup(tmp_path, SEARCH)
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda _: executor.create("create-key", "合成问题", "fake"), range(16)))
    assert set(ids) == {run_id}
    with pytest.raises(RunConflict, match="idempotency_conflict"):
        executor.create("create-key", "不同的问题", "fake")
    command(store, run_id, "pause")
    version = store.get(run_id)["version"]
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda _: store.command(run_id, "resume-key", "resume", version=version), range(16)
            )
        )
    with ThreadPoolExecutor(8) as pool:
        claims = list(
            pool.map(
                lambda n: store.claim(
                    run_id, str(n), agent.profile_fingerprint, "synthetic-snapshot-v1"
                ),
                range(16),
            )
        )
    assert len([c for c in claims if c is not None]) == 1
    with pytest.raises(RunConflict, match="stale_run_version"):
        store.command(run_id, "different-key", "resume", version=version)


def test_outdated_supplement_cannot_answer_later_question(tmp_path):
    store, agent, _, _, run_id, _ = setup(tmp_path, CLARIFY, CLARIFY, ANSWER)
    run(store, agent, run_id)
    first = store.get(run_id)
    command(store, run_id, "supplement", key="answer-1", wait_id=first["wait_id"], text="甲")
    run(store, agent, run_id)
    second = store.get(run_id)
    assert second["wait_id"] != first["wait_id"]
    with pytest.raises(RunConflict, match="stale_clarification"):
        command(store, run_id, "supplement", key="answer-2", wait_id=first["wait_id"], text="乙")
    store.command(
        run_id,
        "answer-1",
        "supplement",
        version=first["version"],
        wait_id=first["wait_id"],
        text="甲",
    )
    assert store.get(run_id)["status"] == "waiting_input"


def test_cancel_while_blocking_retains_gate_and_discards_answer(tmp_path):
    store, _agent, generator, executor, run_id, _ = setup(tmp_path, CLARIFY)
    entered, release = threading.Event(), threading.Event()
    original = generator.generate

    def block(system, user):
        entered.set()
        assert release.wait(5)
        return original(system, user)

    generator.generate = block
    executor.tick()
    assert entered.wait(5)
    command(store, run_id, "cancel")
    assert store.get(run_id)["status"] == "cancelling"
    assert not executor.admission.try_acquire()
    release.set()
    for _ in range(100):
        if store.get(run_id)["status"] == "cancelled":
            break
        time.sleep(0.02)
    assert store.get(run_id)["status"] == "cancelled"
    assert not store.get(run_id)["response"]


def test_expired_owner_late_result_is_fenced(tmp_path):
    store, agent, _, _, run_id, clock = setup(tmp_path, SEARCH)
    epoch = store.claim(run_id, "old", agent.profile_fingerprint, "synthetic-snapshot-v1")
    cp = store.get(run_id)["checkpoint"]
    attempt = store.save(run_id, "old", epoch, cp, [], begin="decide")
    clock.now += 16
    InvestigationStore(store.path, clock=clock).recover()
    with pytest.raises(LeaseLost):
        store.save(run_id, "old", epoch, cp, [], complete=attempt, result="late")
    assert store.get(run_id)["reason"] == "call_outcome_unknown"


@pytest.mark.parametrize("changed", ["model", "snapshot"])
def test_mismatching_model_or_snapshot_blocks_resume(tmp_path, changed):
    store, agent, _, _, run_id, _ = setup(tmp_path, SEARCH)
    command(store, run_id, "pause")
    command(store, run_id, "resume", key="resume-key")
    assert (
        store.claim(
            run_id,
            "new",
            "other" if changed == "model" else agent.profile_fingerprint,
            "other" if changed == "snapshot" else "synthetic-snapshot-v1",
        )
        is None
    )
    assert store.get(run_id)["reason"] == "snapshot_or_model_mismatch"


def test_total_step_and_token_budget_cannot_be_reset_by_supplement(tmp_path):
    store, agent, generator, _, run_id, _ = setup(
        tmp_path, CLARIFY, ANSWER, settings=AgentSettings(max_steps=1)
    )
    run(store, agent, run_id)
    before = store.get(run_id)
    command(store, run_id, "supplement", wait_id=before["wait_id"], text="甲")
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "budget_exhausted"
    assert len(generator.calls) == 1
    assert store.get(run_id)["response"]["usage"] == before["response"]["usage"]


def test_active_time_is_cumulative_and_waiting_time_is_excluded(tmp_path):
    store, agent, generator, _, run_id, clock = setup(
        tmp_path, CLARIFY, CLARIFY, settings=AgentSettings(max_seconds=5)
    )
    original = generator.generate

    def takes_time(system, user):
        clock.now += 3
        return original(system, user)

    generator.generate = takes_time
    run(store, agent, run_id)
    first = store.get(run_id)
    assert first["active_seconds"] == 3
    clock.now += 10000
    command(store, run_id, "supplement", wait_id=first["wait_id"], text="甲")
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "budget_exhausted"
    assert store.get(run_id)["active_seconds"] == 6


def make_app(path: Path, *actions: object):
    agent, generator = make_agent(*actions)
    app = create_app(
        cast(Any, agent.retriever),
        agent=agent,
        agent_models={"fake": agent},
        published_index_identity="synthetic-snapshot-v1",
        investigation_db=path,
    )
    return app, generator


def wait_status(client, run_id, status):
    for _ in range(200):
        row = client.get(f"/api/investigations/{run_id}").json()
        if row["status"] == status:
            return row
        time.sleep(0.02)
    pytest.fail(f"run did not reach {status}: {row}")


def test_real_app_recreation_restore_history_supplement_and_sse_cursor(tmp_path):
    path = tmp_path / "service.sqlite3"
    app1, first_model = make_app(path, SEARCH, READ, CLARIFY)
    with TestClient(app1) as client:
        assert client.get("/api/capabilities").json()["agent_persistence"]
        payload = {"request_id": "create-key", "query": "合成问题", "model": "fake"}
        row = client.post("/api/investigations", json=payload).json()
        run_id = row["run_id"]
        first = wait_status(client, run_id, "waiting_input")
        assert client.post("/api/investigations", json=payload).json()["run_id"] == run_id
        assert len(first_model.calls) == 3
    # New app, executor, generator and connections, reading the same disk file.
    app2, second_model = make_app(path, ANSWER)
    with TestClient(app2) as client:
        restored = client.get(f"/api/investigations/{run_id}").json()
        assert restored["progress"] == first["progress"]
        assert restored["response"]["sources"] == first["response"]["sources"]
        command_payload = {
            "request_id": "supplement-key",
            "version": restored["version"],
            "wait_id": restored["wait_id"],
            "text": "甲",
        }
        assert (
            client.post(
                f"/api/investigations/{run_id}/supplement", json=command_payload
            ).status_code
            == 200
        )
        final = wait_status(client, run_id, "completed")
        assert len(second_model.calls) == 1
        events = client.get(f"/api/investigations/{run_id}/events").json()["events"]
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        stream = client.get(f"/api/investigations/{run_id}/stream?after={first['cursor']}")
        frames = [
            json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")
        ]
        assert [e["seq"] for e in frames] == list(range(first["cursor"] + 1, final["cursor"] + 1))
        repeated = client.get(
            f"/api/investigations/{run_id}/stream", headers={"Last-Event-ID": str(first["cursor"])}
        )
        assert repeated.text == stream.text
        assert len(second_model.calls) == 1
        assert client.get(f"/api/investigations/{run_id}/events?after=9999").status_code == 422
        assert (
            client.post(
                f"/api/investigations/{run_id}/resume",
                json={"request_id": "resume-key", "version": final["version"]},
            ).status_code
            == 409
        )
        assert client.delete(f"/api/investigations/{run_id}").status_code == 200
        assert client.get(f"/api/investigations/{run_id}").status_code == 404


def test_pre_call_pause_wins_transaction_without_charging_unstarted_call(tmp_path):
    store, agent, generator, _, run_id, _ = setup(tmp_path, SEARCH)
    epoch = store.claim(run_id, "worker", agent.profile_fingerprint, "synthetic-snapshot-v1")
    execution = BoundaryExecution(
        store, agent, run_id, owner="worker", epoch=epoch, shutdown=threading.Event()
    )
    command(store, run_id, "pause")
    execution.model_call()
    saved = store.get(run_id)
    assert saved["status"] == "paused"
    assert unpack(saved["checkpoint"]["state"]).model_calls == 0
    assert not generator.calls and not store.attempts(run_id)


def test_checkpoint_is_json_not_opaque_execution_memory():
    cp = new_checkpoint("合成问题")
    restored = unpack(json.loads(json.dumps(cp))["state"])
    assert restored.question == "合成问题" and not restored.evidence


def test_actual_process_exit_after_model_commit_recovers_from_disk(tmp_path):
    script = """
import os, sys, threading
from pathlib import Path
sys.path.insert(0, "tests")
from test_agent import SEARCH, make_agent
from zhrag.investigation_store import InvestigationStore
from zhrag.investigation_executor import InvestigationExecutor, BoundaryExecution
from zhrag.service.app import _AdmissionGate
store = InvestigationStore(Path(sys.argv[1]), lease=0.5)
agent, _ = make_agent(SEARCH)
executor = InvestigationExecutor(store, {"fake": agent}, "synthetic-snapshot-v1", _AdmissionGate(1))
run_id = executor.create("process-create", "合成问题", "fake")
epoch = store.claim(run_id, "dead-process", agent.profile_fingerprint, "synthetic-snapshot-v1")
def crash(point):
    if point == "before_model_result":
        os._exit(17)
BoundaryExecution(store, agent, run_id, owner="dead-process", epoch=epoch,
                  shutdown=threading.Event(), fault=crash).execute()
"""
    path = tmp_path / "killed.sqlite3"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, timeout=15, check=False
    )
    assert result.returncode == 17, result.stderr
    time.sleep(0.6)
    store = InvestigationStore(path)
    store.recover()
    run_id = store.list_runs()[0]
    assert store.get(run_id)["status"] == "interrupted"
    agent, generator = make_agent(READ, ANSWER)
    command(store, run_id, "resume")
    run(store, agent, run_id, owner="new-process")
    assert len(generator.calls) == 2
    assert store.get(run_id)["status"] == "completed"


def test_completed_search_result_survives_pause_without_duplicate_retrieval(tmp_path):
    store, agent, _, _, run_id, _ = setup(tmp_path, SEARCH, READ, ANSWER)
    retriever = agent.retriever
    original = retriever.retrieve

    def pause_search(query):
        result = original(query)
        command(store, run_id, "pause")
        return result

    retriever.retrieve = pause_search
    run(store, agent, run_id)
    assert store.get(run_id)["checkpoint"]["phase"] == "search_result"
    command(store, run_id, "resume", key="resume-key")
    run(store, agent, run_id)
    assert retriever.queries == ["first"]
    assert store.get(run_id)["status"] == "completed"


def test_pause_during_review_keeps_draft_and_saved_review_result(tmp_path):
    review = {
        "coverage": [{"requirement": "合成问题", "covered": True, "evidence_ids": [1]}],
        "blocks": [
            {
                "block": 1,
                "supported": True,
                "condition_scope_supported": True,
                "evidence_ids": [1],
                "finding": "",
            }
        ],
        "suggested_queries": [],
    }
    store, agent, generator, _, run_id, _ = setup(
        tmp_path, SEARCH, READ, ANSWER, review, settings=AgentSettings(review_answers=True)
    )
    original = generator.generate

    def pause_review(system, user):
        result = original(system, user)
        if len(generator.calls) == 4:
            command(store, run_id, "pause")
        return result

    generator.generate = pause_review
    run(store, agent, run_id)
    cp = store.get(run_id)["checkpoint"]
    assert cp["stage"] == "review_answer" and unpack(cp["state"]).pending_answer
    command(store, run_id, "resume", key="resume-key")
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "completed" and len(generator.calls) == 4


def test_total_prompt_budget_stays_spent_across_wait(tmp_path):
    # Two calls each fit the per-call budget but not this cumulative budget.
    store, agent, generator, _, run_id, _ = setup(
        tmp_path, CLARIFY, CLARIFY, settings=AgentSettings(max_total_prompt_tokens=1800)
    )
    run(store, agent, run_id)
    row = store.get(run_id)
    assert row["status"] == "waiting_input"
    command(store, run_id, "supplement", wait_id=row["wait_id"], text="甲版到乙版" * 200)
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "budget_exhausted"
    assert len(generator.calls) == 1


def test_planning_state_round_trips_and_resume_does_not_replan_without_new_facts(tmp_path):
    plan = {
        "requirements": [
            {"id": 1, "kind": "fact", "question_quote": "合成问题", "description": "合成问题"}
        ],
        "facts": [],
    }

    def decision(action, covered=False):
        return {
            "coverage": [
                {
                    "requirement_id": 1,
                    "covered": covered,
                    "evidence": [{"evidence_id": 1, "quote": "合成证据 first"}] if covered else [],
                    "blocks": [1] if covered else [],
                }
            ],
            "dependencies": [],
            "action": action,
        }

    store, agent, generator, _, run_id, _ = setup(
        tmp_path,
        plan,
        decision(SEARCH),
        decision(READ),
        decision(ANSWER, True),
        settings=AgentSettings(plan_investigation=True),
    )

    def pause_after_plan(point):
        if point == "before_decision" and len(generator.calls) == 1:
            command(store, run_id, "pause")

    run(store, agent, run_id, fault=pause_after_plan)
    saved = unpack(store.get(run_id)["checkpoint"]["state"])
    assert saved.plan.requirements[0].description == "合成问题"
    command(store, run_id, "resume", key="resume-key")
    run(store, agent, run_id)
    assert store.get(run_id)["status"] == "completed"
    assert len(generator.calls) == 4


def test_late_old_worker_cannot_overwrite_new_executor_after_safe_takeover(tmp_path):
    store, agent, _, _, run_id, clock = setup(tmp_path, SEARCH, READ, ANSWER)
    epoch = store.claim(run_id, "old", agent.profile_fingerprint, "synthetic-snapshot-v1")
    old_checkpoint = store.get(run_id)["checkpoint"]
    clock.now += 16
    replacement = InvestigationStore(store.path, clock=clock)
    replacement.recover()
    command(replacement, run_id, "resume")
    run(replacement, agent, run_id, owner="new")
    with pytest.raises(LeaseLost):
        store.save(run_id, "old", epoch, old_checkpoint, [], status="paused")
    assert store.get(run_id)["status"] == "completed"


def test_old_worker_cleanup_does_not_remove_new_epoch_heartbeat(tmp_path):
    store, agent, _, executor, run_id, _ = setup(tmp_path, SEARCH)
    # The replacement is registered between the old worker's final commit and
    # its finally block (possible with admission > 1 and immediate resume).
    executor.active[run_id] = (2, threading.current_thread())
    assert executor.admission.try_acquire()
    executor._work(agent, run_id, 1)
    assert executor.active[run_id][0] == 2
    assert store.get(run_id)["status"] == "queued"


def test_cancelling_waiting_run_keeps_evidence_but_removes_stale_question(tmp_path):
    store, agent, _, executor, run_id, _ = setup(tmp_path, SEARCH, READ, CLARIFY)
    run(store, agent, run_id)
    command(store, run_id, "cancel")
    view = public_run(executor, run_id)
    assert view["status"] == "cancelled" and view["wait_id"] is None
    assert view["response"]["status"] == "cancelled"
    assert not view["response"]["clarification"]
    assert view["response"]["sources"][0]["citation_id"] == 1
