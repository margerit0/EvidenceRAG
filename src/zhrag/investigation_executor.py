"""Persistent boundary executor sharing DocumentAgent's validators and tools.

The model and retriever remain synchronous. Saved result boundaries are consumed
without repeating the external call. Uncommitted call outcomes are never retried.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict
from typing import Any, Protocol
from urllib.parse import urlsplit

from zhrag.agent import AGENT_MESSAGES, AgentOutcome, AgentStatus, DocumentAgent, _Run
from zhrag.agent_checkpoint import new_checkpoint, pack, unpack
from zhrag.agent_progress import ProgressRecorder
from zhrag.answering import GenerationError
from zhrag.generation_control import GenerationInterrupted, generation_control
from zhrag.investigation_store import InvestigationStore, LeaseLost, RunConflict
from zhrag.tokens import estimate_tokens


class Admission(Protocol):
    def try_acquire(self) -> bool: ...
    def release(self) -> None: ...


def outcome_response(outcome: AgentOutcome, model: str) -> dict[str, Any]:
    sources = []
    for evidence in outcome.evidence:
        url = evidence.ranked.passage.metadata.get("source_url")
        try:
            parsed = urlsplit(str(url))
            safe = (
                isinstance(url, str)
                and len(url) <= 4000
                and not any(ord(char) < 33 for char in url)
                and "\\" not in url
                and parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            safe = False
        sources.append(
            {
                "citation_id": evidence.citation_id,
                "rank": evidence.ranked.rank,
                "title": evidence.title,
                "text": evidence.ranked.passage.text,
                "source_url": url if safe else None,
            }
        )
    return {
        "status": outcome.status,
        "message": AGENT_MESSAGES[outcome.status],
        "blocks": [asdict(block) for block in outcome.blocks],
        "sources": sources,
        "clarification": outcome.clarification,
        "events": [asdict(e) for e in outcome.events],
        "usage": {
            "model_calls": outcome.model_calls,
            "search_calls": outcome.search_calls,
            "read_calls": outcome.read_calls,
            "prompt_estimated_tokens": outcome.prompt_estimated_tokens,
        },
        "total_seconds": outcome.total_seconds,
        "agent_profile": outcome.profile_fingerprint,
        "model": model,
    }


class BoundaryExecution:
    def __init__(
        self,
        store: InvestigationStore,
        agent: DocumentAgent,
        run_id: str,
        *,
        owner: str,
        epoch: int,
        shutdown: threading.Event,
        fault: Callable[[str], None] = lambda _: None,
    ):
        self.store, self.agent, self.run_id = store, agent, run_id
        self.owner, self.epoch, self.shutdown, self.fault = owner, epoch, shutdown, fault
        row = store.get(run_id)
        self.model = row["model"]
        self.cp = row["checkpoint"]
        if self.cp["schema_version"] != 1:
            raise ValueError("unsupported checkpoint schema")
        self.state: _Run = unpack(self.cp["state"])
        self.state.started = agent.clock() - row["active_seconds"]
        self.progress: list[dict[str, Any]] = []
        recorder = ProgressRecorder(
            lambda event: self.progress.append(asdict(event)), lambda: agent._elapsed(self.state)
        )
        recorder.seq = self.cp["recorder"]["seq"]
        recorder.invocation = self.cp["recorder"]["invocation"]
        active = self.cp["recorder"]["active"]
        recorder.active = (active[0], active[1]) if active else None
        self.state.progress = recorder
        supplements = row["supplements"]
        if len(supplements) > self.cp["supplements_applied"]:
            # Facts changed: rebuild the bounded plan/assessment, retaining all
            # evidence IDs, prior observations and cumulative budgets.
            self.state.question = (
                row["query"]
                + "\n"
                + "\n".join(
                    f"补充问题：{item['question']}\n用户补充：{item['text']}"
                    for item in supplements
                )
            )
            self.state.plan = None
            self.state.assessment = None
            self.state.planning_feedback = None
            self.state.pending_answer = None
            self.state.review_feedback = None
            self.cp["supplements_applied"] = len(supplements)

    def save(self, **kwargs: Any) -> str | None:
        recorder = self.state.progress
        assert recorder is not None
        self.cp["state"] = pack(self.state)
        self.cp["recorder"] = {
            "seq": recorder.seq,
            "invocation": recorder.invocation,
            "active": recorder.active,
        }
        result = self.store.save(
            self.run_id, self.owner, self.epoch, self.cp, self.progress, **kwargs
        )
        self.progress.clear()
        return result

    def control(self) -> str:
        return self.store.heartbeat(self.run_id, self.owner, self.epoch)

    def finish(self, outcome: AgentOutcome) -> None:
        status = {
            "answered": "completed",
            "insufficient_evidence": "completed",
            "clarification_needed": "waiting_input",
            "cancelled": "cancelled",
            "budget_exhausted": "budget_exhausted",
        }.get(outcome.status, "failed")
        self.cp["phase"] = "decision" if status == "waiting_input" else "finished"
        self.cp["raw"] = None
        self.state.action = None
        if status == "waiting_input":
            for event in self.progress:
                if event["action"] == "finish" and event["outcome"] == "clarification_needed":
                    event["action"] = "wait_for_input"
        self.save(
            status=status, reason=outcome.status, response=outcome_response(outcome, self.model)
        )

    def execute(self) -> None:  # noqa: PLR0912 - explicit persisted execution boundaries
        while True:
            status = self.control()
            if status == "cancelling":
                self.finish(self.agent._outcome(self.state, "cancelled"))
                return
            if self.agent._elapsed(self.state) >= self.agent.settings.max_seconds:
                self.finish(self.agent._outcome(self.state, "budget_exhausted"))
                return
            if status == "pausing" or self.shutdown.is_set():
                self.save(
                    status="paused",
                    reason="service_shutdown" if self.shutdown.is_set() else "user_paused",
                )
                return
            phase = self.cp["phase"]
            self.fault(f"before_{phase}")
            if phase == "decision":
                if not self.model_call():
                    return
            elif phase == "model_result":
                if not self.consume_model():
                    return
            elif phase == "dispatch":
                if not self.dispatch():
                    return
            elif phase == "search_result":
                if self.cp["error"]:
                    self.agent._event(self.state, "search_docs", "tool_failed")
                else:
                    self.agent._add_candidates(self.state, unpack(self.cp["search_result"]))
                    self.state.successful_searches += 1
                self.cp.update(phase="decision", search_result=None, error=None)
                self.state.action = None
                self.save()
            else:
                raise ValueError("checkpoint is not at a resumable boundary")
            self.fault("after_boundary")

    def model_call(self) -> bool:
        state, agent = self.state, self.agent
        system, prompt = agent._next_prompt(state)
        tokens = estimate_tokens(system + "\n" + prompt) + 32
        if (
            state.model_calls >= agent.settings.max_steps
            or not agent._fits(prompt, system)
            or state.prompt_tokens + tokens > agent.settings.max_total_prompt_tokens
        ):
            self.finish(agent._outcome(state, "budget_exhausted"))
            return False
        state.model_calls += 1
        state.prompt_tokens += tokens
        stage = "review_answer" if state.pending_answer is not None else "decide"
        if agent.settings.plan_investigation and state.plan is None:
            stage = "plan"
        assert state.progress is not None
        state.progress.start(stage, state.model_calls)
        self.cp.update(phase="model_call", stage=stage)
        attempt = self.save(begin=stage)
        if attempt is None:
            # The transactional pre-call check rejected this reservation. Load
            # the last committed boundary, without charging an unstarted call.
            self._stop_before_call()
            return False
        raw, error = None, None
        try:
            with generation_control(
                lambda: self.control() == "cancelling",
                lambda: agent.settings.max_seconds - agent._elapsed(state),
            ):
                raw = agent.generator.generate(system, prompt)
        except (GenerationInterrupted, GenerationError) as exc:
            error = exc.code
        except (Exception, SystemExit):
            error = "generation_failed"
        self.fault("model_returned_before_commit")
        self.cp.update(phase="model_result", raw=raw, error=error)
        self.save(complete=attempt, result=raw, error=error)
        return True

    def _stop_before_call(self) -> None:
        previous = self.store.get(self.run_id)["checkpoint"]
        # Save the persisted boundary, not our speculative counters/progress.
        self.store.save(self.run_id, self.owner, self.epoch, previous, [], status="paused")

    def consume_model(self) -> bool:
        if self.cp["error"]:
            error: AgentStatus = self.cp["error"]
            self.finish(self.agent._outcome(self.state, error))
            return False
        stage = self.cp["stage"]
        self.agent._event(self.state, stage, "ok", observe=stage != "decide")
        self.state.defer_actions = True
        outcome = self.agent._handle_reply(
            self.state, self.cp["raw"], reviewing=stage == "review_answer"
        )
        self.state.defer_actions = False
        self.cp["raw"] = None
        if outcome is not None:
            self.finish(outcome)
            return False
        self.cp["phase"] = "dispatch" if self.state.action else "decision"
        self.save()
        return True

    def dispatch(self) -> bool:
        action = self.state.action
        assert action is not None
        if action["action"] != "search_docs":
            outcome = self.agent._dispatch(self.state, action)
            self.state.action = None
            if outcome is not None:
                self.finish(outcome)
                return False
            self.cp["phase"] = "decision"
            self.save(
                local_operation="read_passage" if action["action"] == "read_passage" else None,
                result={"evidence_ids": list(self.state.evidence)},
            )
            return True
        query = str(action["query"])
        if not self.agent._prepare_search(self.state, query):
            self.state.action = None
            self.cp["phase"] = "decision"
            self.save()
            return True
        self.cp["phase"] = "search_call"
        attempt = self.save(begin="search_docs")
        if attempt is None:
            self._stop_before_call()
            return False
        result, error = None, None
        try:
            result = pack(self.agent.retriever.retrieve(query).passages)
        except (Exception, SystemExit):
            error = "retrieval_failed"
        self.fault("search_returned_before_commit")
        self.cp.update(phase="search_result", search_result=result, error=error)
        self.save(complete=attempt, result=result, error=error)
        return True


class InvestigationExecutor:
    """A local dispatcher; subscribers never own or cancel execution threads."""

    def __init__(
        self,
        store: InvestigationStore,
        models: Mapping[str, DocumentAgent],
        snapshot: str,
        admission: Admission,
    ):
        self.store, self.models, self.snapshot, self.admission = store, models, snapshot, admission
        self.owner = uuid.uuid4().hex
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active: dict[str, tuple[int, threading.Thread]] = {}
        self.admitted: set[tuple[str, int]] = set()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._dispatch, daemon=True)
            self.thread.start()

    def begin_shutdown(self) -> None:
        # Called before the HTTP server drains SSE, not only during lifespan teardown.
        self.stop.set()

    def close(self, timeout: float = 5.0) -> None:
        self.begin_shutdown()
        deadline = time.monotonic() + timeout
        if self.thread:
            self.thread.join(timeout=max(0, deadline - time.monotonic()))
        with self.lock:
            workers = [thread for _, thread in self.active.values()]
        for thread in workers:
            if thread.ident is not None:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        # Unresponsive providers may outlive this deadline. No lease is renewed;
        # expiry fences late writes and unknown outcomes never auto-retry.

    def _dispatch(self) -> None:
        while not self.stop.is_set():
            with suppress(Exception, SystemExit):
                self.tick()
            self.stop.wait(0.25)

    def tick(self) -> None:
        if self.stop.is_set():
            return
        self.store.recover()
        with self.lock:
            active = list(self.active.items())
        for run_id, (active_epoch, worker) in active:
            if not worker.is_alive():
                self._release(run_id, active_epoch)
                continue  # Never keep a dead worker's lease alive.
            with suppress(LeaseLost, RunConflict):
                self.store.heartbeat(run_id, self.owner, active_epoch)
        for run_id in self.store.pending():
            if self.stop.is_set():
                break
            row = self.store.get(run_id)
            if row["status"] not in {"queued", "resuming"}:
                continue
            if not self.admission.try_acquire():
                break
            agent = self.models.get(row["model"])
            try:
                epoch = self.store.claim(
                    run_id, self.owner, agent.profile_fingerprint if agent else "", self.snapshot
                )
            except BaseException:
                self.admission.release()
                raise
            if epoch is None or agent is None:
                self.admission.release()
                continue
            self._start_worker(agent, run_id, epoch)

    def _start_worker(self, agent: DocumentAgent, run_id: str, epoch: int) -> None:
        with self.lock:
            self.admitted.add((run_id, epoch))
        thread = None
        try:
            thread = threading.Thread(target=self._work, args=(agent, run_id, epoch), daemon=True)
            # Registration and start are indivisible to a concurrent tick.
            with self.lock:
                self.active[run_id] = (epoch, thread)
                thread.start()
        except BaseException:
            if thread is None or thread.ident is None:
                try:
                    cp = self.store.get(run_id)["checkpoint"]
                    self.store.save(
                        run_id,
                        self.owner,
                        epoch,
                        cp,
                        [],
                        status="interrupted",
                        reason="worker_start_failed",
                    )
                except (LeaseLost, RunConflict):
                    pass
                finally:
                    self._release(run_id, epoch)
            raise

    def _release(self, run_id: str, epoch: int) -> None:
        with self.lock:
            if (run_id, epoch) in self.admitted:
                self.admitted.remove((run_id, epoch))
                self.admission.release()
            active = self.active.get(run_id)
            if active is not None and active[0] == epoch:
                self.active.pop(run_id)

    def _work(self, agent: DocumentAgent, run_id: str, epoch: int) -> None:
        try:
            BoundaryExecution(
                self.store, agent, run_id, owner=self.owner, epoch=epoch, shutdown=self.stop
            ).execute()
        except LeaseLost:
            pass
        except (Exception, SystemExit):
            try:
                cp = self.store.get(run_id)["checkpoint"]
                self.store.save(
                    run_id,
                    self.owner,
                    epoch,
                    cp,
                    [],
                    status="unrecoverable",
                    reason="executor_failed",
                )
            except (Exception, SystemExit):
                pass
        finally:
            self._release(run_id, epoch)

    def create(self, key: str, query: str, model: str) -> str:
        agent = self.models.get(model)
        if agent is None:
            raise RunConflict("invalid_model", 422)
        return self.store.create(
            key,
            query,
            model,
            profile=agent.profile_fingerprint,
            snapshot=self.snapshot,
            settings=asdict(agent.settings),
            checkpoint=new_checkpoint(query),
        )
