"""Request-local UI observations, separate from model-visible action history."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AgentProgress:
    seq: int
    invocation_id: str
    step: int
    action: str
    phase: Literal["started", "completed"]
    elapsed_seconds: float
    outcome: str | None = None
    evidence_ids: tuple[int, ...] = ()


class ProgressRecorder:
    """Observers cannot change the prompt, tool results, or terminal status."""

    def __init__(self, observe: Callable[[AgentProgress], None], clock: Callable[[], float]):
        self.observe = observe
        self.clock = clock
        self.seq = 0
        self.invocation = 0
        self.active: tuple[str, int] | None = None

    def _emit(
        self,
        action: str,
        step: int,
        phase: Literal["started", "completed"],
        outcome: str | None = None,
        ids: tuple[int, ...] = (),
    ) -> None:
        self.seq += 1
        event = AgentProgress(
            self.seq, f"call-{self.invocation}", step, action, phase, self.clock(), outcome, ids
        )
        # The optional display sink must not turn a valid answer into a model failure.
        with suppress(Exception):
            self.observe(event)

    def start(self, action: str, step: int) -> None:
        self.invocation += 1
        self.active = (action, step)
        self._emit(action, step, "started")

    def complete(self, outcome: str, ids: tuple[int, ...] = ()) -> None:
        if self.active is not None:
            action, step = self.active
            self._emit(action, step, "completed", outcome, ids)
            self.active = None

    def record(self, action: str, step: int, outcome: str, ids: tuple[int, ...] = ()) -> None:
        if self.active is not None:
            matched = self.active[0] == action
            self.complete(outcome, ids)
            if matched:
                return
        # Some validation/terminal records are instantaneous observations. Do not invent starts.
        self.invocation += 1
        self._emit(action, step, "completed", outcome, ids)
