"""Request-local cooperative cancellation and remaining generation budget."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


class GenerationInterrupted(Exception):
    def __init__(self, code: Literal["cancelled", "budget_exhausted"]) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class GenerationControl:
    cancelled: Callable[[], bool]
    seconds_left: Callable[[], float]

    def remaining(self) -> float:
        if self.cancelled():
            raise GenerationInterrupted("cancelled")
        remaining = self.seconds_left()
        if remaining <= 0:
            raise GenerationInterrupted("budget_exhausted")
        return remaining


_CONTROL: ContextVar[GenerationControl | None] = ContextVar("generation_control", default=None)


def remaining_seconds() -> float:
    control = _CONTROL.get()
    return control.remaining() if control is not None else float("inf")


@contextmanager
def generation_control(
    cancelled: Callable[[], bool], seconds_left: Callable[[], float]
) -> Iterator[None]:
    token = _CONTROL.set(GenerationControl(cancelled, seconds_left))
    try:
        yield
    finally:
        _CONTROL.reset(token)
