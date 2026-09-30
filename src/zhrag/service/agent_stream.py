"""Bounded, ephemeral SSE transport for one document investigation.

POST creates exactly one run. A disconnect requests cooperative cancellation;
the worker retains its admission slot until its blocking dependencies return.
There is no automatic retry, persistent trace store, or replay subscription.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from zhrag.agent_progress import AgentProgress

QUEUE_LIMIT = 256


class _CancellableStream(StreamingResponse):
    stop: threading.Event

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # ASGI 2.4 may report a disconnect from send(), while the generator is
            # suspended at yield. Do not rely on async-generator garbage collection.
            self.stop.set()


def _frame(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"


class ProgressFeed:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex
        self.stop = threading.Event()
        self.overflow = threading.Event()
        self.events: queue.Queue[AgentProgress] = queue.Queue(maxsize=QUEUE_LIMIT)

    def observe(self, event: AgentProgress) -> None:
        if self.stop.is_set():
            return
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.overflow.set()
            self.stop.set()

    def response(self, work: asyncio.Task[JSONResponse]) -> StreamingResponse:
        async def frames() -> AsyncIterator[str]:
            try:
                yield _frame("run_started", {"run_id": self.run_id})
                heartbeat = time.monotonic()
                while not work.done() or not self.events.empty():
                    try:
                        event = self.events.get_nowait()
                    except queue.Empty:
                        if time.monotonic() - heartbeat >= 10:
                            yield ": keep-alive\n\n"
                            heartbeat = time.monotonic()
                        await asyncio.sleep(0.025)
                    else:
                        yield _frame("progress", {"run_id": self.run_id, **asdict(event)})
                response = await asyncio.shield(work)
                if self.overflow.is_set():
                    yield _frame("failure", {"run_id": self.run_id, "code": "stream_overflow"})
                else:
                    body = json.loads(bytes(response.body))
                    if "blocks" in body:
                        yield _frame("result", {"run_id": self.run_id, "response": body})
                    else:
                        yield _frame(
                            "failure",
                            {"run_id": self.run_id, "code": body.get("code", "agent_failed")},
                        )
            finally:
                self.stop.set()

        response = _CancellableStream(
            frames(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )
        response.stop = self.stop
        return response
