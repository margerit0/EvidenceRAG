"""Durable run API. Subscription cancellation never controls the executor."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Header, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from zhrag.agent import AgentOutcome, _Run
from zhrag.agent_checkpoint import unpack
from zhrag.investigation_executor import InvestigationExecutor, outcome_response
from zhrag.investigation_store import TERMINAL, RunConflict


class CreateRun(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: Annotated[str, Field(min_length=8, max_length=128)]
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    model: Annotated[str, Field(min_length=1, max_length=120)] | None = None


class RunCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: Annotated[str, Field(min_length=8, max_length=128)]
    version: Annotated[int, Field(strict=True, ge=1)]
    wait_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    text: Annotated[str, Field(min_length=1, max_length=2000)] | None = None


def public_run(executor: InvestigationExecutor, run_id: str) -> dict[str, Any]:
    store = executor.store
    row = store.get(run_id)
    state: _Run = unpack(row["checkpoint"]["state"])
    cursor = row["next_seq"] - 1
    events: list[dict[str, Any]] = []
    after = 0
    while after < cursor:
        batch = store.events(run_id, after)
        events.extend(item for item in batch if item["seq"] <= cursor)
        if not batch:
            break
        after = batch[-1]["seq"]
    progress = [
        {**item["payload"], "seq": item["seq"], "run_id": run_id}
        for item in events
        if item["kind"] == "progress"
    ]
    response = row["response"] or outcome_response(
        AgentOutcome(
            "cancelled" if row["status"] == "cancelled" else "clarification_needed",
            (),
            tuple(state.evidence.values()),
            row["clarification"],
            tuple(state.events),
            state.model_calls,
            state.search_calls,
            state.read_calls,
            state.prompt_tokens,
            row["active_seconds"],
            row["profile"],
        ),
        row["model"],
    )
    response["total_seconds"] = row["active_seconds"]
    return {
        "schema_version": row["schema_version"],
        "run_id": run_id,
        "query": row["query"],
        "model": row["model"],
        "status": row["status"],
        "reason": row["reason"],
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "active_seconds": row["active_seconds"],
        "settings": row["settings"],
        "wait_id": row["wait_id"],
        "clarification": row["clarification"],
        "supplements": row["supplements"],
        "response": response,
        "progress": progress,
        "cursor": cursor,
        "boundary": row["checkpoint"]["phase"],
        "checkpoint_revision": row["checkpoint_revision"],
        "attempts": store.attempts(run_id),
    }


def install_investigations(
    app: FastAPI, executor: InvestigationExecutor, default_model: str
) -> None:
    router = APIRouter(prefix="/api/investigations")
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}

    @app.exception_handler(RunConflict)
    async def conflict(_request: Any, exc: RunConflict) -> JSONResponse:
        return JSONResponse(
            {"code": exc.code, "message": exc.code}, status_code=exc.status, headers=headers
        )

    @router.post("")
    def create(payload: CreateRun) -> JSONResponse:
        run_id = executor.create(payload.request_id, payload.query, payload.model or default_model)
        return JSONResponse(public_run(executor, run_id), status_code=201, headers=headers)

    @router.get("")
    def listing() -> JSONResponse:
        executor.store.recover()
        # The list is intentionally compact. Full evidence/history is fetched on selection.
        rows = [executor.store.get(run_id) for run_id in executor.store.list_runs()]
        return JSONResponse(
            {
                "schema_version": 1,
                "runs": [
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "query",
                            "model",
                            "status",
                            "reason",
                            "created_at",
                            "version",
                        )
                    }
                    for row in rows
                ],
            },
            headers=headers,
        )

    @router.get("/{run_id}")
    def get(run_id: str) -> JSONResponse:
        executor.store.recover()
        return JSONResponse(public_run(executor, run_id), headers=headers)

    @router.post("/{run_id}/{action}")
    def command(run_id: str, action: str, payload: RunCommand) -> JSONResponse:
        if action not in {"pause", "resume", "cancel", "supplement"}:
            raise RunConflict("unknown_operation", 404)
        if action != "supplement" and (payload.wait_id is not None or payload.text is not None):
            raise RunConflict("unexpected_supplement", 422)
        executor.store.command(
            run_id,
            payload.request_id,
            action,
            version=payload.version,
            wait_id=payload.wait_id,
            text=payload.text,
        )
        return JSONResponse(public_run(executor, run_id), headers=headers)

    @router.delete("/{run_id}")
    def delete(run_id: str) -> JSONResponse:
        executor.store.delete(run_id)
        return JSONResponse({"deleted": run_id}, headers=headers)

    @router.get("/{run_id}/events")
    def events(run_id: str, after: Annotated[int, Query(ge=0)] = 0) -> JSONResponse:
        return JSONResponse({"events": executor.store.events(run_id, after)}, headers=headers)

    @router.get("/{run_id}/stream")
    def stream(
        run_id: str,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        if last_event_id is not None:
            try:
                after = int(last_event_id)
            except ValueError:
                raise RunConflict("invalid_event_cursor", 422) from None
        executor.store.events(run_id, after)  # Validate before sending headers.

        async def frames() -> AsyncIterator[str]:
            cursor = after
            heartbeat = time.monotonic()
            while not executor.stop.is_set():
                # Durable log is the one source for both catch-up and live events;
                # there is no lossy handoff to an in-memory subscriber queue.
                batch = await asyncio.to_thread(executor.store.events, run_id, cursor)
                for item in batch:
                    cursor = item["seq"]
                    data = json.dumps(item, ensure_ascii=False)
                    yield f"id: {cursor}\nevent: run_event\ndata: {data}\n\n"
                row = await asyncio.to_thread(executor.store.get, run_id)
                if row["status"] in TERMINAL and cursor >= row["next_seq"] - 1:
                    return
                if time.monotonic() - heartbeat >= 10:
                    yield ": keep-alive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.2)

        return StreamingResponse(
            frames(), media_type="text/event-stream", headers={**headers, "X-Accel-Buffering": "no"}
        )

    app.include_router(router)
