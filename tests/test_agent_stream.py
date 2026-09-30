from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from test_agent import ANSWER, READ, SEARCH
from test_agent_service import app_for
from test_service import FakeRetriever
from zhrag.agent_progress import AgentProgress
from zhrag.io_utils import write_text
from zhrag.service.agent_stream import QUEUE_LIMIT, ProgressFeed
from zhrag.service.app import create_app


def parse_frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (block.splitlines()[0][7:], json.loads(block.splitlines()[1][6:]))
        for block in text.strip().split("\n\n")
        if block.startswith("event:")
    ]


def test_stream_http_has_ordered_calls_and_authoritative_result() -> None:
    client = TestClient(app_for(SEARCH, READ, ANSWER))
    assert client.get("/api/capabilities").json()["agent_streaming"]
    response = client.post("/api/investigate/stream", json={"query": "合成问题"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    frames = parse_frames(response.text)
    assert frames[0][0] == "run_started" and frames[-1][0] == "result"
    assert len({payload["run_id"] for _, payload in frames}) == 1
    events = [payload for event, payload in frames if event == "progress"]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["action"] == "decide" and events[0]["phase"] == "started"
    assert frames[-1][1]["response"]["status"] == "answered"
    assert frames[-1][1]["response"]["blocks"][0]["citations"] == [1]
    assert client.post("/api/search", json={"query": "q"}).status_code == 200


def test_stream_failures_keep_evidence_and_public_status() -> None:
    client = TestClient(app_for(SEARCH, READ, RuntimeError("PRIVATE_PROVIDER_ERROR")))
    response = client.post("/api/investigate/stream", json={"query": "合成问题"})
    body = parse_frames(response.text)[-1][1]["response"]
    assert body["status"] == "generation_failed" and body["sources"]
    assert body["blocks"] == [] and "PRIVATE_PROVIDER_ERROR" not in response.text


def test_stream_checks_disabled_input_and_shared_admission_before_headers() -> None:
    fake = FakeRetriever()
    disabled = TestClient(create_app(cast(Any, fake)))
    assert disabled.post("/api/investigate/stream", json={"query": "q"}).status_code == 503
    assert not fake.queries
    app = app_for(ANSWER)
    client = TestClient(app)
    for payload in ({"query": ""}, {"query": "q", "tools": []}):
        assert client.post("/api/investigate/stream", json=payload).status_code == 422
    assert app.state.admission.try_acquire()
    assert client.post("/api/investigate/stream", json={"query": "q"}).status_code == 429
    app.state.admission.release()


def test_feed_delivers_progress_before_completion_and_disconnect_requests_stop() -> None:
    async def scenario() -> None:
        feed = ProgressFeed()
        finish = asyncio.Event()

        async def worker() -> JSONResponse:
            feed.observe(AgentProgress(1, "call-1", 1, "decide", "started", 0))
            await finish.wait()
            return JSONResponse({"blocks": [], "status": "cancelled"})

        work = asyncio.create_task(worker())
        iterator = feed.response(work).body_iterator
        assert "run_started" in str(await anext(iterator))
        assert "progress" in str(await asyncio.wait_for(anext(iterator), 1))
        assert not work.done()
        await cast(Any, iterator).aclose()
        assert feed.stop.is_set()
        assert not work.done()  # The worker, and therefore its admission slot, stays alive.
        finish.set()
        await work

    asyncio.run(scenario())


def test_overflow_is_bounded_and_never_reports_a_complete_stream() -> None:
    async def scenario() -> None:
        feed = ProgressFeed()
        for seq in range(QUEUE_LIMIT + 1):
            feed.observe(AgentProgress(seq + 1, f"call-{seq}", 1, "decide", "started", 0))
        assert feed.events.qsize() == QUEUE_LIMIT and feed.stop.is_set()

        async def worker() -> JSONResponse:
            return JSONResponse({"blocks": [], "status": "cancelled"})

        work = asyncio.create_task(worker())
        chunks = [str(chunk) async for chunk in feed.response(work).body_iterator]
        assert "stream_overflow" in chunks[-1] and "event: result" not in "".join(chunks)

    asyncio.run(scenario())


def test_asgi_send_disconnect_immediately_requests_worker_cancellation() -> None:
    async def scenario() -> None:
        feed = ProgressFeed()
        finish = asyncio.Event()

        async def worker() -> JSONResponse:
            await finish.wait()
            return JSONResponse({"blocks": []})

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b""}

        async def send(_message: object) -> None:
            raise OSError("client disconnected")

        work = asyncio.create_task(worker())
        response = feed.response(work)
        try:
            with pytest.raises(ClientDisconnect):
                await response(
                    {"type": "http", "asgi": {"spec_version": "2.4"}},
                    cast(Any, receive),
                    send,
                )
            assert feed.stop.is_set() and not work.done()
        finally:
            finish.set()
            await work

    asyncio.run(scenario())


def test_workbench_serves_only_explicit_build_and_blocks_traversal(tmp_path: Path) -> None:
    write_text(tmp_path / "index.html", "<title>Workbench fixture</title>")
    app = create_app(cast(Any, FakeRetriever()), frontend_dir=tmp_path)
    client = TestClient(app)
    assert "Workbench fixture" in client.get("/workbench/").text
    assert client.get("/workbench/missing.js").status_code == 404
    assert client.get("/workbench/%2e%2e/pyproject.toml").status_code == 404
    assert client.get("/api/capabilities").status_code == 200
