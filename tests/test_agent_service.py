from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from test_agent import ABSTAIN, ANSWER, READ, SEARCH, ScriptedGenerator
from test_service import FakeRetriever
from zhrag.agent import DocumentAgent
from zhrag.service.app import create_app


def app_for(*actions: object) -> Any:
    retriever = FakeRetriever()
    return create_app(
        cast(Any, retriever),
        agent=DocumentAgent(retriever, ScriptedGenerator(*actions), "test-index"),
    )


def test_agent_disabled_and_bad_inputs_do_not_call_dependencies() -> None:
    fake = FakeRetriever()
    client = TestClient(create_app(cast(Any, fake)))
    assert client.post("/api/investigate", json={"query": "q"}).status_code == 503
    assert not client.get("/api/capabilities").json()["agent_enabled"]
    for payload in ({"query": " "}, {"query": "q", "top_k": 1}, {"query": "q", "tools": []}):
        assert client.post("/api/investigate", json=payload).status_code == 422
    assert not fake.queries


def test_http_answer_sources_events_and_optional_capability() -> None:
    client = TestClient(app_for(SEARCH, READ, ANSWER))
    capabilities = client.get("/api/capabilities").json()
    assert capabilities["agent_enabled"] and not capabilities["generation_enabled"]
    response = client.post("/api/investigate", json={"query": "合成问题"})
    body = response.json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert body["status"] == "answered" and body["blocks"][0]["citations"] == [1]
    assert len(body["sources"][0]["text"]) == 4005
    assert body["sources"][0]["source_url"] == "https://docs.example.invalid/backup"
    assert body["usage"]["model_calls"] == 3
    assert body["agent_profile"] == capabilities["agent_profile"]
    assert body["events"][-1]["outcome"] == "answered"


def test_http_failure_preserves_evidence_and_releases_admission() -> None:
    app = app_for(SEARCH, READ, SystemExit("private-provider-error"))
    client = TestClient(app)
    response = client.post("/api/investigate", json={"query": "q"})
    assert response.status_code == 503 and response.json()["sources"]
    assert response.json()["blocks"] == [] and "private-provider" not in response.text
    assert client.post("/api/search", json={"query": "q"}).status_code == 200


def test_request_cancellation_keeps_gate_until_worker_stops_and_prevents_next_call() -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        generator = ScriptedGenerator(SEARCH, ABSTAIN)
        original = generator.generate

        def blocking(system: str, user: str) -> str:
            entered.set()
            assert release.wait(5)
            return original(system, user)

        generator.generate = blocking
        fake = FakeRetriever()
        app = create_app(cast(Any, fake), agent=DocumentAgent(fake, generator, "test-index"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(client.post("/api/investigate", json={"query": "q"}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                response = await client.post("/api/search", json={"query": "other"})
                assert response.status_code == 429
            finally:
                release.set()
            for _ in range(100):
                if app.state.admission._active == 0:
                    break
                await asyncio.sleep(0.01)
            assert app.state.admission._active == 0
            assert not fake.queries and len(generator.calls) == 1

    asyncio.run(scenario())
