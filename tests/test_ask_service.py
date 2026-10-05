from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from test_answering import FakeGenerator
from test_answering_provider import Recorder, body
from test_online import retriever
from test_service import FakeRetriever, result
from zhrag.answering import Answerer, AnswerSettings, GenerationError
from zhrag.providers.answering import ChatAnswerGenerator
from zhrag.providers.chat import ChatConfig
from zhrag.service.app import create_app
from zhrag.service.observability import InMemoryTraceSink


def app_for(
    generator: FakeGenerator,
    *,
    fake: FakeRetriever | None = None,
    settings: AnswerSettings | None = None,
    sink: Any = None,
) -> Any:
    return create_app(
        cast(Any, fake or FakeRetriever()),
        answerer=Answerer(generator, settings or AnswerSettings()),
        trace_sink=sink,
    )


def test_disabled_ask_and_capabilities_do_not_retrieve() -> None:
    fake = FakeRetriever()
    http = TestClient(create_app(cast(Any, fake)))
    assert http.get("/api/capabilities").json() == {
        "generation_enabled": False,
        "output_limit": 2,
        "generation_profile": None,
        "agent_enabled": False,
        "agent_profile": None,
        "agent_streaming": False,
        "agent_models": [],
        "default_agent_model": None,
    }
    response = http.post("/api/ask", json={"query": "question"})
    assert response.status_code == 503
    assert response.json()["code"] == "generation_unavailable"
    assert not fake.queries


def test_answer_uses_untruncated_evidence_and_returns_sources() -> None:
    generator = FakeGenerator()
    sink = InMemoryTraceSink()
    http = TestClient(app_for(generator, sink=sink))
    response = http.post("/api/ask", json={"query": " question ", "top_k": 1})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "answered"
    assert body["blocks"][0]["citations"] == [1]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_url"] == "https://docs.example.invalid/backup"
    assert len(body["sources"][0]["text"]) == 4005
    assert len(body["retrieval"]["passages"][0]["text"]) == 4000
    prompt = json.loads(generator.calls[0][1])
    assert prompt["question"] == "question"
    assert len(prompt["evidence"][0]["text"]) == 4005
    assert body["generation_seconds"] >= 0 and body["total_seconds"] >= 0
    assert len(body["generation_profile"]) == 64
    assert len(sink.events) == 2
    serialized = json.dumps(sink.events)
    assert "question" not in serialized and "AAAA" not in serialized
    assert "合成回答" not in serialized
    assert http.get("/api/capabilities").json()["generation_enabled"] is True


@pytest.mark.parametrize(
    "reply,status,http_status",
    [
        ('{"answerable":false,"blocks":[]}', "insufficient_evidence", 200),
        (
            '{"answerable":true,"blocks":[{"text":"private","citations":[99]}]}',
            "invalid_answer",
            503,
        ),
        (GenerationError("generation_timeout"), "generation_timeout", 503),
        (SystemExit("private provider token"), "generation_failed", 503),
        (
            json.dumps({"answerable": True, "blocks": [{"text": "\ud800", "citations": [1]}]}),
            "invalid_answer",
            503,
        ),
    ],
)
def test_failed_generation_keeps_evidence_and_never_publishes_partial_answer(
    reply: str | BaseException,
    status: str,
    http_status: int,
) -> None:
    http = TestClient(app_for(FakeGenerator(reply)))
    response = http.post("/api/ask", json={"query": "question"})
    assert response.status_code == http_status
    assert response.json()["status"] == status
    assert response.json()["blocks"] == []
    assert response.json()["retrieval"]["passages"]
    assert "private" not in response.text
    assert http.post("/api/search", json={"query": "q"}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "q", "top_k": 3},
        {"query": "q", "top_k": True},
        {"query": "q", "model": "x"},
    ],
)
def test_invalid_ask_does_not_call_dependencies(payload: object) -> None:
    fake, generator = FakeRetriever(), FakeGenerator()
    response = TestClient(app_for(generator, fake=fake)).post("/api/ask", json=payload)
    assert response.status_code == 422
    assert not generator.calls and not fake.queries


def test_retrieval_error_stops_generation() -> None:
    fake, generator = FakeRetriever(error=RuntimeError("PRIVATE")), FakeGenerator()
    response = TestClient(app_for(generator, fake=fake)).post("/api/ask", json={"query": "q"})
    assert response.status_code == 503
    assert response.json()["code"] == "retrieval_failed"
    assert "PRIVATE" not in response.text
    assert not generator.calls


def test_context_budget_failure_returns_retrieval_without_chat() -> None:
    generator = FakeGenerator()
    response = TestClient(app_for(generator, settings=AnswerSettings(max_prompt_tokens=1))).post(
        "/api/ask", json={"query": "q"}
    )
    assert response.json()["status"] == "context_limit"
    assert response.json()["retrieval"]["passages"]
    assert not generator.calls


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "https://user:pass@example.com",
        "https://:pass@example.com",
        "file:///C:/private",
        "https://example.com\n/path",
    ],
)
def test_source_urls_are_restricted(url: str) -> None:
    fake = FakeRetriever()

    def source(query: str) -> Any:
        original = result(query, settings=fake.settings)
        row = original.passages[0]
        row = replace(row, passage=replace(row.passage, metadata={"source_url": url}))
        return replace(original, passages=(row,))

    app = create_app(cast(Any, fake), answerer=Answerer(FakeGenerator()), search_runner=source)
    response = TestClient(app).post("/api/ask", json={"query": "q"})
    assert response.json()["sources"][0]["source_url"] is None


def test_real_retrieval_and_chat_adapter_complete_an_http_answer() -> None:
    recorder = Recorder(
        body(
            content=json.dumps(
                {
                    "answerable": True,
                    "blocks": [{"text": "合成端到端回答", "citations": [1, 2]}],
                }
            )
        )
    )
    online = retriever()
    generator = ChatAnswerGenerator(
        ChatConfig("synthetic", "https://example.invalid", "model"), transport=recorder
    )
    http = TestClient(create_app(online, answerer=Answerer(generator)))
    response = http.post("/api/ask", json={"query": "合成端到端问题"})
    assert response.status_code == 200
    assert response.json()["status"] == "answered"
    request = json.loads(recorder.requests[0].data or b"{}")
    evidence = json.loads(request["messages"][1]["content"])["evidence"]
    assert [item["text"] for item in evidence] == [
        item["text"] for item in response.json()["sources"]
    ]
    assert len(evidence) == 2
    assert response.json()["blocks"][0]["citations"] == [1, 2]
    assert len(recorder.requests) == 1
    assert http.post("/api/search", json={"query": "独立检索"}).status_code == 200
    assert len(recorder.requests) == 1


def test_generation_retry_reuses_retrieval_and_holds_admission() -> None:
    fake = FakeRetriever()
    calls = 0
    sleeps: list[float] = []
    app: Any = None

    def transport(_request: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("private")
        return body(
            content=json.dumps(
                {
                    "answerable": True,
                    "blocks": [{"text": "合成回答", "citations": [1]}],
                }
            )
        )

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        assert app.state.admission._active == 1
        assert fake.queries == ["q"]

    generator = ChatAnswerGenerator(
        ChatConfig("synthetic", "https://example.invalid", "model"),
        transport=transport,
        sleep=sleep,
    )
    app = create_app(cast(Any, fake), answerer=Answerer(generator))
    response = TestClient(app).post("/api/ask", json={"query": "q"})
    assert response.status_code == 200
    assert calls == 2 and sleeps == [5]
    assert fake.queries == ["q"]
    assert app.state.admission._active == 0


def test_cancellation_does_not_release_generation_admission_early() -> None:
    entered, release = threading.Event(), threading.Event()

    class BlockingGenerator(FakeGenerator):
        def generate(self, system: str, user: str) -> str:
            entered.set()
            assert release.wait(5)
            return super().generate(system, user)

    app = app_for(BlockingGenerator())

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            pending = asyncio.create_task(http.post("/api/ask", json={"query": "first"}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                assert (await http.post("/api/search", json={"query": "busy"})).status_code == 429
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert (
                    await http.post("/api/ask", json={"query": "still-busy"})
                ).status_code == 429
            finally:
                release.set()
            for _ in range(200):
                if app.state.admission._active == 0:
                    break
                await asyncio.sleep(0.005)
            assert app.state.admission._active == 0
            assert (await http.post("/api/search", json={"query": "available"})).status_code == 200

    asyncio.run(exercise())
