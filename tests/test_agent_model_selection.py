from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from test_agent import ABSTAIN, ANSWER, READ, SEARCH, ScriptedGenerator
from test_service import FakeRetriever
from zhrag.agent import DocumentAgent
from zhrag.service.app import create_app


def configured_app() -> tuple[Any, ScriptedGenerator, ScriptedGenerator]:
    fake = FakeRetriever()
    first = ScriptedGenerator(SEARCH, READ, ANSWER)
    second = ScriptedGenerator(ABSTAIN)
    second.profile_fingerprint = "b" * 64
    agents = {
        "deepseek-v4.1-flash": DocumentAgent(fake, first, "test-index"),
        "z-ai/glm-5.3": DocumentAgent(fake, second, "test-index"),
    }
    return (
        create_app(cast(Any, fake), agent=agents["deepseek-v4.1-flash"], agent_models=agents),
        first,
        second,
    )


def final_response(response: httpx.Response, streamed: bool) -> dict[str, Any]:
    if not streamed:
        return cast(dict[str, Any], response.json())
    frame = next(
        block for block in response.text.split("\n\n") if block.startswith("event: result")
    )
    return cast(dict[str, Any], json.loads(frame.splitlines()[1][6:])["response"])


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("model", [None, "deepseek-v4.1-flash", "z-ai/glm-5.3"])
def test_selected_model_controls_agent_and_result_profile(
    streamed: bool, model: str | None
) -> None:
    app, first, second = configured_app()
    client = TestClient(app)
    caps = client.get("/api/capabilities").json()
    assert caps["agent_models"] == [
        {"id": "deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash"},
        {"id": "z-ai/glm-5.3", "name": "GLM 5.3"},
    ]
    assert caps["default_agent_model"] == "deepseek-v4.1-flash"
    response = client.post(
        "/api/investigate" + ("/stream" if streamed else ""),
        json={"query": "合成模型选择问题", **({"model": model} if model else {})},
    )
    assert response.status_code == 200
    result = final_response(response, streamed)
    assert result["model"] == (model or caps["default_agent_model"])
    if model == "z-ai/glm-5.3":
        assert result["status"] == "insufficient_evidence"
        assert not first.calls and len(second.calls) == 1
        assert result["agent_profile"] != caps["agent_profile"]
    else:
        assert result["status"] == "answered"
        assert len(first.calls) == 3 and not second.calls
        assert result["agent_profile"] == caps["agent_profile"]


@pytest.mark.parametrize("endpoint", ["/api/investigate", "/api/investigate/stream"])
@pytest.mark.parametrize("model", ["unknown", "https://other.invalid", "", 5])
def test_unavailable_or_malformed_model_does_not_call_any_agent(
    endpoint: str, model: object
) -> None:
    app, first, second = configured_app()
    client = TestClient(app)
    response = client.post(endpoint, json={"query": "q", "model": model})
    assert response.status_code == 422
    assert not first.calls and not second.calls
    assert app.state.admission._active == 0


def test_different_models_share_admission_without_switching_active_run() -> None:
    async def scenario() -> None:
        app, first, second = configured_app()
        entered, release = threading.Event(), threading.Event()
        original = first.generate

        def blocked(system: str, user: str) -> str:
            entered.set()
            assert release.wait(5)
            return original(system, user)

        first.generate = blocked
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            running = asyncio.create_task(
                client.post("/api/investigate", json={"query": "q", "model": "deepseek-v4.1-flash"})
            )
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                other = await client.post(
                    "/api/investigate/stream", json={"query": "q", "model": "z-ai/glm-5.3"}
                )
                assert other.status_code == 429 and not second.calls
            finally:
                release.set()
            result = (await running).json()
            assert result["model"] == "deepseek-v4.1-flash" and result["status"] == "answered"
            assert app.state.admission._active == 0

    asyncio.run(scenario())


def test_model_options_must_share_the_default_agent_and_retriever() -> None:
    fake = FakeRetriever()
    default = DocumentAgent(fake, ScriptedGenerator(ABSTAIN), "test-index")
    other = DocumentAgent(FakeRetriever(), ScriptedGenerator(ABSTAIN), "test-index")
    with pytest.raises(ValueError, match="share"):
        create_app(cast(Any, fake), agent=default, agent_models={"other": other})
    with pytest.raises(ValueError, match="default agent"):
        create_app(cast(Any, fake), agent_models={"default": default})
