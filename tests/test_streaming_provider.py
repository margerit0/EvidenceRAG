from __future__ import annotations

import http.client
import io
import json
import urllib.request
from dataclasses import replace
from email.message import Message
from typing import Any

import pytest

from test_agent import make_agent
from zhrag.agent import AgentSettings
from zhrag.answering import GenerationError
from zhrag.generation_control import GenerationInterrupted, generation_control, remaining_seconds
from zhrag.providers import streaming
from zhrag.providers.answering import ChatAnswerGenerator
from zhrag.providers.chat import ChatConfig


def event(content: str | None = None, *, finish: str | None = None, **delta: object) -> bytes:
    return sse(
        {
            "model": "synthetic",
            "choices": [
                {"index": 0, "delta": {"content": content, **delta}, "finish_reason": finish}
            ],
        }
    )


def sse(value: object) -> bytes:
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode("utf-8")


def complete(text: str = '{"action":"abstain"}') -> bytes:
    return event(text) + event(finish="stop") + b"data: [DONE]\n\n"


class Connection:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/event-stream"):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Retry-After"] = "0"
        self.sock = self
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.timeouts: list[float] = []
        self.on_read = lambda: None

    def request(self, *args: object, **kwargs: Any) -> None:
        self.calls.append({"args": args, **kwargs})

    def getresponse(self) -> Connection:
        return self

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name)

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def readline(self, size: int) -> bytes:
        self.on_read()
        return self.body.readline(size)

    def close(self) -> None:
        self.closed = True


def install(monkeypatch: pytest.MonkeyPatch, *connections: Connection) -> list[dict[str, Any]]:
    created = []

    def factory(host: str, port: int, **kwargs: Any) -> Connection:
        created.append({"host": host, "port": port, **kwargs})
        return connections[len(created) - 1]

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stream must not discover proxies or use urllib")

    monkeypatch.setattr(streaming.http.client, "HTTPSConnection", factory)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "getproxies", forbidden)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    return created


def generator(**kwargs: Any) -> ChatAnswerGenerator:
    return ChatAnswerGenerator(
        ChatConfig("PRIVATE_KEY", "https://relay.invalid/v1", "synthetic"),
        stream=True,
        max_retries=0,
        max_output_tokens=None,
        reasoning_effort="high",
        **kwargs,
    )


def test_stream_assembles_unicode_and_separates_reasoning_without_using_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = Connection(
        b": keepalive\r\n\r\n"
        + event(reasoning_content="PRIVATE_REASONING")
        + event('{"合成":')
        + event("true}")
        + event(finish="stop")
        + sse(
            {"error": None, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 0}}
        )
        + b"data: [DONE]\r\n\r\n"
    )
    created = install(monkeypatch, conn)
    observed = []
    answerer = generator(transport=streaming.StreamingTransport(observe=observed.append))
    assert answerer.generate("system", "user") == '{"合成":true}'
    assert conn.closed and len(created) == 1
    assert created[0]["context"].check_hostname
    request = conn.calls[0]
    assert request["args"] == ("POST", "/v1/chat/completions")
    payload = json.loads(request["body"])
    assert payload["stream"] is True and payload["reasoning_effort"] == "high"
    assert payload["stream_options"] == {"include_usage": True}
    assert not {"max_tokens", "max_completion_tokens"} & payload.keys()
    assert "PRIVATE_REASONING" not in repr(observed)
    assert sum(e.get("reasoning_chars", 0) for e in observed) == len("PRIVATE_REASONING")


def test_model_case_variations_pass_both_gates_and_preserve_served_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = Connection(
        event('{"ok":true}').replace(b'"synthetic"', b'"SyNtHeTiC"')
        + event(finish="stop").replace(b'"synthetic"', b'"SYNTHETIC"')
        + b"data: [DONE]\n\n"
    )
    install(monkeypatch, conn)
    replies = []

    def transport(request: urllib.request.Request) -> bytes:
        reply = streaming.StreamingTransport()(request)
        replies.append(json.loads(reply))
        return reply

    assert generator(transport=transport).generate("system", "user") == '{"ok":true}'
    assert replies[0]["model"] == "SyNtHeTiC"
    assert json.loads(conn.calls[0]["body"])["model"] == "synthetic"
    assert conn.closed


@pytest.mark.parametrize(
    "served_model", ["other", "synthetic-v2", "provider/synthetic", " synthetic", "synthetic ", 7]
)
def test_model_drift_after_valid_content_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch, served_model: object
) -> None:
    conn = Connection(
        event("PRIVATE_PARTIAL")
        + sse({"model": served_model, "choices": []})
        + event(finish="stop")
        + b"data: [DONE]\n\n"
    )
    created = install(monkeypatch, conn)
    with pytest.raises(GenerationError, match="generation_failed"):
        replace(generator(), max_retries=9).generate("s", "u")
    assert len(created) == 1 and conn.closed


@pytest.mark.parametrize(
    "bad",
    [
        event("PRIVATE_CONTENT", finish="length") + b"data: [DONE]\n\n",
        event("PRIVATE_CONTENT") + b"data: [DONE]\n\n",
        sse({"model": "wrong", "choices": []}),
        sse({"error": {"message": "PRIVATE_ERROR"}}),
        event(refusal="PRIVATE_REFUSAL"),
        event(tool_calls=[{"PRIVATE_TOOL": True}]),
        sse({"choices": [{"index": True, "delta": {}}]}),
        sse({"choices": None}),
        sse({"choices": [{"index": 0, "delta": {"content": 5}}]}),
        b'data: {"PRIVATE":\n\n',
        b'data: {"choices":[],"choices":[]}\n\n',
        b'data: {"usage":NaN}\n\n',
        b"data: \xff\n\n",
        event(finish="stop") + b"data: [DONE]\n\n",
        event("first", finish="stop") + event("PRIVATE_LATE") + b"data: [DONE]\n\n",
        b'event: error\ndata: {"message":"PRIVATE_ERROR"}\n\n',
    ],
    ids=lambda value: str(len(value)),
)
def test_malformed_stream_fails_closed_without_retry_or_private_error(
    monkeypatch: pytest.MonkeyPatch, bad: bytes
) -> None:
    conn = Connection(bad)
    created = install(monkeypatch, conn)
    with pytest.raises(GenerationError) as error:
        replace(generator(), max_retries=9).generate("s", "u")
    assert str(error.value) == "generation_failed"
    assert "PRIVATE" not in repr(error.value)
    assert conn.closed and len(created) == 1


@pytest.mark.parametrize("status", [200, 302, 403])
def test_wrong_content_type_and_permanent_http_fail_without_reading(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    conn = Connection(b"PRIVATE_ERROR_PAGE", status, "application/json")
    created = install(monkeypatch, conn)
    with pytest.raises(GenerationError):
        replace(generator(), max_retries=9).generate("s", "u")
    assert len(created) == 1 and conn.closed and conn.body.tell() == 0


@pytest.mark.parametrize("attempts", [1, 5, 10])
def test_retryable_headers_then_success_uses_at_most_ten_attempts(
    monkeypatch: pytest.MonkeyPatch, attempts: int
) -> None:
    conns = [Connection(b"PRIVATE_ERROR_PAGE", 504) for _ in range(attempts - 1)] + [
        Connection(complete())
    ]
    created = install(monkeypatch, *conns)
    sleeps = []
    answerer = replace(generator(), max_retries=9, sleep=sleeps.append)
    assert answerer.generate("s", "u") == '{"action":"abstain"}'
    assert len(created) == attempts and all(c.closed for c in conns)
    assert sum(sleeps) == 5 * sum(range(attempts))


def test_tenth_failure_stops_without_an_eleventh_request(monkeypatch: pytest.MonkeyPatch) -> None:
    conns = [Connection(b"PRIVATE_ERROR_PAGE", 504) for _ in range(10)]
    created = install(monkeypatch, *conns)
    with pytest.raises(GenerationError):
        replace(generator(), max_retries=9, sleep=lambda _: None).generate("s", "u")
    assert len(created) == 10 and all(c.closed for c in conns)


def test_disconnected_attempt_never_contaminates_retry_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = Connection(event("PRIVATE_PARTIAL")), Connection(complete())
    created = install(monkeypatch, first, second)
    answerer = replace(generator(), max_retries=1, sleep=lambda _: None)
    assert answerer.generate("s", "u") == '{"action":"abstain"}'
    assert len(created) == 2 and first.closed and second.closed


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("PRIVATE"),
        ConnectionResetError("PRIVATE"),
        http.client.IncompleteRead(b"PRIVATE"),
    ],
)
def test_transport_failure_is_bounded_and_safe(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    conn = Connection(complete())

    def fail() -> None:
        raise failure

    conn.on_read = fail
    install(monkeypatch, conn)
    with pytest.raises(GenerationError) as error:
        generator().generate("s", "u")
    assert error.value.code == (
        "generation_timeout" if isinstance(failure, TimeoutError) else "generation_failed"
    )
    assert "PRIVATE" not in str(error.value) and conn.closed


@pytest.mark.parametrize("cancel", [True, False])
def test_agent_cancels_or_exhausts_budget_while_collecting_without_dispatch(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    now, stopped = [0.0], [False]
    conn = Connection(complete('{"action":"search_docs","query":"must not run"}'))

    def stop() -> None:
        stopped[0] = cancel
        now[0] = 0.0 if cancel else 11.0

    conn.on_read = stop
    install(monkeypatch, conn)
    agent, _ = make_agent(settings=AgentSettings(max_seconds=10))
    agent = replace(agent, generator=generator(), clock=lambda: now[0])
    result = agent.run("synthetic", cancelled=lambda: stopped[0])
    assert result.status == ("cancelled" if cancel else "budget_exhausted")
    assert result.model_calls == 1 and result.search_calls == 0 and not result.blocks
    assert conn.closed and conn.timeouts[0] <= 10
    assert remaining_seconds() == float("inf")


def test_cancel_during_retry_backoff_prevents_next_request(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = Connection(b"", 504)
    created = install(monkeypatch, conn)
    stopped = [False]

    def sleep(_seconds: float) -> None:
        stopped[0] = True

    with (
        generation_control(lambda: stopped[0], lambda: 600),
        pytest.raises(GenerationInterrupted) as error,
    ):
        replace(generator(), max_retries=9, sleep=sleep).generate("s", "u")
    assert error.value.code == "cancelled" and len(created) == 1 and conn.closed


@pytest.mark.parametrize(
    "body,cap",
    [(b":" + b"x" * 70_000 + b"\n", 256 * 1024), (complete(), 30)],
    ids=["line-cap", "wire-cap"],
)
def test_wire_and_line_limits_bound_stream_memory(
    monkeypatch: pytest.MonkeyPatch, body: bytes, cap: int
) -> None:
    conn = Connection(body)
    install(monkeypatch, conn)
    with pytest.raises(GenerationError):
        replace(generator(), max_response_bytes=cap).generate("s", "u")
    assert conn.closed


def test_stream_profile_is_opt_in_and_normal_payload_is_unchanged() -> None:
    answerer = generator()
    assert answerer.profile_fingerprint != replace(answerer, stream=False).profile_fingerprint
    for value in (None, 1, "true"):
        with pytest.raises(ValueError, match="stream must be boolean"):
            replace(answerer, stream=value)
