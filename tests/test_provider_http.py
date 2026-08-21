"""Contract tests for the shared One Hub JSON transport."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from zhrag.providers.http import JsonClient


class Recorder:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> bytes:
        self.requests.append(request)
        return self.body

    def client(self) -> JsonClient:
        return JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            transport=self,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )


class TestResponseEnvelope:
    def test_accepts_a_json_object(self) -> None:
        recorder = Recorder(b'{"ok":true}')
        assert recorder.client().post({"x": 1}) == {"ok": True}

    @pytest.mark.parametrize("value", [[], [1], "text", 1, True, None])
    def test_rejects_a_non_object_top_level(self, value: Any) -> None:
        body = json.dumps(value).encode()
        with pytest.raises(SystemExit, match="expected a JSON object"):
            Recorder(body).client().post({})

    def test_rejects_malformed_json_with_endpoint_context(self) -> None:
        with pytest.raises(SystemExit, match=r"malformed JSON from https://relay\.example"):
            Recorder(b"{not-json").client().post({})

    def test_rejects_invalid_utf8_with_endpoint_context(self) -> None:
        with pytest.raises(SystemExit, match=r"malformed JSON from https://relay\.example"):
            Recorder(b"\xff").client().post({})

    def test_sends_bearer_json_and_explicit_user_agent(self) -> None:
        recorder = Recorder(b"{}")
        recorder.client().post({"text": "中文"})
        request = recorder.requests[0]
        assert request.get_header("Authorization") == "Bearer secret"
        assert request.get_header("Content-type") == "application/json"
        assert request.get_header("User-agent", "").startswith("zhrag/")
        assert json.loads(request.data or b"{}") == {"text": "中文"}


class TestTransientFailures:
    def test_retries_a_connection_reset(self) -> None:
        outcomes: list[bytes | Exception] = [
            ConnectionResetError(10054, "connection reset"),
            b'{"ok":true}',
        ]
        requests: list[urllib.request.Request] = []
        slept: list[float] = []

        def transport(request: urllib.request.Request) -> bytes:
            requests.append(request)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        client = JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            transport=transport,
            sleep=slept.append,
            log=lambda _message: None,
        )

        assert client.post({}) == {"ok": True}
        assert len(requests) == 2
        assert slept == [5.0]

    def test_retries_cloudflare_520(self) -> None:
        outcomes: list[bytes | Exception] = [
            urllib.error.HTTPError(
                "https://relay.example/v1/test",
                520,
                "web server returned an unknown error",
                email.message.Message(),
                io.BytesIO(b"provider body"),
            ),
            b'{"ok":true}',
        ]
        slept: list[float] = []

        def transport(_request: urllib.request.Request) -> bytes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        client = JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            transport=transport,
            sleep=slept.append,
            log=lambda _message: None,
        )

        assert client.post({}) == {"ok": True}
        assert slept == [5.0]

    def test_connection_reset_exhaustion_has_controlled_diagnostic(self) -> None:
        client = JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            retries=2,
            transport=lambda _request: (_ for _ in ()).throw(
                ConnectionResetError(10054, "connection reset")
            ),
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )

        with pytest.raises(SystemExit, match="giving up after 2 attempts"):
            client.post({})


class TestErrorPrivacy:
    def test_permanent_http_error_does_not_expose_provider_body(self) -> None:
        canary = "PRIVATE_DOCUMENT_CANARY_d91e"
        error = urllib.error.HTTPError(
            "https://relay.example/v1/test",
            400,
            "bad request",
            email.message.Message(),
            io.BytesIO(json.dumps({"error": canary}).encode()),
        )
        client = JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            transport=lambda _request: (_ for _ in ()).throw(error),
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )

        with pytest.raises(SystemExit) as raised:
            client.post({"documents": [canary]})

        message = str(raised.value)
        assert "HTTP 400" in message
        assert canary not in message

    @pytest.mark.parametrize(
        ("body", "hint"),
        [
            (b"error code: 1010", "Cloudflare rejected the User-Agent"),
            ("不在可调用时段".encode(), "time-of-day window"),
        ],
    )
    def test_403_keeps_fixed_relay_hint_without_exposing_body(self, body: bytes, hint: str) -> None:
        error = urllib.error.HTTPError(
            "https://relay.example/v1/test",
            403,
            "forbidden",
            email.message.Message(),
            io.BytesIO(body),
        )
        client = JsonClient(
            url="https://relay.example/v1/test",
            key="secret",
            transport=lambda _request: (_ for _ in ()).throw(error),
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )

        with pytest.raises(SystemExit, match=hint) as raised:
            client.post({})

        assert body.decode() not in str(raised.value)
