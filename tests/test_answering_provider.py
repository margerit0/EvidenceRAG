from __future__ import annotations

import errno
import http.client
import io
import json
import socket
import ssl
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from email.message import Message
from email.utils import format_datetime

import pytest

from zhrag.answering import GenerationError
from zhrag.providers.answering import (
    GENERATION_RETRY_STATUS,
    MAX_RESPONSE_BYTES,
    ChatAnswerGenerator,
    transport_error_details,
)
from zhrag.providers.chat import ChatConfig
from zhrag.providers.http import RETRY_STATUS


class Recorder:
    def __init__(self, body: bytes | Exception) -> None:
        self.body = body
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> bytes:
        self.requests.append(request)
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def body(*, model: object = "model", content: object = '{"answerable":true}') -> bytes:
    return json.dumps(
        {
            "model": model,
            "choices": [
                {"finish_reason": "stop", "message": {"content": content}},
            ],
        }
    ).encode("utf-8")


def generator(recorder: Recorder, **kwargs: object) -> ChatAnswerGenerator:
    return ChatAnswerGenerator(
        ChatConfig("secret-canary", "https://relay.example", "model"),
        transport=recorder,
        sleep=lambda _: None,
        **kwargs,
    )


class TestChatAnswerGenerator:
    def test_generates_once_with_bounded_output_and_no_reasoning_effort(self) -> None:
        recorder = Recorder(body())
        answerer = generator(recorder)

        assert answerer.generate("system", "user") == '{"answerable":true}'
        assert len(recorder.requests) == 1
        payload = json.loads(recorder.requests[0].data or b"{}")
        assert payload["max_completion_tokens"] == 2048
        assert "reasoning_effort" not in payload
        assert answerer.profile_fingerprint == answerer.profile_fingerprint
        assert len(answerer.profile_fingerprint) == 64
        assert "secret-canary" not in answerer.profile_fingerprint

    def test_rejects_served_model_drift_without_leaking_provider_data(self) -> None:
        canary = "PRIVATE_PROVIDER_PAYLOAD"
        recorder = Recorder(body(model="other-model", content=canary))
        answerer = generator(recorder)

        with pytest.raises(GenerationError) as raised:
            answerer.generate("system", "user")

        assert raised.value.code == "generation_failed"
        assert canary not in str(raised.value)
        assert "relay.example" not in str(raised.value)

    def test_maps_timeout_to_fixed_error(self) -> None:
        recorder = Recorder(TimeoutError("secret endpoint details"))
        answerer = generator(recorder, max_retries=0)

        with pytest.raises(GenerationError) as raised:
            answerer.generate("system", "user")

        assert raised.value.code == "generation_timeout"
        assert str(raised.value) == "generation_timeout"
        assert len(recorder.requests) == 1

    def test_maps_wrapped_timeout_to_fixed_error(self) -> None:
        recorder = Recorder(urllib.error.URLError(TimeoutError("private")))
        answerer = generator(recorder)

        with pytest.raises(GenerationError) as raised:
            answerer.generate("system", "user")

        assert raised.value.code == "generation_timeout"

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (0, "max_output_tokens"),
            (True, "max_output_tokens"),
            (0.5, "max_output_tokens"),
            (0, "max_output_tokens"),
        ],
    )
    def test_rejects_invalid_configuration(self, value: object, message: str) -> None:
        recorder = Recorder(body())
        with pytest.raises(ValueError, match=message):
            generator(recorder, max_output_tokens=value)

    def test_rejects_invalid_timeout_configuration(self) -> None:
        recorder = Recorder(body())
        with pytest.raises(ValueError, match="timeout_seconds"):
            generator(recorder, timeout_seconds=float("inf"))

    def test_profile_binds_generation_parameters_but_not_the_key(self) -> None:
        first = generator(Recorder(body()), timeout_seconds=1, max_output_tokens=1)
        second = generator(Recorder(body()), timeout_seconds=60, max_output_tokens=4096)
        assert first.profile_fingerprint != second.profile_fingerprint
        third = ChatAnswerGenerator(
            ChatConfig("other-key", "https://relay.example", "model"),
            timeout_seconds=1,
            max_output_tokens=1,
        )
        assert first.profile_fingerprint == third.profile_fingerprint


def http_error(status: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://relay.example",
        status,
        "private-error-canary",
        headers,
        io.BytesIO(b"private-response-canary"),
    )


class TestLinearRetries:
    @pytest.mark.parametrize(
        "error",
        [
            http.client.RemoteDisconnected("private"),
            http.client.IncompleteRead(b"private", 100),
            ConnectionAbortedError("private"),
            ConnectionRefusedError("private"),
            BrokenPipeError("private"),
            OSError(errno.ENETUNREACH, "private"),
            OSError(10054, "private"),
            socket.gaierror(socket.EAI_AGAIN, "private"),
            socket.gaierror(11002, "private"),
            ssl.SSLEOFError(8, "private"),
        ],
    )
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_transient_transport_errors_recover_with_same_request(
        self, error: Exception, wrapped: bool
    ) -> None:
        requests: list[urllib.request.Request] = []
        sleeps: list[float] = []

        def transport(request: urllib.request.Request) -> bytes:
            requests.append(request)
            if len(requests) < 3:
                raise urllib.error.URLError(error) if wrapped else error
            return body()

        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=transport,
            sleep=sleeps.append,
            max_retries=5,
        )
        assert answerer.generate("system", "user") == '{"answerable":true}'
        assert len(requests) == 3 and sleeps == [5, 10]
        assert all(request is requests[0] for request in requests)

    def test_mixed_transient_failures_exhaust_exact_budget(self) -> None:
        failures = [
            http_error(504),
            urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "private")),
            urllib.error.URLError(http.client.RemoteDisconnected("private")),
        ]
        sleeps: list[float] = []
        calls = 0

        def transport(_request: urllib.request.Request) -> bytes:
            nonlocal calls
            failure = failures[calls]
            calls += 1
            raise failure

        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=transport,
            sleep=sleeps.append,
            max_retries=2,
        )
        with pytest.raises(GenerationError, match="generation_failed"):
            answerer.generate("system", "user")
        assert calls == 3 and sleeps == [5, 10]

    @pytest.mark.parametrize(
        "error",
        [
            ssl.SSLCertVerificationError(1, "private"),
            ssl.SSLError(1, "private"),
            socket.gaierror(socket.EAI_NONAME, "private"),
            PermissionError(errno.EACCES, "private"),
            OSError(errno.EINVAL, "private"),
            http.client.InvalidURL("private"),
        ],
    )
    def test_permanent_wrapped_failures_stop_without_retry(self, error: Exception) -> None:
        recorder = Recorder(urllib.error.URLError(error))
        answerer = generator(recorder, max_retries=5)
        with pytest.raises(GenerationError, match="generation_failed") as raised:
            answerer.generate("system", "user")
        assert len(recorder.requests) == 1
        assert "private" not in str(raised.value)

    def test_nested_reason_diagnostics_and_cycle_are_safe(self) -> None:
        error = urllib.error.URLError(
            urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "private"))
        )
        assert transport_error_details(error) == {
            "error_type": "URLError",
            "reason_type": "gaierror",
            "errno": socket.EAI_AGAIN,
            "failure_kind": "dns_temporary",
            "chat_retryable": True,
        }
        error.reason = error
        recorder = Recorder(error)
        with pytest.raises(GenerationError, match="generation_failed"):
            generator(recorder).generate("system", "user")
        assert len(recorder.requests) == 1

    def test_transient_policy_version_changes_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answerer = generator(Recorder(body()))
        before = answerer.profile_fingerprint
        monkeypatch.setattr("zhrag.providers.answering.TRANSPORT_RETRY_CONTRACT", "future-policy")
        assert answerer.profile_fingerprint != before

    @pytest.mark.parametrize("failure", ["timeout", 401, 504])
    def test_fifteen_retries_have_exact_linear_delays_and_no_final_sleep(
        self, failure: str | int
    ) -> None:
        recorder = Recorder(
            TimeoutError("private") if failure == "timeout" else http_error(int(failure))
        )
        sleeps: list[float] = []
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=recorder,
            sleep=sleeps.append,
        )
        code = "generation_timeout" if failure == "timeout" else "generation_failed"
        with pytest.raises(GenerationError, match=code):
            answerer.generate("system", "user")
        assert len(recorder.requests) == 16
        assert sleeps == [5.0 * n for n in range(1, 16)]
        assert sum(sleeps) == 600.0
        assert len({request.data for request in recorder.requests}) == 1

    @pytest.mark.parametrize(
        "failure", [401, 429, 500, 502, 503, 504, 520, "timeout", "reset", "wrapped-reset"]
    )
    def test_retryable_failure_then_success_stops_immediately(self, failure: int | str) -> None:
        requests: list[urllib.request.Request] = []
        sleeps: list[float] = []

        def transport(request: urllib.request.Request) -> bytes:
            requests.append(request)
            if len(requests) < 3:
                if failure == "timeout":
                    raise urllib.error.URLError(TimeoutError("private"))
                if failure == "reset":
                    raise ConnectionResetError("private")
                if failure == "wrapped-reset":
                    raise urllib.error.URLError(ConnectionResetError("private"))
                raise http_error(int(failure))
            return body()

        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=transport,
            sleep=sleeps.append,
        )
        assert answerer.generate("system", "user") == '{"answerable":true}'
        assert len(requests) == 3
        assert sleeps == [5, 10]
        assert all(request is requests[0] for request in requests)

    @pytest.mark.parametrize("status", [400, 403, 404, 408, 422])
    def test_permanent_http_failure_never_retries(self, status: int) -> None:
        error = http_error(status)
        recorder = Recorder(error)
        sleeps: list[float] = []
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=recorder,
            sleep=sleeps.append,
        )
        with pytest.raises(GenerationError, match="generation_failed") as raised:
            answerer.generate("system", "user")
        assert len(recorder.requests) == 1 and sleeps == []
        assert "private" not in str(raised.value)
        assert error.closed

    @pytest.mark.parametrize("status", [401, 429])
    @pytest.mark.parametrize(
        "retry_after,delay",
        [
            ("2", 5),
            ("20", 20),
            ("999", 300),
            ("NaN", 5),
            ("-1", 5),
            ("broken", 5),
        ],
    )
    def test_retry_after_is_respected_with_existing_cap(
        self, status: int, retry_after: str, delay: int
    ) -> None:
        sleeps: list[float] = []
        calls = 0

        def transport(_request: urllib.request.Request) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise http_error(status, retry_after)
            return body()

        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=transport,
            sleep=sleeps.append,
        )
        answerer.generate("system", "user")
        assert sleeps == [delay]

    def test_retry_after_http_date(self) -> None:
        from zhrag.providers.answering import _retry_after_seconds  # noqa: PLC0415

        future = format_datetime(datetime.now(UTC) + timedelta(seconds=120), usegmt=True)
        assert 115 <= _retry_after_seconds(future) <= 120

    @pytest.mark.parametrize("reply", [b"not-json", body(model="wrong"), body(content="")])
    def test_bad_responses_do_not_trigger_paid_retries(self, reply: bytes) -> None:
        recorder = Recorder(reply)
        sleeps: list[float] = []
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=recorder,
            sleep=sleeps.append,
        )
        with pytest.raises(GenerationError):
            answerer.generate("system", "user")
        assert len(recorder.requests) == 1 and not sleeps

    @pytest.mark.parametrize("max_retries", [0, 3])
    def test_configured_retry_budget_limits_calls_and_sleeps(self, max_retries: int) -> None:
        recorder = Recorder(TimeoutError("private"))
        sleeps: list[float] = []
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=recorder,
            sleep=sleeps.append,
            max_retries=max_retries,
        )
        with pytest.raises(GenerationError, match="generation_timeout"):
            answerer.generate("system", "user")
        assert len(recorder.requests) == max_retries + 1
        assert sleeps == [5.0 * n for n in range(1, max_retries + 1)]

    def test_other_url_errors_do_not_retry(self) -> None:
        recorder = Recorder(urllib.error.URLError("private configuration error"))
        sleeps: list[float] = []
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            transport=recorder,
            sleep=sleeps.append,
        )
        with pytest.raises(GenerationError, match="generation_failed"):
            answerer.generate("system", "user")
        assert len(recorder.requests) == 1
        assert sleeps == []

    @pytest.mark.parametrize("value", [-1, 16, True, 1.5])
    def test_invalid_retry_budget_fails_locally(self, value: object) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            generator(Recorder(body()), max_retries=value)

    def test_401_retry_is_online_only_and_can_be_disabled(self) -> None:
        assert RETRY_STATUS | {401} == GENERATION_RETRY_STATUS
        assert 401 not in RETRY_STATUS
        error = http_error(401)
        recorder = Recorder(error)
        answerer = generator(recorder, max_retries=0)
        with pytest.raises(GenerationError, match="generation_failed"):
            answerer.generate("system", "user")
        assert len(recorder.requests) == 1
        assert error.closed

    def test_retry_status_changes_fingerprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answerer = generator(Recorder(body()))
        current = answerer.profile_fingerprint
        monkeypatch.setattr("zhrag.providers.answering.GENERATION_RETRY_STATUS", RETRY_STATUS)
        assert answerer.profile_fingerprint != current

    def test_retry_policy_changes_fingerprint(self) -> None:
        first = generator(Recorder(body()), max_retries=0)
        second = generator(Recorder(body()), max_retries=15)
        assert first.profile_fingerprint != second.profile_fingerprint

    def test_transport_contract_changes_fingerprint_and_must_be_named(self) -> None:
        default = generator(Recorder(body()))
        assert default.transport_contract == "urllib-default-v1"
        direct = generator(Recorder(body()), transport_contract="http-client-direct-no-redirect-v2")
        assert direct.profile_fingerprint != default.profile_fingerprint
        assert direct.generate("system", "user") == '{"answerable":true}'
        for value in ("", "  ", None):
            with pytest.raises(ValueError, match="transport_contract"):
                generator(Recorder(body()), transport_contract=value)


class TestBoundedDefaultTransport:
    def test_default_transport_passes_timeout_and_rejects_oversized_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self, size: int = -1) -> bytes:
                assert size == MAX_RESPONSE_BYTES + 1
                return b"x" * size

        calls: list[float] = []

        def urlopen(_request: urllib.request.Request, *, timeout: float) -> Response:
            calls.append(timeout)
            return Response()

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        answerer = ChatAnswerGenerator(
            ChatConfig("secret", "https://relay.example", "model"),
            timeout_seconds=12,
        )

        with pytest.raises(GenerationError) as raised:
            answerer.generate("system", "user")

        assert raised.value.code == "generation_failed"
        assert calls == [12.0]
