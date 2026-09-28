"""Strict bounded SSE assembly; only complete final content leaves the transport."""

from __future__ import annotations

import http.client
import json
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from zhrag.generation_control import remaining_seconds
from zhrag.providers.direct import DirectTransport

STREAM_CONTRACT = "chat-sse-direct-v1"
STREAM_MAX_SECONDS = 600.0
MAX_LINE_BYTES = 65_536


class StreamProtocolError(ValueError):
    """Fixed diagnostics, never response fragments or arbitrary provider messages."""

    def __init__(self, reason: str) -> None:
        allowed = {
            "json",
            "object",
            "error_event",
            "model",
            "choices",
            "delta",
            "nontext",
            "unsupported_delta",
            "finish",
            "after_finish",
            "incomplete",
            "empty",
            "content_type",
            "size",
            "encoding",
            "event_type",
        }
        if reason not in allowed:
            raise ValueError("invalid stream diagnostic")
        self.reason = reason
        super().__init__(f"stream_{reason}")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StreamProtocolError("json")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise StreamProtocolError("json")


@dataclass
class StreamCollector:
    model: str
    parts: list[str] = field(default_factory=list)
    model_seen: bool = False
    stopped: bool = False
    done: bool = False
    usage: dict[str, object] = field(default_factory=dict)

    def feed(self, data: str) -> dict[str, int | str]:
        if self.done:
            raise StreamProtocolError("after_finish")
        if data == "[DONE]":
            if not self.stopped:
                raise StreamProtocolError("incomplete")
            self.done = True
            return {"event": "done", "content_chars": 0, "reasoning_chars": 0}
        try:
            obj = json.loads(data, object_pairs_hook=_unique, parse_constant=_constant)
        except (ValueError, RecursionError):
            raise StreamProtocolError("json") from None
        if not isinstance(obj, dict):
            raise StreamProtocolError("object")
        if obj.get("error") is not None:
            raise StreamProtocolError("error_event")
        if obj.get("model") is not None:
            if obj["model"] != self.model:
                raise StreamProtocolError("model")
            self.model_seen = True
        self._usage(obj.get("usage"))
        choices = obj.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamProtocolError("choices")
        event: dict[str, int | str] = {"event": "chunk", "content_chars": 0, "reasoning_chars": 0}
        if choices:
            self._choice(choices[0], event)
        return event

    def _usage(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            number = value.get(key)
            if type(number) is int and number >= 0:
                self.usage[key] = number
        details = value.get("completion_tokens_details")
        if isinstance(details, dict):
            number = details.get("reasoning_tokens")
            if type(number) is int and number >= 0:
                self.usage["completion_tokens_details"] = {"reasoning_tokens": number}

    def _choice(self, choice: object, event: dict[str, int | str]) -> None:
        if (
            not isinstance(choice, dict)
            or type(choice.get("index")) is not int
            or choice["index"] != 0
        ):
            raise StreamProtocolError("choices")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise StreamProtocolError("delta")
        if any(delta.get(key) for key in ("refusal", "tool_calls", "function_call")):
            raise StreamProtocolError("unsupported_delta")
        content = delta.get("content")
        reasoning = delta.get("reasoning_content", delta.get("reasoning"))
        if any(value is not None and not isinstance(value, str) for value in (content, reasoning)):
            raise StreamProtocolError("nontext")
        content, reasoning = content or "", reasoning or ""
        if self.stopped and (content or reasoning or choice.get("finish_reason") is not None):
            raise StreamProtocolError("after_finish")
        self.parts.append(content)
        event.update(content_chars=len(content), reasoning_chars=len(reasoning))
        finish = choice.get("finish_reason")
        if finish is not None:
            if finish != "stop":
                raise StreamProtocolError("finish")
            self.stopped = True
            event["finish_reason"] = "stop"

    def response(self) -> bytes:
        if not self.done or not self.stopped or not self.model_seen:
            raise StreamProtocolError("incomplete")
        content = "".join(self.parts)
        if not content.strip():
            raise StreamProtocolError("empty")
        return json.dumps(
            {
                "model": self.model,
                "usage": self.usage,
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
            ensure_ascii=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class StreamingTransport:
    timeout_seconds: float = 60.0
    max_response_bytes: int = 256 * 1024
    clock: Callable[[], float] = time.monotonic
    observe: Callable[[dict[str, int | float | str]], None] = lambda _event: None

    def __post_init__(self) -> None:
        DirectTransport(self.timeout_seconds, self.max_response_bytes)
        if self.max_response_bytes > 256 * 1024:
            raise ValueError("stream response size exceeds limit")

    def __call__(self, request: urllib.request.Request) -> bytes:
        parts = urlsplit(request.full_url)
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
            or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError("invalid stream endpoint")
        if not isinstance(request.data, bytes):
            raise ValueError("stream request bytes required")
        payload = json.loads(request.data)
        if payload.get("stream") is not True or not isinstance(payload.get("model"), str):
            raise ValueError("stream request required")
        started = self.clock()

        def timeout() -> float:
            remaining = min(STREAM_MAX_SECONDS - (self.clock() - started), remaining_seconds())
            if remaining <= 0:
                raise TimeoutError("stream deadline")
            return min(self.timeout_seconds, remaining)

        connection: http.client.HTTPConnection
        if parts.scheme == "https":
            connection = http.client.HTTPSConnection(
                parts.hostname,
                parts.port or 443,
                timeout=timeout(),
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(
                parts.hostname, parts.port or 80, timeout=timeout()
            )
        response = None
        try:
            connection.request(
                request.get_method(),
                (parts.path or "/") + (f"?{parts.query}" if parts.query else ""),
                body=request.data,
                headers={**dict(request.header_items()), "Accept": "text/event-stream"},
            )
            response = connection.getresponse()
            self.observe(
                {"event": "headers", "status": response.status, "seconds": self.clock() - started}
            )
            if not 200 <= response.status < 300:
                raise urllib.error.HTTPError(
                    request.full_url, response.status, "stream HTTP error", response.headers, None
                )
            if "text/event-stream" not in (response.getheader("Content-Type") or "").lower():
                raise StreamProtocolError("content_type")
            collector = StreamCollector(payload["model"])
            self._read(response, connection, collector, timeout, started)
            return collector.response()
        finally:
            if response is not None:
                response.close()
            connection.close()

    def _read(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPConnection,
        collector: StreamCollector,
        timeout: Callable[[], float],
        started: float,
    ) -> None:
        size, lines, event_type = 0, [], "message"
        while not collector.done:
            seconds = timeout()
            if connection.sock is not None:
                connection.sock.settimeout(seconds)
            line = response.readline(min(self.max_response_bytes - size + 1, MAX_LINE_BYTES + 1))
            timeout()
            size += len(line)
            if size > self.max_response_bytes or len(line) > MAX_LINE_BYTES:
                raise StreamProtocolError("size")
            if not line:
                raise http.client.IncompleteRead(b"")
            try:
                text = line.decode("utf-8").rstrip("\r\n")
            except UnicodeError:
                raise StreamProtocolError("encoding") from None
            if text.startswith("data:"):
                lines.append(text[5:].removeprefix(" "))
            elif text.startswith("event:"):
                event_type = text[6:].strip()
            elif not text:
                if lines:
                    if event_type not in {"message", ""}:
                        raise StreamProtocolError(
                            "error_event" if event_type == "error" else "event_type"
                        )
                    event = collector.feed("\n".join(lines))
                    self.observe({**event, "seconds": self.clock() - started, "bytes": size})
                lines, event_type = [], "message"
