"""Online chat adapter used by the small grounded-answering service.

The batch generation client retains its own retry policy. Online generation uses
an independent linear retry schedule, a finite response body, and errors that do
not disclose credentials, URLs, or provider payloads.
"""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from zhrag.answering import GenerationError
from zhrag.providers.chat import REASONING_EFFORTS, ChatClient, ChatConfig
from zhrag.providers.http import MAX_RETRY_AFTER, RETRY_STATUS, Transport

__all__ = ["MAX_RESPONSE_BYTES", "ChatAnswerGenerator"]

MAX_RESPONSE_BYTES = 256 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 15
RETRY_STEP_SECONDS = 5.0
# Online-only policy: retry 401 without changing credentials or batch clients.
GENERATION_RETRY_STATUS = RETRY_STATUS | {401}
TRANSPORT_RETRY_CONTRACT = "transient-network-v2"
_TRANSIENT_ERRNOS = frozenset(
    (
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.EPIPE,
        errno.ETIMEDOUT,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
    )
)
# Winsock errors are not portable errno values; inspect winerror explicitly too.
_TRANSIENT_WINERRORS = frozenset((10050, 10051, 10052, 10053, 10054, 10060, 10061, 10064, 10065))


def _transport_reason(error: BaseException) -> BaseException:
    seen: set[int] = set()
    while isinstance(error, urllib.error.URLError) and not isinstance(
        error, urllib.error.HTTPError
    ):
        seen.add(id(error))
        if not isinstance(error.reason, BaseException) or id(error.reason) in seen:
            break
        error = error.reason
    return error


def _network_failure_kind(error: BaseException) -> str:  # noqa: PLR0911 - explicit failure classes
    reason = _transport_reason(error)
    if isinstance(reason, urllib.error.HTTPError):
        return "http"
    if isinstance(reason, (ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return "connection_interrupted"
    if isinstance(reason, ssl.SSLError):
        return "tls_error"
    if isinstance(reason, socket.gaierror):
        return "dns_temporary" if reason.errno in (socket.EAI_AGAIN, 11002) else "dns_error"
    if isinstance(reason, TimeoutError):
        return "timeout"
    if isinstance(
        reason, (ConnectionError, http.client.RemoteDisconnected, http.client.IncompleteRead)
    ):
        return "connection_interrupted"
    if isinstance(reason, OSError) and (
        reason.errno in _TRANSIENT_ERRNOS
        or reason.errno in _TRANSIENT_WINERRORS
        or getattr(reason, "winerror", None) in _TRANSIENT_WINERRORS
    ):
        return "connection_interrupted"
    # Preserve the existing urllib string-timeout compatibility, without matching
    # arbitrary error text for DNS, TLS, configuration or permission failures.
    if (
        isinstance(reason, urllib.error.URLError)
        and isinstance(reason.reason, str)
        and "timed out" in reason.reason.lower()
    ):
        return "timeout"
    return "unclassified"


def transport_error_details(error: BaseException) -> dict[str, str | int | bool]:
    """Diagnostic codes only: never stringify errors, endpoints or response bodies."""
    reason = _transport_reason(error)
    details: dict[str, str | int | bool] = {
        "error_type": type(error).__name__,
        "reason_type": type(reason).__name__,
        "failure_kind": _network_failure_kind(error),
        "chat_retryable": _retry_delay(error, 1) is not None,
    }
    for name in ("errno", "winerror"):
        value = getattr(reason, name, None)
        if type(value) is int:
            details[name] = value
    if isinstance(error, urllib.error.HTTPError):
        details["http_status"] = error.code
    return details


def _bounded_urlopen(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    max_response_bytes: int,
) -> bytes:
    """Read a bounded provider response with an explicit network timeout."""
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body: bytes = response.read(max_response_bytes + 1)
    if len(body) > max_response_bytes:
        raise ValueError("provider response exceeded the configured size limit")
    return body


def _is_timeout(error: BaseException) -> bool:
    """Inspect wrapped transport errors without exposing them to the caller."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (TimeoutError, socket.timeout)):
            return True
        if isinstance(current, urllib.error.URLError):
            reason = current.reason
            if isinstance(reason, BaseException):
                pending.append(reason)
            if isinstance(reason, str) and "timed out" in reason.lower():
                return True
        cause = current.__cause__ or current.__context__
        if cause is not None:
            pending.append(cause)
    return False


def _retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0
    return min(seconds, MAX_RETRY_AFTER)


def _retry_delay(error: BaseException, retry_number: int) -> float | None:
    # HTTPError subclasses URLError; permanent HTTP failures must stop here.
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in GENERATION_RETRY_STATUS:
            return None
        retry_after = _retry_after_seconds(error.headers.get("Retry-After"))
        return max(RETRY_STEP_SECONDS * retry_number, retry_after)
    if _network_failure_kind(error) in {"timeout", "connection_interrupted", "dns_temporary"}:
        return RETRY_STEP_SECONDS * retry_number
    return None


def _profile_fingerprint(
    config: ChatConfig,
    *,
    max_output_tokens: int,
    timeout_seconds: float,
    reasoning_effort: str | None,
    max_response_bytes: int,
    max_retries: int,
    transport_contract: str = "urllib-default-v1",
) -> str:
    canonical = json.dumps(
        {
            "contract": "zhrag-chat-answer-v2",
            "endpoint": config.endpoint,
            "model": config.model,
            "max_output_tokens": max_output_tokens,
            "timeout_seconds": timeout_seconds,
            "reasoning_effort": reasoning_effort,
            "max_response_bytes": max_response_bytes,
            "max_retries": max_retries,
            "retry_step_seconds": RETRY_STEP_SECONDS,
            "retry_statuses": sorted(GENERATION_RETRY_STATUS),
            "retry_after_cap_seconds": MAX_RETRY_AFTER,
            "transport_retry_contract": TRANSPORT_RETRY_CONTRACT,
            "transport_contract": transport_contract,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatAnswerGenerator:
    """Generate one answer with bounded linear retries for configured failures."""

    config: ChatConfig
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    reasoning_effort: str | None = None
    transport: Transport | None = None
    max_response_bytes: int = MAX_RESPONSE_BYTES
    max_retries: int = DEFAULT_MAX_RETRIES
    sleep: Callable[[float], None] = time.sleep
    transport_contract: str = "urllib-default-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.transport_contract, str) or not self.transport_contract.strip():
            raise ValueError("transport_contract must be nonempty")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= DEFAULT_MAX_RETRIES:
            raise ValueError("max_retries must be an integer between 0 and 15")
        if (
            not isinstance(self.max_output_tokens, int)
            or isinstance(self.max_output_tokens, bool)
            or not 1 <= self.max_output_tokens <= 8_192
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            not isinstance(self.max_response_bytes, int)
            or isinstance(self.max_response_bytes, bool)
            or self.max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")

        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("invalid reasoning_effort")
        if self.timeout_seconds > 300:
            raise ValueError("timeout_seconds must not exceed 300")
        if self.max_response_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("max_response_bytes exceeds limit")

    @property
    def profile_fingerprint(self) -> str:
        """Bind public generation settings, never the API key."""
        return _profile_fingerprint(
            self.config,
            max_output_tokens=self.max_output_tokens,
            timeout_seconds=float(self.timeout_seconds),
            reasoning_effort=self.reasoning_effort,
            max_response_bytes=self.max_response_bytes,
            max_retries=self.max_retries,
            transport_contract=self.transport_contract,
        )

    def _transport(self) -> Transport:
        if self.transport is not None:
            return self.transport

        def transport(request: urllib.request.Request) -> bytes:
            return _bounded_urlopen(
                request,
                timeout_seconds=float(self.timeout_seconds),
                max_response_bytes=self.max_response_bytes,
            )

        return transport

    def _retrying_transport(self) -> Transport:
        transport = self._transport()

        def send(request: urllib.request.Request) -> bytes:
            for attempt in range(self.max_retries + 1):
                try:
                    return transport(request)
                except (OSError, http.client.HTTPException) as exc:
                    delay = _retry_delay(exc, attempt + 1)
                    if isinstance(exc, urllib.error.HTTPError):
                        # Consume no error payload: it can be large or contain private text.
                        exc.close()
                    if delay is None or attempt == self.max_retries:
                        code = "generation_timeout" if _is_timeout(exc) else "generation_failed"
                        raise GenerationError(code) from None
                    self.sleep(delay)
            raise AssertionError("unreachable retry loop")

        return send

    def generate(self, system: str, user: str) -> str:
        """Return provider text or one fixed, privacy-safe generation error."""
        client = ChatClient(
            self.config,
            reasoning_effort=self.reasoning_effort,
            # Transport owns the linear schedule; do not multiply it by batch retries.
            retries=1,
            transport=self._retrying_transport(),
            log=lambda _message: None,
        )
        try:
            reply = client.complete(
                system,
                user,
                json_object=True,
                max_output_tokens=self.max_output_tokens,
            )
        except GenerationError:
            raise
        except (Exception, SystemExit) as exc:
            code = "generation_timeout" if _is_timeout(exc) else "generation_failed"
            raise GenerationError(code) from None

        if reply.model != self.config.model:
            raise GenerationError("generation_failed")
        return reply.content
