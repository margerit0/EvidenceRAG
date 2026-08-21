"""Shared retrying JSON transport for the One Hub relay.

Embedding and reranking use different request/response contracts but the same
network behavior. Keeping HTTP here prevents their User-Agent, 403 diagnostics,
and long upstream-saturation retry ladder from drifting apart.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MAX_RETRY_AFTER",
    "RETRY_STATUS",
    "JsonClient",
    "Transport",
    "backoff_seconds",
    "explain_http_error",
]

USER_AGENT = "zhrag/0.1 (+https://github.com/margerit0/zhrag)"
ERROR_DETAIL_CHARS = 600
RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 520})
MAX_RETRY_AFTER = 300.0


def _flush_print(message: str) -> None:
    print(message, flush=True)


def explain_http_error(detail: str) -> str:
    """Translate the two relay 403 bodies that otherwise resemble bad keys."""
    if "1010" in detail:
        return "\n  -> Cloudflare rejected the User-Agent, not your key."
    if "可调用时段" in detail:
        return (
            "\n  -> Not an auth or code failure: this relay's key group is gated to a"
            "\n     time-of-day window. Re-run inside the window shown in the message."
        )
    return ""


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before retry ``attempt``; long for upstream saturation."""
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")
    if retry_after is not None:
        return retry_after
    return min(60.0, 5.0 * float(2**attempt))


def _parse_retry_after(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER)


Transport = Callable[[urllib.request.Request], bytes]


def _urlopen(request: urllib.request.Request) -> bytes:
    with urllib.request.urlopen(request, timeout=300) as response:
        body: bytes = response.read()
        return body


@dataclass(frozen=True, slots=True)
class JsonClient:
    """POST JSON with the relay-specific headers, retries, and diagnostics."""

    url: str
    key: str
    retries: int = 7
    transport: Transport = _urlopen
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = _flush_print

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
                method="POST",
            )
            try:
                raw = json.loads(self.transport(request).decode("utf-8"))
                if not isinstance(raw, dict):
                    raise SystemExit("! malformed provider response: expected a JSON object")
                return raw
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"! malformed JSON from {self.url}: {exc}") from exc
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionResetError,
            ) as exc:
                detail, retry_after = "", None
                if isinstance(exc, urllib.error.HTTPError):
                    detail = exc.read().decode("utf-8", errors="replace")[:ERROR_DETAIL_CHARS]
                    hint = explain_http_error(detail)
                    if exc.code not in RETRY_STATUS:
                        raise SystemExit(f"! HTTP {exc.code} from {self.url}{hint}") from exc
                    retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                if attempt == self.retries - 1:
                    raise SystemExit(
                        f"! giving up after {self.retries} attempts: {exc}"
                        f"{explain_http_error(detail)}"
                    ) from exc
                wait = backoff_seconds(attempt, retry_after)
                self.log(f"    retry {attempt + 1}/{self.retries} in {wait:.0f}s ({exc})")
                self.sleep(wait)
        raise SystemExit("unreachable")
