"""Bounded direct connections; never consult environment or system proxies."""

from __future__ import annotations

import http.client
import io
import math
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlsplit

DIRECT_CONTRACT = "http-client-direct-no-redirect-v2"
# Error pages (for example a Cloudflare 504 HTML page of ~850 KB) must surface as
# their HTTP status so retry policy can act; only a short diagnostic slice is kept.
ERROR_BODY_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class DirectTransport:
    timeout_seconds: float = 60.0
    max_response_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (float, int))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 300
        ):
            raise ValueError("timeout must be in (0, 300]")
        if (
            type(self.max_response_bytes) is not int
            or not 1 <= self.max_response_bytes <= 8_388_608
        ):
            raise ValueError("response size must be in [1, 8388608]")

    def __call__(self, request: urllib.request.Request) -> bytes:
        parts = urlsplit(request.full_url)
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError("invalid provider endpoint")
        # HTTP is supported only for explicitly configured local development endpoints.
        if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("remote provider endpoints must use HTTPS")
        connection: http.client.HTTPConnection
        if parts.scheme == "https":
            connection = http.client.HTTPSConnection(
                parts.hostname,
                parts.port or 443,
                timeout=self.timeout_seconds,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(
                parts.hostname,
                parts.port or 80,
                timeout=self.timeout_seconds,
            )
        try:
            # Origin-form path and direct destination: no CONNECT, proxy lookup or redirect.
            connection.request(
                request.get_method(),
                (parts.path or "/") + (f"?{parts.query}" if parts.query else ""),
                body=request.data,
                headers=dict(request.header_items()),
            )
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                # Status before size: an oversized error page is still that status.
                raise urllib.error.HTTPError(
                    request.full_url,
                    response.status,
                    response.reason,
                    response.headers,
                    io.BytesIO(response.read(ERROR_BODY_BYTES)),
                )
            body = response.read(self.max_response_bytes + 1)
            if len(body) > self.max_response_bytes:
                raise ValueError("provider response exceeded size limit")
            return body
        except (http.client.HTTPException, OSError) as exc:
            if isinstance(exc, (urllib.error.HTTPError, TimeoutError, ConnectionResetError)):
                raise
            raise urllib.error.URLError(exc) from exc
        finally:
            connection.close()
