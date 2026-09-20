from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from zhrag.providers import direct


class Connection:
    def __init__(self, *, status: int = 200, body: bytes = b'{"ok":true}') -> None:
        self.status, self.body = status, body
        self.reason = "synthetic"
        self.headers = Message()
        self.headers["Retry-After"] = "5"
        self.calls: list[tuple[object, ...]] = []
        self.closed = False

    def request(self, *args: object, **kwargs: object) -> None:
        self.calls.append((*args, kwargs))

    def getresponse(self) -> Connection:
        return self

    def read(self, limit: int) -> bytes:
        return self.body[:limit]

    def close(self) -> None:
        self.closed = True


def install(monkeypatch: pytest.MonkeyPatch, connection: Connection) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []

    def factory(host: str, port: int, **kwargs: Any) -> Connection:
        created.append({"host": host, "port": port, **kwargs})
        return connection

    monkeypatch.setattr(direct.http.client, "HTTPSConnection", factory)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("proxy discovery must not run")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "getproxies", forbidden)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:8888")
    return created


def test_direct_origin_and_verified_tls_ignore_proxy_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = Connection()
    created = install(monkeypatch, connection)
    request = urllib.request.Request(
        "https://api.example.invalid/v1/chat?q=x",
        data=b"{}",
        headers={"Authorization": "Bearer synthetic"},
    )
    assert direct.DirectTransport()(request) == b'{"ok":true}'
    assert created[0]["host"] == "api.example.invalid" and created[0]["port"] == 443
    assert created[0]["context"].verify_mode == ssl.CERT_REQUIRED
    assert created[0]["context"].check_hostname
    assert connection.calls[0][0:2] == ("POST", "/v1/chat?q=x")
    assert connection.closed


@pytest.mark.parametrize("status", [301, 403, 429, 500])
def test_errors_preserve_status_for_retry_without_redirect(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    connection = Connection(status=status, body=b"synthetic error")
    created = install(monkeypatch, connection)
    with pytest.raises(urllib.error.HTTPError) as error:
        direct.DirectTransport()(urllib.request.Request("https://example.invalid/v1"))
    assert error.value.code == status and error.value.read() == b"synthetic error"
    assert error.value.headers["Retry-After"] == "5"
    error.value.close()
    assert connection.closed and len(created) == 1


def test_response_cap_and_failure_always_close_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Connection(body=b"x" * 100)
    install(monkeypatch, connection)
    with pytest.raises(ValueError, match="size"):
        direct.DirectTransport(max_response_bytes=5)(
            urllib.request.Request("https://example.invalid")
        )
    assert connection.closed


def test_oversized_error_page_keeps_its_status_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Observed live: Cloudflare 504 with an ~850 KB HTML body. The status must win
    # over the success-body cap, otherwise 504 degrades into a non-retryable error.
    connection = Connection(status=504, body=b"<html>" + b"x" * 900_000)
    install(monkeypatch, connection)
    with pytest.raises(urllib.error.HTTPError) as error:
        direct.DirectTransport(max_response_bytes=5)(
            urllib.request.Request("https://example.invalid")
        )
    assert error.value.code == 504
    assert len(error.value.read()) == direct.ERROR_BODY_BYTES
    error.value.close()
    assert connection.closed


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.invalid",
        "http://example.invalid",
        "https://u:p@example.invalid",
        "https://example.invalid/#x",
    ],
)
def test_invalid_targets_rejected_before_connect(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install(monkeypatch, Connection())
    with pytest.raises(ValueError):
        direct.DirectTransport()(urllib.request.Request(url))
    assert not calls
