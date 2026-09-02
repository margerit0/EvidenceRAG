"""Provider-free, privacy-preserving request observability for M11.

The service emits only an in-memory event contract. A sink may forward these
already-sanitized events elsewhere, but this module never receives a query,
passage, document identifier, score, provider payload, or exception detail.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid
from collections.abc import Callable, Mapping
from threading import Lock
from typing import Protocol

__all__ = [
    "FAILURE_CATEGORIES",
    "OBSERVABILITY_SCHEMA",
    "OUTCOMES",
    "STAGE_NAMES",
    "InMemoryTraceSink",
    "NoOpTraceSink",
    "TraceRecorder",
    "TraceSink",
    "TraceSpan",
    "profile_fingerprint",
    "validate_trace_event",
]

OBSERVABILITY_SCHEMA = "zhrag-m11-observability-v1"
STAGE_NAMES = (
    "dense_encode_seconds",
    "sparse_encode_seconds",
    "dense_search_seconds",
    "sparse_search_seconds",
    "fusion_seconds",
    "fetch_seconds",
    "rerank_seconds",
    "total_seconds",
)
FAILURE_CATEGORIES = (
    "none",
    "invalid_request",
    "invalid_top_k",
    "admission_rejected",
    "retrieval_failed",
)
OUTCOMES = ("success", *FAILURE_CATEGORIES[1:])
_EVENTS = ("span_start", "span_end")
_FORBIDDEN_KEY_PARTS = (
    "answer",
    "authorization",
    "credential",
    "cookie",
    "doc_id",
    "embedding",
    "exception",
    "passage",
    "password",
    "payload",
    "provider",
    "query",
    "question",
    "secret",
    "source",
    "score",
    "text",
    "token",
    "vector",
    "url",
)


class TraceSink(Protocol):
    """Receive one already-sanitized trace event."""

    def emit(self, event: Mapping[str, object]) -> None: ...


class NoOpTraceSink:
    """Default sink; it deliberately performs no I/O or event storage."""

    def emit(self, event: Mapping[str, object]) -> None:
        del event


class InMemoryTraceSink:
    """Thread-safe sink used by tests and local diagnostics."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._lock = Lock()

    def emit(self, event: Mapping[str, object]) -> None:
        validate_trace_event(event)
        with self._lock:
            self._events.append(copy.deepcopy(dict(event)))

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(copy.deepcopy(event) for event in self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


def _framed_update(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def profile_fingerprint(
    *,
    profile_name: str,
    embedding_profile: str,
    rerank_profile: str,
    rerank_enabled: bool,
    output_limit: int,
    retrieval_settings: Mapping[str, object] | None = None,
    published_index_identity: str | None = None,
) -> str:
    """Hash all supplied public retrieval-profile metadata."""
    names = (profile_name, embedding_profile, rerank_profile)
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError("profile names must be non-empty strings")
    if not isinstance(rerank_enabled, bool):
        raise TypeError("rerank_enabled must be boolean")
    if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 1:
        raise ValueError("output_limit must be a positive integer")
    settings = dict(retrieval_settings or {})
    if any(not isinstance(key, str) for key in settings):
        raise TypeError("retrieval setting keys must be strings")
    if any(any(part in key.lower() for part in _FORBIDDEN_KEY_PARTS) for key in settings):
        raise ValueError("retrieval settings contain a forbidden field")
    if published_index_identity is not None:
        if not isinstance(published_index_identity, str) or not published_index_identity:
            raise ValueError("published_index_identity must be a non-empty string")
        if any(part in published_index_identity.lower() for part in _FORBIDDEN_KEY_PARTS):
            raise ValueError("published index identity contains a forbidden field")
    payload = {
        "embedding_profile": embedding_profile,
        "output_limit": output_limit,
        "profile_name": profile_name,
        "rerank_enabled": rerank_enabled,
        "rerank_profile": rerank_profile,
        "retrieval_settings": settings,
        "schema": OBSERVABILITY_SCHEMA,
    }
    if published_index_identity is not None:
        payload["published_index_identity"] = published_index_identity
    rendered = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    return value


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _exact_keys(value: object, expected: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} contains a non-string key")
    return value


def _walk_forbidden(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if any(part in key.lower() for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"forbidden trace field at {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def _validate_stage_timings(value: object) -> dict[str, float]:
    stages = _exact_keys(value, set(STAGE_NAMES), "stage_timings")
    return {
        name: _finite_non_negative(stages[name], f"stage_timings.{name}") for name in STAGE_NAMES
    }


def validate_trace_event(event: Mapping[str, object]) -> None:  # noqa: PLR0912
    """Strictly validate one event and reject raw-content fields recursively."""
    _walk_forbidden(event)
    event_type = event.get("event")
    expected = {
        "schema",
        "event",
        "span_id",
        "profile_name",
        "profile_fingerprint",
        "rerank_enabled",
        "started_at_ns",
    }
    if event_type == "span_end":
        expected |= {
            "ended_at_ns",
            "status_code",
            "outcome",
            "failure_category",
            "candidate_count",
            "output_count",
            "stage_timings",
        }
    root = _exact_keys(event, expected, "trace event")
    if root["schema"] != OBSERVABILITY_SCHEMA:
        raise ValueError("trace event schema drift")
    if event_type not in _EVENTS:
        raise ValueError("trace event type is unknown")
    span_id = _non_empty(root["span_id"], "span_id")
    if not span_id.startswith("span-"):
        raise ValueError("span_id has an invalid format")
    _non_empty(root["profile_name"], "profile_name")
    _require_sha256(root["profile_fingerprint"], "profile_fingerprint")
    if not isinstance(root["rerank_enabled"], bool):
        raise ValueError("rerank_enabled must be boolean")
    started = _non_negative_integer(root["started_at_ns"], "started_at_ns")
    if event_type == "span_start":
        return

    ended = _non_negative_integer(root["ended_at_ns"], "ended_at_ns")
    if ended < started:
        raise ValueError("span end precedes span start")
    status = root["status_code"]
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise ValueError("status_code must be an HTTP status integer")
    outcome = root["outcome"]
    if outcome not in OUTCOMES:
        raise ValueError("unknown trace outcome")
    category = root["failure_category"]
    if category not in FAILURE_CATEGORIES:
        raise ValueError("unknown failure category")
    if category not in {"none", outcome}:
        raise ValueError("failure category does not match outcome")
    if category == "none" and outcome != "success":
        raise ValueError("only success may have no failure category")
    expected_status = {
        "success": 200,
        "invalid_request": 422,
        "invalid_top_k": 422,
        "admission_rejected": 429,
        "retrieval_failed": 503,
    }[outcome]
    if status != expected_status:
        raise ValueError("trace status does not match outcome")
    candidate_count = _non_negative_integer(root["candidate_count"], "candidate_count")
    output_count = _non_negative_integer(root["output_count"], "output_count")
    if output_count > candidate_count and outcome == "success":
        raise ValueError("output count exceeds candidate count")
    timings = root["stage_timings"]
    if outcome == "success":
        _validate_stage_timings(timings)
    elif timings is not None:
        raise ValueError("failed spans must not carry stage timings")


class TraceRecorder:
    """Create sanitized spans while isolating all sink and clock failures."""

    def __init__(
        self,
        sink: TraceSink | None,
        *,
        profile_name: str,
        profile_fingerprint_value: str,
        rerank_enabled: bool,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._sink = sink if sink is not None else NoOpTraceSink()
        self._profile_name = _non_empty(profile_name, "profile_name")
        self._profile_fingerprint = _require_sha256(
            profile_fingerprint_value,
            "profile_fingerprint",
        )
        if not isinstance(rerank_enabled, bool):
            raise TypeError("rerank_enabled must be boolean")
        self._rerank_enabled = rerank_enabled
        self._clock_ns = clock_ns

    def start_span(self) -> TraceSpan:
        try:
            started_at_ns = self._clock_ns()
            valid_clock = isinstance(started_at_ns, int) and started_at_ns >= 0
        except Exception:
            started_at_ns = 0
            valid_clock = False
        span = TraceSpan(
            self,
            span_id=f"span-{uuid.uuid4().hex}",
            started_at_ns=started_at_ns,
            enabled=valid_clock,
        )
        if valid_clock:
            self._emit(span.start_event())
        return span

    def _emit(self, event: Mapping[str, object]) -> None:
        try:
            validate_trace_event(event)
            self._sink.emit(event)
        except Exception:
            # Observability is strictly best effort and may never alter HTTP semantics.
            return


class TraceSpan:
    """One start/end pair. Repeated completion is ignored fail-closed."""

    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        span_id: str,
        started_at_ns: int,
        enabled: bool,
    ) -> None:
        self._recorder = recorder
        self._span_id = span_id
        self._started_at_ns = started_at_ns
        self._enabled = enabled
        self._ended = False
        self._lock = Lock()

    def _base(self, event: str) -> dict[str, object]:
        return {
            "schema": OBSERVABILITY_SCHEMA,
            "event": event,
            "span_id": self._span_id,
            "profile_name": self._recorder._profile_name,
            "profile_fingerprint": self._recorder._profile_fingerprint,
            "rerank_enabled": self._recorder._rerank_enabled,
            "started_at_ns": self._started_at_ns,
        }

    def start_event(self) -> dict[str, object]:
        return self._base("span_start")

    def end(
        self,
        *,
        status_code: int,
        outcome: str,
        candidate_count: int = 0,
        output_count: int = 0,
        stage_timings: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        if not self._enabled:
            return
        try:
            try:
                ended_at_ns = self._recorder._clock_ns()
            except Exception:
                return
            event = self._base("span_end")
            event.update(
                {
                    "ended_at_ns": ended_at_ns,
                    "status_code": status_code,
                    "outcome": outcome,
                    "failure_category": "none" if outcome == "success" else outcome,
                    "candidate_count": candidate_count,
                    "output_count": output_count,
                    "stage_timings": (
                        _validate_stage_timings(stage_timings) if outcome == "success" else None
                    ),
                }
            )
            self._recorder._emit(event)
        except Exception:
            return

    def __enter__(self) -> TraceSpan:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is not None:
            self.end(status_code=503, outcome="retrieval_failed")
