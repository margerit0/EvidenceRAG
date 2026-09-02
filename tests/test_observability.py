"""Synthetic tests for the provider-free M11 trace contract."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from test_service import FakeRetriever, result
from zhrag.retrieval.online import OnlineRetrievalResult
from zhrag.service.app import create_app
from zhrag.service.observability import (
    InMemoryTraceSink,
    TraceRecorder,
    profile_fingerprint,
    validate_trace_event,
)


def _recorder(sink: InMemoryTraceSink) -> TraceRecorder:
    return TraceRecorder(
        sink,
        profile_name="test-profile",
        profile_fingerprint_value=profile_fingerprint(
            profile_name="test-profile",
            embedding_profile="test-embedding",
            rerank_profile="test-reranker",
            rerank_enabled=True,
            output_limit=2,
        ),
        rerank_enabled=True,
        clock_ns=iter([10, 20, 30, 40]).__next__,
    )


def test_trace_span_emits_sanitized_start_and_end() -> None:
    sink = InMemoryTraceSink()
    recorder = _recorder(sink)
    span = recorder.start_span()
    span.end(
        status_code=200,
        outcome="success",
        candidate_count=2,
        output_count=1,
        stage_timings={
            "dense_encode_seconds": 0.01,
            "sparse_encode_seconds": 0.02,
            "dense_search_seconds": 0.03,
            "sparse_search_seconds": 0.04,
            "fusion_seconds": 0.05,
            "fetch_seconds": 0.06,
            "rerank_seconds": 0.07,
            "total_seconds": 0.28,
        },
    )

    events = sink.events
    assert len(events) == 2
    assert events[0]["event"] == "span_start"
    assert events[1]["event"] == "span_end"
    assert events[1]["failure_category"] == "none"
    assert "query" not in str(events)


def test_trace_validator_rejects_raw_fields() -> None:
    sink = InMemoryTraceSink()
    recorder = _recorder(sink)
    event = recorder.start_span().start_event()
    event["query"] = "SECRET_QUERY"
    with pytest.raises(ValueError, match="forbidden"):
        validate_trace_event(event)


def test_falsey_sink_is_still_used_and_events_are_deep_copied() -> None:
    class FalseySink(InMemoryTraceSink):
        def __bool__(self) -> bool:
            return False

    sink = FalseySink()
    recorder = _recorder(sink)
    recorder.start_span().end(
        status_code=200,
        outcome="success",
        candidate_count=1,
        output_count=1,
        stage_timings={
            name: 0.0
            for name in (
                "dense_encode_seconds",
                "sparse_encode_seconds",
                "dense_search_seconds",
                "sparse_search_seconds",
                "fusion_seconds",
                "fetch_seconds",
                "rerank_seconds",
                "total_seconds",
            )
        },
    )

    events = sink.events
    events[0]["profile_name"] = "mutated"
    cast(dict[str, object], events[1]["stage_timings"])["total_seconds"] = 99.0
    fresh = sink.events
    assert fresh[0]["profile_name"] == "test-profile"
    assert cast(dict[str, object], fresh[1]["stage_timings"])["total_seconds"] == 0.0


def test_span_ids_are_unique_across_recorder_instances() -> None:
    first = _recorder(InMemoryTraceSink()).start_span().start_event()["span_id"]
    second = _recorder(InMemoryTraceSink()).start_span().start_event()["span_id"]
    assert first != second


def test_start_clock_failure_disables_the_span() -> None:
    sink = InMemoryTraceSink()

    def broken_clock() -> int:
        raise RuntimeError("SECRET_CLOCK_DETAIL")

    recorder = TraceRecorder(
        sink,
        profile_name="test-profile",
        profile_fingerprint_value=profile_fingerprint(
            profile_name="test-profile",
            embedding_profile="test-embedding",
            rerank_profile="test-reranker",
            rerank_enabled=True,
            output_limit=2,
        ),
        rerank_enabled=True,
        clock_ns=broken_clock,
    )
    span = recorder.start_span()
    span.end(status_code=503, outcome="retrieval_failed")
    assert sink.events == ()


def test_end_clock_failure_keeps_the_valid_start_event() -> None:
    sink = InMemoryTraceSink()
    values = iter([10])
    recorder = TraceRecorder(
        sink,
        profile_name="test-profile",
        profile_fingerprint_value=profile_fingerprint(
            profile_name="test-profile",
            embedding_profile="test-embedding",
            rerank_profile="test-reranker",
            rerank_enabled=True,
            output_limit=2,
        ),
        rerank_enabled=True,
        clock_ns=values.__next__,
    )
    recorder.start_span().end(status_code=503, outcome="retrieval_failed")
    assert [event["event"] for event in sink.events] == ["span_start"]


def test_published_index_identity_is_bound_to_profile_fingerprint() -> None:
    common = {
        "profile_name": "test-profile",
        "embedding_profile": "test-embedding",
        "rerank_profile": "test-reranker",
        "rerank_enabled": True,
        "output_limit": 2,
    }
    without_identity = profile_fingerprint(**common)
    with_identity = profile_fingerprint(
        **common,
        published_index_identity="tidb_chunks_v1:abc123",
    )
    changed_identity = profile_fingerprint(
        **common,
        published_index_identity="tidb_chunks_v2:abc123",
    )
    assert without_identity != with_identity
    assert with_identity != changed_identity
    with pytest.raises(ValueError, match="forbidden"):
        profile_fingerprint(**common, published_index_identity="source-key")


def test_sink_failure_does_not_change_http_response() -> None:
    class BrokenSink:
        def emit(self, event: object) -> None:
            raise RuntimeError("SECRET_PROVIDER_DETAIL")

    fake = FakeRetriever()
    client = create_app(cast(Any, fake), trace_sink=cast(Any, BrokenSink()))
    response = TestClient(client, raise_server_exceptions=False).post(
        "/api/search", json={"query": "safe"}
    )
    assert response.status_code == 200
    assert "SECRET_PROVIDER_DETAIL" not in response.text


def test_http_success_and_errors_have_lifecycle_pairs() -> None:
    sink = InMemoryTraceSink()
    fake = FakeRetriever()
    app = create_app(cast(Any, fake), trace_sink=sink)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.post("/api/search", json={"query": "ok"}).status_code == 200
    assert client.post("/api/search", json={"query": "ok", "top_k": 3}).status_code == 422
    assert client.post("/api/search", json={"query": "ok", "extra": "SECRET"}).status_code == 422

    events = sink.events
    assert len(events) == 6
    for start, end in zip(events[::2], events[1::2], strict=True):
        assert start["event"] == "span_start"
        assert end["event"] == "span_end"
        assert start["span_id"] == end["span_id"]
    assert [event["outcome"] for event in events[1::2]] == [
        "success",
        "invalid_top_k",
        "invalid_request",
    ]


def test_profile_drift_result_is_still_failure_and_not_raw() -> None:
    sink = InMemoryTraceSink()
    fake = FakeRetriever()

    def drifted(query: str) -> OnlineRetrievalResult:
        return replace(result(query, settings=fake.settings), profile_name="changed")

    app = create_app(cast(Any, fake), trace_sink=sink, search_runner=drifted)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/search", json={"query": "safe"}
    )
    assert response.status_code == 503
    assert sink.events[-1]["outcome"] == "retrieval_failed"


def test_retrieval_failure_emits_a_sanitized_lifecycle_pair() -> None:
    sink = InMemoryTraceSink()
    detail = "SECRET_PROVIDER_DETAIL for SECRET_QUERY"
    app = create_app(cast(Any, FakeRetriever(error=RuntimeError(detail))), trace_sink=sink)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/search", json={"query": "SECRET_QUERY"}
    )

    assert response.status_code == 503
    assert [event["event"] for event in sink.events] == ["span_start", "span_end"]
    assert sink.events[-1]["outcome"] == "retrieval_failed"
    serialized = str(sink.events)
    assert "SECRET_QUERY" not in serialized
    assert "SECRET_PROVIDER_DETAIL" not in serialized


def test_admission_rejection_emits_its_own_lifecycle_pair() -> None:
    sink = InMemoryTraceSink()
    fake = FakeRetriever()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(query: str) -> OnlineRetrievalResult:
        entered.set()
        await release.wait()
        return result(query, settings=fake.settings)

    app = create_app(
        cast(Any, fake),
        max_concurrency=1,
        search_runner=blocked,
        trace_sink=sink,
    )

    async def exercise() -> tuple[int, int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            first = asyncio.create_task(http.post("/api/search", json={"query": "first"}))
            await entered.wait()
            second = await http.post("/api/search", json={"query": "second"})
            release.set()
            return (await first).status_code, second.status_code

    assert asyncio.run(exercise()) == (200, 429)
    end_events = [event for event in sink.events if event["event"] == "span_end"]
    assert sorted(str(event["outcome"]) for event in end_events) == [
        "admission_rejected",
        "success",
    ]
    for end in end_events:
        assert sum(event["span_id"] == end["span_id"] for event in sink.events) == 2
