"""HTTP service tests with no provider, corpus, or Milvus dependency."""

from __future__ import annotations

import asyncio
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from zhrag.retrieval.online import (
    OnlineRetrievalResult,
    OnlineSettings,
    RankedPassage,
    StageTimings,
)
from zhrag.service.app import DISPLAY_TEXT_MAX_LENGTH, ServiceInfo, create_app
from zhrag.store import Passage

SHA256 = "a" * 64
SECRET_QUERY = "SENTINEL_QUERY_SECRET"
SECRET_TEXT = "SENTINEL_PASSAGE_SECRET"


class FakeRetriever:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.settings = OnlineSettings(
            profile_name="service-test-profile",
            embedding_profile="service-test-embedding",
            rerank_profile="service-test-reranker",
            dense_dimensions=2,
            arm_depth=2,
            fusion_depth=2,
            rrf_k=10,
            rerank_request_depth=2,
            rerank_apply_depth=2,
            output_limit=2,
        )
        self.error = error
        self.queries: list[str] = []

    def retrieve(self, query: str) -> OnlineRetrievalResult:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return result(query, settings=self.settings)


def passage(
    doc_id: str,
    *,
    text: str,
    metadata: dict[str, object] | None = None,
) -> Passage:
    return Passage(
        doc_id=doc_id,
        text=text,
        source_key=f"source:{doc_id}",
        document_sha256=SHA256,
        ordinal=1,
        metadata=cast(Any, metadata or {}),
    )


def timings() -> StageTimings:
    return StageTimings(
        dense_encode_seconds=0.01,
        sparse_encode_seconds=0.002,
        dense_search_seconds=0.003,
        sparse_search_seconds=0.004,
        fusion_seconds=0.0005,
        fetch_seconds=0.005,
        rerank_seconds=0.02,
        total_seconds=0.05,
    )


def result(query: str, *, settings: OnlineSettings) -> OnlineRetrievalResult:
    rows = (
        RankedPassage(
            rank=1,
            fused_rank=2,
            rerank_score=0.9,
            passage=passage(
                "doc-a",
                text="A" * (DISPLAY_TEXT_MAX_LENGTH + 5),
                metadata={
                    "path": "backup/overview.md",
                    "heading_path": "BR > 全量备份",
                    "source_url": "https://docs.example.invalid/backup",
                    "secret": SECRET_TEXT,
                },
            ),
        ),
        RankedPassage(
            rank=2,
            fused_rank=1,
            rerank_score=0.8,
            passage=passage("doc-b", text="second passage"),
        ),
    )
    return OnlineRetrievalResult(
        query=query,
        profile_name=settings.profile_name,
        embedding_profile=settings.embedding_profile,
        rerank_profile=settings.rerank_profile,
        dense_hits=(),
        sparse_hits=(),
        fused_candidates=("doc-b", "doc-a"),
        rerank_request=("doc-b", "doc-a"),
        passages=rows,
        timings=timings(),
    )


def client(
    retriever: FakeRetriever | None = None,
    *,
    static_dir: Path | None = None,
    max_concurrency: int = 1,
    search_runner: Any = None,
    trace_sink: Any = None,
) -> TestClient:
    fake = retriever or FakeRetriever()
    app = create_app(
        cast(Any, fake),
        static_dir=static_dir,
        max_concurrency=max_concurrency,
        search_runner=search_runner,
        trace_sink=trace_sink,
    )
    return TestClient(app, raise_server_exceptions=False)


class TestFactory:
    def test_importing_optional_service_package_does_not_load_fastapi(self) -> None:
        for name in [
            module
            for module in sys.modules
            if module == "zhrag.service" or module.startswith("zhrag.service.")
        ]:
            sys.modules.pop(name)
        sys.modules.pop("fastapi", None)

        imported = importlib.import_module("zhrag.service")

        assert imported.__name__ == "zhrag.service"
        assert "fastapi" not in sys.modules

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_concurrency(self, value: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            create_app(cast(Any, FakeRetriever()), max_concurrency=value)

    def test_rejects_mismatched_service_info(self) -> None:
        fake = FakeRetriever()
        info = ServiceInfo(
            profile_name="wrong",
            embedding_profile=fake.settings.embedding_profile,
            rerank_profile=fake.settings.rerank_profile,
            rerank_enabled=False,
        )
        with pytest.raises(ValueError, match="does not match"):
            create_app(cast(Any, fake), info=info)


class TestHttpContract:
    def test_health_is_metadata_only_and_does_not_retrieve(self) -> None:
        fake = FakeRetriever()
        response = client(fake).get("/healthz")

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "status": "ok",
            "profile_name": "service-test-profile",
            "embedding_profile": "service-test-embedding",
            "rerank_profile": "service-test-reranker",
            "rerank_enabled": True,
            "max_concurrency": 1,
        }
        assert fake.queries == []

    def test_root_serves_the_single_static_file(self) -> None:
        response = client().get("/")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "TiDB 文档检索" in response.text
        assert 'fetch("/api/search"' in response.text

    def test_missing_static_file_has_stable_error(self, tmp_path: Path) -> None:
        response = client(static_dir=tmp_path).get("/")
        assert response.status_code == 404
        assert response.json() == {
            "code": "static_unavailable",
            "message": "The web interface is unavailable.",
        }

    def test_success_response_is_allowlisted_truncated_and_top_k_limited(self) -> None:
        fake = FakeRetriever()
        response = client(fake).post(
            "/api/search",
            json={"query": "  如何备份？  ", "top_k": 1},
        )

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert fake.queries == ["如何备份？"]
        body = response.json()
        assert set(body) == {
            "profile_name",
            "embedding_profile",
            "rerank_profile",
            "rerank_enabled",
            "passages",
            "timings",
        }
        assert body["timings"] == {
            "dense_encode_seconds": 0.01,
            "sparse_encode_seconds": 0.002,
            "dense_search_seconds": 0.003,
            "sparse_search_seconds": 0.004,
            "fusion_seconds": 0.0005,
            "fetch_seconds": 0.005,
            "rerank_seconds": 0.02,
            "total_seconds": 0.05,
        }
        assert len(body["passages"]) == 1
        row = body["passages"][0]
        assert set(row) == {
            "rank",
            "fused_rank",
            "rerank_score",
            "doc_id",
            "source_key",
            "ordinal",
            "metadata",
            "text",
            "text_truncated",
        }
        assert row["metadata"] == {
            "heading_path": "BR > 全量备份",
            "path": "backup/overview.md",
            "source_url": "https://docs.example.invalid/backup",
        }
        assert SECRET_TEXT not in response.text
        assert row["text"] == "A" * DISPLAY_TEXT_MAX_LENGTH
        assert row["text_truncated"] is True

    @pytest.mark.parametrize(
        "payload, code",
        [
            ({"query": ""}, "invalid_request"),
            ({"query": "   \n"}, "invalid_request"),
            ({"query": "q", "extra": "x"}, "invalid_request"),
            ({"query": "q", "top_k": True}, "invalid_request"),
            ({"query": "q", "top_k": 0}, "invalid_request"),
        ],
    )
    def test_schema_errors_use_one_sanitized_contract(
        self,
        payload: dict[str, object],
        code: str,
    ) -> None:
        response = client().post("/api/search", json=payload)
        assert response.status_code == 422
        assert response.json() == {
            "code": code,
            "message": "The request body is invalid.",
        }

    def test_top_k_above_profile_limit_is_rejected_without_retrieval(self) -> None:
        fake = FakeRetriever()
        response = client(fake).post("/api/search", json={"query": "q", "top_k": 3})
        assert response.status_code == 422
        assert response.json() == {
            "code": "invalid_top_k",
            "message": "top_k must be between 1 and 2.",
        }
        assert fake.queries == []

    def test_dependency_exception_is_not_reflected(self) -> None:
        detail = f"provider exploded for {SECRET_QUERY} at D:/private/key.txt"
        response = client(FakeRetriever(error=RuntimeError(detail))).post(
            "/api/search",
            json={"query": SECRET_QUERY},
        )

        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "code": "retrieval_failed",
            "message": "Retrieval failed; try again later.",
        }
        assert SECRET_QUERY not in response.text
        assert "private" not in response.text

    def test_profile_drift_fails_closed(self) -> None:
        fake = FakeRetriever()

        def drifted(query: str) -> OnlineRetrievalResult:
            return replace(result(query, settings=fake.settings), profile_name="changed")

        response = client(fake, search_runner=drifted).post(
            "/api/search",
            json={"query": "q"},
        )
        assert response.status_code == 503
        assert response.json()["code"] == "retrieval_failed"


class TestAdmission:
    def test_concurrency_limit_rejects_instead_of_queueing(self) -> None:
        fake = FakeRetriever()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked(query: str) -> OnlineRetrievalResult:
            entered.set()
            await release.wait()
            return result(query, settings=fake.settings)

        app = create_app(cast(Any, fake), max_concurrency=1, search_runner=blocked)

        async def exercise() -> tuple[httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as http:
                first = asyncio.create_task(http.post("/api/search", json={"query": "first"}))
                await entered.wait()
                second = await http.post("/api/search", json={"query": "second"})
                release.set()
                return await first, second

        first, second = asyncio.run(exercise())
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json() == {
            "code": "service_busy",
            "message": "The service is at its concurrency limit.",
        }
