"""HTTP benchmark runner tests against a hermetic ASGI application."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from zhrag.service.bench import BENCH_REPORT_SCHEMA, STAGE_NAMES, validate_benchmark_report

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "bench.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_script_test_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_script()


def health() -> dict[str, object]:
    return {
        "status": "ok",
        "profile_name": "cached-local-no-rerank-v1",
        "embedding_profile": "cached-query-embedding-v1",
        "rerank_profile": "disabled-identity-fused-order-v1",
        "rerank_enabled": False,
        "max_concurrency": 2,
    }


def search_response() -> dict[str, object]:
    return {
        "profile_name": "cached-local-no-rerank-v1",
        "embedding_profile": "cached-query-embedding-v1",
        "rerank_profile": "disabled-identity-fused-order-v1",
        "rerank_enabled": False,
        "passages": [{"text": "SENTINEL_RESPONSE_TEXT", "doc_id": "SECRET_DOC"}],
        "timings": {name: 0.001 for name in STAGE_NAMES},
    }


def app(*, fail_query: str | None = None, delay: float = 0.0) -> FastAPI:
    service = FastAPI()
    service.state.queries = []
    service.state.active = 0
    service.state.max_active = 0

    @service.get("/healthz")
    async def health_route() -> dict[str, object]:
        return health()

    @service.post("/api/search")
    async def search_route(payload: dict[str, object]) -> JSONResponse:
        query = cast(str, payload["query"])
        service.state.queries.append(query)
        service.state.active += 1
        service.state.max_active = max(service.state.max_active, service.state.active)
        try:
            if delay:
                await asyncio.sleep(delay)
            if query == fail_query:
                return JSONResponse(status_code=503, content={"code": "failed", "message": "x"})
            return JSONResponse(content=search_response())
        finally:
            service.state.active -= 1

    return service


class TestInputs:
    def test_loads_question_or_query_without_other_payload_fields(self, tmp_path: Path) -> None:
        fixture = tmp_path / "queries.jsonl"
        fixture.write_text(
            json.dumps({"question": " 问题甲 ", "answer": "SECRET"}, ensure_ascii=False)
            + "\n"
            + json.dumps({"query": "问题乙", "text": "SECRET"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        assert bench._load_queries(fixture) == ["问题甲", "问题乙"]

    @pytest.mark.parametrize(
        "row",
        [
            {},
            {"question": "a", "query": "b"},
            {"question": ""},
            {"question": 1},
        ],
    )
    def test_malformed_fixture_fails_closed(self, tmp_path: Path, row: dict[str, object]) -> None:
        fixture = tmp_path / "queries.jsonl"
        fixture.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="query fixture"):
            bench._load_queries(fixture)

    def test_schedule_is_deterministic_and_covers_before_repeating(self) -> None:
        queries = ["a", "b", "c"]
        assert bench._query_schedule(queries, count=5, seed=1) == ["b", "c", "a", "b", "c"]
        assert bench._query_schedule(queries, count=5, seed=1) == bench._query_schedule(
            queries,
            count=5,
            seed=1,
        )

    @pytest.mark.parametrize(
        "url",
        ["127.0.0.1:8000", "ftp://example.com", "http://user:pass@example.com"],
    )
    def test_url_rejects_missing_scheme_or_credentials(self, url: str) -> None:
        with pytest.raises(ValueError):
            bench._validated_origin(url)


class TestHttpMeasurement:
    def test_warmup_is_sent_but_excluded_from_numeric_samples(self) -> None:
        service = app()
        transport = httpx.ASGITransport(app=service)
        profile, measured = asyncio.run(
            bench._run_http(
                base_url="http://testserver",
                warmup_queries=["warm-a", "warm-b"],
                measured_queries=["measure-a", "measure-b", "measure-c"],
                concurrency=1,
                timeout=5.0,
                transport=transport,
            )
        )
        assert profile.name == "cached-local-no-rerank-v1"
        assert service.state.queries == [
            "warm-a",
            "warm-b",
            "measure-a",
            "measure-b",
            "measure-c",
        ]
        assert [row.sequence for row in measured.samples] == [0, 1, 2]
        assert all(row.status_code == 200 for row in measured.samples)

    def test_concurrency_is_bounded(self) -> None:
        service = app(delay=0.02)
        transport = httpx.ASGITransport(app=service)
        _profile, measured = asyncio.run(
            bench._run_http(
                base_url="http://testserver",
                warmup_queries=[],
                measured_queries=[str(index) for index in range(6)],
                concurrency=2,
                timeout=5.0,
                transport=transport,
            )
        )
        assert len(measured.samples) == 6
        assert service.state.max_active == 2

    def test_failure_has_no_stages_and_does_not_store_response_payload(self) -> None:
        service = app(fail_query="bad")
        transport = httpx.ASGITransport(app=service)
        _profile, measured = asyncio.run(
            bench._run_http(
                base_url="http://testserver",
                warmup_queries=[],
                measured_queries=["good", "bad"],
                concurrency=1,
                timeout=5.0,
                transport=transport,
            )
        )
        assert measured.samples[0].stages is not None
        assert measured.samples[1].status_code == 503
        assert measured.samples[1].stages is None
        assert "SENTINEL" not in repr(measured)

    def test_failed_warmup_aborts_before_measurement(self) -> None:
        service = app(fail_query="bad")
        transport = httpx.ASGITransport(app=service)
        with pytest.raises(ValueError, match="warm-up"):
            asyncio.run(
                bench._run_http(
                    base_url="http://testserver",
                    warmup_queries=["bad"],
                    measured_queries=["never"],
                    concurrency=1,
                    timeout=5.0,
                    transport=transport,
                )
            )
        assert service.state.queries == ["bad"]

    def test_cache_backed_profile_is_named_as_provider_excluded(self) -> None:
        profile = bench.HealthProfile(
            name="cached-local",
            embedding_profile="cached-query-embedding-v1",
            rerank_profile="disabled-identity-fused-order-v1",
            rerank_enabled=False,
        )
        assert bench._provider_stages_included(profile) is False
        assert any("excluded" in value for value in bench._limitations(profile))


class TestPublication:
    def test_publishes_numeric_samples_before_report_marker(self, tmp_path: Path) -> None:
        samples_path = tmp_path / "samples.json"
        report_path = tmp_path / "report.json"
        samples = {
            "schema": "zhrag-m8-http-benchmark-samples-v1",
            "measurement_wall_seconds": 1.0,
            "samples": [
                {
                    "elapsed_seconds": 0.1,
                    "sequence": 0,
                    "stages": {name: 0.01 for name in STAGE_NAMES},
                    "status_code": 200,
                }
            ],
        }
        report = {
            "schema": BENCH_REPORT_SCHEMA,
            "sentinel": "report-marker",
        }
        bench._publish(
            samples_path=samples_path,
            report_path=report_path,
            samples=samples,
            report=report,
        )
        assert json.loads(samples_path.read_text(encoding="utf-8"))["schema"].endswith("samples-v1")
        assert json.loads(report_path.read_text(encoding="utf-8"))["sentinel"] == "report-marker"
        assert not list(tmp_path.glob("*.tmp"))
        assert not (tmp_path / bench.M8_ARTIFACT_LOCK).exists()

    def test_publication_failure_rolls_back_both_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        samples_path = tmp_path / "samples.json"
        report_path = tmp_path / "report.json"
        samples_path.write_text('{"old":"samples"}\n', encoding="utf-8")
        report_path.write_text('{"old":"report"}\n', encoding="utf-8")
        before = (
            samples_path.read_text(encoding="utf-8"),
            report_path.read_text(encoding="utf-8"),
        )
        real_replace = bench.os.replace
        calls = {"n": 0}

        def flaky(source: object, target: object) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("injected replacement failure")
            real_replace(source, target)

        monkeypatch.setattr(bench.os, "replace", flaky)
        with pytest.raises(OSError, match="injected replacement failure"):
            bench._publish(
                samples_path=samples_path,
                report_path=report_path,
                samples={"schema": "new-samples"},
                report={"schema": "new-report"},
            )
        monkeypatch.undo()
        assert (
            samples_path.read_text(encoding="utf-8"),
            report_path.read_text(encoding="utf-8"),
        ) == before
        assert not list(tmp_path.glob("*.tmp"))
        assert not list(tmp_path.glob("*.rollback"))
        assert not (tmp_path / bench.M8_ARTIFACT_LOCK).exists()

    def test_builds_valid_aggregate_without_response_or_fixture_text(self) -> None:
        service = app()
        transport = httpx.ASGITransport(app=service)
        profile, measured = asyncio.run(
            bench._run_http(
                base_url="http://testserver",
                warmup_queries=["SENTINEL_WARMUP"],
                measured_queries=["SENTINEL_MEASURED"],
                concurrency=1,
                timeout=5.0,
                transport=transport,
            )
        )
        numeric = bench.samples_artifact(
            measured.samples,
            measurement_wall_seconds=measured.wall_seconds,
        )
        aggregate = bench.build_benchmark_report(
            samples=measured.samples,
            measurement_wall_seconds=measured.wall_seconds,
            samples_sha256=bench.samples_sha256(numeric),
            profile={
                "name": profile.name,
                "embedding_profile": profile.embedding_profile,
                "rerank_profile": profile.rerank_profile,
                "rerank_enabled": profile.rerank_enabled,
                "provider_stages_included": False,
            },
            fixture={"sha256": "a" * 64, "count": 2},
            configuration={
                "url_origin": "http://testserver",
                "warmup": 1,
                "requests": 1,
                "concurrency": 1,
                "timeout_seconds": 5.0,
                "seed": 0,
            },
            environment={
                "python": "3.13",
                "platform": "test",
                "machine": "test",
                "processor_count": 1,
            },
            limitations=["test"],
        )
        validate_benchmark_report(aggregate)
        rendered = json.dumps({"numeric": numeric, "aggregate": aggregate})
        assert "SENTINEL" not in rendered
        assert "SECRET_DOC" not in rendered
