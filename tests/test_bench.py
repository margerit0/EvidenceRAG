"""M8 benchmark statistics and aggregate-only artifact tests."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from zhrag.service.bench import (
    BENCH_REPORT_SCHEMA,
    BENCH_SAMPLES_SCHEMA,
    STAGE_NAMES,
    BenchmarkSample,
    build_benchmark_report,
    query_fixture_sha256,
    samples_artifact,
    samples_sha256,
    validate_benchmark_report,
    validate_benchmark_samples,
)


def stages(value: float) -> dict[str, float]:
    return {name: value for name in STAGE_NAMES}


def sample(sequence: int, elapsed: float, *, status: int = 200) -> BenchmarkSample:
    return BenchmarkSample(
        sequence=sequence,
        status_code=status,
        elapsed_seconds=elapsed,
        stages=stages(elapsed / 2) if status == 200 else None,
    )


def profile() -> dict[str, object]:
    return {
        "name": "tidb-no-rerank-local-v1",
        "embedding_profile": "cached-query-embedding-v1",
        "rerank_profile": "disabled-identity-fused-order-v1",
        "rerank_enabled": False,
        "provider_stages_included": False,
    }


def fixture() -> dict[str, object]:
    return {"sha256": "a" * 64, "count": 3}


def configuration(*, requests: int = 3) -> dict[str, object]:
    return {
        "url_origin": "http://127.0.0.1:8000",
        "warmup": 1,
        "requests": requests,
        "concurrency": 1,
        "timeout_seconds": 30.0,
        "seed": 0,
    }


def environment() -> dict[str, object]:
    return {
        "python": "3.13.7",
        "platform": "Windows-11",
        "machine": "AMD64",
        "processor_count": 8,
    }


def report(
    rows: list[BenchmarkSample] | None = None,
    *,
    wall: float = 2.0,
) -> dict[str, object]:
    measured = rows or [sample(0, 1.0), sample(1, 2.0), sample(2, 100.0, status=503)]
    artifact = samples_artifact(measured, measurement_wall_seconds=wall)
    return build_benchmark_report(
        samples=measured,
        measurement_wall_seconds=wall,
        samples_sha256=samples_sha256(artifact),
        profile=profile(),
        fixture=fixture(),
        configuration=configuration(requests=len(measured)),
        environment=environment(),
        limitations=["Local HTTP benchmark; not a production SLA."],
    )


class TestSamples:
    def test_fixture_hash_binds_order_and_exact_text_without_storing_it(self) -> None:
        first = query_fixture_sha256(["问题甲", "问题乙"])
        assert first == query_fixture_sha256(["问题甲", "问题乙"])
        assert first != query_fixture_sha256(["问题乙", "问题甲"])
        assert first != hashlib.sha256("问题甲问题乙".encode()).hexdigest()
        assert len(first) == 64

    @pytest.mark.parametrize("queries", [[], [""], ["  \n"]])
    def test_fixture_hash_rejects_empty_queries(self, queries: list[str]) -> None:
        with pytest.raises(ValueError, match="query"):
            query_fixture_sha256(queries)

    def test_samples_artifact_is_numeric_and_hashes_canonical_bytes(self) -> None:
        artifact = samples_artifact(
            [sample(0, 1.0), sample(1, 2.0, status=429)],
            measurement_wall_seconds=3.0,
        )
        assert artifact["schema"] == BENCH_SAMPLES_SCHEMA
        rendered = (
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        assert samples_sha256(artifact) == hashlib.sha256(rendered.encode()).hexdigest()
        lowered = rendered.lower()
        for forbidden in ("query", "question", "answer", "doc_id", "text", "passage"):
            assert forbidden not in lowered

    def test_failed_sample_cannot_smuggle_stage_data(self) -> None:
        with pytest.raises(ValueError, match="failed samples"):
            BenchmarkSample(0, 500, 1.0, stages(0.1))

    def test_success_requires_all_stages(self) -> None:
        with pytest.raises(ValueError, match="every stage"):
            BenchmarkSample(0, 200, 1.0, {"total_seconds": 0.5})

    def test_validator_rejects_reordered_or_duplicate_sequences(self) -> None:
        artifact = samples_artifact(
            [sample(0, 1.0), sample(1, 2.0)],
            measurement_wall_seconds=3.0,
        )
        cast_rows = artifact["samples"]
        assert isinstance(cast_rows, list)
        cast_rows[1]["sequence"] = 0
        with pytest.raises(ValueError, match="contiguous"):
            validate_benchmark_samples(artifact)


class TestAggregation:
    def test_failed_requests_are_excluded_from_success_latency_and_stage_summaries(self) -> None:
        built = report()
        assert built["schema"] == BENCH_REPORT_SCHEMA
        measurement = built["measurement"]
        assert isinstance(measurement, dict)
        assert measurement["request_count"] == 3
        assert measurement["success_count"] == 2
        assert measurement["error_count"] == 1
        assert measurement["error_rate"] == pytest.approx(1 / 3)
        assert measurement["status_counts"] == {"200": 2, "503": 1}
        assert measurement["successful_qps"] == 1.0
        # If the 100-second failure leaked in, p99 would be near 100 rather than 1.99.
        assert measurement["http_latency"] == {
            "p50_seconds": 1.5,
            "p95_seconds": 1.95,
            "p99_seconds": 1.99,
        }
        for summary in measurement["stages"].values():
            assert summary == {
                "p50_seconds": 0.75,
                "p95_seconds": 0.975,
                "p99_seconds": 0.995,
            }

    def test_qps_uses_success_count_and_measurement_wall(self) -> None:
        built = report([sample(0, 0.2), sample(1, 0.2), sample(2, 0.2)], wall=0.5)
        measurement = built["measurement"]
        assert isinstance(measurement, dict)
        assert measurement["successful_qps"] == 6.0

    def test_all_failed_run_is_rejected(self) -> None:
        rows = [sample(0, 1.0, status=500), sample(1, 2.0, status=429)]
        with pytest.raises(ValueError, match="no successful"):
            build_benchmark_report(
                samples=rows,
                measurement_wall_seconds=3.0,
                samples_sha256="a" * 64,
                profile=profile(),
                fixture=fixture(),
                configuration=configuration(requests=2),
                environment=environment(),
                limitations=["local"],
            )

    def test_non_finite_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            BenchmarkSample(0, 200, math.nan, stages(1.0))


class TestReportValidator:
    @pytest.mark.parametrize(
        "path, value, message",
        [
            (("measurement", "error_rate"), 0.0, "error_rate"),
            (("measurement", "successful_qps"), 999.0, "qps"),
            (("measurement", "success_count"), 3, "counts"),
            (("measurement", "status_counts"), {"200": 3}, "status_counts"),
            (("measurement", "percentile_method"), "nearest", "method"),
        ],
    )
    def test_rejects_internally_inconsistent_aggregates(
        self,
        path: tuple[str, str],
        value: object,
        message: str,
    ) -> None:
        corrupted = copy.deepcopy(report())
        container = corrupted[path[0]]
        assert isinstance(container, dict)
        container[path[1]] = value
        with pytest.raises(ValueError, match=message):
            validate_benchmark_report(corrupted)

    @pytest.mark.parametrize(
        "forbidden",
        ["query", "question", "answer", "doc_id", "source_key", "vectors", "per_request"],
    )
    def test_rejects_raw_or_per_request_fields_at_any_depth(self, forbidden: str) -> None:
        corrupted = copy.deepcopy(report())
        profile_value = corrupted["profile"]
        assert isinstance(profile_value, dict)
        profile_value[forbidden] = "SENTINEL"
        with pytest.raises(ValueError, match="forbidden"):
            validate_benchmark_report(corrupted)

    def test_rejects_extra_non_forbidden_fields(self) -> None:
        corrupted = copy.deepcopy(report())
        corrupted["notes"] = "manual claim"
        with pytest.raises(ValueError, match="keys differ"):
            validate_benchmark_report(corrupted)

    def test_rerank_enabled_requires_provider_stages(self) -> None:
        corrupted = copy.deepcopy(report())
        profile_value = corrupted["profile"]
        assert isinstance(profile_value, dict)
        profile_value["rerank_enabled"] = True
        with pytest.raises(ValueError, match="provider stages"):
            validate_benchmark_report(corrupted)

    def test_report_contains_no_fixture_text_or_document_identity(self) -> None:
        rendered = json.dumps(report(), ensure_ascii=False, sort_keys=True).lower()
        for forbidden in (
            "问题甲",
            '"query"',
            '"question"',
            '"answer"',
            '"doc_id"',
            '"source_key"',
            '"passage"',
        ):
            assert forbidden not in rendered

    def test_same_inputs_produce_byte_identical_aggregate(self) -> None:
        left = json.dumps(report(), sort_keys=True, separators=(",", ":"))
        right = json.dumps(report(), sort_keys=True, separators=(",", ":"))
        assert left == right
