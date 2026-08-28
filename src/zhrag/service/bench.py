"""Aggregate-only statistics and validation for the M8 HTTP benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "BENCH_REPORT_SCHEMA",
    "BENCH_SAMPLES_SCHEMA",
    "STAGE_NAMES",
    "BenchmarkSample",
    "build_benchmark_report",
    "query_fixture_sha256",
    "samples_artifact",
    "samples_sha256",
    "validate_benchmark_report",
    "validate_benchmark_samples",
]

BENCH_SAMPLES_SCHEMA = "zhrag-m8-http-benchmark-samples-v1"
BENCH_REPORT_SCHEMA = "zhrag-m8-http-benchmark-report-v1"
PERCENTILE_METHOD: Literal["linear"] = "linear"
PERCENTILES = (50, 95, 99)
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
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "doc_id",
        "embedding",
        "passages",
        "per_request",
        "query",
        "question",
        "raw_responses",
        "scores",
        "source_key",
        "text",
        "vectors",
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One numeric measurement; it intentionally cannot hold query or passage data."""

    sequence: int
    status_code: int
    elapsed_seconds: float
    stages: Mapping[str, float] | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP status integer")
        _finite_non_negative(self.elapsed_seconds, "elapsed_seconds")
        if self.status_code == 200:
            if self.stages is None or set(self.stages) != set(STAGE_NAMES):
                raise ValueError("successful samples require every stage timing")
            for name in STAGE_NAMES:
                _finite_non_negative(self.stages[name], name)
        elif self.stages is not None:
            raise ValueError("failed samples must not carry stage timings")


def query_fixture_sha256(queries: Sequence[str]) -> str:
    """Hash the ordered in-memory fixture without retaining its text in artifacts."""
    if not queries:
        raise ValueError("query fixture must be non-empty")
    digest = hashlib.sha256()
    digest.update(b"zhrag-m8-query-fixture-v1\0")
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("every benchmark query must contain text")
        encoded = query.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exact_keys(value: object, *, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty latency sample")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()) or bool((array < 0).any()):
        raise ValueError("latency samples must be finite and non-negative")
    result = np.percentile(array, list(PERCENTILES), method=PERCENTILE_METHOD)
    return {
        f"p{percentile}_seconds": float(value)
        for percentile, value in zip(PERCENTILES, result, strict=True)
    }


def _sample_object(sample: BenchmarkSample) -> dict[str, object]:
    return {
        "elapsed_seconds": sample.elapsed_seconds,
        "sequence": sample.sequence,
        "stages": dict(sample.stages) if sample.stages is not None else None,
        "status_code": sample.status_code,
    }


def _canonical_samples_payload(samples_artifact: Mapping[str, object]) -> bytes:
    rendered = json.dumps(
        samples_artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (rendered + "\n").encode("utf-8")


def validate_benchmark_samples(raw: object) -> tuple[BenchmarkSample, ...]:
    """Validate the numeric artifact from which aggregate claims are recomputed."""
    root = _exact_keys(
        raw,
        expected={"schema", "measurement_wall_seconds", "samples"},
        name="samples artifact",
    )
    if root["schema"] != BENCH_SAMPLES_SCHEMA:
        raise ValueError(f"samples artifact is not {BENCH_SAMPLES_SCHEMA}")
    _finite_non_negative(root["measurement_wall_seconds"], "measurement_wall_seconds")
    rows = root["samples"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("samples must be a non-empty array")
    samples: list[BenchmarkSample] = []
    for index, row in enumerate(rows):
        record = _exact_keys(
            row,
            expected={"elapsed_seconds", "sequence", "stages", "status_code"},
            name=f"samples[{index}]",
        )
        raw_stages = record["stages"]
        stages: dict[str, float] | None = None
        if raw_stages is not None:
            stage_map = _exact_keys(
                raw_stages,
                expected=set(STAGE_NAMES),
                name=f"samples[{index}].stages",
            )
            stages = {name: _finite_non_negative(stage_map[name], name) for name in STAGE_NAMES}
        samples.append(
            BenchmarkSample(
                sequence=record["sequence"],
                status_code=record["status_code"],
                elapsed_seconds=_finite_non_negative(record["elapsed_seconds"], "elapsed_seconds"),
                stages=stages,
            )
        )
    sequences = [sample.sequence for sample in samples]
    if sequences != list(range(len(samples))):
        raise ValueError("sample sequences must be contiguous and ordered from zero")
    return tuple(samples)


def build_benchmark_report(
    *,
    samples: Sequence[BenchmarkSample],
    measurement_wall_seconds: float,
    samples_sha256: str,
    profile: Mapping[str, object],
    fixture: Mapping[str, object],
    configuration: Mapping[str, object],
    environment: Mapping[str, object],
    limitations: Sequence[str],
) -> dict[str, object]:
    """Build a report solely from numeric samples and allowlisted run metadata."""
    if not samples:
        raise ValueError("benchmark must contain at least one measured request")
    measurement_wall = _finite_non_negative(measurement_wall_seconds, "measurement_wall_seconds")
    if measurement_wall == 0:
        raise ValueError("measurement_wall_seconds must be positive")
    if len(samples_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in samples_sha256):
        raise ValueError("samples_sha256 must be lowercase SHA-256")
    successes = [sample for sample in samples if sample.status_code == 200]
    if not successes:
        raise ValueError("benchmark has no successful measured requests")
    statuses = Counter(str(sample.status_code) for sample in samples)
    request_count = len(samples)
    success_count = len(successes)
    stage_summary = {
        name: _percentiles([cast_stage(sample, name) for sample in successes])
        for name in STAGE_NAMES
    }
    report: dict[str, object] = {
        "schema": BENCH_REPORT_SCHEMA,
        "profile": dict(profile),
        "fixture": dict(fixture),
        "configuration": dict(configuration),
        "environment": dict(environment),
        "samples": {
            "schema": BENCH_SAMPLES_SCHEMA,
            "sha256": samples_sha256,
        },
        "measurement": {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": request_count - success_count,
            "error_rate": (request_count - success_count) / request_count,
            "status_counts": dict(sorted(statuses.items())),
            "wall_seconds": measurement_wall,
            "successful_qps": success_count / measurement_wall,
            "percentile_method": PERCENTILE_METHOD,
            "http_latency": _percentiles([sample.elapsed_seconds for sample in successes]),
            "stages": stage_summary,
        },
        "limitations": list(limitations),
    }
    validate_benchmark_report(report)
    return report


def cast_stage(sample: BenchmarkSample, name: str) -> float:
    if sample.stages is None:
        raise RuntimeError("successful sample lost its stages")
    return sample.stages[name]


def _walk_forbidden(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden report key at {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def validate_benchmark_report(raw: object) -> None:  # noqa: PLR0912, PLR0915
    """Reject extra/raw fields and enforce internally coherent aggregates."""
    _walk_forbidden(raw)
    root = _exact_keys(
        raw,
        expected={
            "schema",
            "profile",
            "fixture",
            "configuration",
            "environment",
            "samples",
            "measurement",
            "limitations",
        },
        name="benchmark report",
    )
    if root["schema"] != BENCH_REPORT_SCHEMA:
        raise ValueError(f"benchmark report is not {BENCH_REPORT_SCHEMA}")
    profile = _exact_keys(
        root["profile"],
        expected={
            "name",
            "embedding_profile",
            "rerank_profile",
            "rerank_enabled",
            "provider_stages_included",
        },
        name="profile",
    )
    for key in ("name", "embedding_profile", "rerank_profile"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"profile.{key} must be non-empty")
    for key in ("rerank_enabled", "provider_stages_included"):
        if not isinstance(profile[key], bool):
            raise ValueError(f"profile.{key} must be boolean")
    if profile["rerank_enabled"] and not profile["provider_stages_included"]:
        raise ValueError("rerank-enabled reports must include provider stages")
    fixture = _exact_keys(
        root["fixture"],
        expected={"sha256", "count"},
        name="fixture",
    )
    if (
        not isinstance(fixture["sha256"], str)
        or len(fixture["sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in fixture["sha256"])
    ):
        raise ValueError("fixture.sha256 must be lowercase SHA-256")
    _positive_integer(fixture["count"], "fixture.count")
    configuration = _exact_keys(
        root["configuration"],
        expected={
            "url_origin",
            "warmup",
            "requests",
            "concurrency",
            "timeout_seconds",
            "seed",
        },
        name="configuration",
    )
    if not isinstance(configuration["url_origin"], str) or not configuration["url_origin"]:
        raise ValueError("configuration.url_origin must be non-empty")
    warmup = configuration["warmup"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("configuration.warmup must be a non-negative integer")
    for key in ("requests", "concurrency"):
        _positive_integer(configuration[key], f"configuration.{key}")
    seed = configuration["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("configuration.seed must be an integer")
    if _finite_non_negative(configuration["timeout_seconds"], "timeout_seconds") == 0:
        raise ValueError("timeout_seconds must be positive")
    environment = _exact_keys(
        root["environment"],
        expected={"python", "platform", "machine", "processor_count"},
        name="environment",
    )
    for key in ("python", "platform", "machine"):
        if not isinstance(environment[key], str):
            raise ValueError(f"environment.{key} must be a string")
    if environment["processor_count"] is not None:
        _positive_integer(environment["processor_count"], "processor_count")
    sample_ref = _exact_keys(
        root["samples"],
        expected={"schema", "sha256"},
        name="samples",
    )
    if sample_ref["schema"] != BENCH_SAMPLES_SCHEMA:
        raise ValueError("samples schema drift")
    if (
        not isinstance(sample_ref["sha256"], str)
        or len(sample_ref["sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in sample_ref["sha256"])
    ):
        raise ValueError("samples.sha256 must be lowercase SHA-256")
    measurement = _exact_keys(
        root["measurement"],
        expected={
            "request_count",
            "success_count",
            "error_count",
            "error_rate",
            "status_counts",
            "wall_seconds",
            "successful_qps",
            "percentile_method",
            "http_latency",
            "stages",
        },
        name="measurement",
    )
    requests = _positive_integer(measurement["request_count"], "request_count")
    success = _positive_integer(measurement["success_count"], "success_count")
    error = measurement["error_count"]
    if isinstance(error, bool) or not isinstance(error, int) or error < 0:
        raise ValueError("error_count must be a non-negative integer")
    if success + error != requests or configuration["requests"] != requests:
        raise ValueError("measurement counts are inconsistent")
    error_rate = _finite_non_negative(measurement["error_rate"], "error_rate")
    if not math.isclose(error_rate, error / requests, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("error_rate is inconsistent")
    status_counts = measurement["status_counts"]
    if not isinstance(status_counts, Mapping) or not status_counts:
        raise ValueError("status_counts must be a non-empty object")
    counted = 0
    for status, count in status_counts.items():
        if not isinstance(status, str) or len(status) != 3 or not status.isdecimal():
            raise ValueError("status_counts keys must be HTTP status strings")
        counted += _positive_integer(count, f"status_counts.{status}")
    if counted != requests or status_counts.get("200") != success:
        raise ValueError("status_counts are inconsistent")
    wall = _finite_non_negative(measurement["wall_seconds"], "wall_seconds")
    if wall == 0:
        raise ValueError("wall_seconds must be positive")
    qps = _finite_non_negative(measurement["successful_qps"], "successful_qps")
    if not math.isclose(qps, success / wall, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("successful_qps is inconsistent")
    if measurement["percentile_method"] != PERCENTILE_METHOD:
        raise ValueError("percentile method drift")
    _validate_summary(measurement["http_latency"], "http_latency")
    stages = _exact_keys(measurement["stages"], expected=set(STAGE_NAMES), name="stages")
    for name in STAGE_NAMES:
        _validate_summary(stages[name], f"stages.{name}")
    limitations = root["limitations"]
    if (
        not isinstance(limitations, Sequence)
        or isinstance(limitations, (str, bytes))
        or not limitations
        or any(not isinstance(item, str) or not item for item in limitations)
    ):
        raise ValueError("limitations must be a non-empty string array")


def _validate_summary(raw: object, name: str) -> None:
    summary = _exact_keys(
        raw,
        expected={"p50_seconds", "p95_seconds", "p99_seconds"},
        name=name,
    )
    values = [
        _finite_non_negative(summary[f"p{percentile}_seconds"], name) for percentile in PERCENTILES
    ]
    if values != sorted(values):
        raise ValueError(f"{name} percentiles must be non-decreasing")


def samples_artifact(
    samples: Sequence[BenchmarkSample],
    *,
    measurement_wall_seconds: float,
) -> dict[str, object]:
    """Return the canonical gitignored numeric artifact used for certification."""
    artifact: dict[str, object] = {
        "schema": BENCH_SAMPLES_SCHEMA,
        "measurement_wall_seconds": measurement_wall_seconds,
        "samples": [_sample_object(sample) for sample in samples],
    }
    validate_benchmark_samples(artifact)
    return artifact


def samples_sha256(samples_artifact: Mapping[str, object]) -> str:
    validate_benchmark_samples(samples_artifact)
    return hashlib.sha256(_canonical_samples_payload(samples_artifact)).hexdigest()
