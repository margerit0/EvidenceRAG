"""Benchmark the running M8 HTTP service and publish aggregate-only artifacts.

    uv run --extra service python scripts/bench.py --requests 100 --warmup 10

The query fixture remains local and gitignored. Its exact ordered text is hashed
in memory; neither it nor any response passage is written to the numeric samples
or aggregate report. This script makes HTTP calls only and never reads ``.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from zhrag.io_utils import exclusive_lock, read_jsonl, replace_files, write_json
from zhrag.service.bench import (
    STAGE_NAMES,
    BenchmarkSample,
    build_benchmark_report,
    query_fixture_sha256,
    samples_artifact,
    samples_sha256,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES = ROOT / "indexes" / "tidb" / "eval" / "queries.jsonl"
DEFAULT_SAMPLES = ROOT / "indexes" / "tidb" / "eval" / "m8_http_samples.json"
DEFAULT_REPORT = ROOT / "indexes" / "tidb" / "eval" / "m8_http_report.json"
M8_ARTIFACT_LOCK = ".m8.lock"
FIXTURE_KEYS = ("question", "query")


@dataclass(frozen=True, slots=True)
class HealthProfile:
    name: str
    embedding_profile: str
    rerank_profile: str
    rerank_enabled: bool


@dataclass(frozen=True, slots=True)
class RunResult:
    samples: tuple[BenchmarkSample, ...]
    wall_seconds: float


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _validated_origin(raw_url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("url must be an absolute HTTP(S) origin")
    if parts.username is not None or parts.password is not None:
        raise ValueError("url must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("url must not contain query parameters or a fragment")
    path = parts.path.rstrip("/")
    base_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    origin = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    return base_url, origin


def _load_queries(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError("query fixture is missing")
    queries: list[str] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        present = [key for key in FIXTURE_KEYS if key in row]
        if len(present) != 1:
            raise ValueError(f"query fixture row {index} must contain exactly one query field")
        value = row[present[0]]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"query fixture row {index} has no non-empty query")
        queries.append(value.strip())
    if not queries:
        raise ValueError("query fixture is empty")
    return queries


def _query_schedule(
    queries: Sequence[str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    if not queries:
        raise ValueError("query fixture is empty")
    if count < 0:
        raise ValueError("request count must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    # A modular rotation is deterministic without retaining query ids. It covers
    # every fixture row before repeating and changes the starting row by seed.
    start = seed % len(queries)
    return [queries[(start + index) % len(queries)] for index in range(count)]


def _health_profile(raw: object) -> HealthProfile:
    if not isinstance(raw, Mapping):
        raise ValueError("health response is not an object")
    expected = {
        "status",
        "profile_name",
        "embedding_profile",
        "rerank_profile",
        "rerank_enabled",
        "max_concurrency",
    }
    if set(raw) != expected or raw["status"] != "ok":
        raise ValueError("health response contract drift")
    for key in ("profile_name", "embedding_profile", "rerank_profile"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"health response has no {key}")
    if not isinstance(raw["rerank_enabled"], bool):
        raise ValueError("health rerank_enabled is not boolean")
    maximum = raw["max_concurrency"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("health max_concurrency is invalid")
    return HealthProfile(
        name=raw["profile_name"],
        embedding_profile=raw["embedding_profile"],
        rerank_profile=raw["rerank_profile"],
        rerank_enabled=raw["rerank_enabled"],
    )


def _stages(raw: object, profile: HealthProfile) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError("successful search response is not an object")
    expected = {
        "profile_name",
        "embedding_profile",
        "rerank_profile",
        "rerank_enabled",
        "passages",
        "timings",
    }
    if set(raw) != expected:
        raise ValueError("search response contract drift")
    observed = (
        raw["profile_name"],
        raw["embedding_profile"],
        raw["rerank_profile"],
        raw["rerank_enabled"],
    )
    expected_profile = (
        profile.name,
        profile.embedding_profile,
        profile.rerank_profile,
        profile.rerank_enabled,
    )
    if observed != expected_profile:
        raise ValueError("search response profile differs from health response")
    timings = raw["timings"]
    if not isinstance(timings, Mapping) or set(timings) != set(STAGE_NAMES):
        raise ValueError("search response timings contract drift")
    stages: dict[str, float] = {}
    for name in STAGE_NAMES:
        value = timings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"search timing {name} is not numeric")
        stages[name] = float(value)
    return stages


async def _one_request(
    client: httpx.AsyncClient,
    query: str,
    *,
    profile: HealthProfile,
) -> tuple[int, float, dict[str, float] | None]:
    started = time.perf_counter()
    try:
        response = await client.post("/api/search", json={"query": query})
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            return response.status_code, elapsed, None
        return response.status_code, elapsed, _stages(response.json(), profile)
    except (httpx.HTTPError, json.JSONDecodeError, ValueError):
        elapsed = time.perf_counter() - started
        # 599 is a conventional local-client/network error bucket. It is never
        # confused with a server response and remains aggregate-only.
        return 599, elapsed, None


async def _measure(
    client: httpx.AsyncClient,
    queries: Sequence[str],
    *,
    concurrency: int,
    profile: HealthProfile,
) -> RunResult:
    semaphore = asyncio.Semaphore(concurrency)

    async def measured(index: int, query: str) -> BenchmarkSample:
        async with semaphore:
            status, elapsed, stages = await _one_request(client, query, profile=profile)
            return BenchmarkSample(index, status, elapsed, stages)

    started = time.perf_counter()
    samples = await asyncio.gather(*(measured(index, query) for index, query in enumerate(queries)))
    wall = time.perf_counter() - started
    return RunResult(samples=tuple(samples), wall_seconds=wall)


async def _run_http(
    *,
    base_url: str,
    warmup_queries: Sequence[str],
    measured_queries: Sequence[str],
    concurrency: int,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[HealthProfile, RunResult]:
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        transport=transport,
    ) as client:
        health_response = await client.get("/healthz")
        health_response.raise_for_status()
        profile = _health_profile(health_response.json())
        for query in warmup_queries:
            status, _elapsed, _stages_value = await _one_request(client, query, profile=profile)
            if status != 200:
                raise ValueError("warm-up request failed")
        measured = await _measure(
            client,
            measured_queries,
            concurrency=concurrency,
            profile=profile,
        )
    return profile, measured


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
    }


def _provider_stages_included(profile: HealthProfile) -> bool:
    return not profile.embedding_profile.startswith("cached-")


def _limitations(profile: HealthProfile) -> list[str]:
    providers_included = _provider_stages_included(profile)
    if not providers_included:
        provider = "Provider latency is excluded by the cache-backed service profile."
    elif profile.rerank_enabled:
        provider = "Embedding and rerank provider latency are included."
    else:
        provider = "Embedding provider latency is included; rerank is disabled."

    return [
        "This is a local HTTP benchmark, not a production or cloud SLA.",
        "Quality metrics and significance tests are reported separately.",
        provider,
        "Warm-up requests are excluded from every measured aggregate.",
    ]


def _publish(
    *,
    samples_path: Path,
    report_path: Path,
    samples: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    if samples_path.parent.resolve() != report_path.parent.resolve():
        raise ValueError("samples and report must share an artifact directory")
    staged_samples = samples_path.with_suffix(samples_path.suffix + ".tmp")
    staged_report = report_path.with_suffix(report_path.suffix + ".tmp")
    with exclusive_lock(report_path.parent / M8_ARTIFACT_LOCK):
        try:
            write_json(staged_samples, dict(samples))
            write_json(staged_report, dict(report))
            replace_files(((staged_samples, samples_path), (staged_report, report_path)))
        finally:
            staged_samples.unlink(missing_ok=True)
            staged_report.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    try:
        if args.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if args.requests < 1:
            raise ValueError("requests must be positive")
        if args.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if not 0 < args.timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        base_url, origin = _validated_origin(args.url)
        queries = _load_queries(args.queries)
        fixture_hash = query_fixture_sha256(queries)
        schedule = _query_schedule(
            queries,
            count=args.warmup + args.requests,
            seed=args.seed,
        )
        profile, measured = asyncio.run(
            _run_http(
                base_url=base_url,
                warmup_queries=schedule[: args.warmup],
                measured_queries=schedule[args.warmup :],
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
        )
        numeric = samples_artifact(
            measured.samples,
            measurement_wall_seconds=measured.wall_seconds,
        )
        aggregate = build_benchmark_report(
            samples=measured.samples,
            measurement_wall_seconds=measured.wall_seconds,
            samples_sha256=samples_sha256(numeric),
            profile={
                "name": profile.name,
                "embedding_profile": profile.embedding_profile,
                "rerank_profile": profile.rerank_profile,
                "rerank_enabled": profile.rerank_enabled,
                "provider_stages_included": _provider_stages_included(profile),
            },
            fixture={"sha256": fixture_hash, "count": len(queries)},
            configuration={
                "url_origin": origin,
                "warmup": args.warmup,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "timeout_seconds": args.timeout,
                "seed": args.seed,
            },
            environment=_environment(),
            limitations=_limitations(profile),
        )
        _publish(
            samples_path=args.samples,
            report_path=args.report,
            samples=numeric,
            report=aggregate,
        )
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"! benchmark failed ({type(exc).__name__})")
        return 1
    measurement = aggregate["measurement"]
    if not isinstance(measurement, Mapping):
        raise RuntimeError("validated report lost its measurement object")
    print(
        f"wrote {args.report.name}: success={measurement['success_count']}/"
        f"{measurement['request_count']} qps={measurement['successful_qps']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
