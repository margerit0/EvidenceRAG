"""Synchronize M8 service and performance evidence from local numeric artifacts.

The benchmark query fixture and response passages remain gitignored. This script
validates the aggregate report, authenticates its canonical numeric samples, then
rebuilds every latency/QPS/status aggregate before tracked documentation changes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.io_utils import (
    exclusive_lock,
    read_json,
    read_text,
    replace_files,
    write_text,
)
from zhrag.service.bench import (
    STAGE_NAMES,
    build_benchmark_report,
    samples_sha256,
    validate_benchmark_report,
    validate_benchmark_samples,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb" / "eval"
README = ROOT / "README.md"
EVALUATION_DOC = ROOT / "docs" / "evaluation.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
CLAUDE_CONTEXT = ROOT / "CLAUDE.md"
SAMPLES_NAME = "m8_http_samples.json"
REPORT_NAME = "m8_http_report.json"
ARTIFACT_LOCK = ".m8.lock"
DOCS_LOCK = ".docs.lock"

_STAGE_LABELS = {
    "dense_encode_seconds": "dense encode",
    "sparse_encode_seconds": "sparse encode",
    "dense_search_seconds": "dense search",
    "sparse_search_seconds": "sparse search",
    "fusion_seconds": "fusion",
    "fetch_seconds": "fetch",
    "rerank_seconds": "rerank",
    "total_seconds": "service total",
}


@dataclass(frozen=True, slots=True)
class _Target:
    path: Path
    regions: Mapping[str, Callable[[Mapping[str, Any]], str]]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync tracked M8 documentation from authenticated local artifacts."
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="directory holding the shared docs lock taken by every synchronizer",
    )
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--evaluation", type=Path, default=EVALUATION_DOC)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--claude-context", type=Path, default=CLAUDE_CONTEXT)
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _object(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(f"! expected a JSON object: {path}")
    return raw


def _mapping(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row.get(name)
    if not isinstance(value, dict):  # pragma: no cover - strict validator guards this
        raise AssertionError(f"validated benchmark section {name!r} is absent")
    return value


def load_status(artifacts: Path) -> dict[str, Any]:
    """Authenticate the aggregate by rebuilding it from numeric HTTP samples."""
    samples_path = artifacts / SAMPLES_NAME
    report_path = artifacts / REPORT_NAME
    numeric = _object(samples_path)
    report = _object(report_path)
    try:
        samples = validate_benchmark_samples(numeric)
        validate_benchmark_report(report)
        digest = samples_sha256(numeric)
        sample_ref = _mapping(report, "samples")
        if sample_ref["sha256"] != digest:
            raise ValueError("samples SHA-256 does not match the canonical numeric artifact")
        wall = numeric["measurement_wall_seconds"]
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise ValueError("measurement_wall_seconds must be numeric")
        profile = dict(_mapping(report, "profile"))
        fixture = dict(_mapping(report, "fixture"))
        configuration = dict(_mapping(report, "configuration"))
        environment = dict(_mapping(report, "environment"))
        limitations = tuple(report["limitations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! M8 benchmark authentication failed: {exc}") from exc
    recomputed = build_benchmark_report(
        samples=samples,
        measurement_wall_seconds=float(wall),
        samples_sha256=digest,
        profile=profile,
        fixture=fixture,
        configuration=configuration,
        environment=environment,
        limitations=limitations,
    )
    if recomputed != report:
        raise SystemExit(
            "! M8 aggregate report does not match deterministic recomputation from numeric samples"
        )
    return report


def _seconds(summary: Mapping[str, Any], percentile: int) -> float:
    return float(summary[f"p{percentile}_seconds"])


def _milliseconds(summary: Mapping[str, Any], percentile: int) -> float:
    return _seconds(summary, percentile) * 1000.0


def _fmt_ms(value: float) -> str:
    return f"{value:,.1f} ms"


def _fmt_qps(value: float) -> str:
    return f"{value:.2f}"


def _profile_description(report: Mapping[str, Any]) -> str:
    profile = _mapping(report, "profile")
    if bool(profile["provider_stages_included"]):
        provider = (
            "包含 embedding 与已启用的 rerank provider 墙钟"
            if bool(profile["rerank_enabled"])
            else "包含 embedding provider 墙钟；rerank 已禁用"
        )
    else:
        provider = "cache-backed，本地测量不包含 provider 墙钟"
    return (
        f"`{profile['name']}`（embedding=`{profile['embedding_profile']}`；"
        f"rerank=`{profile['rerank_profile']}`；{provider}）"
    )


def _readme_headline(report: Mapping[str, Any]) -> str:
    """One plain-language inline sentence for the README; numbers come from the report."""
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    configuration = _mapping(report, "configuration")
    profile = _mapping(report, "profile")
    if bool(profile["provider_stages_included"]):
        scope = (
            "包含向量模型与重排模型的调用耗时"
            if bool(profile["rerank_enabled"])
            else "包含向量模型的调用耗时，未启用重排"
        )
    else:
        scope = "查询向量走本地缓存、未启用重排，不含模型调用耗时"
    return (
        f"本机 HTTP 基准（{int(measurement['request_count']):,} 次正式请求，并发 "
        f"{int(configuration['concurrency'])}，{scope}）："
        f"p50 {_fmt_ms(_milliseconds(latency, 50))}，p95 {_fmt_ms(_milliseconds(latency, 95))}，"
        f"吞吐 {_fmt_qps(float(measurement['successful_qps']))} QPS，成功 "
        f"{int(measurement['success_count']):,}/{int(measurement['request_count']):,}。"
    )


def _evaluation_benchmark(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    stages = _mapping(measurement, "stages")
    configuration = _mapping(report, "configuration")
    fixture = _mapping(report, "fixture")
    environment = _mapping(report, "environment")
    status_counts = _mapping(measurement, "status_counts")
    lines = [
        "### HTTP 服务与 M8 性能基准",
        "",
        (
            "FastAPI 服务复用同一条同步 `OnlineRetriever`：dense / sparse 两臂各取 100，"
            "客户端 exact RRF，再按 profile 选择 rerank；单文件前端只展示检索 passage 与阶段耗时，"
            "**不生成答案，也不调用 chat completion**。"
        ),
        "",
        (
            f"**正式 profile**：{_profile_description(report)}。通过 HTTP 完成 "
            f"{int(measurement['request_count']):,} 次正式请求（另有 "
            f"{int(configuration['warmup']):,} 次 warm-up，全部排除）；fixture 含 "
            f"{int(fixture['count']):,} 条本地 query，仅发布其 SHA-256。"
        ),
        "",
        "| HTTP 指标 | p50 | p95 | p99 |",
        "|---|---:|---:|---:|",
        (
            f"| 客户端端到端 | {_fmt_ms(_milliseconds(latency, 50))} | "
            f"{_fmt_ms(_milliseconds(latency, 95))} | "
            f"{_fmt_ms(_milliseconds(latency, 99))} |"
        ),
        "",
        (
            f"成功 **{int(measurement['success_count']):,}/"
            f"{int(measurement['request_count']):,}**，错误率 "
            f"**{float(measurement['error_rate']):.2%}**，成功吞吐 "
            f"**{_fmt_qps(float(measurement['successful_qps']))} QPS**；"
            f"测量窗口 {float(measurement['wall_seconds']):,.3f}s、并发 "
            f"{int(configuration['concurrency'])}。HTTP 状态："
            + " / ".join(f"{status}={int(count):,}" for status, count in status_counts.items())
            + "。"
        ),
        "",
        "| 服务阶段 | p50 | p95 | p99 |",
        "|---|---:|---:|---:|",
    ]
    for name in STAGE_NAMES:
        summary = _mapping(stages, name)
        lines.append(
            f"| {_STAGE_LABELS[name]} | {_fmt_ms(_milliseconds(summary, 50))} | "
            f"{_fmt_ms(_milliseconds(summary, 95))} | "
            f"{_fmt_ms(_milliseconds(summary, 99))} |"
        )
    lines.extend(
        (
            "",
            (
                f"> 百分位固定用 NumPy `linear`；环境为 `{environment['platform']}` / "
                f"Python `{environment['python']}` / `{environment['machine']}`。"
                "失败请求不进入成功 latency 或阶段百分位，QPS=成功数/正式测量墙钟。"
            ),
            (
                "> 这是本机 HTTP profile 的观测，不是公网或生产 SLA；质量指标与显著性检验另见 "
                "TiDB pooled-qrels 评估。numeric samples 只含 status、elapsed 与阶段秒数，"
                "不含 query、passage、doc id、向量或 provider payload。"
            ),
        )
    )
    return "\n".join(lines)


def _architecture_m8(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    configuration = _mapping(report, "configuration")
    profile = _mapping(report, "profile")
    provider = "provider-included" if profile["provider_stages_included"] else "cache-backed"
    return (
        "| **M8** | **服务层 + HTTP 延迟/QPS ✅ 2026-08-28** | **1.5** | "
        "✅ FastAPI + 单文件静态前端（阶段耗时条）+ `scripts/serve.py`；✅ "
        "`scripts/bench.py` + numeric-samples 认证 + `sync_m8_docs.py` | "
        f"profile `{profile['name']}`（{provider}），正式请求 "
        f"{int(measurement['success_count']):,}/{int(measurement['request_count']):,} 成功，"
        f"HTTP p50/p95/p99 **{_milliseconds(latency, 50):,.1f}/"
        f"{_milliseconds(latency, 95):,.1f}/{_milliseconds(latency, 99):,.1f} ms**，"
        f"**{float(measurement['successful_qps']):.2f} QPS**，并发 "
        f"{int(configuration['concurrency'])}；本机结果，不是生产 SLA，不与其他 profile 混写 |"
    )


def _architecture_status(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    profile = _mapping(report, "profile")
    return (
        "> M8 在线 pipeline、FastAPI/静态前端与 HTTP benchmark 已完成。当前认证 headline 只绑定 "
        f"`{profile['name']}`：p95 **{_milliseconds(latency, 95):,.1f} ms** / "
        f"**{float(measurement['successful_qps']):.2f} QPS**；"
        "provider-included 与 cache-backed profile 必须分表，不能混成一个性能数字。"
    )


def _architecture_checklist(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    return (
        "- [x] **M8：FastAPI 服务与 HTTP 性能基准。** 服务层只编排现有 "
        "`OnlineRetriever`，有严格输入、脱敏错误、metadata allowlist、fail-fast 并发 admission 与"
        "单文件静态 UI；benchmark 排除 warm-up，报告 p50/p95/p99、成功 QPS、错误率/status 与八阶段"
        f"耗时。当前认证 HTTP p95 **{_milliseconds(latency, 95):,.1f} ms** / "
        f"**{float(measurement['successful_qps']):.2f} QPS**。"
        "同步器从无文本 numeric samples 重算 aggregate；不调用 chat completion。"
    )


def _architecture_resume_evidence(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    profile = _mapping(report, "profile")
    return (
        "> M8 复现证据：`scripts/bench.py` 已对 cache-backed HTTP profile "
        f"`{profile['name']}` 完成 {int(measurement['request_count']):,} 次正式请求；"
        "原始 numeric samples 与聚合报告均在 gitignored `indexes/tidb/eval/`，"
        "由 `sync_m8_docs.py` 校验 SHA-256 并离线重算。"
    )


def _resume_performance(report: Mapping[str, Any]) -> str:
    """The inline performance sentence embedded in the ADR resume bullet."""
    measurement = _mapping(report, "measurement")
    latency = _mapping(measurement, "http_latency")
    configuration = _mapping(report, "configuration")
    profile = _mapping(report, "profile")
    return (
        f"当前认证的 cache-backed HTTP profile `{profile['name']}`（并发 "
        f"{int(configuration['concurrency'])}，不含 provider 墙钟）端到端 **p95 "
        f"{float(latency['p95_seconds']) * 1000.0:,.1f} ms / "
        f"{float(measurement['successful_qps']):.2f} QPS**。"
    )


def _claude_artifact(report: Mapping[str, Any]) -> str:
    measurement = _mapping(report, "measurement")
    profile = _mapping(report, "profile")
    return (
        "- `indexes/tidb/eval/{m8_http_samples,m8_http_report}.json` — "
        "`scripts/bench.py` 经 HTTP 测量；samples 仅含 status / elapsed / 八阶段秒数，report 仅含"
        f"聚合值。当前 `{profile['name']}` 为 "
        f"{int(measurement['request_count']):,} 个正式请求；由 `scripts/sync_m8_docs.py` "
        "校验 canonical SHA-256 并从 samples 精确重算后才同步文档。\n\n"
        "M8 artifacts 继续位于 `indexes/` gitignore 边界，禁止加入 query、passage、doc id、"
        "embedding、rerank score 或 provider payload；不同 provider/cache profile "
        "必须各自命名与报告。"
    )


def _replace_region(
    text: str,
    name: str,
    body: str,
    *,
    path: Path,
    inline: bool = False,
) -> str:
    start = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {name!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    if inline:
        return f"{prefix}{start}{body.strip()}{end}{suffix}"
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _render_target(
    target: _Target,
    report: Mapping[str, Any],
) -> tuple[str, str]:
    before = read_text(target.path)
    after = before
    for name, render in target.regions.items():
        if name in {"M8-RESUME-EVIDENCE", "M8-RESUME-PERFORMANCE"} and (
            f"<!-- BEGIN {name} -->" not in after or f"<!-- END {name} -->" not in after
        ):
            continue
        after = _replace_region(
            after,
            name,
            render(report),
            path=target.path,
            inline=name in {"M8-RESUME-PERFORMANCE", "M8-README-HEADLINE"},
        )
    return before, after


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(args.readme, {"M8-README-HEADLINE": _readme_headline}),
        _Target(args.evaluation, {"M8-SERVICE-BENCHMARK": _evaluation_benchmark}),
        _Target(
            args.architecture,
            {
                "M8-STATUS": _architecture_status,
                "M8-ROADMAP": _architecture_m8,
                "M8-CHECKLIST": _architecture_checklist,
                # Older fixture documents may not yet carry these optional markers;
                # the canonical repository document does, and new renders own them.
                "M8-RESUME-EVIDENCE": _architecture_resume_evidence,
                "M8-RESUME-PERFORMANCE": _resume_performance,
            },
        ),
        _Target(args.claude_context, {"M8-LOCAL-ARTIFACTS": _claude_artifact}),
    )


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    # Every docs synchronizer takes the repository lock first, then its artifact
    # lock. The benchmark publisher takes only .m8.lock, so no reverse order exists.
    with (
        exclusive_lock(args.repo_root / DOCS_LOCK),
        exclusive_lock(args.artifacts / ARTIFACT_LOCK),
    ):
        report = load_status(args.artifacts)
        rendered = [(target.path, *_render_target(target, report)) for target in _targets(args)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                joined = ", ".join(str(path) for path in stale)
                raise SystemExit(f"! generated M8 documentation is stale: {joined}")
            return ()
        if not stale:
            return ()

        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".m8-sync.tmp")
                write_text(temporary, after)
                staged.append((temporary, path))
            replace_files(staged)
        finally:
            for temporary, _path in staged:
                temporary.unlink(missing_ok=True)
    return stale


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    changed = synchronize(args)
    if args.check:
        print("M8 documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} M8 documentation files")
    else:
        print("M8 documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
