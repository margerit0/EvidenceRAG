"""Synchronize tracked documentation from authenticated M7 chunk-sweep artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.tidb_chunk_sweep import (
    CHUNK_SWEEP_REPORT_SCHEMA,
    PROFILE_IDS,
    validate_report,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    ChunkSweepPaths,
    load_canonical_snapshot,
    load_finalized_profile,
)
from zhrag.eval.tidb_chunk_sweep_evaluation import evaluate_chunk_sweep
from zhrag.eval.tidb_runs import RRF_LABEL
from zhrag.io_utils import exclusive_lock, read_json, read_text, replace_files, write_text

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
CLAUDE_CONTEXT = ROOT / "CLAUDE.md"
DOCS_LOCK = ".docs.lock"


@dataclass(frozen=True, slots=True)
class _Target:
    path: Path
    regions: Mapping[str, Callable[[Mapping[str, Any]], str]]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--claude-context", type=Path, default=CLAUDE_CONTEXT)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    if not 0 <= args.seed <= (1 << 63) - 1:
        parser.error("--seed must be in [0, 2**63 - 1]")
    return args


def _object(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(f"! expected a JSON object: {path}")
    return raw


def _mapping(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row.get(name)
    if not isinstance(value, Mapping):
        raise SystemExit(f"! M7 report section {name!r} is not an object")
    return value


def _load_report(paths: ChunkSweepPaths, *, resamples: int, seed: int) -> dict[str, Any]:
    report_path = paths.sweep_root / "report.json"
    samples_path = paths.sweep_root / "numeric_samples.json"
    report = _object(report_path)
    samples = _object(samples_path)
    try:
        validate_report(report)
        snapshot = load_canonical_snapshot(paths)
        finalized = {
            profile_id: load_finalized_profile(paths, snapshot, profile_id)
            for profile_id in PROFILE_IDS
        }
        evaluation = evaluate_chunk_sweep(
            snapshot,
            finalized,
            resamples=resamples,
            seed=seed,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! M7 report authentication failed: {exc}") from exc
    if dict(evaluation.numeric_samples) != samples:
        raise SystemExit("! M7 numeric samples differ from deterministic offline recomputation")
    if dict(evaluation.report) != report:
        raise SystemExit("! M7 report differs from deterministic offline recomputation")
    if report.get("schema") != CHUNK_SWEEP_REPORT_SCHEMA:
        raise SystemExit("! M7 report schema drift")
    return report


def _profile_summary(report: Mapping[str, Any], profile_id: str) -> Mapping[str, Any]:
    return _mapping(_mapping(report, "profiles"), profile_id)


def _profile_estimate(report: Mapping[str, Any], profile_id: str) -> Mapping[str, Any]:
    return _mapping(_mapping(_mapping(report, "estimates"), profile_id), RRF_LABEL)


def _format_interval(row: Mapping[str, Any], metric: str) -> str:
    value = _mapping(row, metric)
    return (
        f"{float(value['mean']):.3f} [{float(value['ci_low']):.3f}, {float(value['ci_high']):.3f}]"
    )


def _readme(report: Mapping[str, Any]) -> str:
    lines = [
        "### M7 分块粒度 sweep（TiDB source-level known-item）",
        "",
        "> **探索性声明**：这是 `400-origin exploratory known-item source retrieval sweep`。",
        "> 先在 raw chunk 上分别构建 BM25 / dense-4096，再做 exact RRF k=10/depth=100，",
        "> 最终 arm 才按 source 首次出现折叠；不启用 rerank、Milvus 或 chat。",
        "",
        (
            "| profile | docs | chunks | split-trigger exceedance | "
            "exact reuse / required new vectors |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for profile_id in PROFILE_IDS:
        summary = _profile_summary(report, profile_id)
        exceedance = _mapping(summary, "split_trigger_exceedance")
        lines.append(
            f"| `{profile_id}` | {int(summary['documents']):,} | "
            f"{int(summary['chunks']):,} | {int(exceedance['count']):,} "
            f"({float(exceedance['rate']):.2%}) | "
            f"{int(summary['canonical_exact_reuse']):,} / "
            f"{int(summary['required_new_document_vectors']):,} |"
        )
    lines.extend(
        (
            "",
            (
                "> **主终点（overall，pair 内 direct/paraphrase 取均值；95% CI 为 "
                "245 个 source cluster 整簇 bootstrap）**："
            ),
            "",
            "| profile | origin-source MRR@10 | Hit@1 | Hit@10 |",
            "|---|---:|---:|---:|",
        )
    )
    for profile_id in PROFILE_IDS:
        estimate = _profile_estimate(report, profile_id)["overall"]
        lines.append(
            f"| `{profile_id}` | {_format_interval(estimate, 'origin_mrr_at_10')} | "
            f"{_format_interval(estimate, 'origin_hit_at_1')} | "
            f"{_format_interval(estimate, 'origin_hit_at_10')} |"
        )
    lines.extend(
        (
            "",
            "> **预声明 Family A（Holm 2-test）**：主终点比较 256−400 与 800−400；"
            "Family B 独立检验 direct/paraphrase difference-in-differences。",
            "",
            "| comparison | Δ MRR@10 [95% CI] | p | p(Holm) | W/L/T |",
            "|---|---:|---:|---:|---:|",
        )
    )
    families = _mapping(report, "families")
    rows = families["origin-source-efficacy"]
    if not isinstance(rows, list):
        raise SystemExit("! M7 efficacy family is malformed")
    for row in rows:
        if not isinstance(row, Mapping):
            raise SystemExit("! M7 efficacy row is malformed")
        lines.append(
            f"| `{row['comparison']}` | {float(row['delta']):.4f} "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{float(row['raw_p']):.4f} | {float(row['adjusted_p']):.4f} | "
            f"{row['wins']}/{row['losses']}/"
            f"{int(row['ties_nonzero']) + int(row['ties_zero'])} |"
        )
    lines.extend(
        (
            "",
            (
                "> 不能把未检出差异写成等价、无损或全局最优；source-level known-item "
                "retrieval 也不等于 answer-bearing passage recall。"
            ),
            (
                "> confirmed-source 只是 canonical 400-only judgement pool 的 secondary "
                "sensitivity，未判断 source 不是可靠负例。"
            ),
        )
    )
    return "\n".join(lines)


def _architecture(report: Mapping[str, Any]) -> str:
    profiles: list[str] = []
    for profile_id in PROFILE_IDS:
        summary = _profile_summary(report, profile_id)
        estimate = _profile_estimate(report, profile_id)["overall"]
        profiles.append(
            f"`{profile_id}` {int(summary['chunks']):,} chunks；"
            f"origin-source MRR@10 {_format_interval(estimate, 'origin_mrr_at_10')}"
        )
    details = "<br>".join(profiles)
    return (
        "| **M7** | **TiDB chunk sweep（source-level known-item）✅** | **0.5** | "
        "256/400/800；dense-4096 + char-bigram BM25 + exact RRF k=10；"
        "RRF 先于最终 arm 的 source collapse；不启用 rerank/chat | "
        f"{details}<br>"
        "主问题是 `400-origin exploratory known-item source retrieval sweep`，"
        "不是 passage-level answer relevance |"
    )


def _claude(_report: Mapping[str, Any]) -> str:
    return (
        "- `indexes/tidb/eval/chunk_sweep/v1/` — `build_tidb_chunk_sweep.py` 规划并认证 "
        "256/400/800 profile；`evaluate_tidb_chunk_sweep.py` 完全离线重建 BM25/dense/RRF、"
        "source collapse、numeric samples 与 source-cluster CI/paired bootstrap/Holm report。\n\n"
        "M7 需补齐的 document vector ID 仅为 256/800 相对 canonical 400 缺失的 "
        "exact chunk IDs，固定 batch=16；"
        "不调用 query embedding、rerank、chat 或 Milvus。所有 cache、runs、qrels 映射与报告"
        "均位于 `indexes/` gitignore 边界。"
    )


def _replace_region(text: str, name: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {name!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _render_target(target: _Target, report: Mapping[str, Any]) -> tuple[str, str]:
    before = read_text(target.path)
    after = before
    for name, renderer in target.regions.items():
        after = _replace_region(after, name, renderer(report), path=target.path)
    return before, after


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(args.readme, {"M7-CHUNK-SWEEP": _readme}),
        _Target(args.architecture, {"M7-CHUNK-SWEEP": _architecture}),
        _Target(args.claude_context, {"M7-CHUNK-SWEEP": _claude}),
    )


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = ChunkSweepPaths(args.artifacts, ROOT / "tidb-rag-curated")
    with (
        exclusive_lock(args.readme.parent / DOCS_LOCK),
        exclusive_lock(paths.sweep_root / ".bundle.lock"),
        exclusive_lock(paths.eval_root / ".artifacts.lock"),
    ):
        report = _load_report(paths, resamples=args.resamples, seed=args.seed)
        rendered = [(target.path, *_render_target(target, report)) for target in _targets(args)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                raise SystemExit(
                    f"! generated M7 documentation is stale: {', '.join(map(str, stale))}"
                )
            return ()
        if not stale:
            return ()
        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".m7-sync.tmp")
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
        print("M7 chunk-sweep documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} M7 documentation files")
    else:
        print("M7 chunk-sweep documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
