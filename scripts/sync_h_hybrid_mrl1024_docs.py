"""Synchronize the H hybrid baseline from its authenticated local report.

The aggregate report and all of its raw inputs are gitignored.  This script
recomputes the report byte-for-byte in memory before allowing any number into a
tracked document.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.hybrid_mrl1024 import (
    A_LABEL,
    ARM_LABELS,
    E_LABEL,
    G_LABEL,
    H_LABEL,
    evaluate_hybrid_mrl1024,
    load_hybrid_mrl1024_inputs,
    validate_hybrid_mrl1024_report,
)
from zhrag.io_utils import (
    exclusive_lock,
    read_json,
    read_text,
    replace_files,
    write_text,
)

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
CLAUDE_CONTEXT = ROOT / "CLAUDE.md"
REPORT_NAME = "h_hybrid_rrf_mrl1024_report.json"
ARTIFACT_LOCK = ".h_hybrid_mrl1024.lock"
DOCS_LOCK = ".docs.lock"

_ARM_NAMES = {
    A_LABEL: "A：BM25 char-bigram",
    E_LABEL: "E：dense-1024",
    H_LABEL: "H：A+E exact RRF",
    G_LABEL: "G：A+dense-4096 exact RRF",
}
_FAMILY_NAMES = {
    "efficacy-binary": "efficacy / binary",
    "efficacy-continuous": "efficacy / continuous",
    "retention-binary": "retention / binary",
    "retention-continuous": "retention / continuous",
}
_PERCENT_METRICS = {"R@1", "hit@1", "ALL@10"}


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
        description="Sync tracked H hybrid documentation from an authenticated local report."
    )
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--claude-context", type=Path, default=CLAUDE_CONTEXT)
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SystemExit(f"! expected a JSON object: {path}")
    return value


def load_status(expanded: Path) -> dict[str, Any]:
    """Validate and exactly recompute the aggregate report from frozen artifacts."""
    report_path = expanded / REPORT_NAME
    report = _object(report_path)
    try:
        validate_hybrid_mrl1024_report(report)
        design = report["evaluation_design"]
        if not isinstance(design, dict):
            raise ValueError("evaluation_design must be an object")
        resamples = design.get("resamples")
        seed = design.get("base_seed")
        if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
            raise ValueError("evaluation_design.resamples is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("evaluation_design.base_seed is invalid")
        recomputed = evaluate_hybrid_mrl1024(
            load_hybrid_mrl1024_inputs(expanded, require_frozen=True),
            resamples=resamples,
            seed=seed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! H aggregate report validation failed: {exc}") from exc
    if recomputed != report:
        raise SystemExit(
            "! H aggregate report failed deterministic recomputation from the frozen "
            "manifest/corpus/qrels/caches"
        )
    return report


def _summary(
    report: Mapping[str, Any],
    *,
    scope: str,
    arm: str,
    metric: str,
) -> Mapping[str, Any]:
    metrics = report["metrics"]
    block = metrics["headline"] if scope == "headline" else metrics["by_arity"][scope]
    value = block["systems"][arm][metric]
    if not isinstance(value, dict):
        raise AssertionError("validated metric summary is not an object")
    return value


def _fmt_metric(summary: Mapping[str, Any], metric: str) -> str:
    mean = float(summary["mean"])
    low = float(summary["low"])
    high = float(summary["high"])
    if metric in _PERCENT_METRICS:
        return f"{mean:.1%} [{low:.1%}, {high:.1%}]"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _fmt_p(value: float, *, at_floor: bool) -> str:
    rendered = f"{value:.2e}" if 0.0 < value < 1e-4 else f"{value:.4f}"
    return f"{rendered}†" if at_floor else rendered


def _fmt_delta(row: Mapping[str, Any]) -> str:
    metric = str(row["metric"])
    delta = float(row["delta"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if metric in _PERCENT_METRICS:
        return f"{delta * 100:+.2f}pp [{low * 100:+.2f}, {high * 100:+.2f}]"
    return f"{delta:+.4f} [{low:+.4f}, {high:+.4f}]"


def _short_arm(arm: str) -> str:
    return arm.split("-", 1)[0]


def _family_counts(report: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return [
        (
            family,
            len(rows),
            sum(bool(row["reject"]) for row in rows),
        )
        for family, rows in report["contrasts"].items()
    ]


def _readme_h(report: Mapping[str, Any]) -> str:
    inputs = report["inputs"]
    config = report["configuration"]
    design = report["evaluation_design"]
    lines = [
        "### H：dense-1024 + BM25 的独立 hybrid 基线",
        "",
        (
            f"H 已在 **{inputs['documents']:,} 篇完整新闻文档 / {inputs['queries']:,} 条 query** "
            "上独立运行。文档和 query 都从同一份已冻结 4096 维 cache 取前 1024 维后逐行 "
            "L2 重归一化；A/E 各取 top-100，再做等权客户端 exact RRF "
            f"(`k={config['rrf_k']}`，无 rerank、零 API 调用)。G 由同一 4096 维矩阵离线重建，"
            "只作 retention difference reference；I/J 仍是 G 上的付费 rerank，未改名。"
        ),
        "",
        (
            f"**历史 1doc 对齐（n={report['metrics']['headline']['n']:,}；不重复进入显著性 "
            "family）**"
        ),
        "",
        "| arm | R@1 [95% CI] | MRR@10 [95% CI] | nDCG@10 [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARM_LABELS:
        lines.append(
            "| "
            + " | ".join(
                (
                    _ARM_NAMES[arm],
                    _fmt_metric(_summary(report, scope="headline", arm=arm, metric="R@1"), "R@1"),
                    _fmt_metric(
                        _summary(report, scope="headline", arm=arm, metric="MRR@10"),
                        "MRR@10",
                    ),
                    _fmt_metric(
                        _summary(report, scope="headline", arm=arm, metric="nDCG@10"),
                        "nDCG@10",
                    ),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "**全部 query 按实际 gold arity 分层（每格均为 mean [pointwise 95% CI]）**",
            "",
            "| arity | n | arm | R@1 | hit@1 | ALL@10 | MRR@10 | nDCG@10 |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        )
    )
    for arity in (1, 2, 3):
        block = report["metrics"]["by_arity"][str(arity)]
        for index, arm in enumerate(ARM_LABELS):
            cells = [
                str(arity) if index == 0 else "",
                f"{block['n']:,}" if index == 0 else "",
                _ARM_NAMES[arm],
            ]
            cells.extend(
                _fmt_metric(_summary(report, scope=str(arity), arm=arm, metric=metric), metric)
                for metric in ("R@1", "hit@1", "ALL@10", "MRR@10", "nDCG@10")
            )
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        (
            "",
            (
                f"**预声明检验**：双尾；连续指标用 {design['resamples']:,} 次 centred paired "
                "query bootstrap，二元指标用 exact McNemar；四个 family 各自 Holm。"
            ),
            "",
            (
                "| family | arity | contrast | metric | delta [paired 95% CI] | "
                "W/L/T | raw p | Holm p |"
            ),
            "|---|---:|---|---|---:|---:|---:|---:|",
        )
    )
    for family, rows in report["contrasts"].items():
        for row in rows:
            raw_p = _fmt_p(float(row["raw_p"]), at_floor=bool(row["raw_p_at_floor"]))
            adjusted = _fmt_p(
                float(row["adjusted_p"]),
                at_floor=bool(row["adjusted_p_inherits_floor"]),
            )
            ties = int(row["ties_nonzero"]) + int(row["ties_zero"])
            lines.append(
                "| "
                + " | ".join(
                    (
                        _FAMILY_NAMES[family],
                        str(row["arity"]),
                        (
                            f"{_short_arm(str(row['treatment']))} − "
                            f"{_short_arm(str(row['comparator']))}"
                        ),
                        str(row["metric"]),
                        _fmt_delta(row),
                        f"{row['wins']}/{row['losses']}/{ties}",
                        raw_p,
                        f"{adjusted}{' *' if row['reject'] else ''}",
                    )
                )
                + " |"
            )

    counts = _family_counts(report)
    lines.extend(
        (
            "",
            "校正后拒绝数："
            + "；".join(
                f"{_FAMILY_NAMES[family]} {rejected}/{total}" for family, total, rejected in counts
            )
            + "。",
        )
    )
    retention_rejections = [
        row
        for family in ("retention-binary", "retention-continuous")
        for row in report["contrasts"][family]
        if row["reject"]
    ]
    if retention_rejections:
        detected = "、".join(
            f"arity={row['arity']} {row['metric']}" for row in retention_rejections
        )
        lines.append(f"H 与 G 检测到经校正差异：{detected}；方向必须结合 delta 读取。")
    else:
        lines.append(
            "H 与 G 的 retention families 未检测到经校正差异；这不是 non-inferiority / "
            "equivalence 证据，不能写成“1024 无损”或“两者等价”。"
        )
    lines.extend(
        (
            "",
            (
                "> `R@1` 在 arity=2/3 是分数型、上限分别为 1/2 与 1/3，只作描述，未塞进 "
                "McNemar。`†` 是 add-one Monte Carlo floor，不是严格 `<` 上界。全部 2,394 条 "
                "query 已参与既往探索，因此这是 exploratory benchmark，不是未触碰 test；CI 是 "
                "pointwise，结论只绑定本次 cache fingerprints。"
            ),
        )
    )
    return "\n".join(lines)


def _architecture_status(report: Mapping[str, Any]) -> str:
    h_r1 = _fmt_metric(_summary(report, scope="headline", arm=H_LABEL, metric="R@1"), "R@1")
    h_mrr = _fmt_metric(
        _summary(report, scope="headline", arm=H_LABEL, metric="MRR@10"),
        "MRR@10",
    )
    g_r1 = _fmt_metric(_summary(report, scope="headline", arm=G_LABEL, metric="R@1"), "R@1")
    counts = "；".join(
        f"{_FAMILY_NAMES[family]} {rejected}/{total} reject"
        for family, total, rejected in _family_counts(report)
    )
    return (
        f"H（A+dense-1024）已独立离线完成：1doc R@1 **{h_r1}**、MRR@10 **{h_mrr}**；"
        f"同次重建的 G R@1 **{g_r1}**。{counts}。H−G 是双尾 difference test，"
        "不是 non-inferiority/equivalence；I/J 仍绑定 dense-4096 的 G 与原 rerank cache。"
    )


def _architecture_m4(report: Mapping[str, Any]) -> str:
    h_r1 = _fmt_metric(_summary(report, scope="headline", arm=H_LABEL, metric="R@1"), "R@1")
    h_mrr = _fmt_metric(
        _summary(report, scope="headline", arm=H_LABEL, metric="MRR@10"),
        "MRR@10",
    )
    g_r1 = _fmt_metric(_summary(report, scope="headline", arm=G_LABEL, metric="R@1"), "R@1")
    return (
        "| **M4** | 混合检索 + RRF | ~~1.0~~ **0.5** | 客户端 char-bigram → "
        "SPARSE_FLOAT_VECTOR（IP）；`hybrid_search` + RRFRanker | **离线部分已完成 "
        "2026-08-19，分层于 2026-08-20 补齐**（`scripts/compare_dense_bm25.py` + "
        "`retrieval/fusion.py`）：1doc hybrid **79.9 / 0.881** vs BM25 75.9（p=1.6e-03，"
        "显著）vs dense 78.0（Holm p=0.231，不显著）；多证据上 dense 的完整证据召回更强，"
        "而同一 RRF 在 arity=3 ALL@10 比 dense 低 5.11pp。在线链路已用客户端精确 RRF "
        "落地（`retrieval/online.py`，两臂各 100 → 本地 RRF k=10/depth=100）；"
        f"H（A+dense-1024）已独立离线运行：1doc R@1 **{h_r1}** / MRR@10 "
        f"**{h_mrr}**，同次重建的 G R@1 **{g_r1}**。H−G 只作双尾 difference test，"
        "I/J 仍绑定 G；M4 剩把融合搬进 Milvus 服务端并复现数字（服务端 tie 顺序与本地 "
        "doc-id tie-break 不保证一致，属优化路径而非默认精确路径） |"
    )


def _architecture_checklist(report: Mapping[str, Any]) -> str:
    inputs = report["inputs"]
    h_ndcg = _fmt_metric(
        _summary(report, scope="headline", arm=H_LABEL, metric="nDCG@10"),
        "nDCG@10",
    )
    retention_families = ("retention-binary", "retention-continuous")
    retention = sum(
        bool(row["reject"]) for family in retention_families for row in report["contrasts"][family]
    )
    retention_total = sum(len(report["contrasts"][family]) for family in retention_families)
    return (
        "- [x] **H：dense-1024 + char-bigram BM25 的独立等权 exact RRF 基线。** "
        f"已在 {inputs['documents']:,} 篇完整文档 / {inputs['queries']:,} 条 query 上从同一 "
        "4096 维 cache 双侧前缀切片并重归一化，零 API、无 rerank；1doc nDCG@10 "
        f"**{h_ndcg}**。H−G 两个 retention family 共 {retention}/{retention_total} 项通过 Holm；"
        "未拒绝项只写“未检测到差异”，不写“无损/等价”。复现："
        "`scripts/evaluate_h_hybrid_mrl1024.py`。"
    )


def _claude_artifact(report: Mapping[str, Any]) -> str:
    inputs = report["inputs"]
    return (
        "- `crud-rag-subset/eval-expanded/h_hybrid_rrf_mrl1024_report.json` — "
        "`scripts/evaluate_h_hybrid_mrl1024.py`；"
        f"{inputs['documents']:,} docs / {inputs['queries']:,} queries，4096→1024 双侧前缀 "
        "L2 重归一化，A/E/H/G 聚合 CI + paired tests/Holm（完全离线）。\n\n"
        "该报告、有效矩阵 fingerprints 与所有输入继续位于目录级 gitignore 边界；"
        "cache miss 会失败，不会读取 `.env` 或调用 embedding/rerank provider。"
    )


def _replace_region(text: str, name: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {name!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _render_target(
    target: _Target,
    report: Mapping[str, Any],
) -> tuple[str, str]:
    before = read_text(target.path)
    after = before
    for name, render in target.regions.items():
        after = _replace_region(after, name, render(report), path=target.path)
    return before, after


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(args.readme, {"H-HYBRID-MRL1024-EVIDENCE": _readme_h}),
        _Target(
            args.architecture,
            {
                "H-HYBRID-MRL1024-STATUS": _architecture_status,
                "H-HYBRID-MRL1024-M4": _architecture_m4,
                "H-HYBRID-MRL1024-CHECKLIST": _architecture_checklist,
            },
        ),
        _Target(
            args.claude_context,
            {"H-HYBRID-MRL1024-ARTIFACT": _claude_artifact},
        ),
    )


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    docs_lock = args.readme.parent / DOCS_LOCK
    # All synchronizers take the repository-wide docs lock first, then their own
    # artifact lock. The fixed order prevents two report readers from deadlocking
    # while both want to replace README/ADR/CLAUDE.md.
    with exclusive_lock(docs_lock), exclusive_lock(args.expanded / ARTIFACT_LOCK):
        report = load_status(args.expanded)
        rendered = [(target.path, *_render_target(target, report)) for target in _targets(args)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                joined = ", ".join(str(path) for path in stale)
                raise SystemExit(f"! generated H documentation is stale: {joined}")
            return ()
        if not stale:
            return ()

        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".h-sync.tmp")
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
        print("H hybrid documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} H documentation files")
    else:
        print("H hybrid documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
