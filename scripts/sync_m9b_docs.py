"""Synchronize tracked M9b1 contract documentation without reading artifacts.

The M9b1 implementation is an offline-first orchestration contract. This
synchronizer publishes only its frozen design status and roadmap row; it never
reads a corpus, cache, report, .env file, provider, or optional model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import zhrag.eval.crud_generation as contract
from zhrag.io_utils import exclusive_lock, read_text, replace_files, write_text

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EVALUATION_DOC = ROOT / "docs" / "evaluation.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
DOCS_LOCK = ".docs.lock"
STATUS_MARKER = "M9B-GENERATION-STATUS"
ROADMAP_MARKER = "M9B-GENERATION-ROADMAP"


@dataclass(frozen=True, slots=True)
class _Target:
    path: Path
    regions: tuple[tuple[str, Callable[[], str]], ...]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize M9b1 generation-orchestration contract documentation."
    )
    parser.add_argument(
        "--evaluation",
        "--readme",
        dest="evaluation",
        type=Path,
        default=EVALUATION_DOC,
        help="detailed evaluation document (legacy alias: --readme)",
    )
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="directory holding the shared docs lock",
    )
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _tasks() -> str:
    return ", ".join(f"`{task}`" for task in contract.M9B_TASKS)


def _status_lines(heading: str) -> list[str]:
    return [
        heading,
        "",
        (
            f"M9b1 冻结合同 `{contract.M9B_CONTRACT_VERSION}`，当前 profile 是 known-context："
            "生成请求直接使用每条 case 的已知 context，暂不包含 retrieval stage。支持的 task 为 "
            f"{_tasks()}。这不是 CRUD-RAG Table 8 reproduction，也不产生端到端检索质量结论。"
        ),
        "",
        "**已交付的离线边界**：",
        (
            "- 输入、prompt、hash、cache row、question/reference bank 与 finalizer 均为纯内存合同；"
            f"输入 schema=`{contract.GENERATION_EXPERIMENT_SCHEMA}`，"
            f"cache row schema=`{contract.CACHE_ROW_SCHEMA}`。"
        ),
        (
            f"- generation、QG、reference QA、prediction QA 和 semantic stage 按依赖图断点续跑；"
            f"每 case 最多 {contract.MAX_QUESTIONS_PER_CASE} 个问题，prompt estimator 超过 "
            f"{contract.MAX_INPUT_ESTIMATED_TOKENS:,} 个估算 token 即拒绝截断。"
        ),
        (
            "- status/dry-run 是只读检查，不读取 `.env`，不导入 provider 或可选模型，不创建 cache、"
            "artifact directory 或 lock；付费 chat 与模型加载必须分别显式开启 guard。"
        ),
        (
            "- cache sidecar 重新认证 input manifest、prompt、model/endpoint、served model、"
            "父级 fingerprint 和完整性；final report 只能由完整、认证过的 cache 离线重建。"
        ),
        (
            "- 完成后的 provenance hardening 统一拒绝 Windows 路径别名与保留设备名，固定 artifact "
            "root 为项目内 `indexes/crud/generation/v1/`，拒绝运行/参考路径上的 symlink/junction，"
            "并要求 provider 显式返回且全程匹配 requested model；chat profile 域与公开 provenance "
            "envelope 已轮换到 v2，旧 v1 sidecar/samples/report 不会被新 runner 续跑、原地改写"
            "或发布。"
        ),
        "",
        "**artifact 与隐私边界**：",
        (
            "- 文本型 cache、question/reference bank 和中间答案只允许存在于 "
            "`indexes/crud/generation/v1/` 的 gitignored 本地树；不提交 corpus、source、question、"
            "answer、prediction、embedding、provider payload 或 BERTScore 权重。"
        ),
        (
            "- 发布的 `numeric_samples.json` 与 `report.json` 只含匿名 sequence/cluster、task、"
            "metric、allowlisted fingerprint 和统计计数，不含任何原文或可回溯的 "
            "case/question identity。"
        ),
        (
            "- generation 指标按 case 统计，RAGQuestEval 按 question 统计，多问题 case 以 parent "
            "case 作 cluster；单 profile 只发布 descriptive cluster-bootstrap 95% CI，"
            "不做假设检验。"
        ),
        "",
        (
            "**RAGQuestEval 口径**：问题只从 exact ground-truth reference 生成；reference QA 与 "
            "prediction QA 分别使用 reference 和 evaluated prediction 作为 context，source article "
            "不进入 QG 或任一 QA context。空的 conditional denominator 保持为 null/0，"
            "而不是伪造为零分。"
        ),
        "",
        "**真实实验状态**：",
        "尚无通过 `numeric_samples.json` 重算认证并同步到文档的 M9b2 真实结果；",
        "本 marker 不发布任何真实生成、QG、QA、BERTScore 或 RAGQuestEval 数字。",
        "真实实验继续受显式授权、append-only cache、固定 case 上限与 ignored-artifact 边界约束。",
        "",
        "**入口**：",
        "```bash",
        "uv run python scripts/run_crud_generation.py --status",
        "uv run python scripts/run_crud_generation.py --dry-run",
        "# 付费 chat：显式加 --allow-paid-provider，并按 stage 的 prerequisite 顺序执行",
        "# BERTScore：另需显式加 --allow-model-download；finalize 全程离线",
        "```",
    ]


def render_readme_status() -> str:
    """Render the README M9b1 status region from core contract constants."""

    return "\n".join(_status_lines("### M9b1：CRUD-RAG 生成评测编排合同"))


def render_architecture_status() -> str:
    """Render the ADR M9b1 status region from core contract constants."""

    return "\n".join(_status_lines("**M9b1：CRUD-RAG 生成评测编排合同**"))


def render_roadmap() -> str:
    """Render the single M9 roadmap row without experimental measurements."""

    tasks = " / ".join(contract.M9B_TASKS)
    return (
        "| **M9** | **生成侧评估（M9a ✅ / M9b1 编排合同 ✅；真实实验待授权）** | **2.0** | "
        "**M9a**：`eval/metrics_gen.py`、`eval/quest_eval.py`、合成测试、aggregate-only "
        "Table 8 evidence、独立文档同步；**M9b1**：纯内存合同、离线优先 CLI、QG/QA/生成 "
        f"stage DAG（{tasks}）与 text-free finalizer | M9b1 已冻结 cache/provenance/privacy "
        "合同；真实 provider 实验与结果数字不在本阶段发布 |"
    )


def _replace_region(text: str, marker: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {marker!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(args.evaluation, ((STATUS_MARKER, render_readme_status),)),
        _Target(
            args.architecture,
            (
                (STATUS_MARKER, render_architecture_status),
                (ROADMAP_MARKER, render_roadmap),
            ),
        ),
    )


def _render_target(target: _Target) -> tuple[str, str]:
    before = read_text(target.path)
    after = before
    for marker, render in target.regions:
        after = _replace_region(after, marker, render(), path=target.path)
    return before, after


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    """Render and atomically publish the two tracked M9b1 document regions."""

    with exclusive_lock(args.repo_root / DOCS_LOCK):
        rendered = [(target.path, *_render_target(target)) for target in _targets(args)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                joined = ", ".join(str(path) for path in stale)
                raise SystemExit(f"! generated M9b1 documentation is stale: {joined}")
            return ()
        if not stale:
            return ()

        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".m9b-sync.tmp")
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
        print("M9b1 documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} M9b1 documentation files")
    else:
        print("M9b1 documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
