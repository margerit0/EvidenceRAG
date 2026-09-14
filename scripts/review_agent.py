"""Prepare separate local reviews and certify aggregate-only Agent quality reports.

Default is read-only status. No mode imports a provider, reads .env or calls LLMs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from zhrag.eval.agent_review import (
    CertifiedRun,
    certify_run,
    fingerprint,
    quality_report,
    review_template,
)
from zhrag.io_utils import (
    exclusive_lock,
    read_bytes,
    replace_files,
    write_json,
    write_jsonl,
    write_text,
)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _reject_constant(_value: str) -> object:
    raise ValueError("nonfinite JSON")


def _load(path: Path, *, jsonl: bool = False) -> Any:
    text = read_bytes(path).decode("utf-8", errors="strict")

    def parse(raw: str) -> object:
        return json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)

    return [parse(line) for line in text.splitlines() if line.strip()] if jsonl else parse(text)


def review_packet(run: CertifiedRun) -> str:
    """Plain local text; shuffled content hashes hide explicit method labels only."""
    lines = [
        "文档调查审核材料（本地文件；不得提交语料或答案）",
        "原始 trials.jsonl 不修改；在 reviews.jsonl 中按完整 trial_sha256 填写审核结果。",
        "方法标签已隐藏，但答案风格可能暴露方法；这不是严格双盲。",
        "criteria_met 逐项检查，不能依据 status 直接把任务标为成功。",
        "把答案拆成可核对陈述，只有被所引段落支持的陈述才计入 supported_claims。",
        "没有回答时 total_claims/supported_claims 均填 0。",
        "仅追问或拒答填写 appropriate_clarification_or_refusal，其他情况填 null。",
        "task_success = 全部验收项通过、动作匹配 expected_status、陈述全部有支持，"
        "且追问/拒答恰当。",
        "故障试次也需审核并标为失败，不得删除；reviewer 为实际审核者，完成后设 reviewed=true。",
        "哈希检查不能证明审核者身份或判断正确；任务 reviewed=false 时不会发布质量报告。",
    ]
    for trial in sorted(run.trials, key=lambda item: item.sha256):
        task, result = trial.task, trial.result
        lines.extend(
            [
                "\n" + "=" * 72,
                f"trial_sha256: {trial.sha256}",
                f"任务: {task.question}",
                f"快照: {task.snapshot}",
                f"任务已审核: {task.reviewed}; 期望动作: {task.expected_status}",
                "验收条件:",
                *[
                    f"  {index}. {criterion}"
                    for index, criterion in enumerate(task.acceptance_criteria, 1)
                ],
                "参考来源:",
                *[f"  {source}" for source in task.reference_sources],
                f"实际状态: {result['status']}",
                f"追问: {result['clarification']}",
                "答案:",
            ]
        )
        for block in result["blocks"]:
            lines.extend([str(block["text"]), f"引用编号: {block['citations']}"])
        lines.append("本次证据:")
        for source in result["sources"]:
            lines.extend(
                [
                    f"[{source['citation_id']}] {source['title']}",
                    f"来源: {source['source_key']}; {source['source_url']}",
                    str(source["text"]),
                ]
            )
    return "\n".join(lines) + "\n"


def _publish_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    try:
        write_json(temporary, value)
        replace_files(((temporary, path),))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare", action="store_true", help="create reviews and plain review packet"
    )
    modes.add_argument(
        "--report", action="store_true", help="require full review and recompute statistics"
    )
    modes.add_argument("--check", action="store_true", help="recompute and verify the saved report")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except (OSError, ValueError, TypeError, RecursionError, SystemExit):
        print("Review failed: check complete v2 run, fingerprints, paired trials and reviews.")
        return 1


def _execute(args: argparse.Namespace) -> int:
    from agent_tasks import _local_path  # noqa: PLC0415 - shared local artifact policy

    target = _local_path(args.run_dir)
    if not target.is_dir():
        raise ValueError("run directory is missing")
    # Validate every existing child, including possible output symlinks, before I/O.
    names = (
        "manifest.json",
        "trials.jsonl",
        "reviews.jsonl",
        "review-packet.txt",
        "quality_report.json",
        "quality_report.json.tmp",
        ".review.lock",
    )
    for name in names:
        _local_path(target / name)
    with exclusive_lock(target / ".review.lock"):
        run = certify_run(
            _load(target / "manifest.json"), _load(target / "trials.jsonl", jsonl=True)
        )
        if args.prepare:
            if any((target / name).exists() for name in ("reviews.jsonl", "review-packet.txt")):
                raise ValueError("existing review work must not be overwritten")
            rows = sorted(review_template(run), key=lambda row: str(row["trial_sha256"]))
            write_text(target / "review-packet.txt", review_packet(run))
            write_jsonl(target / "reviews.jsonl", rows)
            print("Prepared separate local review records and review-packet.txt; no labels filled.")
            return 0
        if args.report or args.check:
            reviews = _load(target / "reviews.jsonl", jsonl=True)
            report = quality_report(run, reviews, resamples=args.resamples, seed=args.seed)
            path = target / "quality_report.json"
            if args.check:
                if fingerprint(_load(path)) != fingerprint(report):
                    raise ValueError("saved report does not match recomputation")
                print("Quality report matches certified trials and current reviews.")
            else:
                _publish_json(path, report)
                print("Published aggregate-only quality_report.json from complete reviews.")
            return 0
        print(
            json.dumps(
                {
                    "certified_trials": len(run.trials),
                    "task_count": len(run.tasks),
                    "reviewed_tasks": sum(task.reviewed for task in run.tasks),
                    "methods": run.methods,
                    "reviews_exist": (target / "reviews.jsonl").is_file(),
                    "trials_sha256": run.trials_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
