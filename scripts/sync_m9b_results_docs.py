"""Publish M9b2 generation results from an authenticated, text-free report.

Hard rule: no number in a tracked document is ever transcribed by hand. So this
synchronizer does not trust ``report.json``. It reloads the numeric samples,
re-runs ``build_generation_report`` with the design parameters the report itself
records, and refuses to render unless the recomputation is byte-identical. A
report that drifted from its samples -- or samples edited after publication --
fails closed rather than reaching the README.

Only ``numeric_samples.json`` and ``report.json`` are read. Both are text-free
by construction: anonymous case/cluster ordinals, task names, metric values,
allowlisted fingerprints and counts. No corpus, cache, question, answer,
prediction, ``.env`` or provider payload is opened here.
"""

from __future__ import annotations

import argparse
import ntpath
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.crud_generation import (
    M9B_TASKS,
    PUBLIC_INPUT_FINGERPRINT_KEYS,
    build_generation_report,
    numeric_samples_sha256,
    validate_generation_report,
    validate_numeric_samples,
)
from zhrag.io_utils import exclusive_lock, read_json, read_text, replace_files, write_text

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
DEFAULT_ARTIFACTS = ROOT / "indexes" / "crud" / "generation" / "v1"
ARTIFACT_LOCK = ".artifacts.lock"
DOCS_LOCK = ".docs.lock"
NUMERIC_SAMPLES = "numeric_samples.json"
REPORT = "report.json"
RESULTS_MARKER = "M9B-GENERATION-RESULTS"

_SLUG = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")

#: Display label per published metric. Keyed by the frozen metric constants so a
#: metric added upstream fails the render instead of silently disappearing.
_GENERATION_LABELS: Mapping[str, str] = {
    "mean_sentence_bleu4": "BLEU-4（标准 sentence BLEU）",
    "crud_mean_sentence_bleu4_no_bp": "BLEU-4（CRUD 兼容口径，无 BP）",
    "mean_sentence_rouge_l_precision": "ROUGE-L Precision",
    "mean_sentence_rouge_l_recall": "ROUGE-L Recall",
    "mean_sentence_rouge_l_f1": "ROUGE-L F1",
    "mean_bertscore_precision_zh_rescaled": "BERTScore Precision（中文，rescaled）",
    "mean_bertscore_recall_zh_rescaled": "BERTScore Recall（中文，rescaled）",
    "mean_bertscore_f1_zh_rescaled": "BERTScore F1（中文，rescaled）",
}
_RAGQUEST_LABELS: Mapping[str, str] = {
    "paper_recall_all_questions": "Recall（论文口径：全部问题为分母）",
    "paper_precision_all_questions": "Precision（论文口径：全部问题为分母）",
    "code_recall_reference_answerable": "Recall（历史代码口径：reference 可答为分母）",
    "code_precision_generated_answerable": "Precision（历史代码口径：双方可答为分母）",
}
_SENTINEL_LABELS: Mapping[str, str] = {
    "reference_exact": "reference 答案 = 精确 sentinel",
    "reference_normalized": "reference 答案 = 归一化后 sentinel",
    "reference_near": "reference 答案 ≈ 近似 sentinel（未计入不可答）",
    "generated_exact": "prediction 答案 = 精确 sentinel",
    "generated_normalized": "prediction 答案 = 归一化后 sentinel",
    "generated_near": "prediction 答案 ≈ 近似 sentinel（未计入不可答）",
}


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
        description="Publish M9b2 generation results from an authenticated numeric report."
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--run-id", dest="run_id", required=True)
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="read runs/canonical/<run-id> instead of runs/trials/<run-id>",
    )
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _slug(value: str, *, label: str) -> str:
    # ntpath.isreserved covers CON/COM1/trailing dots on every platform, so a run
    # id that would alias an existing directory under Windows semantics is
    # rejected here rather than resolving to a neighbouring run's numbers.
    if not _SLUG.fullmatch(value) or ntpath.isreserved(value):
        raise SystemExit(f"! {label} must be a safe slug: {value!r}")
    return value


def _run_root(args: argparse.Namespace) -> Path:
    kind = "canonical" if args.canonical else "trials"
    return args.artifacts / "runs" / kind / _slug(args.run_id, label="--run-id")


def load_results(run_root: Path) -> dict[str, Any]:
    """Authenticate the published report by rebuilding it from numeric samples."""

    samples_path = run_root / NUMERIC_SAMPLES
    report_path = run_root / REPORT
    for path in (samples_path, report_path):
        if not path.is_file():
            raise SystemExit(f"! missing finalized artifact: {path}")
    try:
        samples = read_json(samples_path)
        report = read_json(report_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"! unreadable M9b report bundle under {run_root}: {exc}") from exc

    try:
        validate_numeric_samples(samples)
        validate_generation_report(report)
        typed = _mapping(report, "report")
        design = _mapping(typed["design"], "report design")
        digest = numeric_samples_sha256(_mapping(samples, "samples"))
        if _mapping(typed["samples"], "report samples")["sha256"] != digest:
            raise ValueError("report does not reference this numeric samples artifact")
        confidence = float(design["confidence"])
        resamples = int(design["resamples"])
        seed = int(design["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! M9b report authentication failed: {exc}") from exc

    recomputed = build_generation_report(
        _mapping(samples, "samples"),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    if recomputed != report:
        raise SystemExit(
            "! M9b report does not match deterministic recomputation from its numeric samples"
        )
    return dict(typed)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def _estimate(entry: Mapping[str, Any]) -> str:
    """Render one estimate as ``mean [low, high]``; an undefined ratio stays ``—``."""

    if entry["mean"] is None:
        return "—（分母为空）"
    return f"{_fmt(entry['mean'])} [{_fmt(entry['ci_low'])}, {_fmt(entry['ci_high'])}]"


def _labelled(names: Mapping[str, str], block: Mapping[str, Any], *, label: str) -> None:
    unknown = set(block) - set(names)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise SystemExit(f"! {label} has unlabelled entries; update the renderer: {joined}")


def _table(header: str, rows: list[tuple[str, ...]], columns: tuple[str, ...]) -> list[str]:
    lines = [header, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _result_lines(report: Mapping[str, Any], heading: str) -> list[str]:
    design = _mapping(report["design"], "design")
    tasks = {task: _mapping(report["tasks"][task], f"task {task}") for task in M9B_TASKS}
    confidence = float(design["confidence"])
    percent = f"{confidence * 100:.0f}%"

    lines = [
        heading,
        "",
        (
            f"合同 `{report['contract']}`，profile 为 {design['profile']}。"
            f"下表全部由 `scripts/sync_m9b_results_docs.py` 从 text-free 的 "
            f"`numeric_samples.json` 重算并与 `report.json` 逐字段核对后写入；"
            f"区间是 {percent} {design['bootstrap']}（resamples={int(design['resamples']):,}，"
            f"seed={int(design['seed'])}），"
            f"假设检验：{design['hypothesis_tests']}。"
        ),
        "",
    ]

    columns = (
        "指标",
        *(f"`{task}`（{int(tasks[task]['case_count']):,} cases）" for task in M9B_TASKS),
    )
    generation_rows: list[tuple[str, ...]] = []
    for task in M9B_TASKS:
        _labelled(
            _GENERATION_LABELS,
            _mapping(tasks[task]["generation"], "generation"),
            label=f"{task} generation block",
        )
    for metric, label in _GENERATION_LABELS.items():
        generation_rows.append(
            (
                label,
                *(
                    _estimate(_mapping(tasks[task]["generation"][metric], metric))
                    for task in M9B_TASKS
                ),
            )
        )
    lines.extend(
        _table("**生成质量（每 case 一个观测，case 即 cluster）**", generation_rows, columns)
    )
    lines.append("")

    quest_columns = (
        "指标",
        *(f"`{task}`（{int(tasks[task]['question_count']):,} questions）" for task in M9B_TASKS),
    )
    quest_rows: list[tuple[str, ...]] = []
    for task in M9B_TASKS:
        _labelled(
            _RAGQUEST_LABELS,
            _mapping(tasks[task]["ragquest"], "ragquest"),
            label=f"{task} ragquest block",
        )
    for metric, label in _RAGQUEST_LABELS.items():
        cells: list[str] = []
        for task in M9B_TASKS:
            entry = _mapping(tasks[task]["ragquest"][metric], metric)
            cells.append(f"{_estimate(entry)}（分母 {int(entry['denominator']):,}）")
        quest_rows.append((label, *cells))
    lines.extend(
        _table(
            "**RAGQuestEval（每 question 一个观测，多问题 case 以 parent case 作 cluster）**",
            quest_rows,
            quest_columns,
        )
    )
    lines.append("")

    sentinel_rows: list[tuple[str, ...]] = []
    for task in M9B_TASKS:
        _labelled(
            _SENTINEL_LABELS,
            _mapping(tasks[task]["sentinel_counts"], "sentinel_counts"),
            label=f"{task} sentinel block",
        )
    for key, label in _SENTINEL_LABELS.items():
        sentinel_rows.append(
            (
                label,
                *(f"{int(tasks[task]['sentinel_counts'][key]):,}" for task in M9B_TASKS),
            )
        )
    lines.extend(_table("**「无法推断」sentinel 判定分布**", sentinel_rows, quest_columns))
    lines.append("")

    inputs = _mapping(report["inputs"], "inputs")
    missing = PUBLIC_INPUT_FINGERPRINT_KEYS - set(inputs)
    if missing:
        raise SystemExit(f"! report inputs are incomplete: {', '.join(sorted(missing))}")
    lines.append("**输入与产物 fingerprint**（全部为 allowlisted SHA-256，不含路径或原文）：")
    lines.append("")
    lines.append("```text")
    lines.extend(f"{key}: {inputs[key]}" for key in sorted(inputs))
    lines.append(f"numeric_samples_sha256: {_mapping(report['samples'], 'samples')['sha256']}")
    lines.append(f"tokenizer_fingerprint: {report['tokenizer_fingerprint']}")
    lines.append(f"semantic_provenance_fingerprint: {report['semantic_provenance_fingerprint']}")
    lines.append("```")
    lines.append("")
    lines.append("**口径限制**：")
    lines.extend(f"- {item}" for item in report["limitations"])
    return lines


def _renderer(report: Mapping[str, Any], heading: str) -> Callable[[], str]:
    return lambda: "\n".join(_result_lines(report, heading))


def _replace_region(text: str, marker: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {marker!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _targets(args: argparse.Namespace, report: Mapping[str, Any]) -> tuple[_Target, ...]:
    return (
        _Target(
            args.readme,
            (
                (
                    RESULTS_MARKER,
                    _renderer(report, "### M9b2：CRUD-RAG 生成评测结果（known-context）"),
                ),
            ),
        ),
        _Target(
            args.architecture,
            (
                (
                    RESULTS_MARKER,
                    _renderer(report, "**M9b2：CRUD-RAG 生成评测结果（known-context）**"),
                ),
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
    """Authenticate the report, then atomically publish both document regions."""

    run_root = _run_root(args)
    with exclusive_lock(args.artifacts / ARTIFACT_LOCK):
        report = load_results(run_root)
    with exclusive_lock(args.readme.parent / DOCS_LOCK):
        rendered = [(target.path, *_render_target(target)) for target in _targets(args, report)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                joined = ", ".join(str(path) for path in stale)
                raise SystemExit(f"! generated M9b2 result documentation is stale: {joined}")
            return ()
        if not stale:
            return ()

        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".m9b-results.tmp")
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
        print("M9b2 result documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} M9b2 result documentation files")
    else:
        print("M9b2 result documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
