"""Attribute frozen TiDB retrieval bad cases without provider calls.

The command reads only authenticated local artifacts and publishes one
aggregate-only report under the gitignored TiDB index tree::

    uv run python scripts/evaluate_tidb_bad_cases.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from zhrag.eval.tidb_bad_cases import (
    ATTRIBUTION_CATEGORIES,
    BAD_CASES_SCHEMA,
    evaluate_tidb_bad_cases,
    validate_bad_case_report,
)
from zhrag.io_utils import exclusive_lock, read_json, read_jsonl, replace_files, write_json

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
REPORT_NAME = "bad_case_report.json"
OPERATION_LOCK = ".bad-cases.lock"
ARTIFACT_LOCK = ".artifacts.lock"


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute frozen TiDB retrieval bad cases without network calls."
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    return parser.parse_args(argv)


def _object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise SystemExit(f"! expected a JSON object: {path}")
    return value


def _load_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    eval_root = args.artifacts / "eval"
    active_locks = tuple(
        path for path in (eval_root / ".pool.lock", eval_root / ".qrels.lock") if path.exists()
    )
    if active_locks:
        joined = ", ".join(str(path) for path in active_locks)
        raise SystemExit(
            f"! input writer lock exists; refusing to read a changing bundle: {joined}"
        )
    required = (
        args.artifacts / "state.json",
        eval_root / "pool_report.json",
        eval_root / "qrels_report.json",
        eval_root / "runs.jsonl",
        eval_root / "qrels.jsonl",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "! TiDB bad-case inputs are absent:\n  " + "\n  ".join(str(path) for path in missing)
        )
    try:
        report = evaluate_tidb_bad_cases(
            state=_object(args.artifacts / "state.json"),
            pool_report=_object(eval_root / "pool_report.json"),
            qrels_report=_object(eval_root / "qrels_report.json"),
            run_rows=read_jsonl(eval_root / "runs.jsonl"),
            qrel_rows=read_jsonl(eval_root / "qrels.jsonl"),
        )
        validate_bad_case_report(report)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! TiDB bad-case evaluation failed: {exc}") from exc
    return report


def _publish(args: argparse.Namespace, report: dict[str, Any]) -> Path:
    eval_root = args.artifacts / "eval"
    target = eval_root / REPORT_NAME
    staged = eval_root / f".{REPORT_NAME}.tmp"
    try:
        write_json(staged, report)
        replace_files(((staged, target),))
    finally:
        staged.unlink(missing_ok=True)
    return Path(target)


def _print_summary(report: dict[str, Any], path: Path) -> None:
    design = report["evaluation_design"]
    print(
        f"attributed {design['queries']:,} query surfaces across "
        f"{len(design['systems'])} frozen systems"
    )
    print(
        f"candidate window top-{design['candidate_depth']}; "
        f"evaluation cutoff top-{design['evaluation_cutoff']}"
    )
    print(f"{'system':<24} {'recall':>10} {'ranking':>10} {'success':>10}")
    categories = ATTRIBUTION_CATEGORIES
    for label in design["systems"]:
        counts = report["system_categories"][label]["counts"]
        print(
            f"{label:<24} "
            f"{counts[categories[0]]:>10} "
            f"{counts[categories[1]]:>10} "
            f"{counts[categories[2]]:>10}"
        )
    print("generation: not_evaluated (retrieval service exposes no generation stage)")
    print(f"wrote {BAD_CASES_SCHEMA} aggregate report to {path}")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    eval_root = args.artifacts / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    # The operation lock is acquired first; the shared lock remains held while
    # loading, authenticating, computing, and publishing the report.
    with exclusive_lock(eval_root / OPERATION_LOCK), exclusive_lock(eval_root / ARTIFACT_LOCK):
        report = _load_and_evaluate(args)
        target = _publish(args, report)
    _print_summary(report, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
