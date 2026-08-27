"""Evaluate the frozen TiDB runs against finalized qrels, entirely offline.

The command has no provider mode. It reads only local frozen artifacts and
publishes one aggregate-only report under the gitignored index tree:

    uv run python scripts/evaluate_tidb_retrieval.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from zhrag.eval.tidb_quality import (
    RUN_LABELS,
    evaluate_tidb_quality,
    validate_quality_report,
)
from zhrag.io_utils import (
    exclusive_lock,
    read_json,
    read_jsonl,
    replace_files,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
QUALITY_REPORT = Path("quality_report.json")
ARTIFACT_LOCK = ".artifacts.lock"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen TiDB retrieval artifacts without network calls."
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be >= 1")
    if not 0 <= args.seed <= (1 << 63) - 1:
        parser.error("--seed must be in [0, 2**63 - 1]")
    return args


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
            "! TiDB quality inputs are absent:\n  " + "\n  ".join(str(path) for path in missing)
        )
    try:
        report = evaluate_tidb_quality(
            state=_object(args.artifacts / "state.json"),
            pool_report=_object(eval_root / "pool_report.json"),
            qrels_report=_object(eval_root / "qrels_report.json"),
            run_rows=read_jsonl(eval_root / "runs.jsonl"),
            qrel_rows=read_jsonl(eval_root / "qrels.jsonl"),
            resamples=args.resamples,
            seed=args.seed,
        )
        validate_quality_report(report)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! TiDB quality evaluation failed: {exc}") from exc
    return report


def _publish(args: argparse.Namespace, report: dict[str, Any]) -> Path:
    eval_root = args.artifacts / "eval"
    target = eval_root / QUALITY_REPORT
    staged = eval_root / f"{QUALITY_REPORT.name}.tmp"
    try:
        write_json(staged, report)
        replace_files(((staged, target),))
    finally:
        staged.unlink(missing_ok=True)
    return target


def _fmt_p(value: float, *, at_floor: bool) -> str:
    rendered = f"{value:.2e}" if 0.0 < value < 1e-4 else f"{value:.4f}"
    return f"{rendered}*" if at_floor else rendered


def _print_summary(report: dict[str, Any], path: Path) -> None:
    design = report["evaluation_design"]
    metrics = report["system_metrics"]
    print(
        f"evaluated {design['pairs']:,} pair observations / {design['queries']:,} query surfaces "
        f"in {design['clusters']:,} {design['cluster_key']} clusters"
    )
    print("all returned top-10 prefixes are judged; no provider calls were available")
    print(f"{'system':<24} {'Hit@1':>8} {'R@1':>8} {'MRR@10':>8} {'bin nDCG':>9} {'graded':>8}")
    for label in RUN_LABELS:
        overall = metrics[label]["overall"]
        print(
            f"{label:<24} "
            f"{overall['full_hit_at_1']['mean']:>8.3f} "
            f"{overall['full_recall_at_1']['mean']:>8.3f} "
            f"{overall['full_mrr_at_10']['mean']:>8.3f} "
            f"{overall['full_binary_ndcg_at_10']['mean']:>9.3f} "
            f"{overall['graded_ndcg_at_10']['mean']:>8.3f}"
        )
    print("\nprimary pair-mean binary nDCG@10 contrasts (treatment - comparator):")
    for row in report["primary_contrasts"]:
        delta = row["delta"]
        counts = row["counts"]
        adjusted = _fmt_p(
            float(row["adjusted_p"]),
            at_floor=bool(row["adjusted_p_inherits_floor"]),
        )
        print(
            f"  {row['id']:<18} {delta['mean']:+.4f} "
            f"[{delta['low']:+.4f}, {delta['high']:+.4f}] "
            f"W/L/T={counts['wins']}/{counts['losses']}/"
            f"{counts['ties_nonzero'] + counts['ties_zero']} "
            f"p(Holm)={adjusted}{' reject' if row['reject'] else ''}"
        )
    print("* add-one Monte Carlo floor; not a strict less-than bound")
    print(f"wrote aggregate report to {path}")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    eval_root = args.artifacts / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    # The operation lock first, then the shared bundle lock held across loading
    # *and* publication: a writer that starts after the diagnostic check below
    # still cannot replace runs/qrels while they are being streamed.
    with exclusive_lock(eval_root / ".quality.lock"), exclusive_lock(eval_root / ARTIFACT_LOCK):
        report = _load_and_evaluate(args)
        target = _publish(args, report)
    _print_summary(report, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
