"""Evaluate the independent CRUD-RAG H hybrid baseline, entirely offline.

The command can only consume the frozen 4096-dimensional caches. It never reads
``.env`` and has no provider, embedding, or rerank mode:

    uv run python scripts/evaluate_h_hybrid_mrl1024.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from zhrag.eval.hybrid_mrl1024 import (
    ARM_LABELS,
    H_LABEL,
    evaluate_hybrid_mrl1024,
    load_hybrid_mrl1024_inputs,
    validate_hybrid_mrl1024_report,
)
from zhrag.io_utils import exclusive_lock, replace_files, write_json

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
REPORT_NAME = "h_hybrid_rrf_mrl1024_report.json"
LOCK_NAME = ".h_hybrid_mrl1024.lock"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen CRUD-RAG H retrieval artifacts without network calls."
    )
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be >= 1")
    if not 0 <= args.seed <= (1 << 63) - 1:
        parser.error("--seed must be in [0, 2**63 - 1]")
    return args


def _required_paths(expanded: Path) -> tuple[Path, ...]:
    document_cache = expanded / "emb_cache_4096.jsonl"
    query_cache = expanded / "emb_cache_queries_4096.jsonl"
    return (
        expanded / "manifest.json",
        expanded / "corpus.jsonl",
        expanded / "qrels.jsonl",
        document_cache,
        document_cache.with_suffix(document_cache.suffix + ".meta.json"),
        query_cache,
        query_cache.with_suffix(query_cache.suffix + ".meta.json"),
    )


def _load_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    missing = [path for path in _required_paths(args.expanded) if not path.is_file()]
    if missing:
        raise SystemExit(
            "! frozen H inputs are absent; this command will not populate them:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    try:
        inputs = load_hybrid_mrl1024_inputs(args.expanded, require_frozen=True)
        report = evaluate_hybrid_mrl1024(
            inputs,
            resamples=args.resamples,
            seed=args.seed,
        )
        validate_hybrid_mrl1024_report(report)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! H hybrid evaluation failed: {exc}") from exc
    return report


def _publish(args: argparse.Namespace, report: dict[str, Any]) -> Path:
    target = args.expanded / REPORT_NAME
    staged = args.expanded / f"{REPORT_NAME}.tmp"
    try:
        write_json(staged, report)
        replace_files(((staged, target),))
    finally:
        staged.unlink(missing_ok=True)
    return target


def _fmt_p(value: float, *, at_floor: bool) -> str:
    rendered = f"{value:.2e}" if 0.0 < value < 1e-4 else f"{value:.4f}"
    return f"{rendered}†" if at_floor else rendered


def _print_summary(report: dict[str, Any], path: Path) -> None:
    inputs = report["inputs"]
    headline = report["metrics"]["headline"]
    print(
        f"evaluated {inputs['documents']:,} complete documents / "
        f"{inputs['queries']:,} queries; headline n={headline['n']:,}"
    )
    print("4096 cache -> client prefix-1024 + row L2; no provider or rerank calls")
    print(f"{'system':<34} {'R@1':>8} {'MRR@10':>8} {'nDCG@10':>9}")
    for arm in ARM_LABELS:
        metrics = headline["systems"][arm]
        print(
            f"{arm:<34} {metrics['R@1']['mean']:>8.3f} "
            f"{metrics['MRR@10']['mean']:>8.3f} "
            f"{metrics['nDCG@10']['mean']:>9.3f}"
        )

    print("\npredeclared Holm families (H is treatment):")
    for family, rows in report["contrasts"].items():
        rejected = sum(bool(row["reject"]) for row in rows)
        floors = sum(bool(row["adjusted_p_inherits_floor"]) for row in rows)
        print(f"  {family:<23} {len(rows):>2} tests, {rejected} rejected, {floors} floor-marked")
    retention = report["contrasts"]["retention-continuous"]
    example = next(row for row in retention if row["arity"] == 1 and row["metric"] == "nDCG@10")
    adjusted = _fmt_p(
        float(example["adjusted_p"]),
        at_floor=bool(example["adjusted_p_inherits_floor"]),
    )
    print(
        f"  H-G arity=1 nDCG@10 {example['delta']:+.4f} "
        f"[{example['ci_low']:+.4f}, {example['ci_high']:+.4f}], "
        f"p(Holm)={adjusted}"
    )
    print(f"H arm label: {H_LABEL}; G is a full-width difference reference, not rerank")
    print("† add-one Monte Carlo floor; not a strict less-than bound")
    print(f"wrote aggregate report to {path}")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    if not args.expanded.is_dir():
        raise SystemExit(f"! expanded artifact directory is absent: {args.expanded}")
    with exclusive_lock(args.expanded / LOCK_NAME):
        report = _load_and_evaluate(args)
        target = _publish(args, report)
    _print_summary(report, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
