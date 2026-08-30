"""Evaluate the finalized TiDB 256/400/800 chunk profiles completely offline."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

from zhrag.eval.tidb_chunk_sweep import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PROFILE_IDS,
    sweep_runs_rows,
)
from zhrag.eval.tidb_chunk_sweep_artifacts import (
    ChunkSweepPaths,
    load_canonical_snapshot,
    load_finalized_profile,
)
from zhrag.eval.tidb_chunk_sweep_evaluation import (
    ChunkSweepEvaluation,
    evaluate_chunk_sweep,
)
from zhrag.io_utils import exclusive_lock, replace_files, write_json, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
CURATED = ROOT / "tidb-rag-curated"


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--curated", type=Path, default=CURATED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    if not 0 <= args.seed <= (1 << 63) - 1:
        parser.error("--seed must be in [0, 2**63 - 1]")
    return args


def _load_and_evaluate(
    paths: ChunkSweepPaths,
    *,
    resamples: int,
    seed: int,
) -> ChunkSweepEvaluation:
    snapshot = load_canonical_snapshot(paths)
    finalized = {
        profile_id: load_finalized_profile(paths, snapshot, profile_id)
        for profile_id in PROFILE_IDS
    }
    return evaluate_chunk_sweep(
        snapshot,
        finalized,
        resamples=resamples,
        seed=seed,
    )


def _stage(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _publish(paths: ChunkSweepPaths, evaluation: ChunkSweepEvaluation) -> Path:
    run_targets = {
        profile_id: paths.sweep_root / "runs" / f"{profile_id}.jsonl" for profile_id in PROFILE_IDS
    }
    samples_target = paths.sweep_root / "numeric_samples.json"
    report_target = paths.sweep_root / "report.json"
    run_staged = {profile_id: _stage(path) for profile_id, path in run_targets.items()}
    samples_staged = _stage(samples_target)
    report_staged = _stage(report_target)
    staged_paths = (*run_staged.values(), samples_staged, report_staged)
    try:
        for profile_id in PROFILE_IDS:
            write_jsonl(run_staged[profile_id], sweep_runs_rows(evaluation.runs[profile_id]))
        write_json(samples_staged, dict(evaluation.numeric_samples))
        write_json(report_staged, dict(evaluation.report))
        replace_files(
            (
                *((run_staged[profile_id], run_targets[profile_id]) for profile_id in PROFILE_IDS),
                (samples_staged, samples_target),
                (report_staged, report_target),
            )
        )
        return report_target
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    paths = ChunkSweepPaths(args.artifacts, args.curated)
    try:
        with exclusive_lock(paths.sweep_root / ".evaluate.lock"), ExitStack() as stack:
            for profile_id in PROFILE_IDS:
                stack.enter_context(
                    exclusive_lock(paths.profile_root(profile_id) / ".profile.lock")
                )
            stack.enter_context(exclusive_lock(paths.sweep_root / ".bundle.lock"))
            stack.enter_context(exclusive_lock(paths.artifacts / ".index.lock"))
            stack.enter_context(exclusive_lock(paths.eval_root / ".artifacts.lock"))
            evaluation = _load_and_evaluate(
                paths,
                resamples=args.resamples,
                seed=args.seed,
            )
            report = _publish(paths, evaluation)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! M7 chunk-sweep evaluation failed: {exc}") from exc
    print(f"published M7 chunk-sweep report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
