"""Draw a deterministic M9b generation subset from the local CRUD-RAG split.

The full M9b1 case universe is 2,800 cases, which costs roughly 32k-36k paid
chat calls across the four stages -- about six days of sequential provider
traffic. A subset makes one honest end-to-end run affordable, but only if the
subset itself is reproducible: ``report.json`` pins ``dataset_snapshot_sha256``
over the exact input bytes, and nobody can rebuild those bytes from a prose
description of "we sampled 300 cases".

So the sample is a pure function of ``(input records, per-task size, seed)``:
records are ordered by upstream ID before drawing, so the source file's own
ordering cannot bias the draw, and the same seed reproduces the same bytes.

Only the fields M9b1 actually consumes are copied. That keeps the derived file
as small a corpus derivative as possible; it stays under the same gitignored
boundary as its source and must never be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from zhrag.eval.crud_generation import M9B_TASKS, build_generation_cases
from zhrag.io_utils import read_bytes, read_json, replace_files, write_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "crud-rag-subset" / "raw" / "split_merged.json"
DEFAULT_OUTPUT = ROOT / "crud-rag-subset" / "raw" / "split_merged_m9b_subset.json"

#: Exactly the fields ``build_generation_cases`` reads for each task. Copying
#: only these keeps unrelated upstream columns out of the derived file.
TASK_FIELDS: Mapping[str, tuple[str, ...]] = {
    "event_summary": ("ID", "text", "summary"),
    "questanswer_1doc": ("ID", "news1", "questions", "answers"),
}


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", dest="output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-summary", type=int, default=200)
    parser.add_argument("--questanswer-1doc", dest="questanswer_1doc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _sizes(args: argparse.Namespace) -> dict[str, int]:
    sizes = {
        "event_summary": args.event_summary,
        "questanswer_1doc": args.questanswer_1doc,
    }
    for task, size in sizes.items():
        if size < 1:
            raise SystemExit(f"! --{task.replace('_', '-')} must be positive")
    return sizes


def _records(raw: object, task: str) -> list[Mapping[str, object]]:
    if not isinstance(raw, Mapping):
        raise SystemExit("! dataset root must be a JSON object")
    rows = raw.get(task)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise SystemExit(f"! dataset task {task!r} must be a non-empty list")
    records: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SystemExit(f"! dataset task {task!r} contains a non-object record")
        records.append(row)
    return records


def _projected(record: Mapping[str, object], task: str) -> dict[str, object]:
    projected: dict[str, object] = {}
    for field in TASK_FIELDS[task]:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"! {task}: record is missing a non-blank {field!r}")
        projected[field] = value
    return projected


def sample_subset(raw: object, sizes: Mapping[str, int], *, seed: int) -> dict[str, list[object]]:
    """Draw ``sizes[task]`` records per task deterministically from ``raw``."""

    subset: dict[str, list[object]] = {}
    for task in M9B_TASKS:
        size = sizes[task]
        records = _records(raw, task)
        projected = [_projected(record, task) for record in records]
        identifiers = [str(record["ID"]) for record in projected]
        if len(set(identifiers)) != len(identifiers):
            raise SystemExit(f"! {task}: upstream IDs are not unique")
        if size > len(projected):
            raise SystemExit(f"! {task}: asked for {size:,} of only {len(projected):,} records")
        # Sort first so the draw depends on the record set, not on file order.
        ordered = sorted(projected, key=lambda record: str(record["ID"]))
        drawn = random.Random(f"{seed}:{task}").sample(range(len(ordered)), size)
        subset[task] = [ordered[index] for index in sorted(drawn)]
    return subset


def _publish(path: Path, subset: Mapping[str, list[object]]) -> bool:
    staged = Path(f"{path}.subset.tmp")
    try:
        write_json(staged, dict(subset))
        if path.exists():
            # The snapshot hash is baked into every downstream cache sidecar, so a
            # silent rewrite would strand an in-flight run against a vanished input.
            if read_bytes(staged) == read_bytes(path):
                return False
            raise SystemExit(f"! refusing to overwrite an existing subset: {path}")
        replace_files(((staged, path),))
    finally:
        staged.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    sizes = _sizes(args)
    try:
        raw = read_json(args.input_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"! unreadable dataset {args.input_path}: {exc}") from exc

    subset = sample_subset(raw, sizes, seed=args.seed)
    cases = build_generation_cases(subset)
    created = _publish(args.output_path, subset)

    digest = hashlib.sha256(read_bytes(args.output_path)).hexdigest()
    status = "wrote" if created else "already present"
    print(f"{status} {args.output_path}")
    for task in M9B_TASKS:
        print(f"  {task}: {len(subset[task]):,} records")
    print(f"  cases: {len(cases):,}  seed: {args.seed}")
    print(f"  dataset_snapshot_sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
