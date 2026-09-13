"""Initialize or validate a local task set; never call providers or read .env."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zhrag.eval.agent_tasks import draft_tasks, task_summary, validate_tasks
from zhrag.io_utils import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = ROOT / "indexes" / "agent_eval"


def _local_path(path: Path) -> Path:
    # Existing junctions/symlinks in the artifact tree must not route writes elsewhere.
    absolute = Path(path).absolute()
    if not absolute.is_relative_to(TASKS_ROOT):
        raise ValueError("tasks must stay under indexes/agent_eval")
    for parent in (absolute, *absolute.parents):
        if parent == ROOT:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError("task path cannot traverse symlinks or junctions")
    resolved = absolute.resolve()
    if not resolved.is_relative_to(TASKS_ROOT.resolve()):
        raise ValueError("tasks must stay under indexes/agent_eval")
    return resolved


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=TASKS_ROOT / "v1" / "tasks.jsonl")
    parser.add_argument(
        "--initialize", action="store_true", help="create unreviewed scenario drafts"
    )
    parser.add_argument(
        "--require-reviewed", action="store_true", help="reject all unreviewed tasks"
    )
    args = parser.parse_args(argv)
    try:
        path = _local_path(args.tasks)
        if args.initialize:
            if args.require_reviewed:
                raise ValueError("draft initialization cannot satisfy reviewed evaluation")
            if path.exists():
                raise ValueError("task file exists; refusing to overwrite review work")
            write_jsonl(path, draft_tasks())
        tasks = validate_tasks(read_jsonl(path), require_reviewed=args.require_reviewed)
        print(json.dumps(task_summary(tasks), ensure_ascii=False, indent=2))
    except (OSError, ValueError):
        print("Task validation failed; check schema, local path, review state and split isolation.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
