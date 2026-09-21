"""Assemble grounded task drafts into an unreviewed task set; fully offline.

Reads draft JSON arrays, verifies quotes against the local corpus snapshot and the
published index state, then writes tasks.jsonl plus an evidence sidecar for human
review. Never calls providers, never reads .env, never marks tasks reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from zhrag.eval.agent_task_drafts import assemble_tasks, verify_draft
from zhrag.eval.agent_tasks import task_summary, validate_tasks
from zhrag.io_utils import read_json, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = ROOT / "indexes" / "agent_eval"


def _split_plan(items: list[str]) -> dict[str, str]:
    plan: dict[str, str] = {}
    for item in items:
        group, _, split = item.partition("=")
        if split not in {"dev", "test"} or not group:
            raise ValueError(f"split plan entries look like group=dev|test, got {item!r}")
        plan[group] = split
    return plan


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, default=TASKS_ROOT / "v2" / "drafts")
    parser.add_argument("--output", type=Path, default=TASKS_ROOT / "v2" / "tasks.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "tidb-rag-curated")
    parser.add_argument("--index-state", type=Path, default=ROOT / "indexes/tidb/state.json")
    parser.add_argument("--snapshot", required=True, help="corpus snapshot label recorded per task")
    parser.add_argument("--split", action="append", default=[], help="source_group=dev|test")
    parser.add_argument("--write", action="store_true", help="write outputs; default only verifies")
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        if not output.is_relative_to(TASKS_ROOT):
            raise ValueError("tasks must stay under indexes/agent_eval")
        manifest = read_jsonl(args.corpus / "corpus_manifest.jsonl")
        local_paths = {f"pingcap/docs-cn:{row['path']}": str(row["local_path"]) for row in manifest}
        indexed = frozenset(read_json(args.index_state)["documents"])
        drafts: list[dict[str, object]] = []
        problems = []
        for path in sorted(args.drafts.glob("*.json")):
            for draft in read_json(path):
                drafts.append(draft)
                problems.extend(
                    verify_draft(
                        draft,
                        corpus_root=args.corpus,
                        local_paths=local_paths,
                        indexed_sources=indexed,
                    )
                )
        for problem in problems:
            print(f"! {problem.task_id}: {problem.message}")
        if problems:
            print(f"{len(problems)} problems in {len(drafts)} drafts; nothing written.")
            return 1
        plan = _split_plan(args.split)
        tasks, evidence = assemble_tasks(drafts, split_plan=plan, snapshot=args.snapshot)
        summary = task_summary(validate_tasks(tasks))
        summary["quotes"] = sum(len(row["evidence_quotes"]) for row in evidence)  # type: ignore[arg-type]
        summary["groups_by_split"] = dict(Counter(plan.values()))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.write:
            if output.exists():
                raise ValueError("task file exists; refusing to overwrite review work")
            write_jsonl(output, tasks)
            write_jsonl(output.with_name("draft_evidence.jsonl"), evidence)
            print(f"wrote {len(tasks)} unreviewed tasks to {output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Draft assembly failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
