"""Plan local comparisons by default; --run explicitly enables paid provider calls.

Reuses the live service composition root. Each run uses a fresh directory, writes
private trials for manual review, and never calls an automatic quality judge.
"""

from __future__ import annotations

import argparse
import json
import ntpath
import re
import sys
from dataclasses import asdict
from pathlib import Path

from zhrag.eval.agent_comparison import COMPARISON_CONTRACT, run_method
from zhrag.eval.agent_tasks import METHODS, task_summary, validate_tasks
from zhrag.io_utils import append_jsonl, read_jsonl, write_json

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=ROOT / "indexes/agent_eval/v1/tasks.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--allow-drafts", action="store_true", help="development only; no quality claims"
    )
    parser.add_argument(
        "--run", action="store_true", help="allow paid embedding, rerank and chat calls"
    )
    parser.add_argument("--run-id", help="fresh local run slug, required with --run")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except (OSError, RuntimeError, ValueError, SystemExit):
        print("Comparison failed; check local tasks, review status, run ID and service artifacts.")
        return 1


def _execute(args: argparse.Namespace) -> int:
    # Sibling script import supports direct `python scripts/compare_agent.py`.
    # Neither local-path validation nor planning imports providers or reads .env.
    from agent_tasks import _local_path  # noqa: PLC0415

    if not 1 <= args.limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("duplicate comparison method")
    if args.allow_drafts and args.split != "dev":
        raise ValueError("drafts are only allowed for development")
    tasks = validate_tasks(
        read_jsonl(_local_path(args.tasks)),
        require_reviewed=not args.allow_drafts,
    )
    selected = tuple(task for task in tasks if task.split == args.split)[: args.limit]
    if not selected:
        raise ValueError("no tasks selected")
    manifest: dict[str, object] = {
        "contract": COMPARISON_CONTRACT,
        "task_set": task_summary(tasks),
        "selection": task_summary(selected),
        "split": args.split,
        "methods": args.methods,
        "trial_count": len(selected) * len(args.methods),
        "complete": False,
        "human_review_required": True,
        "provider_calls_enabled": args.run,
        "order": "task-order-with-cyclic-method-rotation-v1",
    }
    if not args.run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if not isinstance(args.run_id, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", args.run_id):
        raise ValueError("a canonical run-id is required")
    if ntpath.isreserved(args.run_id):
        raise ValueError("run-id is reserved on Windows")
    target = _local_path(ROOT / "indexes/agent_eval/runs" / args.run_id)
    if target.exists():
        raise ValueError("run directory exists; use a fresh run-id")

    import serve  # noqa: PLC0415 - explicitly paid composition boundary

    # All comparison methods share the exact same model, retrieval and context caps.
    service_args = serve._parse_args(
        [
            "--enable-agent",
            "--env",
            str(args.env),
            "--generation-reasoning-effort",
            "low",
            "--generation-max-tokens",
            "4096",
        ]
    )
    state, index = serve._load_published_artifacts(service_args.artifacts)
    store = serve.MilvusStore(
        serve.MilvusConfig(
            uri=service_args.uri,
            collection_name=service_args.alias,
            dense_dimensions=serve.DENSE_WIDTH,
        )
    )
    try:
        retriever, _info = serve._build_retriever(
            service_args, state=state, index=index, store=store
        )
        agent = serve._build_agent(
            service_args,
            retriever,
            index_identity=f"{state.collection_name}:{state.sparse_fingerprint}",
        )
        assert agent is not None
        manifest["agent_profile"] = agent.profile_fingerprint
        # mkdir is exclusive: two concurrent invocations cannot publish into one run.
        target.mkdir(parents=True, exist_ok=False)
        write_json(target / "manifest.json", manifest)
        for ordinal, task in enumerate(selected):
            rotation = ordinal % len(args.methods)
            methods = args.methods[rotation:] + args.methods[:rotation]
            for method in methods:
                trial = run_method(agent, task.question, method)
                append_jsonl(
                    target / "trials.jsonl",
                    [
                        {
                            "task": asdict(task),
                            "result": trial,
                            "review": {
                                "reviewed": False,
                                "reviewer": "",
                                "task_success": None,
                                "supported_claims": None,
                                "total_claims": None,
                                "appropriate_clarification_or_refusal": None,
                                "notes": "",
                            },
                        }
                    ],
                )
                print(f"completed trial {ordinal + 1}/{len(selected)}: {method}")
        manifest["complete"] = True
        write_json(target / "manifest.json", manifest)
    finally:
        store.close()
    print("Trials saved locally. Execution status is not task accuracy; manual review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
