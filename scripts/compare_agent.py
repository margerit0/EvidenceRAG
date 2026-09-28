"""Plan local comparisons by default; --run explicitly enables paid provider calls.

Reuses the live service composition root. Each run uses a fresh directory, writes
private trials for manual review, and never calls an automatic quality judge.
"""

from __future__ import annotations

import argparse
import json
import math
import ntpath
import re
import sys
from dataclasses import asdict
from pathlib import Path

from zhrag.agent import AgentSettings
from zhrag.eval.agent_comparison import COMPARISON_CONTRACT, run_method
from zhrag.eval.agent_review import fingerprint, pending_review
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
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-reads", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--generation-timeout", type=float, default=60.0)
    output_limit = parser.add_mutually_exclusive_group()
    output_limit.add_argument("--generation-max-tokens", type=int, default=4096)
    output_limit.add_argument(
        "--generation-no-token-limit",
        dest="generation_max_tokens",
        action="store_const",
        const=None,
        help="omit the request output-token cap; provider defaults and model limits still apply",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=0,
        help="retries after the first chat attempt (0-9); at most 10 total attempts",
    )
    parser.add_argument("--agent-review-answers", action="store_true")
    parser.add_argument(
        "--generation-reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="low",
    )
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except (OSError, RuntimeError, ValueError, SystemExit):
        print("Comparison failed; check local tasks, review status, run ID and service artifacts.")
        return 1


def _execute(  # noqa: PLR0912, PLR0915 - explicit offline/paid boundary
    args: argparse.Namespace,
) -> int:
    # Sibling script import supports direct `python scripts/compare_agent.py`.
    # Neither local-path validation nor planning imports providers or reads .env.
    from agent_tasks import _local_path  # noqa: PLC0415

    if not 1 <= args.limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    budgets = AgentSettings(
        max_steps=args.max_steps,
        max_searches=args.max_searches,
        max_reads=args.max_reads,
        max_seconds=args.max_seconds,
        review_answers=getattr(args, "agent_review_answers", False),
    )
    if not math.isfinite(args.generation_timeout) or not 0 < args.generation_timeout <= 300:
        raise ValueError("invalid generation timeout")
    if args.generation_max_tokens is not None and not 1 <= args.generation_max_tokens <= 8192:
        raise ValueError("invalid generation output cap")
    if not 0 <= args.generation_retries <= 9:
        raise ValueError("generation-retries must be in [0, 9] (at most 10 total attempts)")
    reasoning_effort = getattr(args, "generation_reasoning_effort", "low")
    if reasoning_effort not in {"minimal", "low", "medium", "high"}:
        raise ValueError("invalid generation reasoning effort")
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
        "agent_budgets": asdict(budgets),
        "generation": {
            "max_output_tokens": args.generation_max_tokens,
            "timeout_seconds": min(args.generation_timeout, args.max_seconds),
            "max_retries": args.generation_retries,
            "reasoning_effort": reasoning_effort,
        },
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
    service_options = [
        "--enable-agent",
        "--env",
        str(args.env),
        "--generation-reasoning-effort",
        reasoning_effort,
        "--generation-timeout",
        str(args.generation_timeout),
        "--agent-max-steps",
        str(args.max_steps),
        "--agent-max-searches",
        str(args.max_searches),
        "--agent-max-seconds",
        str(args.max_seconds),
        "--context-passages",
        str(args.max_reads),
        "--agent-generation-retries",
        str(args.generation_retries),
    ]
    if args.generation_max_tokens is None:
        service_options.append("--generation-no-token-limit")
    else:
        service_options.extend(["--generation-max-tokens", str(args.generation_max_tokens)])
    if budgets.review_answers:
        service_options.append("--agent-review-answers")
    service_args = serve._parse_args(service_options)
    state, index = serve._load_published_artifacts(service_args.artifacts)
    serve._direct_loopback()
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
        trial_rows: list[dict[str, object]] = []
        method_profiles: dict[str, str] = {}
        for ordinal, task in enumerate(selected):
            rotation = ordinal % len(args.methods)
            methods = args.methods[rotation:] + args.methods[:rotation]
            for method in methods:
                trial = run_method(agent, task.question, method)
                profile = str(trial["profile_fingerprint"])
                if method in method_profiles and method_profiles[method] != profile:
                    raise ValueError("method profile changed during the run")
                method_profiles[method] = profile
                row: dict[str, object] = {
                    "task": asdict(task),
                    "result": trial,
                    "review": pending_review(),
                }
                append_jsonl(
                    target / "trials.jsonl",
                    [row],
                )
                trial_rows.append(row)
                print(f"completed trial {ordinal + 1}/{len(selected)}: {method}")
        manifest["complete"] = True
        manifest["method_profiles"] = method_profiles
        manifest["trials_sha256"] = fingerprint(trial_rows)
        write_json(target / "manifest.json", manifest)
    finally:
        store.close()
    print("Trials saved locally. Execution status is not task accuracy; manual review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
