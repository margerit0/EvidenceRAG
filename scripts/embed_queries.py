"""Fill the query-embedding cache for whichever CRUD-RAG tasks you need.

    uv run python scripts/embed_queries.py --dry-run     # what is missing, and what it costs
    uv run python scripts/embed_queries.py               # all three QA tasks

Exists so that ``scripts/compare_dense_bm25.py`` can keep its "no network calls"
guarantee. That guarantee is worth a separate script: an analysis script that
silently re-embeds on a cache miss can spend money and, worse, drift from the
vectors an earlier run reported, which is exactly the failure this project's
caching exists to prevent.

The document side is already covered -- ``probe_mrl_quality.py`` embedded all
5,681 documents, and every task's gold documents come from that same pool. Only
queries are ever missing here, because the MRL ablation held itself to
``questanswer_1doc`` and so never asked for the other 1,594.

Queries carry the instruct prefix and documents do not; see
:data:`zhrag.providers.embedding.QUERY_PROMPT`. The two therefore live in
separate caches, and the prefix is recorded in each cache's sidecar so a later
run under a different instruction cannot silently reuse these vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhrag.eval.crud import QA_TASKS
from zhrag.io_utils import read_jsonl
from zhrag.providers.embedding import (
    QUERY_PROMPT,
    EmbeddingClient,
    EmbeddingConfig,
    load_env,
    load_or_embed,
)
from zhrag.tokens import estimate_tokens

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
QUERY_CACHE = EXPANDED / "emb_cache_queries_4096.jsonl"

#: Provider list price per million input tokens, in CNY, as of 2026-08.
CNY_PER_MTOK = 0.28


def _questions(tasks: set[str]) -> dict[str, str]:
    return {
        r["query_id"]: r["question"]
        for r in read_jsonl(EXPANDED / "qrels.jsonl")
        if r["task"] in tasks
    }


def _already_cached() -> set[str]:
    if not QUERY_CACHE.exists():
        return set()
    return {r["doc_id"] for r in read_jsonl(QUERY_CACHE)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(QA_TASKS), choices=list(QA_TASKS))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what is missing, and the model that would embed it, without calling out",
    )
    args = parser.parse_args()

    if not (EXPANDED / "qrels.jsonl").exists():
        print(f"! {EXPANDED} not built -- run scripts/build_eval_corpus.py first")
        return 1

    questions = _questions(set(args.tasks))
    # Hoisted out of the comprehension deliberately: the query cache is 220 MB of
    # decimal floats, so calling _already_cached() per query would re-parse it
    # 2,394 times and turn a --dry-run into a multi-minute wait.
    cached = _already_cached()
    missing = {qid: text for qid, text in questions.items() if qid not in cached}
    # Per-bucket rather than one characters-per-token constant: QUERY_PROMPT is
    # 88 English characters against a question of ~50 Chinese ones, so the
    # payload is over half Latin and the CJK ratio alone would overstate the
    # bill by roughly a third.
    tokens = sum(estimate_tokens(QUERY_PROMPT + text) for text in missing.values())

    print(f"tasks     {', '.join(sorted(args.tasks))}")
    print(f"queries   {len(questions):,} requested, {len(missing):,} not yet cached")
    print(f"estimate  ~{tokens:,} tokens ~= CNY {tokens / 1e6 * CNY_PER_MTOK:.3f}")
    if QUERY_CACHE.exists():
        print(f"cache     {QUERY_CACHE.stat().st_size / 1e6:.0f} MB")

    config = EmbeddingConfig.from_env(load_env(ROOT / ".env"))
    print(f"model     {config.model}\n")

    if args.dry_run:
        print("--dry-run: stopping before any API call.")
        return 0

    # Called even when nothing is missing. load_or_embed's first act is the
    # sidecar check, which is what catches a cache written under a different
    # model or prompt -- returning early on an empty `missing` would skip
    # exactly the run that most needs it, since "nothing to embed" is what a
    # silently-mismatched cache looks like from here.
    load_or_embed(
        QUERY_CACHE,
        questions,
        EmbeddingClient(config=config),
        model=config.model,
        prompt=QUERY_PROMPT,
        batch=args.batch,
    )
    if not missing:
        print("\nnothing to embed -- the cache already covered every requested query,")
        print("and the sidecar agrees it was written by this model under this prompt.")
        return 0
    print("\ndone. scripts/compare_dense_bm25.py can now run offline over these tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
