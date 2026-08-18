"""Build the expanded CRUD-RAG evaluation corpus and its qrels.

    uv run python scripts/build_eval_corpus.py

Reads ``crud-rag-subset/raw/split_merged.json`` and writes, into
``crud-rag-subset/eval-expanded/``:

    corpus.jsonl    every distinct news body across all six CRUD tasks
    qrels.jsonl     questions with gold document ids, for all three QA tasks
    manifest.json   counts and a checksum of the source file

Nothing is downloaded. The raw file already holds all 7,661 records; the shipped
500-document subset simply discards most of them, and that discarding is what
makes the benchmark saturate. See :mod:`zhrag.eval.crud` for the measurements.

The output directory is gitignored: it is derived data, reproducible from the
raw file, and the news text carries unclear redistribution rights.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from zhrag.eval.crud import QA_TASKS, build_queries, harvest_documents
from zhrag.io_utils import read_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "crud-rag-subset" / "raw" / "split_merged.json"
OUT = ROOT / "crud-rag-subset" / "eval-expanded"


def main() -> int:
    if not RAW.exists():
        print(f"! {RAW} not found -- run crud-rag-subset/build_subset.ps1 first")
        return 1

    raw = read_json(RAW)
    pool = harvest_documents(raw)
    queries = build_queries(raw, pool=pool)

    gold = {doc_id for q in queries for doc_id in q.gold_doc_ids}
    by_task = Counter(q.task for q in queries)
    by_arity = Counter(len(q.gold_doc_ids) for q in queries)

    n_docs = write_jsonl(
        OUT / "corpus.jsonl",
        ({"doc_id": doc_id, "text": text} for doc_id, text in sorted(pool.items())),
    )
    n_queries = write_jsonl(
        OUT / "qrels.jsonl",
        (
            {
                "query_id": q.query_id,
                "question": q.question,
                "answer": q.answer,
                "gold_doc_ids": list(q.gold_doc_ids),
                "task": q.task,
            }
            for q in queries
        ),
    )

    write_json(
        OUT / "manifest.json",
        {
            "source_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
            "records_per_task": {task: len(raw.get(task, [])) for task in sorted(raw)},
            "documents": n_docs,
            "gold_documents": len(gold),
            "distractors": n_docs - len(gold),
            "queries": n_queries,
            "queries_per_task": dict(sorted(by_task.items())),
            "queries_per_gold_arity": {str(k): v for k, v in sorted(by_arity.items())},
        },
    )

    print(
        f"corpus.jsonl   {n_docs:>6,} documents "
        f"({len(gold):,} gold, {n_docs - len(gold):,} distractors)"
    )
    print(f"qrels.jsonl    {n_queries:>6,} queries")
    for task in QA_TASKS:
        print(f"  {task:<20} {by_task.get(task, 0):>5,}")
    print(f"gold arity     {dict(sorted(by_arity.items()))}")
    print(f"written to     {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
