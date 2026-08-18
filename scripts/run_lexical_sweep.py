"""Regenerate the retrieval tables quoted in README.md and zhrag.eval.crud.

    uv run python scripts/run_lexical_sweep.py            # full sweep
    uv run python scripts/run_lexical_sweep.py --quick    # saturation curve only

Produces three tables:

1. Corpus-size scaling -- the saturation evidence. Held to ``questanswer_1doc``
   on purpose: this table varies corpus size, so gold arity must stay fixed or
   the trend would confound two variables at once.
2. Analyzer comparison over all 2,394 queries, **stratified by task**.
3. Task difficulty, with the arity ceiling that makes stratification necessary.

Why stratified rather than one pooled R@1: :func:`~zhrag.eval.metrics.recall_at_k`
returns the fraction of gold retrieved, so at k=1 a 3-gold query cannot exceed
1/3. Averaging 1-, 2- and 3-gold queries into a single R@1 produces a number
whose value depends on the task mix rather than on retrieval quality.
``ALL-gold@10`` is a clean 0/1 success indicator at any arity, so that column
*is* pooled across all 2,394 queries.

All 2,394 queries are used because the ablation table needs the statistical
power: under Holm correction across ~40 cells, an effect that fixes some queries
and breaks others is undetectable at n=800 and detectable at n=2,394. See
``scripts/power_analysis.py``.

Everything runs through the shipped, unit-tested :class:`zhrag.lexical.BM25`
and :mod:`zhrag.eval.metrics`, not a throwaway script. That is the point: the
numbers in the README have to come out of the same code a reader can run.

Requires ``scripts/build_eval_corpus.py`` to have been run first. The jieba rows
need the optional extra (``uv pip install jieba``) and are skipped without it.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from zhrag.eval.crud import QA_TASKS, Query, sample_corpus
from zhrag.eval.metrics import all_gold_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from zhrag.io_utils import read_jsonl
from zhrag.lexical import BM25, char_ngram, union
from zhrag.lexical.analyzers import Analyzer

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
#: Distractor counts added on top of the gold floor. The smallest corpus is the
#: gold documents alone -- every query's evidence and nothing else -- which is
#: the most favourable corpus a retriever can be given and therefore the right
#: place to start a saturation curve.
DISTRACTOR_STEPS = (0, 200, 1200, 3200, None)
TOP_K = 20


def _load() -> tuple[dict[str, str], list[Query]]:
    pool = {r["doc_id"]: r["text"] for r in read_jsonl(EXPANDED / "corpus.jsonl")}
    queries = [
        Query(
            query_id=r["query_id"],
            question=r["question"],
            answer=r["answer"],
            gold_doc_ids=tuple(r["gold_doc_ids"]),
            task=r["task"],
        )
        for r in read_jsonl(EXPANDED / "qrels.jsonl")
    ]
    return pool, queries


def _build(corpus: dict[str, str], analyzer: Analyzer) -> tuple[BM25, float]:
    start = time.perf_counter()
    index = BM25(analyzer=analyzer).index(list(corpus), list(corpus.values()))
    return index, time.perf_counter() - start


def _score(index: BM25, queries: Sequence[Query]) -> dict[str, float]:
    start = time.perf_counter()
    runs = [[doc for doc, _ in index.search(q.question, k=TOP_K)] for q in queries]
    query_ms = (time.perf_counter() - start) / len(queries) * 1000

    pairs = list(zip(runs, queries, strict=True))
    n = len(pairs)

    def mean(fn: Callable[[Sequence[str], Sequence[str]], float]) -> float:
        return sum(fn(r, q.gold_doc_ids) for r, q in pairs) / n

    return {
        "R@1": mean(lambda r, g: recall_at_k(r, g, 1)),
        "R@5": mean(lambda r, g: recall_at_k(r, g, 5)),
        "MRR@10": mean(lambda r, g: mrr_at_k(r, g, 10)),
        "nDCG@10": mean(lambda r, g: ndcg_at_k(r, g, 10)),
        "ALL@10": mean(lambda r, g: all_gold_at_k(r, g, 10)),
        "query_ms": query_ms,
    }


def _analyzers() -> list[tuple[str, Analyzer]]:
    out: list[tuple[str, Analyzer]] = [
        ("char unigram", char_ngram(1)),
        ("char bigram", char_ngram(2)),
        ("char trigram", char_ngram(3)),
    ]
    try:
        from zhrag.lexical import jieba_words  # noqa: PLC0415

        out += [
            ("jieba precise", jieba_words()),
            ("jieba cut_for_search", jieba_words(for_search=True)),
            ("jieba + char bigram", union(jieba_words(), char_ngram(2))),
        ]
    except ImportError:
        print("  (jieba not installed -- skipping word-segmentation rows)")
    return out


def table_scaling(pool: dict[str, str], one_doc: Sequence[Query]) -> None:
    print("## 1. Corpus-size scaling (char bigram, questanswer_1doc only)\n")
    gold_floor = len({d for q in one_doc for d in q.gold_doc_ids})
    print("   Arity held fixed at 1 so the only variable is corpus size.")
    print(f"   Gold floor = {gold_floor:,} documents (evidence for all {len(one_doc):,} queries)\n")
    print(f"   {'corpus':>8} | {'R@1':>7} {'R@5':>7} {'MRR@10':>7} {'nDCG@10':>8}")
    print(f"   {'-' * 8} | {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")
    for extra in DISTRACTOR_STEPS:
        size = None if extra is None else gold_floor + extra
        corpus = sample_corpus(pool, one_doc, size=size)
        index, _ = _build(corpus, char_ngram(2))
        m = _score(index, one_doc)
        print(
            f"   {len(corpus):>8,} | {m['R@1']:>6.1%} {m['R@5']:>7.1%} "
            f"{m['MRR@10']:>7.3f} {m['nDCG@10']:>8.3f}"
        )


def table_analyzers(full: dict[str, str], queries: Sequence[Query]) -> None:
    by_task = {t: [q for q in queries if q.task == t] for t in QA_TASKS}
    print(
        f"\n\n## 2. Analyzer comparison ({len(full):,} documents, all {len(queries):,} queries)\n"
    )
    print("   R@1 stratified by task; ALL-gold@10 pooled (valid at any arity).\n")
    header = (
        f"   {'analyzer':<22} {'vocab':>9} | "
        f"{'R@1 1doc':>8} {'R@1 2doc':>8} {'R@1 3doc':>8} | "
        f"{'ALL@10':>7} | {'build':>6} {'query':>7}"
    )
    print(header)
    print(
        f"   {'-' * 22} {'-' * 9} | {'-' * 8} {'-' * 8} {'-' * 8} | {'-' * 7} | {'-' * 6} {'-' * 7}"
    )
    for label, analyzer in _analyzers():
        index, build_s = _build(full, analyzer)
        per_task = {t: _score(index, qs) for t, qs in by_task.items()}
        pooled = _score(index, queries)
        print(
            f"   {label:<22} {index.vocabulary_size:>9,} | "
            + " ".join(f"{per_task[t]['R@1']:>7.1%}" for t in QA_TASKS)
            + f" | {pooled['ALL@10']:>6.1%} | {build_s:>5.1f}s {pooled['query_ms']:>6.1f}ms"
        )


def table_difficulty(full: dict[str, str], queries: Sequence[Query]) -> None:
    print("\n\n## 3. Task difficulty and the arity ceiling (char bigram)\n")
    index, _ = _build(full, char_ngram(2))
    print(
        f"   {'task':<20} {'n':>6} {'gold':>5} | {'R@1':>7} {'ceiling':>8} {'% of ceil':>10} | "
        f"{'MRR@10':>7} {'ALL@10':>7}"
    )
    print(
        f"   {'-' * 20} {'-' * 6} {'-' * 5} | {'-' * 7} {'-' * 8} {'-' * 10} | {'-' * 7} {'-' * 7}"
    )
    for task in QA_TASKS:
        subset = [q for q in queries if q.task == task]
        m = _score(index, subset)
        arity = sum(len(q.gold_doc_ids) for q in subset) / len(subset)
        ceiling = 1 / arity
        print(
            f"   {task:<20} {len(subset):>6,} {arity:>5.2f} | {m['R@1']:>6.1%} "
            f"{ceiling:>7.1%} {m['R@1'] / ceiling:>9.1%} | "
            f"{m['MRR@10']:>7.3f} {m['ALL@10']:>6.1%}"
        )
    print("\n   R@1 cannot exceed 1/arity, so the raw R@1 column is not comparable")
    print("   across rows. '% of ceil' and ALL-gold@10 are.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="saturation curve only")
    args = parser.parse_args()

    if not (EXPANDED / "corpus.jsonl").exists():
        print(f"! {EXPANDED} not built -- run scripts/build_eval_corpus.py first")
        return 1

    pool, queries = _load()
    one_doc = [q for q in queries if q.task == "questanswer_1doc"]
    print(
        f"pool {len(pool):,} documents | {len(queries):,} queries "
        f"({len(one_doc):,} single-evidence)\n"
    )

    table_scaling(pool, one_doc)
    if args.quick:
        return 0

    full = sample_corpus(pool, queries)
    table_analyzers(full, queries)
    table_difficulty(full, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
