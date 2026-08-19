"""Align dense-4096 and char-bigram BM25 query by query, and price the fusion.

    uv run python scripts/compare_dense_bm25.py
    uv run python scripts/compare_dense_bm25.py --resamples 100000

README currently reports dense-4096 at R@1 78.0% against BM25's 75.9% and
flags the 2.1pp gap as a bare point estimate. This script closes that: it scores
both arms on the *same* 800 single-evidence queries over the *same* 5,681
documents, keeps the per-query vectors aligned, and reports the paired tests.

The gap is not the interesting part. Two arms two points apart can be two arms
that rank almost identically, or two arms that fail on disjoint sets of queries
and happen to fail equally often. Those cases have the same headline number and
opposite consequences for M4: fusion is worthless in the first and maximal in
the second. Section 2's contingency table is what separates them, and section 4
settles it outright by *running* RRF over the two rankings rather than reasoning
about whether it would help.

**No network calls.** Both arms are recomputed from artefacts already on disk:
the BM25 index is built from ``corpus.jsonl`` in seconds, and the dense arm
reads the 4096-d embedding caches written by ``probe_mrl_quality.py``. A missing
cache entry is a hard error rather than a silent re-embed, so this script cannot
quietly spend money or drift from the vectors the MRL ablation used.

Held to ``questanswer_1doc``: Recall@1 on a 3-gold query cannot exceed 1/3 (see
:func:`zhrag.eval.metrics.recall_at_k`), and a contingency table over a mixture
of arities would count "partial credit changed" as a win.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import Query, sample_corpus
from zhrag.eval.metrics import (
    WinLossTie,
    bootstrap_ci,
    bootstrap_p_floor,
    holm_bonferroni,
    mcnemar_exact,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_test,
    recall_at_k,
    win_loss_tie,
)
from zhrag.io_utils import read_jsonl
from zhrag.lexical import BM25, char_ngram
from zhrag.retrieval import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
DOC_CACHE = EXPANDED / "emb_cache_4096.jsonl"
QUERY_CACHE = EXPANDED / "emb_cache_queries_4096.jsonl"

WIDTH = 4096
#: How deep each arm retrieves. 100 rather than 10 because fusion needs somewhere
#: to find the documents the other arm ranked 1st -- capping both runs at 10
#: would measure a shallower system and hide exactly the recoveries section 3 is
#: looking for. Metrics themselves never look past rank 10.
DEPTH = 100

BM25_LABEL = "BM25 char-bigram"
DENSE_LABEL = "dense-4096"


def _load_queries(limit: int | None) -> list[Query]:
    queries = [
        Query(
            query_id=r["query_id"],
            question=r["question"],
            answer=r["answer"],
            gold_doc_ids=tuple(r["gold_doc_ids"]),
            task=r["task"],
        )
        for r in read_jsonl(EXPANDED / "qrels.jsonl")
        if r["task"] == "questanswer_1doc"
    ]
    return queries[:limit] if limit else queries


def _load_matrix(cache: Path, ids: Sequence[str]) -> NDArray[np.float32]:
    """Fill a preallocated matrix from an append-only embedding cache.

    Streaming into a fixed array rather than building ``{id: [float, ...]}``
    first: 5,681 rows of 4,096 Python floats is roughly 560 MB of boxed objects
    against 93 MB for the float32 matrix that is actually wanted. The cache may
    hold an id more than once (it is appended per batch and resumable), and a
    later row simply overwrites an earlier one -- the same last-write-wins rule
    the rest of the codebase uses.
    """
    position = {doc_id: i for i, doc_id in enumerate(ids)}
    matrix = np.zeros((len(ids), WIDTH), dtype=np.float32)
    seen: set[str] = set()
    for row in read_jsonl(cache):
        i = position.get(row["doc_id"])
        if i is None:
            continue
        vector = row["embedding"]
        if len(vector) != WIDTH:
            raise SystemExit(
                f"! {cache.name}: {row['doc_id']} has width {len(vector)}, not {WIDTH}"
            )
        matrix[i] = vector
        seen.add(row["doc_id"])

    if len(seen) != len(ids):
        raise SystemExit(
            f"! {cache.name} is missing {len(ids) - len(seen):,} of {len(ids):,} vectors.\n"
            f"  This script never calls the embedding API. Populate the cache first:\n"
            f"    uv run python scripts/probe_mrl_quality.py --docs 5681 --queries 800"
        )
    return matrix


def _bm25_runs(corpus: dict[str, str], queries: Sequence[Query]) -> list[list[str]]:
    index = BM25(analyzer=char_ngram(2)).index(list(corpus), list(corpus.values()))
    return [[doc for doc, _ in index.search(q.question, k=DEPTH)] for q in queries]


def _dense_runs(
    query_matrix: NDArray[np.float32], doc_matrix: NDArray[np.float32], doc_ids: Sequence[str]
) -> list[list[str]]:
    """Top-``DEPTH`` ids per query by cosine. Vectors arrive L2-normalised."""
    # kth=DEPTH-1, not DEPTH: argpartition's kth is a 0-based index into the
    # partitioned row, so DEPTH would demand a corpus of at least DEPTH+1
    # documents and raise on one of exactly DEPTH. Same selection either way.
    depth = min(DEPTH, len(doc_ids))
    sims = query_matrix @ doc_matrix.T
    top = np.argpartition(-sims, depth - 1, axis=1)[:, :depth]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, 1), axis=1), 1)
    return [[doc_ids[i] for i in row] for row in ordered]


def _per_query(runs: Sequence[Sequence[str]], queries: Sequence[Query]) -> dict[str, list[float]]:
    pairs = list(zip(runs, queries, strict=True))
    return {
        "R@1": [recall_at_k(r, q.gold_doc_ids, 1) for r, q in pairs],
        "MRR@10": [mrr_at_k(r, q.gold_doc_ids, 10) for r, q in pairs],
        "nDCG@10": [ndcg_at_k(r, q.gold_doc_ids, 10) for r, q in pairs],
    }


def _mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores)


def _first_gold_rank(run: Sequence[str], gold: Sequence[str]) -> int | None:
    goldset = set(gold)
    return next((i for i, doc in enumerate(run, start=1) if doc in goldset), None)


def _fmt_p(p: float, floor: float) -> str:
    """Render a bootstrap p-value, refusing to print its floor as an estimate."""
    return f"<{floor:.1e}" if p <= floor + 1e-12 else f"{p:.4f}"


def _fmt_exact_p(p: float) -> str:
    """Render an exact p-value without rounding a real number down to 0.0000.

    ``.4f`` would print an exact McNemar p of 1.7e-18 as ``0.0000`` -- the same
    "a floor is not an estimate" misreading :func:`_fmt_p` exists to prevent,
    reintroduced three lines from prose boasting that the exact test has no
    floor.
    """
    return f"{p:.2e}" if 0.0 < p < 1e-4 else f"{p:.4f}"


def report_head_to_head(
    bm25: dict[str, list[float]], dense: dict[str, list[float]], resamples: int
) -> None:
    print("\n## 1. Head to head, same queries, same corpus\n")
    print(f"   {'arm':<20} {'R@1 [95% CI]':>24} {'MRR@10':>9} {'nDCG@10':>9}")
    print(f"   {'-' * 20} {'-' * 24} {'-' * 9} {'-' * 9}")
    for label, arm in ((BM25_LABEL, bm25), (DENSE_LABEL, dense)):
        ci = bootstrap_ci(arm["R@1"], resamples=resamples)
        print(
            f"   {label:<20} {ci.mean:>9.1%} [{ci.low:.1%}, {ci.high:.1%}] "
            f"{_mean(arm['MRR@10']):>9.3f} {_mean(arm['nDCG@10']):>9.3f}"
        )

    floor = bootstrap_p_floor(resamples)
    print(f"\n   {'metric':<10} {'delta':>10} {'95% CI on the paired delta':>26} {'p':>10}  test")
    print(f"   {'-' * 10} {'-' * 10} {'-' * 26} {'-' * 10}  {'-' * 20}")
    verdicts: dict[str, tuple[float, float]] = {}
    for metric in ("R@1", "MRR@10", "nDCG@10"):
        diffs = [d - b for b, d in zip(bm25[metric], dense[metric], strict=True)]
        ci = bootstrap_ci(diffs, resamples=resamples)
        if metric == "R@1":
            # Binary outcome: the exact test applies and is preferred.
            p = mcnemar_exact(bm25[metric], dense[metric])
            rendered, test = _fmt_exact_p(p), "McNemar exact"
        else:
            p = paired_bootstrap_test(bm25[metric], dense[metric], resamples=resamples)
            rendered, test = _fmt_p(p, floor), "bootstrap, 1-sided"
        scale, unit = (100, "pp") if metric == "R@1" else (1, "")
        interval = f"[{ci.low * scale:+.2f}, {ci.high * scale:+.2f}]{unit}"
        verdicts[metric] = (ci.low, ci.high)
        print(
            f"   {metric:<10} {f'{ci.mean * scale:+.2f}{unit}':>10} {interval:>26} "
            f"{rendered:>10}  {test}"
        )

    n = len(bm25["R@1"])
    one_sided = mcnemar_exact(bm25["R@1"], dense["R@1"], alternative="greater")
    straddles = [m for m, (low, high) in verdicts.items() if low <= 0 <= high]
    print("\n   Delta is dense minus BM25, so positive favours dense. This is one")
    print("   pre-registered comparison -- it is the open question in the roadmap, not")
    print("   a family -- so no multiplicity correction applies here. Section 4's")
    print("   fusion arms are corrected because they are a family.")
    print("   R@1 uses the exact two-sided McNemar test: the outcome is binary, so the")
    print("   exact null is available and there is no resampling floor. MRR@10 and")
    print("   nDCG@10 are graded, so they fall back to the one-sided paired bootstrap")
    print(f"   ('dense > BM25'), whose floor at {resamples:,} resamples is {floor:.1e}.")

    high = verdicts["R@1"][1] * 100
    if straddles:
        print(f"\n   => the 95% CI straddles zero for: {', '.join(straddles)}.")
        print("      A one-sided McNemar, the most favourable framing dense can be given,")
        print(f"      still returns p = {_fmt_exact_p(one_sided)}. **The gap is a point")
        print("      estimate this query set cannot separate from zero.** Not 'the arms")
        print(f"      are equal' -- the interval also admits a {high:.1f}pp win for dense;")
        print(f"      n={n:,} is what runs out first.")
    else:
        print("\n   => no metric's 95% CI straddles zero; the separation is real at this n.")


def report_contingency(bm25: dict[str, list[float]], dense: dict[str, list[float]]) -> None:
    print("\n\n## 2. Where the two arms disagree (Recall@1)\n")
    counts = win_loss_tie(bm25["R@1"], dense["R@1"])
    n = counts.n
    both, neither = counts.ties_nonzero, counts.ties_zero
    only_bm25, only_dense = counts.losses, counts.wins

    print(f"   {'':<14} {'dense hit':>11} {'dense miss':>11} {'total':>8}")
    print(f"   {'-' * 14} {'-' * 11} {'-' * 11} {'-' * 8}")
    print(f"   {'BM25 hit':<14} {both:>11,} {only_bm25:>11,} {both + only_bm25:>8,}")
    print(f"   {'BM25 miss':<14} {only_dense:>11,} {neither:>11,} {only_dense + neither:>8,}")
    print(f"   {'total':<14} {both + only_dense:>11,} {only_bm25 + neither:>11,} {n:>8,}")

    # Phi: the correlation of the two 0/1 outcome vectors. High phi means the two
    # arms fail on the same queries, which is precisely when fusion cannot help.
    rows = (both + only_bm25, only_dense + neither)
    cols = (both + only_dense, only_bm25 + neither)
    denominator = float(rows[0]) * rows[1] * cols[0] * cols[1]
    phi = (both * neither - only_bm25 * only_dense) / denominator**0.5 if denominator else 0.0

    print(
        f"\n   discordant pairs        {counts.discordant:>7,}   "
        f"({counts.discordant / n:.1%} of queries carry all the evidence)"
    )
    print(f"   only BM25 gets it       {only_bm25:>7,}   dense loses these")
    print(f"   only dense gets it      {only_dense:>7,}   dense wins these")
    print(f"   neither                 {neither:>7,}   unreachable by any fusion of these two")
    print(
        f"   oracle ceiling          {counts.union_rate:>7.1%}   perfect fusion of these two runs"
    )
    print(f"   phi (outcome agreement) {phi:>7.3f}")

    print("\n   Read the ceiling against the better single arm, not against 100%.")
    print("   The headline delta is the *net* of two disagreement columns; the columns")
    print("   themselves are several times larger, which is why the gap looks small and")
    print("   the fusion headroom does not.")


def report_recoverable(
    bm25_runs: Sequence[Sequence[str]],
    dense_runs: Sequence[Sequence[str]],
    queries: Sequence[Query],
) -> None:
    print("\n\n## 3. When one arm misses rank 1, where does the other put the gold?\n")
    print("   The oracle ceiling in section 2 assumes a fusion that always picks the")
    print("   right arm. What a real fusion can reach depends on how far down the")
    print("   losing arm buried the document -- rank 2 is recoverable, rank 90 is not.\n")

    buckets = ((1, 1), (2, 3), (4, 10), (11, DEPTH))
    labels = [str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in buckets] + [f">{DEPTH}"]
    print(f"   {'arm that missed @1':<24} {'n':>6} " + "".join(f"{label:>9}" for label in labels))
    print(f"   {'-' * 24} {'-' * 6} " + " ".join("-" * 8 for _ in labels))

    for missed_label, missed, other in (
        (BM25_LABEL, bm25_runs, dense_runs),
        (DENSE_LABEL, dense_runs, bm25_runs),
    ):
        ranks = [
            _first_gold_rank(other[i], q.gold_doc_ids)
            for i, q in enumerate(queries)
            if _first_gold_rank(missed[i], q.gold_doc_ids) != 1
        ]
        cells = [sum(1 for r in ranks if r is not None and lo <= r <= hi) for lo, hi in buckets]
        cells.append(len(ranks) - sum(cells))
        shares = "".join(f"{c / len(ranks):>9.1%}" if ranks else f"{'-':>9}" for c in cells)
        print(f"   {missed_label:<24} {len(ranks):>6,} {shares}")

    print("\n   Row 1 reads: of the queries BM25 got wrong at rank 1, this is where")
    print("   dense had the gold document. Mass in the 1 and 2-3 columns is fusion")
    print("   headroom; mass in the >100 column is a document neither arm surfaced.")


def _fusion_arms(
    bm25_runs: Sequence[Sequence[str]], dense_runs: Sequence[Sequence[str]]
) -> dict[str, list[list[str]]]:
    """The fusion configurations to price. Deliberately few -- see the caveat printed."""
    pairs = list(zip(bm25_runs, dense_runs, strict=True))
    return {
        "RRF k=60 depth=100": [reciprocal_rank_fusion([b, d], k=60) for b, d in pairs],
        "RRF k=60 depth=10": [reciprocal_rank_fusion([b, d], k=60, depth=10) for b, d in pairs],
        "RRF k=10 depth=100": [reciprocal_rank_fusion([b, d], k=10) for b, d in pairs],
        "RRF k=60 w=.3/.7": [
            reciprocal_rank_fusion([b, d], k=60, weights=[0.3, 0.7]) for b, d in pairs
        ],
    }


def report_fusion(
    bm25_runs: Sequence[Sequence[str]],
    dense_runs: Sequence[Sequence[str]],
    queries: Sequence[Query],
    *,
    bm25: dict[str, list[float]],
    dense: dict[str, list[float]],
    resamples: int,
) -> None:
    print("\n\n## 4. What RRF actually buys, computed rather than assumed\n")
    print("   Both rankings are already on disk, so fusion is a client-side")
    print("   rearrangement: this is M4's headline number, available before any")
    print("   vector store exists. Baseline is dense-4096, the better single arm.\n")

    arms = {
        label: _per_query(runs, queries)
        for label, runs in _fusion_arms(bm25_runs, dense_runs).items()
    }
    raw_p = {label: mcnemar_exact(dense["R@1"], arm["R@1"]) for label, arm in arms.items()}
    adjusted = holm_bonferroni(raw_p)
    base_r1 = _mean(dense["R@1"])

    print(
        f"   {'arm':<22} {'R@1 [95% CI]':>24} {'MRR@10':>8} {'vs dense':>10} "
        f"{'win/loss':>9} {'p(Holm)':>9}"
    )
    print(f"   {'-' * 22} {'-' * 24} {'-' * 8} {'-' * 10} {'-' * 9} {'-' * 9}")
    ci = bootstrap_ci(dense["R@1"], resamples=resamples)
    print(
        f"   {DENSE_LABEL:<22} {ci.mean:>9.1%} [{ci.low:.1%}, {ci.high:.1%}] "
        f"{_mean(dense['MRR@10']):>8.4f} {'baseline':>10} {'':>9} {'':>9}"
    )
    tallies: dict[str, WinLossTie] = {}
    for label, arm in arms.items():
        ci = bootstrap_ci(arm["R@1"], resamples=resamples)
        p, reject = adjusted[label]
        tallies[label] = win_loss_tie(dense["R@1"], arm["R@1"])
        print(
            f"   {label:<22} {ci.mean:>9.1%} [{ci.low:.1%}, {ci.high:.1%}] "
            f"{_mean(arm['MRR@10']):>8.4f} {(ci.mean - base_r1) * 100:>+8.2f}pp "
            f"{f'{tallies[label].wins}/{tallies[label].losses}':>9} "
            f"{_fmt_exact_p(p):>8}{'*' if reject else ' '}"
        )

    n = len(dense["R@1"])
    print(f"\n   * = significant at family-wise alpha=0.05 after Holm over {len(arms)} arms.")

    # The roadmap's actual question is not "does fusion beat dense" but "does
    # anything beat the 40-line BM25 baseline". Answer it directly rather than
    # leaving the reader to chain two comparisons that were never chained.
    best = max(arms, key=lambda label: _mean(arms[label]["R@1"]))
    against_bm25 = win_loss_tie(bm25["R@1"], arms[best]["R@1"])
    p_bm25 = mcnemar_exact(bm25["R@1"], arms[best]["R@1"])
    print(f"\n   Against the BM25 baseline rather than dense -- '{best}':")
    print(
        f"     R@1 {_mean(bm25['R@1']):.1%} -> {_mean(arms[best]['R@1']):.1%} "
        f"({(_mean(arms[best]['R@1']) - _mean(bm25['R@1'])) * 100:+.2f}pp), "
        f"win {against_bm25.wins} / loss {against_bm25.losses}, "
        f"McNemar p = {_fmt_exact_p(p_bm25)}"
    )
    print("     This is the comparison the roadmap actually asked for.")

    _report_churn(arms, adjusted, tallies, base_r1)

    print(f"\n   Two caveats, both load-bearing. These {len(arms)} configurations were scored on")
    print(f"   the same {n:,} queries the winner is reported on, so the best row is an")
    print("   upper bound rather than a generalisation estimate -- a held-out split is")
    print("   owed before this number goes in a resume. And the p-values use the exact")
    print(f"   McNemar test, so unlike a bootstrap at {resamples:,} resamples they have no")
    print("   resolution floor to hide behind.")

    _report_depth(arms, n)


def _report_churn(
    arms: dict[str, dict[str, list[float]]],
    adjusted: dict[str, tuple[float, bool]],
    tallies: dict[str, WinLossTie],
    base_r1: float,
) -> None:
    """Explain an inverted significance ordering, but only when there is one.

    Derived from the table rather than asserted: on a smaller ``--queries`` run
    the ordering does not hold, and a script whose whole premise is "computed,
    not assumed" must not print a conclusion its own numbers contradict.
    """
    significant = [label for label in arms if adjusted[label][1]]
    biggest = max(arms, key=lambda label: _mean(arms[label]["R@1"]) - base_r1)
    if len(significant) != 1 or significant[0] == biggest:
        return
    quiet, loud = tallies[significant[0]], tallies[biggest]
    loud_net, quiet_net = loud.wins - loud.losses, quiet.wins - quiet.losses
    print("\n   The win/loss column explains an ordering that otherwise looks broken.")
    print(f"   '{significant[0]}' posts a smaller delta than '{biggest}' and is")
    print("   nonetheless the only arm to survive correction, because McNemar sees only")
    print("   the queries the two systems disagree on. Rearranging")
    print(f"   {loud.discordant} of them to net {loud_net:+d} is a noisier claim than rearranging")
    print(
        f"   {quiet.discordant} to net {quiet_net:+d}, and the delta column cannot separate them."
    )
    print("   This is the churn effect the roadmap's power analysis predicted, seen")
    print("   here on real runs rather than in a simulation.")


def _report_depth(arms: dict[str, dict[str, list[float]]], n: int) -> None:
    """The candidate-depth finding, gated on it actually holding this run."""
    deep, shallow = arms["RRF k=60 depth=100"], arms["RRF k=60 depth=10"]
    if deep["R@1"] != shallow["R@1"]:
        return
    moved = sum(1 for a, b in zip(deep["MRR@10"], shallow["MRR@10"], strict=True) if a != b)
    print(f"\n   Note for M4's design: retrieving {DEPTH} candidates per arm instead of 10")
    print(f"   changes rank 1 on 0 of the {n:,} queries -- identical R@1 vectors, not")
    print("   merely equal means. For a document at rank 11+ to reach the fused top")
    print("   spot it must outscore everything both arms put in their top 10, which")
    print("   at k=60 never happens here.")
    if moved:
        print("   Deep candidates are still a quality knob *below* rank 1 -- they reorder")
        print(f"   {moved} of the {n:,} queries somewhere inside the top 10, which is where the")
        print("   MRR@10 column's last digit comes from -- so read this as a rank-1 result")
        print("   and not as 'depth is free'.")
    else:
        print("   Nor does anything move inside the top 10 on this run: the MRR@10")
        print("   vectors are identical too, so at this query count depth buys nothing")
        print("   at any rank the metrics look at.")
    print("   Either way this is a fusion-depth finding, not a rerank-window one: a")
    print("   reranker scores every candidate it is handed, so its window is bounded by")
    print("   recall, not by what RRF would have promoted.")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=0, help="0 = all single-evidence queries")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    if not (EXPANDED / "corpus.jsonl").exists():
        print(f"! {EXPANDED} not built -- run scripts/build_eval_corpus.py first")
        return 1
    # Checked before anything touches the files: the crafted "populate the cache"
    # message below is only reachable if the file exists, so an absent cache
    # would otherwise surface as a raw FileNotFoundError from the size banner --
    # whose OS-supplied text is itself mojibake under cp936.
    for cache in (DOC_CACHE, QUERY_CACHE):
        if not cache.exists():
            print(
                f"! {cache} is absent. This script never calls the embedding API.\n"
                f"  Populate the caches first (~15 min, one full embedding pass):\n"
                f"    uv run python scripts/probe_mrl_quality.py --docs 5681 --queries 800"
            )
            return 1

    pool = {r["doc_id"]: r["text"] for r in read_jsonl(EXPANDED / "corpus.jsonl")}
    queries = _load_queries(args.queries or None)
    corpus = sample_corpus(pool, queries)
    doc_ids = list(corpus)
    print(f"corpus {len(corpus):,} documents | {len(queries):,} questanswer_1doc queries")
    print(f"retrieval depth {DEPTH} per arm | {args.resamples:,} bootstrap resamples")

    print("\nbuilding BM25 (char bigram) ...", flush=True)
    bm25_runs = _bm25_runs(corpus, queries)

    print(f"loading {DOC_CACHE.name} ({DOC_CACHE.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    doc_matrix = _load_matrix(DOC_CACHE, doc_ids)
    print(f"loading {QUERY_CACHE.name} ...", flush=True)
    query_matrix = _load_matrix(QUERY_CACHE, [q.query_id for q in queries])
    dense_runs = _dense_runs(query_matrix, doc_matrix, doc_ids)

    bm25, dense = _per_query(bm25_runs, queries), _per_query(dense_runs, queries)
    report_head_to_head(bm25, dense, args.resamples)
    report_contingency(bm25, dense)
    report_recoverable(bm25_runs, dense_runs, queries)
    report_fusion(bm25_runs, dense_runs, queries, bm25=bm25, dense=dense, resamples=args.resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
