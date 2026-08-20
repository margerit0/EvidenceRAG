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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import QA_TASKS, Query, sample_corpus
from zhrag.eval.metrics import (
    BootstrapCI,
    WinLossTie,
    all_gold_at_k,
    bootstrap_ci,
    bootstrap_p_floor,
    hit_at_k,
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


def _load_queries(tasks: Sequence[str], limit: int | None = None) -> list[Query]:
    queries = [
        Query(
            query_id=r["query_id"],
            question=r["question"],
            answer=r["answer"],
            gold_doc_ids=tuple(r["gold_doc_ids"]),
            task=r["task"],
        )
        for r in read_jsonl(EXPANDED / "qrels.jsonl")
        if r["task"] in set(tasks)
    ]
    return queries[:limit] if limit else queries


def _load_matrix(
    cache: Path, ids: Sequence[str], *, require_all: bool = True
) -> tuple[NDArray[np.float32], list[str]]:
    """Fill a preallocated matrix from an append-only embedding cache.

    Streaming into a fixed array rather than building ``{id: [float, ...]}``
    first: 5,681 rows of 4,096 Python floats is roughly 560 MB of boxed objects
    against 93 MB for the float32 matrix that is actually wanted. The cache may
    hold an id more than once (it is appended per batch and resumable), and a
    later row simply overwrites an earlier one -- the same last-write-wins rule
    the rest of the codebase uses.

    Returns the matrix and the ids the cache did not cover. ``require_all``
    defaults to True because an uncovered row stays all-zero, which cosines to 0
    against every document and so ranks last silently instead of announcing
    itself. The one caller that passes False is deciding *whether* an optional
    section can run at all, and drops the affected rows before scoring anything.
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

    missing = [doc_id for doc_id in ids if doc_id not in seen]
    if missing and require_all:
        raise SystemExit(
            f"! {cache.name} is missing {len(missing):,} of {len(ids):,} vectors.\n"
            f"  This script never calls the embedding API. Populate the cache first:\n"
            f"    uv run python scripts/probe_mrl_quality.py --docs 5681 --queries 800\n"
            f"    uv run python scripts/embed_queries.py     # multi-evidence queries"
        )
    return matrix, missing


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
        # Binary at any arity, unlike R@1. hit@1 asks "is *a* gold document
        # first"; ALL-gold@10 asks "did we retrieve everything the question
        # needs". Both are 0/1, so both admit the exact McNemar test.
        "hit@1": [hit_at_k(r, q.gold_doc_ids, 1) for r, q in pairs],
        "ALL@10": [all_gold_at_k(r, q.gold_doc_ids, 10) for r, q in pairs],
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
) -> str:
    """Price every fusion configuration; returns the label of the best by R@1.

    The winner is returned rather than recomputed downstream so that section 5
    audits *this* choice. Re-deriving it there from a different query set would
    let the two sections quietly disagree about which arm they are discussing.
    """
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
    return best


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


#: The two 0/1 metrics. Both are reported in every arity block; the ordering
#: function below only decides which one the prose leads with.
BINARY_METRICS = ("ALL@10", "hit@1")

#: Above this, a metric has stopped separating systems and leading with it would
#: report a ceiling rather than a difference.
SATURATION_CEILING = 0.95


def _binary_metrics(baseline: Mapping[str, Sequence[float]]) -> tuple[str, str, str]:
    """Order the two binary metrics for this stratum, unsaturated one first.

    Returns ``(primary, secondary, why)``. Both are always printed -- this only
    chooses which one the block leads with.

    ALL-gold@10 is the honest success indicator for a multi-evidence question,
    since answering it needs every passage, but on single-gold queries it
    collapses to hit@10, which BM25 already scores 99.5%: a column that cannot
    separate two systems. hit@1 is the reverse, discriminative everywhere but
    blind to whether the *rest* of the evidence was found.

    The rule looks only at the baseline arm, which makes the choice
    treatment-blind: it cannot be tuned toward whichever arm happened to win.
    That is worth something, and it is emphatically not the same as
    verdict-neutral. The two metrics answer different questions -- "is one gold
    document first" against "is the whole evidence set in the top ten" -- and on
    this data a treatment wins one while losing the other. Designating a primary
    picks the question; printing both, and correcting over both, is what keeps
    the designation from deciding the answer.
    """
    saturation = _mean(list(baseline["ALL@10"]))
    if saturation < SATURATION_CEILING:
        return (
            "ALL@10",
            "hit@1",
            f"BM25 scores {saturation:.1%} on ALL@10, below the {SATURATION_CEILING:.0%} bar",
        )
    return "hit@1", "ALL@10", f"ALL@10 is saturated at {saturation:.1%} for BM25"


class _ArityRow(NamedTuple):
    """One printed line of the per-arity test table, with its family key."""

    arity: int
    arm: str
    metric: str
    ci_low: str
    ci_high: str
    tally: str
    key: str
    primary: bool


def report_by_arity(
    queries: Sequence[Query],
    arms: Mapping[str, Sequence[Sequence[str]]],
    resamples: int,
    fusion_label: str,
) -> None:
    """Everything above, re-cut by how many documents the question actually needs."""
    print("\n\n## 5. Stratified by gold arity\n")
    print("   Sections 1-4 are held to questanswer_1doc. This one is not, and it is")
    print("   the open question the roadmap flags as most likely to overturn them:")
    print("   dense retrieval is supposed to pull ahead where lexical overlap runs")
    print("   out, and a 3-document question is where that should show.\n")
    print("   Grouped by the **actual** gold count, not by task name. The names do not")
    print("   fix arity -- questanswer_2docs holds 8 single-gold queries and")
    print("   questanswer_3docs holds 13 two-gold and 1 single-gold -- and since R@1")
    print("   is capped at 1/arity, a row mixing them would carry three ceilings.")

    groups: dict[int, list[int]] = {}
    for i, query in enumerate(queries):
        groups.setdefault(len(query.gold_doc_ids), []).append(i)

    scored = {
        arity: {
            label: _per_query([runs[i] for i in idx], [queries[i] for i in idx])
            for label, runs in arms.items()
        }
        for arity, idx in sorted(groups.items())
    }

    print(
        f"\n   {'arity':>5} {'n':>6} {'arm':<22} {'R@1':>7} {'MRR@10':>8} "
        f"{'nDCG@10':>8} {'hit@1':>7} {'ALL@10':>7}"
    )
    print(f"   {'-' * 5} {'-' * 6} {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 7}")
    for arity, per_arm in scored.items():
        for j, (label, metrics) in enumerate(per_arm.items()):
            head = f"   {arity:>5} {len(groups[arity]):>6,}" if j == 0 else "   " + " " * 12
            print(
                f"{head} {label:<22} {_mean(metrics['R@1']):>6.1%} "
                f"{_mean(metrics['MRR@10']):>8.3f} {_mean(metrics['nDCG@10']):>8.3f} "
                f"{_mean(metrics['hit@1']):>6.1%} {_mean(metrics['ALL@10']):>6.1%}"
            )

    print("\n   R@1 is comparable down a column only *within* an arity block: its")
    print("   ceiling is 1/arity, so 3-gold rows top out at 33.3%. hit@1 and ALL@10")
    print("   are 0/1 at any arity and are the columns to read across blocks.")
    extra = len(groups.get(1, [])) - sum(1 for q in queries if q.task == "questanswer_1doc")
    if extra:
        print("\n   The arity-1 block is not the questanswer_1doc row from sections 1-4:")
        print(f"   it is {extra} queries larger, because that many rows filed under the")
        print("   2docs and 3docs tasks carry a single gold document. Small differences")
        print("   against the README's per-task numbers are that regrouping, not drift.")

    print("\n   All four fusion configurations appear in the table so the reader can see")
    print(f"   they cluster, but only '{fusion_label}' -- the one section 4 selected --")
    print("   is carried into the tests below. Adding three near-identical arms would")
    print("   inflate the correction's family size without adding a question.")

    _report_arity_tests(scored, resamples, fusion_label)
    _report_arity_verdict(scored, fusion_label, resamples)


def _report_arity_verdict(
    scored: Mapping[int, Mapping[str, dict[str, list[float]]]],
    fusion_label: str,
    resamples: int,
) -> None:
    """State plainly whether one fusion configuration survives a change of arity."""
    print("\n   Fusion against dense alone, per arity -- the configuration in section 4")
    print("   was chosen on single-evidence queries, so this is where that choice gets")
    print("   audited rather than assumed. Both binary metrics, Holm-corrected over the")
    print("   whole block:\n")

    raw: dict[str, float] = {}
    deltas: dict[str, float] = {}
    cis: dict[str, BootstrapCI] = {}
    tallies: dict[str, WinLossTie] = {}
    for arity, per_arm in scored.items():
        dense, fused = per_arm[DENSE_LABEL], per_arm[fusion_label]
        for metric in BINARY_METRICS:
            key = f"arity {arity} {metric}"
            raw[key] = mcnemar_exact(dense[metric], fused[metric])
            diffs = [t - d for d, t in zip(dense[metric], fused[metric], strict=True)]
            cis[key] = bootstrap_ci(diffs, resamples=resamples)
            deltas[key] = (_mean(fused[metric]) - _mean(dense[metric])) * 100
            tallies[key] = win_loss_tie(dense[metric], fused[metric])
    adjusted = holm_bonferroni(raw)

    regressions: list[str] = []
    for key in raw:
        delta, ci, counts = deltas[key], cis[key], tallies[key]
        p, reject = adjusted[key]
        if delta > 0:
            verdict = "fusion ahead"
        elif delta < 0:
            verdict = "**dense alone ahead**"
        else:
            verdict = "dead level"
        if delta < 0 and reject:
            regressions.append(key)
        interval = f"[{ci.low * 100:+.2f}, {ci.high * 100:+.2f}]pp"
        print(
            f"     {key:<16} {delta:+6.2f}pp {interval:>18}, win/loss "
            f"{f'{counts.wins}/{counts.losses}':>7}, p(Holm) = {_fmt_exact_p(p):>9}"
            f"{'*' if reject else ' '} -- {verdict}"
        )
    print(f"\n   * = significant at family-wise alpha=0.05 after Holm over {len(raw)} tests.")

    if not regressions:
        print("\n   No fusion regression survives the correction: one configuration is")
        print("   defensible across every arity on this data.")
        return
    print(f"\n   Fusion is significantly behind dense alone on: {', '.join(regressions)}.")
    print("   Read the metric, not just the sign. The losses are on ALL-gold@10 -- the")
    print("   whole evidence set -- while hit@1 moves by amounts the same test cannot")
    print("   separate from zero. So the bill section 4 ran up is specific: the")
    print("   lexical arm helps put *a* passage first on questions one passage answers,")
    print("   and dilutes dense's ability to surface *every* passage on questions that")
    print("   need several. A single fusion configuration across all arities is")
    print("   therefore not supported here -- either the weights vary with the question,")
    print("   or the ablation reports per-arity rows and declines to name one winner.")


def _report_arity_tests(
    scored: Mapping[int, Mapping[str, dict[str, list[float]]]],
    resamples: int,
    fusion_label: str,
) -> None:
    """Per-arity verdicts against the BM25 baseline, on both binary metrics."""
    print("\n   Against the BM25 baseline, per arity. Both 0/1 metrics are reported in")
    print("   every block; '>' marks the one the block leads with, chosen by looking")
    print("   only at the baseline's saturation. The graded row underneath is the")
    print("   one-sided paired bootstrap on nDCG@10.\n")

    floor = bootstrap_p_floor(resamples)
    rows: list[_ArityRow] = []
    binary_p: dict[str, float] = {}
    graded_p: dict[str, float] = {}
    why_by_arity: dict[int, str] = {}

    for arity, per_arm in scored.items():
        baseline = per_arm[BM25_LABEL]
        primary, secondary, why = _binary_metrics(baseline)
        why_by_arity[arity] = why
        for arm in (DENSE_LABEL, fusion_label):
            treatment = per_arm[arm]
            for metric in (primary, secondary):
                key = f"a{arity} {arm} {metric}"
                binary_p[key] = mcnemar_exact(baseline[metric], treatment[metric])
                counts = win_loss_tie(baseline[metric], treatment[metric])
                diffs = [t - b for b, t in zip(baseline[metric], treatment[metric], strict=True)]
                ci = bootstrap_ci(diffs, resamples=resamples)
                rows.append(
                    _ArityRow(
                        arity=arity,
                        arm=arm,
                        metric=metric,
                        ci_low=f"{ci.low * 100:+.2f}",
                        ci_high=f"{ci.high * 100:+.2f}",
                        tally=f"{counts.wins}/{counts.losses}",
                        key=key,
                        primary=metric == primary,
                    )
                )
            key = f"a{arity} {arm} nDCG@10"
            graded_p[key] = paired_bootstrap_test(
                baseline["nDCG@10"], treatment["nDCG@10"], resamples=resamples
            )
            diffs = [t - b for b, t in zip(baseline["nDCG@10"], treatment["nDCG@10"], strict=True)]
            ci = bootstrap_ci(diffs, resamples=resamples)
            rows.append(
                _ArityRow(
                    arity=arity,
                    arm=arm,
                    metric="nDCG@10",
                    ci_low=f"{ci.low:+.3f}",
                    ci_high=f"{ci.high:+.3f}",
                    tally="",
                    key=key,
                    primary=False,
                )
            )

    # Two families, not one. The binary tests are exact and two-sided; the graded
    # ones are one-sided with a Monte-Carlo floor at 1/(resamples+1). Pooling
    # them would let floored values -- which are "at most this small", not "this
    # small" -- set the step-down order for the exact ones.
    binary_adj = holm_bonferroni(binary_p)
    graded_adj = holm_bonferroni(graded_p)

    print(
        f"   {'':>1}{'arity':>5} {'arm':<22} {'metric':>8} {'delta [95% CI]':>24} "
        f"{'win/loss':>9} {'p':>10} {'p(Holm)':>10}"
    )
    print(f"    {'-' * 5} {'-' * 22} {'-' * 8} {'-' * 24} {'-' * 9} {'-' * 10} {'-' * 10}")
    last_arity: int | None = None
    for row in rows:
        if last_arity is not None and row.arity != last_arity:
            print(f"    {'':>5} ({why_by_arity[last_arity]})")
        last_arity = row.arity
        graded = row.metric == "nDCG@10"
        raw_p = graded_p[row.key] if graded else binary_p[row.key]
        adj, reject = (graded_adj if graded else binary_adj)[row.key]
        shown_raw = _fmt_p(raw_p, floor) if graded else _fmt_exact_p(raw_p)
        shown_adj = _fmt_p(adj, floor) if graded else _fmt_exact_p(adj)
        baseline = scored[row.arity][BM25_LABEL][row.metric]
        treatment = scored[row.arity][row.arm][row.metric]
        mean = (_mean(treatment) - _mean(baseline)) * (1 if graded else 100)
        interval = (
            f"{mean:+.3f} [{row.ci_low}, {row.ci_high}]"
            if graded
            else f"{mean:+.2f}pp [{row.ci_low}, {row.ci_high}]"
        )
        print(
            f"   {'>' if row.primary else ' '}{row.arity:>5} {row.arm:<22} "
            f"{row.metric:>8} {interval:>24} {row.tally:>9} {shown_raw:>10} "
            f"{shown_adj:>9}{'*' if reject else ' '}"
        )
    if last_arity is not None:
        print(f"    {'':>5} ({why_by_arity[last_arity]})")

    print(
        f"\n   Holm runs over the {len(binary_p)} binary tests as one family and the "
        f"{len(graded_p)} graded ones"
    )
    print("   as another. The binary family spans both metrics on purpose: every one of")
    print(f"   those {len(binary_p)} is a chance to claim an improvement, and splitting them into")
    print("   a family per column would buy power by redrawing the family after seeing")
    print("   the table. A result that needs that split is not a result.")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=0, help="0 = all single-evidence queries")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--no-arity",
        action="store_true",
        help="skip section 5 even when the multi-evidence query embeddings are cached",
    )
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
    one_doc = _load_queries(["questanswer_1doc"], args.queries or None)
    multi = [q for q in _load_queries(list(QA_TASKS)) if q.task != "questanswer_1doc"]
    corpus = sample_corpus(pool, _load_queries(list(QA_TASKS)))
    doc_ids = list(corpus)

    print(f"corpus {len(corpus):,} documents | {len(one_doc):,} questanswer_1doc queries")
    print(f"retrieval depth {DEPTH} per arm | {args.resamples:,} bootstrap resamples")

    print(f"loading {DOC_CACHE.name} ({DOC_CACHE.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    doc_matrix, _ = _load_matrix(DOC_CACHE, doc_ids)

    # Section 5 needs the multi-evidence queries embedded, which the MRL ablation
    # never asked for. Degrade to sections 1-4 rather than failing: the
    # pre-registered comparison must stay runnable on a machine that has only
    # ever run probe_mrl_quality.py. The coverage question is answered by the
    # one pass that has to read this 220 MB file anyway -- asking it separately
    # first would parse every vector twice to learn only which ids exist.
    print(f"loading {QUERY_CACHE.name} ...", flush=True)
    query_matrix, absent = _load_matrix(
        QUERY_CACHE, [q.query_id for q in one_doc + multi], require_all=False
    )
    one_doc_absent = [qid for qid in absent if qid in {q.query_id for q in one_doc}]
    if one_doc_absent:
        raise SystemExit(
            f"! {QUERY_CACHE.name} is missing {len(one_doc_absent):,} of the "
            f"{len(one_doc):,} single-evidence queries\n"
            f"  sections 1-4 depend on. Populate the cache first:\n"
            f"    uv run python scripts/probe_mrl_quality.py --docs 5681 --queries 800"
        )

    stratified = bool(multi) and not absent and not args.no_arity and not args.queries
    if stratified:
        print(f"section 5 additionally covers {len(multi):,} multi-evidence queries")
    elif args.no_arity:
        print("section 5 skipped: --no-arity")
    elif args.queries:
        print(
            f"section 5 skipped: --queries {args.queries} truncates the single-evidence\n"
            f"  arm, and an arity-1 block scored on {args.queries} queries next to arity-2 and\n"
            f"  arity-3 blocks scored on all of theirs would put three sample sizes in one\n"
            f"  column. Drop --queries to include it."
        )
    elif absent:
        print(
            f"section 5 skipped: {len(absent):,} multi-evidence queries are not in "
            f"{QUERY_CACHE.name}\n  populate them with: uv run python scripts/embed_queries.py"
        )

    queries = one_doc + (multi if stratified else [])
    query_matrix = query_matrix[: len(queries)]

    print("\nbuilding BM25 (char bigram) ...", flush=True)
    bm25_runs = _bm25_runs(corpus, queries)
    dense_runs = _dense_runs(query_matrix, doc_matrix, doc_ids)

    # Sections 1-4 are the pre-registered single-evidence comparison and stay
    # held to it; slicing here rather than re-running keeps both halves on
    # byte-identical runs.
    n1 = len(one_doc)
    bm25 = _per_query(bm25_runs[:n1], one_doc)
    dense = _per_query(dense_runs[:n1], one_doc)
    report_head_to_head(bm25, dense, args.resamples)
    report_contingency(bm25, dense)
    report_recoverable(bm25_runs[:n1], dense_runs[:n1], one_doc)
    best = report_fusion(
        bm25_runs[:n1],
        dense_runs[:n1],
        one_doc,
        bm25=bm25,
        dense=dense,
        resamples=args.resamples,
    )

    if stratified:
        report_by_arity(
            queries,
            {
                BM25_LABEL: bm25_runs,
                DENSE_LABEL: dense_runs,
                **_fusion_arms(bm25_runs, dense_runs),
            },
            args.resamples,
            best,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
