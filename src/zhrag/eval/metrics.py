"""Information-retrieval metrics, implemented rather than imported.

Two reasons this module is hand-rolled. First, the eval libraries in this space
disagree subtly about multi-gold cases -- whether Recall@k means "fraction of
gold retrieved" or "any gold retrieved" -- and CRUD-RAG's ``questanswer_2docs``
and ``questanswer_3docs`` tasks have 2 and 3 gold documents respectively, so the
distinction decides the headline number. Second, these are the numbers the whole
project is judged on; they need tests, not a transitive dependency.

Metric choice for this benchmark is constrained by a measured fact: on the
5,681-document pool a plain char-bigram BM25 already reaches Recall@5 = 99.2%.
Recall@5 is therefore saturated and cannot separate two systems. Report
**Recall@1, MRR@10 and nDCG@10**; Recall@5 belongs in an appendix if anywhere.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "BootstrapCI",
    "all_gold_at_k",
    "bootstrap_ci",
    "evaluate",
    "hit_at_k",
    "holm_bonferroni",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_test",
    "recall_at_k",
]


def _validate(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold documents appearing in the top ``k``.

    For single-gold queries this collapses to hit-or-miss; for the 2docs/3docs
    tasks it gives partial credit, which is why it differs from
    :func:`all_gold_at_k`.

    Beware when ``k < len(gold)``: the score is then capped at ``k/len(gold)``,
    so Recall@1 tops out at 50% for a 2-gold query and 33.3% for a 3-gold one.
    Putting Recall@1 for 1-, 2- and 3-gold tasks in one column makes a healthy
    system look like it is collapsing. Compare across arities with
    :func:`all_gold_at_k` instead, or hold arity fixed within a row.
    """
    _validate(k)
    goldset = set(gold)
    if not goldset:
        raise ValueError("gold must be non-empty")
    return len(goldset & set(ranked[:k])) / len(goldset)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if *any* gold document is in the top ``k``."""
    _validate(k)
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0


def all_gold_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 only if *every* gold document is in the top ``k``.

    The honest metric for multi-document QA: answering a 3-document question
    correctly requires all three passages, so partial recall overstates success.
    """
    _validate(k)
    goldset = set(gold)
    if not goldset:
        raise ValueError("gold must be non-empty")
    return 1.0 if goldset <= set(ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """Reciprocal rank of the first gold document within the top ``k``, else 0."""
    _validate(k)
    goldset = set(gold)
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in goldset:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k with the standard ``1/log2(rank+1)`` discount."""
    _validate(k)
    goldset = set(gold)
    if not goldset:
        raise ValueError("gold must be non-empty")
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1) if doc in goldset)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(goldset)) + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap_ci(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval over per-query scores.

    Report intervals rather than bare point estimates: at n=500 an *unpaired*
    comparison needs several points of separation before the intervals stop
    overlapping. Note this is the weaker of the two tools here -- for comparing
    ablation arms that share a query set, use :func:`paired_bootstrap_test`,
    which is far more sensitive.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(resamples))
    alpha = (1 - confidence) / 2
    return BootstrapCI(
        mean=sum(scores) / n,
        low=means[int(alpha * resamples)],
        high=means[min(int((1 - alpha) * resamples), resamples - 1)],
        n=n,
    )


def paired_bootstrap_test(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """One-sided paired bootstrap p-value for ``treatment > baseline``.

    Queries are resampled as pairs, which is what makes this *paired*: the two
    systems saw identical queries, so the per-query difference is far less noisy
    than either system's absolute score. This is standard practice in IR and is
    the right test for an ablation table where every arm shares a query set.

    Sensitivity depends on how correlated the arms are, not on the headline gap
    alone. At n=500 a +1pp gain where nothing regresses is detectable; the same
    +1pp net gain is not, once some queries improve and others break. Report the
    win/loss/tie counts alongside the p-value so the reader can tell which case
    they are looking at.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if not baseline:
        raise ValueError("scores must be non-empty")

    diffs = [t - b for b, t in zip(baseline, treatment, strict=True)]
    observed = sum(diffs) / len(diffs)
    if observed <= 0:
        return 1.0

    rng = random.Random(seed)
    n = len(diffs)
    # Centre the differences so the resampling distribution matches H0: mean = 0.
    centred = [d - observed for d in diffs]
    at_least_as_extreme = sum(
        1 for _ in range(resamples) if sum(rng.choices(centred, k=n)) / n >= observed
    )
    return (at_least_as_extreme + 1) / (resamples + 1)


def holm_bonferroni(
    pvalues: dict[str, float], *, alpha: float = 0.05
) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down correction over a family of comparisons.

    An ablation table with ~40 arms compared against one baseline runs ~40
    hypothesis tests. At alpha=0.05 roughly two come back "significant" by
    chance alone, so an uncorrected table contains false discoveries that a
    reviewer who knows IR will find before reading the headline number.

    Holm rather than plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate; Holm rather than Benjamini-Hochberg because
    the goal is to avoid claiming *any* false improvement, not to control the
    expected proportion of them.

    Returns ``{name: (adjusted_p, reject_null)}``. Adjusted p-values are forced
    monotone non-decreasing, and once the step-down procedure stops rejecting it
    stops for every larger p-value -- that is what makes it a step-*down* test
    rather than 40 independent thresholds.
    """
    if not pvalues:
        return {}
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    rejecting = True
    for i, (name, p_raw) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - i) * p_raw))
        running = adjusted
        rejecting = rejecting and adjusted <= alpha
        out[name] = (adjusted, rejecting)
    return out


def evaluate(
    run: dict[str, Sequence[str]],
    qrels: dict[str, Sequence[str]],
    *,
    metrics: dict[str, Callable[[Sequence[str], Sequence[str]], float]] | None = None,
) -> dict[str, BootstrapCI]:
    """Score a full run and return a confidence interval per metric.

    ``run`` maps query id -> ranked document ids; ``qrels`` maps query id -> gold
    document ids. Queries present in ``qrels`` but missing from ``run`` score 0
    rather than being silently dropped, so a retriever that returns nothing for
    hard queries is penalised instead of flattered.
    """
    metrics = metrics or {
        "R@1": lambda r, g: recall_at_k(r, g, 1),
        "MRR@10": lambda r, g: mrr_at_k(r, g, 10),
        "nDCG@10": lambda r, g: ndcg_at_k(r, g, 10),
    }
    out: dict[str, BootstrapCI] = {}
    for name, fn in metrics.items():
        out[name] = bootstrap_ci([fn(run.get(qid, []), gold) for qid, gold in qrels.items()])
    return out
