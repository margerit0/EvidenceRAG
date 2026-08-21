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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "BootstrapCI",
    "WinLossTie",
    "all_gold_at_k",
    "bootstrap_ci",
    "bootstrap_p_floor",
    "evaluate",
    "hit_at_k",
    "holm_bonferroni",
    "holm_floor_flags",
    "mcnemar_exact",
    "mrr_at_k",
    "ndcg_at_k",
    "paired_bootstrap_test",
    "recall_at_k",
    "win_loss_tie",
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


def bootstrap_p_floor(resamples: int) -> float:
    """Smallest p-value :func:`paired_bootstrap_test` can return for ``resamples``.

    The add-one estimator is ``(count + 1) / (resamples + 1)``, so with the
    default 10,000 resamples nothing below ``9.999e-05`` is representable. A
    value at that floor means zero resampled null statistics were at least as
    extreme as the observation. It does *not* prove that the underlying tail
    probability is below the floor: zero exceedances can still occur when that
    probability is nonzero. Callers should mark the value as a Monte Carlo floor
    (ideally with ``0 / resamples``), never render it as a ``<`` bound. Raising
    ``resamples`` gives a finer estimate when the result remains at the floor.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    return 1.0 / (resamples + 1)


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
    :func:`win_loss_tie` counts alongside the p-value so the reader can tell
    which case they are looking at.

    Two properties worth knowing before quoting the number:

    * **It is one-sided.** A p of 0.60 does not mean "baseline wins"; it means
      "no evidence that treatment wins". Swap the arguments to ask the other
      question, and say which direction was tested when reporting.
    * **It has a resolution floor** of :func:`bootstrap_p_floor`. For a *binary*
      metric prefer :func:`mcnemar_exact`, which is exact and has no floor.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if not baseline:
        raise ValueError("scores must be non-empty")

    diffs = [t - b for b, t in zip(baseline, treatment, strict=True)]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    # Centre the differences so the resampling distribution matches H0: mean = 0.
    #
    # A regression (observed <= 0) is *not* short-circuited to 1.0 here. That
    # shortcut is conservative and never produces a false discovery, but it
    # discards the distinction between "clearly worse" and "a hair below zero",
    # and it feeds a family of identical 1.0s into Holm where the real values
    # would have been spread out. The general path costs one more resampling
    # loop and returns the actual one-sided p-value.
    #
    # The constant-difference case is worth knowing about because it is not
    # symmetric. Centring leaves a zero-variance null, so every resample ties
    # the observation and the test reduces to asking `0 >= observed`: exactly
    # 1.0 when the constant is <= 0, and the floor -- the most significant value
    # the estimator can emit -- when it is > 0, however tiny. A change that
    # shifts every query by the same epsilon therefore reports as the strongest
    # arm in the table. That is the bootstrap being asked a question it cannot
    # answer (there is no per-query variation to resample), not a defect to
    # patch here; report win/loss counts and the effect size next to any p-value
    # and the case is self-evidently degenerate.
    centred = [d - observed for d in diffs]
    at_least_as_extreme = sum(
        1 for _ in range(resamples) if sum(rng.choices(centred, k=n)) / n >= observed
    )
    return (at_least_as_extreme + 1) / (resamples + 1)


@dataclass(frozen=True, slots=True)
class WinLossTie:
    """Per-query outcome breakdown for two systems on the same queries.

    The counts a bare p-value hides. ``+2pp`` produced by 100 wins and 84 losses
    is a different system from ``+2pp`` produced by 16 wins and 0 losses: the
    first has found a genuinely different ranking and is a candidate for fusion,
    the second is the same ranking nudged. Both print the same delta.
    """

    wins: int
    """Queries where ``treatment`` scored strictly higher."""

    losses: int
    """Queries where ``baseline`` scored strictly higher."""

    ties_nonzero: int
    """Queries both systems scored equally and above zero -- both succeeded."""

    ties_zero: int
    """Queries both systems scored zero. **Nothing built on top of these two
    runs can fix them**, so this count is the hard floor on any fusion."""

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.ties_nonzero + self.ties_zero

    @property
    def discordant(self) -> int:
        """Queries the two systems disagree on -- the only ones a paired test sees."""
        return self.wins + self.losses

    @property
    def union_rate(self) -> float:
        """Fraction where *at least one* system scored above zero.

        For a binary metric this is the **oracle ceiling**: the score a perfect
        fusion of exactly these two runs would reach, and therefore the honest
        upper bound to quote before building one. For a graded metric it is only
        a coverage rate, because a fusion could also improve a query both
        systems scored partially.
        """
        return 1.0 - self.ties_zero / self.n if self.n else 0.0

    def __str__(self) -> str:
        # "reachable", not "oracle": union_rate is an oracle ceiling only when
        # the metric is binary, and this dataclass never learns which it was.
        reach = f"{self.union_rate:.1%}" if self.n else "n/a"
        return (
            f"win {self.wins} / loss {self.losses} / tie {self.ties_nonzero}+{self.ties_zero} "
            f"(n={self.n}, reachable {reach})"
        )


def win_loss_tie(baseline: Sequence[float], treatment: Sequence[float]) -> WinLossTie:
    """Count per-query wins, losses and the two kinds of tie.

    Ties are split because they are not interchangeable: a both-succeeded tie is
    headroom already taken, a both-failed tie is headroom no combination of
    these two systems can reach. Collapsing them into one "tie" column is how an
    ablation table ends up claiming a fusion ceiling it cannot hit.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if not baseline:
        raise ValueError("scores must be non-empty")

    wins = losses = ties_nonzero = ties_zero = 0
    for b, t in zip(baseline, treatment, strict=True):
        if t > b:
            wins += 1
        elif b > t:
            losses += 1
        elif b > 0.0:
            ties_nonzero += 1
        else:
            ties_zero += 1
    return WinLossTie(wins=wins, losses=losses, ties_nonzero=ties_nonzero, ties_zero=ties_zero)


def mcnemar_exact(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    alternative: Literal["two-sided", "greater"] = "two-sided",
) -> float:
    """Exact McNemar test for two systems on the same queries, binary outcomes only.

    The right test for a paired *binary* metric, and strictly better than
    :func:`paired_bootstrap_test` there for two reasons. It is exact -- the null
    is ``Binomial(discordant, 0.5)``, evaluated in closed form -- so it has no
    Monte-Carlo resolution floor and no seed. And it is the same number on every
    machine, which a bootstrap only is because a seed was pinned.

    **Which of this project's metrics qualify.** :func:`hit_at_k` and
    :func:`all_gold_at_k` are 0/1 by construction. :func:`recall_at_k` is *not*,
    except when every query has one gold document: on the 2docs/3docs tasks it
    returns 0.5, 1/3 or 2/3, and this function will refuse them rather than
    quietly treating "partial credit changed" as a win. So "R@1" is admissible
    on ``questanswer_1doc`` and inadmissible on the pooled 2,394-query set --
    which is the same restriction :func:`recall_at_k` documents for a different
    reason.

    Only the queries the two systems *disagree* on carry information: a query
    both got right and a query both got wrong say nothing about which system is
    better. That is why a 2pp gap over 800 queries can rest on as few as 30
    discordant pairs, and why quoting :func:`win_loss_tie` next to the p-value
    is not optional.

    ``alternative='greater'`` tests ``treatment > baseline``, matching
    :func:`paired_bootstrap_test`'s direction. The default is two-sided, which
    is the appropriate choice when comparing two systems neither of which was
    designated the incumbent beforehand.

    The exact tail underflows to 0.0 once the discordant count passes ~1074,
    since the denominator ``2**n`` leaves no representable double. That needs
    over a thousand queries on which the two systems disagree; if a table ever
    reaches it, the printed p is a floor rather than a value.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if not baseline:
        raise ValueError("scores must be non-empty")
    if alternative not in ("two-sided", "greater"):
        # Literal is erased at runtime, and 'less' is a plausible thing to try --
        # scipy's binomtest accepts it. Falling through to the two-sided branch
        # would hand back a p-value exactly twice the intended one.
        raise ValueError(f"alternative must be 'two-sided' or 'greater', got {alternative!r}")
    for name, scores in (("baseline", baseline), ("treatment", treatment)):
        # A flag rather than a None sentinel: None is itself a value that must
        # be rejected, and `next(..., None)` cannot tell it from "nothing found".
        for score in scores:
            if score != 0.0 and score != 1.0:  # noqa: PLR1714 - NaN must fail both
                raise ValueError(
                    f"{name} is not binary: found {score!r}; McNemar needs 0/1 outcomes"
                )

    counts = win_loss_tie(baseline, treatment)
    wins, n = counts.wins, counts.discordant
    if n == 0:
        return 1.0

    # Under H0 the wins among discordant pairs are Binomial(n, 0.5). Integer
    # arithmetic throughout: comb() and the 2**n denominator (spelled as a shift
    # so it stays an int for the type checker) are exact, so the tail is correct
    # to the last bit rather than accumulated from n floating-point terms.
    total = 1 << n
    upper = sum(math.comb(n, i) for i in range(wins, n + 1)) / total
    if alternative == "greater":
        return upper
    lower = sum(math.comb(n, i) for i in range(wins + 1)) / total
    return min(1.0, 2 * min(lower, upper))


def holm_floor_flags(
    raw: Mapping[str, float],
    raw_at_floor: Mapping[str, bool],
) -> dict[str, bool]:
    """Propagate Monte Carlo floor provenance through a Holm adjustment.

    This is provenance, not a claim that an adjusted p-value is an upper bound.
    A Holm value inherits the marker only when its active running maximum is
    formed entirely from raw values at their bootstrap floors. An exact or
    otherwise resolved contributor at the same maximum removes the marker.
    """
    if set(raw) != set(raw_at_floor):
        raise ValueError("raw p-values and floor flags must have identical keys")

    ordered = sorted(raw.items(), key=lambda item: item[1])
    running = 0.0
    running_at_floor = False
    floor_flags: dict[str, bool] = {}
    family_size = len(ordered)
    for index, (key, raw_p) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * raw_p)
        candidate_at_floor = raw_at_floor[key]
        if candidate > running and not math.isclose(
            candidate, running, rel_tol=1e-12, abs_tol=1e-15
        ):
            running = candidate
            running_at_floor = candidate_at_floor and candidate < 1.0
        elif math.isclose(candidate, running, rel_tol=1e-12, abs_tol=1e-15):
            running_at_floor = running_at_floor and candidate_at_floor and candidate < 1.0
        floor_flags[key] = running_at_floor
    return floor_flags


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
