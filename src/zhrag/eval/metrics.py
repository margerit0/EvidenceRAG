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
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "BootstrapCI",
    "WinLossTie",
    "all_gold_at_k",
    "bootstrap_ci",
    "bootstrap_p_floor",
    "clustered_bootstrap_ci",
    "clustered_paired_bootstrap_test",
    "evaluate",
    "graded_ndcg_at_k",
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


def _validate_unique_ranked(ranked: Sequence[str], k: int) -> None:
    prefix = ranked[:k]
    if len(set(prefix)) != len(prefix):
        raise ValueError(f"ranked top-{k} must not contain duplicate document ids")


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k with the standard ``1/log2(rank+1)`` discount."""
    _validate(k)
    _validate_unique_ranked(ranked, k)
    goldset = set(gold)
    if not goldset:
        raise ValueError("gold must be non-empty")
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1) if doc in goldset)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(goldset)) + 1))
    return dcg / ideal


def graded_ndcg_at_k(
    ranked: Sequence[str],
    grades: Mapping[str, int],
    k: int = 10,
) -> float:
    """Graded nDCG@k with exponential gain ``2**grade - 1``.

    A missing document receives zero gain, but callers must first prove that the
    evaluated prefix is fully judged. This metric cannot distinguish an explicit
    zero from an absent judgement by itself.
    """
    _validate(k)
    _validate_unique_ranked(ranked, k)
    gains: dict[str, int] = {}
    for doc_id, grade in grades.items():
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("grade ids must be non-empty strings")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
            raise ValueError(f"grade for {doc_id!r} must be a non-negative integer")
        gains[doc_id] = (1 << grade) - 1
    positive = sorted((gain for gain in gains.values() if gain > 0), reverse=True)
    if not positive:
        raise ValueError("grades must contain at least one positive relevance grade")
    dcg = sum(
        gains.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(positive[:k], start=1))
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
    """Percentile bootstrap confidence interval over independent score units.

    Report intervals rather than bare point estimates: at n=500 an *unpaired*
    comparison needs several points of separation before the intervals stop
    overlapping. Note this is the weaker of the two tools here -- for comparing
    ablation arms that share a query set, use :func:`paired_bootstrap_test`,
    which is far more sensitive. Correlated observations must first be reduced
    to independent cluster-level values before calling this function.
    """
    _validate_bootstrap_scores(scores, name="scores", resamples=resamples)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
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


def _validate_bootstrap_scores(
    scores: Sequence[float],
    *,
    name: str,
    resamples: int,
) -> None:
    if not scores:
        raise ValueError("scores must be non-empty")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError(f"{name} must contain only finite values")


def _cluster_groups[Cluster: Hashable](
    values: Sequence[float],
    clusters: Sequence[Cluster],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if len(values) != len(clusters):
        raise ValueError(f"{name}/clusters length mismatch: {len(values)} vs {len(clusters)}")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    grouped: dict[Cluster, list[float]] = {}
    for value, cluster in zip(values, clusters, strict=True):
        try:
            grouped.setdefault(cluster, []).append(value)
        except TypeError as exc:
            raise ValueError("clusters must contain hashable values") from exc
    return tuple(tuple(grouped[cluster]) for cluster in grouped)


def _resampled_cluster_mean(
    groups: Sequence[Sequence[float]],
    rng: random.Random,
) -> float:
    sampled = rng.choices(groups, k=len(groups))
    total = sum(sum(group) for group in sampled)
    count = sum(len(group) for group in sampled)
    return total / count


def clustered_bootstrap_ci[Cluster: Hashable](
    scores: Sequence[float],
    clusters: Sequence[Cluster],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Cluster bootstrap CI with a pair-weighted point estimate.

    Clusters, rather than individual observations, are sampled with replacement.
    Every selected cluster contributes all of its original observations, so the
    bootstrap distribution permits arbitrary within-cluster dependence while the
    reported point estimate remains the ordinary observation-weighted mean.
    """
    _validate_bootstrap_scores(scores, name="scores", resamples=resamples)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    groups = _cluster_groups(scores, clusters, name="scores")
    rng = random.Random(seed)
    means = sorted(_resampled_cluster_mean(groups, rng) for _ in range(resamples))
    alpha = (1 - confidence) / 2
    return BootstrapCI(
        mean=sum(scores) / len(scores),
        low=means[int(alpha * resamples)],
        high=means[min(int((1 - alpha) * resamples), resamples - 1)],
        n=len(groups),
    )


def clustered_paired_bootstrap_test[Cluster: Hashable](
    baseline: Sequence[float],
    treatment: Sequence[float],
    clusters: Sequence[Cluster],
    *,
    alternative: Literal["greater", "two-sided"] = "greater",
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Centred paired cluster-bootstrap p-value.

    Treatment-minus-baseline differences stay paired inside each cluster. The
    null centres the observation-weighted differences globally, then samples
    whole clusters with replacement and compares their pooled mean statistic.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if alternative not in ("greater", "two-sided"):
        raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")
    _validate_bootstrap_scores(baseline, name="baseline", resamples=resamples)
    _validate_bootstrap_scores(treatment, name="treatment", resamples=resamples)
    diffs = [right - left for left, right in zip(baseline, treatment, strict=True)]
    observed = sum(diffs) / len(diffs)
    centred = [difference - observed for difference in diffs]
    groups = _cluster_groups(centred, clusters, name="differences")
    rng = random.Random(seed)

    def extreme(sampled: float) -> bool:
        if alternative == "two-sided":
            return abs(sampled) >= abs(observed)
        return sampled >= observed

    at_least_as_extreme = sum(
        1 for _ in range(resamples) if extreme(_resampled_cluster_mean(groups, rng))
    )
    return (at_least_as_extreme + 1) / (resamples + 1)


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
    alternative: Literal["greater", "two-sided"] = "greater",
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Centred paired-bootstrap p-value for one- or two-sided alternatives.

    Score units are resampled as pairs, which is what makes this *paired*: the
    two systems saw identical observations, so their difference is less noisy
    than either system's absolute score. Correlated observations must first be
    reduced to independent cluster-level values.

    ``alternative='greater'`` preserves the original one-sided contract for
    ``treatment > baseline``. ``'two-sided'`` tests any non-zero mean difference
    by comparing absolute statistics. Both forms have the add-one resolution
    floor exposed by :func:`bootstrap_p_floor`; report effect sizes and
    :func:`win_loss_tie` counts next to the p-value.
    """
    if len(baseline) != len(treatment):
        raise ValueError(f"length mismatch: {len(baseline)} vs {len(treatment)}")
    if alternative not in ("greater", "two-sided"):
        raise ValueError(f"alternative must be 'greater' or 'two-sided', got {alternative!r}")
    _validate_bootstrap_scores(baseline, name="baseline", resamples=resamples)
    _validate_bootstrap_scores(treatment, name="treatment", resamples=resamples)

    diffs = [t - b for b, t in zip(baseline, treatment, strict=True)]
    observed = sum(diffs) / len(diffs)
    centred = [difference - observed for difference in diffs]
    rng = random.Random(seed)
    n = len(diffs)

    def extreme(sampled: float) -> bool:
        if alternative == "two-sided":
            return abs(sampled) >= abs(observed)
        return sampled >= observed

    at_least_as_extreme = sum(
        1 for _ in range(resamples) if extreme(sum(rng.choices(centred, k=n)) / n)
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


def _ordered_pvalues(pvalues: Mapping[str, float]) -> list[tuple[str, float]]:
    ordered: list[tuple[str, float]] = []
    for name, value in pvalues.items():
        if not isinstance(name, str) or not name:
            raise ValueError("p-value names must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"p-value for {name!r} must be finite and in [0, 1]")
        ordered.append((name, float(value)))
    return sorted(ordered, key=lambda item: (item[1], item[0]))


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
    if any(not isinstance(flag, bool) for flag in raw_at_floor.values()):
        raise ValueError("floor flags must be boolean")

    ordered = _ordered_pvalues(raw)
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

    ordered = _ordered_pvalues(pvalues)
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
