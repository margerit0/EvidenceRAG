"""How many evaluation queries does this benchmark actually need?

    uv run python scripts/power_analysis.py

Two different questions get confused with each other, so this script answers
them separately.

**Absolute precision** -- "what is this system's R@1?" -- is governed by the
binomial spread of one score vector, and shrinks only as 1/sqrt(n). Halving the
error bar costs 4x the queries.

**Comparative precision** -- "is arm B better than arm A?" -- is what an ablation
table actually needs, and it behaves completely differently. Both arms see the
same queries, so the paired test looks at the per-query *difference*, which is 0
for every query where the two arms agree. For a binary metric where the
treatment fixes ``f`` queries and breaks ``b``, the test statistic reduces to
roughly ``(f - b) / sqrt(f + b)`` -- **n does not appear**. What decides
significance is the absolute number of queries that changed and how one-sided
the change was, not how many queries were asked.

The practical consequence is counterintuitive: n barely matters when a change is
clean, and matters a great deal when a change is churny. Adding queries buys
almost nothing for a strict improvement, and buys real power for the messy
improvements that ablations actually produce.
"""

from __future__ import annotations

import math

from zhrag.eval.metrics import bootstrap_ci, paired_bootstrap_test

BASELINE_R1 = 0.759  # char-bigram BM25 @ 5,681 documents, questanswer_1doc
Z_ONE_SIDED_95 = 1.645
AVAILABLE = {"questanswer_1doc only": 800, "all three QA tasks": 2394}


def _paired_z(n: int, fixed: int, broken: int) -> float:
    """z-statistic for a paired binary comparison, from the difference vector."""
    if fixed + broken == 0 or fixed + broken > n:
        return 0.0
    mean = (fixed - broken) / n
    var = (fixed + broken) / n - mean**2
    if var <= 0:
        return math.inf
    return mean / math.sqrt(var / n)


def _diff_vectors(n: int, fixed: int, broken: int) -> tuple[list[float], list[float]]:
    """Baseline/treatment score vectors where ``fixed`` queries improve and ``broken`` regress.

    A broken query must score 1.0 in the baseline and 0.0 in the treatment.
    Building it the other way round -- 0.0 in both -- silently produces no
    difference at all, and the resulting "churn" column measures nothing.
    """
    if fixed + broken > n:
        raise ValueError(f"fixed+broken ({fixed + broken}) exceeds n ({n})")
    baseline = [0.0] * fixed + [1.0] * broken + [1.0] * (n - fixed - broken)
    treatment = [1.0] * fixed + [0.0] * broken + [1.0] * (n - fixed - broken)
    return baseline, treatment


def minimum_detectable_gain(
    n: int, churn: float, *, resamples: int = 1500, alpha: float = 0.05
) -> tuple[int, int, float]:
    """Smallest one-sided-significant net gain at ``n`` queries, measured empirically.

    Uses the shipped :func:`paired_bootstrap_test` rather than a normal
    approximation, because the approximation is unreliable exactly where the
    interesting answers live: when only a handful of queries change, the
    bootstrap distribution is too discrete for the CLT and the closed form
    overstates power (see the validation table).

    ``churn`` is the ratio of newly-broken to newly-fixed queries: 0.0 is a
    strict improvement, 0.8 means four queries break for every five fixed.
    Returns ``(fixed, broken, net_gain_percentage_points)``.
    """

    def significant(fixed: int) -> bool:
        broken = round(churn * fixed)
        if fixed + broken > n:
            return False
        base, treat = _diff_vectors(n, fixed, broken)
        return paired_bootstrap_test(base, treat, resamples=resamples) < alpha

    lo, hi = 1, n // 2
    if not significant(hi):
        return 0, 0, math.inf
    while lo < hi:  # smallest `fixed` that is significant; p is monotone in it
        mid = (lo + hi) // 2
        if significant(mid):
            hi = mid
        else:
            lo = mid + 1
    broken = round(churn * lo)
    return lo, broken, (lo - broken) / n * 100


def absolute_precision() -> None:
    print("## 1. Absolute precision -- 'what is this system's R@1?'\n")
    print("   Simulated at the measured baseline R@1 = 75.9%.\n")
    print(f"   {'n':>6} | {'95% CI':>18} | {'half-width':>10}")
    print(f"   {'-' * 6} | {'-' * 18} | {'-' * 10}")
    for n in (200, 400, 800, 1600, 2394, 5000, 10000):
        hits = round(BASELINE_R1 * n)
        scores = [1.0] * hits + [0.0] * (n - hits)
        ci = bootstrap_ci(scores, resamples=4000)
        half = (ci.high - ci.low) / 2 * 100
        note = "  <- available" if n in AVAILABLE.values() else ""
        print(f"   {n:>6,} | [{ci.low:>6.1%}, {ci.high:>6.1%}] | {half:>9.2f}pp{note}")
    print("\n   Shrinks as 1/sqrt(n): 4x the queries to halve the error bar.")


def comparative_precision() -> None:
    print("\n\n## 2. Comparative precision -- 'did arm B beat arm A?'\n")
    print("   Minimum net gain clearing the shipped one-sided paired bootstrap at p<0.05,")
    print("   found by binary search. Cell shows net gain, and (fixed/broken) queries.\n")
    churns = (0.0, 0.3, 0.5, 0.8)
    print(f"   {'n':>6} | " + " | ".join(f"churn {c:>4.1f}" for c in churns))
    print(f"   {'-' * 6} | " + " | ".join("-" * 16 for _ in churns))
    for n in (400, 800, 1600, 2394):
        cells = []
        for churn in churns:
            fixed, broken, net = minimum_detectable_gain(n, churn)
            cells.append(f"{net:>6.2f}pp ({fixed:>3}/{broken:<3})")
        note = "  <- available" if n in AVAILABLE.values() else ""
        print(f"   {n:>6,} | " + " | ".join(cells) + note)

    print("\n   churn = newly-broken / newly-fixed. 0.0 is a strict improvement;")
    print("   0.8 means the change breaks 4 queries for every 5 it fixes.")


def validate_against_shipped_test() -> None:
    """Show where the normal approximation stops being trustworthy."""
    print("\n\n## 3. Why the closed form is not used above\n")
    print("   The paired statistic reduces to roughly (f-b)/sqrt(f+b), which suggests")
    print("   n is irrelevant. That holds only once enough queries have changed.\n")
    print(
        f"   {'n':>6} {'fixed':>6} {'broken':>7} | {'z':>6} {'z says':>8} | "
        f"{'actual p':>9} {'bootstrap':>10}"
    )
    print(f"   {'-' * 6} {'-' * 6} {'-' * 7} | {'-' * 6} {'-' * 8} | {'-' * 9} {'-' * 10}")
    cases = [(800, 3, 0), (800, 6, 0), (800, 12, 0), (800, 40, 20), (2394, 40, 20)]
    for n, fixed, broken in cases:
        base, treat = _diff_vectors(n, fixed, broken)
        p = paired_bootstrap_test(base, treat, resamples=4000)
        z = _paired_z(n, fixed, broken)
        flag = "" if (z >= Z_ONE_SIDED_95) == (p < 0.05) else "   <- disagree"
        print(
            f"   {n:>6,} {fixed:>6} {broken:>7} | {z:>6.2f} {z >= Z_ONE_SIDED_95!s:>8} | "
            f"{p:>9.4f} {p < 0.05!s:>10}{flag}"
        )
    print("\n   With only a few changed queries the bootstrap distribution is too discrete")
    print("   for the CLT, and the closed form claims significance the resampling denies.")
    print("   Trust the bootstrap; it is what the ablation table will actually report.")


def main() -> int:
    print("# Statistical power of the CRUD-RAG evaluation set")
    print(f"# Available queries: {AVAILABLE}")
    print("# Hard ceiling: 2,394 -- that is every QA record in split_merged.json.\n")
    absolute_precision()
    comparative_precision()
    validate_against_shipped_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
