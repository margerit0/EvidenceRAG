"""Rank fusion. Currently one function, because one is what the evidence supports.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009) combines runs
by rank rather than by score. That choice is not stylistic. A BM25 score is an
unbounded sum of IDF terms and a cosine similarity lives in [-1, 1]; any
score-level combination has to normalise them first, and every normalisation
(min-max over the returned window, z-score, softmax) is itself a hyper-parameter
that shifts with the query. Ranks are already commensurate, so RRF has exactly
one constant and no per-query calibration.

The constant ``k`` damps the top of each list: a document ranked 1st by one
system and absent from the other scores ``1/(k+1)``, so a small ``k`` lets a
single confident system win outright while a large ``k`` flattens the curve and
rewards documents both systems liked. ``k=60`` is the paper's value and is used
here as the default rather than tuned, so that a tuned row in an ablation table
has something honest to be compared against.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    depth: int | None = None,
) -> list[str]:
    """Fuse ranked document-id lists into one ranking, best first.

    ``k`` is RRF's damping constant, *not* a cut-off -- the unfortunate name
    collision comes from the paper. Use ``depth`` for the cut-off: it truncates
    every input run before fusing, which is what a deployed pipeline does when
    each retriever is asked for its top-N.

    Args:
        runs: one ranked list of document ids per system. Ids may repeat within
            a run; only the best rank counts.
        k: damping constant. Larger values weight agreement between runs more
            heavily relative to one run's confidence.
        weights: per-run multipliers, defaulting to 1.0 each. Only the ratios
            matter, since scaling every weight scales every score.
        depth: keep the first ``depth`` entries of each run. ``None`` keeps all.

    Returns:
        Every document appearing in any (truncated) run, ordered by descending
        fused score. Ties break on document id so the output is stable across
        runs and platforms rather than dependent on dict ordering.
    """
    if not runs:
        raise ValueError("runs must be non-empty")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if depth is not None and depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if weights is None:
        weights = [1.0] * len(runs)
    elif len(weights) != len(runs):
        raise ValueError(f"weights ({len(weights)}) and runs ({len(runs)}) differ in length")

    contributions: dict[str, list[float]] = defaultdict(list)
    for run, weight in zip(runs, weights, strict=True):
        seen: set[str] = set()
        for rank, doc_id in enumerate(run[:depth] if depth is not None else run, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            contributions[doc_id].append(weight / (k + rank))

    # fsum, not a running += : float addition is not associative, so two
    # documents holding the *same multiset* of rank contributions in a different
    # order accumulate to floats one ULP apart. The `(-score, doc_id)` key below
    # would then never reach the doc_id component, and the documented tie-break
    # would silently become "ordered by rounding noise". With two runs the
    # orders coincide and nothing is visible; with three -- which is exactly
    # what adding a reranker arm gives -- it is. fsum rounds the exact sum once,
    # so its result depends on the multiset alone.
    scores = {doc: math.fsum(values) for doc, values in contributions.items()}
    return [doc for doc, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
