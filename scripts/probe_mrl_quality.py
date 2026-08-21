"""Measure what slicing a 4096-d embedding actually costs, on real labelled data.

    uv run python scripts/probe_mrl_quality.py            # 2,000-doc probe
    uv run python scripts/probe_mrl_quality.py --docs 5681  # full eval corpus

``scripts/verify_embedding_api.py`` established that this provider implements
``dimensions=n`` as a prefix slice + renormalise, which is also what Qwen3's
own ``truncate_dim`` does (``modules.json`` slices before ``Normalize``, and
scalar normalisation commutes with slice-then-renormalise). That answers *how*
truncation is implemented. It says nothing about what truncation **costs**.

Two measurements that do say something, neither of which is the L2 energy
profile:

1. **Per-dimension variance across documents.** Magnitude is the wrong signal --
   after L2 normalisation ``||v[:n]|| ~ sqrt(n/N)`` almost by construction, so a
   "front-loading ratio" near 1.0 is an artefact rather than a finding. What
   distinguishes an MRL-trained model is whether early dimensions *discriminate*
   between documents, and that is variance across a corpus, not norm within one
   vector.

2. **Retrieval quality per width.** The only measurement whose outcome changes a
   decision. Reported against a stated ``corpus_size`` because R@1 is meaningless
   without one -- see the saturation table in README.

The whole sweep runs off **one** embedding pass: documents are embedded once at
full width and cached, and every narrower row is a client-side slice of that
cache. That is not a shortcut, it is the finding from ``verify_embedding_api.py``
being cashed in -- and re-running this script costs nothing.

Held to ``questanswer_1doc`` so R@1 is not capped by gold arity (a 3-gold query
cannot exceed 1/3 at k=1; see :func:`zhrag.eval.metrics.recall_at_k`).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import Query, sample_corpus
from zhrag.eval.metrics import (
    bootstrap_ci,
    bootstrap_p_floor,
    holm_bonferroni,
    holm_floor_flags,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_test,
    recall_at_k,
)
from zhrag.io_utils import read_jsonl
from zhrag.providers.embedding import (
    QUERY_PROMPT,
    EmbeddingClient,
    EmbeddingConfig,
    load_env,
    load_or_embed,
)

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
CACHE = EXPANDED / "emb_cache_4096.jsonl"
#: Queries need their own cache because Qwen3 is asymmetric: a query carries the
#: instruct prefix and a document carries none, so the *same* string embeds to
#: two different vectors depending on which side it is on. One shared cache
#: keyed on raw text would silently serve the wrong one.
QUERY_CACHE = EXPANDED / "emb_cache_queries_4096.jsonl"

#: Widths to sweep. 4096 is the model's native width; 64 is the smallest the
#: Qwen3 MRL whitelist mentions. Going below the useful range on purpose: a
#: curve that only degrades at the far end is far more convincing than three
#: points that all look fine.
DIMS = (4096, 2048, 1024, 512, 256, 128, 64)

#: Bootstrap resamples for every CI and p-value in this script. Named rather
#: than defaulted because the p-value column has to be rendered against the
#: floor this number implies -- see :func:`zhrag.eval.metrics.bootstrap_p_floor`.
RESAMPLES = 10_000


def _l2(vector: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in vector))


def _slice(vector: list[float], dim: int) -> list[float]:
    """Prefix slice + L2 renormalise -- identical to Qwen3's own truncate_dim."""
    if dim >= len(vector):
        return vector
    prefix = vector[:dim]
    scale = _l2(prefix)
    return [x / scale for x in prefix]


def report_variance(vectors: NDArray[np.float32]) -> None:
    """Do early dimensions discriminate between documents more than late ones?"""
    print("\n\n## 1. Per-dimension variance across documents\n")
    print("   Magnitude is the wrong probe: after L2 normalisation ||v[:n]|| ~ sqrt(n/N)")
    print("   holds almost by construction. Variance measures whether a dimension")
    print("   actually separates documents -- which is what MRL training targets.\n")

    variances = vectors.var(axis=0)
    width = variances.size
    band = width // 8
    overall = float(variances.mean())

    print(f"   {'dim range':>13} {'mean variance':>15} {'vs overall':>12}")
    print(f"   {'-' * 13} {'-' * 15} {'-' * 12}")
    for i in range(8):
        lo, hi = i * band, (i + 1) * band
        mean_var = float(variances[lo:hi].mean())
        print(f"   {f'{lo:,}-{hi:,}':>13} {mean_var:>15.3e} {mean_var / overall:>11.3f}x")

    first, last = float(variances[:band].mean()), float(variances[-band:].mean())
    print(f"\n   first {band}/last {band} variance ratio = {first / last:.3f}")
    if first / last > 1.5:
        print("   => early dimensions carry more discriminative signal (MRL-consistent).")
    elif first / last > 0.9:
        print("   => variance is flat across dimensions. Information is NOT ordered by")
        print("      index, so truncation discards signal roughly in proportion to width.")
        print("      Whether that costs accuracy is an empirical question -- see table 2.")
    else:
        print("   => later dimensions carry MORE variance than earlier ones.")


def _rank(
    queries: NDArray[np.float32], docs: NDArray[np.float32], doc_ids: list[str], k: int = 10
) -> list[list[str]]:
    """Top-k document ids per query, by cosine on unit-norm slices.

    One matmul rather than a Python loop. At 800 x 5,681 x 4,096 this is ~19
    billion multiply-adds; the pure-Python version of the same sweep would run
    for over half an hour and hold 745 MB of float objects.
    """
    sims = queries @ docs.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, 1), axis=1), 1)
    return [[doc_ids[i] for i in row] for row in ordered]


def _slice_matrix(matrix: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Prefix slice + row-wise L2 renormalise -- Qwen3's own truncate_dim."""
    if dim >= matrix.shape[1]:
        return matrix
    prefix = matrix[:, :dim]
    return prefix / np.linalg.norm(prefix, axis=1, keepdims=True)


def report_retrieval(
    doc_matrix: NDArray[np.float32],
    doc_ids: list[str],
    query_matrix: NDArray[np.float32],
    queries: list[Query],
) -> None:
    print("\n\n## 2. Retrieval quality per width (the measurement that decides)\n")
    print(f"   corpus_size = {len(doc_ids):,} documents, {len(queries):,} queries")
    print("   task = questanswer_1doc only, so R@1 is not capped by gold arity.")
    print("   R@1 carries a 95% bootstrap CI; p is a one-sided paired bootstrap for")
    print("   'dim=4096 beats this row', Holm-corrected across the whole family.\n")
    print(
        f"   {'dim':>6} {'R@1 [95% CI]':>22} {'MRR@10':>8} {'nDCG@10':>8} "
        f"{'vs 4096':>9} {'p(Holm)':>9} {'MB':>7}"
    )
    print(f"   {'-' * 6} {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 7}")

    rows: dict[int, dict[str, Any]] = {}
    per_query: dict[int, list[float]] = {}

    for dim in DIMS:
        ranked = _rank(_slice_matrix(query_matrix, dim), _slice_matrix(doc_matrix, dim), doc_ids)
        pairs = list(zip(ranked, queries, strict=True))
        n = len(pairs)
        # Keep the per-query vector: the paired test needs the same queries in
        # both arms, and that pairing is what makes it sensitive enough to
        # resolve differences this small.
        per_query[dim] = [recall_at_k(r, q.gold_doc_ids, 1) for r, q in pairs]
        rows[dim] = {
            "ci": bootstrap_ci(per_query[dim], resamples=RESAMPLES),
            "mrr": sum(mrr_at_k(r, q.gold_doc_ids, 10) for r, q in pairs) / n,
            "ndcg": sum(ndcg_at_k(r, q.gold_doc_ids, 10) for r, q in pairs) / n,
            "mb": len(doc_ids) * dim * 4 / 1e6,
        }

    full = DIMS[0]
    raw_p = {
        str(dim): paired_bootstrap_test(per_query[dim], per_query[full], resamples=RESAMPLES)
        for dim in DIMS
        if dim != full
    }
    adjusted = holm_bonferroni(raw_p)
    floor = bootstrap_p_floor(RESAMPLES)
    raw_floors = {key: p <= floor + 1e-12 for key, p in raw_p.items()}
    adjusted_floors = holm_floor_flags(raw_p, raw_floors)

    for dim in DIMS:
        row, ci = rows[dim], rows[dim]["ci"]
        delta = (ci.mean - rows[full]["ci"].mean) * 100
        if dim == full:
            verdict = "baseline"
        else:
            p, reject = adjusted[str(dim)]
            marker = "†" if adjusted_floors[str(dim)] else ""
            verdict = f"{p:.3f}{marker}{'*' if reject else ''}"
        print(
            f"   {dim:>6} {ci.mean:>7.1%} [{ci.low:.1%}, {ci.high:.1%}] {row['mrr']:>8.3f} "
            f"{row['ndcg']:>8.3f} {delta:>+8.2f}pp {verdict:>9} {row['mb']:>7.1f}"
        )

    print("\n   * = significant at family-wise alpha=0.05 after Holm correction.")
    print(
        f"   † means the active Holm estimate inherits a raw {RESAMPLES:,}-resample "
        f"floor ({floor:.1e}; 0/{RESAMPLES:,} null exceedances)."
    )
    print("   It marks Monte Carlo resolution, not a proven '<' bound on the true tail.")
    print("   Note the three rows that all print 0.684: Holm forces adjusted p-values to")
    print("   be monotone, so a -0.50pp regression and a +0.37pp improvement land on the")
    print("   same number. Read the delta column, not the p column, for direction.")
    significant = [d for d in DIMS if d != full and adjusted[str(d)][1]]
    if not significant:
        print("   => NO width is significantly worse than 4096 on this query set.")
        print(f"      Read that as 'not resolvable at n={len(queries)}', not 'proven equal'.")
    else:
        print(f"   => significantly worse than {full}: {significant}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=2000, help="corpus size (gold always kept)")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if not (EXPANDED / "corpus.jsonl").exists():
        print(f"! {EXPANDED} not built -- run scripts/build_eval_corpus.py first")
        return 1

    config = EmbeddingConfig.from_env(load_env(ROOT / ".env"))
    client = EmbeddingClient(config=config)

    pool = {r["doc_id"]: r["text"] for r in read_jsonl(EXPANDED / "corpus.jsonl")}
    all_queries = [
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
    queries = all_queries[: args.queries]
    corpus = sample_corpus(pool, queries, size=args.docs)
    print(f"corpus {len(corpus):,} docs | {len(queries):,} queries | model {config.model}\n")

    doc_vectors = load_or_embed(CACHE, corpus, client, model=config.model, batch=args.batch)

    # The query cache is keyed on query_id but holds the *prefixed* text, which
    # is what was actually sent. Passing the prompt to load_or_embed rather than
    # baking it into the values keeps it in the cache sidecar, where a later run
    # that changed the instruction cannot silently reuse these vectors.
    query_map = load_or_embed(
        QUERY_CACHE,
        {q.query_id: q.question for q in queries},
        client,
        model=config.model,
        prompt=QUERY_PROMPT,
        batch=args.batch,
    )

    # float32, not float64: storage error is ~1e-7 relative, five orders of
    # magnitude below this provider's own reproducibility floor of cos 0.99993
    # (see scripts/verify_embedding_api.py). Doubling the width would record
    # noise more precisely, nothing else. Also halves 5,681x4096 to 93 MB.
    doc_ids = list(doc_vectors)
    doc_matrix = np.asarray([doc_vectors[d] for d in doc_ids], dtype=np.float32)
    query_matrix = np.asarray([query_map[q.query_id] for q in queries], dtype=np.float32)
    print(f"matrices: docs {doc_matrix.shape} queries {query_matrix.shape} float32")

    report_variance(doc_matrix)
    report_retrieval(doc_matrix, doc_ids, query_matrix, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
