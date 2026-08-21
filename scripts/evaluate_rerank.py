"""Score and evaluate Qwen3 reranking over the frozen hybrid candidate run.

Dry-run (default; no network):

    uv run python scripts/evaluate_rerank.py

Populate the ignored, resumable top-100 score cache:

    uv run python scripts/evaluate_rerank.py --score --max-queries 1  # paid smoke
    uv run python scripts/evaluate_rerank.py --score                  # resume all

Analyse only after every expected pair is cached:

    uv run python scripts/evaluate_rerank.py --analyze --resamples 100000

The retrieval input is frozen to equal-weight RRF k=10/depth=100 over
char-bigram BM25 and dense-4096. One API request scores a query's first 100
fused candidates. The score cache then supports two offline treatments:
rerank the first 50 and append the untouched tail, or rerank all 100. No second
paid pass is needed.

The score cache is local derived data under ``eval-expanded`` and must not be
committed. Its sidecar binds it to the model, instruction, retrieval settings,
ordered candidate ids, and exact query/document text sent to the API.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from zhrag.eval.crud import QA_TASKS, Query, sample_corpus
from zhrag.eval.rerank import (
    PairwiseComparison,
    candidate_run_fingerprint,
    missing_score_queries,
    paired_metric_family,
    rerank_input_fingerprint,
    rerank_prefix,
)
from zhrag.eval.retrieval import (
    bm25_runs,
    dense_runs,
    load_embedding_matrix,
    load_queries,
    per_query_metrics,
)
from zhrag.io_utils import read_jsonl
from zhrag.providers import (
    DEFAULT_RERANK_INSTRUCTION,
    PairScore,
    RerankClient,
    RerankConfig,
    append_pair_scores,
    estimate_rerank_tokens,
    load_env,
    load_pair_score_provenance,
    load_pair_scores,
    prepare_pair_score_cache,
    validate_pair_score_cache,
)
from zhrag.retrieval import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
DOC_CACHE = EXPANDED / "emb_cache_4096.jsonl"
QUERY_CACHE = EXPANDED / "emb_cache_queries_4096.jsonl"
SCORE_CACHE = EXPANDED / "rerank_scores_top100.jsonl"

WIDTH = 4096
RETRIEVAL_DEPTH = 100
RRF_K = 10
SCORE_DEPTH = 100
EVAL_DEPTHS = (50, 100)
ILLUSTRATIVE_CNY_PER_MTOK = 0.28
RERANK_LABELS = {depth: f"rerank-8b@{depth}" for depth in EVAL_DEPTHS}
BASELINE_LABEL = "hybrid RRF k=10/d=100"


class Experiment(NamedTuple):
    corpus: dict[str, str]
    queries: list[Query]
    candidates: list[list[str]]


def _load_experiment() -> Experiment:
    required = (
        EXPANDED / "corpus.jsonl",
        EXPANDED / "qrels.jsonl",
        DOC_CACHE,
        QUERY_CACHE,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "! rerank experiment inputs are absent:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    pool = {row["doc_id"]: row["text"] for row in read_jsonl(EXPANDED / "corpus.jsonl")}
    queries = load_queries(EXPANDED / "qrels.jsonl", list(QA_TASKS))
    corpus = sample_corpus(pool, queries)
    doc_ids = list(corpus)
    doc_matrix, _ = load_embedding_matrix(DOC_CACHE, doc_ids, width=WIDTH)
    query_matrix, _ = load_embedding_matrix(
        QUERY_CACHE,
        [query.query_id for query in queries],
        width=WIDTH,
    )
    lexical = bm25_runs(corpus, queries, depth=RETRIEVAL_DEPTH)
    dense = dense_runs(query_matrix, doc_matrix, doc_ids, depth=RETRIEVAL_DEPTH)
    candidates = [
        reciprocal_rank_fusion([bm25, semantic], k=RRF_K, depth=RETRIEVAL_DEPTH)
        for bm25, semantic in zip(lexical, dense, strict=True)
    ]
    if any(len(run) < SCORE_DEPTH for run in candidates):
        raise SystemExit("! a fused run has fewer than 100 candidates")
    return Experiment(corpus=corpus, queries=queries, candidates=candidates)


def _provenance(
    experiment: Experiment,
    *,
    model: str,
    endpoint: str,
) -> dict[str, object]:
    return {
        "schema": "zhrag-rerank-pair-scores-v1",
        "model": model,
        "endpoint": endpoint,
        "request_contract": "one query + ordered top-100 documents; top_n=100; v1",
        "instruction": DEFAULT_RERANK_INSTRUCTION,
        "candidate_run": {
            "lexical": "BM25 char-bigram",
            "dense": "Qwen3-Embedding-8B 4096d",
            "fusion": "equal-weight RRF",
            "rrf_k": RRF_K,
            "retrieval_depth": RETRIEVAL_DEPTH,
            "score_depth": SCORE_DEPTH,
            "query_count": len(experiment.queries),
            "fingerprint_sha256": candidate_run_fingerprint(
                experiment.queries,
                experiment.candidates,
                depth=SCORE_DEPTH,
            ),
            "input_fingerprint_sha256": rerank_input_fingerprint(
                experiment.queries,
                experiment.candidates,
                experiment.corpus,
                depth=SCORE_DEPTH,
            ),
        },
    }


def _cached_identity() -> tuple[str, str]:
    recorded = load_pair_score_provenance(SCORE_CACHE)
    model = recorded.get("model")
    endpoint = recorded.get("endpoint")
    if not isinstance(model, str) or not model:
        raise SystemExit("! rerank cache metadata has no valid model")
    if not isinstance(endpoint, str) or not endpoint:
        raise SystemExit("! rerank cache metadata has no valid endpoint")
    return model, endpoint


def _expected_pairs(experiment: Experiment) -> set[tuple[str, str]]:
    return {
        (query.query_id, doc_id)
        for query, run in zip(experiment.queries, experiment.candidates, strict=True)
        for doc_id in run[:SCORE_DEPTH]
    }


def _coverage(
    experiment: Experiment,
    scores: Mapping[tuple[str, str], float],
) -> tuple[list[str], int]:
    missing_queries = missing_score_queries(
        experiment.queries,
        experiment.candidates,
        scores,
        depth=SCORE_DEPTH,
    )
    covered = sum(pair in scores for pair in _expected_pairs(experiment))
    return missing_queries, covered


def _estimate_missing_tokens(
    experiment: Experiment,
    missing_ids: set[str],
) -> int:
    return sum(
        estimate_rerank_tokens(
            query.question,
            [experiment.corpus[doc_id] for doc_id in run[:SCORE_DEPTH]],
        )
        for query, run in zip(experiment.queries, experiment.candidates, strict=True)
        if query.query_id in missing_ids
    )


def _request_tokens(
    experiment: Experiment,
    query_ids: set[str],
) -> list[int]:
    return [
        estimate_rerank_tokens(
            query.question,
            [experiment.corpus[doc_id] for doc_id in run[:SCORE_DEPTH]],
        )
        for query, run in zip(experiment.queries, experiment.candidates, strict=True)
        if query.query_id in query_ids
    ]


def _score(
    experiment: Experiment,
    client: RerankClient,
    scores: dict[tuple[str, str], float],
    *,
    max_queries: int | None,
) -> None:
    missing, _ = _coverage(experiment, scores)
    selected = set(missing[:max_queries] if max_queries is not None else missing)
    total = len(selected)
    prompt_tokens = 0
    done = 0
    for query, run in zip(experiment.queries, experiment.candidates, strict=True):
        if query.query_id not in selected:
            continue
        doc_ids = run[:SCORE_DEPTH]
        result = client.score(
            query.question,
            [experiment.corpus[doc_id] for doc_id in doc_ids],
        )
        rows = [
            PairScore(query.query_id, doc_id, score)
            for doc_id, score in zip(doc_ids, result.scores, strict=True)
        ]
        append_pair_scores(SCORE_CACHE, rows)
        scores.update({(row.query_id, row.doc_id): row.score for row in rows})
        if result.prompt_tokens is not None:
            prompt_tokens += result.prompt_tokens
        done += 1
        print(
            f"scored {done:,}/{total:,} queries | {done * SCORE_DEPTH:,} pairs"
            f" | API prompt tokens {prompt_tokens:,}",
            flush=True,
        )


def _reranked_runs(
    experiment: Experiment,
    scores: Mapping[tuple[str, str], float],
    *,
    depth: int,
) -> list[list[str]]:
    return [
        rerank_prefix(
            run,
            {doc_id: scores[(query.query_id, doc_id)] for doc_id in run[:depth]},
            depth=depth,
        )
        for query, run in zip(experiment.queries, experiment.candidates, strict=True)
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _fmt_p(p: float, *, at_floor: bool) -> str:
    rendered = f"{p:.2e}" if 0.0 < p < 1e-4 else f"{p:.4f}"
    return f"{rendered}†" if at_floor else rendered


def _families(
    scored: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    *,
    resamples: int,
) -> tuple[tuple[str, bool, list[PairwiseComparison]], ...]:
    actual_arities = set(scored)
    expected_arities = {1, 2, 3}
    if actual_arities != expected_arities:
        raise ValueError(
            f"rerank inference requires exactly arities 1, 2, and 3; got {sorted(actual_arities)}"
        )
    efficacy = tuple((BASELINE_LABEL, RERANK_LABELS[depth]) for depth in EVAL_DEPTHS)
    depth = ((RERANK_LABELS[50], RERANK_LABELS[100]),)
    return (
        (
            "Efficacy binary",
            True,
            paired_metric_family(
                scored,
                comparisons=efficacy,
                metrics=("hit@1", "ALL@10"),
                binary=True,
                resamples=resamples,
            ),
        ),
        (
            "Efficacy graded",
            False,
            paired_metric_family(
                scored,
                comparisons=efficacy,
                metrics=("MRR@10", "nDCG@10"),
                binary=False,
                resamples=resamples,
            ),
        ),
        (
            "Depth binary",
            True,
            paired_metric_family(
                scored,
                comparisons=depth,
                metrics=("hit@1", "ALL@10"),
                binary=True,
                resamples=resamples,
            ),
        ),
        (
            "Depth graded",
            False,
            paired_metric_family(
                scored,
                comparisons=depth,
                metrics=("MRR@10", "nDCG@10"),
                binary=False,
                resamples=resamples,
            ),
        ),
    )


def _analyse(
    experiment: Experiment,
    scores: Mapping[tuple[str, str], float],
    *,
    resamples: int,
) -> None:
    missing, covered = _coverage(experiment, scores)
    expected_pairs = _expected_pairs(experiment)
    expected = len(expected_pairs)
    unexpected = set(scores) - expected_pairs
    if missing:
        raise SystemExit(
            f"! score cache is incomplete: {covered:,}/{expected:,} expected pairs; "
            f"{len(missing):,} queries need a complete top-{SCORE_DEPTH} request.\n"
            "  Run with --score to resume. Refusing to analyse a partial query set."
        )
    if unexpected:
        raise SystemExit(
            f"! score cache contains {len(unexpected):,} unexpected query/document pairs; "
            "refusing to analyse mixed experiment data."
        )

    runs = {
        BASELINE_LABEL: experiment.candidates,
        **{
            RERANK_LABELS[depth]: _reranked_runs(
                experiment,
                scores,
                depth=depth,
            )
            for depth in EVAL_DEPTHS
        },
    }
    groups: dict[int, list[int]] = {}
    for index, query in enumerate(experiment.queries):
        groups.setdefault(len(query.gold_doc_ids), []).append(index)

    scored = {
        arity: {
            label: per_query_metrics(
                [run[index] for index in indices],
                [experiment.queries[index] for index in indices],
            )
            for label, run in runs.items()
        }
        for arity, indices in sorted(groups.items())
    }

    print("\n## 1. Rerank point estimates by actual gold arity\n")
    print(
        f"   {'arity':>5} {'n':>6} {'arm':<24} {'R@1':>7} {'MRR@10':>8} "
        f"{'nDCG@10':>8} {'hit@1':>7} {'ALL@10':>7}"
    )
    for arity, per_arm in scored.items():
        for index, (label, metrics) in enumerate(per_arm.items()):
            prefix = f"   {arity:>5} {len(groups[arity]):>6,}" if index == 0 else " " * 15
            print(
                f"{prefix} {label:<24} {_mean(metrics['R@1']):>6.1%} "
                f"{_mean(metrics['MRR@10']):>8.3f} "
                f"{_mean(metrics['nDCG@10']):>8.3f} "
                f"{_mean(metrics['hit@1']):>6.1%} "
                f"{_mean(metrics['ALL@10']):>6.1%}"
            )
    print(
        "\n   Within each arity, R@1 = hit@1 / arity; inference tests hit@1 once "
        "rather than duplicating that hypothesis."
    )

    families = _families(scored, resamples=resamples)
    print("\n## 2. Paired contrasts with predeclared Holm families\n")
    for family_name, binary, rows in families:
        print(f"### {family_name} ({len(rows)} tests)\n")
        print(
            f"   {'arity':>5} {'comparator':<24} {'treatment':<16} {'metric':>8} "
            f"{'delta [95% CI]':>25} {'win/loss':>9} {'p':>10} {'p(Holm)':>10}"
        )
        for row in rows:
            scale = 100 if binary else 1
            unit = "pp" if binary else ""
            interval = (
                f"{row.delta * scale:+.2f}{unit} "
                f"[{row.ci.low * scale:+.2f}, {row.ci.high * scale:+.2f}]{unit}"
            )
            tally = f"{row.counts.wins}/{row.counts.losses}"
            raw = _fmt_p(row.raw_p, at_floor=row.raw_p_at_floor)
            adjusted = _fmt_p(
                row.adjusted_p,
                at_floor=row.adjusted_p_inherits_floor,
            )
            print(
                f"   {row.arity:>5} {row.comparator:<24} {row.treatment:<16} "
                f"{row.metric:>8} {interval:>25} {tally:>9} {raw:>10} "
                f"{adjusted:>10}{'*' if row.reject else ' '}"
            )
        test = "two-sided exact McNemar" if binary else "one-sided paired bootstrap"
        print(f"\n   {test}; Holm correction within this {len(rows)}-test family.\n")
    print("   Direction is treatment − comparator; * = Holm-adjusted family-wise p < 0.05.")
    print(
        f"   † marks a bootstrap value whose active estimate comes from 0/{resamples:,} "
        "null exceedances."
    )
    print("   It is the add-one Monte Carlo floor, not a proven '<' bound on the true tail.")


def _load_cache_for_mode(
    experiment: Experiment,
    *,
    score: bool,
    analyze: bool,
) -> tuple[RerankConfig | None, str, dict[tuple[str, str], float]]:
    if score:
        config = RerankConfig.from_env(load_env(ROOT / ".env"))
        model, endpoint = config.model, config.endpoint
        prepare_pair_score_cache(
            SCORE_CACHE,
            _provenance(experiment, model=model, endpoint=endpoint),
        )
    elif analyze:
        config = None
        model, endpoint = _cached_identity()
        validate_pair_score_cache(
            SCORE_CACHE,
            _provenance(experiment, model=model, endpoint=endpoint),
        )
    else:
        config = RerankConfig.from_env(load_env(ROOT / ".env"))
        model, endpoint = config.model, config.endpoint
        sidecar = Path(f"{SCORE_CACHE}.meta.json")
        if sidecar.exists():
            validate_pair_score_cache(
                SCORE_CACHE,
                _provenance(experiment, model=model, endpoint=endpoint),
            )
        elif SCORE_CACHE.exists() and SCORE_CACHE.stat().st_size > 0:
            raise SystemExit(f"! refusing to use rerank cache without provenance: {SCORE_CACHE}")
    return config, model, load_pair_scores(SCORE_CACHE)


def _print_status(
    experiment: Experiment,
    scores: Mapping[tuple[str, str], float],
    *,
    model: str,
) -> tuple[int, list[str]]:
    missing, covered = _coverage(experiment, scores)
    expected = len(experiment.queries) * SCORE_DEPTH
    missing_set = set(missing)
    missing_tokens = _estimate_missing_tokens(experiment, missing_set)
    request_tokens = _request_tokens(experiment, missing_set)

    print(f"queries     {len(experiment.queries):,}")
    print(f"candidates {BASELINE_LABEL}")
    print(f"score once top-{SCORE_DEPTH}: {expected:,} query/document pairs")
    print(f"cache       {covered:,}/{expected:,} expected pairs")
    print(f"missing     {len(missing):,} complete query requests")
    print(f"estimate    ~{missing_tokens:,} remaining prompt tokens (heuristic)")
    if request_tokens:
        sorted_tokens = sorted(request_tokens)
        p50 = sorted_tokens[len(sorted_tokens) // 2]
        print(
            f"request     ~{min(sorted_tokens):,} min / {p50:,} p50 / "
            f"{max(sorted_tokens):,} max prompt tokens"
        )
    print(
        f"illustration ~CNY {missing_tokens / 1e6 * ILLUSTRATIVE_CNY_PER_MTOK:.2f} "
        f"at the documented CNY {ILLUSTRATIVE_CNY_PER_MTOK:.2f}/M proxy price"
    )
    print("billing      actual One Hub relay price is unverified; check before --score")
    print(f"model       {model}")
    print(f"cache path  {SCORE_CACHE} (gitignored derived data)\n")
    return expected, missing


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", action="store_true", help="call the paid rerank API")
    mode.add_argument("--analyze", action="store_true", help="analyse a complete local cache")
    parser.add_argument(
        "--max-queries",
        type=int,
        help="with --score, process at most this many incomplete queries",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.max_queries is not None and not args.score:
        parser.error("--max-queries requires --score")
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be >= 1")

    if args.resamples < 1:
        parser.error("--resamples must be >= 1")

    experiment = _load_experiment()
    config, model, scores = _load_cache_for_mode(
        experiment,
        score=args.score,
        analyze=args.analyze,
    )
    expected, _ = _print_status(experiment, scores, model=model)

    if args.score:
        if config is None:  # pragma: no cover - kept explicit for static narrowing
            raise AssertionError("score mode requires rerank configuration")
        _score(
            experiment,
            RerankClient.create(config),
            scores,
            max_queries=args.max_queries,
        )
        remaining, covered = _coverage(experiment, scores)
        print(
            f"\ncheckpoint  {covered:,}/{expected:,} expected pairs; "
            f"{len(remaining):,} queries remain"
        )
        if not remaining:
            print("cache complete; run --analyze for paired results.")
        return 0
    if args.analyze:
        _analyse(experiment, scores, resamples=args.resamples)
        return 0

    print("dry-run: no API call. Use --score to populate or --analyze to evaluate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
