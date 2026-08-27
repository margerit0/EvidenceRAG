"""Materialize BM25, dense, RRF and rerank runs for the TiDB judgement pool.

No retrieval result is a relevance label. This script only builds the union of
what the four frozen systems surfaced; ``build_tidb_qrels.py`` performs a
separate, rank-blinded judging pass over that union.

The default is an offline plan. Paid provider calls require explicit flags:

    uv run python scripts/build_tidb_pool.py
    uv run python scripts/build_tidb_pool.py --embed       # paid query embeddings
    uv run python scripts/build_tidb_pool.py --rerank      # paid top-100 scores
    uv run python scripts/build_tidb_pool.py --embed --rerank

Query embeddings and rerank scores are append-only, resumable and gitignored.
Run artifacts bind the exact query set, indexed corpus, persisted sparse
vocabulary, model, endpoint, instruction and ordered candidate ids. A changed
contract fails closed rather than mixing two experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zhrag.eval.crud import Query
from zhrag.eval.pool import POOL_SCHEMA, PooledQuery, build_pool, pool_fingerprint
from zhrag.eval.qgen import EvalChunk
from zhrag.eval.rerank import (
    candidate_run_fingerprint,
    missing_score_queries,
    rerank_input_fingerprint,
)
from zhrag.eval.retrieval import load_embedding_matrix
from zhrag.eval.tidb_runs import (
    DENSE_LABEL,
    LEXICAL_LABEL,
    RERANK_LABEL,
    RRF_LABEL,
    TIDB_RUNS_SCHEMA,
    TiDBRuns,
    build_dense_runs,
    build_lexical_runs,
    build_reranked_runs,
    build_rrf_runs,
)
from zhrag.ingest import (
    Scope,
    chunker_fingerprint,
    document_loader,
    load_manifest,
    plan_documents,
    read_state,
    scope_fingerprint,
)
from zhrag.io_utils import exclusive_lock, read_jsonl, replace_files, write_json, write_jsonl
from zhrag.lexical import read_sparse_index
from zhrag.providers import (
    PairScore,
    RerankClient,
    RerankConfig,
    append_pair_scores,
    load_env,
    load_or_embed,
    load_pair_score_provenance,
    load_pair_scores,
    prepare_pair_score_cache,
    validate_embedding_cache,
    validate_pair_score_cache,
)
from zhrag.providers.embedding import EmbeddingClient, EmbeddingConfig

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tidb-rag-curated" / "selected_manifest.json"
DOCUMENTS = ROOT / "tidb-rag-curated" / "documents"
ARTIFACTS = ROOT / "indexes" / "tidb"
EVAL = ARTIFACTS / "eval"
QUERIES = EVAL / "queries.jsonl"
QUERY_CACHE = EVAL / "query_embeddings_4096.jsonl"
RERANK_CACHE = EVAL / "rerank_scores_top100.jsonl"
RUNS = EVAL / "runs.jsonl"
POOL = EVAL / "pool.jsonl"
POOL_REPORT = EVAL / "pool_report.json"
ARTIFACT_LOCK = ".artifacts.lock"

TARGET_TOKENS = 400
HARD_MAX_TOKENS = 600
WIDTH = 4096
RETRIEVAL_DEPTH = 100
RRF_K = 10
RERANK_REQUEST_DEPTH = 100
RERANK_APPLY_DEPTH = 50
POOL_DEPTH = 20
QUERY_PROMPT = (
    "Instruct: Given a Chinese question about TiDB, retrieve the documentation "
    "passage that answers it\nQuery:"
)
RERANK_INSTRUCTION = (
    "Given a Chinese question about TiDB, retrieve documentation passages that "
    "contain the evidence needed to answer it."
)
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
QUERY_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-query-4096-v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANK_PROFILE = "qwen3-reranker-8b-tidb-v1"
RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"


@dataclass(frozen=True, slots=True)
class Experiment:
    chunks: tuple[EvalChunk, ...]
    corpus: dict[str, str]
    query_rows: tuple[dict[str, Any], ...]
    queries: tuple[Query, ...]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TiDB relevance-judgement pool.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--pool-depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--embed", action="store_true", help="allow paid query embeddings")
    parser.add_argument("--rerank", action="store_true", help="allow paid rerank requests")
    return parser.parse_args(argv)


def _load_experiment(args: argparse.Namespace) -> Experiment:
    eval_root = args.artifacts / "eval"
    queries_path = eval_root / "queries.jsonl"
    state = read_state(args.artifacts / "state.json")
    if state is None:
        raise SystemExit("! published TiDB state is absent -- build the index first")
    expected_scope = scope_fingerprint(Scope.evergreen())
    if state.scope != expected_scope:
        raise SystemExit("! published TiDB scope is not the frozen evergreen scope")
    expected_chunker = chunker_fingerprint(
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    if state.chunker_fingerprint != expected_chunker:
        raise SystemExit("! published TiDB chunker is not the frozen evaluation chunker")
    if state.embedding_profile != DOCUMENT_EMBEDDING_PROFILE:
        raise SystemExit("! published document embedding profile is not the frozen TiDB profile")
    if not queries_path.is_file():
        raise SystemExit(
            "! final TiDB queries are absent -- build_tidb_queries.py must finish first"
        )

    planned = plan_documents(
        load_manifest(args.manifest),
        scope=Scope.evergreen(),
        load=document_loader(args.documents),
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    chunks = tuple(
        EvalChunk(
            chunk_id=chunk.chunk_id,
            source_key=plan.key,
            ordinal=chunk.ordinal,
            collection=plan.metadata["collection"],
            theme=plan.metadata["theme"],
            text=chunk.contextual_text,
            approx_tokens=chunk.approx_tokens,
        )
        for plan in planned
        for chunk in plan.chunks
    )
    corpus = {chunk.chunk_id: chunk.text for chunk in chunks}
    if set(corpus) != set(state.chunk_ids()):
        raise SystemExit("! materialized corpus does not equal the exact published chunk set")

    raw_rows = tuple(read_jsonl(queries_path))
    query_rows = tuple(_validated_query_row(row, queries_path) for row in raw_rows)
    queries = tuple(
        Query(
            query_id=row["query_id"],
            question=row["question"],
            answer=row["answer"],
            gold_doc_ids=tuple(row["gold_doc_ids"]),
            task=row["task"],
        )
        for row in query_rows
    )
    if not queries or len({query.query_id for query in queries}) != len(queries):
        raise SystemExit("! final TiDB query set is empty or has duplicate query ids")
    return Experiment(chunks=chunks, corpus=corpus, query_rows=query_rows, queries=queries)


def _validated_query_row(row: object, path: Path) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise SystemExit(f"! malformed query row in {path}: expected an object")
    for field in ("query_id", "question", "answer", "task"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"! malformed {field} in {path}: {value!r}")
    gold = row.get("gold_doc_ids")
    if not isinstance(gold, list) or len(gold) != 1 or not isinstance(gold[0], str):
        raise SystemExit("! qgen input must still carry exactly one generating chunk")
    return dict(row)


def _fingerprint_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for field in ("query_id", "question", "answer", "task"):
            value = str(row[field]).encode("utf-8")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        for doc_id in row["gold_doc_ids"]:
            encoded = str(doc_id).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _query_vectors(
    experiment: Experiment,
    args: argparse.Namespace,
) -> NDArray[np.float32] | None:
    cache = args.artifacts / "eval" / QUERY_CACHE.name
    items = {query.query_id: query.question for query in experiment.queries}
    if args.embed:
        env = load_env(ROOT / ".env")
        config = EmbeddingConfig.from_env(env)
        if config.model != EMBEDDING_MODEL:
            raise SystemExit(
                f"! {QUERY_EMBEDDING_PROFILE} requires {EMBEDDING_MODEL!r}, "
                f"but .env selects {config.model!r}"
            )
        vectors = load_or_embed(
            cache,
            items,
            EmbeddingClient(config=config),
            model=config.model,
            prompt=QUERY_PROMPT,
            batch=args.batch,
        )
        matrix = np.asarray(
            [vectors[query.query_id] for query in experiment.queries], dtype=np.float32
        )
        expected_shape = (len(experiment.queries), WIDTH)
        if matrix.shape != expected_shape:
            raise SystemExit(
                f"! query vectors have shape {matrix.shape}, expected {expected_shape}"
            )
        return matrix
    if not cache.exists():
        return None
    validate_embedding_cache(cache, model=EMBEDDING_MODEL, prompt=QUERY_PROMPT)
    matrix, missing = load_embedding_matrix(
        cache,
        [query.query_id for query in experiment.queries],
        width=WIDTH,
        require_all=False,
    )
    if missing:
        cached_count = len(experiment.queries) - len(missing)
        print(f"query embeddings: {cached_count:,} cached, {len(missing):,} missing")
        return None
    return matrix


def _doc_vectors(experiment: Experiment, args: argparse.Namespace) -> NDArray[np.float32]:
    cache = args.artifacts / "dense_cache.jsonl"
    if not cache.exists():
        raise SystemExit("! published document embedding cache is absent")
    validate_embedding_cache(cache, model=EMBEDDING_MODEL, prompt="")
    matrix, _ = load_embedding_matrix(cache, list(experiment.corpus), width=WIDTH)
    return matrix


def _rerank_provenance(
    experiment: Experiment,
    fused: Sequence[Sequence[str]],
    *,
    model: str,
    endpoint: str,
) -> dict[str, object]:
    return {
        "schema": "zhrag-tidb-rerank-pair-scores-v1",
        "model": model,
        "endpoint": endpoint,
        "profile": RERANK_PROFILE,
        "request_contract": "one query + ordered top-100 documents; top_n=100; v1",
        "instruction": RERANK_INSTRUCTION,
        "candidate_run": {
            "lexical": LEXICAL_LABEL,
            "dense": DENSE_LABEL,
            "fusion": RRF_LABEL,
            "query_embedding_profile": QUERY_EMBEDDING_PROFILE,
            "retrieval_depth": RETRIEVAL_DEPTH,
            "score_depth": RERANK_REQUEST_DEPTH,
            "query_count": len(experiment.queries),
            "fingerprint_sha256": candidate_run_fingerprint(
                experiment.queries,
                fused,
                depth=RERANK_REQUEST_DEPTH,
            ),
            "input_fingerprint_sha256": rerank_input_fingerprint(
                experiment.queries,
                fused,
                experiment.corpus,
                depth=RERANK_REQUEST_DEPTH,
            ),
        },
    }


def _expected_rerank_pairs(
    experiment: Experiment,
    fused: Sequence[Sequence[str]],
) -> set[tuple[str, str]]:
    return {
        (query.query_id, doc_id)
        for query, run in zip(experiment.queries, fused, strict=True)
        for doc_id in run[:RERANK_REQUEST_DEPTH]
    }


def _missing_rerank_queries(
    experiment: Experiment,
    fused: Sequence[Sequence[str]],
    scores: Mapping[tuple[str, str], float],
) -> list[str]:
    return missing_score_queries(
        experiment.queries,
        fused,
        scores,
        depth=RERANK_REQUEST_DEPTH,
    )


def _validated_rerank_scores(
    cache: Path,
    expected_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    scores = load_pair_scores(cache)
    unexpected = set(scores) - expected_pairs
    if unexpected:
        raise SystemExit(
            f"! rerank cache contains {len(unexpected):,} unexpected query/document pairs"
        )
    return scores


def _score_reranker(
    experiment: Experiment,
    fused: Sequence[Sequence[str]],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    cache = args.artifacts / "eval" / RERANK_CACHE.name
    config: RerankConfig | None = None
    if args.rerank:
        env = load_env(ROOT / ".env")
        config = RerankConfig.from_env(env)
        if config.model != RERANK_MODEL:
            raise SystemExit(
                f"! {RERANK_PROFILE} requires {RERANK_MODEL!r}, but .env selects {config.model!r}"
            )
        provenance = _rerank_provenance(
            experiment,
            fused,
            model=config.model,
            endpoint=config.endpoint,
        )
    else:
        recorded = load_pair_score_provenance(cache)
        model = recorded.get("model")
        endpoint = recorded.get("endpoint")
        if not isinstance(model, str) or not model or not isinstance(endpoint, str) or not endpoint:
            raise SystemExit("! rerank cache metadata has no valid model or endpoint")
        provenance = _rerank_provenance(experiment, fused, model=model, endpoint=endpoint)

    expected_pairs = _expected_rerank_pairs(experiment, fused)
    if not args.rerank:
        validate_pair_score_cache(cache, provenance)
        scores = _validated_rerank_scores(cache, expected_pairs)
        missing = _missing_rerank_queries(experiment, fused, scores)
        complete_count = len(experiment.queries) - len(missing)
        print(f"rerank scores: {complete_count:,} complete, {len(missing):,} missing")
        return scores, provenance

    assert config is not None
    lock = args.artifacts / "eval" / ".rerank.lock"
    with exclusive_lock(lock):
        # A process may have filled this append-only cache while this writer was
        # waiting for the lock. Revalidate and recompute the paid work only after
        # exclusive ownership, otherwise a stale snapshot can repeat requests.
        prepare_pair_score_cache(cache, provenance)
        scores = _validated_rerank_scores(cache, expected_pairs)
        missing = _missing_rerank_queries(experiment, fused, scores)
        complete_count = len(experiment.queries) - len(missing)
        print(f"rerank scores: {complete_count:,} complete, {len(missing):,} missing")
        selected = missing[: args.max_queries] if args.max_queries is not None else missing
        if not selected:
            return scores, provenance

        client = RerankClient.create(config)
        selected_ids = set(selected)
        done = 0
        prompt_tokens = 0
        for query, run in zip(experiment.queries, fused, strict=True):
            if query.query_id not in selected_ids:
                continue
            doc_ids = run[:RERANK_REQUEST_DEPTH]
            result = client.score(
                query.question,
                [experiment.corpus[doc_id] for doc_id in doc_ids],
                instruction=RERANK_INSTRUCTION,
            )
            rows = [
                PairScore(query.query_id, doc_id, score)
                for doc_id, score in zip(doc_ids, result.scores, strict=True)
            ]
            append_pair_scores(cache, rows)
            scores.update({(row.query_id, row.doc_id): row.score for row in rows})
            prompt_tokens += result.prompt_tokens or 0
            done += 1
            print(
                f"rerank {done:,}/{len(selected):,} | pairs {done * RERANK_REQUEST_DEPTH:,} "
                f"| API prompt tokens {prompt_tokens:,}",
                flush=True,
            )
    return scores, provenance


def _query_pairs(
    queries: Sequence[Query],
) -> list[tuple[tuple[int, Query], tuple[int, Query]]]:
    """Recover verified pairs from their shared generating chunk.

    The public query ids are hashes of each surface, so a direct query and its
    paraphrase intentionally have different suffixes. ``queries.jsonl`` is also
    sorted by that full id, which places every direct row before every paraphrase
    row. The one stable pair identity is the generating chunk recorded as the
    single qgen gold.
    """
    if len(queries) % 2:
        raise SystemExit("! TiDB query set has an odd number of rows")
    by_chunk: dict[str, dict[str, tuple[int, Query]]] = {}
    for position, query in enumerate(queries):
        if query.task not in {"direct", "paraphrase"}:
            raise SystemExit(f"! {query.query_id}: unknown query variant {query.task!r}")
        if len(query.gold_doc_ids) != 1:
            raise SystemExit(f"! {query.query_id}: qgen query needs one generating chunk")
        variants = by_chunk.setdefault(query.gold_doc_ids[0], {})
        if query.task in variants:
            raise SystemExit(f"! {query.gold_doc_ids[0]}: duplicate {query.task} query in one pair")
        variants[query.task] = (position, query)

    pairs: list[tuple[tuple[int, Query], tuple[int, Query]]] = []
    for chunk_id in sorted(by_chunk):
        variants = by_chunk[chunk_id]
        if set(variants) != {"direct", "paraphrase"}:
            missing = sorted({"direct", "paraphrase"} - set(variants))
            raise SystemExit(f"! {chunk_id}: incomplete query pair; missing={missing}")
        direct = variants["direct"]
        paraphrase = variants["paraphrase"]
        if direct[1].answer != paraphrase[1].answer:
            raise SystemExit("! direct/paraphrase pair does not share one gold and answer")
        pairs.append((direct, paraphrase))
    return pairs


def _paired_pools(
    experiment: Experiment,
    runs: TiDBRuns,
    *,
    pool_depth: int,
) -> tuple[list[PooledQuery], dict[str, int], dict[str, int]]:
    """Union all systems across both surfaces of every verified query pair."""
    contribution: dict[str, int] = {label: 0 for label in runs.runs}
    exclusive: dict[str, int] = {label: 0 for label in runs.runs}
    units: list[PooledQuery] = []
    for (direct_position, direct), (paraphrase_position, paraphrase) in _query_pairs(
        experiment.queries
    ):
        union_runs: dict[str, tuple[str, ...]] = {}
        for label in runs.runs:
            candidates, _ = build_pool(
                {
                    "direct": runs.runs[label][direct_position],
                    "paraphrase": runs.runs[label][paraphrase_position],
                },
                depth=pool_depth,
            )
            union_runs[label] = candidates
            contribution[label] += len(candidates)
        candidates, contributors = build_pool(
            union_runs,
            depth=2 * pool_depth + 1,
            required=direct.gold_doc_ids,
        )
        for labels in contributors.values():
            if len(labels) == 1:
                exclusive[labels[0]] += 1
        units.append(
            PooledQuery(
                chunk_id=direct.gold_doc_ids[0],
                query_ids=(direct.query_id, paraphrase.query_id),
                questions=(direct.question, paraphrase.question),
                answer=direct.answer,
                candidates=candidates,
                contributors=contributors,
            )
        )
    return units, contribution, exclusive


def _write_runs(
    experiment: Experiment,
    runs: TiDBRuns,
    args: argparse.Namespace,
    *,
    pool_depth: int,
    rerank_provenance: Mapping[str, object],
) -> None:
    eval_root = args.artifacts / "eval"
    run_path = eval_root / RUNS.name
    pool_path = eval_root / POOL.name
    report_path = eval_root / POOL_REPORT.name
    rows = [
        {
            "query_id": query_id,
            "runs": {label: list(run) for label, run in runs.for_query(query_id).items()},
        }
        for query_id in runs.query_ids
    ]

    pair_units, contribution, exclusive = _paired_pools(
        experiment,
        runs,
        pool_depth=pool_depth,
    )
    sizes = [len(unit.candidates) for unit in pair_units]
    pools = [
        {
            "schema": POOL_SCHEMA,
            "query_ids": list(unit.query_ids),
            "questions": list(unit.questions),
            "answer": unit.answer,
            "generating_chunk_id": unit.chunk_id,
            "candidate_doc_ids": list(unit.candidates),
            "contributors": {doc_id: list(labels) for doc_id, labels in unit.contributors.items()},
            "tasks": ["direct", "paraphrase"],
        }
        for unit in pair_units
    ]
    report = {
        "schema": TIDB_RUNS_SCHEMA,
        "queries": len(experiment.queries),
        "corpus_chunks": len(experiment.corpus),
        "query_set_fingerprint": _fingerprint_rows(experiment.query_rows),
        "runs_fingerprint": runs.fingerprint,
        "run_depth": RETRIEVAL_DEPTH,
        "pool_depth_per_system": pool_depth,
        "pairs": len(pair_units),
        "pool_fingerprint": pool_fingerprint(pair_units),
        "pool_candidates_total": sum(sizes),
        "pool_candidates_mean": sum(sizes) / len(sizes),
        "pool_candidates_min": min(sizes),
        "pool_candidates_max": max(sizes),
        "system_candidate_slots": dict(sorted(contribution.items())),
        "system_exclusive_candidates": dict(sorted(exclusive.items())),
        "profiles": {
            "query_embedding": QUERY_EMBEDDING_PROFILE,
            "rerank": RERANK_PROFILE,
            "rrf_k": RRF_K,
            "rerank_request_depth": RERANK_REQUEST_DEPTH,
            "rerank_apply_depth": RERANK_APPLY_DEPTH,
        },
        "rerank_provenance": dict(rerank_provenance),
        "semantics": "candidate pool only; retrieval rank is not a relevance judgement",
    }
    # Validate the entire in-memory bundle before the first final artifact is
    # overwritten. A malformed pair must not leave a new runs.jsonl beside an old
    # pool/report that appear to describe one complete experiment.
    staged_runs = run_path.with_suffix(run_path.suffix + ".tmp")
    staged_pool = pool_path.with_suffix(pool_path.suffix + ".tmp")
    staged_report = report_path.with_suffix(report_path.suffix + ".tmp")
    staged = (staged_runs, staged_pool, staged_report)
    try:
        write_jsonl(staged_runs, rows)
        write_jsonl(staged_pool, pools)
        write_json(staged_report, report)
        # Shared bundle lock, acquired inside the .pool.lock operation lock, so a
        # reader holding it never observes a half-replaced runs/pool/report set.
        with exclusive_lock(eval_root / ARTIFACT_LOCK):
            replace_files(
                (
                    (staged_runs, run_path),
                    (staged_pool, pool_path),
                    (staged_report, report_path),
                )
            )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    print(f"wrote {len(rows):,} aligned runs to {run_path}")
    print(f"wrote {sum(sizes):,} pair/candidate judgements to {pool_path}")


def _estimate(experiment: Experiment, args: argparse.Namespace) -> None:
    query_cache = args.artifacts / "eval" / QUERY_CACHE.name
    cached_queries = (
        {row["doc_id"] for row in read_jsonl(query_cache)} if query_cache.exists() else set()
    )
    missing = [query for query in experiment.queries if query.query_id not in cached_queries]
    print(
        f"queries {len(experiment.queries):,}; query embeddings "
        f"{len(experiment.queries) - len(missing):,} cached / {len(missing):,} missing"
    )
    print(
        "rerank requires one top-100 request per query after embeddings; "
        f"up to {len(experiment.queries):,} requests / "
        f"~{len(experiment.queries) * RERANK_REQUEST_DEPTH:,} pair scores"
    )
    if not args.embed and missing:
        print("offline plan: pass --embed to populate the paid query cache")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    if args.pool_depth < 1 or args.pool_depth > RERANK_APPLY_DEPTH:
        raise SystemExit(f"! --pool-depth must be in [1, {RERANK_APPLY_DEPTH}]")
    if args.batch < 1:
        raise SystemExit("! --batch must be positive")
    if args.max_queries is not None and args.max_queries < 1:
        raise SystemExit("! --max-queries must be positive")

    experiment = _load_experiment(args)
    _estimate(experiment, args)
    query_matrix = _query_vectors(experiment, args)
    if query_matrix is None:
        return 0

    doc_ids = list(experiment.corpus)
    doc_matrix = _doc_vectors(experiment, args)
    sparse = read_sparse_index(args.artifacts / "sparse_index.json")
    state = read_state(args.artifacts / "state.json")
    assert state is not None
    if sparse.fingerprint != state.sparse_fingerprint:
        raise SystemExit("! sparse vocabulary fingerprint differs from published state")

    questions = [query.question for query in experiment.queries]
    lexical = build_lexical_runs(sparse, experiment.corpus, questions, depth=RETRIEVAL_DEPTH)
    dense = build_dense_runs(query_matrix, doc_matrix, doc_ids, depth=RETRIEVAL_DEPTH)
    fused = build_rrf_runs(lexical, dense, depth=RETRIEVAL_DEPTH, rrf_k=RRF_K)
    scores, provenance = _score_reranker(experiment, fused, args)
    complete = all(
        (query.query_id, doc_id) in scores
        for query, run in zip(experiment.queries, fused, strict=True)
        for doc_id in run[:RERANK_REQUEST_DEPTH]
    )
    if not complete:
        print("pool not written: rerank cache is incomplete; pass --rerank to resume")
        return 0

    reranked = build_reranked_runs(
        fused,
        [query.query_id for query in experiment.queries],
        scores,
        request_depth=RERANK_REQUEST_DEPTH,
        apply_depth=RERANK_APPLY_DEPTH,
    )
    runs = TiDBRuns(
        query_ids=tuple(query.query_id for query in experiment.queries),
        runs={
            LEXICAL_LABEL: lexical,
            DENSE_LABEL: dense,
            RRF_LABEL: fused,
            RERANK_LABEL: reranked,
        },
    )
    eval_root = args.artifacts / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(eval_root / ".pool.lock"):
        _write_runs(
            experiment,
            runs,
            args,
            pool_depth=args.pool_depth,
            rerank_provenance=provenance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
