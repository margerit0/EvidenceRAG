"""Query the published TiDB index end to end.

    uv run python scripts/query_index.py "如何用 BR 做全量备份？"
    uv run python scripts/query_index.py "..." --no-rerank      # 只跑检索两臂 + RRF

Composition root for the online path: an embedding client, the persisted sparse
vocabulary, a Milvus collection reached through its alias, and a reranker. Each
one is adapted to a narrow Protocol here, so ``zhrag.retrieval.online`` itself
never sees HTTP, ``.env``, or pymilvus.

The instructions below are a *product* profile. They are named separately and are
not the frozen CRUD-RAG news prompts: the rerank gains measured on that benchmark
were measured with different instructions on a different corpus, and nothing here
claims they transfer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhrag.io_utils import read_json
from zhrag.lexical import read_sparse_index
from zhrag.providers.embedding import EmbeddingClient, EmbeddingConfig, load_env
from zhrag.providers.rerank import RerankClient, RerankConfig
from zhrag.retrieval import (
    DenseQueryAdapter,
    OnlineRetriever,
    OnlineSettings,
    RerankAdapter,
    SparseQueryAdapter,
)
from zhrag.store import MilvusConfig, MilvusStore

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
DENSE_WIDTH = 4096

QUERY_PROMPT = (
    "Instruct: Given a Chinese question about TiDB, retrieve the documentation "
    "passage that answers it\nQuery:"
)
RERANK_INSTRUCTION = (
    "Given a Chinese question about TiDB, retrieve documentation passages that "
    "contain the evidence needed to answer it."
)
PROFILE = OnlineSettings.product(
    profile_name="tidb-docs-exact-rrf10-rerank100to50-v1",
    embedding_profile="qwen3-embedding-8b-tidb-query-4096-v1",
    rerank_profile="qwen3-reranker-8b-tidb-v1",
    dense_dimensions=DENSE_WIDTH,
)


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the published TiDB index.")
    parser.add_argument("query")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--uri", default=str(ARTIFACTS / "milvus.db"))
    parser.add_argument("--alias", default="tidb_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip the reranker; report the fused retrieval order instead",
    )
    return parser.parse_args(argv)


class _IdentityReranker:
    """Score the fused order as-is, so ``--no-rerank`` sends no paid request."""

    def score(self, query: str, documents: list[str]) -> tuple[float, ...]:
        return tuple(float(-index) for index in range(len(documents)))


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)

    state_path = args.artifacts / "state.json"
    vocabulary_path = args.artifacts / "sparse_index.json"
    for path in (state_path, vocabulary_path):
        if not path.is_file():
            print(f"! {path} not found -- run scripts/build_index.py --embed --publish first")
            return 1

    state = read_json(state_path)
    index = read_sparse_index(vocabulary_path)
    if index.fingerprint != state.get("sparse_fingerprint"):
        # A vocabulary from another build assigns different term indexes, so the
        # inner product would score against whatever terms sit at those
        # positions: a silently wrong ranking rather than an error.
        print("! sparse vocabulary does not match the published state -- rebuild the index")
        return 1

    env = load_env(ROOT / ".env")
    settings = OnlineSettings.product(
        profile_name=PROFILE.profile_name,
        embedding_profile=PROFILE.embedding_profile,
        rerank_profile=PROFILE.rerank_profile,
        dense_dimensions=DENSE_WIDTH,
        output_limit=args.top_k,
    )
    reranker: object
    if args.no_rerank:
        reranker = _IdentityReranker()
    else:
        reranker = RerankAdapter(
            RerankClient.create(RerankConfig.from_env(env)),
            instruction=RERANK_INSTRUCTION,
        )

    store = MilvusStore(
        MilvusConfig(
            uri=args.uri,
            collection_name=args.alias,
            dense_dimensions=DENSE_WIDTH,
        )
    )
    try:
        store.ensure_collection()
        retriever = OnlineRetriever(
            settings=settings,
            dense_encoder=DenseQueryAdapter(
                EmbeddingClient(EmbeddingConfig.from_env(env), log=lambda _message: None),
                prompt=QUERY_PROMPT,
                dimensions=DENSE_WIDTH,
            ),
            sparse_encoder=SparseQueryAdapter(index),
            store=store,
            reranker=reranker,  # type: ignore[arg-type]
        )
        result = retriever.retrieve(args.query)
    finally:
        store.close()

    print(f"query: {result.query}")
    print(f"profile: {result.profile_name}")
    dense_ids = {hit.doc_id for hit in result.dense_hits}
    sparse_ids = {hit.doc_id for hit in result.sparse_hits}
    print(
        f"arms: dense {len(dense_ids)}  sparse {len(sparse_ids)}  "
        f"both {len(dense_ids & sparse_ids)}  fused {len(result.fused_candidates)}"
    )
    timings = result.timings
    print(
        "timing(ms): "
        f"encode {1000 * (timings.dense_encode_seconds + timings.sparse_encode_seconds):.0f}  "
        f"search {1000 * (timings.dense_search_seconds + timings.sparse_search_seconds):.0f}  "
        f"fuse {1000 * timings.fusion_seconds:.1f}  "
        f"fetch {1000 * timings.fetch_seconds:.0f}  "
        f"rerank {1000 * timings.rerank_seconds:.0f}  "
        f"total {1000 * timings.total_seconds:.0f}"
    )
    for row in result.passages:
        passage = row.passage
        heading = passage.metadata.get("path", "?")
        first_line = passage.text.splitlines()[0] if passage.text else ""
        print(
            f"\n#{row.rank}  fused_rank={row.fused_rank}  score={row.rerank_score:.4f}\n"
            f"  {heading}\n"
            f"  {first_line[:110]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
