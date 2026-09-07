"""Serve the published TiDB index through the optional FastAPI layer.

    uv run --extra service --extra milvus python scripts/serve.py
    uv run --extra service --extra milvus python scripts/serve.py --no-rerank
    uv run --extra service --extra milvus python scripts/serve.py \
        --no-rerank --query-cache indexes/tidb/eval/query_embeddings_4096.jsonl

This is the composition root. Importing it does not read ``.env``, open Milvus,
or make a provider request; those side effects happen only in :func:`main` after
all local artifacts and profile contracts have been validated.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from zhrag.answering import Answerer, AnswerSettings
from zhrag.embedding_contract import validate_embedding_cache
from zhrag.ingest import IngestState, read_state
from zhrag.io_utils import read_jsonl
from zhrag.lexical import SparseIndex, read_sparse_index
from zhrag.retrieval.online import OnlineRetriever, OnlineSettings, PassageReranker
from zhrag.service.app import ServiceInfo, create_app
from zhrag.store import MilvusConfig, MilvusStore

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
DENSE_WIDTH = 4096
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
QUERY_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-query-4096-v1"
RERANK_PROFILE = "qwen3-reranker-8b-tidb-v1"
NO_RERANK_PROFILE = "disabled-identity-fused-order-v1"
LIVE_PROFILE = "tidb-docs-exact-rrf10-rerank100to50-v1"
NO_RERANK_SERVICE_PROFILE = "tidb-docs-exact-rrf10-no-rerank-v1"
CACHED_NO_RERANK_SERVICE_PROFILE = "tidb-docs-exact-rrf10-cached-query-no-rerank-v1"
CACHED_QUERY_EMBEDDING_PROFILE = "cached-qwen3-embedding-8b-tidb-query-4096-v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
QUERY_PROMPT = (
    "Instruct: Given a Chinese question about TiDB, retrieve the documentation "
    "passage that answers it\nQuery:"
)
RERANK_INSTRUCTION = (
    "Given a Chinese question about TiDB, retrieve documentation passages that "
    "contain the evidence needed to answer it."
)


class _StoreLike(Protocol):
    @property
    def dense_dimensions(self) -> int: ...

    def ensure_collection(self) -> None: ...

    def alias_target(self, alias: str) -> str | None: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CachedQueryEncoder:
    """Serve only query texts already present in a provenance-guarded cache."""

    vectors: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ValueError("query embedding cache is empty")

    def encode(self, query: str) -> tuple[float, ...]:
        try:
            return self.vectors[query]
        except KeyError as exc:
            raise ValueError("query is absent from the cache-backed service profile") from exc


@dataclass(slots=True)
class _IdentityReranker:
    """Preserve fused order without sending passage text to a rerank provider."""

    def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        return tuple(float(-index) for index in range(len(documents)))


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--uri", default=str(ARTIFACTS / "milvus.db"))
    parser.add_argument("--alias", default="tidb_chunks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--query-cache",
        type=Path,
        help=(
            "serve only exact queries in this local embedding cache; implies "
            "--no-rerank and makes the profile provider-free"
        ),
    )
    parser.add_argument(
        "--query-fixture",
        type=Path,
        help="paired local query JSONL for --query-cache (default: sibling queries.jsonl)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="preserve fused order and do not create or call a rerank client",
    )
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help="allow paid chat calls through /api/ask; requires LLM_* configuration",
    )
    parser.add_argument("--generation-timeout", type=float, default=60.0)
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=15,
        help="retries after the first chat attempt (0-15); wait 5, 10, ... seconds",
    )
    parser.add_argument("--generation-max-tokens", type=int, default=2048)
    parser.add_argument("--context-tokens", type=int, default=12_000)
    parser.add_argument("--context-passages", type=int, default=6)
    parser.add_argument(
        "--generation-reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
    )
    return parser.parse_args(argv)


def _build_answerer(args: argparse.Namespace) -> Answerer | None:
    if not args.enable_generation:
        return None
    if args.query_cache is not None:
        raise ValueError("generation is incompatible with query-cache")
    from zhrag.providers.answering import ChatAnswerGenerator  # noqa: PLC0415
    from zhrag.providers.chat import ChatConfig  # noqa: PLC0415
    from zhrag.providers.embedding import load_env  # noqa: PLC0415

    return Answerer(
        ChatAnswerGenerator(
            ChatConfig.from_env(load_env(args.env)),
            max_output_tokens=args.generation_max_tokens,
            timeout_seconds=args.generation_timeout,
            reasoning_effort=args.generation_reasoning_effort,
            max_retries=args.generation_retries,
        ),
        settings=AnswerSettings(
            max_passages=args.context_passages,
            max_prompt_tokens=args.context_tokens,
        ),
    )


def _load_published_artifacts(artifacts: Path) -> tuple[IngestState, SparseIndex]:
    state_path = artifacts / "state.json"
    vocabulary_path = artifacts / "sparse_index.json"
    missing = [path.name for path in (state_path, vocabulary_path) if not path.is_file()]
    if missing:
        raise ValueError(
            "published TiDB artifacts are missing: "
            f"{', '.join(missing)}; run scripts/build_index.py --embed --publish"
        )
    state = read_state(state_path)
    if state is None:
        raise ValueError("the published TiDB state is absent")
    index = read_sparse_index(vocabulary_path)
    if index.fingerprint != state.sparse_fingerprint:
        raise ValueError("sparse vocabulary does not match the published state")
    if state.embedding_profile != DOCUMENT_EMBEDDING_PROFILE:
        raise ValueError("published document embedding profile is incompatible with this service")
    if not state.collection_name:
        raise ValueError("published state has no physical collection name")
    if len(state.chunk_ids()) != index.document_count:
        raise ValueError("published state and sparse index disagree on chunk count")
    return state, index


def _settings(
    *,
    no_rerank: bool,
    cached_query: bool = False,
    output_limit: int = 10,
) -> OnlineSettings:
    if cached_query and not no_rerank:
        raise ValueError("cache-backed query embedding requires no-rerank")
    profile_name = LIVE_PROFILE
    embedding_profile = QUERY_EMBEDDING_PROFILE
    rerank_profile = RERANK_PROFILE
    if cached_query:
        profile_name = CACHED_NO_RERANK_SERVICE_PROFILE
        embedding_profile = CACHED_QUERY_EMBEDDING_PROFILE
        rerank_profile = NO_RERANK_PROFILE
    elif no_rerank:
        profile_name = NO_RERANK_SERVICE_PROFILE
        rerank_profile = NO_RERANK_PROFILE
    return OnlineSettings.product(
        profile_name=profile_name,
        embedding_profile=embedding_profile,
        rerank_profile=rerank_profile,
        dense_dimensions=DENSE_WIDTH,
        output_limit=output_limit,
        rerank_enabled=not no_rerank,
    )


def _load_cached_queries(  # noqa: PLR0912
    cache: Path,
    queries_path: Path,
) -> _CachedQueryEncoder:
    if not cache.is_file():
        raise ValueError("query embedding cache is missing")
    if not queries_path.is_file():
        raise ValueError("query fixture is missing")
    try:
        validate_embedding_cache(cache, model=EMBEDDING_MODEL, prompt=QUERY_PROMPT)
    except SystemExit as exc:
        raise ValueError("query embedding cache provenance is incompatible") from exc

    id_to_query: dict[str, str] = {}
    for row in read_jsonl(queries_path):
        query_id = row.get("query_id")
        question = row.get("question")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query fixture contains an invalid query_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("query fixture contains an invalid question")
        if query_id in id_to_query:
            raise ValueError("query fixture contains duplicate query ids")
        id_to_query[query_id] = question.strip()

    vectors_by_id: dict[str, tuple[float, ...]] = {}
    for row in read_jsonl(cache):
        query_id = row.get("doc_id")
        raw_vector = row.get("embedding")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query embedding cache contains an invalid id")
        if query_id not in id_to_query:
            continue
        if not isinstance(raw_vector, list) or len(raw_vector) != DENSE_WIDTH:
            raise ValueError("query embedding cache contains an invalid vector width")
        vector: list[float] = []
        for raw_value in raw_vector:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("query embedding cache contains a non-numeric value")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("query embedding cache contains a non-finite value")
            vector.append(value)
        vectors_by_id[query_id] = tuple(vector)
    missing = set(id_to_query) - set(vectors_by_id)
    if missing:
        raise ValueError("query embedding cache is incomplete for the fixture")
    if len(set(id_to_query.values())) != len(id_to_query):
        raise ValueError("query fixture contains duplicate question text")
    return _CachedQueryEncoder(
        {question: vectors_by_id[query_id] for query_id, question in id_to_query.items()}
    )


def _verify_store(
    store: _StoreLike,
    *,
    alias: str,
    state: IngestState,
) -> None:
    if store.dense_dimensions != DENSE_WIDTH:
        raise ValueError("store dense width does not match the service profile")
    target = store.alias_target(alias)
    if target != state.collection_name:
        raise ValueError("Milvus alias target does not match the published state")
    store.ensure_collection()
    if store.count() != len(state.chunk_ids()):
        raise ValueError("Milvus row count does not match the published state")


def _build_retriever(
    args: argparse.Namespace,
    *,
    state: IngestState,
    index: SparseIndex,
    store: MilvusStore,
) -> tuple[OnlineRetriever, ServiceInfo]:
    # Provider imports stay below the artifact and Milvus validation boundary so
    # clean module import remains side-effect-free and a broken local publication
    # fails before credentials are read.
    from zhrag.retrieval.adapters import (  # noqa: PLC0415 - provider boundary
        SparseQueryAdapter,
    )

    _verify_store(store, alias=args.alias, state=state)
    cached_query = args.query_cache is not None
    if cached_query:
        fixture = args.query_fixture or args.query_cache.parent / "queries.jsonl"
        embedding: Any = _load_cached_queries(
            args.query_cache,
            fixture,
        )
    else:
        from zhrag.providers.embedding import (  # noqa: PLC0415 - composition boundary
            EmbeddingClient,
            EmbeddingConfig,
            load_env,
        )
        from zhrag.retrieval.adapters import (  # noqa: PLC0415 - provider boundary
            DenseQueryAdapter,
        )

        if not args.env.is_file():
            raise ValueError(".env is missing")
        env = load_env(args.env)
        embedding = DenseQueryAdapter(
            EmbeddingClient(EmbeddingConfig.from_env(env), log=lambda _message: None),
            prompt=QUERY_PROMPT,
            dimensions=DENSE_WIDTH,
        )
    settings = _settings(no_rerank=args.no_rerank, cached_query=cached_query)
    reranker: PassageReranker
    if args.no_rerank:
        reranker = _IdentityReranker()
    else:
        from zhrag.providers.rerank import (  # noqa: PLC0415 - composition boundary
            RerankClient,
            RerankConfig,
        )
        from zhrag.retrieval.adapters import (  # noqa: PLC0415 - provider boundary
            RerankAdapter,
        )

        reranker = RerankAdapter(
            RerankClient.create(RerankConfig.from_env(env)),
            instruction=RERANK_INSTRUCTION,
        )
    retriever = OnlineRetriever(
        settings=settings,
        dense_encoder=embedding,
        sparse_encoder=SparseQueryAdapter(index),
        store=store,
        reranker=reranker,
    )
    return retriever, ServiceInfo(
        profile_name=settings.profile_name,
        embedding_profile=settings.embedding_profile,
        rerank_profile=settings.rerank_profile,
        rerank_enabled=settings.rerank_enabled,
    )


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    try:
        if not 1 <= args.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if args.max_concurrency < 1:
            raise ValueError("max-concurrency must be positive")
        if args.query_cache is not None and not args.no_rerank:
            raise ValueError("query-cache requires --no-rerank")
        if args.query_fixture is not None and args.query_cache is None:
            raise ValueError("query-fixture requires --query-cache")
        if args.enable_generation and args.query_cache is not None:
            raise ValueError("generation is incompatible with query-cache")
        if not math.isfinite(args.generation_timeout) or not 0 < args.generation_timeout <= 300:
            raise ValueError("generation-timeout must be in (0, 300]")
        if not 0 <= args.generation_retries <= 15:
            raise ValueError("generation-retries must be in [0, 15]")
        if not 1 <= args.generation_max_tokens <= 8_192:
            raise ValueError("generation-max-tokens must be in [1, 8192]")
        AnswerSettings(max_passages=args.context_passages, max_prompt_tokens=args.context_tokens)
        state, index = _load_published_artifacts(args.artifacts)
        store = MilvusStore(
            MilvusConfig(
                uri=args.uri,
                collection_name=args.alias,
                dense_dimensions=DENSE_WIDTH,
            )
        )
        try:
            retriever, info = _build_retriever(
                args,
                state=state,
                index=index,
                store=store,
            )
            app = create_app(
                retriever,
                info=info,
                max_concurrency=args.max_concurrency,
                answerer=_build_answerer(args),
                published_index_identity=(f"{state.collection_name}:{state.sparse_fingerprint}"),
            )
            import uvicorn  # noqa: PLC0415 - optional service dependency

            mode = (
                "cached-no-rerank"
                if args.query_cache is not None
                else "no-rerank"
                if args.no_rerank
                else "live-rerank"
            )
            print(
                f"serving {info.profile_name} ({mode}) at "
                f"http://{args.host}:{args.port} with one worker"
            )
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=1,
                log_config=None,
                access_log=False,
            )
        finally:
            store.close()
    except (OSError, RuntimeError, ValueError) as exc:
        # Composition errors may contain local paths or vendor payload fragments.
        # The console receives only the exception class and a curated public hint;
        # request handlers use their own fixed machine-readable errors.
        print(f"! service startup failed ({type(exc).__name__})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
