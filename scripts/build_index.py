"""Build a versioned TiDB index into a shadow collection, then switch the alias.

    uv run python scripts/build_index.py --dry-run
    uv run python scripts/build_index.py --embed --publish

The build is deliberately staged so that an interrupted or failed run can never
leave a half-written collection serving traffic:

1. plan  -- read the manifest, verify bytes, chunk, and compute stable ids.
2. embed -- reuse cached dense vectors by chunk id and embed only what is new.
             This is the only paid step, and it requires ``--embed``.
3. write -- upsert every desired row into a *new* versioned collection.
4. check -- verify the row count and a sample round trip in that collection.
5. switch -- point the alias at it and only then persist the success state.

Sparse vectors are rebuilt for the whole collection on every build, because BM25
statistics are corpus-global: one added chunk changes IDF and average length for
every other row. Reusing them per-document would silently mix two vocabularies.

Nothing written here belongs in Git: chunks, vectors, vocabulary, state, and the
Milvus database all land under gitignored paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhrag.ingest import (
    DocumentPlan,
    IngestState,
    Scope,
    chunker_fingerprint,
    diff_documents,
    document_loader,
    load_manifest,
    plan_documents,
    read_state,
    reusable_chunk_ids,
    scope_fingerprint,
    write_state,
)
from zhrag.lexical import build_sparse_index, write_sparse_index
from zhrag.lexical.sparse import SparseBuild
from zhrag.providers.embedding import EmbeddingClient, EmbeddingConfig, load_env, load_or_embed
from zhrag.store import ChunkRecord, MilvusConfig, MilvusStore

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tidb-rag-curated" / "selected_manifest.json"
DOCUMENTS = ROOT / "tidb-rag-curated" / "documents"
ARTIFACTS = ROOT / "indexes" / "tidb"
TARGET_TOKENS = 400
HARD_MAX_TOKENS = 600
#: Named separately from the frozen news benchmark profile. The instruction and
#: the corpus both differ, so rerank/retrieval numbers measured on CRUD-RAG do
#: not transfer to this index and must not be reported as if they did.
EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
DENSE_WIDTH = 4096


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--uri", default=str(ARTIFACTS / "milvus.db"))
    parser.add_argument("--alias", default="tidb_chunks")
    parser.add_argument("--version", default="v1", help="shadow collection suffix")
    parser.add_argument(
        "--include-releases",
        action="store_true",
        help="also index temporal_releases (excluded by default: release notes date)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and report only; no network, no database, no state write",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="allow paid embedding requests for chunks missing from the cache",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="switch the alias and persist the state after verification",
    )
    return parser.parse_args(argv)


def _scope(include_releases: bool) -> Scope:
    return Scope(exclude_collections=frozenset()) if include_releases else Scope.evergreen()


def _report_plan(planned: tuple[DocumentPlan, ...], delta_counts: dict[str, int]) -> None:
    chunks = sum(len(plan.chunks) for plan in planned)
    tokens = [chunk.approx_tokens for plan in planned for chunk in plan.chunks]
    tokens.sort()
    print(f"documents: {len(planned):,}   chunks: {chunks:,}")
    if tokens:
        p50 = tokens[len(tokens) // 2]
        p90 = tokens[min(len(tokens) - 1, int(len(tokens) * 0.9))]
        print(f"tokens/chunk: p50={p50:,}  p90={p90:,}  max={tokens[-1]:,}")
    print("delta: " + "  ".join(f"{name}={count:,}" for name, count in delta_counts.items()))


def _records(
    planned: tuple[DocumentPlan, ...],
    vectors: dict[str, list[float]],
    sparse: SparseBuild,
) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            doc_id=chunk.chunk_id,
            text=chunk.contextual_text,
            dense=tuple(vectors[chunk.chunk_id]),
            sparse=sparse.vector_for(chunk.chunk_id),
            source_key=plan.key,
            document_sha256=plan.document_sha256,
            ordinal=chunk.ordinal,
            metadata={
                **dict(plan.metadata),
                "heading_path": " > ".join(chunk.heading_path),
                "approx_tokens": chunk.approx_tokens,
            },
        )
        for plan in planned
        for chunk in plan.chunks
    ]


def _write_and_verify(
    store: MilvusStore,
    records: list[ChunkRecord],
    *,
    collection_name: str,
) -> None:
    """Fill the shadow collection and prove it is complete before publication."""
    store.ensure_collection()
    written = 0
    for start in range(0, len(records), 128):
        written += store.upsert(records[start : start + 128])
    print(f"wrote {written:,} rows into {collection_name}")

    rows = store.count()
    if rows != len(records):
        raise SystemExit(f"! {collection_name} holds {rows:,} rows, expected {len(records):,}")
    sample = [record.doc_id for record in records[:5]]
    if [passage.doc_id for passage in store.fetch(sample)] != sample:
        raise SystemExit("! sample round trip did not return the requested rows")


def _publish(
    args: argparse.Namespace,
    planned: tuple[DocumentPlan, ...],
    sparse: SparseBuild,
    scope: Scope,
    collection_name: str,
) -> None:
    write_state(
        args.artifacts / "state.json",
        IngestState(
            scope=scope_fingerprint(scope),
            chunker_fingerprint=chunker_fingerprint(
                target_tokens=TARGET_TOKENS,
                hard_max_tokens=HARD_MAX_TOKENS,
            ),
            embedding_profile=EMBEDDING_PROFILE,
            sparse_fingerprint=sparse.index.fingerprint,
            collection_name=collection_name,
            documents={
                plan.key: {
                    "document_sha256": plan.document_sha256,
                    "metadata_fingerprint": plan.metadata_fingerprint,
                    "chunk_ids": list(plan.chunk_ids),
                }
                for plan in planned
            },
        ),
    )
    print(f"published {args.alias} -> {collection_name}")


def _build(
    args: argparse.Namespace,
    planned: tuple[DocumentPlan, ...],
    corpus: dict[str, str],
    *,
    sparse: SparseBuild,
    scope: Scope,
    collection_name: str,
) -> int:
    env = load_env(ROOT / ".env")
    client = EmbeddingClient(EmbeddingConfig.from_env(env))
    vectors = load_or_embed(
        args.artifacts / "dense_cache.jsonl",
        corpus,
        client,
        model=client.config.model,
        log=print,
    )
    records = _records(planned, vectors, sparse)

    store = MilvusStore(
        MilvusConfig(
            uri=args.uri,
            collection_name=collection_name,
            dense_dimensions=DENSE_WIDTH,
        )
    )
    try:
        _write_and_verify(store, records, collection_name=collection_name)
        if not args.publish:
            print(f"built {collection_name}; re-run with --publish to switch {args.alias!r}")
            return 0
        store.activate_alias(args.alias)
    finally:
        store.close()

    # The vocabulary is written only on the publishing path: a query encoder must
    # never be able to load a vocabulary that no live collection was built with.
    write_sparse_index(args.artifacts / "sparse_index.json", sparse.index)
    _publish(args, planned, sparse, scope, collection_name)
    return 0


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)

    for path, label in ((args.manifest, "manifest"), (args.documents, "documents")):
        if not path.exists():
            print(f"! {path} ({label}) not found -- run tidb-rag-curated/download_curated.ps1")
            return 1

    scope = _scope(args.include_releases)
    manifest = load_manifest(args.manifest)
    planned = plan_documents(
        manifest,
        scope=scope,
        load=document_loader(args.documents),
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    previous = read_state(args.artifacts / "state.json")
    delta = diff_documents(planned, previous, scope=scope, manifest=manifest)
    _report_plan(planned, dict(delta.counts))

    corpus = {chunk.chunk_id: chunk.contextual_text for plan in planned for chunk in plan.chunks}
    total = sum(len(plan.chunks) for plan in planned)
    if not corpus:
        print("! the selected scope produced no chunks")
        return 1
    if len(corpus) != total:
        print(f"! {total - len(corpus):,} chunk ids collide across the planned corpus")
        return 1

    sparse = build_sparse_index(corpus)
    print(
        f"sparse vocabulary: {len(sparse.index.terms):,} bigrams ({sparse.index.fingerprint[:12]})"
    )
    reusable = reusable_chunk_ids(planned, previous, embedding_profile=EMBEDDING_PROFILE)
    print(f"dense vectors reusable by chunk id: {len(reusable):,} of {len(corpus):,}")

    collection_name = f"tidb_chunks_{args.version}"
    if args.dry_run:
        print(f"dry run: would build {collection_name} and leave the alias untouched")
        return 0
    if not args.embed:
        print("! refusing to embed without --embed (this step sends paid requests)")
        return 1
    return _build(
        args,
        planned,
        corpus,
        sparse=sparse,
        scope=scope,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
