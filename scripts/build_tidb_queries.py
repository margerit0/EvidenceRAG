"""Build the TiDB retrieval evaluation set from the published index's chunks.

    uv run python scripts/build_tidb_queries.py                    # plan only
    uv run python scripts/build_tidb_queries.py --generate         # paid

Stages, in order, so that an interrupted run resumes without re-paying:

1. plan     -- re-chunk the manifest exactly as ``build_index.py`` did and prove
               the chunk ids match the published ``state.json``.
2. sample   -- draw a theme-stratified sample deterministically from a seed.
3. generate -- one call per chunk producing a direct question, a paraphrase
               twin, a short grounded answer, and a type label.
4. dedupe   -- drop near-duplicate questions *before* paying to verify them.
5. verify   -- one call per surviving pair, on five quality axes plus same intent.
6. write    -- queries.jsonl plus a report of every drop and the leakage stats.

The gold for a query is the chunk it was written from. That is a single-evidence,
incomplete judgement: other chunks may answer the same question and will be
counted as misses until ``judge_tidb_pool.py`` pools and grades them. Numbers
from this file alone are therefore a floor, not an estimate, and the report says
so rather than leaving a reader to assume otherwise.

Nothing written here belongs in Git. The questions are derived from CC BY-SA 3.0
documentation and are Adaptations of it; they stay under the gitignored index
directory with the vectors and the vocabulary.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.qgen import (
    GENERATION_CACHE_KEY_SCHEMA,
    GENERATION_INSTRUCTIONS,
    QGEN_SCHEMA,
    VERIFICATION_CACHE_KEY_SCHEMA,
    VERIFICATION_INSTRUCTIONS,
    EvalChunk,
    GeneratedQuery,
    bigram_containment,
    dedupe_questions,
    generation_instructions_fingerprint,
    generation_prompt,
    instructions_fingerprint,
    parse_generation,
    parse_verdict,
    stratified_sample,
    theme_counts,
    verification_cache_id,
    verification_instructions_fingerprint,
    verification_prompt,
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
from zhrag.io_utils import (
    append_jsonl,
    exclusive_lock,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from zhrag.providers.cache import prepare_cache_sidecar
from zhrag.providers.chat import ChatClient, ChatConfig
from zhrag.providers.embedding import load_env

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tidb-rag-curated" / "selected_manifest.json"
DOCUMENTS = ROOT / "tidb-rag-curated" / "documents"
ARTIFACTS = ROOT / "indexes" / "tidb"
TARGET_TOKENS = 400
HARD_MAX_TOKENS = 600
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = "zhrag-tidb-eval-2026-08"
#: No verifier override by default: the configured ``LLM_MODEL_NAME`` is the
#: only model the operator has explicitly selected. A second model remains an
#: opt-in via ``--verifier-model``; using the configured model for both stages is
#: reported as self-agreement rather than being presented as independent review.
DEFAULT_VERIFIER: str | None = None
FLUSH_EVERY = 20


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TiDB evaluation query set.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="chunks to sample",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--verifier-model", default=DEFAULT_VERIFIER)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many sampled chunks (for a cheap trial run)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="allow paid chat requests; without it the run plans and reports only",
    )
    return parser.parse_args(argv)


def _load_chunks(manifest_path: Path, documents: Path) -> tuple[EvalChunk, ...]:
    """Re-derive the indexed chunks, with the strata a sample balances over."""
    manifest = load_manifest(manifest_path)
    planned = plan_documents(
        manifest,
        scope=Scope.evergreen(),
        load=document_loader(documents),
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    return tuple(
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


def _verify_against_index(chunks: Sequence[EvalChunk], artifacts: Path) -> None:
    """Refuse to build golds that the live collection does not actually hold.

    A chunker or scope change would still produce a plausible-looking query set,
    but every gold id would be absent from the index and every query would score
    zero -- a failure that reads like "retrieval is broken" rather than "the
    benchmark was built against a different corpus".
    """
    state = read_state(artifacts / "state.json")
    if state is None:
        raise SystemExit(
            f"! no published index state under {artifacts}\n"
            "  run scripts/build_index.py --embed --publish first"
        )
    expected_scope = scope_fingerprint(Scope.evergreen())
    if state.scope != expected_scope:
        raise SystemExit(
            "! published index scope drift: qgen requires the exact evergreen collection "
            f"({state.scope[:12]} != {expected_scope[:12]})"
        )
    expected = chunker_fingerprint(target_tokens=TARGET_TOKENS, hard_max_tokens=HARD_MAX_TOKENS)
    if state.chunker_fingerprint != expected:
        raise SystemExit(
            "! chunker fingerprint drift: this script would generate golds for chunks\n"
            f"  the published index does not contain ({state.chunker_fingerprint[:12]} != "
            f"{expected[:12]})"
        )
    published = state.chunk_ids()
    planned = frozenset(chunk.chunk_id for chunk in chunks)
    missing = planned - published
    extra = published - planned
    if missing or extra:
        raise SystemExit(
            "! published chunk set differs from the exact qgen corpus: "
            f"{len(missing):,} missing, {len(extra):,} extra"
        )


@dataclass(frozen=True, slots=True)
class _Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls_with_token_counts: int = 0

    def plus(
        self,
        prompt: int | None,
        completion: int | None,
        reasoning: int | None,
    ) -> _Usage:
        counts = (prompt, completion, reasoning)
        complete = all(value is not None for value in counts)
        return _Usage(
            self.calls + 1,
            self.prompt_tokens + (prompt or 0),
            self.completion_tokens + (completion or 0),
            self.reasoning_tokens + (reasoning or 0),
            self.calls_with_token_counts + int(complete),
        )

    def combine(self, other: _Usage) -> _Usage:
        return _Usage(
            self.calls + other.calls,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.calls_with_token_counts + other.calls_with_token_counts,
        )

    def as_dict(self) -> dict[str, int | str]:
        coverage = (
            "complete"
            if self.calls_with_token_counts == self.calls
            else f"partial: {self.calls_with_token_counts}/{self.calls} calls carry token counts"
        )
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "token_count_coverage": coverage,
        }


def _required_string(row: object, name: str, path: Path, lineno: int) -> str:
    if not isinstance(row, dict):
        raise SystemExit(f"! malformed cache row in {path}:{lineno}: expected an object")
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"! malformed {name} in {path}:{lineno}: {value!r}")
    return value


def _optional_count(
    row: dict[str, Any],
    name: str,
    path: Path,
    lineno: int,
) -> tuple[int, bool]:
    if name not in row:
        return 0, False
    value = row[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"! malformed {name} in {path}:{lineno}: {value!r}")
    return value, True


def _read_cache(path: Path) -> tuple[dict[str, str], _Usage, dict[str, int]]:
    """Load an append-only cache with usage and actual served-model counts."""
    if not path.exists():
        return {}, _Usage(), {}
    replies: dict[str, str] = {}
    usage_by_id: dict[str, _Usage] = {}
    model_by_id: dict[str, str] = {}
    for lineno, row in enumerate(read_jsonl(path), 1):
        item_id = _required_string(row, "id", path, lineno)
        content = _required_string(row, "content", path, lineno)
        assert isinstance(row, dict)
        model = row.get("model")
        if not isinstance(model, str) or not model:
            raise SystemExit(f"! malformed model in {path}:{lineno}: {model!r}")
        model_by_id[item_id] = model
        prompt_tokens, has_prompt_tokens = _optional_count(row, "prompt_tokens", path, lineno)
        completion_tokens, has_completion_tokens = _optional_count(
            row,
            "completion_tokens",
            path,
            lineno,
        )
        reasoning_tokens, has_reasoning_tokens = _optional_count(
            row,
            "reasoning_tokens",
            path,
            lineno,
        )
        count_fields = (has_prompt_tokens, has_completion_tokens, has_reasoning_tokens)
        if any(count_fields) and not all(count_fields):
            raise SystemExit(f"! incomplete token usage fields in {path}:{lineno}")
        replies[item_id] = content
        usage_by_id[item_id] = _Usage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            calls_with_token_counts=int(all(count_fields)),
        )
    usage = _Usage()
    for row_usage in usage_by_id.values():
        usage = usage.combine(row_usage)
    models: dict[str, int] = {}
    for model in model_by_id.values():
        models[model] = models.get(model, 0) + 1
    return replies, usage, dict(sorted(models.items()))


def _cache_provenance(
    *,
    schema: str,
    cache_key: str,
    instructions: str,
    reasoning_effort: str,
    endpoint: str,
    model: str,
    stage: str,
) -> dict[str, object]:
    return {
        "schema": schema,
        "cache_key": cache_key,
        "instructions": instructions,
        "reasoning_effort": reasoning_effort,
        "endpoint": endpoint,
        "model": model,
        "stage": stage,
    }


def _migrate_generation_sidecar(
    cache: Path,
    provenance: dict[str, object],
    *,
    legacy_instructions: str,
) -> bool:
    """Apply exact, semantics-preserving generation metadata migrations.

    Early qgen versions first hashed both prompt stages, then omitted the endpoint.
    Neither difference changes a cached generation request. Only those exact known
    shapes are accepted; every other provenance mismatch remains fail-closed.
    """
    sidecar = Path(f"{cache}.meta.json")
    if not cache.exists() or cache.stat().st_size == 0 or not sidecar.exists():
        return False
    recorded = read_json(sidecar)
    base = {
        "schema": provenance["schema"],
        "reasoning_effort": provenance["reasoning_effort"],
        "model": provenance["model"],
        "stage": "generate",
    }
    known_shapes = (
        {**base, "instructions": legacy_instructions},
        {
            **base,
            "cache_key": provenance["cache_key"],
            "instructions": provenance["instructions"],
        },
    )
    if recorded not in known_shapes:
        return False
    write_json(sidecar, provenance)
    return True


def _checkpoint(cache: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        append_jsonl(cache, rows)
        rows.clear()


def _complete_many(
    client: ChatClient,
    items: Sequence[tuple[str, str]],
    *,
    system: str,
    cache: Path,
    label: str,
    concurrency: int,
    log: Callable[[str], None],
) -> tuple[dict[str, str], _Usage, dict[str, int]]:
    """Fill a reply cache for ``items``, paying only for the ids it is missing."""
    replies, usage, models = _read_cache(cache)
    pending = [(item_id, user) for item_id, user in items if item_id not in replies]
    log(f"{label}: {len(items) - len(pending):,} cached, {len(pending):,} to request")
    if not pending:
        return replies, usage, models

    buffer: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        try:
            for start in range(0, len(pending), max(1, concurrency)):
                batch = pending[start : start + max(1, concurrency)]
                futures = {
                    pool.submit(client.complete, system, user): item_id for item_id, user in batch
                }
                for future in as_completed(futures):
                    item_id = futures[future]
                    reply = future.result()
                    replies[item_id] = reply.content
                    usage = usage.plus(
                        reply.prompt_tokens,
                        reply.completion_tokens,
                        reply.reasoning_tokens,
                    )
                    models[reply.model] = models.get(reply.model, 0) + 1
                    row: dict[str, Any] = {
                        "id": item_id,
                        "content": reply.content,
                        "model": reply.model,
                    }
                    usage_counts = (
                        reply.prompt_tokens,
                        reply.completion_tokens,
                        reply.reasoning_tokens,
                    )
                    if all(value is not None for value in usage_counts):
                        row.update(
                            {
                                "prompt_tokens": reply.prompt_tokens,
                                "completion_tokens": reply.completion_tokens,
                                "reasoning_tokens": reply.reasoning_tokens,
                            }
                        )
                    buffer.append(row)
                    done += 1
                    if len(buffer) >= FLUSH_EVERY:
                        _checkpoint(cache, buffer)
                        log(f"  {label}: {done:,}/{len(pending):,}")
        finally:
            _checkpoint(cache, buffer)
    log(f"{label}: {done:,} new replies, {usage.prompt_tokens + usage.completion_tokens:,} tokens")
    return replies, usage, dict(sorted(models.items()))


def _generate(
    chunks: Sequence[EvalChunk],
    replies: dict[str, str],
) -> tuple[list[GeneratedQuery], dict[str, int], list[dict[str, str]]]:
    """Parse generator replies into query pairs, counting every rejection."""
    queries: list[GeneratedQuery] = []
    rejections: dict[str, int] = {}
    ledger: list[dict[str, str]] = []
    for chunk in chunks:
        raw = replies.get(chunk.chunk_id)
        if raw is None:
            continue
        outcome = parse_generation(chunk, raw)
        if outcome.rejection is not None:
            kind = outcome.rejection.split(":", 1)[0]
            rejections[kind] = rejections.get(kind, 0) + 1
            ledger.append(
                {
                    "stage": "generation",
                    "chunk_id": chunk.chunk_id,
                    "reason": outcome.rejection,
                }
            )
            continue
        queries.extend(outcome.queries)
    return queries, dict(sorted(rejections.items())), ledger


def _overlap_report(
    queries: Sequence[GeneratedQuery],
    text_by_chunk: dict[str, str],
) -> dict[str, Any]:
    """Summarise how much of each query was copied from its own gold chunk."""
    report: dict[str, Any] = {}
    for variant in ("direct", "paraphrase"):
        values = [
            bigram_containment(query.question, text_by_chunk[query.chunk_id])
            for query in queries
            if query.variant == variant
        ]
        if not values:
            continue
        values.sort()
        report[variant] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "p10": round(values[int(len(values) * 0.10)], 4),
            "p50": round(values[len(values) // 2], 4),
            "p90": round(values[min(len(values) - 1, int(len(values) * 0.90))], 4),
        }
    paired = [
        (
            bigram_containment(direct.question, text_by_chunk[direct.chunk_id]),
            bigram_containment(twin.question, text_by_chunk[twin.chunk_id]),
        )
        for direct, twin in _pairs(queries)
    ]
    if paired:
        deltas = [twin - direct for direct, twin in paired]
        report["paraphrase_minus_direct"] = {
            "n": len(deltas),
            "mean": round(statistics.fmean(deltas), 4),
            "reduced": sum(1 for value in deltas if value < 0),
        }
    return report


def _pairs(queries: Sequence[GeneratedQuery]) -> list[tuple[GeneratedQuery, GeneratedQuery]]:
    by_chunk: dict[str, dict[str, GeneratedQuery]] = {}
    for query in queries:
        by_chunk.setdefault(query.chunk_id, {})[query.variant] = query
    return [
        (group["direct"], group["paraphrase"])
        for group in by_chunk.values()
        if "direct" in group and "paraphrase" in group
    ]


def _plan_only(
    sample: Sequence[EvalChunk],
    chunks: Sequence[EvalChunk],
    args: argparse.Namespace,
) -> int:
    print(f"sample: {len(sample):,} chunks of {len(chunks):,}  seed={args.seed}")
    population = theme_counts(chunks)
    drawn = theme_counts(sample)
    for theme in sorted(population):
        print(f"  {theme:32s} corpus={population[theme]:5,}  sample={drawn.get(theme, 0):4,}")
    calls = len(sample) * 2
    print(
        f"would send ~{calls:,} chat requests "
        f"({len(sample):,} generate + {len(sample):,} pair verify)\n"
        "re-run with --generate to spend"
    )
    return 0


def _verified(
    pairs: Sequence[tuple[GeneratedQuery, GeneratedQuery]],
    replies: dict[str, str],
) -> tuple[
    list[GeneratedQuery],
    dict[str, int],
    tuple[str, ...],
    list[dict[str, str]],
]:
    kept: list[GeneratedQuery] = []
    failures: dict[str, int] = {}
    dropped_chunks: list[str] = []
    ledger: list[dict[str, str]] = []
    for direct, paraphrase in pairs:
        raw = replies.get(verification_cache_id(direct, paraphrase))
        if raw is None:
            failures["missing_reply"] = failures.get("missing_reply", 0) + 1
            dropped_chunks.append(direct.chunk_id)
            ledger.append(
                {
                    "stage": "verification",
                    "chunk_id": direct.chunk_id,
                    "reason": "missing_reply",
                }
            )
            continue
        try:
            verdict = parse_verdict(raw)
        except ValueError as exc:
            failures["unparseable"] = failures.get("unparseable", 0) + 1
            dropped_chunks.append(direct.chunk_id)
            ledger.append(
                {
                    "stage": "verification",
                    "chunk_id": direct.chunk_id,
                    "reason": f"unparseable: {exc}",
                }
            )
            continue
        if verdict.keep:
            kept.extend((direct, paraphrase))
            continue
        dropped_chunks.append(direct.chunk_id)
        axes = verdict.failures()
        ledger.append(
            {
                "stage": "verification",
                "chunk_id": direct.chunk_id,
                "reason": ",".join(axes),
            }
        )
        for axis in axes:
            failures[axis] = failures.get(axis, 0) + 1
    return kept, dict(sorted(failures.items())), tuple(sorted(dropped_chunks)), ledger


def _build(args: argparse.Namespace, sample: Sequence[EvalChunk], out: Path) -> int:
    env = load_env(ROOT / ".env")
    generator = ChatClient(ChatConfig.from_env(env), reasoning_effort=args.reasoning_effort or None)
    verifier_model = args.verifier_model or env["LLM_MODEL_NAME"]
    verifier = ChatClient(
        ChatConfig.from_env(env, model=verifier_model),
        reasoning_effort=args.reasoning_effort or None,
    )

    if generator.config.model == verifier.config.model:
        print(
            f"! generator and verifier are both {generator.config.model}: "
            "the verification pass is self-agreement, not an independent check"
        )

    generation_provenance = _cache_provenance(
        schema=QGEN_SCHEMA,
        cache_key=GENERATION_CACHE_KEY_SCHEMA,
        instructions=generation_instructions_fingerprint(),
        reasoning_effort=args.reasoning_effort,
        endpoint=generator.config.endpoint,
        model=generator.config.model,
        stage="generate",
    )
    verification_provenance = _cache_provenance(
        schema=QGEN_SCHEMA,
        cache_key=VERIFICATION_CACHE_KEY_SCHEMA,
        instructions=verification_instructions_fingerprint(),
        reasoning_effort=args.reasoning_effort,
        endpoint=verifier.config.endpoint,
        model=verifier.config.model,
        stage="verify",
    )
    generation_cache = out / "generation_cache.jsonl"
    verification_cache = out / "verification_cache.jsonl"
    if _migrate_generation_sidecar(
        generation_cache,
        generation_provenance,
        legacy_instructions=instructions_fingerprint(),
    ):
        print("migrated generation cache provenance to the stage-specific fingerprint")
    prepare_cache_sidecar(
        generation_cache,
        generation_provenance,
        label="qgen cache",
    )
    prepare_cache_sidecar(
        verification_cache,
        verification_provenance,
        label="qgen verification cache",
    )

    generated, generate_usage, generation_models = _complete_many(
        generator,
        [(chunk.chunk_id, generation_prompt(chunk)) for chunk in sample],
        system=GENERATION_INSTRUCTIONS,
        cache=generation_cache,
        label="generate",
        concurrency=args.concurrency,
        log=print,
    )
    queries, rejections, drop_ledger = _generate(sample, generated)
    print(f"parsed {len(queries):,} queries from {len(sample):,} chunks; rejected {rejections}")

    deduped, dropped = dedupe_questions(queries)
    drop_ledger.extend(
        {
            "stage": "dedupe",
            "chunk_id": chunk_id,
            "reason": f"near-duplicate of {other_id}",
        }
        for chunk_id, other_id in dropped
    )
    print(f"deduped: kept {len(deduped):,}, dropped {len(dropped):,} near-duplicate chunks")

    by_chunk = {chunk.chunk_id: chunk for chunk in sample}
    pairs = _pairs(deduped)
    verified_replies, verify_usage, verification_models = _complete_many(
        verifier,
        [
            (
                verification_cache_id(direct, paraphrase),
                verification_prompt(by_chunk[direct.chunk_id], direct, paraphrase),
            )
            for direct, paraphrase in pairs
        ],
        system=VERIFICATION_INSTRUCTIONS,
        cache=verification_cache,
        label="verify",
        concurrency=args.concurrency,
        log=print,
    )
    kept, failures, incomplete_chunks, verification_ledger = _verified(
        pairs,
        verified_replies,
    )
    drop_ledger.extend(verification_ledger)
    print(
        f"verified: kept {len(kept):,} of {len(deduped):,}; "
        f"axis failures {failures}; dropped pairs {len(incomplete_chunks):,}"
    )

    state = read_state(args.artifacts / "state.json")
    assert state is not None
    return _write(
        args,
        kept=kept,
        sample=sample,
        by_chunk=by_chunk,
        out=out,
        report_extra={
            "reasoning_effort": args.reasoning_effort,
            "instruction_fingerprints": {
                "generation": generation_instructions_fingerprint(),
                "verification": verification_instructions_fingerprint(),
            },
            "published_index": {
                "scope": state.scope,
                "chunker": state.chunker_fingerprint,
                "embedding_profile": state.embedding_profile,
                "sparse": state.sparse_fingerprint,
                "collection": state.collection_name,
                "chunks": len(state.chunk_ids()),
            },
            "generator_endpoint": generator.config.endpoint,
            "generator_model_requested": generator.config.model,
            "generator_models_served": generation_models,
            "verifier_endpoint": verifier.config.endpoint,
            "verifier_model_requested": verifier.config.model,
            "verifier_models_served": verification_models,
            "verification_independence": (
                "same-requested-model self-agreement"
                if generator.config.model == verifier.config.model
                else "different-requested-model review"
            ),
            "drop_ledger": sorted(
                drop_ledger,
                key=lambda row: (row["stage"], row["chunk_id"]),
            ),
            "generation_rejections": rejections,
            "near_duplicate_chunks_dropped": len(dropped),
            "verification_failures": failures,
            "pairs_verified": len(pairs),
            "pairs_dropped": len(incomplete_chunks),
            "usage": {
                "generate": generate_usage.as_dict(),
                "verify": verify_usage.as_dict(),
                "accounting": (
                    "all unique cache rows; legacy rows without token fields are reported "
                    "as partial coverage"
                ),
            },
        },
    )


def _write(
    args: argparse.Namespace,
    *,
    kept: Sequence[GeneratedQuery],
    sample: Sequence[EvalChunk],
    by_chunk: dict[str, EvalChunk],
    out: Path,
    report_extra: dict[str, Any],
) -> int:
    text_by_chunk = {chunk_id: chunk.text for chunk_id, chunk in by_chunk.items()}
    rows = [
        {
            "query_id": query.query_id,
            "question": query.question,
            "answer": query.answer,
            "gold_doc_ids": [query.chunk_id],
            "gold_source_key": by_chunk[query.chunk_id].source_key,
            "task": query.variant,
            "question_type": query.question_type,
            "theme": by_chunk[query.chunk_id].theme,
            "bigram_containment": round(
                bigram_containment(query.question, text_by_chunk[query.chunk_id]), 4
            ),
        }
        for query in sorted(kept, key=lambda query: query.query_id)
    ]
    # Do not truncate a previous canonical query set when a paid run produces
    # no usable pair. The report is still written below so the failed run is
    # auditable, while the query artifact remains the last known-good output.
    written = len(rows)
    if kept:
        write_jsonl(out / "queries.jsonl", rows)

    complete_pairs = len(_pairs(kept))
    report = {
        "schema": QGEN_SCHEMA,
        "seed": args.seed,
        "sampled_chunks": len(sample),
        "queries": written,
        "complete_pairs": complete_pairs,
        "queries_by_variant": {
            variant: sum(1 for query in kept if query.variant == variant)
            for variant in ("direct", "paraphrase")
        },
        "queries_by_type": {
            question_type: sum(1 for query in kept if query.question_type == question_type)
            for question_type in sorted({query.question_type for query in kept})
        },
        "theme_coverage": theme_counts(
            [by_chunk[chunk_id] for chunk_id in {query.chunk_id for query in kept}]
        ),
        "bigram_containment": _overlap_report(kept, text_by_chunk),
        "gold_semantics": (
            "single generating chunk; incomplete until judge_tidb_pool.py grades a pool"
        ),
        **report_extra,
    }
    write_json(out / "report.json", report)
    if not kept:
        print("! no query survived verification")
        return 1
    print(
        f"wrote {written:,} queries ({complete_pairs:,} complete pairs) to {out / 'queries.jsonl'}"
    )
    overlap = report["bigram_containment"]
    if isinstance(overlap, dict) and "direct" in overlap:
        print(
            f"bigram containment p50: direct {overlap['direct']['p50']:.3f}  "
            f"paraphrase {overlap['paraphrase']['p50']:.3f}"
        )
    return 0


def _output_dir(args: argparse.Namespace) -> Path:
    canonical = args.limit is None and args.size == DEFAULT_SAMPLE_SIZE
    if canonical:
        return args.artifacts / "eval"
    suffix = f"size-{args.size}-limit-{args.limit or 'all'}"
    return args.artifacts / "eval-trials" / suffix


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)

    for path, label in ((args.manifest, "manifest"), (args.documents, "documents")):
        if not path.exists():
            print(f"! {path} ({label}) not found -- run tidb-rag-curated/download_curated.ps1")
            return 1

    if args.size < 1:
        raise SystemExit("! --size must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("! --limit must be positive")
    if args.concurrency < 1:
        raise SystemExit("! --concurrency must be positive")

    chunks = _load_chunks(args.manifest, args.documents)
    _verify_against_index(chunks, args.artifacts)
    requested_size = min(args.size, len(chunks))
    if args.limit is not None:
        requested_size = min(requested_size, args.limit)
    sample = stratified_sample(chunks, size=requested_size, seed=args.seed)

    if not args.generate:
        return _plan_only(sample, chunks, args)

    out = _output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".build.lock"):
        return _build(args, sample, out)


if __name__ == "__main__":
    raise SystemExit(main())
