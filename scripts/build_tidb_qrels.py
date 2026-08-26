"""Judge the completed TiDB candidate pool and publish multi-gold qrels.

The default is an offline status check. Provider calls require ``--judge`` and
use the configured ``LLM_*`` credentials with the frozen high reasoning effort.
Judgement responses are cached by the exact direct/paraphrase pair and the
rank-blinded candidate batch, so an interrupted run resumes without re-paying
completed batches. A qrels bundle is published only after every pooled
candidate has a valid grade.

    uv run python scripts/build_tidb_qrels.py
    uv run python scripts/build_tidb_qrels.py --judge
    uv run python scripts/build_tidb_qrels.py --finalize
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.pool import (
    JUDGING_CACHE_KEY_SCHEMA,
    JUDGING_INSTRUCTIONS,
    POOL_SCHEMA,
    JudgedQuery,
    PooledQuery,
    batched,
    judging_cache_id,
    judging_input_fingerprint,
    judging_instructions_fingerprint,
    judging_order,
    judging_prompt,
    parse_judgements,
    pool_fingerprint,
    qrels_rows,
)
from zhrag.eval.qgen import EvalChunk
from zhrag.eval.tidb_runs import (
    DENSE_LABEL,
    LEXICAL_LABEL,
    RERANK_LABEL,
    RRF_LABEL,
    TIDB_RUNS_SCHEMA,
    TiDBRuns,
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
    replace_files,
    write_json,
    write_jsonl,
)
from zhrag.providers.cache import (
    load_cache_provenance,
    prepare_cache_sidecar,
    validate_cache_sidecar,
)
from zhrag.providers.chat import ChatClient, ChatConfig
from zhrag.providers.embedding import load_env

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tidb-rag-curated" / "selected_manifest.json"
DOCUMENTS = ROOT / "tidb-rag-curated" / "documents"
ARTIFACTS = ROOT / "indexes" / "tidb"
EVAL = ARTIFACTS / "eval"
QUERIES = EVAL / "queries.jsonl"
RUNS = EVAL / "runs.jsonl"
POOL = EVAL / "pool.jsonl"
POOL_REPORT = EVAL / "pool_report.json"
JUDGING_CACHE = EVAL / "judging_cache.jsonl"
QRELS = EVAL / "qrels.jsonl"
QRELS_REPORT = EVAL / "qrels_report.json"

TARGET_TOKENS = 400
HARD_MAX_TOKENS = 600
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
DEFAULT_BATCH_SIZE = 8
DEFAULT_ORDER_SEED = "zhrag-tidb-qrels-2026-08"
REASONING_EFFORT = "high"
JUDGING_SCHEMA = "zhrag-tidb-judging-cache-v1"
JUDGING_REQUEST_CONTRACT = (
    "one direct/paraphrase pair + rank-blinded fixed candidate batch; JSON object; v1"
)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


@dataclass(frozen=True, slots=True)
class _Batch:
    cache_id: str
    unit: PooledQuery
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Experiment:
    corpus: dict[str, str]
    query_rows: dict[str, dict[str, Any]]
    runs: TiDBRuns
    units: tuple[PooledQuery, ...]
    pool_report: dict[str, Any]
    query_set_fingerprint: str


class _Usage:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls_with_token_counts = 0

    def add(self, entry: _CacheEntry) -> None:
        self.calls += 1
        values = (entry.prompt_tokens, entry.completion_tokens, entry.reasoning_tokens)
        if all(value is not None for value in values):
            self.calls_with_token_counts += 1
        self.prompt_tokens += entry.prompt_tokens or 0
        self.completion_tokens += entry.completion_tokens or 0
        self.reasoning_tokens += entry.reasoning_tokens or 0

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


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge the TiDB candidate pool.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--order-seed", default=DEFAULT_ORDER_SEED)
    parser.add_argument("--judge", action="store_true", help="allow paid chat judgement calls")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="validate the complete cache and publish qrels without provider calls",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="request at most this many missing batches (requires --judge)",
    )
    return parser.parse_args(argv)


def _load_corpus(args: argparse.Namespace) -> dict[str, str]:
    state = read_state(args.artifacts / "state.json")
    if state is None:
        raise SystemExit("! published TiDB state is absent -- build the index first")
    if state.scope != scope_fingerprint(Scope.evergreen()):
        raise SystemExit("! published TiDB scope is not the frozen evergreen scope")
    expected_chunker = chunker_fingerprint(
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    if state.chunker_fingerprint != expected_chunker:
        raise SystemExit("! published TiDB chunker is not the frozen evaluation chunker")
    if state.embedding_profile != DOCUMENT_EMBEDDING_PROFILE:
        raise SystemExit("! published document embedding profile is not the frozen TiDB profile")

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
    published = set(state.chunk_ids())
    if set(corpus) != published:
        raise SystemExit(
            "! materialized corpus does not equal the exact published chunk set: "
            f"{len(set(corpus) - published):,} missing, {len(published - set(corpus)):,} extra"
        )
    return corpus


def _query_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            raise SystemExit(f"! malformed query row in {path}: expected an object")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise SystemExit(f"! malformed query id in {path}: {query_id!r}")
        if query_id in rows:
            raise SystemExit(f"! duplicate query id in {path}: {query_id}")
        for name in ("question", "answer", "task"):
            value = row.get(name)
            if not isinstance(value, str) or not value:
                raise SystemExit(f"! malformed {name} for {query_id}")
        gold = row.get("gold_doc_ids")
        if not isinstance(gold, list) or len(gold) != 1 or not isinstance(gold[0], str):
            raise SystemExit(f"! {query_id}: qgen rows must have exactly one generating chunk")
        rows[query_id] = dict(row)
    if not rows:
        raise SystemExit(f"! query set is empty: {path}")
    return rows


def _string_tuple(value: object, *, field: str, row_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SystemExit(f"! pool row {row_number}: {field} must be a list of non-empty strings")
    return tuple(value)


def _pool_unit(  # noqa: PLR0912
    row: Mapping[str, object],
    *,
    row_number: int,
    query_rows: Mapping[str, Mapping[str, Any]],
    corpus: Mapping[str, str],
) -> PooledQuery:
    if row.get("schema") != POOL_SCHEMA:
        raise SystemExit(f"! malformed pool row {row_number}: schema mismatch")
    query_ids = _string_tuple(row.get("query_ids"), field="query_ids", row_number=row_number)
    questions = _string_tuple(row.get("questions"), field="questions", row_number=row_number)
    candidates = _string_tuple(
        row.get("candidate_doc_ids"), field="candidate_doc_ids", row_number=row_number
    )
    raw_contributors = row.get("contributors")
    if not isinstance(raw_contributors, dict):
        raise SystemExit(f"! pool row {row_number}: contributors must be an object")
    contributors: dict[str, tuple[str, ...]] = {}
    for doc_id, systems in raw_contributors.items():
        if not isinstance(doc_id, str):
            raise SystemExit(f"! pool row {row_number}: contributor id is not a string")
        contributors[doc_id] = _string_tuple(
            systems, field=f"contributors[{doc_id}]", row_number=row_number
        )
    chunk_id = row.get("generating_chunk_id")
    answer = row.get("answer")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise SystemExit(f"! pool row {row_number}: generating chunk is malformed")
    if not isinstance(answer, str) or not answer.strip():
        raise SystemExit(f"! pool row {row_number}: answer is malformed")
    if row.get("tasks") != ["direct", "paraphrase"]:
        raise SystemExit(f"! pool row {row_number}: tasks are not the frozen pair")
    try:
        unit = PooledQuery(
            chunk_id=chunk_id,
            query_ids=query_ids,
            questions=questions,
            answer=answer,
            candidates=candidates,
            contributors=contributors,
        )
    except ValueError as exc:
        raise SystemExit(f"! malformed pool row {row_number}: {exc}") from exc
    if unit.chunk_id not in corpus or any(doc_id not in corpus for doc_id in unit.candidates):
        raise SystemExit(f"! pool row {row_number}: candidate is absent from published corpus")
    for query_id, question in zip(unit.query_ids, unit.questions, strict=True):
        source = query_rows.get(query_id)
        if source is None:
            raise SystemExit(f"! pool row {row_number}: unknown query id {query_id}")
        if source["question"] != question or source["answer"] != unit.answer:
            raise SystemExit(f"! pool row {row_number}: query surface or answer drift")
        if source["task"] != query_id.partition(":")[0]:
            raise SystemExit(f"! pool row {row_number}: query task drift")
        if source["gold_doc_ids"] != [unit.chunk_id]:
            raise SystemExit(f"! pool row {row_number}: generating gold drift")
    return unit


def _load_pool(
    path: Path,
    *,
    query_rows: Mapping[str, Mapping[str, Any]],
    corpus: Mapping[str, str],
) -> tuple[PooledQuery, ...]:
    units: list[PooledQuery] = []
    seen_query_ids: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), 1):
        unit = _pool_unit(
            row,
            row_number=row_number,
            query_rows=query_rows,
            corpus=corpus,
        )
        if seen_query_ids & set(unit.query_ids):
            raise SystemExit(f"! pool repeats a query id at row {row_number}")
        seen_query_ids.update(unit.query_ids)
        units.append(unit)
    if not units:
        raise SystemExit(f"! pool is empty: {path}")
    if seen_query_ids != set(query_rows):
        raise SystemExit(
            "! pool/query coverage differs: "
            f"{len(set(query_rows) - seen_query_ids):,} queries absent from pool, "
            f"{len(seen_query_ids - set(query_rows)):,} unknown pool queries"
        )
    return tuple(units)


def _load_runs(path: Path, query_ids: Sequence[str]) -> TiDBRuns:
    required = {LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL}
    rows: dict[str, dict[str, tuple[str, ...]]] = {}
    seen: set[str] = set()
    for row_number, row in enumerate(read_jsonl(path), 1):
        if not isinstance(row, dict):
            raise SystemExit(f"! malformed run row {row_number}")
        query_id = row.get("query_id")
        raw_runs = row.get("runs")
        if not isinstance(query_id, str) or not query_id or not isinstance(raw_runs, dict):
            raise SystemExit(f"! malformed run row {row_number}")
        if query_id in seen:
            raise SystemExit(f"! duplicate run query id {query_id}")
        seen.add(query_id)
        if set(raw_runs) != required:
            raise SystemExit(f"! run labels differ at row {row_number}")
        rows[query_id] = {
            label: _string_tuple(raw_runs[label], field=f"runs[{label}]", row_number=row_number)
            for label in sorted(required)
        }
    if tuple(sorted(seen)) != tuple(sorted(query_ids)):
        raise SystemExit("! runs/query coverage differs")
    aligned = {
        label: tuple(rows[query_id][label] for query_id in query_ids) for label in sorted(required)
    }
    try:
        return TiDBRuns(query_ids=tuple(query_ids), runs=aligned)
    except ValueError as exc:
        raise SystemExit(f"! malformed runs: {exc}") from exc


def _load_experiment(args: argparse.Namespace) -> _Experiment:
    corpus = _load_corpus(args)
    query_rows = _query_rows(args.artifacts / "eval" / QUERIES.name)
    units = _load_pool(args.artifacts / "eval" / POOL.name, query_rows=query_rows, corpus=corpus)
    runs_path = args.artifacts / "eval" / RUNS.name
    runs = _load_runs(runs_path, tuple(query_rows))
    report_path = args.artifacts / "eval" / POOL_REPORT.name
    report = read_json(report_path)
    if not isinstance(report, dict) or report.get("schema") != TIDB_RUNS_SCHEMA:
        raise SystemExit("! pool report schema is malformed")
    expected_pool = pool_fingerprint(units)
    if report.get("pool_fingerprint") != expected_pool:
        raise SystemExit("! pool fingerprint differs from pool report")
    if report.get("runs_fingerprint") != runs.fingerprint:
        raise SystemExit("! runs fingerprint differs from pool report")
    if report.get("queries") != len(query_rows) or report.get("pairs") != len(units):
        raise SystemExit("! pool report counts differ from its artifacts")
    if report.get("corpus_chunks") != len(corpus):
        raise SystemExit("! pool report corpus count differs from published corpus")
    query_set_fingerprint = _query_set_fingerprint(query_rows)
    return _Experiment(corpus, query_rows, runs, units, report, query_set_fingerprint)


def _query_set_fingerprint(rows: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(rows):
        row = rows[query_id]
        for value in (
            query_id,
            str(row["question"]),
            str(row["answer"]),
            str(row["task"]),
            *[str(doc_id) for doc_id in row["gold_doc_ids"]],
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _batches(units: Sequence[PooledQuery], *, seed: str, batch_size: int) -> tuple[_Batch, ...]:
    result: list[_Batch] = []
    for unit in units:
        ordered = judging_order(unit.candidates, seed=f"{seed}:{unit.chunk_id}")
        for batch in batched(ordered, batch_size):
            result.append(_Batch(judging_cache_id(unit.questions, unit.answer, batch), unit, batch))
    return tuple(result)


def _cache_entries(path: Path) -> dict[str, _CacheEntry]:
    if not path.exists():
        return {}
    entries: dict[str, _CacheEntry] = {}
    for lineno, row in enumerate(read_jsonl(path), 1):
        if not isinstance(row, dict):
            raise SystemExit(f"! malformed judging cache row {lineno}: expected an object")
        item_id = row.get("id")
        content = row.get("content")
        model = row.get("model")
        if not isinstance(item_id, str) or not item_id:
            raise SystemExit(f"! malformed judging cache id at row {lineno}")
        if not isinstance(content, str) or not content:
            raise SystemExit(f"! malformed judging cache content at row {lineno}")
        if not isinstance(model, str) or not model:
            raise SystemExit(f"! malformed judging cache model at row {lineno}")
        values: list[int | None] = []
        for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            value = row.get(name)
            invalid = isinstance(value, bool) or not isinstance(value, int) or value < 0
            if value is not None and invalid:
                raise SystemExit(f"! malformed {name} at cache row {lineno}")
            values.append(value)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise SystemExit(f"! incomplete token usage fields at cache row {lineno}")
        entries[item_id] = _CacheEntry(content, model, *values)
    return entries


def _base_provenance(experiment: _Experiment, *, seed: str, batch_size: int) -> dict[str, object]:
    return {
        "schema": JUDGING_SCHEMA,
        "pool_schema": POOL_SCHEMA,
        "cache_key": JUDGING_CACHE_KEY_SCHEMA,
        "instructions": judging_instructions_fingerprint(),
        "reasoning_effort": REASONING_EFFORT,
        "batch_size": batch_size,
        "order_seed": seed,
        "pool_fingerprint_sha256": pool_fingerprint(experiment.units),
        "query_set_fingerprint_sha256": experiment.query_set_fingerprint,
        "judging_input_fingerprint_sha256": judging_input_fingerprint(
            experiment.units, corpus=experiment.corpus, seed=seed, batch_size=batch_size
        ),
        "request_contract": JUDGING_REQUEST_CONTRACT,
    }


def _provenance(
    experiment: _Experiment,
    *,
    seed: str,
    batch_size: int,
    model: str,
    endpoint: str,
) -> dict[str, object]:
    return {
        **_base_provenance(experiment, seed=seed, batch_size=batch_size),
        "model": model,
        "endpoint": endpoint,
    }


def _prepare_or_validate_sidecar(
    cache: Path,
    expected_base: Mapping[str, object],
    *,
    judge: bool,
    config: ChatConfig | None,
) -> dict[str, object] | None:
    if judge:
        assert config is not None
        expected = {**expected_base, "model": config.model, "endpoint": config.endpoint}
        prepare_cache_sidecar(cache, expected, label="TiDB judging cache")
        return expected
    if not cache.exists() and not Path(f"{cache}.meta.json").exists():
        return None
    recorded = load_cache_provenance(cache, label="TiDB judging cache")
    dynamic = {"model": recorded.get("model"), "endpoint": recorded.get("endpoint")}
    if not isinstance(dynamic["model"], str) or not dynamic["model"]:
        raise SystemExit("! judging cache metadata has no valid model")
    if not isinstance(dynamic["endpoint"], str) or not dynamic["endpoint"]:
        raise SystemExit("! judging cache metadata has no valid endpoint")
    expected = {**expected_base, **dynamic}
    validate_cache_sidecar(cache, expected, label="TiDB judging cache")
    return expected


def _valid_grades(
    entries: Mapping[str, _CacheEntry], batches: Sequence[_Batch]
) -> tuple[dict[str, dict[str, int]], set[str]]:
    parsed: dict[str, dict[str, int]] = {}
    invalid: set[str] = set()
    for batch in batches:
        entry = entries.get(batch.cache_id)
        if entry is None:
            invalid.add(batch.cache_id)
            continue
        try:
            parsed[batch.cache_id] = parse_judgements(entry.content, batch.candidates)
        except ValueError:
            invalid.add(batch.cache_id)
    return parsed, invalid


def _judge_missing(
    experiment: _Experiment,
    batches: Sequence[_Batch],
    entries: dict[str, _CacheEntry],
    missing: set[str],
    config: ChatConfig,
    *,
    cache: Path,
    max_batches: int | None,
) -> None:
    selected = [batch for batch in batches if batch.cache_id in missing]
    if max_batches is not None:
        selected = selected[:max_batches]
    if not selected:
        return
    client = ChatClient(config=config, reasoning_effort=REASONING_EFFORT)
    for number, batch in enumerate(selected, 1):
        prompt = judging_prompt(
            batch.unit.questions,
            batch.unit.answer,
            batch.candidates,
            experiment.corpus,
        )
        reply = client.complete(JUDGING_INSTRUCTIONS, prompt)
        try:
            parse_judgements(reply.content, batch.candidates)
        except ValueError as exc:
            raise SystemExit(
                f"! invalid judgement reply for batch {batch.cache_id}: {exc}"
            ) from exc
        entry = _CacheEntry(
            content=reply.content,
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            reasoning_tokens=reply.reasoning_tokens,
        )
        row: dict[str, Any] = {"id": batch.cache_id, "content": reply.content, "model": reply.model}
        counts = (reply.prompt_tokens, reply.completion_tokens, reply.reasoning_tokens)
        if all(value is not None for value in counts):
            row.update(
                {
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "reasoning_tokens": reply.reasoning_tokens,
                }
            )
        append_jsonl(
            cache,
            [row],
        )
        entries[batch.cache_id] = entry
        print(f"judged {number:,}/{len(selected):,} batches", flush=True)


def _coverage_report(
    experiment: _Experiment,
    qrels: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, int | float]]:
    report: dict[str, dict[str, int | float]] = {}
    for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL):
        rows = 0
        rank1 = 0
        rank10 = 0
        arm_runs = experiment.runs.runs[label]
        for query_id, per_arm in zip(experiment.runs.query_ids, arm_runs, strict=True):
            raw_judged = qrels[query_id]["judged_doc_ids"]
            if not isinstance(raw_judged, list):
                raise ValueError(f"{query_id}: qrels judged_doc_ids must be a list")
            judged = set(raw_judged)
            top1 = set(per_arm[:1]) <= judged
            top10 = set(per_arm[:10]) <= judged
            rows += 1
            rank1 += int(top1)
            rank10 += int(top10)
        report[label] = {
            "queries": rows,
            "top1_complete": rank1,
            "top1_rate": rank1 / rows,
            "top10_complete": rank10,
            "top10_rate": rank10 / rows,
        }
    return report


def _publish(
    args: argparse.Namespace,
    experiment: _Experiment,
    judged: Sequence[JudgedQuery],
    entries: Mapping[str, _CacheEntry],
    batches: Sequence[_Batch],
    *,
    provenance: Mapping[str, object],
) -> None:
    source = experiment.query_rows
    rows = qrels_rows(judged, queries=source)
    qrels_by_id = {str(row["query_id"]): row for row in rows}
    grades: list[int] = []
    generator_grades: list[int] = []
    gold_arities: list[int] = []
    for unit in judged:
        grades.extend(unit.grades.values())
        generator_grades.append(unit.generator_chunk_grade)
        gold_arities.append(len(unit.gold_doc_ids))
    usage = _Usage()
    served_models: dict[str, int] = {}
    for batch in batches:
        entry = entries[batch.cache_id]
        usage.add(entry)
        served_models[entry.model] = served_models.get(entry.model, 0) + 1
    report = {
        "schema": "zhrag-tidb-qrels-v1",
        "pool_fingerprint_sha256": pool_fingerprint(experiment.units),
        "runs_fingerprint_sha256": experiment.runs.fingerprint,
        "query_set_fingerprint_sha256": experiment.query_set_fingerprint,
        "corpus_chunks": len(experiment.corpus),
        "pairs": len(judged),
        "queries": len(rows),
        "batches": len(batches),
        "batch_size": provenance["batch_size"],
        "order_seed": provenance["order_seed"],
        "judging_input_fingerprint_sha256": provenance["judging_input_fingerprint_sha256"],
        "grades": {str(grade): grades.count(grade) for grade in (0, 1, 2)},
        "gold_arity": {
            "mean": sum(gold_arities) / len(gold_arities),
            "min": min(gold_arities),
            "max": max(gold_arities),
        },
        "generating_chunk_grades": {
            str(grade): generator_grades.count(grade) for grade in (0, 1, 2)
        },
        "generating_chunk_disagreement_rate": sum(grade != 2 for grade in generator_grades)
        / len(generator_grades),
        "requested_model": provenance["model"],
        "endpoint": provenance["endpoint"],
        "reasoning_effort": REASONING_EFFORT,
        "served_models": dict(sorted(served_models.items())),
        "usage": usage.as_dict(),
        "cache_batches": len(entries),
        "cache_valid_batches": len(batches),
        "run_judged_coverage": _coverage_report(experiment, qrels_by_id),
        "semantics": (
            "grade 2 is binary gold; grade 1 is partial context; documents outside "
            "the pooled candidates remain unjudged"
        ),
    }
    eval_root = args.artifacts / "eval"
    staged_qrels = eval_root / f"{QRELS.name}.tmp"
    staged_report = eval_root / f"{QRELS_REPORT.name}.tmp"
    staged = (staged_qrels, staged_report)
    try:
        write_jsonl(staged_qrels, rows)
        write_json(staged_report, report)
        replace_files(
            (
                (staged_qrels, eval_root / QRELS.name),
                (staged_report, eval_root / QRELS_REPORT.name),
            )
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    print(f"wrote {len(rows):,} qrels rows to {eval_root / QRELS.name}")


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    if args.judge and args.finalize:
        raise SystemExit("! --judge and --finalize are mutually exclusive")
    if args.max_batches is not None and (args.max_batches < 1 or not args.judge):
        raise SystemExit("! --max-batches requires a positive value and --judge")
    if args.batch_size < 1:
        raise SystemExit("! --batch-size must be positive")
    if not args.order_seed:
        raise SystemExit("! --order-seed must be non-empty")

    experiment = _load_experiment(args)
    batches = _batches(experiment.units, seed=args.order_seed, batch_size=args.batch_size)
    expected_ids = {batch.cache_id for batch in batches}
    cache = args.artifacts / "eval" / JUDGING_CACHE.name
    config: ChatConfig | None = None
    if args.judge:
        config = ChatConfig.from_env(load_env(ROOT / ".env"))
    base = _base_provenance(experiment, seed=args.order_seed, batch_size=args.batch_size)
    with exclusive_lock(args.artifacts / "eval" / ".qrels.lock"):
        provenance = _prepare_or_validate_sidecar(cache, base, judge=args.judge, config=config)
        entries = _cache_entries(cache)
        unexpected = sorted(set(entries) - expected_ids)
        if unexpected:
            raise SystemExit(f"! judging cache contains {len(unexpected):,} unexpected batch ids")
        parsed, invalid = _valid_grades(entries, batches)
        print(
            f"judging batches: {len(batches):,} expected; "
            f"{len(batches) - len(invalid):,} valid; {len(invalid):,} missing or invalid"
        )
        if args.judge:
            assert config is not None
            assert provenance is not None
            _judge_missing(
                experiment,
                batches,
                entries,
                invalid,
                config,
                cache=cache,
                max_batches=args.max_batches,
            )
            parsed, invalid = _valid_grades(entries, batches)
        if invalid:
            print("qrels not published: judging cache is incomplete")
            return 0
        if not args.finalize:
            print("judging cache is complete; pass --finalize to publish qrels")
            return 0
        judged: list[JudgedQuery] = []
        for unit in experiment.units:
            grades: dict[str, int] = {}
            ordered = judging_order(unit.candidates, seed=f"{args.order_seed}:{unit.chunk_id}")
            for batch in batched(ordered, args.batch_size):
                cache_id = judging_cache_id(unit.questions, unit.answer, batch)
                grades.update(parsed[cache_id])
            judged.append(
                JudgedQuery(chunk_id=unit.chunk_id, query_ids=unit.query_ids, grades=grades)
            )
        assert provenance is not None
        _publish(
            args,
            experiment,
            judged,
            entries,
            batches,
            provenance=provenance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
