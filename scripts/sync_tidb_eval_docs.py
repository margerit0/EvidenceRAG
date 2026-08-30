"""Synchronize TiDB evaluation status from validated local aggregate reports.

The report inputs live under the gitignored index tree. Only an explicit
allowlist of aggregate counts and methodological statements is rendered into
tracked documentation; query text, answers, chunk ids, raw judgements and
provider responses never cross that boundary.

Run tests with JUnit output before synchronization:

    uv run pytest --junitxml=indexes/tidb/eval/pytest.xml
    uv run python scripts/sync_tidb_eval_docs.py
    uv run python scripts/sync_tidb_eval_docs.py --check
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from zhrag.eval.tidb_quality import (
    PRIMARY_METRIC,
    RUN_LABELS,
    evaluate_tidb_quality,
    validate_quality_report,
)
from zhrag.eval.tidb_runs import DENSE_LABEL, LEXICAL_LABEL, RERANK_LABEL, RRF_LABEL
from zhrag.io_utils import (
    exclusive_lock,
    read_json,
    read_jsonl,
    read_text,
    replace_files,
    write_text,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
CLAUDE_CONTEXT = ROOT / "CLAUDE.md"

STATE_SCHEMA = "zhrag-ingest-state-v1"
QGEN_SCHEMA = "zhrag-tidb-qgen-v1"
POOL_SCHEMA = "zhrag-tidb-runs-v1"
QRELS_SCHEMA = "zhrag-tidb-qrels-v1"
ARTIFACT_LOCK = ".artifacts.lock"
DOCS_LOCK = ".docs.lock"
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"


@dataclass(frozen=True, slots=True)
class EvalStatus:
    documents: int
    chunks: int
    collection: str
    sampled_chunks: int
    verified_pairs: int
    source_clusters: int
    label_provenance: str
    queries: int
    direct_queries: int
    paraphrase_queries: int
    pool_candidates: int
    pool_min: int
    pool_max: int
    judging_batches: int
    grade_zero: int
    grade_one: int
    grade_two: int
    generating_disagreements: int
    generating_disagreement_rate: float
    tests: int
    quality: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Target:
    path: Path
    regions: Mapping[str, Callable[[EvalStatus], str]]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync tracked TiDB evaluation status from local aggregate reports."
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--claude-context", type=Path, default=CLAUDE_CONTEXT)
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _object(path: Path) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(f"! expected a JSON object: {path}")
    return raw


def _integer(
    row: Mapping[str, Any],
    name: str,
    *,
    source: Path,
    minimum: int = 0,
) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SystemExit(f"! {source}: {name} must be an integer >= {minimum}")
    return value


def _number(row: Mapping[str, Any], name: str, *, source: Path) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"! {source}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"! {source}: {name} must be finite")
    return result


def _mapping(row: Mapping[str, Any], name: str, *, source: Path) -> Mapping[str, Any]:
    value = row.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"! {source}: {name} must be an object")
    return value


def _string(row: Mapping[str, Any], name: str, *, source: Path) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"! {source}: {name} must be a non-empty string")
    return value


def _schema(row: Mapping[str, Any], expected: str, *, source: Path) -> None:
    if row.get("schema") != expected:
        raise SystemExit(f"! {source}: schema must be {expected!r}")


def _same(name: str, *values: object) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise SystemExit(f"! aggregate reports disagree on {name}: {values}")


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    try:
        root = ET.fromstring(read_text(path))
    except ET.ParseError as exc:
        raise SystemExit(f"! malformed JUnit XML: {path}: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if root.tag not in {"testsuite", "testsuites"} or not suites:
        raise SystemExit(f"! malformed JUnit XML root: {path}")

    counts: list[int] = []
    for name in ("tests", "failures", "errors", "skipped"):
        total = 0
        for suite in suites:
            raw = suite.get(name, "0")
            try:
                value = int(raw)
            except ValueError as exc:
                raise SystemExit(f"! {path}: JUnit {name} is not an integer") from exc
            if value < 0:
                raise SystemExit(f"! {path}: JUnit {name} must be non-negative")
            total += value
        counts.append(total)
    return cast(tuple[int, int, int, int], tuple(counts))


def load_status(artifacts: Path) -> EvalStatus:  # noqa: PLR0912, PLR0915
    eval_root = artifacts / "eval"
    state_path = artifacts / "state.json"
    qgen_path = eval_root / "report.json"
    pool_path = eval_root / "pool_report.json"
    qrels_path = eval_root / "qrels_report.json"
    quality_path = eval_root / "quality_report.json"
    junit_path = eval_root / "pytest.xml"
    state = _object(state_path)
    qgen = _object(qgen_path)
    pool = _object(pool_path)
    qrels = _object(qrels_path)
    quality = _object(quality_path)
    _schema(state, STATE_SCHEMA, source=state_path)
    _schema(qgen, QGEN_SCHEMA, source=qgen_path)
    _schema(pool, POOL_SCHEMA, source=pool_path)
    _schema(qrels, QRELS_SCHEMA, source=qrels_path)
    try:
        validate_quality_report(quality)
    except ValueError as exc:
        raise SystemExit(f"! {quality_path}: invalid quality report: {exc}") from exc

    documents = _mapping(state, "documents", source=state_path)
    if any(
        not isinstance(key, str) or not isinstance(value, dict) for key, value in documents.items()
    ):
        raise SystemExit(f"! {state_path}: documents must map string ids to objects")
    chunk_ids: list[str] = []
    for document in documents.values():
        raw_ids = document.get("chunk_ids")
        if not isinstance(raw_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id for chunk_id in raw_ids
        ):
            raise SystemExit(f"! {state_path}: every document needs non-empty string chunk ids")
        chunk_ids.extend(raw_ids)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise SystemExit(f"! {state_path}: chunk ids must be globally unique")
    chunks = len(chunk_ids)
    if state.get("embedding_profile") != DOCUMENT_EMBEDDING_PROFILE:
        raise SystemExit(f"! {state_path}: unexpected embedding profile")
    collection = _string(state, "collection_name", source=state_path)

    published = _mapping(qgen, "published_index", source=qgen_path)
    _same("published chunk count", chunks, _integer(published, "chunks", source=qgen_path))
    _same("published collection", collection, _string(published, "collection", source=qgen_path))
    for state_name, report_name in (
        ("chunker_fingerprint", "chunker"),
        ("embedding_profile", "embedding_profile"),
        ("scope", "scope"),
        ("sparse_fingerprint", "sparse"),
    ):
        _same(
            f"published {report_name}",
            _string(state, state_name, source=state_path),
            _string(published, report_name, source=qgen_path),
        )

    sampled = _integer(qgen, "sampled_chunks", source=qgen_path, minimum=1)
    independence = _string(qgen, "verification_independence", source=qgen_path)
    qgen_model = _string(qgen, "generator_model_requested", source=qgen_path)
    verifier_model = _string(qgen, "verifier_model_requested", source=qgen_path)
    judge_model = _string(qrels, "requested_model", source=qrels_path)
    served_model_fields = (
        (qgen, "generator_models_served", qgen_path),
        (qgen, "verifier_models_served", qgen_path),
        (qrels, "served_models", qrels_path),
    )
    served_models: set[str] = set()
    for report, name, source in served_model_fields:
        counts = _mapping(report, name, source=source)
        if not counts:
            raise SystemExit(f"! {source}: {name} must be non-empty")
        for model in counts:
            if not isinstance(model, str) or not model:
                raise SystemExit(f"! {source}: {name} keys must be non-empty strings")
            _integer(counts, model, source=source, minimum=1)
            served_models.add(model)
    label_provenance: str
    if independence == "same-requested-model self-agreement":
        _same("synthetic label requested model", qgen_model, verifier_model, judge_model)
        if served_models != {qgen_model}:
            raise SystemExit(
                "! aggregate reports disagree on served synthetic-label models: "
                f"requested={qgen_model!r}, served={sorted(served_models)!r}"
            )
        label_provenance = "same-model self-agreement"
    elif independence == "different-requested-model review":
        if qgen_model == verifier_model:
            raise SystemExit(f"! {qgen_path}: different-model review uses identical models")
        label_provenance = "different-model verification; judge provenance reported separately"
    else:
        raise SystemExit(f"! {qgen_path}: unknown verification independence {independence!r}")
    _same(
        "synthetic label reasoning effort",
        _string(qgen, "reasoning_effort", source=qgen_path),
        _string(qrels, "reasoning_effort", source=qrels_path),
    )

    pairs = _integer(qgen, "complete_pairs", source=qgen_path, minimum=1)
    verified = _integer(qgen, "pairs_verified", source=qgen_path, minimum=1)
    dropped = _integer(qgen, "pairs_dropped", source=qgen_path)
    if verified - dropped != pairs:
        raise SystemExit(f"! {qgen_path}: verified - dropped must equal complete pairs")
    queries = _integer(qgen, "queries", source=qgen_path, minimum=1)
    variants = _mapping(qgen, "queries_by_variant", source=qgen_path)
    direct = _integer(variants, "direct", source=qgen_path)
    paraphrase = _integer(variants, "paraphrase", source=qgen_path)
    if queries != direct + paraphrase or direct != pairs or paraphrase != pairs:
        raise SystemExit(f"! {qgen_path}: query variants do not form complete pairs")

    pool_queries = _integer(pool, "queries", source=pool_path, minimum=1)
    pool_pairs = _integer(pool, "pairs", source=pool_path, minimum=1)
    pool_chunks = _integer(pool, "corpus_chunks", source=pool_path, minimum=1)
    _same("query count", queries, pool_queries, _integer(qrels, "queries", source=qrels_path))
    _same("pair count", pairs, pool_pairs, _integer(qrels, "pairs", source=qrels_path))
    _same(
        "corpus chunk count",
        chunks,
        pool_chunks,
        _integer(qrels, "corpus_chunks", source=qrels_path),
    )
    _same(
        "query-set fingerprint",
        _string(pool, "query_set_fingerprint", source=pool_path),
        _string(qrels, "query_set_fingerprint_sha256", source=qrels_path),
    )
    _same(
        "pool fingerprint",
        _string(pool, "pool_fingerprint", source=pool_path),
        _string(qrels, "pool_fingerprint_sha256", source=qrels_path),
    )
    _same(
        "runs fingerprint",
        _string(pool, "runs_fingerprint", source=pool_path),
        _string(qrels, "runs_fingerprint_sha256", source=qrels_path),
    )
    quality_inputs = _mapping(quality, "inputs", source=quality_path)
    quality_design = _mapping(quality, "evaluation_design", source=quality_path)
    quality_qrels = _mapping(quality, "qrels_quality", source=quality_path)
    _same(
        "quality query count",
        queries,
        _integer(quality_design, "queries", source=quality_path, minimum=1),
        _integer(quality_qrels, "queries", source=quality_path, minimum=1),
    )
    _same(
        "quality pair count",
        pairs,
        _integer(quality_design, "pairs", source=quality_path, minimum=1),
        _integer(quality_qrels, "pairs", source=quality_path, minimum=1),
    )
    _same(
        "quality corpus count",
        chunks,
        _integer(quality_qrels, "corpus_chunks", source=quality_path, minimum=1),
    )
    for name, expected in (
        ("query_set_fingerprint_sha256", _string(pool, "query_set_fingerprint", source=pool_path)),
        ("pool_fingerprint_sha256", _string(pool, "pool_fingerprint", source=pool_path)),
        ("runs_fingerprint_sha256", _string(pool, "runs_fingerprint", source=pool_path)),
    ):
        _same(
            f"quality {name}",
            expected,
            _string(quality_inputs, name, source=quality_path),
        )

    pool_total = _integer(pool, "pool_candidates_total", source=pool_path, minimum=1)
    pool_min = _integer(pool, "pool_candidates_min", source=pool_path, minimum=1)
    pool_max = _integer(pool, "pool_candidates_max", source=pool_path, minimum=1)
    pool_mean = _number(pool, "pool_candidates_mean", source=pool_path)
    if not pool_min <= pool_mean <= pool_max or abs(pool_mean * pairs - pool_total) > 1e-7:
        raise SystemExit(f"! {pool_path}: pool candidate aggregates do not reconcile")
    quality_pool = _mapping(quality_qrels, "pool_candidates", source=quality_path)
    _same("quality pool total", pool_total, _integer(quality_pool, "total", source=quality_path))
    _same("quality pool minimum", pool_min, _integer(quality_pool, "min", source=quality_path))
    _same("quality pool mean", pool_mean, _number(quality_pool, "mean", source=quality_path))
    _same("quality pool maximum", pool_max, _integer(quality_pool, "max", source=quality_path))

    batches = _integer(qrels, "batches", source=qrels_path, minimum=1)
    _same(
        "judging cache batch count",
        batches,
        _integer(qrels, "cache_batches", source=qrels_path),
        _integer(qrels, "cache_valid_batches", source=qrels_path),
    )
    _same(
        "quality judging batch count",
        batches,
        _integer(quality_qrels, "judging_batches", source=quality_path, minimum=1),
    )
    grades = _mapping(qrels, "grades", source=qrels_path)
    quality_grades = _mapping(quality_qrels, "raw_grades", source=quality_path)
    grade_zero = _integer(grades, "0", source=qrels_path)
    grade_one = _integer(grades, "1", source=qrels_path)
    grade_two = _integer(grades, "2", source=qrels_path)
    if grade_zero + grade_one + grade_two != pool_total:
        raise SystemExit(f"! {qrels_path}: grade counts do not equal pooled candidates")
    for grade, count in zip(
        ("0", "1", "2"),
        (grade_zero, grade_one, grade_two),
        strict=True,
    ):
        _same(
            f"quality raw grade {grade}",
            count,
            _integer(quality_grades, grade, source=quality_path),
        )

    generating = _mapping(qrels, "generating_chunk_grades", source=qrels_path)
    quality_generating = _mapping(
        quality_qrels,
        "generating_chunk_grades",
        source=quality_path,
    )
    generating_counts = tuple(
        _integer(generating, str(grade), source=qrels_path) for grade in (0, 1, 2)
    )
    if sum(generating_counts) != pairs:
        raise SystemExit(f"! {qrels_path}: generating-chunk grades do not equal pairs")
    disagreements = generating_counts[0] + generating_counts[1]
    disagreement_rate = _number(qrels, "generating_chunk_disagreement_rate", source=qrels_path)
    if abs(disagreement_rate - disagreements / pairs) > 1e-12:
        raise SystemExit(f"! {qrels_path}: generating-chunk disagreement rate is inconsistent")
    for grade, count in zip(("0", "1", "2"), generating_counts, strict=True):
        _same(
            f"quality generating grade {grade}",
            count,
            _integer(quality_generating, grade, source=quality_path),
        )
    _same(
        "quality promoted generating count",
        disagreements,
        _integer(quality_qrels, "promoted_generating_chunks", source=quality_path),
    )

    arity = _mapping(qrels, "gold_arity", source=qrels_path)
    quality_arity = _mapping(quality_qrels, "gold_arity", source=quality_path)
    arity_values = tuple(_number(arity, name, source=qrels_path) for name in ("min", "mean", "max"))
    if not 1.0 <= arity_values[0] <= arity_values[1] <= arity_values[2]:
        raise SystemExit(f"! {qrels_path}: gold arity aggregates are inconsistent")
    for name, value in zip(("min", "mean", "max"), arity_values, strict=True):
        _same(
            f"quality gold arity {name}",
            value,
            _number(quality_arity, name, source=quality_path),
        )

    coverage = _mapping(qrels, "run_judged_coverage", source=qrels_path)
    quality_coverage = _mapping(
        quality_qrels,
        "run_judged_coverage",
        source=quality_path,
    )
    if set(coverage) != set(RUN_LABELS):
        raise SystemExit(f"! {qrels_path}: run coverage labels differ from the frozen systems")
    if set(quality_coverage) != set(RUN_LABELS):  # pragma: no cover
        raise AssertionError("validated quality coverage labels drifted")
    for label in RUN_LABELS:
        arm = _mapping(coverage, label, source=qrels_path)
        quality_arm = _mapping(quality_coverage, label, source=quality_path)
        _same(f"{label} coverage queries", queries, _integer(arm, "queries", source=qrels_path))
        _same(
            f"quality {label} coverage queries",
            queries,
            _integer(quality_arm, "queries", source=quality_path),
        )
        for depth in (1, 10):
            complete = _integer(arm, f"top{depth}_complete", source=qrels_path)
            rate = _number(arm, f"top{depth}_rate", source=qrels_path)
            if complete != queries or abs(rate - 1.0) > 1e-12:
                raise SystemExit(f"! {qrels_path}: {label} top-{depth} coverage is incomplete")
            _same(
                f"quality {label} top-{depth} coverage",
                complete,
                _integer(
                    quality_arm,
                    f"top{depth}_all_returned_judged",
                    source=quality_path,
                ),
            )
            _same(
                f"quality {label} top-{depth} coverage rate",
                rate,
                _number(
                    quality_arm,
                    f"top{depth}_all_returned_judged_rate",
                    source=quality_path,
                ),
            )

    tests, failures, errors, skipped = _junit_counts(junit_path)
    if tests < 1 or failures or errors or skipped:
        raise SystemExit(
            f"! quality gate is not clean: tests={tests}, failures={failures}, "
            f"errors={errors}, skipped={skipped}"
        )
    # Cross-report reconciliation above gives precise diagnostics; this is the
    # check that actually authenticates the rendered numbers.
    _authenticate_quality(
        quality,
        state=state,
        pool=pool,
        qrels=qrels,
        eval_root=eval_root,
        quality_path=quality_path,
    )
    return EvalStatus(
        documents=len(documents),
        chunks=chunks,
        collection=collection,
        sampled_chunks=sampled,
        verified_pairs=pairs,
        source_clusters=_integer(quality_design, "clusters", source=quality_path, minimum=1),
        label_provenance=label_provenance,
        queries=queries,
        direct_queries=direct,
        paraphrase_queries=paraphrase,
        pool_candidates=pool_total,
        pool_min=pool_min,
        pool_max=pool_max,
        judging_batches=batches,
        grade_zero=grade_zero,
        grade_one=grade_one,
        grade_two=grade_two,
        generating_disagreements=disagreements,
        generating_disagreement_rate=disagreement_rate,
        tests=tests,
        quality=quality,
    )


def _authenticate_quality(
    quality: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    pool: Mapping[str, Any],
    qrels: Mapping[str, Any],
    eval_root: Path,
    quality_path: Path,
) -> None:
    """Rebuild the quality report from frozen raw artifacts and demand equality.

    Structural validation alone cannot authenticate a number: any in-range mean,
    interval or p-value passes it. The evaluator is seed-deterministic and JSON
    float round-trips are exact, so recomputing from ``runs.jsonl`` / ``qrels.jsonl``
    with the report's own resamples and base seed and comparing objects binds
    every rendered statistic, and the qrels semantic digest, to the artifacts.
    """
    runs_path = eval_root / "runs.jsonl"
    qrels_rows_path = eval_root / "qrels.jsonl"
    missing = [path for path in (runs_path, qrels_rows_path) if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(
            f"! cannot authenticate {quality_path}: raw artifacts are absent: {joined}"
        )
    design = _mapping(quality, "evaluation_design", source=quality_path)
    resamples = _integer(design, "resamples", source=quality_path, minimum=1)
    base_seed = _integer(design, "base_seed", source=quality_path)
    try:
        recomputed = evaluate_tidb_quality(
            state=state,
            pool_report=pool,
            qrels_report=qrels,
            run_rows=read_jsonl(runs_path),
            qrel_rows=read_jsonl(qrels_rows_path),
            resamples=resamples,
            seed=base_seed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"! {quality_path}: offline recomputation failed: {exc}") from exc
    if recomputed != quality:
        raise SystemExit(
            f"! {quality_path}: does not match deterministic recomputation from frozen artifacts"
        )


def _quality_section(status: EvalStatus, name: str) -> Mapping[str, Any]:
    value = status.quality.get(name)
    if not isinstance(value, dict):  # pragma: no cover - strict validator guards this
        raise AssertionError(f"validated quality section {name!r} is absent")
    return value


def _quality_mean(status: EvalStatus, label: str, view: str, metric: str) -> float:
    systems = _quality_section(status, "system_metrics")
    row = systems[label][view][metric]
    return float(row["mean"])


def _fmt_interval(row: Mapping[str, Any], *, scale: float = 1.0, digits: int = 3) -> str:
    mean = float(row["mean"]) * scale
    low = float(row["low"]) * scale
    high = float(row["high"]) * scale
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _fmt_p(value: float, *, at_floor: bool) -> str:
    rendered = f"{value:.2e}" if 0.0 < value < 1e-4 else f"{value:.4f}"
    return f"{rendered}†" if at_floor else rendered


def _system_name(label: str) -> str:
    return {
        LEXICAL_LABEL: "BM25 char-bigram",
        DENSE_LABEL: "dense Qwen3-4096",
        RRF_LABEL: "RRF k=10/depth=100",
        RERANK_LABEL: "Qwen3 rerank@50",
    }[label]


def _readme_status(status: EvalStatus) -> str:
    lines = [
        "> **状态**：评估层、语料层、词法/稠密检索、MRL、RRF 与离线 rerank 深度消融均已完成；",
        "> 在线检索链路已通过真实 Milvus Lite 校验，TiDB evergreen 索引也已付费嵌入并发布：",
        (
            f"> **{status.documents:,} 篇文档 / {status.chunks:,} chunks**。另已构建 "
            f"**{status.verified_pairs:,} 组 direct/paraphrase、"
            f"{status.queries:,} 条 query** 的合成 pooled qrels；"
        ),
        (
            "> 系统级 TiDB Hit@1 / R@1 / MRR@10 / binary + graded nDCG@10 已完成；"
            f"点估计以 {status.verified_pairs:,} 个 pair 观测为权重，CI 与检验按 "
            f"{status.source_clusters:,} 个 `gold_source_key` 源聚类重采样。"
        ),
        "> 服务端 `hybrid_search` 融合仍是待验证路径，不等同于本地 exact RRF。",
        "> 下方所有数字均为本仓库脚本在真实语料上跑出的结果，非引用。",
    ]
    return "\n".join(lines)


def _readme_evidence(status: EvalStatus) -> str:
    percent = status.generating_disagreement_rate * 100.0
    systems = _quality_section(status, "system_metrics")
    design = _quality_section(status, "evaluation_design")
    lines = [
        (
            "> **TiDB 合成评测集（本地报告生成）**：从已发布的 "
            f"{status.chunks:,} 个 chunk 中按主题确定性抽样 {status.sampled_chunks:,} 个，"
            f"双阶段生成并验证后保留 {status.verified_pairs:,} 个完整 pair"
            f"（direct / paraphrase 各 {status.direct_queries:,} 条）。"
        ),
        (
            "> 四条冻结 run 在两种表面形式上按系统 top-20 取并集，并强制纳入生成 chunk，"
            f"得到 {status.pool_candidates:,} 个 pair-candidate 判断槽"
            f"（每 pair {status.pool_min:,}–{status.pool_max:,}）。"
        ),
        (
            f"> rank-blinded、固定顺序的 LLM judge 共完成 {status.judging_batches:,} 个 batch；"
            f"原始 grade 0/1/2 为 "
            f"{status.grade_zero:,}/{status.grade_one:,}/{status.grade_two:,}。"
        ),
        (
            f"> 生成 chunk 与 grade 2 规则不一致 {status.generating_disagreements:,}/"
            f"{status.verified_pairs:,}（{percent:.3f}%）。四系统实际返回的 top-1/top-10 "
            "均达到 100% **已判断覆盖**。"
        ),
        "",
        (
            "> **离线质量（overall，direct/paraphrase 先在 pair 内取均值；"
            f"括号为 source-cluster bootstrap 95% CI，{status.source_clusters:,} 个源聚类）**："
        ),
        "",
        "| 冻结系统 | Hit@1 | R@1（备选完整答案覆盖） | MRR@10 | binary nDCG@10 | graded nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in RUN_LABELS:
        overall = systems[label]["overall"]
        cells = [
            _fmt_interval(overall["full_hit_at_1"]),
            _fmt_interval(overall["full_recall_at_1"]),
            _fmt_interval(overall["full_mrr_at_10"]),
            _fmt_interval(overall["full_binary_ndcg_at_10"]),
            _fmt_interval(overall["graded_ndcg_at_10"]),
        ]
        lines.append(f"| {_system_name(label)} | " + " | ".join(cells) + " |")

    lines.extend(
        (
            "",
            "> **预声明主检验族**：主终点为 pair-mean binary nDCG@10；双尾 centred paired "
            f"source-cluster bootstrap（{int(design['resamples']):,} 次，"
            f"{int(design['clusters']):,} 个源聚类）并在以下 4 个比较内做 Holm 校正。",
            "",
            "| treatment − comparator | Δ [95% CI] | win/loss/tie | p | p(Holm) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in status.quality["primary_contrasts"]:
        delta = row["delta"]
        counts = row["counts"]
        tie = int(counts["ties_nonzero"]) + int(counts["ties_zero"])
        raw = _fmt_p(float(row["raw_p"]), at_floor=bool(row["raw_p_at_floor"]))
        adjusted = _fmt_p(
            float(row["adjusted_p"]),
            at_floor=bool(row["adjusted_p_inherits_floor"]),
        )
        label = f"{_system_name(str(row['treatment']))} − {_system_name(str(row['comparator']))}"
        lines.append(
            f"| {label} | {_fmt_interval(delta, digits=4)} | "
            f"{counts['wins']}/{counts['losses']}/{tie} | {raw} | "
            f"{adjusted}{' *' if row['reject'] else ''} |"
        )

    lines.extend(
        (
            "",
            "> **direct → paraphrase robustness（独立 4-test Holm family）**：",
            "",
            "| 系统 | direct nDCG | paraphrase nDCG | Δ(para-direct) [95% CI] | p(Holm) |",
            "|---|---:|---:|---:|---:|",
        )
    )
    robustness = {str(row["id"]): row for row in status.quality["surface_robustness"]}
    for label in RUN_LABELS:
        row = robustness[label]
        direct_mean = _quality_mean(status, label, "direct", PRIMARY_METRIC)
        paraphrase_mean = _quality_mean(status, label, "paraphrase", PRIMARY_METRIC)
        adjusted = _fmt_p(
            float(row["adjusted_p"]),
            at_floor=bool(row["adjusted_p_inherits_floor"]),
        )
        lines.append(
            f"| {_system_name(label)} | {direct_mean:.3f} | {paraphrase_mean:.3f} | "
            f"{_fmt_interval(row['delta'], digits=4)} | {adjusted}{' *' if row['reject'] else ''} |"
        )

    lines.extend(
        (
            "",
            "> **词面重叠分层（描述性，不做 subgroup p 值）**：每个 surface 内按 stored "
            "bigram containment 做保留 ties 的 mid-CDF 三分位；下表为主指标。",
            "",
            "| surface / stratum | n | overlap 范围 | BM25 | dense | RRF | rerank |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    overlap = status.quality["overlap_strata"]["tasks"]
    for task in ("direct", "paraphrase"):
        for stratum in ("low", "middle", "high"):
            row = overlap[task][stratum]
            values = [f"{float(row['systems'][label]['mean']):.3f}" for label in RUN_LABELS]
            lines.append(
                f"| {task} / {stratum} | {row['n']} | "
                f"{float(row['overlap_min']):.3f}–{float(row['overlap_max']):.3f} | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        (
            "",
            (
                f"> **边界**：这些是 {status.label_provenance} 的 synthetic pooled labels，"
                "不是 TiDB 上游人工 gold；100% 是 judged coverage，不是质量。"
            ),
            (
                "> grade 2 文档是可独立完整回答的**替代证据**，所以 Hit@1 / MRR / nDCG 是主视图，"
                "不报告要求找齐所有替代答案的 ALL@10；graded nDCG 采用 full=3、partial=1 gain。"
            ),
            (
                f"> 生成 chunk 经单独 verification pass 证实（{status.label_provenance}）；"
                "即使 relevance judge 给 0/1，仍作为 operational full gold。"
                "pool 外保持未判断；本评测未用于反向调参。"
            ),
            (
                "> † 表示 add-one Monte Carlo floor，不是严格 `<` 上界；CI 是 pointwise，"
                "同一 source 的 pair 已整簇重采样，跨 source/theme 的残余相关性未建模。"
            ),
        )
    )
    return "\n".join(lines)


def _quality_gate(status: EvalStatus) -> str:
    # Only the JUnit report is parsed here, so only the pytest result may be
    # claimed. Ruff / format / mypy are separate pre-commit gates that this
    # synchronizer does not read and therefore must not certify.
    return (
        f"tests/                   {status.tests:,} 个单元测试\n"
        "```\n\n"
        f"质量门禁（本行仅由 `pytest.xml` 生成）：`pytest` {status.tests:,} passed。"
        "`ruff check` / `ruff format --check` / `mypy --strict` 是独立的提交前门禁，"
        "不由本报告认证。"
    )


def _architecture_corpus(status: EvalStatus) -> str:
    lines = [
        (
            "| | `crud-rag-subset`（5,681 篇干扰语料） | "
            f"`tidb-rag-curated`（{status.documents:,} 篇 / "
            f"{status.documents:,} 篇入库 / {status.chunks:,} chunks） |"
        ),
        "|---|---|---|",
        (
            "| **角色** | **探索性检索 benchmark**——现有消融数字来源 | "
            "**部署集 + 合成 pooled qrels**——公网 demo 与域内评测输入 |"
        ),
        (
            "| **有无 gold label** | 有上游 `evidence_document_id` | "
            "**无上游人工 gold**；已有 direct/paraphrase QG、四系统 pooling、"
            "rank-blinded LLM judge 形成的合成 qrels |"
        ),
        (
            "| **在它上面调什么** | 已探索 analyzer、fusion 权重、rerank 深度、MRL 维度 | "
            "当前冻结配置，不用这批 query 反向调参；若要调，先拆 dev/test 或明确为探索性 |"
        ),
        (
            "| **在它上面测什么** | R@1 / MRR@10 / nDCG@10 / ALL-gold@10 + 配对检验 | "
            "已完成 Hit@1 / R@1 / MRR@10 / binary+graded nDCG@10、source-cluster bootstrap、"
            "direct/paraphrase 与描述性词面重叠分层；HTTP 延迟/QPS 已由 M8 独立认证，"
            "增量重建仍待测 |"
        ),
        "",
        (
            f"当前 TiDB qrels 覆盖 {status.verified_pairs:,} 个 pair / "
            f"{status.queries:,} 条 query / {status.pool_candidates:,} 个 pooled candidates，"
            f"分属 {status.source_clusters:,} 个 `gold_source_key` 源聚类。"
        ),
        (
            "四条冻结 run 的 top-1/top-10 候选均已判断，但 **100% judged coverage "
            "不是 100% retrieval quality**；质量表已用 source-clustered 95% CI、双尾配对 "
            "source-cluster bootstrap 与 Holm 校正生成。"
        ),
    ]
    return "\n".join(lines)


def _architecture_m3(status: EvalStatus) -> str:
    bm25 = _quality_mean(status, LEXICAL_LABEL, "overall", PRIMARY_METRIC)
    dense = _quality_mean(status, DENSE_LABEL, "overall", PRIMARY_METRIC)
    rrf = _quality_mean(status, RRF_LABEL, "overall", PRIMARY_METRIC)
    rerank = _quality_mean(status, RERANK_LABEL, "overall", PRIMARY_METRIC)
    return "".join(
        (
            "| **M3** | **TiDB 全量索引 + 合成 pooled qrels + 离线质量 ✅ 2026-08-26** | ",
            "**1.5** | ✅ `ingest.py` + `scripts/build_index.py` + `scripts/query_index.py`；✅ ",
            "`scripts/build_tidb_{queries,pool,qrels}.py`；✅ `evaluate_tidb_retrieval.py` | ",
            f"{status.documents:,} 篇 evergreen → **{status.chunks:,} chunks**；",
            f"**{status.verified_pairs:,} pairs / {status.queries:,} queries / ",
            f"{status.pool_candidates:,} pooled candidates**。overall binary nDCG@10：",
            f"BM25 {bm25:.3f} / dense {dense:.3f} / RRF {rrf:.3f} / rerank {rerank:.3f}；",
            f"以 {status.source_clusters:,} 个 source cluster 为重采样单位的 95% CI + ",
            "两个预声明 4-test 双尾 source-cluster bootstrap/Holm family。",
            "无上游人工 gold；本地 exact RRF 不代表服务端 hybrid_search |",
        )
    )


def _claude_artifacts(status: EvalStatus) -> str:
    rows = [
        ("路径", "大小", "怎么来的"),
        ("---", "---", "---"),
        (
            "`crud-rag-subset/raw/split_merged.json`",
            "26 MB",
            "`crud-rag-subset/build_subset.ps1`",
        ),
        (
            "`crud-rag-subset/eval-expanded/{corpus,qrels}.jsonl`",
            "11 MB",
            "`scripts/build_eval_corpus.py`（不联网）",
        ),
        (
            "`crud-rag-subset/eval-expanded/emb_cache_4096.jsonl`",
            "523 MB",
            "`scripts/probe_mrl_quality.py`，5,681 篇 @4096 维",
        ),
        (
            "`crud-rag-subset/eval-expanded/emb_cache_queries_4096.jsonl`",
            "220 MB",
            "同上 + `scripts/embed_queries.py`，2,394 条 query（独立 prompt）",
        ),
        (
            "`tidb-rag-curated/documents/`",
            "6.6 MB",
            "`tidb-rag-curated/download_curated.ps1`",
        ),
        (
            "`indexes/tidb/dense_cache.jsonl`",
            "161 MB",
            f"`scripts/build_index.py --embed --publish`，{status.chunks:,} chunks（付费）",
        ),
        (
            "`indexes/tidb/{sparse_index.json,state.json,milvus.db}`",
            "2.1 MB / 263 KB / —",
            "同上；发布词表、状态与本地库",
        ),
        (
            "`indexes/tidb/eval/{generation,verification}_cache.jsonl` 等",
            "—",
            (
                "`scripts/build_tidb_queries.py --generate/--verify`；"
                f"{status.verified_pairs:,} pairs / {status.queries:,} queries（付费 chat）"
            ),
        ),
        (
            "`indexes/tidb/eval/query_embeddings_4096.jsonl`",
            "—",
            "`scripts/build_tidb_pool.py --embed`（付费 embedding）",
        ),
        (
            "`indexes/tidb/eval/rerank_scores_top100.jsonl`",
            "—",
            "`scripts/build_tidb_pool.py --rerank`（付费 rerank）",
        ),
        (
            "`indexes/tidb/eval/{runs,pool}.jsonl`、`pool_report.json`",
            "—",
            f"`scripts/build_tidb_pool.py`；{status.pool_candidates:,} slots（离线）",
        ),
        (
            "`indexes/tidb/eval/judging_cache.jsonl`",
            "—",
            f"`scripts/build_tidb_qrels.py --judge`；{status.judging_batches:,} batches（付费）",
        ),
        (
            "`indexes/tidb/eval/qrels.jsonl`、`qrels_report.json`",
            "—",
            f"`scripts/build_tidb_qrels.py --finalize`；{status.queries:,} qrels（离线）",
        ),
        (
            "`indexes/tidb/eval/quality_report.json`",
            "—",
            (
                "`scripts/evaluate_tidb_retrieval.py`；source-clustered CI / "
                "paired source-cluster bootstrap / Holm（离线）"
            ),
        ),
    ]
    table = "\n".join(f"| {path} | {size} | {source} |" for path, size, source in rows)
    note = (
        "以上全部是本地输入或派生产物，均不得提交。TiDB 链路的付费步骤不止索引 "
        "embedding：还包括 QG/验证、query embedding、rerank 与 qrels judging；缓存完整后的 "
        "pool、finalize、指标计算和文档同步才是离线步骤。"
    )
    return f"{table}\n\n{note}"


def _replace_region(text: str, name: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {name!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _render_target(target: _Target, status: EvalStatus) -> tuple[str, str]:
    before = read_text(target.path)
    after = before
    for name, render in target.regions.items():
        after = _replace_region(after, name, render(status), path=target.path)
    return before, after


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(
            args.readme,
            {
                "TIDB-EVAL-STATUS": _readme_status,
                "TIDB-EVAL-EVIDENCE": _readme_evidence,
                "QUALITY-GATE-STATUS": _quality_gate,
            },
        ),
        _Target(
            args.architecture,
            {
                "TIDB-CORPUS-EVAL-STATUS": _architecture_corpus,
                "M3-TIDB-EVAL-STATUS": _architecture_m3,
            },
        ),
        _Target(
            args.claude_context,
            {"TIDB-LOCAL-ARTIFACTS": _claude_artifacts},
        ),
    )


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    # Every documentation synchronizer takes the repository-wide docs lock first,
    # then its artifact lock. The fixed order serializes replacement of the three
    # shared tracked documents without introducing a cross-report deadlock.
    with (
        exclusive_lock(args.readme.parent / DOCS_LOCK),
        exclusive_lock(args.artifacts / "eval" / ARTIFACT_LOCK),
    ):
        status = load_status(args.artifacts)
        rendered = [(target.path, *_render_target(target, status)) for target in _targets(args)]
        stale = tuple(path for path, before, after in rendered if before != after)
        if args.check:
            if stale:
                joined = ", ".join(str(path) for path in stale)
                raise SystemExit(f"! generated documentation is stale: {joined}")
            return ()
        if not stale:
            return ()

        staged: list[tuple[Path, Path]] = []
        try:
            for path, before, after in rendered:
                if before == after:
                    continue
                temporary = path.with_suffix(path.suffix + ".sync.tmp")
                write_text(temporary, after)
                staged.append((temporary, path))
            replace_files(staged)
        finally:
            for temporary, _path in staged:
                temporary.unlink(missing_ok=True)
    return stale


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    changed = synchronize(args)
    if args.check:
        print("TiDB evaluation documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} documentation files")
    else:
        print("TiDB evaluation documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
