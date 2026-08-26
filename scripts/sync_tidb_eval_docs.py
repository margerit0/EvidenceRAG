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
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from zhrag.eval.tidb_runs import DENSE_LABEL, LEXICAL_LABEL, RERANK_LABEL, RRF_LABEL
from zhrag.io_utils import read_json, read_text, replace_files, write_text

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "indexes" / "tidb"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
CLAUDE_CONTEXT = ROOT / "CLAUDE.md"

STATE_SCHEMA = "zhrag-ingest-state-v1"
QGEN_SCHEMA = "zhrag-tidb-qgen-v1"
POOL_SCHEMA = "zhrag-tidb-runs-v1"
QRELS_SCHEMA = "zhrag-tidb-qrels-v1"
DOCUMENT_EMBEDDING_PROFILE = "qwen3-embedding-8b-tidb-doc-4096-v1"
RUN_LABELS = (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)


@dataclass(frozen=True, slots=True)
class EvalStatus:
    documents: int
    chunks: int
    collection: str
    sampled_chunks: int
    verified_pairs: int
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
    return float(value)


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
    junit_path = eval_root / "pytest.xml"
    state = _object(state_path)
    qgen = _object(qgen_path)
    pool = _object(pool_path)
    qrels = _object(qrels_path)
    _schema(state, STATE_SCHEMA, source=state_path)
    _schema(qgen, QGEN_SCHEMA, source=qgen_path)
    _schema(pool, POOL_SCHEMA, source=pool_path)
    _schema(qrels, QRELS_SCHEMA, source=qrels_path)

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

    pool_total = _integer(pool, "pool_candidates_total", source=pool_path, minimum=1)
    pool_min = _integer(pool, "pool_candidates_min", source=pool_path, minimum=1)
    pool_max = _integer(pool, "pool_candidates_max", source=pool_path, minimum=1)
    pool_mean = _number(pool, "pool_candidates_mean", source=pool_path)
    if not pool_min <= pool_mean <= pool_max or abs(pool_mean * pairs - pool_total) > 1e-7:
        raise SystemExit(f"! {pool_path}: pool candidate aggregates do not reconcile")

    batches = _integer(qrels, "batches", source=qrels_path, minimum=1)
    _same(
        "judging cache batch count",
        batches,
        _integer(qrels, "cache_batches", source=qrels_path),
        _integer(qrels, "cache_valid_batches", source=qrels_path),
    )
    grades = _mapping(qrels, "grades", source=qrels_path)
    grade_zero = _integer(grades, "0", source=qrels_path)
    grade_one = _integer(grades, "1", source=qrels_path)
    grade_two = _integer(grades, "2", source=qrels_path)
    if grade_zero + grade_one + grade_two != pool_total:
        raise SystemExit(f"! {qrels_path}: grade counts do not equal pooled candidates")

    generating = _mapping(qrels, "generating_chunk_grades", source=qrels_path)
    generating_counts = tuple(
        _integer(generating, str(grade), source=qrels_path) for grade in (0, 1, 2)
    )
    if sum(generating_counts) != pairs:
        raise SystemExit(f"! {qrels_path}: generating-chunk grades do not equal pairs")
    disagreements = generating_counts[0] + generating_counts[1]
    disagreement_rate = _number(qrels, "generating_chunk_disagreement_rate", source=qrels_path)
    if abs(disagreement_rate - disagreements / pairs) > 1e-12:
        raise SystemExit(f"! {qrels_path}: generating-chunk disagreement rate is inconsistent")

    coverage = _mapping(qrels, "run_judged_coverage", source=qrels_path)
    if set(coverage) != set(RUN_LABELS):
        raise SystemExit(f"! {qrels_path}: run coverage labels differ from the frozen systems")
    for label in RUN_LABELS:
        arm = _mapping(coverage, label, source=qrels_path)
        _same(f"{label} coverage queries", queries, _integer(arm, "queries", source=qrels_path))
        for depth in (1, 10):
            complete = _integer(arm, f"top{depth}_complete", source=qrels_path)
            rate = _number(arm, f"top{depth}_rate", source=qrels_path)
            if complete != queries or abs(rate - 1.0) > 1e-12:
                raise SystemExit(f"! {qrels_path}: {label} top-{depth} coverage is incomplete")

    tests, failures, errors, skipped = _junit_counts(junit_path)
    if tests < 1 or failures or errors or skipped:
        raise SystemExit(
            f"! quality gate is not clean: tests={tests}, failures={failures}, "
            f"errors={errors}, skipped={skipped}"
        )
    return EvalStatus(
        documents=len(documents),
        chunks=chunks,
        collection=collection,
        sampled_chunks=sampled,
        verified_pairs=pairs,
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
    )


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
            "> 系统级 TiDB R@1 / MRR@10 / nDCG@10 尚未从该 qrels 计算，"
            "服务端 `hybrid_search` 融合仍是待验证的优化路径。"
        ),
        "> 下方所有数字均为本仓库脚本在真实语料上跑出的结果，非引用。",
    ]
    return "\n".join(lines)


def _readme_evidence(status: EvalStatus) -> str:
    percent = status.generating_disagreement_rate * 100.0
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
            f"{status.verified_pairs:,}（{percent:.3f}%）。四系统 top-1/top-10 均达到 "
            "100% **已判断覆盖**。"
        ),
        (
            "> **边界**：这些是同一请求模型完成生成、验证与相关性判断的合成 pooled labels，"
            "不是 TiDB 上游人工 gold；100% 表示候选已被判断，不是检索准确率。"
        ),
        (
            "> 当前还没有系统质量表。若用这批 query 继续调参，必须另拆 dev/test，"
            "或把结果明确标为探索性。"
        ),
        (
            "> 生成 chunk 是构题后由独立 verification pass 验证的证据；即使相关性 judge "
            "给 0/1，发布 qrels 仍将其保留为 verified gold。pool 外文档保持未判断。"
        ),
    ]
    return "\n".join(lines)


def _quality_gate(status: EvalStatus) -> str:
    return (
        f"tests/                   {status.tests:,} 个单元测试\n"
        "```\n\n"
        f"质量门禁：`pytest` {status.tests:,} passed · `ruff check` 全通过 · "
        "`ruff format --check` 全通过 · `mypy --strict` 无告警。"
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
            "待生成系统级指标、direct/paraphrase 与词面重叠分层；另测延迟/QPS/重建 |"
        ),
        "",
        (
            f"当前 TiDB qrels 覆盖 {status.verified_pairs:,} 个 pair / "
            f"{status.queries:,} 条 query / {status.pool_candidates:,} 个 pooled candidates。"
        ),
        (
            "四条冻结 run 的 top-1/top-10 候选均已判断，但 **100% judged coverage "
            "不是 100% retrieval quality**；在系统级指标与 95% CI / 配对检验产出前，"
            "不从覆盖率推导质量结论。"
        ),
    ]
    return "\n".join(lines)


def _architecture_m3(status: EvalStatus) -> str:
    return "".join(
        (
            "| **M3** | **TiDB 全量索引 + 合成 pooled qrels ✅ 2026-08-25** | **1.5** | ",
            "✅ `ingest.py` + `scripts/build_index.py` + `scripts/query_index.py`；✅ ",
            "`scripts/build_tidb_{queries,pool,qrels}.py`（双表面 QG、四系统 pool、",
            "显式 finalize） | ",
            f"{status.documents:,} 篇 evergreen → **{status.chunks:,} chunks** 已索引发布；",
            f"合成评测集为 **{status.verified_pairs:,} pairs / {status.queries:,} queries / ",
            f"{status.pool_candidates:,} pooled candidates / ",
            f"{status.judging_batches:,} judge batches**。",
            "四系统 top-1/top-10 判断覆盖完整，但系统级指标尚待离线计算；",
            "无上游人工 gold |",
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
