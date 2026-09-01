"""Synchronize tracked M9a documentation from aggregate-only evidence.

The evidence file is deliberately a small, reviewed allowlist.  It contains
source metadata and historical Table 8 aggregates, never a query, answer,
passage, document id, chunk or provider payload.  This script does not access
any ignored evaluation artifact and performs no network or model operation.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.eval.metrics_gen import (
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_IDF,
    BERTSCORE_LANG,
    BERTSCORE_MODEL,
    BERTSCORE_NUM_LAYERS,
    BERTSCORE_RESCALE_WITH_BASELINE,
    BERTSCORE_USE_FAST_TOKENIZER,
    CRUD_BLEU_METRIC,
    ROUGE_L_METRIC,
    STANDARD_BLEU_METRIC,
)
from zhrag.eval.quest_eval import QUEST_EVAL_SCHEMA, UNANSWERABLE_SENTINEL
from zhrag.io_utils import exclusive_lock, read_json, read_text, replace_files, write_text

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "docs" / "evidence" / "crud_rag_table8_v3.json"
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "architecture-decision.md"
DOCS_LOCK = ".docs.lock"

SCHEMA = "zhrag-crud-rag-table8-evidence-v1"
README_MARKER = "M9A-GENERATION-METRICS"
ARCHITECTURE_MARKER = "M9A-GENERATION-METRICS"
EXPECTED_PDF_SHA256 = "2e4ae0cb708fdca9d96bcf8d1c0713dae121195a0a31b78e3132e9ef4fa7db8a"
EXPECTED_TASK_MODELS = (
    ("summarization", "Qwen-14B"),
    ("summarization", "GPT-4-0613"),
    ("question answering 1-document", "Qwen-14B"),
    ("question answering 1-document", "GPT-4-0613"),
)
_NUMERIC_ROW_KEYS = (
    "bleu",
    "rouge_l",
    "bert_score",
    "ragquest_precision",
    "ragquest_recall",
    "length",
)
_EXPECTED_ROW_VALUES = {
    ("summarization", "Qwen-14B"): (32.51, 33.33, 85.62, 68.94, 40.57, 139.1),
    ("summarization", "GPT-4-0613"): (24.54, 35.91, 89.39, 71.24, 50.53, 194.6),
    ("question answering 1-document", "Qwen-14B"): (
        37.95,
        55.13,
        83.25,
        53.03,
        73.92,
        73.8,
    ),
    ("question answering 1-document", "GPT-4-0613"): (
        33.87,
        51.42,
        80.92,
        53.14,
        62.39,
        95.9,
    ),
}
_SOURCE_KEYS = {
    "title",
    "arxiv_id",
    "arxiv_version",
    "doi",
    "pdf_url",
    "html_url",
    "pdf_sha256",
    "pdf_page",
    "table",
}
_UPSTREAM_KEYS = {
    "repository",
    "commit",
    "license_file",
    "github_license_metadata",
    "license_status_date",
}
_ROW_KEYS = {
    "task",
    "model",
    "bleu",
    "rouge_l",
    "bert_score",
    "ragquest_precision",
    "ragquest_recall",
    "length",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TEXT_KEYS = {
    "answer",
    "answers",
    "chunk",
    "chunk_id",
    "document",
    "document_id",
    "embedding",
    "passage",
    "payload",
    "provider",
    "query",
    "question",
    "reference",
    "text",
    "vector",
}


@dataclass(frozen=True, slots=True)
class _Target:
    path: Path
    marker: str
    render: Callable[[Mapping[str, Any]], str]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize M9a generation-metric documentation from tracked evidence."
    )
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--check", action="store_true", help="fail if tracked docs are stale")
    return parser.parse_args(argv)


def _object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        forbidden = sorted(set(extra) & _FORBIDDEN_TEXT_KEYS)
        suffix = f"; forbidden text keys={forbidden}" if forbidden else ""
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}{suffix}")


def _string(value: Mapping[str, Any], name: str, *, label: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{label}.{name} must be a non-empty string")
    return result


def _number(
    value: Mapping[str, Any],
    name: str,
    *,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{label}.{name} must be numeric")
    number = float(result)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{label}.{name} must be finite and >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label}.{name} must be <= {maximum}")
    return number


def _validate_source(evidence: Mapping[str, Any]) -> None:
    source = _object(evidence.get("source"), label="evidence.source")
    _exact_keys(source, _SOURCE_KEYS, label="evidence.source")
    expected = {
        "title": (
            "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented "
            "Generation of Large Language Models"
        ),
        "arxiv_id": "2401.17043",
        "arxiv_version": 3,
        "doi": "10.48550/arXiv.2401.17043",
        "pdf_url": "https://arxiv.org/pdf/2401.17043v3",
        "html_url": "https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8",
        "pdf_page": 26,
        "table": 8,
    }
    for name, expected_value in expected.items():
        if source.get(name) != expected_value:
            raise ValueError(f"evidence.source.{name} must be {expected_value!r}")
    digest = _string(source, "pdf_sha256", label="evidence.source")
    if _HEX64.fullmatch(digest) is None:
        raise ValueError("evidence.source.pdf_sha256 must be 64 lowercase hex characters")
    if digest != EXPECTED_PDF_SHA256:
        raise ValueError("evidence.source.pdf_sha256 does not match the audited PDF")


def _validate_upstream(evidence: Mapping[str, Any]) -> None:
    upstream = _object(evidence.get("upstream"), label="evidence.upstream")
    _exact_keys(upstream, _UPSTREAM_KEYS, label="evidence.upstream")
    if _string(upstream, "repository", label="evidence.upstream") != (
        "https://github.com/IAAR-Shanghai/CRUD_RAG"
    ):
        raise ValueError("evidence.upstream.repository is not the fixed CRUD-RAG repository")
    if _string(upstream, "commit", label="evidence.upstream") != (
        "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"
    ):
        raise ValueError("evidence.upstream.commit is not the audited commit")
    if upstream.get("license_file") is not False:
        raise ValueError("evidence.upstream.license_file must be false")
    if upstream.get("github_license_metadata") is not None:
        raise ValueError("evidence.upstream.github_license_metadata must be null")
    if _string(upstream, "license_status_date", label="evidence.upstream") != "2026-08-31":
        raise ValueError("evidence.upstream.license_status_date is not frozen")


def _validate_rows(evidence: Mapping[str, Any]) -> None:
    raw_rows = evidence.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(EXPECTED_TASK_MODELS):
        raise ValueError(f"evidence.rows must contain exactly {len(EXPECTED_TASK_MODELS)} rows")
    seen: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(raw_rows):
        row = _object(raw_row, label=f"evidence.rows[{index}]")
        _exact_keys(row, _ROW_KEYS, label=f"evidence.rows[{index}]")
        key = (
            _string(row, "task", label=f"evidence.rows[{index}]"),
            _string(row, "model", label=f"evidence.rows[{index}]"),
        )
        if key not in EXPECTED_TASK_MODELS:
            raise ValueError(f"evidence.rows[{index}] has an unexpected task/model key: {key!r}")
        if key in seen:
            raise ValueError(f"evidence.rows repeats task/model key: {key!r}")
        seen.add(key)
        label = f"evidence.rows[{index}]"
        actual_values = tuple(
            _number(
                row,
                name,
                label=label,
                minimum=0.0,
                maximum=None if name == "length" else 100.0,
            )
            for name in _NUMERIC_ROW_KEYS
        )
        if actual_values != _EXPECTED_ROW_VALUES[key]:
            raise ValueError(f"{label} values do not match the audited Table 8 row")
    if seen != set(EXPECTED_TASK_MODELS):
        raise ValueError("evidence.rows do not contain the fixed four Table 8 anchors")


def validate_evidence(value: object) -> Mapping[str, Any]:
    """Validate the complete aggregate-only evidence contract."""

    evidence = _object(value, label="evidence")
    _exact_keys(evidence, {"schema", "source", "upstream", "rows"}, label="evidence")
    if evidence.get("schema") != SCHEMA:
        raise ValueError(f"evidence.schema must be {SCHEMA!r}")
    _validate_source(evidence)
    _validate_upstream(evidence)
    _validate_rows(evidence)
    return evidence


def load_evidence(path: Path) -> Mapping[str, Any]:
    """Read and validate evidence through the project's UTF-8 JSON port."""

    try:
        return validate_evidence(read_json(path))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"! invalid M9a evidence {path}: {exc}") from exc


def _rows(evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = evidence["rows"]
    if not isinstance(rows, list):  # pragma: no cover - validator guards this
        raise AssertionError("validated evidence rows are not a list")
    return tuple(_object(row, label="validated Table 8 row") for row in rows)


def _fmt(value: object, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _source_line(evidence: Mapping[str, Any]) -> str:
    source = _object(evidence["source"], label="validated source")
    return (
        f"历史值来自 [arXiv 2401.17043v3]({source['pdf_url']}) 的 Table "
        f"{source['table']}（HTML 锚点：[{source['html_url']}]({source['html_url']})，"
        f"PDF 第 {source['pdf_page']} 页；副本 SHA-256 `{source['pdf_sha256']}`）。"
    )


def _table(evidence: Mapping[str, Any]) -> str:
    lines = [
        (
            "| task | model | BLEU | ROUGE-L | `bertScore`（上游列名） | "
            "RAGQuest precision | RAGQuest recall | length |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _rows(evidence):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["task"]),
                    str(row["model"]),
                    _fmt(row["bleu"], 2),
                    _fmt(row["rouge_l"], 2),
                    _fmt(row["bert_score"], 2),
                    _fmt(row["ragquest_precision"], 2),
                    _fmt(row["ragquest_recall"], 2),
                    _fmt(row["length"], 1),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _contract_lines() -> tuple[str, ...]:
    return (
        f"- `{STANDARD_BLEU_METRIC}`：单 reference、token-level modified 1–4 gram precision、",
        "  几何均值与标准 brevity penalty；无 smoothing、无 effective-order，逐样本后取算术平均。",
        f"- `{CRUD_BLEU_METRIC}`：同一逐样本算法但移除 brevity penalty，只作兼容审计列，",
        "  不称为 standard BLEU；两者都不是 corpus BLEU。",
        f"- `{ROUGE_L_METRIC}`：token LCS 的 sentence-level precision/recall/F1（beta=1），",
        "  逐样本后取算术平均；是 rougeL，不是 rougeLsum。",
        (
            f"- `{BERTSCORE_MODEL}` 的真实 BERTScore 为可选模型适配器：lang=`{BERTSCORE_LANG}`、"
            f"layer={BERTSCORE_NUM_LAYERS}、rescale_with_baseline={BERTSCORE_RESCALE_WITH_BASELINE}、"
            f"idf={BERTSCORE_IDF}、batch={BERTSCORE_BATCH_SIZE}、"
            f"use_fast_tokenizer={BERTSCORE_USE_FAST_TOKENIZER}。"
        ),
        (
            f"- `{QUEST_EVAL_SCHEMA}`：无法回答哨兵为 `{UNANSWERABLE_SENTINEL}`；"
            "报告论文全问题分母与历史代码条件分母，不把空条件集合伪造为 0。"
        ),
    )


def render_readme(evidence: Mapping[str, Any]) -> str:
    """Render the README marker body from validated evidence."""

    validate_evidence(evidence)
    lines = [
        "### 中文生成指标与 CRUD-RAG Table 8 历史证据",
        "",
        "M9a 冻结的是可复现的指标合同，不是一次真实生成实验。词法指标使用注入式 tokenizer，",
        "默认 CI 不加载模型；任一行 tokenization、语义评分、长度或数值校验失败，",
        "整个 report 拒绝发布，不会把失败行静默转成 0 或从 denominator 中删除。",
        "",
        "**本项目的 canonical contracts**：",
        *_contract_lines(),
        "",
        "**CRUD-RAG Table 8（历史 0–100 表格值，仅作来源锚点）**：",
        "",
        _table(evidence),
        "",
        _source_line(evidence),
        (
            "这些值不是本项目实现的 golden test：论文没有冻结足够的 tokenization、依赖版本、"
            "smoothing 或模型参数，且上游 `bertScore` 实际是 `text2vec-base-chinese` 句向量相似度，"
            "不是真 BERTScore。不要把它与本项目的真实 BERTScore 列直接横比。"
        ),
        (
            "上游仓库固定为 `IAAR-Shanghai/CRUD_RAG@"
            f"{_object(evidence['upstream'], label='validated upstream')['commit']}`；截至 "
            "2026-08-31 仍无 LICENSE。这里只引用聚合事实和公开定义，未复制上游代码或数据。"
        ),
    ]
    return "\n".join(lines)


def render_architecture(evidence: Mapping[str, Any]) -> str:
    """Render the concise ADR M9a contract/evidence marker body."""

    validate_evidence(evidence)
    upstream = _object(evidence["upstream"], label="validated upstream")
    return "\n".join(
        (
            "**M9a：独立生成指标合同 + Table 8 证据（无付费调用）**",
            "",
            "M9a 只交付纯内存评分、合成测试、来源证据和默认 CI 隔离；不读取 CRUD-RAG/TiDB 语料，"
            "不生成答案，不提交 `quest_gt`，不调用 QG/QA/chat/embedding/rerank/judge/Milvus。真实 "
            "event summarization 与 QA-1doc 生成实验留给 M9b。",
            "",
            "**固定合同**：",
            *_contract_lines(),
            "",
            (
                f"Table 8 只作为历史来源锚点：{_source_line(evidence)} 上游固定为 "
                f"`{upstream['repository']}@{upstream['commit']}`，许可状态仍为无 LICENSE；"
                "其聚合值不构成本项目数值等价或实现复用许可。"
            ),
        )
    )


def _replace_region(text: str, marker: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {marker!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def _targets(args: argparse.Namespace) -> tuple[_Target, ...]:
    return (
        _Target(args.readme, README_MARKER, render_readme),
        _Target(args.architecture, ARCHITECTURE_MARKER, render_architecture),
    )


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    """Validate, render and atomically publish the two tracked M9a regions."""

    with exclusive_lock(args.readme.parent / DOCS_LOCK):
        evidence = load_evidence(args.evidence)
        rendered: list[tuple[Path, str, str]] = []
        for target in _targets(args):
            before = read_text(target.path)
            after = _replace_region(
                before,
                target.marker,
                target.render(evidence),
                path=target.path,
            )
            rendered.append((target.path, before, after))
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
        print("M9a documentation is synchronized")
    elif changed:
        print(f"synchronized {len(changed)} M9a documentation files")
    else:
        print("M9a documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
