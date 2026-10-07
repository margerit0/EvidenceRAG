"""Recompute the README ablation summary offline, preserving historical test families.

No provider is instantiated. Existing experiment scripts supply their original
ranking, cache provenance and M7 authentication contracts; only aggregates are
published. Cache misses fail before either document is replaced.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from zhrag.eval.crud import sample_corpus
from zhrag.eval.hybrid_mrl1024 import HybridInputs, load_hybrid_mrl1024_inputs
from zhrag.eval.metrics import (
    bootstrap_ci,
    bootstrap_p_floor,
    holm_bonferroni,
    holm_floor_flags,
    mcnemar_exact,
    paired_bootstrap_test,
    recall_at_k,
)
from zhrag.eval.rerank import paired_metric_family
from zhrag.eval.retrieval import bm25_runs, dense_runs, per_query_metrics, prefix_l2_normalize
from zhrag.eval.tidb_chunk_sweep_artifacts import ChunkSweepPaths
from zhrag.io_utils import exclusive_lock, read_text, replace_files, write_text
from zhrag.retrieval import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parent.parent
EXPANDED = ROOT / "crud-rag-subset" / "eval-expanded"
DOCS_LOCK = ".docs.lock"
RESAMPLES = 10_000
MARKER = "ABLATION-SUMMARY"


def _script(name: str) -> ModuleType:
    """Reuse a CLI's existing pure helpers without executing its main function."""
    spec = importlib.util.spec_from_file_location(
        f"ablation_{name}", ROOT / "scripts" / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import experiment helper: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    label: str,
    metric: str,
    before: Sequence[float],
    after: Sequence[float],
    p: float,
    *,
    method: str,
    at_floor: bool = False,
    scale: float = 100,
) -> str:
    diffs = [right - left for left, right in zip(before, after, strict=True)]
    ci = bootstrap_ci(diffs, resamples=RESAMPLES)
    return _interval_row(
        label, metric, ci.mean, ci.low, ci.high, p=p, method=method, scale=scale, at_floor=at_floor
    )


def _interval_row(
    label: str,
    metric: str,
    delta: float,
    low: float,
    high: float,
    *,
    p: float,
    method: str,
    scale: float = 100,
    at_floor: bool = False,
) -> str:
    unit = "pp" if scale == 100 else ""
    digits = 2 if scale == 100 else 4
    interval = f"[{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]{unit}"
    rendered_p = f"{p:.2e}" if 0 < p < 0.0001 else f"{p:.4f}"
    rendered_p += "†" if at_floor else ""
    return (
        f"| {label} | {metric} | {delta * scale:+.{digits}f}{unit} | "
        f"{interval} | {rendered_p} | {method} |"
    )


def _mrl_rows(inputs: HybridInputs) -> list[str]:
    # Use the original probe's task, document order and ranking contract, not
    # the rerank experiment's actual-arity groups or a newly selected test.
    probe = _script("probe_mrl_quality")
    selected = [i for i, q in enumerate(inputs.queries) if q.task == "questanswer_1doc"]
    queries = [inputs.queries[i] for i in selected]
    if len(queries) != 800 or any(len(q.gold_doc_ids) != 1 for q in queries):
        raise ValueError("MRL requires the frozen 800 single-gold queries")
    corpus = sample_corpus(dict(inputs.corpus), queries, size=5681)
    positions = {doc_id: i for i, doc_id in enumerate(inputs.doc_ids)}
    docs = inputs.doc_matrix[[positions[doc_id] for doc_id in corpus]]
    query_matrix = inputs.query_matrix[selected]
    scores: dict[int, list[float]] = {}
    for dim in probe.DIMS:
        ranked = probe._rank(
            query_matrix if dim == 4096 else prefix_l2_normalize(query_matrix, dim),
            docs if dim == 4096 else prefix_l2_normalize(docs, dim),
            list(corpus),
        )
        scores[dim] = [
            recall_at_k(run, query.gold_doc_ids, 1)
            for run, query in zip(ranked, queries, strict=True)
        ]
    raw = {
        str(dim): paired_bootstrap_test(scores[dim], scores[4096], resamples=RESAMPLES)
        for dim in probe.DIMS
        if dim != 4096
    }
    corrected = holm_bonferroni(raw)
    floors = holm_floor_flags(
        raw, {key: value <= bootstrap_p_floor(RESAMPLES) + 1e-12 for key, value in raw.items()}
    )
    return [
        _row(
            f"向量 {dim} 维 vs 4096 维",
            "R@1（原单证据任务）",
            scores[4096],
            scores[dim],
            corrected[str(dim)][0],
            method="单尾配对 bootstrap；六项 Holm",
            at_floor=floors[str(dim)],
        )
        for dim in (1024, 64)
    ]


def _crud_rows() -> list[str]:
    print("validating local CRUD inputs and rebuilding retrieval runs ...", flush=True)
    inputs = load_hybrid_mrl1024_inputs(EXPANDED, require_frozen=True)
    lexical = bm25_runs(inputs.corpus, inputs.queries, depth=100)
    dense = dense_runs(inputs.query_matrix, inputs.doc_matrix, inputs.doc_ids, depth=100)
    fused = [
        reciprocal_rank_fusion([left, right], k=10, depth=100)
        for left, right in zip(lexical, dense, strict=True)
    ]
    headline = [i for i, q in enumerate(inputs.queries) if q.task == "questanswer_1doc"]
    scores = [
        per_query_metrics([runs[i] for i in headline], [inputs.queries[i] for i in headline])["R@1"]
        for runs in (lexical, dense, fused)
    ]
    rows = [
        _row(
            label,
            "R@1（原单证据任务）",
            scores[0],
            after,
            mcnemar_exact(scores[0], after),
            method=method,
        )
        for label, after, method in (
            ("语义检索 vs 关键词检索", scores[1], "双尾精确 McNemar；原始 p"),
            ("两路融合 vs 关键词检索", scores[2], "双尾精确 McNemar；探索性原始 p"),
        )
    ]
    rerank = _script("evaluate_rerank")
    print("authenticating rerank scores and recomputing paired families ...", flush=True)
    experiment = rerank.Experiment(dict(inputs.corpus), list(inputs.queries), fused)
    _config, _model, cached = rerank._load_cache_for_mode(experiment, score=False, analyze=True)
    if set(cached) != rerank._expected_pairs(experiment):
        raise ValueError("rerank cache is incomplete or contains unexpected pairs")
    runs = {
        rerank.BASELINE_LABEL: fused,
        **{
            rerank.RERANK_LABELS[depth]: rerank._reranked_runs(experiment, cached, depth=depth)
            for depth in rerank.EVAL_DEPTHS
        },
    }
    grouped = {}
    for arity in (1, 2, 3):
        indices = [i for i, q in enumerate(inputs.queries) if len(q.gold_doc_ids) == arity]
        grouped[arity] = {
            label: per_query_metrics(
                [run[i] for i in indices], [inputs.queries[i] for i in indices]
            )
            for label, run in runs.items()
        }
    for comparisons, method in (
        (
            [(rerank.BASELINE_LABEL, rerank.RERANK_LABELS[d]) for d in rerank.EVAL_DEPTHS],
            "双尾精确 McNemar；效益十二项 Holm",
        ),
        (
            [(rerank.RERANK_LABELS[50], rerank.RERANK_LABELS[100])],
            "双尾精确 McNemar；深度六项 Holm",
        ),
    ):
        family = paired_metric_family(
            grouped,
            comparisons=comparisons,
            metrics=("hit@1", "ALL@10"),
            binary=True,
            resamples=RESAMPLES,
        )
        for row in family:
            if row.metric != "hit@1":
                continue
            if row.comparator == rerank.BASELINE_LABEL:
                if row.arity != 1 or row.treatment != rerank.RERANK_LABELS[50]:
                    continue
                label = "融合 + 重排前 50 vs 仅融合"
            else:
                label = f"重排前 100 vs 前 50（arity={row.arity}）"
            rows.append(
                _interval_row(
                    label,
                    f"hit@1（实际 {row.arity} 个 gold）",
                    row.delta,
                    row.ci.low,
                    row.ci.high,
                    p=row.adjusted_p,
                    method=method,
                )
            )
    print("recomputing the historical MRL family ...", flush=True)
    return rows + _mrl_rows(inputs)


def build_summary(paths: ChunkSweepPaths) -> str:
    rows = _crud_rows()
    print("authenticating the TiDB chunk sweep from its canonical inputs ...", flush=True)
    chunk = _script("sync_tidb_chunk_sweep_docs")
    report = chunk._load_report(paths, resamples=RESAMPLES, seed=0)
    for row in report["families"]["origin-source-efficacy"]:
        rows.append(
            _interval_row(
                f"TiDB 分块 {row['comparison']}",
                "origin-source MRR@10",
                row["delta"],
                row["ci_low"],
                row["ci_high"],
                p=row["adjusted_p"],
                method="双尾来源簇配对 bootstrap；两项 Holm",
                scale=1,
            )
        )
    return "\n".join(
        [
            "由 `scripts/sync_ablation_docs.py` 从完整本地缓存离线重算；差值均为前者减后者。",
            "CRUD 使用 5,681 篇文档；原单证据任务为 800 条，"
            "重排按实际 gold 数对全部 2,394 条分层。",
            "TiDB 分块是独立的 source-level known-item 实验，不与 CRUD 文档级结果混合。",
            "",
            "| 比较 | 指标与范围 | 差值 | 配对 95% 置信区间 | p 值 | 检验口径 |",
            "|---|---|---:|---:|---:|---|",
            *rows,
            "",
            "差值区间使用 10,000 次重采样、seed=0；TiDB 按来源簇重采样，其余按查询配对。",
            "MRL 保留历史的单尾退化检验及完整六项比较族；R@1 在这组单证据任务上仍是二元指标。",
            "† 表示 Holm 结果继承蒙特卡洛分辨率下限。未检出差异不证明等价或对小效应有足够功效。",
            "融合对 BM25 的历史比较来自同批数据上的融合选型，"
            "原始 p 未校正该选择，按探索性结果解释。",
            "词法 analyzer 只保留描述性比较；其原始表、重排完整族和历史报告见详细评估文档。",
        ]
    )


def _replace_region(text: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {MARKER} -->", f"<!-- END {MARKER} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"expected exactly one {MARKER} region")
    left, right = text.index(begin) + len(begin), text.index(end)
    if left >= right:
        raise ValueError(f"reversed {MARKER} markers")
    return text[:left] + "\n" + body + "\n" + text[right:]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "docs" / "evaluation.md")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = ChunkSweepPaths(ROOT / "indexes" / "tidb", ROOT / "tidb-rag-curated")
    with (
        exclusive_lock(args.repo_root / DOCS_LOCK),
        exclusive_lock(EXPANDED / ".h_hybrid_mrl1024.lock"),
        exclusive_lock(paths.sweep_root / ".bundle.lock"),
        exclusive_lock(paths.eval_root / ".artifacts.lock"),
    ):
        targets = (args.readme, args.evaluation)
        originals = {path: read_text(path) for path in targets}
        for value in originals.values():
            _replace_region(value, "")
        body = build_summary(paths)
        rendered = {path: _replace_region(before, body) for path, before in originals.items()}
        stale = tuple(path for path in targets if rendered[path] != originals[path])
        if args.check:
            if stale:
                raise SystemExit("! generated ablation summary is stale")
            return ()
        staged: list[tuple[Path, Path]] = []
        try:
            for path in stale:
                temporary = path.with_suffix(path.suffix + ".ablation-sync.tmp")
                staged.append((temporary, path))
                write_text(temporary, rendered[path])
            if staged:
                replace_files(staged)
        finally:
            for temporary, _path in staged:
                temporary.unlink(missing_ok=True)
        return stale


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        changed = synchronize(_parse_args(argv))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"! offline ablation synchronization failed: {exc}") from exc
    print(f"ablation summary verified; {len(changed)} document(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
