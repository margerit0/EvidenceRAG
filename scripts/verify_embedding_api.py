"""Probe undocumented embedding-provider behaviour before building any index.

Run once per provider, *before* the first bulk embed:

    uv run python scripts/verify_embedding_api.py

Four properties decide whether an index is even correct, and no vendor
documents any of them:

1. **Are the returned vectors L2-normalised?** A cosine/IP index over
   un-normalised vectors does not raise -- it silently ranks by magnitude as
   well as by direction, so a long document outranks a relevant one and the
   only symptom is a mediocre Recall@1 that looks like a modelling problem.
   Checked across several inputs, because one vector with norm 1.0 could be a
   coincidence; a *constant* norm across inputs of very different lengths
   cannot be.

2. **How reproducible is the provider?** Measured first, because it is the
   yardstick for everything after it. Batched GPU inference is not
   element-wise deterministic: the reduction order inside a fused matmul
   depends on batch shape and position, so the same text can come back
   slightly different. Until this noise floor is known, any other "the vectors
   differ by 1e-3" observation is uninterpretable.

3. **Is ``dimensions=n`` a prefix slice or a separately trained head?** If a
   slice, one stored 4096-d vector serves every dimension and the MRL ablation
   costs zero extra API calls. If a trained head, each dimension needs its own
   full re-embed and its own index. This decides the collection schema, which
   cannot be changed after the corpus is indexed. Judged by cosine *relative to
   the noise floor from (2)* -- an absolute threshold cannot tell "different
   head" from "same head, re-run".

4. **Does client-side slicing preserve ranking?** The only question that
   actually matters downstream. Two vectors can differ numerically and induce
   identical retrieval. Ranking agreement is measured directly rather than
   inferred from cosine.

Norms and dot products use :func:`math.fsum`, not ``numpy.linalg.norm``. fsum is
exactly rounded, so a deviation it reports is real rather than an artefact of
pairwise summation -- which matters when the whole question is whether a number
equals 1.0.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from zhrag.providers.embedding import EmbeddingClient, EmbeddingConfig, load_env

ROOT = Path(__file__).resolve().parent.parent

#: float32 has ~7 decimal digits, and the values arrive as decimal JSON, so a
#: genuinely normalised vector lands within ~1e-6 of 1.0 rather than exactly on
#: it. Anything outside this is a real deviation, not rounding.
TOLERANCE = 1e-5

#: Deliberately diverse: normalisation is a per-vector property, so the
#: discriminating evidence is a *constant* norm across inputs whose raw
#: magnitudes should differ a lot (one character vs. a full paragraph).
PROBES: list[tuple[str, str]] = [
    ("zh-short", "TiDB 的向量索引怎么创建？"),
    (
        "zh-long",
        "TiDB 是一个开源的分布式 SQL 数据库，支持水平扩展、强一致性和高可用。"
        "它兼容 MySQL 协议，采用计算存储分离架构，由 TiDB Server、PD 和 TiKV 三个组件构成。"
        "从 v7.0 起，TiDB 支持向量数据类型与向量索引，可用于检索增强生成等场景。" * 2,
    ),
    ("en-only", "How do I create a vector index in TiDB?"),
    ("zh-1char", "库"),
    ("mixed-code", "执行 `tiup cluster deploy` 之前需要确认 TiFlash 副本数"),
    ("zh-short-dup", "TiDB 的向量索引怎么创建？"),  # intra-batch determinism check
]

#: Near-duplicates on purpose. Ranking is only at risk when candidates are
#: close together, so a corpus of obviously-unrelated sentences would pass the
#: equivalence test without testing anything.
RANK_QUERY = "怎么给 TiDB 表加向量索引？"
RANK_DOCS: list[str] = [
    "在 TiDB 中可以通过 CREATE VECTOR INDEX 语句为向量列创建索引。",
    "TiDB 的向量索引要求表必须先有 TiFlash 副本，否则无法创建。",
    "向量索引目前在 TiDB 中标记为实验特性，可能在后续版本变更。",
    "TiDB 支持 VECTOR 数据类型，可以存储浮点数组用于相似度检索。",
    "使用 ALTER TABLE 语句可以为已有的 TiDB 表添加列。",
    "TiDB 的二级索引和主键索引在存储层都由 TiKV 负责维护。",
    "创建索引时可以指定索引名称、索引列以及索引类型。",
    "TiFlash 是 TiDB 的列式存储引擎，用于加速分析型查询。",
    "余弦距离和 L2 距离是向量检索中常用的两种距离度量。",
    "TiDB Server 是无状态的计算节点，不存储实际数据。",
    "PD 负责 TiDB 集群的元数据管理和调度决策。",
    "可以使用 tiup cluster deploy 命令部署一个新的 TiDB 集群。",
    "TiDB 兼容 MySQL 5.7 协议，大部分客户端可以直接连接。",
    "分布式事务在 TiDB 中通过两阶段提交协议实现。",
    "TiDB 的执行计划可以通过 EXPLAIN 语句查看。",
    "向量检索通常需要先把文本通过嵌入模型转换成稠密向量。",
    "HNSW 是一种基于图的近似最近邻检索算法。",
    "在建立索引之前应当确认嵌入向量是否已经做过 L2 归一化。",
    "TiDB 从 v7.0 开始所有文档采用 CC BY-SA 3.0 许可证。",
    "备份和恢复可以使用 BR 工具完成，支持全量和增量。",
]


def _l2(vector: list[float]) -> float:
    """Exactly-rounded L2 norm. Equivalent to np.linalg.norm, but not lossy."""
    return math.sqrt(math.fsum(x * x for x in vector))


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors here are unit-norm, but do not assume it."""
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    return dot / (_l2(a) * _l2(b))


def _renorm(vector: list[float], dim: int) -> list[float]:
    """MRL by construction: take the first `dim` components, then L2-renormalise."""
    prefix = vector[:dim]
    scale = _l2(prefix)
    return [x / scale for x in prefix]


def _vectors(response: dict[str, Any]) -> list[list[float]]:
    """Extract embeddings in request order; OpenAI does not promise sorted data."""
    rows = sorted(response["data"], key=lambda d: d.get("index", 0))
    return [row["embedding"] for row in rows]


def _embed(
    client: EmbeddingClient, texts: list[str], *, dim: int | None = None
) -> list[list[float]]:
    payload: dict[str, Any] = {"model": client.config.model, "input": texts}
    if dim is not None:
        payload["dimensions"] = dim
    return _vectors(client.post(payload))


def check_normalisation(client: EmbeddingClient) -> list[list[float]]:
    print("## 1. L2 normalisation\n")
    response = client.post({"model": client.config.model, "input": [t for _, t in PROBES]})
    vectors = _vectors(response)

    tokens = response.get("usage", {}).get("total_tokens")
    print(f"   model={response.get('model', client.config.model)}  usage={tokens} tokens\n")
    print(f"   {'probe':<14} {'chars':>6} {'dim':>6} {'L2 norm':>19} {'|norm-1|':>10}")
    print(f"   {'-' * 14} {'-' * 6} {'-' * 6} {'-' * 19} {'-' * 10}")

    norms: list[float] = []
    for (label, text), vector in zip(PROBES, vectors, strict=True):
        norm = _l2(vector)
        norms.append(norm)
        print(
            f"   {label:<14} {len(text):>6,} {len(vector):>6,} "
            f"{norm:>19.15f} {abs(norm - 1):>10.2e}"
        )

    spread = max(norms) - min(norms)
    print(f"\n   spread across probes: {spread:.2e}")
    if all(abs(n - 1.0) < TOLERANCE for n in norms):
        print("   => L2-NORMALISED. Safe for COSINE or IP; the two are equivalent here.")
        print("      No client-side normalisation needed before indexing.")
    elif spread < TOLERANCE:
        print(f"   => NOT unit norm, but constant at {norms[0]:.6f} -- rescaled, not raw.")
        print("      Ranking is unaffected, but do not assume dot product == cosine.")
    else:
        print("   => NOT NORMALISED and norm varies with input.")
        print("      MUST normalise client-side before indexing, or IP ranking is wrong.")
    return vectors


def check_reproducibility(client: EmbeddingClient, batch: list[list[float]]) -> float:
    """Return the noise floor as a cosine, for use as the yardstick in check_mrl."""
    print("\n\n## 2. Reproducibility -- establishing the noise floor\n")

    intra = _cos(batch[0], batch[-1])
    print(f"   same text, same batch  (positions 0 and 5): cos = {intra:.12f}")

    repeat = _embed(client, [PROBES[0][1]])[0]
    inter = _cos(batch[0], repeat)
    print(f"   same text, separate request:                cos = {inter:.12f}")

    floor = min(intra, inter)
    print(f"\n   noise floor = {floor:.12f}  (1 - {1 - floor:.2e})")
    if floor > 1 - 1e-9:
        print("   => deterministic. Bit-level caching and exact-match tests are sound.")
    else:
        print("   => NOT deterministic. Batched GPU reduction order varies, so the same")
        print("      text re-embedded differs slightly. Consequences:")
        print("      - cache on (text, model) is still correct, but tests must compare")
        print("        with a tolerance, never assert equality;")
        print("      - any two vectors closer than this floor are indistinguishable.")
    return floor


def check_mrl(client: EmbeddingClient, full: list[float], floor: float, *, dims: list[int]) -> None:
    print("\n\n## 3. dimensions=n -- prefix slice or separately trained head?\n")
    print("   Judged against the noise floor, not an absolute threshold: a re-embed of")
    print(f"   the *same* text at full width already only reaches cos {floor:.9f}.\n")
    print(f"   {'dim':>6} {'cos(slice, native)':>20} {'vs floor':>12} {'front-load':>11}")
    print(f"   {'-' * 6} {'-' * 20} {'-' * 12} {'-' * 11}")

    verdicts: list[bool] = []
    for dim in dims:
        try:
            native = _embed(client, [PROBES[0][1]], dim=dim)[0]
        except SystemExit as exc:
            print(f"   {dim:>6}  rejected: {exc}")
            verdicts.append(False)
            continue
        if len(native) != dim:
            print(f"   {dim:>6}  provider ignored the parameter, returned {len(native)}")
            verdicts.append(False)
            continue

        sliced = _renorm(full, dim)
        cos = _cos(sliced, native)
        # A slice that is merely as different as a re-run is not a different head.
        indistinguishable = cos >= floor
        # Front-loading: an MRL-trained model concentrates information early, so
        # ||full[:n]|| exceeds the uniform share sqrt(n/N).
        front = _l2(full[:dim]) / math.sqrt(dim / len(full))
        verdicts.append(indistinguishable)
        mark = "at/above" if indistinguishable else "BELOW"
        print(f"   {dim:>6} {cos:>20.12f} {mark:>12} {front:>11.4f}")

    print()
    if all(verdicts):
        print("   => PREFIX SLICE. At every width the client-side slice is at least as")
        print("      close to the native vector as a plain re-embed of the same text is")
        print("      to itself. There is no evidence of a separate head to detect.")
        print("      Store ONE 4096-d vector; the whole MRL ablation is client-side.")
    elif any(verdicts):
        print("   => MIXED. Some widths track the slice, others do not -- see the table.")
    else:
        print("   => SEPARATELY TRAINED HEAD(s). Each width needs its own re-embed")
        print("      and its own index; budget the MRL ablation accordingly.")


def check_ranking_equivalence(client: EmbeddingClient, *, dim: int) -> None:
    """The only test whose outcome changes a downstream decision."""
    print(f"\n\n## 4. Ranking equivalence at dim={dim} (the decision test)\n")
    print(f"   {len(RANK_DOCS)} near-duplicate Chinese technical sentences, 1 query.")
    print("   Near-duplicates on purpose: ranking only flips when scores are close.\n")

    q_full = _embed(client, [RANK_QUERY])[0]
    d_full = _embed(client, RANK_DOCS)
    d_native = _embed(client, RANK_DOCS, dim=dim)
    q_native = _embed(client, [RANK_QUERY], dim=dim)[0]

    sliced_scores = [_cos(_renorm(q_full, dim), _renorm(d, dim)) for d in d_full]
    native_scores = [_cos(q_native, d) for d in d_native]

    order_sliced = sorted(range(len(RANK_DOCS)), key=lambda i: -sliced_scores[i])
    order_native = sorted(range(len(RANK_DOCS)), key=lambda i: -native_scores[i])

    print(f"   {'rank':>4} | {'client-side slice':<34} | {'native dimensions=' + str(dim):<34}")
    print(f"   {'-' * 4} | {'-' * 34} | {'-' * 34}")
    for rank in range(5):
        a, b = order_sliced[rank], order_native[rank]
        flag = "  " if a == b else " !"
        print(f"   {rank + 1:>4}{flag}| {RANK_DOCS[a][:32]:<34} | {RANK_DOCS[b][:32]:<34}")

    for k in (1, 3, 5, 10):
        overlap = len(set(order_sliced[:k]) & set(order_native[:k])) / k
        print(f"\n   top-{k:<2} set overlap : {overlap:.0%}", end="")
    identical = sum(a == b for a, b in zip(order_sliced, order_native, strict=True))
    print(f"\n   exact position match: {identical}/{len(RANK_DOCS)}")

    drift = max(abs(a - b) for a, b in zip(sliced_scores, native_scores, strict=True))
    # The gap between adjacent ranks is the scale that decides whether `drift`
    # can reorder anything: a score perturbation smaller than the typical gap
    # cannot flip neighbours, however large it looks in isolation.
    descending = sorted(native_scores, reverse=True)
    gap = statistics.median(descending[i] - descending[i + 1] for i in range(len(descending) - 1))
    print(f"   max |score delta|   : {drift:.2e}")
    print(f"   median adjacent gap : {gap:.2e}   (delta/gap = {drift / gap:.2f})")

    if order_sliced[:10] == order_native[:10]:
        print("\n   => IDENTICAL top-10 ordering. Client-side slicing is safe: the MRL")
        print("      ablation can run off one stored 4096-d vector with no re-embedding.")
    else:
        print("\n   => ORDER DIFFERS. Compare the flipped rows against the median score")
        print("      gap above before concluding it matters.")


def main() -> int:
    # This script prints Chinese -- both its own probes and, more importantly,
    # the provider's error bodies, which are Chinese. On a cp936 console that
    # mojibakes (stderr) or raises UnicodeEncodeError (stdout). Both streams
    # need reconfiguring: SystemExit messages are written to stderr, so fixing
    # only stdout leaves exactly the error text you need to read unreadable.
    # Explicit and local, unlike PYTHONUTF8, which would also mask bare-open()
    # bugs that the Windows CI leg exists to catch.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--rank-dim", type=int, default=1024)
    parser.add_argument("--skip-ranking", action="store_true")
    args = parser.parse_args()

    config = EmbeddingConfig.from_env(load_env(ROOT / ".env"))
    client = EmbeddingClient(config=config)

    print(f"endpoint {config.url}\nmodel    {config.model}\n")
    vectors = check_normalisation(client)
    floor = check_reproducibility(client, vectors)
    check_mrl(client, vectors[0], floor, dims=args.dims)
    if not args.skip_ranking:
        check_ranking_equivalence(client, dim=args.rank_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
