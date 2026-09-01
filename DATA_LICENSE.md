# 数据来源与授权

**本仓库不包含任何语料原文。** 一个字节都没有。两份语料都通过各自的下载/抽取脚本在本地重建，
`.gitignore` 明确排除了下列目录：

```
crud-rag-subset/raw/            crud-rag-subset/corpus/
crud-rag-subset/eval/           crud-rag-subset/eval-expanded/
tidb-rag-curated/documents/
```

仓库里保留的只有**清单、校验值和可复现的下载脚本**（`corpus_manifest.jsonl`、
`subset_manifest.json`、`*.ps1`），它们描述数据但不复制数据。

以下授权状态是截至 2026-08-31 冻结的审计记录，包含 GitHub API 的查询结果；本文件不声称本次会话重新联网核验。

---

## 1. TiDB 中文文档 — `pingcap/docs-cn`

| | |
|---|---|
| 上游 | https://github.com/pingcap/docs-cn |
| 固定 commit | `26f202bcb1b314ca21f63bdb931f3c90426a14a1` |
| 许可证 | **CC BY-SA 3.0 Unported**（仓库根目录 `LICENSE`，首行 `Attribution-ShareAlike 3.0 Unported`） |
| 上游声明 | README：「自 TiDB v7.0 起，所有文档的许可证均为 CC BY-SA 3.0」 |
| 本快照是否适用 | **是**。语料中出现的最高版本号为 `v9.0.0`，远在 v7.0 阈值之后 |

> GitHub API 的 `license` 字段返回 `{"key": "other", "spdx_id": "NOASSERTION"}` —— 这只是
> 因为该 LICENSE 是 CC BY-SA 3.0 全文而非 GitHub 能自动识别的 SPDX 模板，**不代表授权不明**。

### 这对本项目意味着什么

CC BY-SA 3.0 的 **ShareAlike** 条款是有实质约束的，且很容易被忽略：

- 本项目的 chunk 输出（`zhrag.chunking` 切分出的文本片段）在该许可证下构成
  **Adaptation**（改编作品），因为它是「以任何可辨认地衍生自原作的形式对原作的改写、转换或改编」。
- 因此**若要分发这些 chunk 或由其构建的索引**，必须：
  1. 署名 PingCAP 并指明原作已被修改；
  2. 以 CC BY-SA 3.0 或兼容许可证分发该改编作品。
- 这就是为什么派生产物（切分结果、向量索引）同样不进仓库，而是由脚本本地生成。

**仓库中的代码以 Apache-2.0 授权；文档语料及其派生物不适用该许可证。**

---

## 2. CRUD-RAG 评测数据 — `IAAR-Shanghai/CRUD_RAG`

| | |
|---|---|
| 上游 | https://github.com/IAAR-Shanghai/CRUD_RAG |
| 使用的文件 | `data/crud_split/split_merged.json` |
| 论文 | CRUD-RAG: A Comprehensive Chinese Benchmark for RAG（arXiv:2401.17043） |
| 许可证 | ⚠️ **无**。GitHub API 的 `license` 字段返回 `null`，仓库根目录**不存在 LICENSE 文件** |

> README 中的 Apache-2.0 徽章是一张 shields.io 图片，**背后没有对应的许可证文件**。
> 徽章不是授权。截至 2026-08-18 查询时，该仓库有 403 stars，仍无 LICENSE。

### 这对本项目意味着什么

风险有两层，且第二层更重：

1. **代码层**：无 LICENSE 文件意味着默认「保留所有权利」。本项目没有复制、移植或改写其
   `src/metric/` 或其它源文件；M9a 的 `metrics_gen.py` 与 `quest_eval.py` 只依据公开的
   BLEU、ROUGE-L 和 answer-scoring 定义独立实现，并由本项目自己的合成测试覆盖。
   **独立重实现不等于获得复制上游代码的权利**，也不授予使用上游数据的权利。

2. **数据层（更重要）**：`split_merged.json` 内含约 8 万篇中文新闻正文，**上游未声明这些新闻的
   出处、版权归属或再分发许可**。这些文本的著作权属于原始新闻机构，不属于数据集作者，
   数据集作者也未展示获得转授权的依据。

因此本项目：

- **不重新分发任何新闻正文**，包括「小样本」「示例几条」；
- `docs/evidence/crud_rag_table8_v3.json` 只保存论文 Table 8 的聚合事实、来源定位、hash 和
  上游审计元数据，作为 **aggregate-only historical evidence**；它不包含 question、answer、
  reference、passage、chunk、embedding 或 provider payload。引用聚合事实不等于获得上游代码
  或底层数据的再分发许可；
- 不复制 CRUD-RAG 上游代码，也不把其 `quest_gt`、generated question/answer 或 generation
  cache 加入仓库、发行包或公开 artifact；这些内容若在 M9b 本地生成，只能放在明确的
  gitignored 本地目录；
- 仅将原始数据用于**非商业的学术评测与个人技术验证**；所有语料、chunk、向量、检索分数、
  QG/QA/生成文本及其缓存均为本地派生，已被 `.gitignore` 排除。

**建议**：若要将本项目用于任何商业或公开托管场景，请先就数据授权向上游提 issue 澄清。

---

## 3. 如何在本地获得数据

```powershell
# TiDB 文档（500 篇，按 git blob SHA-1 校验完整性）
powershell -NoProfile -ExecutionPolicy Bypass -File tidb-rag-curated\download_curated.ps1

# CRUD-RAG 原始 split 文件与 500 篇子集
powershell -NoProfile -ExecutionPolicy Bypass -File crud-rag-subset\build_subset.ps1
```

```bash
# 扩展评测语料（5,681 篇 + 2,394 条 qrels）——纯本地派生，不下载任何东西
uv run python scripts/build_eval_corpus.py
```

---

## 4. 引用

使用本评测框架或其结论时，请一并引用上游：

```bibtex
@article{lyu2024crud,
  title  = {CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented
            Generation of Large Language Models},
  author = {Lyu, Yuanjie and Li, Zhiyu and Niu, Simin and Xiong, Feiyu and Tang, Bo
            and Wang, Wenjin and Wu, Hao and Liu, Huanyong and Xu, Tong and Chen, Enhong},
  journal = {arXiv preprint arXiv:2401.17043},
  year   = {2024}
}
```

TiDB 文档署名：© PingCAP, Inc.，依 CC BY-SA 3.0 授权，快照取自 commit `26f202bc`。
