# zhrag — 企业级中文 RAG 与可复现评估框架

面向中文语料的检索增强生成系统。项目的核心不是"又一个 RAG demo"，而是**一套能真正区分配置优劣的评估框架**，以及在此之上用实测数据驱动的每一个工程决策。

<!-- BEGIN TIDB-EVAL-STATUS -->
> **状态**：评估层、语料层、词法/稠密检索、MRL、RRF 与离线 rerank 深度消融均已完成；
> 在线检索链路已通过真实 Milvus Lite 校验，TiDB evergreen 索引也已付费嵌入并发布：
> **450 篇文档 / 1,832 chunks**。另已构建 **490 组 direct/paraphrase、980 条 query** 的合成 pooled qrels；
> 系统级 TiDB Hit@1 / R@1 / MRR@10 / binary + graded nDCG@10 已完成；点估计以 490 个 pair 观测为权重，CI 与检验按 245 个 `gold_source_key` 源聚类重采样。
> 服务端 `hybrid_search` 融合仍是待验证路径，不等同于本地 exact RRF。
> 下方所有数字均为本仓库脚本在真实语料上跑出的结果，非引用。
<!-- END TIDB-EVAL-STATUS -->

---

## 为什么先做评估

项目起步时的第一个实验就推翻了原定方案。

CRUD-RAG 官方子集只索引 500 篇文档。把评测集换成**全部 800 条单证据查询、语料只放这 800 篇证据文档**（对检索器最有利的配置），一个**纯标准库、字符 bigram 的 BM25**（无 embedding、无 rerank、无调参）得到：

```
Recall@1 = 98.0%    Recall@5 = 100.0%    MRR@10 = 0.990    nDCG@10 = 0.992
```

也就是说：**任何**检索配置在这个评估集上都会得到接近满分，ablation table 的每一行都会是同一个数字。评估集完全饱和，不具备区分能力。

原因有两点：语料规模太小；且问题由证据文档生成，实体、数字、专有名词大量原文重叠，字符匹配即可命中。

### 修复：扩大语料，更换指标

`crud-rag-subset/raw/split_merged.json` 中本就包含全部六个 CRUD 任务（7,661 条记录）。从中可去重抽取 **5,681 篇**文档作为干扰集，无需额外下载。

以下由 `scripts/run_lexical_sweep.py` 生成（800 条单证据查询）：

| 语料规模 | R@1 | R@5 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|
| 800（仅证据文档） | 98.0% | 100.0% | 0.990 | 0.992 |
| 1,000 | 96.2% | 100.0% | 0.981 | 0.986 |
| 2,000 | 90.4% | 100.0% | 0.949 | 0.962 |
| 4,000 | 80.6% | 99.5% | 0.890 | 0.918 |
| **5,681** | **75.9%** | 99.2% | **0.857** | 0.893 |

R@1 让出 22 个百分点的可优化空间。**R@5 即使在 5,681 篇下仍有 99.2%，不能作为主指标。**
本项目的主指标固定为 **R@1 / MRR@10 / nDCG@10**。

### 多文档任务才是真正的战场

同一 5,681 篇语料，字符 bigram BM25：

| 任务 | n | 平均 gold | R@1 | R@1 上限 | 占上限 | ALL-gold@10 |
|---|---:|---:|---:|---:|---:|---:|
| `questanswer_1doc` | 800 | 1.00 | 75.9% | 100.0% | 75.9% | 99.9% |
| `questanswer_2docs` | 797 | 1.99 | 35.4% | 50.3% | 70.5% | 87.3% |
| **`questanswer_3docs`** | 797 | 2.98 | 23.0% | 33.5% | 68.5% | **68.9%** |

因此评估纳入 2docs / 3docs 任务，而非仅用最简单的 1doc。

> ⚠️ **R@1 这一列不能跨行比较。** `recall_at_k` 返回的是"命中的 gold 占比"，因此 k=1 时 3-gold 查询的上限就是 1/3。23.0% 其实是上限的 68.5%，不是崩溃 —— 三行的"占上限"分别是 75.9% / 70.5% / 68.5%，系统随 arity **平滑退化而非雪崩**。跨 arity 比较要用 **ALL-gold@10** 或"占上限"，不要直接比 R@1。

### 评测集够大吗？

"2,394 条查询够不够"这个问题里其实混着两个完全不同的问题，功效分析由 `scripts/power_analysis.py` 生成。

**绝对精度**（"这个系统的 R@1 是多少？"）只受单个分数向量的二项分布支配，按 1/√n 收敛 —— 误差棒减半要 4 倍查询量：

| n | 95% CI 半宽 |
|---:|---:|
| 800 | ±2.88pp |
| 2,394 | ±1.69pp |
| 10,000 | ±0.87pp |

**比较精度**（"B 臂比 A 臂好吗？"）才是消融表真正需要的，而它的行为完全不同。两臂看同一批查询，配对检验只看**逐查询的差值**，两臂一致的查询差值为 0。对二元指标，检验统计量约等于 `(f−b)/√(f+b)`，其中 f 是被修好的查询数、b 是被弄坏的 —— **n 根本不出现**。

实测证实了这一点：达到显著所需的绝对改变数几乎与 n 无关（churn=0.5 时，n=400/800/2394 都是约 20 修好 / 10 弄坏）。n 只改变这个固定数量折算成百分点后的大小。

真正决定性的是**多重比较校正**。消融表约 40 格，Holm 校正后最严的那格要求 p ≤ 0.05/40 = 0.00125：

| n | churn | α=0.05 | α=0.00125（Holm 后） |
|---:|---:|---:|---:|
| 800 | 0.0 | 0.50pp | 1.75pp |
| 800 | 0.5 | 1.25pp | 3.50pp |
| **800** | **0.8** | 3.38pp | **不可检测** |
| 2,394 | 0.0 | 0.21pp | 0.58pp |
| 2,394 | 0.5 | 0.42pp | 1.21pp |
| **2,394** | **0.8** | 1.09pp | **3.09pp** |

> churn = 弄坏数 / 修好数。0.0 是纯改进；0.8 表示每修好 5 条就弄坏 4 条 —— 这才是真实消融的常见形态。

**在 n=800 上，一个 churny 的改动经 Holm 校正后在任何效应量下都无法显著**，因为需要改变的查询数会超过 n/2。所以本项目用满全部 2,394 条。

**这张表随后被本项目自己的数据命中了。** dense-4096 vs BM25 在 R@1 上是 **86 条修好 / 69 条弄坏**，
churn = 0.80，delta 2.12pp —— 正落在上表「n=800、churn 0.8、α=0.05 需要 3.38pp」那一格的下方，
于是 p = 0.199，不显著。而 RRF 融合对 BM25 是 65 胜 33 负（churn 0.51）、delta 4.00pp，
越过了 churn 0.5 那行的 1.25pp 门槛，p = 0.0016。**功效分析不是纸上谈兵**：它提前说清了
哪一格能出结论、哪一格无论怎么跑都出不了，而两个预测都兑现了。

**但到 2,394 就停。** 唯一的扩充途径是用 LLM 从文档合成查询，而那会**重新引入把这个 benchmark 一开始搞饱和的那个缺陷**：从证据文档生成问题 → 实体与数字原文重叠 → 字符匹配直接命中。那是拿一个更严重的效度问题去换一个精度问题。

**真正的杠杆始终是语料规模，不是查询数。** 语料 800 → 5,681 创造了 22 个百分点的可测空间；查询数只是把尺子刻度变细，语料规模决定尺子上有没有东西可量。

<details>
<summary>副产品：闭式解在稀疏情形下会高估功效</summary>

上面的 MDE 表是用真实的 `paired_bootstrap_test` 二分搜索出来的，不是套 `(f−b)/√(f+b)` 公式。因为当改变的查询数很少时，bootstrap 重采样分布过于离散，中心极限定理不成立，闭式解会声称一个 bootstrap 并不认可的显著性：

```
     n  fixed  broken |      z   z 判定 |  actual p  bootstrap 判定
   800      3       0 |   1.74     True |    0.0795      False   <- 分歧
   800      6       0 |   2.46     True |    0.0187       True
   800     12       0 |   3.49     True |    0.0017       True
```

以 bootstrap 为准 —— 它才是消融表实际会报告的东西。
</details>

---

## 实测驱动的工程决策

<!-- BEGIN M9A-GENERATION-METRICS -->
### 中文生成指标与 CRUD-RAG Table 8 历史证据

M9a 冻结的是可复现的指标合同，不是一次真实生成实验。词法指标使用注入式 tokenizer，
默认 CI 不加载模型；任一行 tokenization、语义评分、长度或数值校验失败，
整个 report 拒绝发布，不会把失败行静默转成 0 或从 denominator 中删除。

**本项目的 canonical contracts**：
- `mean_sentence_bleu4`：单 reference、token-level modified 1–4 gram precision、
  几何均值与标准 brevity penalty；无 smoothing、无 effective-order，逐样本后取算术平均。
- `crud_mean_sentence_bleu4_no_bp`：同一逐样本算法但移除 brevity penalty，只作兼容审计列，
  不称为 standard BLEU；两者都不是 corpus BLEU。
- `mean_sentence_rouge_l_f1`：token LCS 的 sentence-level precision/recall/F1（beta=1），
  逐样本后取算术平均；是 rougeL，不是 rougeLsum。
- `bert-base-chinese` 的真实 BERTScore 为可选模型适配器：lang=`zh`、layer=8、rescale_with_baseline=True、idf=False、batch=64、use_fast_tokenizer=False。
- `zhrag-rag-quest-eval-v1`：无法回答哨兵为 `无法推断`；报告论文全问题分母与历史代码条件分母，不把空条件集合伪造为 0。

**CRUD-RAG Table 8（历史 0–100 表格值，仅作来源锚点）**：

| task | model | BLEU | ROUGE-L | `bertScore`（上游列名） | RAGQuest precision | RAGQuest recall | length |
|---|---|---:|---:|---:|---:|---:|---:|
| summarization | Qwen-14B | 32.51 | 33.33 | 85.62 | 68.94 | 40.57 | 139.1 |
| summarization | GPT-4-0613 | 24.54 | 35.91 | 89.39 | 71.24 | 50.53 | 194.6 |
| question answering 1-document | Qwen-14B | 37.95 | 55.13 | 83.25 | 53.03 | 73.92 | 73.8 |
| question answering 1-document | GPT-4-0613 | 33.87 | 51.42 | 80.92 | 53.14 | 62.39 | 95.9 |

历史值来自 [arXiv 2401.17043v3](https://arxiv.org/pdf/2401.17043v3) 的 Table 8（HTML 锚点：[https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8](https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8)，PDF 第 26 页；副本 SHA-256 `2e4ae0cb708fdca9d96bcf8d1c0713dae121195a0a31b78e3132e9ef4fa7db8a`）。
这些值不是本项目实现的 golden test：论文没有冻结足够的 tokenization、依赖版本、smoothing 或模型参数，且上游 `bertScore` 实际是 `text2vec-base-chinese` 句向量相似度，不是真 BERTScore。不要把它与本项目的真实 BERTScore 列直接横比。
上游仓库固定为 `IAAR-Shanghai/CRUD_RAG@1aace383994e1f68efa12cf2a8e2dadfb4102ceb`；截至 2026-08-31 仍无 LICENSE。这里只引用聚合事实和公开定义，未复制上游代码或数据。
<!-- END M9A-GENERATION-METRICS -->

### 中文 BM25：字符 bigram 优于 jieba 分词

5,681 篇语料、全部 2,394 条查询，由 `scripts/run_lexical_sweep.py` 生成。
R@1 按任务分层（不可跨行比较），ALL-gold@10 汇总（任意 arity 下都有效）：

| analyzer | vocab | R@1 1doc | R@1 2docs | R@1 3docs | **ALL@10** | build | query |
|---|---:|---:|---:|---:|---:|---:|---:|
| char unigram | 4,886 | 74.2% | 27.5% | 17.3% | 77.5% | 0.8s | 36.0ms |
| **char bigram** | 351,779 | 75.9% | **35.4%** | **23.0%** | **85.4%** | 2.9s | 6.6ms |
| char trigram | 1,204,066 | 74.0% | 34.4% | 21.7% | 79.4% | 4.3s | 2.1ms |
| jieba precise | 85,365 | 74.8% | 35.1% | 22.4% | 84.3% | 10.9s | 12.0ms |
| jieba cut_for_search | 91,246 | 73.4% | 34.4% | 22.2% | 84.2% | 7.3s | 12.5ms |
| jieba + char bigram | 393,862 | **76.4%** | 35.6% | 22.8% | 85.3% | 8.9s | 15.4ms |

字符 bigram 胜过两种 jieba 模式，建索引快 3.8 倍，且**零依赖**。
并集方案在 1doc 上多 0.5pp，但汇总的 ALL@10 反而略低（85.3% vs 85.4%）—— 这 0.5pp 要靠翻倍的词表和一个 2020 年后再未发版的依赖来换。→ **jieba 被移出主依赖**（仅作为可选 extra 保留对照行）。

> **分层汇报立刻还了债。** 只看 1doc 的 R@1，char unigram 与 bigram 相差 1.7pp，看起来无关紧要；换成汇总的 ALL-gold@10，差距是 **7.9pp**。单证据任务把 analyzer 之间的差异压缩了约 4 倍 —— 只在 1doc 上做选型，会让人以为分词方案根本不重要。

### Markdown 分块：必须两阶段

TiDB 文档的 h2/h3 段落长度极不均匀（p50 = 332 字符，p90 = 1,743，max = 70,843）。按标题切分是常见建议，但在此语料上失效。以下全部由 `scripts/corpus_stats.py` 生成，token 计数使用与 Qwen3 实测标定的估算器：

| 策略 | n | p10 | p50 | p90 | max | <100 tok | 代码块截断 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 仅按标题切分 | 5,504 | 14 | 65 | 318 | 16,111 | **63.5%** | — |
| 两阶段 target=400 | **1,832** | 165 | 371 | 734 | 10,914 | **5.0%** | **0** |
| 两阶段 target=512 | 1,486 | 215 | 476 | 890 | 16,111 | 3.2% | 0 |
| 两阶段 target=700 | 1,154 | 224 | 645 | 1,165 | 16,111 | 2.7% | 0 |

近**三分之二**的朴素切块小于 100 token（一个标题加一句话），embedding 后基本是噪声。
两阶段策略 = 按标题层级切分 → 合并过小相邻段 + 按段落边界拆分超长段。

代码块与表格在切分前被占位符保护、切分后还原，**实测三种 target 下代码块截断数均为 0**。这对本语料至关重要：检索强依赖 `tiup cluster deploy` 这类标识符的精确匹配。

合并相邻小节时，只有共同的标题前缀留在 metadata 里，各小节剩余的标题层级会写进被索引的正文。否则「甲」「乙」两个兄弟小节合成一块后，「乙」的标题会在生成永久 chunk ID 与向量之前就消失。

选定 target=400：共 1,832 个 chunk、829,140 tokens。其中 282 个（15.4%）超过 `hard_max`，因为单个代码块或表格本身就超预算——这是有意为之，Qwen3-Embedding-8B 的 32k 上下文放得下，而把表格与表头拆开的代价更大。

向量存储量：**4096 维 float32 仅 30.0 MB**，MRL 截断到 1024 维只要 7.5 MB。规模完全不构成向量库选型的约束。

### 分块尺寸必须用对的 tokenizer 估算

用实测的 Qwen3 tokenizer 标定（`src/zhrag/tokens.py`）：

| tokenizer | 中文 | 中英混排 | 英文 | 512 token 相当于多少字符（中/混/英） |
|---|---:|---:|---:|---|
| **Qwen3 (151k)** | **1.570** | 2.275 | 4.491 | 804 / 1,165 / 2,299 |
| cl100k_base | 0.849 | 1.959 | 4.491 | 435 / 1,003 / 2,299 |
| o200k_base | 1.331 | 2.305 | 4.491 | 682 / 1,180 / 2,299 |
| BGE-M3 / XLM-R | 1.524 | 2.076 | 3.456 | 780 / 1,063 / 1,770 |

用 cl100k（GPT-3.5 时代）给中文估算 token，每个 chunk 会少装近一半内容。本项目早期原型用固定 1.15 字符/token，得出的 chunk 数与最终实现相差 2.4 倍——README 中所有数字因此改为由脚本生成而非手工誊写。

**附带发现**：Qwen3 的 tokenizer 对技术标识符的切分明显优于 BGE-M3：

```
tidb_mem_quota_query
  Qwen3  : tid · b · _mem · _quota · _query          (5)
  BGE-M3 : ▁tid·b·_·mem·_·quot·a·_·que·ry            (10，语义被打碎)
```

对一个满是 SQL 关键字与 CLI 参数的语料，这是选用 Qwen3-Embedding 而非 BGE-M3 的一个实质理由。

### 向量检索：8B 嵌入模型只比 40 行 BM25 高 2.1 个点，而这 2.1 个点不显著

同一 5,681 篇语料、同一批 800 条单证据查询，由 `scripts/compare_dense_bm25.py` 逐查询对齐后生成
（dense 向量取自 `probe_mrl_quality.py` 写下的 4096 维缓存，不重新调用 API；
Qwen3-Embedding-8B，query 侧加 instruct 前缀、document 侧不加）：

| 检索器 | R@1 | MRR@10 | nDCG@10 | 向量存储 |
|---|---:|---:|---:|---:|
| 字符 bigram BM25（纯标准库） | 75.9% | 0.857 | 0.893 | — |
| **dense-4096** | **78.0%** | **0.866** | **0.898** | 93.1 MB |
| dense-1024（MRL 截断） | 77.5% | 0.861 | 0.894 | 23.3 MB |
| dense-128（MRL 截断） | 77.0% | 0.853 | 0.887 | 2.9 MB |

一个 80 亿参数的嵌入模型、93 MB 向量、每次查询一次 API 调用，**只换来 2.1 个百分点 ——
而这 2.1 个点在 n=800 上与零不可区分**：

| 指标 | delta（dense − BM25） | 95% CI（配对） | p | 检验 |
|---|---:|---:|---:|---|
| R@1 | +2.12pp | **[−0.88, +5.12]pp** | **0.1986** | McNemar 精确，双尾 |
| MRR@10 | +0.01 | [−0.01, +0.03] | 0.1571 | 配对 bootstrap，单尾 |
| nDCG@10 | +0.01 | [−0.01, +0.02] | 0.2056 | 配对 bootstrap，单尾 |

三个主指标的置信区间**全部跨过零**。即使换成对 dense 最有利的单尾框架，McNemar 仍给出 p = 0.0993。
截断到 128 维（2.9 MB）名义上还高 BM25 1.1 个点 —— 那个差更小，只能读作"与 BM25 打平"，不是"赢"。

> **这不是"两臂相等"。** 同一个区间也容得下 dense 赢 5.1pp；先耗尽的是 n=800，不是差距。
> R@1 用的是**精确 McNemar** 而非 bootstrap：结局是二元的，闭式零分布可用，因而没有蒙特卡洛
> 分辨率地板、不需要随机种子、换台机器还是同一个数。

**而"打平"恰恰是融合最赚的信号 —— 这一点也已被验证，不是推测。** 两臂在 R@1 上的列联表：

|  | dense 命中 | dense 未中 | 合计 |
|---|---:|---:|---:|
| **BM25 命中** | 538 | 69 | 607 |
| **BM25 未中** | 86 | 107 | 193 |
| **合计** | 624 | 176 | 800 |

- **155 条（19.4%）两臂判定相反**，φ = 0.455 —— 相关但远非冗余。头条那个 +2.12pp 是
  69 与 86 两个大列相减的**净值**，列本身比它大一个量级。
- **两臂取并集的 oracle 上限是 86.6%**，比最好的单臂高 8.6pp。这是任何融合的天花板，
  也是唯一诚实的对标基准 —— 不是 100%。
- **失败几乎都是"差一点"**：某一臂 rank-1 落空时，gold 在另一臂里有 **83–88% 落在前 3 名**，
  掉出 top-100 的是 **0.0%**。够得着的东西才谈得上融合。

### 混合检索：RRF 显著打赢 BM25，但打不赢 dense 单臂

两条 run 都已在磁盘上，所以融合是纯客户端重排 —— 这一节的数字在接入任何向量库之前就拿到了
（`scripts/compare_dense_bm25.py`，零 API 调用）。基线取 dense-4096，较强的那条单臂：

| 融合配置 | R@1 | MRR@10 | vs dense | win/loss | p (Holm) |
|---|---:|---:|---:|---:|---:|
| dense-4096（基线） | 78.0% | 0.8661 | — | — | — |
| RRF k=60, depth=100 | 79.8% | 0.8803 | +1.75pp | 38/24 | 0.231 |
| RRF k=60, depth=10 | 79.8% | 0.8802 | +1.75pp | 38/24 | 0.231 |
| RRF k=10, depth=100 | **79.9%** | **0.8810** | +1.87pp | 39/24 | 0.231 |
| RRF k=60, 权重 .3/.7 | 79.5% | 0.8779 | +1.50pp | **16/4** | **0.047** ✱ |

**对 BM25 基线：75.9% → 79.9%，+4.00pp，65 胜 33 负，McNemar 精确 p = 0.0016 —— 显著。**
这才是路线图真正问的那个比较。

三个结论，第二个是这一节真正的内容：

1. **增益最小的那一行是唯一显著的。** 加权 0.3/0.7 只涨 1.50pp 却过了 Holm 校正，
   而涨 1.87pp 的没过 —— 因为 McNemar 只看两个系统判定相反的查询：重排 20 条净赚 12，
   比重排 63 条净赚 15 更可信。**裸 delta 分不出这两者**，win/loss 列才分得出。
   等权 RRF 每修好 5 条就弄坏 3 条；加权 RRF 赢在**少破坏**，不在多修好。
2. **可调项其实只有权重。** k 从 60 调到 10 只动 0.1pp；融合深度 10 与 100 的**逐查询 R@1
   向量逐位相同**（800 条里 0 条 top-1 发生改变，仅 3 条在 top-10 内部重排）。
   → M4 里检索深度是延迟/成本旋钮，不是质量旋钮。
3. 79.9% 只吃掉了 86.6% 那个 oracle 上限的约 37%（自 BM25 起算），剩下的 6.7pp 是 rerank 的作业。

> ⚠️ **这四个配置是在同一批 800 条查询上选出来又在同一批上汇报的**，所以"最好那一行"是上界，
> 不是泛化估计；泛化结论仍需独立的 held-out 切分。另外**「深度无关」只对 1doc 的 R@1 成立，不要外推到
> rerank 窗口** —— 重排器会给它拿到的每个候选打分，窗口大小由召回决定，不由 RRF 会不会提升它决定。

### 多证据分层：dense 强在找齐证据，不强在把任一证据排第一

同一脚本把全部 2,394 条查询按**实际 gold 数量**重分层，而不是相信任务名（2docs 中有 8 条
实际只有 1 个 gold，3docs 中还有 13 条 2-gold 和 1 条 1-gold）。因此样本数是
**809 / 802 / 783**。R@1 在 arity=2/3 时上限只有 1/2、1/3，不能跨块比较；以下同时报告
任意 arity 都是二元的 `hit@1`（是否有一篇 gold 排第一）和 `ALL@10`（是否把整套证据都放进 top-10）：

| arity | n | arm | R@1 | MRR@10 | nDCG@10 | hit@1 | ALL@10 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 809 | BM25 | 75.6% | 0.854 | 0.889 | 75.6% | 99.5% |
|  |  | dense-4096 | 77.6% | 0.863 | 0.895 | 77.6% | 99.3% |
|  |  | RRF k=10, depth=100 | **79.7%** | **0.879** | **0.909** | **79.7%** | **99.8%** |
| 2 | 802 | BM25 | 35.2% | 0.817 | 0.796 | 70.4% | 87.4% |
|  |  | dense-4096 | **35.4%** | **0.828** | 0.824 | **70.8%** | **93.9%** |
|  |  | RRF k=10, depth=100 | 35.1% | 0.827 | **0.827** | 70.2% | 93.8% |
| 3 | 783 | BM25 | 22.7% | 0.799 | 0.748 | 68.2% | 68.7% |
|  |  | dense-4096 | **24.2%** | **0.840** | **0.819** | **72.5%** | **84.4%** |
|  |  | RRF k=10, depth=100 | 23.9% | 0.835 | 0.804 | 71.8% | 79.3% |

这里最容易写错的一句话是“arity 越高，dense 越赢”。在跨 arity 可比的 `hit@1` 上，dense 相对
BM25 依次是 **+1.98pp / +0.37pp / +4.34pp**，配对 95% CI 分别为
**[−1.11, +4.94]pp / [−3.49, +4.24]pp / [+0.64, +8.17]pp**；经同一 family 的 Holm
校正，三行都不显著（校正 p = 1.000 / 1.000 / 0.176）。arity=2 甚至比 arity=1 更弱，
所以“dense 碾压 2docs”只是从 `hit@1` 偷换成 `ALL@10` 后的指标假象。

**真正成立、而且更有用的结论是完整证据召回。** `ALL@10` 上，dense 相对 BM25 在 arity=2
提升 **+6.48pp**（95% CI **[+4.24, +8.85]pp**，71 胜 19 负，Holm p = **2.81e-07**），
arity=3 提升 **+15.71pp**（95% CI **[+12.90, +18.52]pp**，133 胜 10 负，Holm
p = **1.66e-27**）。也就是说，dense 不擅长把“某一篇”推到第 1，但明显更擅长把回答问题所需的
**整套证据**捞进 top-10。

这也推翻了“一套 RRF 权重通吃”的设想。第 4 节在 1doc 上选出的 RRF k=10/depth=100，
到 arity=3 时相对 dense 的 `ALL@10` **84.4% → 79.3%（−5.11pp，95% CI
[−7.02, −3.19]pp，11 胜 51 负，6 项 Holm p = 1.67e-06）**；同一块的 `hit@1`
只有 −0.77pp（95% CI [−3.58, +2.04]pp，校正 p = 1.000）。退化只发生在“找齐整套证据”这个口径，
不能泛化成“融合整体伤害多文档查询”。因此后续要么让权重随问题变化，要么如实按 arity 报告，
不再命名一个全局赢家。

### 重排：单证据改善 top-1，多证据改善完整证据召回；top-100 未胜 top-50

在冻结的 **dense-4096 + char-bigram BM25、等权 RRF k=10/depth=100** 候选上，
Qwen3-Reranker-8B 对每条查询的前 100 个候选统一打分一次；同一份本地缓存随后离线构造两个 treatment：
只重排前 50 个并原样接回尾部，或重排全部 100 个。以下由
`scripts/evaluate_rerank.py --analyze --resamples 100000` 生成：

| arity | n | arm | R@1 | MRR@10 | nDCG@10 | hit@1 | ALL@10 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 809 | hybrid RRF k=10/d=100 | 79.7% | 0.879 | 0.909 | 79.7% | 99.8% |
|  |  | **rerank-8b@50** | **85.8%** | **0.919** | **0.939** | **85.8%** | 99.8% |
|  |  | rerank-8b@100 | 85.8% | 0.919 | 0.939 | 85.8% | 99.8% |
| 2 | 802 | hybrid RRF k=10/d=100 | 35.1% | 0.827 | 0.827 | 70.2% | 93.8% |
|  |  | **rerank-8b@50** | **35.7%** | **0.834** | **0.836** | **71.3%** | **94.3%** |
|  |  | rerank-8b@100 | 35.7% | 0.834 | 0.836 | 71.3% | 94.4% |
| 3 | 783 | hybrid RRF k=10/d=100 | 23.9% | 0.835 | 0.804 | 71.8% | 79.3% |
|  |  | **rerank-8b@50** | **24.6%** | **0.850** | **0.830** | **73.7%** | **86.5%** |
|  |  | rerank-8b@100 | 24.6% | 0.849 | 0.830 | 73.7% | 86.7% |

四个检验族在看结果前固定：efficacy 的二元/连续指标各一个，top-100 对 top-50 的 depth
二元/连续指标各一个；每族分别做 Holm 校正。由此得到的结论不是一个全局平均数：

- **arity=1 的 `hit@1` 显著提升 6.06pp**：79.7% → 85.8%，95% CI
  **[+3.34, +8.78]pp**，91 胜 / 42 负，McNemar 原始 p = 2.59e−05，12 项 Holm
  p = **0.0003**。MRR@10 与 nDCG@10 同样显著，分别约 +0.04 与 +0.03。
- **arity=3 的 `ALL@10` 显著提升**：top-50 为 **+7.15pp**（95% CI
  **[+4.98, +9.32]pp**，68/12，Holm p = **1.32e−09**），top-100 为
  **+7.41pp**（**[+5.24, +9.58]pp**，70/12，Holm p = **4.93e−10**）；
  nDCG@10 约 +0.03，也通过校正。
- **arity=2 没有任何 efficacy 比较通过 Holm 校正。** 点估计虽略升，但置信区间跨零；不能写成
  “rerank 对所有查询都有效”。
- **top-100 没有显著优于 top-50。** 三个 arity 的 `hit@1` 完全相同；ALL@10 只多
  0.00 / 0.12 / 0.26pp，6 项二元 depth family 的 Holm p 全为 1.000；连续指标也无一通过校正。
  因而当前证据支持部署时选 **top-50**，而不是把“没检出差异”夸成“两者等价”。

> ⚠️ 这次付费 sweep 跨越多个时间段断点续跑，查询顺序又按任务排列，与 arity 高度相关；sidecar
> 固定了 endpoint、model、instruction、候选顺序和实际输入文本，却无法记录中转站未公开的后端部署版本。
> 未观察到缓存污染或后端漂移，但也没有独立测过 reranker 的重复性。因此逐查询 baseline-vs-rerank
> 比较代表已接受的这一次打分实现；**跨 arity 解读额外依赖 endpoint 在整轮期间保持平稳**。这是未量化的
> stationarity 威胁，不是已经发生漂移的证据。另请注意，本实验的 dense 臂是 **4096 维**；它没有回答
> 另一个尚未运行的 dense-1024 hybrid 问题，不能把本表改名成 1024 结果。

### MRL 降维：存储降 75%，R@1 无显著损失

Qwen3 的技术报告没有发布任何维度-质量曲线（arXiv v3 全文里 "MRL" 只出现两次，都在表头注释）。
以下为 5,681 篇语料 / 800 条查询实测，R@1 带 95% bootstrap CI，p 为「4096 优于该行」的
单尾配对 bootstrap 检验、经 Holm 校正跨全表：

| dim | R@1 [95% CI] | MRR@10 | vs 4096 | p (Holm) | 存储 |
|---:|---|---:|---:|---:|---:|
| **4096** | 78.0% [75.1%, 80.8%] | 0.866 | — | baseline | 93.1 MB |
| 2048 | 77.6% [74.6%, 80.4%] | 0.863 | −0.38pp | 0.684 | 46.5 MB |
| **1024** | 77.5% [74.6%, 80.4%] | 0.861 | −0.50pp | 0.684 | **23.3 MB** |
| 512 | 78.4% [75.5%, 81.2%] | 0.864 | +0.37pp | 0.684 | 11.6 MB |
| 256 | 76.6% [73.6%, 79.5%] | 0.854 | −1.38pp | 0.312 | 5.8 MB |
| 128 | 77.0% [74.1%, 79.9%] | 0.853 | −1.00pp | 0.521 | 2.9 MB |
| **64** | 73.8% [70.8%, 76.9%] | 0.833 | **−4.25pp** | **0.001†** ✱ | 1.5 MB |

**只有 64 维显著变差。128 维往上全部不可区分。**

> 两处 p 值的读法，都是统计层审计的产物：
> **`†` 表示蒙特卡洛分辨率，而不是 `<` 上界。** 64 维那格的原始 add-one 估计触到
> 10,000 次重采样的地板（`1/(10001)` ≈ 1.0e−04；0/10,000 个零分布样本达到观测值），
> 经 6 项 Holm 校正后显示为 0.001†。这一轮只能说估计停在地板；**不能据此证明真实尾概率
> 小于该值**。要提高分辨率需增加重采样次数；这里的 R@1 在不同维度间并非二元配对结局，不能
> 像单证据系统间 hit/miss 比较那样直接换成 McNemar。
> **2048 / 1024 / 512 三行的 0.684 也不是三个相同的发现**：Holm 强制校正后的 p 单调不减，
> 于是 −0.50pp 的回退和 +0.37pp 的改进被压到同一个数上。方向看 delta 列，不要看 p 列。

三点必须一起读，否则容易过度解读：

1. **检验是有分辨力的。** 它抓到了 64 维（Holm p=0.001†），所以「128–2048 不显著」不是检验太钝。
2. **512 维比 4096 高 0.37pp —— 这是噪声，而且是有用的噪声。** 曲线非单调（256 的 76.6%
   低于 128 的 77.0%）本身就说明这一段全落在测量误差内。n=800、R@1≈78% 时标准误约 1.46pp。
3. **机制不是「MRL 前置聚集」。** 跨文档逐维方差首尾比 **1.047**，是平的 —— 信息并没有按下标
   排序。更可能的解释是向量内在维度远低于 4096，前缀截断近似随机投影；Johnson–Lindenstrauss
   在 n=5,681 时给出约 200 维的保距下界，与实测「128–256 是拐点」吻合。
   **报告行为，不宣称机制。**

<details>
<summary>饱和第二次咬人：同一配置，语料一扩，效应量增大 1.6 倍并跨过显著线</summary>

先在 2,000 篇的子集上跑过一轮，结论是「没有任何维度显著变差」：

| 语料 | 64 维 vs 4096 | p (Holm) |
|---|---:|---:|
| 2,000 篇 | −2.67pp | 0.178（不显著） |
| **5,681 篇** | **−4.25pp** | **0.001（显著）** |

2,000 篇上 R@5 从 4096 一路 99.7% 平到 128 —— 那一列已经饱和。这与本文开篇诊断 CRUD-RAG
官方子集时的机制完全相同，只是这次落在自己头上。**语料规模决定尺子上有没有东西可量**，
这条规律对被评测的对象和评测者本身一视同仁。
</details>

### 测量供应商之前，先测它的噪声底

判断「`dimensions=n` 是前缀切片还是独立训练的 head」时，第一版用了绝对阈值（逐分量最大差
< 1e-4），得出「独立 head」——**这个结论是错的**。因为没有先问：这个 provider 自己重现得了吗？

```
同一段文本，同一批次内重复      cos = 1.000000000000
同一段文本，分两次请求          cos = 0.999930725934   ← 噪声底
```

**同一段文本分两次请求，向量就已经不一样了**，每分量差正好也是 1e-3 量级 —— 第一版的阈值在拿噪声当信号。
成因不是 bug：批式 GPU 推理中融合矩阵乘的规约顺序随 batch 形状与位置变化，非结合律的浮点加法给出不同结果。

以噪声底为基准重测，客户端切片与原生 `dimensions=n` 的余弦是 `0.999932 / 0.999932 / 0.999934`
（512 / 1024 / 2048），**全部落在噪声底之上**：切片与原生的差异，不大于同一次调用重跑一遍的差异。
排序检验独立确认（20 条近义句，top-10 顺序完全一致，分数扰动仅为相邻名次间隔的 7%）。

三个直接后果：

- **只存一份 4096 维向量**，全部 MRL 档位是它的客户端切片 —— 上表 7 行共用**一次**嵌入，零额外 API 调用；
- 向量相关的单元测试**只能带容差比较，绝不能断言相等**；
- 任何比 `1 − 6.9e-05` 更接近的两个向量在本 provider 上不可区分，这是所有向量比较的分辨率上限。

> 另：`.env` 指向的是一个 One Hub 中转站而非 SiliconFlow。它有两个状态码相同、成因完全不同的 403
> —— Cloudflare 拦截 `Python-urllib/3.x` 默认 UA（`error code 1010`），以及 key 分组的时段限制。
> 两者都不是鉴权失败，不读 response body 会误判成 key 有问题。

<!-- BEGIN H-HYBRID-MRL1024-EVIDENCE -->
### H：dense-1024 + BM25 的独立 hybrid 基线

H 已在 **5,681 篇完整新闻文档 / 2,394 条 query** 上独立运行。文档和 query 都从同一份已冻结 4096 维 cache 取前 1024 维后逐行 L2 重归一化；A/E 各取 top-100，再做等权客户端 exact RRF (`k=10`，无 rerank、零 API 调用)。G 由同一 4096 维矩阵离线重建，只作 retention difference reference；I/J 仍是 G 上的付费 rerank，未改名。

**历史 1doc 对齐（n=800；不重复进入显著性 family）**

| arm | R@1 [95% CI] | MRR@10 [95% CI] | nDCG@10 [95% CI] |
|---|---:|---:|---:|
| A：BM25 char-bigram | 75.9% [72.9%, 78.9%] | 0.857 [0.838, 0.874] | 0.893 [0.879, 0.906] |
| E：dense-1024 | 77.5% [74.5%, 80.4%] | 0.861 [0.842, 0.879] | 0.894 [0.880, 0.909] |
| H：A+E exact RRF | 79.6% [76.8%, 82.4%] | 0.878 [0.860, 0.894] | 0.908 [0.895, 0.921] |
| G：A+dense-4096 exact RRF | 79.9% [77.0%, 82.6%] | 0.881 [0.863, 0.898] | 0.911 [0.898, 0.924] |

**全部 query 按实际 gold arity 分层（每格均为 mean [pointwise 95% CI]）**

| arity | n | arm | R@1 | hit@1 | ALL@10 | MRR@10 | nDCG@10 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 809 | A：BM25 char-bigram | 75.6% [72.7%, 78.6%] | 75.6% [72.7%, 78.6%] | 99.5% [98.9%, 99.9%] | 0.854 [0.835, 0.872] | 0.889 [0.875, 0.903] |
|  |  | E：dense-1024 | 77.1% [74.3%, 80.0%] | 77.1% [74.3%, 80.0%] | 99.1% [98.4%, 99.8%] | 0.858 [0.840, 0.877] | 0.891 [0.877, 0.906] |
|  |  | H：A+E exact RRF | 79.5% [76.8%, 82.3%] | 79.5% [76.8%, 82.3%] | 99.8% [99.4%, 100.0%] | 0.875 [0.858, 0.893] | 0.906 [0.893, 0.919] |
|  |  | G：A+dense-4096 exact RRF | 79.7% [76.9%, 82.4%] | 79.7% [76.9%, 82.4%] | 99.8% [99.4%, 100.0%] | 0.879 [0.861, 0.896] | 0.909 [0.896, 0.921] |
| 2 | 802 | A：BM25 char-bigram | 35.2% [33.6%, 36.8%] | 70.4% [67.2%, 73.6%] | 87.4% [85.0%, 89.7%] | 0.817 [0.796, 0.838] | 0.796 [0.780, 0.812] |
|  |  | E：dense-1024 | 35.7% [34.2%, 37.3%] | 71.4% [68.3%, 74.6%] | 93.0% [91.3%, 94.8%] | 0.832 [0.813, 0.851] | 0.822 [0.809, 0.836] |
|  |  | H：A+E exact RRF | 36.2% [34.6%, 37.7%] | 72.3% [69.2%, 75.4%] | 93.3% [91.4%, 95.0%] | 0.838 [0.819, 0.857] | 0.831 [0.817, 0.844] |
|  |  | G：A+dense-4096 exact RRF | 35.1% [33.5%, 36.7%] | 70.2% [67.1%, 73.3%] | 93.8% [92.0%, 95.4%] | 0.827 [0.809, 0.846] | 0.827 [0.813, 0.840] |
| 3 | 783 | A：BM25 char-bigram | 22.7% [21.6%, 23.8%] | 68.2% [64.9%, 71.5%] | 68.7% [65.5%, 71.9%] | 0.799 [0.778, 0.821] | 0.748 [0.730, 0.766] |
|  |  | E：dense-1024 | 24.1% [23.1%, 25.2%] | 72.4% [69.2%, 75.5%] | 83.1% [80.5%, 85.7%] | 0.839 [0.819, 0.857] | 0.816 [0.801, 0.830] |
|  |  | H：A+E exact RRF | 24.1% [23.0%, 25.1%] | 72.3% [69.1%, 75.4%] | 80.5% [77.7%, 83.3%] | 0.836 [0.817, 0.855] | 0.805 [0.790, 0.820] |
|  |  | G：A+dense-4096 exact RRF | 23.9% [22.9%, 25.0%] | 71.8% [68.7%, 75.0%] | 79.3% [76.5%, 82.1%] | 0.835 [0.816, 0.854] | 0.804 [0.788, 0.818] |

**预声明检验**：双尾；连续指标用 10,000 次 centred paired query bootstrap，二元指标用 exact McNemar；四个 family 各自 Holm。

| family | arity | contrast | metric | delta [paired 95% CI] | W/L/T | raw p | Holm p |
|---|---:|---|---|---:|---:|---:|---:|
| efficacy / binary | 1 | H − A | hit@1 | +3.83pp [+1.48, +6.18] | 64/33/712 | 0.0022 | 0.0215 * |
| efficacy / binary | 1 | H − A | ALL@10 | +0.25pp [+0.00, +0.62] | 2/0/807 | 0.5000 | 1.0000 |
| efficacy / binary | 1 | H − E | hit@1 | +2.35pp [+0.49, +4.33] | 41/22/746 | 0.0226 | 0.1580 |
| efficacy / binary | 1 | H − E | ALL@10 | +0.62pp [+0.12, +1.24] | 5/0/804 | 0.0625 | 0.3750 |
| efficacy / binary | 2 | H − A | hit@1 | +1.87pp [-1.00, +4.74] | 74/59/669 | 0.2246 | 1.0000 |
| efficacy / binary | 2 | H − A | ALL@10 | +5.86pp [+4.11, +7.73] | 52/5/745 | 6.40e-11 | 7.04e-10 * |
| efficacy / binary | 2 | H − E | hit@1 | +0.87pp [-2.00, +3.62] | 70/63/669 | 0.6030 | 1.0000 |
| efficacy / binary | 2 | H − E | ALL@10 | +0.25pp [-1.12, +1.62] | 17/15/770 | 0.8601 | 1.0000 |
| efficacy / binary | 3 | H − A | hit@1 | +4.09pp [+1.40, +6.77] | 74/42/667 | 0.0038 | 0.0343 * |
| efficacy / binary | 3 | H − A | ALL@10 | +11.75pp [+9.45, +14.18] | 96/4/683 | 6.45e-24 | 7.74e-23 * |
| efficacy / binary | 3 | H − E | hit@1 | -0.13pp [-3.07, +2.94] | 70/71/642 | 1.0000 | 1.0000 |
| efficacy / binary | 3 | H − E | ALL@10 | -2.68pp [-4.47, -0.89] | 15/36/732 | 0.0046 | 0.0368 * |
| efficacy / continuous | 1 | H − A | MRR@10 | +0.0218 [+0.0083, +0.0354] | 96/61/652 | 0.0020 | 0.0140 * |
| efficacy / continuous | 1 | H − A | nDCG@10 | +0.0168 [+0.0065, +0.0270] | 96/61/652 | 0.0016 | 0.0128 * |
| efficacy / continuous | 1 | H − E | MRR@10 | +0.0176 [+0.0065, +0.0289] | 82/51/676 | 0.0031 | 0.0186 * |
| efficacy / continuous | 1 | H − E | nDCG@10 | +0.0148 [+0.0063, +0.0235] | 82/51/676 | 0.0012 | 0.0108 * |
| efficacy / continuous | 2 | H − A | MRR@10 | +0.0204 [+0.0047, +0.0368] | 131/78/593 | 0.0134 | 0.0590 |
| efficacy / continuous | 2 | H − A | nDCG@10 | +0.0344 [+0.0245, +0.0445] | 292/157/353 | 1.00e-04† | 0.0012† * |
| efficacy / continuous | 2 | H − E | MRR@10 | +0.0054 [-0.0110, +0.0211] | 103/90/609 | 0.5056 | 1.0000 |
| efficacy / continuous | 2 | H − E | nDCG@10 | +0.0083 [-0.0008, +0.0173] | 265/200/337 | 0.0722 | 0.2166 |
| efficacy / continuous | 3 | H − A | MRR@10 | +0.0369 [+0.0212, +0.0527] | 129/62/592 | 1.00e-04† | 0.0012† * |
| efficacy / continuous | 3 | H − A | nDCG@10 | +0.0568 [+0.0476, +0.0660] | 382/143/258 | 1.00e-04† | 0.0012† * |
| efficacy / continuous | 3 | H − E | MRR@10 | -0.0023 [-0.0186, +0.0145] | 92/95/596 | 0.7900 | 1.0000 |
| efficacy / continuous | 3 | H − E | nDCG@10 | -0.0107 [-0.0189, -0.0024] | 241/276/266 | 0.0118 | 0.0590 |
| retention / binary | 1 | H − G | hit@1 | -0.25pp [-1.36, +0.87] | 9/11/789 | 0.8238 | 1.0000 |
| retention / binary | 1 | H − G | ALL@10 | +0.00pp [+0.00, +0.00] | 0/0/809 | 1.0000 | 1.0000 |
| retention / binary | 2 | H − G | hit@1 | +2.12pp [+0.62, +3.62] | 26/9/767 | 0.0060 | 0.0359 * |
| retention / binary | 2 | H − G | ALL@10 | -0.50pp [-1.25, +0.12] | 2/6/794 | 0.2891 | 1.0000 |
| retention / binary | 3 | H − G | hit@1 | +0.51pp [-0.89, +1.92] | 18/14/751 | 0.5966 | 1.0000 |
| retention / binary | 3 | H − G | ALL@10 | +1.15pp [+0.00, +2.30] | 15/6/762 | 0.0784 | 0.3918 |
| retention / continuous | 1 | H − G | MRR@10 | -0.0033 [-0.0092, +0.0027] | 14/22/773 | 0.2848 | 1.0000 |
| retention / continuous | 1 | H − G | nDCG@10 | -0.0025 [-0.0070, +0.0019] | 14/22/773 | 0.2673 | 1.0000 |
| retention / continuous | 2 | H − G | MRR@10 | +0.0105 [+0.0029, +0.0181] | 43/26/733 | 0.0079 | 0.0474 * |
| retention / continuous | 2 | H − G | nDCG@10 | +0.0039 [-0.0001, +0.0081] | 83/73/646 | 0.0594 | 0.2970 |
| retention / continuous | 3 | H − G | MRR@10 | +0.0015 [-0.0061, +0.0088] | 29/32/722 | 0.6842 | 1.0000 |
| retention / continuous | 3 | H − G | nDCG@10 | +0.0017 [-0.0020, +0.0053] | 115/107/561 | 0.3557 | 1.0000 |

校正后拒绝数：efficacy / binary 5/12；efficacy / continuous 7/12；retention / binary 1/6；retention / continuous 1/6。
H 与 G 检测到经校正差异：arity=2 hit@1、arity=2 MRR@10；方向必须结合 delta 读取。

> `R@1` 在 arity=2/3 是分数型、上限分别为 1/2 与 1/3，只作描述，未塞进 McNemar。`†` 是 add-one Monte Carlo floor，不是严格 `<` 上界。全部 2,394 条 query 已参与既往探索，因此这是 exploratory benchmark，不是未触碰 test；CI 是 pointwise，结论只绑定本次 cache fingerprints。
<!-- END H-HYBRID-MRL1024-EVIDENCE -->

### 在线链路：把离线证据原样搬上去，而不是搬一个像它的东西

离线冻结的配置是「dense-4096 与 char-bigram BM25 各取 100 → 等权 RRF k=10/depth=100 →
一次性送 100 篇给 reranker → 只用前 50 个分数改排序」。在线实现里有三处**容易在不知不觉中偏离**它：

**① 稀疏检索不能交给数据库去分词。** Milvus 的 BM25 function 会重新切词并估算它自己的 IDF，
Lite 上的 IDF 还是 segment 局部的。所以文档侧在客户端算完整的 BM25 贡献存进 `SPARSE_FLOAT_VECTOR`，
查询侧是去重词的二值向量，metric 用 `IP` —— 内积**就是**本地 BM25 的分数（`tests/test_sparse.py` 逐查询对齐）。
代价是 BM25 统计量全局耦合：新增一个 chunk 会改变每一行的 IDF 与平均长度，所以**每次构建都整体重算稀疏向量**，
不能只更新变化的那篇文档。

**② 融合放在客户端，而不是 `RRFRanker`。** 公式一致，但并列名次的顺序不一致：本地实现用
`math.fsum` + 文档 id 打破并列，服务端按先到顺序；且服务端的最终 `limit` 会在并列边界上直接丢掉一个，
事后补不回来。所以默认路径是两臂分别取回、调用仓库已有的 `reciprocal_rank_fusion`；
`hybrid_search` 保留为通过 arm/fusion 对齐测试之后的优化项。

**③ 「top-50」是应用深度，不是请求形状。** 离线那一行是**发了 100 篇**再取前 50 个分数，
不是发 50 篇。二者的 provider 输入不同，结论不能互换，所以 `OnlineSettings` 把
`rerank_request_depth=100` 与 `rerank_apply_depth=50` 显式分成两个字段。
同理，`benchmark_exact()`（新闻语料、dense-4096）与面向 TiDB 文档的 `product()` profile 分别命名、分别指纹 ——
CRUD-RAG 上测出的 +6.06pp 不会因为换了语料就自动成立。

在线核心不导入 `pymilvus`、不读 `.env`、不碰文件系统：编码器、存储、reranker 和时钟都是注入的 Protocol，
所以默认 CI 不需要 API key、语料或原生依赖就能验证请求形状、融合顺序、失败边界与各阶段耗时。
真实 Milvus Lite 的 schema / 两臂 / alias / 重开由 opt-in 的 `scripts/verify_milvus_store.py` 覆盖。

**④ 词表必须跟着集合一起落盘。** 查询向量的 term index 只有在**构建文档时用的那份词表**下才有意义；
换一份词表，内积就会打到恰好占着那些位置的词上——**排序静默出错，不报错**。所以发布路径会把
vocabulary + IDF 写成 `sparse_index.json`，查询端加载后先比对 `state.json` 里的 fingerprint，不一致直接拒绝启动。
（只有发布成功才写这个文件：否则查询端可能加载到一份没有任何在线集合与之对应的词表。）

> **索引构建是先建后切。** `scripts/build_index.py` 把整批期望行写进一个带版本号的 shadow collection，
> 校验行数与抽样回读之后才切 alias，最后才落盘成功状态；任何一步失败，线上 alias 与状态文件都不动。
> **实际构建（2026-08-22）**：450 篇 evergreen 文档 → **1,832 chunks**、**75,620** 个 bigram 词表，
> 1,832 条 4096 维向量、115 个批次、**零重试零错误**；重跑一次是 `unchanged=450`、`reusable=1,832/1,832`，
> 零新增请求——幂等性由 chunk id 而非时间戳保证。
>
> 单条查询的端到端冒烟（`scripts/query_index.py`，「如何用 BR 做全量快照备份？」）：两臂各 100、重合 68、
> 融合 132 个候选；重排把「快照备份使用指南 > 对集群进行快照备份」从融合第 17 位提到第 1 位。
> 阶段耗时 encode 2.2s / search 1.0s / fuse 0.1ms / fetch 26ms / **rerank 11.8s**（100 篇一次请求）。
> ⚠️ 这是**一条查询的观察**，不是延迟基准也不是质量证据——p50/p95/QPS 属于 M8；
> 系统级检索质量见下方 490-pair 离线评估，不能用这一条冒烟查询替代。

<!-- BEGIN M8-SERVICE-BENCHMARK -->
### HTTP 服务与 M8 性能基准

FastAPI 服务复用同一条同步 `OnlineRetriever`：dense / sparse 两臂各取 100，客户端 exact RRF，再按 profile 选择 rerank；单文件前端只展示检索 passage 与阶段耗时，**不生成答案，也不调用 chat completion**。

**正式 profile**：`tidb-docs-exact-rrf10-cached-query-no-rerank-v1`（embedding=`cached-qwen3-embedding-8b-tidb-query-4096-v1`；rerank=`disabled-identity-fused-order-v1`；cache-backed，本地测量不包含 provider 墙钟）。通过 HTTP 完成 980 次正式请求（另有 20 次 warm-up，全部排除）；fixture 含 980 条本地 query，仅发布其 SHA-256。

| HTTP 指标 | p50 | p95 | p99 |
|---|---:|---:|---:|
| 客户端端到端 | 197.4 ms | 228.3 ms | 244.4 ms |

成功 **980/980**，错误率 **0.00%**，成功吞吐 **4.97 QPS**；测量窗口 197.019s、并发 1。HTTP 状态：200=980。

| 服务阶段 | p50 | p95 | p99 |
|---|---:|---:|---:|
| dense encode | 0.0 ms | 0.1 ms | 0.1 ms |
| sparse encode | 0.1 ms | 0.1 ms | 0.1 ms |
| dense search | 84.0 ms | 105.5 ms | 111.9 ms |
| sparse search | 92.9 ms | 114.0 ms | 123.3 ms |
| fusion | 0.1 ms | 0.2 ms | 0.2 ms |
| fetch | 14.1 ms | 24.7 ms | 36.1 ms |
| rerank | 0.0 ms | 0.0 ms | 0.0 ms |
| service total | 195.0 ms | 225.9 ms | 242.1 ms |

> 百分位固定用 NumPy `linear`；环境为 `Windows-11-10.0.26200-SP0` / Python `3.13.3` / `AMD64`。失败请求不进入成功 latency 或阶段百分位，QPS=成功数/正式测量墙钟。
> 这是本机 HTTP profile 的观测，不是公网或生产 SLA；质量指标与显著性检验另见 TiDB pooled-qrels 评估。numeric samples 只含 status、elapsed 与阶段秒数，不含 query、passage、doc id、向量或 provider payload。
<!-- END M8-SERVICE-BENCHMARK -->

<!-- BEGIN M7-CHUNK-SWEEP -->
### M7 分块粒度 sweep（TiDB source-level known-item）

> **探索性声明**：这是 `400-origin exploratory known-item source retrieval sweep`。
> 先在 raw chunk 上分别构建 BM25 / dense-4096，再做 exact RRF k=10/depth=100，
> 最终 arm 才按 source 首次出现折叠；不启用 rerank、Milvus 或 chat。

| profile | docs | chunks | split-trigger exceedance | exact reuse / required new vectors |
|---|---:|---:|---:|---:|
| `tidb-chunk-t256-h384-v1` | 450 | 2,802 | 430 (15.35%) | 252 / 2,550 |
| `tidb-chunk-t400-h600-v1` | 450 | 1,832 | 282 (15.39%) | 1,832 / 0 |
| `tidb-chunk-t800-h1200-v1` | 450 | 1,019 | 116 (11.38%) | 190 / 829 |

> **主终点（overall，pair 内 direct/paraphrase 取均值；95% CI 为 245 个 source cluster 整簇 bootstrap）**：

| profile | origin-source MRR@10 | Hit@1 | Hit@10 |
|---|---:|---:|---:|
| `tidb-chunk-t256-h384-v1` | 0.870 [0.846, 0.893] | 0.787 [0.749, 0.823] | 0.994 [0.987, 0.999] |
| `tidb-chunk-t400-h600-v1` | 0.862 [0.839, 0.884] | 0.771 [0.734, 0.806] | 0.992 [0.984, 0.998] |
| `tidb-chunk-t800-h1200-v1` | 0.850 [0.824, 0.874] | 0.753 [0.715, 0.791] | 0.991 [0.982, 0.998] |

> **预声明 Family A（Holm 2-test）**：主终点比较 256−400 与 800−400；Family B 独立检验 direct/paraphrase difference-in-differences。

| comparison | Δ MRR@10 [95% CI] | p | p(Holm) | W/L/T |
|---|---:|---:|---:|---:|
| `tidb-chunk-t256-h384-v1-vs-tidb-chunk-t400-h600-v1` | 0.0080 [-0.0049, 0.0206] | 0.2239 | 0.2239 | 65/63/362 |
| `tidb-chunk-t800-h1200-v1-vs-tidb-chunk-t400-h600-v1` | -0.0125 [-0.0262, 0.0009] | 0.0703 | 0.1406 | 70/90/330 |

> 不能把未检出差异写成等价、无损或全局最优；source-level known-item retrieval 也不等于 answer-bearing passage recall。
> confirmed-source 只是 canonical 400-only judgement pool 的 secondary sensitivity，未判断 source 不是可靠负例。
<!-- END M7-CHUNK-SWEEP -->

<!-- BEGIN TIDB-EVAL-EVIDENCE -->
> **TiDB 合成评测集（本地报告生成）**：从已发布的 1,832 个 chunk 中按主题确定性抽样 500 个，双阶段生成并验证后保留 490 个完整 pair（direct / paraphrase 各 490 条）。
> 四条冻结 run 在两种表面形式上按系统 top-20 取并集，并强制纳入生成 chunk，得到 24,525 个 pair-candidate 判断槽（每 pair 23–74）。
> rank-blinded、固定顺序的 LLM judge 共完成 3,287 个 batch；原始 grade 0/1/2 为 18,840/4,347/1,338。
> 生成 chunk 与 grade 2 规则不一致 1/490（0.204%）。四系统实际返回的 top-1/top-10 均达到 100% **已判断覆盖**。

> **离线质量（overall，direct/paraphrase 先在 pair 内取均值；括号为 source-cluster bootstrap 95% CI，245 个源聚类）**：

| 冻结系统 | Hit@1 | R@1（备选完整答案覆盖） | MRR@10 | binary nDCG@10 | graded nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 char-bigram | 0.660 [0.626, 0.695] | 0.429 [0.397, 0.460] | 0.758 [0.731, 0.784] | 0.701 [0.675, 0.727] | 0.639 [0.618, 0.660] |
| dense Qwen3-4096 | 0.789 [0.759, 0.819] | 0.495 [0.460, 0.529] | 0.862 [0.841, 0.883] | 0.793 [0.774, 0.812] | 0.739 [0.725, 0.752] |
| RRF k=10/depth=100 | 0.764 [0.734, 0.795] | 0.484 [0.452, 0.516] | 0.855 [0.836, 0.875] | 0.796 [0.776, 0.816] | 0.739 [0.724, 0.754] |
| Qwen3 rerank@50 | 0.935 [0.916, 0.952] | 0.614 [0.575, 0.651] | 0.964 [0.953, 0.974] | 0.914 [0.899, 0.928] | 0.833 [0.823, 0.844] |

> **预声明主检验族**：主终点为 pair-mean binary nDCG@10；双尾 centred paired source-cluster bootstrap（10,000 次，245 个源聚类）并在以下 4 个比较内做 Holm 校正。

| treatment − comparator | Δ [95% CI] | win/loss/tie | p | p(Holm) |
|---|---:|---:|---:|---:|
| dense Qwen3-4096 − BM25 char-bigram | 0.0917 [0.0682, 0.1161] | 254/123/113 | 1.00e-04† | 0.0004† * |
| RRF k=10/depth=100 − BM25 char-bigram | 0.0951 [0.0816, 0.1088] | 282/58/150 | 1.00e-04† | 0.0004† * |
| RRF k=10/depth=100 − dense Qwen3-4096 | 0.0034 [-0.0110, 0.0180] | 169/162/159 | 0.6464 | 0.6464 |
| Qwen3 rerank@50 − RRF k=10/depth=100 | 0.1177 [0.1019, 0.1340] | 265/46/179 | 1.00e-04† | 0.0004† * |

> **direct → paraphrase robustness（独立 4-test Holm family）**：

| 系统 | direct nDCG | paraphrase nDCG | Δ(para-direct) [95% CI] | p(Holm) |
|---|---:|---:|---:|---:|
| BM25 char-bigram | 0.783 | 0.619 | -0.1641 [-0.1930, -0.1358] | 0.0004† * |
| dense Qwen3-4096 | 0.806 | 0.780 | -0.0257 [-0.0410, -0.0107] | 0.0012 * |
| RRF k=10/depth=100 | 0.833 | 0.759 | -0.0740 [-0.0928, -0.0555] | 0.0004† * |
| Qwen3 rerank@50 | 0.923 | 0.906 | -0.0167 [-0.0286, -0.0052] | 0.0044 * |

> **词面重叠分层（描述性，不做 subgroup p 值）**：每个 surface 内按 stored bigram containment 做保留 ties 的 mid-CDF 三分位；下表为主指标。

| surface / stratum | n | overlap 范围 | BM25 | dense | RRF | rerank |
|---|---:|---:|---:|---:|---:|---:|
| direct / low | 167 | 0.280–0.625 | 0.678 | 0.763 | 0.776 | 0.905 |
| direct / middle | 157 | 0.630–0.711 | 0.817 | 0.815 | 0.845 | 0.927 |
| direct / high | 166 | 0.714–0.917 | 0.857 | 0.841 | 0.880 | 0.936 |
| paraphrase / low | 164 | 0.050–0.419 | 0.384 | 0.744 | 0.642 | 0.882 |
| paraphrase / middle | 163 | 0.421–0.552 | 0.689 | 0.788 | 0.791 | 0.905 |
| paraphrase / high | 163 | 0.553–0.833 | 0.786 | 0.809 | 0.846 | 0.931 |

> **边界**：这些是 same-model self-agreement 的 synthetic pooled labels，不是 TiDB 上游人工 gold；100% 是 judged coverage，不是质量。
> grade 2 文档是可独立完整回答的**替代证据**，所以 Hit@1 / MRR / nDCG 是主视图，不报告要求找齐所有替代答案的 ALL@10；graded nDCG 采用 full=3、partial=1 gain。
> 生成 chunk 经单独 verification pass 证实（same-model self-agreement）；即使 relevance judge 给 0/1，仍作为 operational full gold。pool 外保持未判断；本评测未用于反向调参。
> † 表示 add-one Monte Carlo floor，不是严格 `<` 上界；CI 是 pointwise，同一 source 的 pair 已整簇重采样，跨 source/theme 的残余相关性未建模。
<!-- END TIDB-EVAL-EVIDENCE -->

---

## 已实现

```
src/zhrag/
  io_utils.py            单一 UTF-8 文件 I/O 出入口（含增量缓存用的 append_jsonl）
  embedding_contract.py  provider-neutral 的 embedding cache provenance 合同
  tokens.py              按 Qwen3 tokenizer 实测标定的中英双分量 token 估算
  lexical/
    analyzers.py         字符 n-gram / jieba（可选）/ 并集
    bm25.py              Okapi BM25，可插拔 analyzer
  chunking/
    markdown.py          标题感知的两阶段分块，保护代码块与表格
  eval/
    crud.py              CRUD-RAG 语料重建：去重文档池 + 三个任务的 qrels
    metrics.py           R@k / MRR / nDCG / ALL-gold + bootstrap CI
                         + 配对 bootstrap（含蒙特卡洛分辨率标记）
                         + 精确 McNemar（二元指标，无地板、无种子）
                         + 逐查询胜/负/平计数
                         + Holm-Bonferroni 多重比较校正
    retrieval.py         共享的 BM25 / dense run 构造、嵌入缓存读取、MRL 前缀 L2 与逐查询指标
    hybrid_mrl1024.py    H 的冻结 A/E/H/G run、query bootstrap、四族配对检验与聚合报告认证
    rerank.py            rerank 窗口语义、覆盖检查、输入指纹与四个配对检验族
    qgen.py              TiDB 分层抽样、双表面 QG、严格解析与双阶段验证
    tidb_runs.py         冻结四系统 run、确定性排序、RRF 与 rerank 应用语义
    pool.py              pair-level pooling、rank-blinded 判断顺序与 multi-gold qrels
    tidb_quality.py      TiDB pair-aware 指标、95% CI、配对 bootstrap/Holm 与分层报告
    metrics_gen.py       provider-free 生成指标合同：sentence BLEU/ROUGE-L + lazy BERTScore
    quest_eval.py        provider-free RAGQuestEval answer scoring 与显式 denominator
  quality_gate.py        严格 JUnit 解析与 clean quality gate
  providers/
    http.py               One Hub JSON transport：显式 UA、长退避、Retry-After、隐私化错误
    embedding.py          共享嵌入客户端：配置 / 批缓存 / 模型与 prompt sidecar
    rerank.py             Qwen3 rerank 请求与完整响应校验
    chat.py               LLM_* chat completion：高推理强度、JSON 与截断校验
    cache.py              忽略的 append-only 配对分数缓存 + provenance sidecar
  retrieval/
    fusion.py            RRF 融合（可加权、可指定融合深度）
    online.py            在线编排：两臂各取 100 → 本地精确 RRF → 请求 100 / 应用 50 的重排
    adapters.py          provider 与在线 Protocol 的唯一接缝（instruction 必须显式传入）
  store/
    base.py              与厂商无关的 ChunkRecord / ArmHit / Passage 与 VectorStore Protocol
    milvus.py            惰性导入的 pymilvus 适配器：固定 schema、完整行 upsert、alias 切换
  lexical/
    sparse.py            客户端 char-bigram BM25 稀疏向量（与本地 BM25 内积等价）
  ingest.py              manifest 校验、稳定身份、scope 隔离与文档级 delta
scripts/
  build_eval_corpus.py   由 raw/split_merged.json 生成 5,681 篇语料与 qrels
  run_lexical_sweep.py   重新生成上方三张检索表（全部 2,394 条，分层）
  power_analysis.py      功效分析：评测集需要多少条查询
  corpus_stats.py        重新生成上方语料与分块统计
  verify_embedding_api.py  嵌入供应商行为探针：L2 归一化 / 噪声底 / 切片等价 / 排序等价
  probe_mrl_quality.py   MRL 维度-质量曲线（逐维方差 + R@1 曲线 + 配对检验）
  embed_queries.py       补齐多证据 query 嵌入缓存（可预估 token 与成本）
  compare_dense_bm25.py  dense 与 BM25 逐查询对齐：列联表 / RRF / arity 分层 / 配对检验（不联网）
  evaluate_rerank.py     top-100 断点续评分 + top-50/100 离线重排与分层配对检验
  smoke_milvus_lite.py   Milvus Lite 在 Windows + Python 3.13 的冒烟测试
  verify_milvus_store.py 正式 store 的 Milvus Lite 集成校验（schema / 两臂 / alias / 重开）
  build_index.py         manifest → chunk → 稀疏重建 → shadow collection → alias 切换（默认干跑）
  query_index.py         在线组合根：嵌入 + 词表 + Milvus alias + 重排，打印排序与各阶段耗时
  build_tidb_queries.py  分层抽样 → 生成 direct/paraphrase → 单独验证 pass → 发布 query pair
  build_tidb_pool.py     冻结 BM25/dense/RRF/rerank runs 并构造 pair-level 判断池
  build_tidb_qrels.py    rank-blinded 判断缓存；只有 --finalize 发布 qrels/report
  evaluate_tidb_retrieval.py 只读冻结 runs/qrels，离线生成聚合质量报告
  sync_tidb_eval_docs.py 只同步 TiDB 聚合状态/质量区域（支持 --check）
  sync_m9a_docs.py       从 tracked aggregate-only evidence 同步生成指标合同与 Table 8 区域
  sync_quality_gate_docs.py 从独立 JUnit 报告同步 README 测试数（支持 --check）
  evaluate_h_hybrid_mrl1024.py 只读完整 4096 cache，离线重建 A/E/H/G 并生成聚合报告
  sync_h_hybrid_mrl1024_docs.py 重算认证 H 报告并同步 tracked 文档（支持 --check）
<!-- BEGIN QUALITY-GATE-STATUS -->
tests/                   1,030 个单元测试
```

质量门禁（本行仅由 `pytest.xml` 生成）：`pytest` 1,030 passed。`ruff check` / `ruff format --check` / `mypy --strict` 是独立的提交前门禁，不由本报告认证。
<!-- END QUALITY-GATE-STATUS -->

README 中每一个数字都由上述脚本生成，没有手工誊写。这不是洁癖：早期原型用固定 1.15 字符/token 估算，得出的 chunk 数与最终实现相差 2.4 倍；而最初那次 BM25 饱和实验是一次性脚本跑的、从未提交，导致 README 里的核心结论一度**无法被任何人复现**。

### 关于 `eval/crud.py`

官方子集只索引 500 篇文档，而 `raw/split_merged.json` 里本就有 7,661 条记录。重建语料时哪些字段可以进检索池，是一个**正确性**问题而非计数问题：`hallu_modified` 同时携带 `hallucinatedContinuation` 与 `hallucinatedMod` —— 为幻觉检测任务**刻意编造**的文本。用 `startswith('news')` 这类前缀匹配去收割字段会把它们放进检索池，于是语料里混入了断言虚假事实的、读起来毫无破绽的中文新闻；任何检索到它的生成器都会被一份它此刻完全有理由去违背的标准答案判为错误。

因此字段用**显式白名单**而非前缀匹配，并在测试中把"编造文本绝不可检索"这一条单独钉死。

### 关于 `io_utils.py`

开发机上 Python 默认编码为 cp936：

```python
>>> sys.stdout.encoding, locale.getpreferredencoding(), sys.flags.utf8_mode
('gbk', 'cp936', 0)
>>> open('tidb-rag-curated/README.md').read()
UnicodeDecodeError: 'gbk' codec can't decode byte 0xad in position 9
```

失败是**非对称**的：`sys.getfilesystemencoding()` 已是 utf-8，所以路径正常、只有文件内容报错，很容易误诊。且 Linux CI 默认 UTF-8 会掩盖该问题。
因此全项目文件读写只经由一个模块，`ruff` 的 `PLW1514` 规则禁止其它位置直接调用 `open()`，CI 保留 `windows-latest` 分支专门捕捉这类 bug。

### 关于指标自研

`recall_at_k` 在多证据文档场景下的定义各家评估库并不一致（"命中比例" vs "是否命中任一"），而 2docs / 3docs 任务恰好有 2 和 3 个证据文档，该定义直接决定主指标数值。因此指标自研并配单元测试，同时提供 `all_gold_at_k` 作为多文档任务的诚实口径。

统计口径：n=500 时约 3–4 个百分点才是可辨差异下限，故所有结果报 bootstrap 置信区间；ablation 各臂之间，连续指标用**配对 bootstrap 检验**（各臂共享同一查询集，配对可大幅降低方差），二元指标用**精确 McNemar 检验** —— 它没有蒙特卡洛分辨率下界、不需要随机种子、换台机器还是同一个数。单证据查询的 R@1 是二元的；多证据查询的 R@1 会取 1/2、1/3、2/3 等分数，所以改用 `hit@1`（任一 gold 排第一）与 `ALL-gold@10`（整套 gold 都在 top-10）做精确检验。dense vs BM25 那一格就是由它判定的。所有比较同时报逐查询**胜/负/平计数**：同样是 +2pp，由 86 胜 69 负得来和由 16 胜 4 负得来，是两个完全不同的系统，而 delta 列一模一样。

---

## 开发

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

uv run pytest          # 单元测试
uv run ruff check src tests scripts  # 含 PLW1514：禁止裸 open()
uv run mypy            # strict
```

### 复现本 README 的全部数字

```bash
# 1. 由原始 split_merged.json 重建评测语料（5,681 篇 + 2,394 条 qrels）
uv run python scripts/build_eval_corpus.py

# 2. 检索表：饱和曲线、analyzer 对比、多证据难度
uv pip install jieba                       # 仅 jieba 对照行需要
uv run python scripts/run_lexical_sweep.py

# 3. 语料与分块统计
uv run python scripts/corpus_stats.py

# 4. 功效分析：评测集够不够大
uv run python scripts/power_analysis.py
```

以上四步**不需要 API key**，也不下载任何东西 —— 全部六个 CRUD 任务本来就在你已有的
`raw/split_merged.json` 里。输出目录 `crud-rag-subset/eval-expanded/` 是派生数据，已 gitignore。

向量部分需要一个 OpenAI 兼容的嵌入端点，在 `.env` 里配置
`Embedding_BASE_URL` / `Embedding_API_KEY` / `Embedding_MODEL_NAME`：

```bash
# 5. 供应商行为探针（约 10 次 API 调用）：归一化 / 噪声底 / 切片等价 / 排序等价
uv run python scripts/verify_embedding_api.py

# 6. MRL 维度-质量曲线。全量约 2M tokens（≈ ¥0.6），只嵌入一次；
#    7 个维度档全部是该次嵌入的客户端切片，重跑读缓存、零成本。
uv run python scripts/probe_mrl_quality.py --docs 5681 --queries 800

# 7. dense 与 BM25 逐查询对齐：头对头 / RRF / 按实际 gold 数分层
#    第 6 步先写下 800 条 1doc query；补齐另外 1,594 条后，全程离线复算。
uv run python scripts/embed_queries.py --dry-run  # 先看缺口、token 与费用
uv run python scripts/embed_queries.py            # 有缺口时才调用 API
uv run python scripts/compare_dense_bm25.py --resamples 100000

# 8. 冻结 dense-4096 hybrid 的前 100 个候选，统一打分一次；top-50/100 共用缓存。
#    scoring 是付费且可断点续跑；analysis 不读 .env、不联网，并拒绝不完整或混入异实验的缓存。
uv run python scripts/evaluate_rerank.py                         # dry-run：先看缺口与 token 估计
uv run python scripts/evaluate_rerank.py --score --max-queries 1 # 付费 smoke
uv run python scripts/evaluate_rerank.py --score                 # 仅补齐缺失 query
uv run python scripts/evaluate_rerank.py --analyze --resamples 100000

# 9. TiDB 合成评测集。无 flag 都是离线计划/状态检查；付费步骤必须显式开启。
uv run python scripts/build_tidb_queries.py                       # 计划 QG，不调用 chat
uv run python scripts/build_tidb_queries.py --generate            # 付费生成 + 验证，可续跑
uv run python scripts/build_tidb_pool.py                           # 离线状态/构池（缓存须完整）
uv run python scripts/build_tidb_pool.py --embed --rerank          # 仅补齐付费缓存
uv run python scripts/build_tidb_qrels.py                          # 只检查，不发布
uv run python scripts/build_tidb_qrels.py --judge                  # 只补判断 cache，不发布
uv run python scripts/build_tidb_qrels.py --finalize               # 离线显式发布 qrels/report
uv run python scripts/evaluate_tidb_retrieval.py --resamples 10000 --seed 0  # 严格离线质量

# 10. 文档同步分三条独立边界：M9a evidence、TiDB 聚合报告、JUnit 质量门禁。
#     下面的 pytest.xml 只供质量门禁同步器读取；TiDB 同步器不依赖它。
uv run pytest --junitxml=.pytest_tmp/pytest.xml
uv run python scripts/sync_m9a_docs.py
uv run python scripts/sync_m9a_docs.py --check
uv run python scripts/sync_tidb_eval_docs.py
uv run python scripts/sync_tidb_eval_docs.py --check
uv run python scripts/sync_quality_gate_docs.py --junit .pytest_tmp/pytest.xml
uv run python scripts/sync_quality_gate_docs.py --junit .pytest_tmp/pytest.xml --check

# 11. 独立 H（A+dense-1024）基线：只读完整本地 cache，严格离线；缺失/漂移即失败。
uv run python scripts/evaluate_h_hybrid_mrl1024.py --resamples 10000 --seed 0
uv run python scripts/sync_h_hybrid_mrl1024_docs.py
uv run python scripts/sync_h_hybrid_mrl1024_docs.py --check
```

第 6 步把 4096 维文档向量缓存到 `eval-expanded/emb_cache_4096.jsonl`（当前约 523 MB，
十进制 JSONL，已 gitignore）。query 因 Qwen3 的非对称前缀而使用独立缓存文件；补齐全部 2,394 条后
当前约 220 MB。两个缓存都按批追加写入、中断可续；同目录 sidecar 记录 model 与 prompt，防止换模型后
静默复用旧向量。

第 8 步的 239,400 个 rerank 配对分数同样是**语料派生物**，只保存在
`eval-expanded/rerank_scores_top100.jsonl`，不得提交。sidecar 绑定 endpoint、model、instruction、候选 run
指纹与实际 query/document 文本指纹；评分只在完整响应校验通过后按 query 追加，续跑会跳过完整 query，
分析阶段要求全部 2,394 × 100 个预期配对齐全且没有额外配对。

---

## 数据来源与许可

本仓库**不提交任何语料原文**，仅提供可复现的下载与抽取脚本。完整说明见 **[DATA_LICENSE.md](DATA_LICENSE.md)**。

要点（TiDB/CRUD-RAG 授权状态按 2026-08-31 的审计记录整理；详见 DATA_LICENSE.md）：

- **TiDB 中文文档**：[pingcap/docs-cn](https://github.com/pingcap/docs-cn) @ `26f202bc`，**CC BY-SA 3.0**。
  注意 ShareAlike 是有牙齿的：本项目切出的 chunk 构成该许可证定义的 **Adaptation**，
  分发它们需署名并以兼容许可证发布 —— 这正是派生产物也不进仓库的原因。
- **CRUD-RAG**：[IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)，
  ⚠️ **仓库根目录没有 LICENSE 文件**，GitHub API 的 `license` 字段返回 `null`
  —— README 里的 Apache-2.0 徽章只是一张 shields.io 图片，徽章不是授权。
  其 `split_merged.json` 内含约 8 万篇中文新闻正文，上游未声明出处与再分发许可，
  故本项目不重分发其中任何一条，仅作非商业学术评测使用。

代码部分以 Apache-2.0 授权；语料及其派生物不适用该许可证。
