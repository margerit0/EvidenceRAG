# zhrag — 企业级中文 RAG 与可复现评估框架

面向中文语料的检索增强生成系统。项目的核心不是"又一个 RAG demo"，而是**一套能真正区分配置优劣的评估框架**，以及在此之上用实测数据驱动的每一个工程决策。

> **状态**：评估与语料层已完成并有测试覆盖；检索栈（向量库 / 混合检索 / rerank）选型进行中。
> 下方所有数字均为本仓库脚本在真实语料上跑出的结果，非引用。

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

### 中文 BLEU / ROUGE 的默认配置会产出"看起来合理的垃圾"

生成侧指标在中文上极易配错，且错得**没有任何报错**。用一条真实 CRUD-RAG 样本测量（`ref` = 标准答案，`good` = 正确改写，`bad` = 完全不相关的另一段新闻）：

| 指标配置 | good | bad | 判定 |
|---|---:|---:|---|
| BLEU `tokenize=13a`（sacrebleu **默认**） | 31.95 | 0.00 | 低估约 24 分 |
| BLEU `tokenize=zh` | **55.98** | 1.06 | ✅ 正确 |
| BLEU `tokenize=char` | 67.67 | 0.65 | 可用 |
| ROUGE-L `rouge_score` 直接跑中文 | **1.0000** | 0.0000 | ⚠️ **退化，不可用** |
| ROUGE-L `rouge-chinese` + jieba 分词 | **0.8276** | 0.1154 | ✅ 正确 |

`rouge_score` 对一个**改写句**给出满分 1.0——它按空格切词，而中文没有空格，整句坍缩成一个 token。数字很漂亮，但完全没有意义。

因此本项目固定：**BLEU 用 sacrebleu `tokenize='zh'`；ROUGE-L 用 `rouge-chinese`（jieba 词级）；BERTScore 必须指定中文模型**（不能用默认的英文 RoBERTa）。

> 注意这里的反直觉之处：**jieba 在 BM25 检索上输给字符 bigram，但在 ROUGE 评估上是正确选择。** 分词方案要按用途分别决定，不能一刀切。

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
| 两阶段 target=400 | **1,725** | 169 | 375 | 747 | 10,914 | **5.3%** | **0** |
| 两阶段 target=512 | 1,399 | 204 | 480 | 913 | 16,111 | 3.4% | 0 |
| 两阶段 target=700 | 1,084 | 228 | 640 | 1,164 | 16,111 | 3.0% | 0 |

近**三分之二**的朴素切块小于 100 token（一个标题加一句话），embedding 后基本是噪声。
两阶段策略 = 按标题层级切分 → 合并过小相邻段 + 按段落边界拆分超长段。

代码块与表格在切分前被占位符保护、切分后还原，**实测三种 target 下代码块截断数均为 0**。这对本语料至关重要：检索强依赖 `tiup cluster deploy` 这类标识符的精确匹配。

选定 target=400：共 1,725 个 chunk、793,449 tokens。其中 273 个（15.8%）超过 `hard_max`，因为单个代码块或表格本身就超预算——这是有意为之，Qwen3-Embedding-8B 的 32k 上下文放得下，而把表格与表头拆开的代价更大。

向量存储量：**4096 维 float32 仅 28.3 MB**，MRL 截断到 1024 维只要 7.1 MB。规模完全不构成向量库选型的约束。

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

---

## 已实现

```
src/zhrag/
  io_utils.py            单一 UTF-8 文件 I/O 出入口
  tokens.py              按 Qwen3 tokenizer 实测标定的中英双分量 token 估算
  lexical/
    analyzers.py         字符 n-gram / jieba（可选）/ 并集
    bm25.py              Okapi BM25，可插拔 analyzer
  chunking/
    markdown.py          标题感知的两阶段分块，保护代码块与表格
  eval/
    crud.py              CRUD-RAG 语料重建：去重文档池 + 三个任务的 qrels
    metrics.py           R@k / MRR / nDCG / ALL-gold + bootstrap CI
                         + 配对检验 + Holm-Bonferroni 多重比较校正
scripts/
  build_eval_corpus.py   由 raw/split_merged.json 生成 5,681 篇语料与 qrels
  run_lexical_sweep.py   重新生成上方三张检索表（全部 2,394 条，分层）
  power_analysis.py      功效分析：评测集需要多少条查询
  corpus_stats.py        重新生成上方语料与分块统计
  smoke_milvus_lite.py   Milvus Lite 在 Windows + Python 3.13 的冒烟测试
tests/                   135 个单元测试
```

质量门禁：`pytest` 135 passed · `ruff check` 全通过 · `mypy --strict` 无告警。

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

统计口径：n=500 时约 3–4 个百分点才是可辨差异下限，故所有结果报 bootstrap 置信区间，ablation 各臂之间用**配对 bootstrap 检验**（各臂共享同一查询集，配对可大幅降低方差）。

---

## 开发

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

uv run pytest          # 单元测试
uv run ruff check .    # 含 PLW1514：禁止裸 open()
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

第 1 步不下载任何东西 —— 全部六个 CRUD 任务本来就在你已有的 `raw/split_merged.json` 里。
输出目录 `crud-rag-subset/eval-expanded/` 是派生数据，已 gitignore。

---

## 数据来源与许可

本仓库**不提交任何语料原文**，仅提供可复现的下载与抽取脚本。完整说明见 **[DATA_LICENSE.md](DATA_LICENSE.md)**。

要点（均于 2026-08-18 经 GitHub API 实测确认）：

- **TiDB 中文文档**：[pingcap/docs-cn](https://github.com/pingcap/docs-cn) @ `26f202bc`，**CC BY-SA 3.0**。
  注意 ShareAlike 是有牙齿的：本项目切出的 chunk 构成该许可证定义的 **Adaptation**，
  分发它们需署名并以兼容许可证发布 —— 这正是派生产物也不进仓库的原因。
- **CRUD-RAG**：[IAAR-Shanghai/CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)，
  ⚠️ **仓库根目录没有 LICENSE 文件**，GitHub API 的 `license` 字段返回 `null`
  —— README 里的 Apache-2.0 徽章只是一张 shields.io 图片，徽章不是授权。
  其 `split_merged.json` 内含约 8 万篇中文新闻正文，上游未声明出处与再分发许可，
  故本项目不重分发其中任何一条，仅作非商业学术评测使用。

代码部分以 Apache-2.0 授权；语料及其派生物不适用该许可证。
