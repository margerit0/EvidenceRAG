# 中文企业级 RAG 系统 —— 最终技术方案

> 定稿日期 2026-08-17。以下所有版本号、价格、限制均为该日实测值，会漂移。标注 ⚠️ 的是研究阶段未能闭环、需要你自己动手验证的点。

> **数字勘误说明**：本报告综合时使用的分块统计来自早期原型（固定 1.15 字符/token）。
> 生产实现按 Qwen3 tokenizer 实测标定（中文 1.57 / 英文 4.49），已全部更正为
> `scripts/corpus_stats.py` 的输出：朴素按标题切分的欠长块占比是 **63.5%**（非 31.4%），
> target=400 得 **1,725** 块（非 4,191），4096 维存储 **28.3 MB**（非 115 MB）。
> 结论方向不变，且更强。

> **验证状态更新**：以下三项阻塞验证已于 2026-08-18 在本机 Windows 11 + Python 3.13.5 完成。旧 checklist 已保留为审计轨迹，状态在 §13 更新。

---


**自研薄检索层（Protocol + YAML config）+ Milvus 三级部署（Lite on Windows → Standalone in WSL2 → Zilliz Cloud Free 公网）+ Qwen3-Embedding-8B（客户端拼 instruct 前缀、MRL 降至 1024 维）+ Qwen3-Reranker-4B（top-50）+ 客户端 char-bigram BM25 稀疏向量做词法臂 + 服务端 RRF 融合，评估侧以已建成的 R@1/MRR@10/nDCG@10 为主指标、移植 CRUD-RAG 的 ~150 行生成指标为辅，全部跑在 GitHub Actions 双 OS CI 上。**

---

## 2. 技术栈选型表

| 层次 | 选型 | 理由 | 备选（及改选条件） |
|---|---|---|---|
| **向量数据库** | **Milvus**：本地 `milvus-lite 3.2.0`（纯 Python，支持 Windows）→ WSL2 Ubuntu-24.04 Standalone → Zilliz Cloud Free（5 GB / 5 collections / $0） | 4096 维无压力（上限 32,768）；`hybrid_search` + `RRFRanker` 服务端融合；collection alias 支持原子换索引；同一份 pymilvus 代码跑三种部署 | **Qdrant v1.19.0**（原生 `qdrant-x86_64-pc-windows-msvc.zip`，无需 Docker/WSL2）。改选条件：你决定公网 demo 自己托管 VPS / HF Space，或彻底拒绝 Docker。**pgvector 已被排除**，见 §3 |
| **检索框架** | **自研薄层**：`Retriever` / `Fusion` / `Reranker` / `Chunker` 四个 Protocol + registry + pydantic-settings YAML | `langchain-community` 已于 2026-05-22 正式 sunset，检索半壁江山被弃；llama-index-core 近 12 周提交量同比 -70%、8 周无发版。你现有 `BM25.search(query, k) -> list[tuple[str, float]]` **本身就是 Retriever Protocol** | **Haystack 3.0**（YAML 原生序列化 + `MultiRetriever` RRF）。改选条件：消融维度超过 6 个，且确实需要框架级配置编排。可另加 LangChain/LlamaIndex 各 ~50 行 adapter 作为**消融表的一行**（用于框架适配对照） |
| **embedding** | **Qwen/Qwen3-Embedding-8B** @ SiliconFlow，`dimensions=1024`（先按 4096 跑一遍作天花板） | 已持有；C-MTEB Retrieval 78.21；一次性索引成本可忽略（全量 4.88M tokens ≈ ¥1.37） | Qwen3-Embedding-4B（¥0.14/M，2560 维，C-MTEB 77.03）。改选条件：MRL 消融显示 8B 相对 4B 的 +1.18 分在你的语料上不显著 |
| **rerank** | **Qwen/Qwen3-Reranker-4B** @ SiliconFlow（用 `instruction` 字段） | 8B 只比 4B 高 1.51 CMTEB-R，但 4B 的 FollowIR 14.84 vs 8B 8.05——你要传中文领域自定义 instruction，指令遵循能力才是决定项；价格减半 | Qwen3-Reranker-8B 作为消融表最后一行跑一次。**不要用 0.6B**：CMTEB-R 71.31，输给 bge-reranker-v2-m3 的 72.16 |
| **LLM（生成）** | Qwen 系（你已有 key） | 与 embedding/rerank 同族，叙事一致 | — |
| **LLM（评判）** | **DeepSeek-V3 类 或 Kimi**，必须≠生成模型 | 自偏好偏差已被因果证实（GPT-4 自评胜率 +10%，Claude-v1 +25%；Panickssery et al. 2404.13076 证明自我识别能力与自偏好强度线性相关）。deepeval 内置 `deepseek_model.py` / `kimi_model.py` | 任一非 Qwen 家族强中文模型 |
| **分块** | 已建成的两阶段：header split → 掩码 code fence/table → target=400 合并小块/拆大块 | 实测 n=1,725，p50 375，p90 747，欠长块 5.3%，代码块破损 0 | 必须补跑 256/400/800 sweep 出曲线（验证分块目标的选择依据） |
| **检索管线** | dense(4096/1024, HNSW, COSINE) + sparse(char-bigram BM25, **IP**) 双字段 → 服务端 `hybrid_search` + `RRFRanker` → 客户端 Qwen3-Reranker-4B 重排 top-50 | 三段式，每段可单独消融 | 融合方式备选 WeightedRanker / Qdrant 的 dbsf，作为消融表一列 |
| **词法检索** | **客户端算 char-bigram BM25 权重，作为 SPARSE_FLOAT_VECTOR 推给 DB** | Milvus 内置 `chinese` analyzer 就是 jieba，且默认 `mode="search"` = `cut_for_search`——正是你实测最差的 72.8%，比 char bigram 的 75.8% 低 3 分。**开服务端分词器会让系统变差**。另外可绕开 Milvus Lite「BM25 IDF 按 segment 局部统计」的坑 | 无（这是本项目最有说服力的设计决策之一） |
| **服务层** | FastAPI + httpx（异步）+ tenacity（429 指数退避） | 三个依赖，全部薄，不侵入检索层 | — |
| **前端** | FastAPI 挂一个单文件静态 HTML（检索框 + 结果卡片 + 命中 chunk 高亮 + 各阶段耗时条） | 界面优先展示**阶段耗时**和**检索证据**，便于检查系统行为 | Gradio / Streamlit（若你想 5 分钟部署到 HF Space）。改选条件：你决定公网 demo 放 HF Space 而非 Zilliz |
| **评估** | 检索侧：**已建成**（R@k / MRR / nDCG / ALL-gold / bootstrap CI / paired bootstrap）。生成侧：**移植** CRUD-RAG `src/metric/` 约 150 行进自己的包 | 绝不 `pip install` CRUD_RAG（它 pin 了 `llama_index==0.9.32` / `langchain==0.1.4` / `pymilvus==2.3.3`，在 3.13 上装不上） | RAGAS 只作为「我知道这个框架」的一行说明。**不要当主力**：最后一次 commit 2026-02-24，559 open issues，而竞品当天都在发版 |
| **可观测性** | **Phoenix**（`pip install arize-phoenix && phoenix serve`，SQLite 后端，**零 Docker**） | 你机器没 Docker；Phoenix 是唯一 pip 即用的 tracing UI；requires_python `>=3.10,<3.15` 覆盖 3.13.5 | Langfuse（33.2k stars，国内认知度最高）但需 6 个容器；改选条件：你已经在 WSL2 里装了 Docker。⚠️ Phoenix 是 Elastic-2.0，非 OSI 许可 |
| **工程化/CI** | GitHub Actions：`ubuntu-latest`（全量快子集）+ `windows-latest`（**故意不设 PYTHONUTF8/PYTHONIOENCODING**）；ruff（含 PLW1514 禁裸 `open()`）+ mypy + pytest | Linux runner 是 UTF-8，会掩盖你本机 cp936 的裸 `open()` 崩溃；双系统 CI 验证不同默认编码下的行为 | — |

---

## 3. 向量数据库为什么选它（正面回答）

### 3.1 pgvector：**直接出局，且理由反直觉**

pgvector 0.8.6（2026-07-29）的索引维度上限至今是：`vector` **2,000**、`halfvec` **4,000**、`bit` 64,000、`sparsevec` 1,000 非零元。**4096 > 4000**——网上人人复述的「维度超了就转 halfvec」这条建议，恰好在 Qwen3-Embedding-8B 的宽度上**差 96 维失效**。`vector` 类型本身能存 16,000 维，但**存得下不等于索引得了**。

pgvector README 的 FAQ 只给三条出路：half-precision（≤4000，还是不够）、binary quantization（`bit`，召回大幅损失）、切子向量/降维。也就是说，走 Postgres 只能放弃原生 4096。

数据库维度限制应以对应版本的官方文档核验，并区分可存储维度与可索引维度。

### 3.2 Top-3 正面对比

| 维度 | Milvus | Qdrant | TiDB Vector |
|---|---|---|---|
| **4096 维原生支持** | ✅ 上限 32,768 | ✅ 上限 65,535 | ✅ 上限 16,383 |
| **混合检索** | ✅ 服务端 `hybrid_search([AnnSearchRequest...], ranker=RRFRanker())`，Lite 本地也支持 | ✅ `prefetch` + `rrf`/`dbsf`（v1.10+），RRF 的 k 可调（v1.16+），加权 RRF（v1.17+）——融合 API 比 Milvus 更干净 | ❌ **融合在 pytidb 客户端做，不是一条 SQL**。官方文档自己说「用 pytidb 完全可选，你也可以直接写 SQL 然后自带 rerank 模型」 |
| **中文全文检索** | ✅ 内置 `chinese` analyzer = jieba + cnalphanumonly；另有独立 `jieba`/ICU/Lindera tokenizer。**但你不该用**（见下） | ❌ 无 jieba，只有 `multilingual` tokenizer + BM25-as-sparse-vector + `idf` modifier | ⚠️ `WITH PARSER MULTILINGUAL` 支持中日韩，**但只在 TiDB Cloud Starter/Essential 的两个 region（Frankfurt、Singapore）可用，无中国大陆 region**，文档自称「still in the early stages」 |
| **元数据过滤** | ✅ 完整 Milvus 表达式：`text_match()`、JSON path、`array_contains`、`$meta[...]` | ✅ payload filter，每个 prefetch 可独立带 filter | ✅ 就是 SQL WHERE（这点最强） |
| **Windows / WSL2** | ✅ Lite 3.2.0 是纯 Python `py3-none-any` wheel，官方 Requirements 明写 Windows；faiss-cpu 1.15.0 有 cp313 win_amd64 wheel。⚠️ **Windows+3.13 组合不在上游 CI 矩阵内** | ✅✅ 官方发布 `qdrant-x86_64-pc-windows-msvc.zip`，解压即跑，**安装路径最干净** | ☁️ 纯云，本地无关 |
| **免费额度长期存活** | ✅ Zilliz Cloud Free：5 GB / 2.5M vCU/月 / 5 collections。官方页面**没有写 Free 集群闲置 N 天自动删除/挂起**；但 Terms 保留服务终止权，且终止后不承诺保留数据。不能承诺永久存活，必须保留可重建 artifact | ⚠️ 原生 Windows 运行路径干净；免费云集群的闲置/删除策略必须按当前官方条款自行核对，不能把未经一手来源确认的“1 周挂起、4 周删除”写成定论 | ⚠️ Starter：5 GiB 行存 + 5 GiB 列存 + 50M RU/月，免信用卡。配额耗尽会影响新请求，长期闲置/归档条款仍需单独核对 |

### 3.3 关键设计决策：**不要用数据库的中文分词器**

Milvus 内置 `chinese` analyzer 等价于 `{"tokenizer": "jieba", "filter": ["cnalphanumonly"]}`，而简写 `{"tokenizer": "jieba"}` 的默认是 `mode="search"`，即 jieba 的 `cut_for_search`——你实测 **72.8% R@1**，比 char bigram 的 **75.8%** 低整整 3 分。**开这个开关会让系统变差。**

正确做法：进程内算 char-bigram BM25 权重 → 作为 `SPARSE_FLOAT_VECTOR` 写入 → `SPARSE_INVERTED_INDEX` + `metric_type="IP"`。数据库只负责 ANN + 服务端融合。

三个副产品：
1. 绕开 Milvus Lite「BM25 IDF 按 segment 局部统计」的限制，否则 Lite 的分数与 Standalone/Zilliz 对不上，会**悄悄污染你上报的指标**；
2. 数据库选型变得基本可逆（只委托了 ANN + fusion）；
3. 选型依据：**「实测数据库内置 jieba analyzer 与客户端 char-bigram BM25 臂，后者高 3.0 pt R@1，遂将词法臂移出数据库、以预计算稀疏向量喂入」**——这是量化的设计决策，不是框架默认值。

### 3.4 对「用 TiDB 向量检索服务 TiDB 文档」这个叙事的直接裁决

**做，但只做成 adapter 接口后面的第二后端，绝不做主存储。**

理由是 2026 年的事实不支持它承重：
- 向量检索在 2026-08 的官方文档里**仍标注 beta**（"might be changed without prior notice"）；
- 向量索引**强制要求 TiFlash 副本**；
- 全文检索只在两个 region 可用，**无中国大陆节点**——这直接拆掉「企业级中文部署」的叙事支点；
- RRF/加权融合在 **pytidb 客户端**完成，不是服务端 SQL；
- **pytidb 版本号是 0.0.14，且无 Python 3.13 classifier**。

把头条指标压在一个 0.0.x SDK 后面的 beta 功能上，会增加尚未验证的部署故障风险。

**正确的吃法**：定义 `VectorStore` Protocol，`MilvusStore` 为主，`TiDBStore` 为第二实现，然后在 README 写一节 **「为什么 TiDB 的文档没有跑在 TiDB 上」**，列出 beta 状态、TiFlash 依赖、两 region 全文检索限制、客户端融合、SDK 版本号。

**这一节比那个噱头本身更有价值**——它证明你拿一手文档验证了厂商宣传并做了判断。可插拔后端接口本身就是整个项目最强的工程信号。

---

## 4. Qwen3 两个模型的正确用法

### 4.1 Instruction 前缀（最多人踩错的地方）

权威来源是 `config_sentence_transformers.json`，字节精确：

```
"query":    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
"document": ""
```

**`Query:` 后面没有空格。** 注意模型卡自相矛盾：Python helper 是 `f'Instruct: {task}\nQuery:{query}'`（无空格），同一页的 TEI curl 示例却写 `Query: What is...`（有空格）。以 JSON config 为准（SentenceTransformers 实际应用的就是它）。

**非对称**：document 侧 prompt 是空字符串，**文档不加任何前缀**。两边都加前缀会静默掉几个点的 R@1 且不报错。

本项目直接抄这段：

```python
QUERY_PROMPT = ("Instruct: Given a Chinese technical question about TiDB documentation, "
                "retrieve the most relevant documentation passage\nQuery:")
query_text = QUERY_PROMPT + q      # documents: 原文，无前缀
```

**指令写英文**，即使语料是中文。Qwen 明说：「In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English.」不加 instruct 掉 1%–5%。（⚠️ 这是通用建议，不是在 TiDB 文档上的实测——顺手做个中英 instruction 的 A/B，两次跑，又是一行诚实消融。）

### 4.2 pooling / normalize（自建推理时才需要）

`pooling_mode_lasttoken: true`，`padding_side` 必须是 `'left'`，最后必过 `F.normalize(p=2, dim=1)`，`similarity_fn_name = cosine`。右 padding + 朴素 `hidden[:, -1]` 会拿到 pad token 的向量——**不报错，只是全错**。

### 4.3 MRL 策略

MRL 就是切片 + L2 重归一化（`modules.json` 是 `[Transformer, Pooling(lasttoken), Normalize]`，`truncate_dim` 在 Normalize 之前切）。

**没有任何官方来源发布过 2048/1024/512 的质量损失表**——arXiv v3 全文里 "MRL" 只出现两次，都在 Table 1 的表头注释里，"Matryoshka" 出现 0 次。任何引用「1024 维只掉 1%」的人引的是 BGE-M3 或 OpenAI，不是 Qwen。

**这是你最高价值的原创消融**：4096 / 2048 / 1024 / 512 四档，跑 5,681 文档的 R@1 / MRR@10。存储侧 1,725 chunks 从 28.3 MB → 7.1 MB（-75%）。

先做一个 5 分钟验证：同一段文本分别请求 `dimensions=4096` 和 `dimensions=1024`，检查 1024 维向量是否等于 4096 维前 1024 分量的 L2 重归一化。**若是，你可以只存一份 4096 维向量、客户端任意截断**，不必重嵌入。

SiliconFlow 的 `dimensions` 是离散白名单，8B 支持 `[64,128,256,512,768,1024,1536,2048,2560,4096]`。

### 4.4 Rerank 用法

Reranker **不是传统 cross-encoder**，是因果 LM 在 yes/no logits 上打分：`score = exp(logsoftmax([false_logit, true_logit])[1])`。线上格式（注意尖括号，与 embedder 的裸 `Instruct:`/`Query:` **不同**）：

```
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
body   = '<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}'
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

截断是 `truncation='longest_first'`，`max_length` 已扣掉 prefix+suffix token 数——**被砍的永远是文档，不是脚手架**。

**候选深度必须 > 10。** Qwen 自己的评测基准是 **top-100**。你实测的 ALL-gold@10 是 1doc 99.7% / 2docs 85.7% / **3docs 65.0%**——top-10 重排窗口会把 3docs 任务的天花板锁死在 65%。跑 **top-50 和 top-100 两档**作为消融。

### 4.5 Provider、价格、限流

| Provider | Embedding-8B | Reranker-4B | 关键特性 |
|---|---|---|---|
| **SiliconFlow（选它）** | ¥0.28/M | ¥0.14/M | 唯一在 `/v1/rerank` 暴露 Qwen3 `instruction` 字段的；Cohere 形状响应（`results[{index, relevance_score}]`，无需客户端排序）；国内原生 |
| DeepInfra（备用） | $0.010/M | $0.025/M | 便宜约 4x，但 **`normalize` 默认 false**，rerank 是 `{queries[], documents[]}` 平行数组、返回裸 `scores[]` 无索引 |
| DashScope/百炼 | ❌ | — | `text-embedding-v4` **不是** Qwen3-Embedding-8B：最大 2048 维、8,192 token。将该端点记录为 Qwen3-Embedding-8B 会造成模型身份与实验说明不一致 |

**限流**：SiliconFlow 是账号级、按模型类别，Embedding RPM 2,000–10,000 / TPM 500K–10M 随 L0–L5 消费等级上升；**Reranker 是平的：RPM 2,000 / TPM 500,000，不随等级涨**。一次 top-100 全量重排约 118M tokens，在 500K TPM 下**至少 4 小时墙钟**。第一天就写指数退避。

**成本结构（重要）**：重排是 embedding 的约 24 倍。全量 embedding 一遍 4.88M tokens ≈ ¥1.37；一次 top-50 重排扫描 2,394 条 QA ≈ 59M tokens ≈ ¥16.4。**必须按 `(query_id, doc_id, model)` 缓存重排分数**，否则消融矩阵会让你反复付钱。

**批处理**：embedding 侧按 32–64 条一批并发 4–8 路；rerank 侧受 TPM 约束，并发 2 路 + tenacity 退避即可。

---

## 5. 两个语料怎么组织

**核心原则：CRUD-RAG 是尺子，TiDB 是产品。尺子上调参，产品上展示，两者数据永不混流。**

| | `crud-rag-subset`（5,681 篇干扰语料） | `tidb-rag-curated`（500 篇 / 1,725 chunks） |
|---|---|---|
| **角色** | **评测集**——所有数字来源 | **部署集**——公网 demo 与工程化能力展示 |
| **有无 gold label** | 有（`evidence_document_id`，免费） | 无 |
| **在它上面调什么** | 检索配置：analyzer、fusion 权重、rerank 深度、MRL 维度 | **什么都不调** |
| **在它上面测什么** | R@1 / MRR@10 / nDCG@10 / ALL-gold@10 + paired bootstrap | 端到端 p50/p95/p99 延迟、QPS、增量重建、alias 原子换索引、Bad Case 归因 |

### 怎么避免看起来像过拟合

1. **配置冻结点写死**：README 明确写「所有超参在 CRUD-RAG 5,681 文档评测集上选定，选定后冻结，原样部署到 TiDB 语料，未在 TiDB 上做任何调参」。这句话本身就是方法论声明。
2. **切分 dev/test**：CRUD-RAG 的 2,394 条 QA 里，用 1doc(800) 做 dev 调参，2docs(797)+3docs(797) 只在最终一次上报——避免在同一批 query 上反复选参。
3. **报告置信区间**：你已经实现了 `bootstrap_ci` 和 `paired_bootstrap_test`。**消融表每一行带 95% CI，配置间差异带 p 值**。一个带 CI 的 +1.2pt 提升是工程结论；一个裸的 +1.2pt 是噪声。⚠️ 但先确认 `paired_bootstrap_test` 的实现是对的（重采样的是 query 还是 per-query 分差？单尾还是双尾？~40 个消融格是否做多重比较校正？）——懂 IR 的评审会先查这个再看你的 R@1。
4. **TiDB 侧只报无标注可测的量**：延迟、吞吐、chunk 分布、代码块完整率、增量重建的 `{added, updated, deleted, skipped}` 计数。**不要在 TiDB 上编造检索指标**。
5. **两个语料的许可都不入库**：`.gitignore` 已经正确排除了四个语料目录，保持。补一个 `DATA_LICENSE.md`（见 §12）。

---

## 6. 评估方案

### 6.1 检索侧消融表骨架（主表）

**已实现**：`src/zhrag/eval/metrics.py` 里的 R@k / MRR@k / nDCG@k / ALL-gold@k / bootstrap_ci / paired_bootstrap_test。以下只是把它们排成表。

**列**：

| 列 | 说明 |
|---|---|
| `config_id` | 与 YAML 文件名一一对应 |
| `corpus_size` | 500 / 2000 / 5681（**永远显式，绝不省略**） |
| `retriever` | bm25-bigram / bm25-jieba / dense-4096 / dense-1024 / hybrid-rrf |
| `chunk_target` | 256 / 400 / 800 |
| `rerank` | none / qwen3-4b@50 / qwen3-4b@100 / qwen3-8b@50 |
| `R@1` ±95%CI | **头条指标** |
| `MRR@10` ±95%CI | 头条指标 |
| `nDCG@10` ±95%CI | 头条指标 |
| `ALL-gold@10` | 按 1doc/2docs/3docs 分列（**3docs 的 65.0% 天花板必须同表标注**） |
| `p_vs_baseline` | paired bootstrap vs char-bigram BM25 基线 |
| `index_build_s` | jieba+bigram 并集慢 7 倍这件事要有列承载 |
| `latency_p95_ms` | 端到端 |
| `cost_usd` | 该配置跑一遍的 API 花费 |

**行（最小可交付的 12 行）**：

```
A. bm25-bigram          @5681, chunk=400, rerank=none      ← 基线，已有 75.8 / 0.856
B. bm25-jieba-precise    @5681, chunk=400, rerank=none      ← 已有 73.8
C. bm25-jieba-search     @5681, chunk=400, rerank=none      ← 已有 72.8
D. dense-4096            @5681, chunk=400, rerank=none      ← 新
E. dense-1024 (MRL)      @5681, chunk=400, rerank=none      ← 新，原创消融
F. dense-512  (MRL)      @5681, chunk=400, rerank=none      ← 新，原创消融
G. hybrid-rrf (A+D)      @5681, chunk=400, rerank=none      ← 新
H. hybrid-rrf (A+E)      @5681, chunk=400, rerank=none      ← 新，最可能是最终配置
I. H + rerank-4b@50      @5681, chunk=400                   ← 新
J. H + rerank-4b@100     @5681, chunk=400                   ← 新
K. H + rerank-8b@50      @5681, chunk=400                   ← 新，4B vs 8B
L. H + rerank-4b@50      @5681, chunk=256 / 800             ← chunk sweep
M. bm25-bigram           @500,  chunk=400, rerank=none      ← 饱和对照行，98.6 / 0.993
```

**M 行必须存在，且和 A 行贴在一起。** 这是全篇最重要的排版决策——非技术筛选人看到孤零零的 75.8% 会当成退化。

### 6.2 中文 BLEU / ROUGE / BERTScore 的正确配置

**⚠️ 最高危陷阱（实测）**：`rouge_score` 的默认 tokenizer **会删掉所有中文字符**。实测 `tokenize('近千家展商参与了UDE2023博览会', None)` 返回 `['ude2023']` 一个 token。在你的 `qa_1doc.jsonl` 第 0 行上，好的改写打 **1.0000**、完全无关的句子打 **0.0000**——分数**完全来自共享的拉丁字母/数字子串**。这会产出一张看起来能发表的噪声表。

**复现 CRUD-RAG 的唯一正确路径**（jieba **词级**）：

```python
import jieba, evaluate
f = lambda text: list(jieba.cut(text))          # 精确模式，HMM 开，无用户词典

bleu = evaluate.load('bleu')
r = bleu.compute(predictions=[gen], references=[[ref]], tokenizer=f)

rouge = evaluate.load('rouge')
r = rouge.compute(predictions=[gen], references=[[ref]], tokenizer=f,
                  rouge_types=['rougeL'])       # 注意是 rougeL，不是 rougeLsum
```

sacrebleu 的实测判别力（好改写分 − 无关句分）：`13a` 31.95 / `zh` 54.91 / `char` 67.02 / jieba 预切+`13a` 52.58。**`tokenize='zh'` 是字符级，不是词级**。选 jieba 词级以对齐 CRUD-RAG 表格；可额外报字符级作稳健性检查。

**BERTScore**：

```python
from bert_score import BERTScorer
scorer = BERTScorer(lang='zh', rescale_with_baseline=True, batch_size=64)
```

`lang='zh'` 一次性解析出 `bert-base-chinese` + `num_layers=8` + 对应 baseline 文件。**必须开 `rescale_with_baseline=True`**：该 baseline 的 layer-8 行是 P/R/F ≈ 0.5476，不做 rescale 时中文 BERTScore 压缩在 0.6–0.9，消融差异肉眼不可见。`hfl/chinese-roberta-wwm-ext` **不被支持**（`model2layers` 无条目、无 baseline TSV），不要用。

⚠️ `bert-score` PyPI 版本 0.3.13 上传于 2023-02-20，仓库 HEAD 停在 2024-04-12——**这是整套栈里最可能装不上 Python 3.13 的包，先装再设计指标表**。

**三个必须在 README 里点破的 CRUD-RAG 事实**（这些本身就是加分项）：

1. 它的 `bertScore` 列**不是 BERTScore**，是 `text2vec-base-chinese` 的句向量余弦（`src/metric/common.py` L74-85）。requirements.txt 里根本没有 `bert-score`。不要把你的真 BERTScore 和他们的数放同一列。
2. 它默认**把 BLEU 的 brevity penalty 除掉了**（`with_penalty=False` → `bleu_avg/brevity_penalty`），所以短生成不受长度惩罚。同时报标准 BLEU 和他们的变体。
3. 论文 Eq.(2) 的 precision 分母是 `|QG(GT)|`（全部问题），**代码却是两次过滤后 `np.mean`**（先去掉 GT 自己答不了的，再去掉 GM 答不了的）。代码版系统性偏高。选一个、写明、全表不混用。

另：所有指标被 `@catch_all_exceptions` 包着，异常返回 `None`，调用方 `or 0.0` ——**崩掉的 ROUGE 会变成 0.0 悄悄进入你的均值**。移植时改成显式失败计数，任何一行失败就拒绝出均值。

### 6.3 LLM-judge 设计

- **判官 ≠ 生成模型**（Qwen 生成 → DeepSeek/Kimi 评判）。
- **RAGQuestEval 自己重实现**（约 60 行）：question generation 用强模型跑**一次**并 commit 结果 JSON（CRUD-RAG 原实现按 `data_point['ID']` 缓存到 `{task}_quest_gt_save.json`），整个消融矩阵**只付一次 QG 的钱**；per-config 的 QA 步是受限抽取任务（「用一两个词或者非常简短的语句回答」，temperature=0.1，max_new_tokens=1280），中档模型足够。
- **无法回答的哨兵字符串是 `无法推断`，做的是精确相等比较**。判官回「无法推断。」带句号就会被算作「答上了」，静默抬高 recall。**必须先归一化再比较，并记录近似哨兵的命中率**。
- README 里写明 judge 模型 + temperature + 日期。judge drift 会无声地作废跨轮次对比。

### 6.4 如果各臂打平怎么办

这是需要**提前决定**的分支，不能等测完再想：

1. **先看 CI 是否重叠**。若 hybrid 与 BM25 的 95% CI 大幅重叠且 paired bootstrap 的 p > 0.05，**如实报告「在本语料上 dense 臂未带来统计显著提升」**。这是第二个诚实的负结果，和 98.6%→75.8% 那个同样值钱。
2. **切分难度桶再看**。按 1doc/2docs/3docs 分层——dense 臂大概率在 2docs/3docs（ALL-gold@10 只有 85.7%/65.0%）上才显出优势，整体均值会把它稀释掉。**分层表是打平时的救命稻草**。
3. **换看 nDCG@10 和 MRR@10**。R@1 打平不代表排序质量打平。
4. **绝不通过换评测集来制造差异**。若换了，必须两个集都报。

---

## 7. 里程碑路线图（业余时间，1 天 ≈ 3 小时有效工时）

已建成：`io_utils.py`、`tokens.py`、`lexical/{analyzers,bm25}.py`、`chunking/markdown.py`、`eval/metrics.py` + 4 个测试文件（~559 行）。

| # | 里程碑 | 天 | 交付物（artifact） | 数字（number） |
|---|---|---|---|---|
| **M0** | 仓库卫生 + CI | **1.0** | `.github/workflows/ci.yml`（ubuntu + windows 双 leg，windows 不设 PYTHONUTF8）；ruff 加 PLW1514；删除 `_research_*.py` / `_enc_test.txt`；`DATA_LICENSE.md` | CI 绿；ruff 0 error；测试通过率 100% |
| **M1** | Protocol + registry + YAML config | **1.5** | `retrieval/base.py`（4 个 Protocol）、`registry.py`、`config.py`（pydantic-settings，`extra='forbid'`）、`experiments/*.yaml` | 现有 BM25 零改动通过 Protocol；1 条命令跑通 1 个 config |
| **M2** | Milvus Lite 冒烟 + provider 客户端 | **1.0** | `store/milvus.py`、`embed/siliconflow.py`（含 instruct 前缀常量）、`rerank/siliconflow.py`；tenacity 退避 | 10 行 create/insert/hybrid_search 通过；embedding L2 范数 = 1.0 验证；dimensions 截断等价性验证 |
| **M3** | 全量索引 + dense 基线 | **1.5** | `scripts/build_index.py`（幂等 upsert，chunk id = hash(path, ordinal, text)）、`corpus_manifest` sha256 变更检测 | 5,681 文档索引完成；dense-4096 的 R@1 / MRR@10 出数 |
| **M4** | 混合检索 + RRF | **1.0** | 客户端 char-bigram → SPARSE_FLOAT_VECTOR（IP）；`hybrid_search` + RRFRanker | hybrid vs BM25 vs dense 三行 + paired bootstrap p 值 |
| **M5** | Rerank + 深度消融 | **1.5** | `rerank` 阶段 + `(qid,docid,model)` 分数缓存 | top-50 / top-100 / 4B vs 8B 四行；**分层报 1doc/2docs/3docs** |
| **M6** | MRL 消融（原创） | **1.0** | 4096/2048/1024/512 四档 | 存储 28.3 MB → 7.1 MB；R@1 损失曲线（**无人发布过的数**） |
| **M7** | chunk sweep | **0.5** | 256/400/800 | 「为什么是 400」有曲线不是故事 |
| **M8** | 服务层 + 延迟/QPS | **1.5** | FastAPI + 单文件静态前端（含各阶段耗时条）；`scripts/bench.py` | **p50/p95/p99 + QPS**——补上「企业级」四条腿里唯一缺的那条 |
| **M9** | 生成侧评估 | **2.0** | 移植 `metrics_gen/`（jieba 词级 BLEU/ROUGE + BERTScore-zh rescaled + 自实现 RAGQuestEval）；commit quest_gt JSON | event_summary / QA-1doc 两个任务对齐 CRUD-RAG Table 8 baseline |
| **M10** | TiDB 第二后端 + 那一节 README | **1.0** | `store/tidb.py` + 「为什么 TiDB 的文档没有跑在 TiDB 上」 | 同一份 Protocol 双后端跑通 |
| **M11** | 可观测性 + Bad Case 归因 | **1.0** | `phoenix serve` tracing；Bad-Case 归因表（召回失败 / 排序失败 / 生成失败三分类，各 20 例） | Bad Case 分布百分比 |
| **M12** | 公网部署 + README 定稿 | **1.0** | Zilliz Cloud Free 集群 + 公网 demo 链接；README 首屏定稿 | 端到端在线可点 |

**合计 ≈ 15.5 天有效工时**。按每周 2 个工作日晚上 + 1 个周末日算，约 **5–6 周**。

**M0 优先完成**：可重复执行的测试、持续集成和数据许可说明是后续实验与发布的基础。

---

## 8. 仓库结构

```
zhrag/
├── .github/workflows/
│   ├── ci.yml                      # ubuntu + windows（windows leg 不设 PYTHONUTF8）
│   └── eval-nightly.yml            # LLM-judge 指标，仅 nightly / 手动触发
├── README.md                       # 首屏：一句话定位 → 500/2000/5681 饱和表 → 3 行 quickstart → 诚实局限
├── DATA_LICENSE.md                 # 两个上游 + pinned commit 26f202bc + CC BY-SA 3.0 URI + CRUD-RAG 引用
├── pyproject.toml
├── src/zhrag/
│   ├── io_utils.py                 # ✅ 已建成（UTF-8 端口）
│   ├── tokens.py                   # ✅ 已建成（中英双分量估算）
│   ├── config.py                   # 🆕 pydantic-settings, extra='forbid'
│   ├── registry.py                 # 🆕 str -> factory
│   ├── lexical/
│   │   ├── analyzers.py            # ✅ char_ngram / jieba_words / union
│   │   ├── bm25.py                 # ✅ 已满足 Retriever Protocol，零改动
│   │   └── sparse.py               # 🆕 BM25 权重 -> {index: weight} 稀疏向量
│   ├── chunking/
│   │   └── markdown.py             # ✅ 两阶段 header-aware
│   ├── retrieval/
│   │   ├── base.py                 # 🆕 Retriever / Fusion / Reranker / Chunker Protocol
│   │   ├── dense.py                # 🆕
│   │   ├── hybrid.py               # 🆕 RRF / Weighted
│   │   └── pipeline.py             # 🆕 retrieve -> fuse -> rerank
│   ├── store/
│   │   ├── base.py                 # 🆕 VectorStore Protocol
│   │   ├── milvus.py               # 🆕 Lite / Standalone / Zilliz 同一份代码
│   │   └── tidb.py                 # 🆕 第二后端（叙事用）
│   ├── providers/
│   │   ├── siliconflow.py          # 🆕 embed + rerank，含 QUERY_PROMPT 常量
│   │   └── cache.py                # 🆕 (qid, docid, model) 重排分数缓存
│   ├── eval/
│   │   ├── metrics.py              # ✅ R@k / MRR / nDCG / ALL-gold / bootstrap
│   │   ├── metrics_gen.py          # 🆕 jieba 词级 BLEU/ROUGE + BERTScore-zh
│   │   ├── quest_eval.py           # 🆕 RAGQuestEval 自实现（~60 行）
│   │   └── runner.py               # 🆕 遍历 experiments/*.yaml -> results/*.jsonl
│   ├── service/
│   │   ├── app.py                  # 🆕 FastAPI
│   │   └── static/index.html       # 🆕 单文件前端 + 阶段耗时条
│   └── ingest.py                   # 🆕 幂等：path 为稳定键，sha256 变更检测，
│                                   #    返回 {added, updated, deleted, skipped}
├── experiments/                    # 🆕 每个 YAML = 消融表一行
│   ├── a_bm25_bigram.yaml
│   ├── h_hybrid_rrf_mrl1024.yaml
│   └── ...
├── results/                        # 🆕 提交（体积小、可复现、可比对）
│   ├── retrieval_ablation.jsonl
│   ├── quest_gt_save.json          # QG 只付一次钱，提交它
│   └── ablation_table.md           # 由脚本生成，非手写
├── scripts/
│   ├── corpus_stats.py             # ✅
│   ├── download_corpora.py         # 🆕 首次运行 stdout 打印许可声明
│   ├── build_index.py              # 🆕
│   └── bench.py                    # 🆕 p50/p95/p99 + QPS
├── tests/                          # ✅ 4 个文件 ~559 行，继续加
└── docs/
    ├── why-not-tidb.md             # 「为什么 TiDB 的文档没有跑在 TiDB 上」
    ├── why-not-pgvector.md         # 4096 > 4000 的 96 维之差
    └── bad-cases.md                # 归因表
```

**语料目录（`crud-rag-subset/`、`tidb-rag-curated/`）保持在 `.gitignore` 里，一个字节都不提交。**

---

## 9. 成本预估

| 项目 | 用量 | 单价 | 成本 |
|---|---|---|---|
| Embedding 全量一遍（两个语料） | 4.88M tokens | ¥0.28/M | **¥1.37**（≈ $0.19） |
| Embedding × 15 遍（MRL 消融 + 重建） | 73M tokens | ¥0.28/M | **¥20.5**（≈ $2.9） |
| Rerank top-50 一次全扫（2,394 queries × 50 × ~490 tok） | ~59M tokens | ¥0.14/M（4B） | **¥8.2** |
| Rerank top-100 一次全扫 | ~118M tokens | ¥0.14/M | **¥16.5** |
| Rerank 消融 ×4 组（有缓存，实际约 2.5 倍成本） | ~250M tokens | ¥0.14/M | **¥35** |
| Rerank-8B 对照一次（top-50） | 59M tokens | ¥0.28/M | **¥16.5** |
| RAGQuestEval — QG（强模型，**只跑一次并 commit**） | ~2,400 题 × ~800 tok | 按 LLM 计费 | **≈ ¥30**（⚠️ 估算） |
| RAGQuestEval — QA（中档判官，per-config） | ~2,400 × N 题 × ~600 tok | 按 LLM 计费 | **≈ ¥20 / 配置**（⚠️ 估算） |
| 生成侧 LLM 输出（4 个任务 × ~2,000 行 × ~250 tok） | ~2M tokens 输出 | 按 LLM 计费 | **≈ ¥40**（⚠️ 估算） |
| BERTScore / BLEU / ROUGE | 本地 CPU | — | **¥0** |
| Milvus Lite / Standalone (WSL2) | 本地 | — | **¥0** |
| Zilliz Cloud Free | 5 GB / 2.5M vCU/月 | 免费 | **¥0** |
| GitHub Actions（公开仓库） | — | 免费 | **¥0** |
| Phoenix tracing | 本地 SQLite | — | **¥0** |
| **总计（含全部消融，保守）** | | | **≈ ¥200–250（约 $28–35）** |

存储侧：1,725 TiDB chunks @ 4096 维 float32 ≈ **28.3 MB**；MRL-1024 ≈ **7.1 MB**；全部落在 Zilliz Free 的 5 GB 里，**约 175 倍余量**。

**成本控制三条铁律**：① 重排分数按 `(qid, docid, model)` 缓存并提交；② QG 结果 JSON 提交进仓库，CI 永不重跑；③ CI 只跑无需 API key 的检索指标，judge 指标 gate 到 nightly。

---

## 10. 明确不做的事

| 不做 | 一句话理由 |
|---|---|
| **不用 LangChain / LlamaIndex 做地基** | `langchain-community` 已于 2026-05-22 sunset（"effective immediately"），llama-index-core 8 周无发版、提交量同比 -70%——建在被弃的那一半上 |
| **不上 DSPy** | prompt optimizer 是消融矩阵里的**不受控混杂因素**：分高的配置可能只是被优化得更狠。要用就单独放到生成阶段实验、检索侧冻结 |
| **不上 GraphRAG / Self-RAG / CRAG** | 每一个都能单独吃掉 5 周，而且和「检索评估」这条主线正交。README 提一句「已知但本项目未覆盖，见 Roadmap」即可 |
| **不做微调（embedding / reranker / LLM）** | 需要标注数据与 GPU 时长，且会让「我的评估集有效吗」这条主线被稀释 |
| **不用 pgvector** | 索引维度 4096 > halfvec 上限 4000，物理上做不到原生索引。写成 `docs/why-not-pgvector.md` 反而是加分项 |
| **不把 TiDB 做主存储** | 向量检索仍标 beta、强依赖 TiFlash、全文检索只在 Frankfurt/Singapore 两地、融合在客户端、pytidb 版本号 0.0.14 |
| **不用 Qdrant Cloud 免费层挂公网链接** | 在本次验证范围内**没有找到一手官方页面支持“闲置 1 周挂起、4 周删除”这一精确说法**，因此不把它写成事实。若使用 Qdrant Cloud，部署前必须按当前计划条款核对；否则使用可重建的 Milvus/Zilliz 或自托管方案 |
| **不用 RAGAS 做主力评估框架** | 2026-02-24 起无提交、559 open issues，而 deepeval/langfuse/opik/phoenix 当天都在发版；且 `adapt()` 默认 `adapt_instruction=False`，中文提示词是半英半中 |
| **不开数据库内置中文分词器** | Milvus `chinese` analyzer = jieba `cut_for_search` = 你实测最差的 72.8%，比 char bigram 低 3 分 |
| **不提交任何语料字节（含「小样本」）** | CRUD_RAG **根本没有 LICENSE 文件**（Apache badge 只是 README 里的 shields.io 图片，GitHub API 报 `license: None`），8 万篇新闻无出处无授权声明；TiDB 文档是 CC BY-SA 3.0，你的 chunk 输出属于 Adaptation |
| **不用 Docker（本阶段）** | 本机没装；Milvus Lite 纯 Python、Phoenix 是 SQLite，全链路可以零容器跑通。真需要 Standalone 时进 WSL2 |
| **不做多租户 / RBAC / 审计** | 没有任何编排框架白送这些，它们来自数据库和应用层；且 Dify 的许可证明确禁止未授权的多租户运营 |
| **不做移动端 / 复杂前端** | 优先展示阶段耗时和检索证据，控制前端维护成本 |

---

## 11. 技术摘要与证据边界

### 实现与实验摘要

**① 主导中文 RAG 评估框架设计与语料重建。** 发现 CRUD-RAG 官方 500 篇子集已饱和——40 行纯标准库字符 bigram BM25 即达 **R@1 98.6%、MRR@10 0.993**，任何检索配置均近满分、消融表无区分度；定位根因为语料规模不足与问句-证据表层重叠。从原始数据去重扩展至 **5,681 篇**干扰语料后 R@1 降至 **75.8%**、MRR@10 **0.856**，释放 **24 个百分点**可优化空间，并将主指标由已饱和的 R@5 改为 **R@1 / MRR@10 / nDCG@10**，全部指标附 95% bootstrap 置信区间与配置间 paired bootstrap 显著性检验。

**② 中文词法检索方案实测选型。** 对比 jieba 精确模式（R@1 **73.8%**）、jieba 搜索模式（**72.8%**）与字符 bigram（**75.8%**）；jieba+bigram 并集达 76.0% 但索引构建耗时 **7 倍**，最终选定字符 bigram。进一步实测**向量数据库内置 jieba analyzer 默认即为搜索模式**，遂将词法臂移出数据库，以客户端预计算 BM25 权重作为 SPARSE_FLOAT_VECTOR（`metric_type=IP`）喂入，数据库仅承担 ANN 与服务端 RRF 融合。

**③ 面向技术文档的两阶段分块策略。** 针对 500 篇 TiDB 中文文档（**3,309** 个代码块 / **5,262** 行表格），纯标题切分导致 **63.5%** 分块 <100 tokens、最大块 **16,111** tokens；改为「标题切分 → 掩码代码块与表格 → 按 target=400 合并小块 / 拆分大块」，得 **1,725** 块，p50 **375** / p90 **747**，欠长块降至 **5.3%**，代码块破损 **0** 例；并给出 256/400/800 的分块尺寸-召回曲线。

**④ 检索栈与成本/性能工程。** 基于 Qwen3-Embedding-8B + Qwen3-Reranker-4B 构建混合检索 + 重排流水线，**R@1 由 75.8% 提升至 [XX.X%]（p=[X.XX]）**；端到端 **p95 [XXX] ms / QPS [XX]**。首次公开 Qwen3-Embedding-8B 的 **MRL 降维质量曲线**（4096→1024 维，存储由 28.3 MB 降至 7.1 MB，**-75%**，R@1 损失 **[X.X] pt**），官方技术报告未发布此数据。以 alias 实现零停机换索引，重嵌入全量成本约 **$0.34**。

> ⚠️ **④ 里的 4 个 `[方括号]` 必须在发布前填实数，绝不能带着括号发出去。** ①②③ 每一个数字都已实测。


---

## 12. 踩坑清单

| 坑 | 后果 | 修法 |
|---|---|---|
| **Windows 默认 cp936** | 裸 `open()` 读项目自己的中文文件直接 `UnicodeDecodeError`；`print()` 中文抛 `UnicodeEncodeError`。文件系统编码却是 utf-8（不对称） | 全项目走 `io_utils.py`；ruff 开 **PLW1514** 禁裸 `open()`；CI 加 `windows-latest` leg 且**故意不设** `PYTHONUTF8`/`PYTHONIOENCODING`（设了这条腿就废了）。⚠️ 注意 GH Actions 的 windows runner 大概率是 cp1252 不是 cp936，它验证的是「非 UTF-8 默认」而非你的具体 codepage |
| **`rouge_score` 默认 tokenizer 删光中文** | `tokenize('近千家展商参与了UDE2023博览会')` → `['ude2023']`；好改写打 1.0000、无关句打 0.0000，全靠共享拉丁子串。**产出一整张看起来能发表的噪声表** | 必须传 `tokenizer=lambda t: list(jieba.cut(t))`；上线前用一对「好改写 / 无关句」做判别力自检 |
| **sacrebleu 默认 `13a`；`zh` 是字符级不是词级** | 数字内部自洽但与 CRUD-RAG 表格不可比 | 选 jieba 词级对齐论文；字符级另列作稳健性检查 |
| **CRUD-RAG 的 `bertScore` 不是 BERTScore** | 是 `text2vec-base-chinese` 句向量余弦；把你的真 BERTScore 和它放同列 = 无意义对比 | 两列分开命名并注明 |
| **CRUD-RAG 默认除掉 BLEU 的 brevity penalty** | 他们的「bleu」不是标准 BLEU，短生成不受惩罚 | 两个都报，注明 |
| **论文 Eq.(2) 与代码的 precision 分母不一致** | 代码版（两次过滤后 `np.mean`）系统性偏高 | 选一个、写明、全表不混 |
| **`@catch_all_exceptions` 返回 `None` + `or 0.0`** | 崩掉的 ROUGE 变成 0.0 混进头条均值（BLEU 因返回 5 元组反而会 TypeError——失败模式还不一致） | 移植时改为显式失败计数，任一行失败拒绝出均值 |
| **`Query:` 后面那个空格** | 模型卡自相矛盾（Python helper 无空格 / TEI curl 有空格）。选错 = 全库向量与线上查询前缀不一致，**索引完之后改不了，只能重嵌入** | 以 `config_sentence_transformers.json` 为准（**无空格**），定义成常量，永不改 |
| **文档侧误加 instruct 前缀** | 模型是非对称的，document prompt 是空串。两边都加会静默掉几个点 R@1，**不报错** | 文档原文入库 |
| **`padding_side` 不是 `'left'`（自建推理时）** | 右 padding + `hidden[:, -1]` → 短样本拿到 pad token 向量，全错且无异常 | 用模型卡的 `last_token_pool()`（按 attention_mask 分支） |
| **DeepInfra `normalize` 默认 false** | 未归一化的 4096 维向量进 cosine 索引 = 排序错误、无报错。SiliconFlow 干脆没文档说归一化默认值 | 第一次响应就 `np.linalg.norm()` 自检 |
| **给稀疏字段挂 `FunctionType.BM25`** | Milvus 会重新分词并**覆盖你预计算的向量** | 用 `SPARSE_INVERTED_INDEX` + `metric_type="IP"`，**不挂** BM25 function。Qdrant 上的对称坑：权重里已含 IDF 就别开 `idf` modifier，否则双重计算 |
| **Milvus Lite 的 BM25 IDF 是 segment 局部的** | Lite 的分数在 Standalone/Zilliz 上复现不出来，指标静默失真 | 走客户端稀疏向量（本方案已规避）；若非要用服务端 BM25，出指标的那一遍只在 Standalone 跑 |
| **Milvus Lite 对 data_dir 加文件锁，单进程** | 并行 eval worker 会死锁或报错 | 每个 worker 独立 data_dir，或 eval 跑 WSL2 Standalone |
| **Milvus Lite 在 Windows + Python 3.13 未经上游 CI 验证** | 上游 CI 只覆盖 Windows+3.10 和 Linux+3.10~3.13 | **第一件事**做 10 行冒烟测试（见 §13），5 分钟去掉整个方案的最大风险 |
| **SiliconFlow reranker 限流是平的** | RPM 2,000 / TPM 500,000 **不随消费等级上涨**；一次 top-100 全扫至少 4 小时墙钟 | 第一天就上 tenacity 指数退避 + 分数缓存 + 断点续跑 |
| **重排成本是 embedding 的 ~24 倍** | 消融矩阵会让你反复付同一笔钱 | 按 `(qid, docid, model)` 缓存并提交结果 |
| **`无法推断` 是精确字符串比较** | 判官回「无法推断。」被算作「答上了」，静默抬高 recall | 归一化后再比，并记录近似哨兵率 |
| **`bert-score` 0.3.13 停在 2023-02-20** | 整套栈里最可能装不上 Python 3.13 的包 | **设计指标表之前先装**；必要时 pin transformers |
| **CRUD_RAG requirements 装不上** | pin 了 `llama_index==0.9.32` / `langchain==0.1.4` / `pymilvus==2.3.3`，全是 namespace 拆分前版本 | **不要 pip install 它**，把 `src/metric/` 那 ~150 行移植进自己的包 |
| **`corpus_manifest.jsonl` 的 `id` 是内容哈希** | `id == git_blob_sha1` 500/500，内容一变 id 就变，不是稳定身份 | 稳定键 = `path`（500/500 唯一）；变更检测 = `sha256`；chunk id = `hash(path, ordinal, chunk_text)` 保证重跑是 upsert 不是重复插入 |
| **README 里「企业级」目前是空头支票** | 尚缺实际延迟与监控验证，不能只凭功能清单作出承诺 | M8 交付 p50/p95/p99 + QPS，或把这个词软化 |

---

## 13. 需要你自己确认的事（清单）

**🔴 阻塞级（做别的之前先做）**

- [x] **Milvus Lite 3.2.0 在 Windows + Python 3.13.5 上的冒烟测试**：已在隔离 `.venv-verify-milvus` 中安装 `pymilvus==3.0.1` + `milvus-lite==3.2.0`，并验证 4096-dim dense、sparse inverted index、`hybrid_search` + `RRFRanker`。结果：**PASS**。提交版脚本：`scripts/smoke_milvus_lite.py`。两个 Windows 特有坑已固化进脚本：① `pymilvus==3.0.1` 的 `[milvus-lite]` extra marker 明确排除了 `win32`，所以 Windows 上必须显式安装 `milvus-lite==3.2.0`；② `MilvusClient.close()` 返回后 `LOCK` 仍可能被 native 后台线程短暂持有，故用 worker subprocess 作为可靠的清理边界。
- [x] **`bert-score` 0.3.13 能否在 Python 3.13.5 + 当前 transformers 上装并跑通**：隔离 `.venv-verify-bert` 安装成功，`bert-score==0.3.13` + `transformers==5.15.0` + `torch==2.13.0` 可 import；`BERTScorer(lang='zh', rescale_with_baseline=True)` 端到端评分也通过。模型下载需可访问 Hugging Face（本机直连超时，使用 `HF_ENDPOINT=https://hf-mirror.com` 通过）。注意 distribution metadata 是 0.3.13，但模块内 `bert_score.__version__` 仍写 0.3.12，这是上游版本字符串不一致，不影响运行。
- [ ] **SiliconFlow 返回的 embedding 是否已 L2 归一化**（无文档）。索引 1,725 个 chunk 之前，对第一个响应算 `np.linalg.norm()`。

**🟡 影响架构决策**

- [ ] **SiliconFlow 的 `dimensions` 截断是「切片+重归一化」还是训练过的 MRL head。** 同一段文本请求 4096 和 1024，检查后者是否等于前者前 1024 维的 L2 重归一化。**若是，你可以只存一份 4096 维、客户端任意截断**，不必重嵌入。两次 API 调用。
- [x] **Zilliz Cloud Free 长期闲置是否会被删除/挂起**：官方 [pricing](https://zilliz.com/pricing) 明确 Free 为 5 GB / 2.5M vCUs per month / up to 5 collections；pricing FAQ 明确“suspended”状态会停止 vector database costs，但 storage costs continue until deletion。官方 [limits](https://docs.zilliz.com/docs/limits) 没有写“闲置 N 天自动挂起/删除”，也没有 Free 专属的闲置删除倒计时。官方 [Terms and Conditions](https://zilliz.com/terms-and-conditions) 则保留较宽的服务终止权，并写明终止后“不承担保留已终止集群或备份快照中数据的义务”。结论：**没有证据支持“闲置 4 周必删”，但也不能承诺永久保留**；不要把 Zilliz Free 当唯一的服务可用性保障，部署前保留本地/可重建 artifact，并在控制台确认具体 region/plan 条款。
- [ ] **Zilliz Cloud Free 跑的是 Milvus 2.6.x 还是 3.0.x。** 3.0.0 于 2026-07-29 GA，托管云通常滞后。建好集群后从版本端点确认，不要信营销页。
- [ ] **TiDB Cloud Starter 免费实例长期闲置是否归档/删除。** `serverless-limitations.md` / `serverless-faqs.md` 里 grep 不到任何相关条款——可能确实没有，也可能在文档仓库外的控制台页面。
- [ ] **`pingcap/docs-cn` 的 pinned commit `26f202bc` 是在 TiDB v7.0 之前还是之后。** README 说 CC BY-SA 3.0 是「自 TiDB v7.0 起」适用，确认你的快照落在该声明覆盖范围内。
- [ ] **就 CRUD_RAG 的许可开一个 issue**（无 LICENSE 文件 + 8 万篇新闻无出处）。一个礼貌的 issue 零成本、可能拿到明确答复，且在 402 star 的仓库上本身就是可见的尽职信号。

**🟢 影响指标可信度**

- [ ] **你现有的 `paired_bootstrap_test` 实现是否统计正确**：重采样 query 还是 per-query 分差？单尾还是双尾？~40 个消融格是否需要多重比较校正？懂 IR 的评审会**先查这个再看你的 R@1**。函数存在，但研究阶段没读实现。
- [ ] **CRUD-RAG 论文 Table 8 的 baseline 数字**是 pypdf 文本抽取得来的，PDF 表格抽取可能错位相邻数字。**你实际引用的那 3–4 行**（summarization、QA-1doc）要对着原 PDF 逐个核对。
- [ ] **GitHub Actions `windows-latest` Python 的实际默认 codepage**。预期是 cp1252（美式 locale）而非你的 cp936——它复现的是「非 UTF-8 默认」，不是你的具体环境。要精确复现 cp936 得强制 locale 或在某个 job 里设 `PYTHONIOENCODING=gbk`。
- [ ] **`rouge-chinese` 与 `evaluate + rouge_score + jieba` 两条路径的数值差多少。** 研究阶段只测了前者（好 0.8276 / 坏 0.1154，判别力正常），没有在同一输入上跑两者做对比。混用或替换前跑一次。
- [ ] **DeepInfra 的 Qwen3-Embedding-4B 标价 $0.020/M、比 8B 的 $0.010/M 贵一倍**，这个反常价格可能是促销或过期数据。做预算前在实时页面确认。
- [ ] **英文 instruction 与中文 instruction 在你的中文语料上到底哪个好。** Qwen 基于训练数据来源推荐英文，但那是通用建议不是在 TiDB 文档上的实测。两次跑，同一评测，又一行诚实消融。
- [ ] **hybrid + rerank 到底能不能在你的 5,681 语料上打赢 char-bigram BM25 的 75.8%。** 尚未测量。**若打不赢，那就是第二个诚实的负结果——现在就决定你会怎么发布它**，应在发布结论前完成验证。
- [ ] **RAGAS / DeepEval 内置指标提示词在中文上的校准度。** 研究只验证了管道（`base_url` 支持、`adapt_instruction` 语义、DeepSeek/Kimi 类），**零中文评测**。人工标 ~50 行，先测判官与你的一致率。

**部署与生态待验证事项**

- [ ] **Langfuse / Opik / Confident AI 的免费额度**（研究阶段定价页被限流）；**Opik 自托管是否强制 Docker/K8s**（未核实，按 Langfuse 类比推断）。
- [ ] **Elasticsearch 若仍在对比范围内**：`dense_vector` 的 `dims` 上限**恰好是 4096**（零余量，换更宽的模型就直接废），且 `infinilabs/analysis-ik` 的 GitHub releases 最新一条停在 2024-05-06，未确认存在兼容 ES 9.5 的 IK 构建。