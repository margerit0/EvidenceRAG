# zhrag 架构决策与验证记录

> 初版规划日期为 2026-08-17，本文随后持续更新。历史调研中的版本、价格和供应商限制并非全部经过本项目实测；重新部署时应核对当前官方文档。当前实现以代码、认证报告和 §13 的验证状态为准，未完成项不代表已交付能力。

> **数字勘误说明**：本报告综合时使用的分块统计来自早期原型（固定 1.15 字符/token）。
> 生产实现按 Qwen3 tokenizer 实测标定（中文 1.57 / 英文 4.49），已全部更正为
> `scripts/corpus_stats.py` 的输出：朴素按标题切分的欠长块占比是 **63.5%**（非 31.4%），
> target=400 得 **1,832** 块（非 4,191），4096 维存储 **30.0 MB**（非 115 MB）。
> 结论方向不变，且更强。

> **验证状态更新**：以下三项阻塞验证已于 2026-08-18 在本机 Windows 11 + Python 3.13.5 完成。旧 checklist 已保留为审计轨迹，状态在 §13 更新。

---


> **在线问答增补（2026-09-07）**：新增显式启用的 `/api/ask` 单轮证据问答，包含完整段落预算、结构化引用校验、拒答/故障分离和问答界面。详见 [问答合同](answering.md)。下文“HTTP 无生成阶段”的历史描述仅适用于原 `/api/search` 与已认证 M8 profile；M11 trace 仍仅覆盖检索子阶段，生成耗时单独返回。当前合成回归已验证功能合同，并完成小规模真实问答冒烟；观察到生成失败与引用支持不完整，尚无答案准确率或端到端性能基准；M9b known-context 评测状态不因此改变。

> **M9a 状态（2026-08-31）**：生成指标合同、RAGQuestEval answer-scoring 语义、aggregate-only Table 8 证据和独立同步器已完成；本阶段没有生成真实答案，也没有付费调用。M9b 才会在本地 gitignored 目录中运行 QG/QA/生成实验。

**当前在线路径：窄 Protocol + 显式组合根；Milvus Lite；Qwen3-Embedding-8B dense-4096 + 客户端 char-bigram BM25 稀疏向量；客户端 exact RRF；Qwen3-Reranker-8B 请求 top-100 分数、应用前 top-50。可选 `/api/ask` 在检索后生成带引用的单轮答案。MRL 降维为独立离线实验，Standalone/Zilliz 公网部署和 TiDB Cloud 真机验证仍属于后续工作。检索评估与生成验证分开报告，默认 CI 使用合成数据且不调用付费模型。**

---

## 2. 技术栈选型表

| 层次 | 选型 | 理由 | 备选（及改选条件） |
|---|---|---|---|
| **向量数据库** | **Milvus**：本地 `milvus-lite 3.2.0`（纯 Python，支持 Windows）→ WSL2 Ubuntu-24.04 Standalone → Zilliz Cloud Free（5 GB / 5 collections / $0） | 4096 维无压力（上限 32,768）；`hybrid_search` + `RRFRanker` 服务端融合；collection alias 支持原子换索引；同一份 pymilvus 代码跑三种部署 | **Qdrant v1.19.0**（原生 `qdrant-x86_64-pc-windows-msvc.zip`，无需 Docker/WSL2）。改选条件：你决定公网 demo 自己托管 VPS / HF Space，或彻底拒绝 Docker。**pgvector 已被排除**，见 §3 |
| **检索框架** | **自研薄层**：窄 Protocol、frozen settings 与显式组合根 | 当前检索步骤固定，依赖注入足以隔离编码器、存储和重排器，便于单独测试 | 只有出现真实的编排需求时才引入框架；新增适配层需验证是否改变检索或评估合同 |
| **embedding** | **Qwen/Qwen3-Embedding-8B** @ One Hub relay；本地保留 4096 维缓存，部署候选为 MRL 1024 维 | 已实测 4096→1024 的 R@1 仅 −0.50pp、Holm p=0.684，存储降 75%；中转站行为与 SiliconFlow 不能混用 | Qwen3-Embedding-4B。改选条件：有可用 endpoint 后，在相同语料与 query 上做配对检验，而不是引用 C-MTEB 的跨模型点估计 |
| **rerank** | **Qwen/Qwen3-Reranker-8B** @ One Hub relay，传 `instruction`，部署窗口候选 **top-50** | 已在冻结的 dense-4096 hybrid 上完成 top-50/100 分层消融：arity=1 `hit@1` +6.06pp（Holm p=0.0003），arity=3 `ALL@10` +7.15pp（p=1.32e−09）；top-100 未显著优于 top-50 | 4B 对照取消：当前中转站不提供。若换 provider，必须新建独立 fingerprint/cache，不能与现有 8B 分数混用 |
| **LLM（生成）** | 由 `LLM_*` 配置注入 OpenAI-compatible chat；真实冒烟模型为 `gpt-5.6-sol` | 严格校验返回模型、答案结构和引用编号，默认不启用生成；已观察到供应商错误和引用支持不完整 | 更换模型需单独验证；生成质量和含模型调用的延迟不能引用检索基准代替 |
| **LLM（评判）** | 当前 TiDB pooled qrels：与 QG 相同的 `gpt-5.6-sol`、`reasoning_effort=high`；未来生成侧对比实验仍要求独立 judge | 本轮可用配置只有同一请求模型，因此报告明确写 **self-agreement / synthetic labels**，不冒充独立复核；若要提升标签可信度，应在冻结 pool 上补不同模型复判或人工校准 | DeepSeek-V3 类或 Kimi 等非生成模型；切换后必须新建 provenance/cache，不与现有判断混用 |
| **分块** | 已建成的两阶段：header split → 掩码 code fence/table → target=400 合并小块/拆大块 | 实测 n=1,832，p50 371，p90 734，欠长块 5.0%，代码块破损 0；M7 已完成 256/400/800 source-level known-item sweep | 400 保留为 canonical reference；M7 两个相对 400 的主终点比较经 Holm 均未显著，不能声称 400 全局最优、等价或无损 |
| **检索管线** | dense-4096（COSINE）+ sparse（char-bigram BM25，**IP**）两臂检索 → 客户端 exact RRF → 客户端 Qwen3-Reranker-8B 重排 top-50（请求窗口 top-100） | 三段式，每段可单独消融；top-50 由 top-100 未检出额外收益的实测决定 | **WeightedRanker 是必测项而非备选**：离线实测等权 RRF 相对 dense 单臂不显著（39 胜 24 负，Holm p=0.231），加权 0.3/0.7 才显著（16 胜 4 负，p=0.047）。反过来 `RRFRanker` 的 **k 几乎不影响结果**（60→10 只动 0.1pp），不值得占消融表一列。Qdrant 的 dbsf 仍可作对照 |
| **词法检索** | **客户端算 char-bigram BM25 权重，作为 SPARSE_FLOAT_VECTOR 推给 DB** | Milvus 内置 `chinese` analyzer 就是 jieba，且默认 `mode="search"` = `cut_for_search`——正是你实测最差的 73.4%，比 char bigram 的 75.9% 低 2.5 分。**开服务端分词器会让系统变差**。另外可绕开 Milvus Lite「BM25 IDF 按 segment 局部统计」的坑 | 无（这是本项目最有说服力的设计决策之一） |
| **服务层** | FastAPI + 线程池中的同步 provider 调用；urllib transport 与独立生成重试策略 | 共享有限并发 admission，固定错误分类，检索和生成耗时分开报告 | 超时不是整个请求的严格墙钟上限，长重试不构成在线可用性保证 |
| **前端** | FastAPI 挂一个单文件静态 HTML（检索框 + 结果卡片 + 命中 chunk 高亮 + 各阶段耗时条） | 界面优先展示**阶段耗时**和**检索证据**，便于检查系统行为 | Gradio / Streamlit（若你想 5 分钟部署到 HF Space）。改选条件：你决定公网 demo 放 HF Space 而非 Zilliz |
| **评估** | 检索侧：**已建成**（R@k / MRR / nDCG / ALL-gold / bootstrap CI / paired bootstrap）。生成侧：**M9a 已冻结独立指标合同**（逐样本 BLEU/ROUGE-L、可选真实 BERTScore、RAGQuestEval 评分语义），真实生成实验留给 M9b | 不安装或移植无授权的 CRUD_RAG 代码；Table 8 只保留 aggregate-only 来源证据 | RAGAS 只作为「我知道这个框架」的一行说明。**不要当主力**：最后一次 commit 2026-02-24，559 open issues，而竞品当天都在发版 |
| **可观测性** | **provider-free retrieval trace contract**：默认 no-op sink，测试用 in-memory sink；可选 sink 只接收脱敏 span lifecycle、profile fingerprint、候选/输出计数与八阶段耗时 | 不把 Phoenix/OTEL 运行时或外部 UI 当作已验证交付；HTTP 检索 trace 不覆盖生成，生成状态固定为 `not_evaluated` | Phoenix / OpenTelemetry 可作为后续 exporter，但必须单独验证部署、许可与数据边界 |
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
| **混合检索** | ✅ 服务端 `hybrid_search([AnnSearchRequest...], ranker=RRFRanker())`，Lite 本地也支持 | ✅ `prefetch` + `rrf`/`dbsf`（v1.10+），RRF 的 k 可调（v1.16+），加权 RRF（v1.17+）——融合 API 比 Milvus 更干净 | SQL 暴露 vector/full-text 两臂；本仓库为保持 frozen exact RRF 语义在客户端融合。`pytidb` 不是 runtime 依赖 |
| **中文全文检索** | ✅ 内置 `chinese` analyzer = jieba + cnalphanumonly；另有独立 `jieba`/ICU/Lindera tokenizer。**但你不该用**（见下） | ❌ 无 jieba，只有 `multilingual` tokenizer + BM25-as-sparse-vector + `idf` modifier | ⚠️ `WITH PARSER MULTILINGUAL` 支持中日韩；当前仍称 early stages，限 TiDB Cloud Starter 的 Oregon / N. Virginia / Tokyo / Frankfurt / Singapore。它不等于本项目实测的 char-bigram arm |
| **元数据过滤** | ✅ 完整 Milvus 表达式：`text_match()`、JSON path、`array_contains`、`$meta[...]` | ✅ payload filter，每个 prefetch 可独立带 filter | ✅ 就是 SQL WHERE（这点最强） |
| **Windows / WSL2** | ✅ Lite 3.2.0 是纯 Python `py3-none-any` wheel，官方 Requirements 明写 Windows；faiss-cpu 1.15.0 有 cp313 win_amd64 wheel。⚠️ **Windows+3.13 组合不在上游 CI 矩阵内** | ✅✅ 官方发布 `qdrant-x86_64-pc-windows-msvc.zip`，解压即跑，**安装路径最干净** | ☁️ 纯云，本地无关 |
| **免费额度长期存活** | ✅ Zilliz Cloud Free：5 GB / 2.5M vCU/月 / 5 collections。官方页面**没有写 Free 集群闲置 N 天自动删除/挂起**；但 Terms 保留服务终止权，且终止后不承诺保留数据。不能承诺永久存活，必须保留可重建 artifact | ⚠️ 原生 Windows 运行路径干净；免费云集群的闲置/删除策略必须按当前官方条款自行核对，不能把未经一手来源确认的“1 周挂起、4 周删除”写成定论 | ⚠️ Starter：5 GiB 行存 + 5 GiB 列存 + 50M RU/月，免信用卡。配额耗尽会影响新请求，长期闲置/归档条款仍需单独核对 |

### 3.3 关键设计决策：**不要用数据库的中文分词器**

Milvus 内置 `chinese` analyzer 等价于 `{"tokenizer": "jieba", "filter": ["cnalphanumonly"]}`，而简写 `{"tokenizer": "jieba"}` 的默认是 `mode="search"`，即 jieba 的 `cut_for_search`——你实测 **73.4% R@1**，比 char bigram 的 **75.9%** 低 2.5 分。**开这个开关会让系统变差。**

正确做法：进程内算 char-bigram BM25 权重 → 作为 `SPARSE_FLOAT_VECTOR` 写入 → `SPARSE_INVERTED_INDEX` + `metric_type="IP"`。数据库负责两臂检索，默认在线路径在客户端执行 exact RRF。

三个副产品：
1. 绕开 Milvus Lite「BM25 IDF 按 segment 局部统计」的限制，否则 Lite 的分数与 Standalone/Zilliz 对不上，会**悄悄污染你上报的指标**；
2. 数据库选型变得基本可逆（只委托了 ANN + fusion）；
3. 选型依据：**「实测数据库内置 jieba analyzer 与客户端 char-bigram BM25 臂，后者高 2.5 pt R@1，遂将词法臂移出数据库、以预计算稀疏向量喂入」**——这是量化的设计决策，不是框架默认值。

### 3.4 对「用 TiDB 向量检索服务 TiDB 文档」这个叙事的直接裁决

**做，但只做成 adapter 接口后面的第二后端，绝不做主存储。**

理由是当前一手文档与本项目验证状态都不支持它承重：
- 向量检索当前仍是 **public preview**（"might be changed without prior notice"）；
- HNSW 向量索引依赖 **TiFlash replica**，ANN 查询形状也有限制；
- 全文检索仍在 early stages，限 TiDB Cloud Starter 的五个 AWS region，且其 `MULTILINGUAL` parser / BM25 不是本项目实测的 char-bigram arm；
- 本项目的 frozen exact RRF 已在客户端实现；换用另一套融合会让“后端对照”同时改变算法；
- 仓库没有真实 TiDB endpoint，尚无 `EXPLAIN annIndex`、建索引、两臂排序、round-trip、publish/reconnect 证据。

把头条指标压在未经本项目 live smoke 的 preview 能力上，会使复现路径缺少完整的验证闭环。

**当前实现**：`VectorStore` Protocol 下保留 `MilvusStore` 主路径，并新增惰性 PyMySQL 的 `TiDBStore`：dense 使用原生 VECTOR/HNSW，char-bigram BM25 使用 companion sparse postings，exact RRF 仍在客户端。README 已写明「为什么 TiDB 的文档没有跑在 TiDB 上」；只有真实 TiDB Cloud 实例通过上述 smoke 后，才声称“双后端跑通”。

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

**这是你最高价值的原创消融**：4096 / 2048 / 1024 / 512 四档，跑 5,681 文档的 R@1 / MRR@10。存储侧 1,832 chunks 从 30.0 MB → 7.5 MB（-75%）。

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

**候选深度必须 > 10。** Qwen 自己的评测基准是 **top-100**。你实测的 ALL-gold@10 是 1doc 99.9% / 2docs 87.3% / **3docs 68.9%**——top-10 重排窗口会把 3docs 任务的天花板锁死在 68.9%。跑 **top-50 和 top-100 两档**作为消融。

### 4.5 Provider、价格、限流

> **⚠️ 2026-08-18 更正：实际在用的不是 SiliconFlow。** `.env` 指向已配置的中转端点，
> 一个 One Hub 中转站，模型为 `Qwen/Qwen3-Embedding-8B` 与 `Qwen/Qwen3-Reranker-8B`（**没有 4B**，
> 所以消融表的 K 行「4B vs 8B」跑不了）。下表所有 SiliconFlow 特有结论——`normalize` 默认值、
> `dimensions` 白名单、`/v1/rerank` 的 `instruction` 字段、RPM/TPM——**对该中转站均未经验证**，
> 后端接的是谁不透明。已实测的部分见 §13。两个实测到的行为差异：
>
> 1. **该 key 分组限时段调用（09:00–18:00）**，窗口外返回 HTTP 403 `one_hub_error`；
> 2. **Cloudflare 拦截 `Python-urllib/3.x` 默认 UA**，返回 HTTP 403 `error code 1010`。
>    两者状态码相同且都不是鉴权失败，不读 body 会误判成 key 有问题。必须显式设置 `User-Agent`。

当前实现只把 **One Hub relay** 当作已验证 provider。旧调研中的 SiliconFlow / DeepInfra / 百炼价格、
白名单与限流可作为换站时的线索，但不能用来描述当前端点，更不能拿 SiliconFlow 标价反推这次账单。

已验证的 transport 契约：显式 `User-Agent`（否则 Cloudflare 403/1010）、最多 7 次尝试、5 秒起步且
60 秒封顶的长退避、尊重 `Retry-After`（最多 300 秒），以及对 429/500/502/503/504/520 和连接重置
重试。429 在该站表示上游负载饱和，不是配额耗尽。HTTP 400 没有 blanket retry：它也可能是请求错误；
本轮少量看似瞬态的 400 在人工确认 checkpoint 后用原请求单条续跑成功。

**成本结构只保留行为结论，不保留未验证价格结论。** Rerank 输入远大于 embedding，必须缓存；本轮
2,394 × 100 = **239,400** 个 `(query_id, doc_id)` 分数只打一次，top-50/100 是同一缓存上的离线
窗口消融。缓存按 query 的完整响应追加，sidecar 绑定 model、endpoint、instruction、候选与输入指纹；
它连同 embedding cache 都是语料派生物，**只留本地并 gitignore，绝不提交**。

> ⚠️ 本轮评分跨多个时间段断点续跑且 query 顺序按 task/arity 排列；relay 没有暴露后端部署 revision。
> 未观察到缓存污染或漂移，但跨 arity 结论依赖 endpoint stationarity。后续 provider 应记录可用的
> request id / response model / deployment revision，并随机化或交错 arity 的请求顺序。

---

## 5. 两个语料怎么组织

**核心原则：CRUD-RAG 是尺子，TiDB 是产品。尺子上调参，产品上展示，两者数据永不混流。**

<!-- BEGIN TIDB-CORPUS-EVAL-STATUS -->
| | `crud-rag-subset`（5,681 篇干扰语料） | `tidb-rag-curated`（450 篇 / 450 篇入库 / 1,832 chunks） |
|---|---|---|
| **角色** | **探索性检索 benchmark**——现有消融数字来源 | **部署集 + 合成 pooled qrels**——公网 demo 与域内评测输入 |
| **有无 gold label** | 有上游 `evidence_document_id` | **无上游人工 gold**；已有 direct/paraphrase QG、四系统 pooling、rank-blinded LLM judge 形成的合成 qrels |
| **在它上面调什么** | 已探索 analyzer、fusion 权重、rerank 深度、MRL 维度 | 当前冻结配置，不用这批 query 反向调参；若要调，先拆 dev/test 或明确为探索性 |
| **在它上面测什么** | R@1 / MRR@10 / nDCG@10 / ALL-gold@10 + 配对检验 | 已完成 Hit@1 / R@1 / MRR@10 / binary+graded nDCG@10、source-cluster bootstrap、direct/paraphrase 与描述性词面重叠分层；HTTP 延迟/QPS 已由 M8 独立认证，增量重建仍待测 |

当前 TiDB qrels 覆盖 490 个 pair / 980 条 query / 24,525 个 pooled candidates，分属 245 个 `gold_source_key` 源聚类。
四条冻结 run 的 top-1/top-10 候选均已判断，但 **100% judged coverage 不是 100% retrieval quality**；质量表已用 source-clustered 95% CI、双尾配对 source-cluster bootstrap 与 Holm 校正生成。
<!-- END TIDB-CORPUS-EVAL-STATUS -->

### 怎么避免看起来像过拟合

1. **配置冻结点写死**：README 明确写「所有超参在 CRUD-RAG 5,681 文档评测集上选定，选定后冻结，原样部署到 TiDB 语料，未在 TiDB 上做任何调参」。这句话本身就是方法论声明。
2. **dev/test 边界必须按事实描述**：早期计划是 1doc(800) 做 dev、2docs+3docs 只最终上报，但后续 arity 分层、RRF 审计与 rerank 深度消融已经查看了全部 2,394 条。现有数字因此是**探索性 benchmark 结果，不是未触碰的 test 泛化估计**；发布前若要声称泛化，必须另建 held-out split 或外部语料，不能继续沿用这条已失效的计划。
3. **报告置信区间**：**消融表每一行带 95% CI，配置间差异带 p 值与 win/loss**。✅ `paired_bootstrap_test` 已于 2026-08-19 审计（详见 §13）：配对重采样 query、单尾、Holm 已实现；另修了蒙特卡洛分辨率下界与 `observed<=0` 的保守短路，并为二元指标加了精确 McNemar。**一个带 CI 与 win/loss 的 +1.2pt 是工程结论；一个裸的 +1.2pt 是噪声** —— 本项目的 dense vs BM25 正是「裸看 +2.1pt 像结论、配对检验后是噪声」的实例。
4. **TiDB 侧按标签来源限定结论**：无标注量继续报告延迟、吞吐、chunk 分布、代码块完整率与增量重建计数；合成 pooled qrels 可用于域内系统指标，但必须显式标为合成标签，并与上游人工 gold 隔离。**不要把 judged coverage 写成质量，也不要把合成 qrels 写成人工标注。**
5. **两个语料及其派生物都不入库**：`.gitignore` 覆盖语料目录，向量、chunk、rerank pair score 与可能复述原文的 QG 输出同样不得提交；许可边界记录在 `DATA_LICENSE.md`。

---

## 6. 评估方案

### 6.1 检索侧消融表骨架（主表）

**已实现**：`src/zhrag/eval/metrics.py` 里的 R@k / MRR@k / nDCG@k / ALL-gold@k / bootstrap_ci / paired_bootstrap_test / **mcnemar_exact** / **win_loss_tie** / **bootstrap_p_floor** / **holm_floor_flags** / holm_bonferroni，`src/zhrag/retrieval/fusion.py` 的 **RRF（可加权、可指定融合深度）**，以及 `eval/rerank.py` 的窗口语义、输入指纹与四个预声明检验族。以下只是把它们排成表。

**列**：

| 列 | 说明 |
|---|---|
| `config_id` | 与 YAML 文件名一一对应 |
| `corpus_size` | 500 / 2000 / 5681（**永远显式，绝不省略**） |
| `retriever` | bm25-bigram / bm25-jieba / dense-4096 / dense-1024 / hybrid-rrf |
| `chunk_target` | 256 / 400 / 800 |
| `rerank` | none / qwen3-8b@50 / qwen3-8b@100（当前 provider 无 4B） |
| `R@1` ±95%CI | **头条指标** |
| `MRR@10` ±95%CI | 头条指标 |
| `nDCG@10` ±95%CI | 头条指标 |
| `ALL-gold@10` | 按 1doc/2docs/3docs 分列（**3docs 的 68.9% 天花板必须同表标注**） |
| `p_vs_baseline` | 二元指标用**精确 McNemar**（多 gold 时测试 `hit@1` / `ALL@10`，不把分数型 R@1 塞进列联表），连续指标用**配对 bootstrap**；按预声明 family 做 Holm 校正 |
| `win/loss` | 逐查询胜/负计数。**必列** —— 同样是 +2pp，由 86 胜 69 负得来（churn 0.80，不可检测）和由 16 胜 4 负得来（显著）是两回事，而 delta 列一模一样 |
| `index_build_s` | jieba+bigram 并集慢 7 倍这件事要有列承载 |
| `latency_p95_ms` | 端到端 |
| `cost_usd` | 该配置跑一遍的 API 花费 |

**行（最小可交付的 12 行）**：

```
A. bm25-bigram          @5681, unit=document, chunk=N/A, rerank=none      ← 基线，已有 75.9 / 0.857
B. bm25-jieba-precise    @5681, unit=document, chunk=N/A, rerank=none      ← 已有 74.8
C. bm25-jieba-search     @5681, unit=document, chunk=N/A, rerank=none      ← 已有 73.4
D. dense-4096            @5681, unit=document, chunk=N/A, rerank=none      ← ✅ 已测 78.0 / 0.866
E. dense-1024 (MRL)      @5681, unit=document, chunk=N/A, rerank=none      ← ✅ 已测 77.5 / 0.861（p=0.684，不显著劣于 D）
F. dense-512  (MRL)      @5681, unit=document, chunk=N/A, rerank=none      ← ✅ 已测 78.4 / 0.864（p=0.684）
G. hybrid-rrf (A+D)      @5681, unit=document, chunk=N/A, rerank=none      ← ✅ 已测（离线）79.9 / 0.881；vs A +4.00pp（p=1.6e-03，显著）；vs D +1.87pp（Holm p=0.231，不显著）
G'. hybrid-rrf 加权 .3/.7 @5681, unit=document, chunk=N/A, rerank=none     ← ✅ 已测（离线）79.5 / 0.878；vs D +1.50pp（16 胜 4 负，Holm p=0.047，全表唯一显著优于 D）
H. hybrid-rrf (A+E)      @5681, unit=document, chunk=N/A, rerank=none      ← ✅ 已独立离线运行；结果见下方报告生成区
I. G + rerank-8b@50      @5681, unit=document, chunk=N/A                   ← ✅ 已测：按 arity 报告；当前部署候选
J. G + rerank-8b@100     @5681, unit=document, chunk=N/A                   ← ✅ 已测：未显著优于 I
K. ~~4B vs 8B~~                                             ← ❌ 取消：中转站没有 4B（见 §4.5）
L. H + rerank-8b@50      @5681, chunk=256 / 800             ← 待跑；先完成独立 H baseline 与 chunk sweep
M. bm25-bigram           @500,  unit=document, chunk=N/A, rerank=none      ← 饱和对照行，98.0 / 0.990
```

<!-- BEGIN H-HYBRID-MRL1024-STATUS -->
H（A+dense-1024）已独立离线完成：1doc R@1 **79.6% [76.8%, 82.4%]**、MRR@10 **0.878 [0.860, 0.894]**；同次重建的 G R@1 **79.9% [77.0%, 82.6%]**。efficacy / binary 5/12 reject；efficacy / continuous 7/12 reject；retention / binary 1/6 reject；retention / continuous 1/6 reject。H−G 是双尾 difference test，不是 non-inferiority/equivalence；I/J 仍绑定 dense-4096 的 G 与原 rerank cache。
<!-- END H-HYBRID-MRL1024-STATUS -->

> **D/E/F 已于 2026-08-19 实测完成**（`scripts/probe_mrl_quality.py`，800 条 1doc 查询）。
> 三个结论改变了后续排期：
>
> 1. **dense 只比 BM25 高 2.1pp，且已证不显著**（78.0 vs 75.9；配对 95% CI [−0.88, +5.12]pp，
>    McNemar 精确双尾 p = 0.199）。头条指标不能靠 dense 单臂 —— 但**融合的理由反而更硬了**：
>    两臂 R@1 列联表 538 / 69 / 86 / 107，φ = 0.455，**并集 oracle 上限 86.6%**。G 行已把其中
>    4.00pp 兑现（对 A 显著）；I/J 已在 G 上完成；H 已作为独立 1024 维实验运行，结果由上方
>    `H-HYBRID-MRL1024-STATUS` 报告区生成。
> 2. **M6（MRL 消融）实际已经做完**，且零 API 成本：1024 维 −0.50pp（p=0.684）、128 维 −1.00pp
>    （p=0.521）、**只有 64 维显著劣化**（−4.25pp，Holm p=0.001†；† 表示 0/10,000
>    零分布样本达到观测值的 add-one 蒙特卡洛地板，不是 `<` 上界）。存储 93.1 MB → 23.3 MB（−75%）。
> 3. 因此 **E 而非 D 被用于独立 H 行的 dense 臂**：存储少 75%，H 的实际质量与配对检验由上方
>    认证报告区生成。但已完成的 I/J 付费实验明确建立在 G（dense-4096）上；不能事后把它们改名成 H。

**M 行必须存在，且和 A 行贴在一起。** 这是全篇最重要的排版决策——非技术筛选人看到孤零零的 75.9% 会当成退化。

### 6.2 中文生成指标与 CRUD-RAG Table 8 证据

<!-- BEGIN M9A-GENERATION-METRICS -->
**M9a：独立生成指标合同 + Table 8 证据（无付费调用）**

M9a 只交付纯内存评分、合成测试、来源证据和默认 CI 隔离；不读取 CRUD-RAG/TiDB 语料，不生成答案，不提交 `quest_gt`，不调用 QG/QA/chat/embedding/rerank/judge/Milvus。真实 event summarization 与 QA-1doc 生成实验留给 M9b。

**固定合同**：
- `mean_sentence_bleu4`：单 reference、token-level modified 1–4 gram precision、
  几何均值与标准 brevity penalty；无 smoothing、无 effective-order，逐样本后取算术平均。
- `crud_mean_sentence_bleu4_no_bp`：同一逐样本算法但移除 brevity penalty，只作兼容审计列，
  不称为 standard BLEU；两者都不是 corpus BLEU。
- `mean_sentence_rouge_l_f1`：token LCS 的 sentence-level precision/recall/F1（beta=1），
  逐样本后取算术平均；是 rougeL，不是 rougeLsum。
- `bert-base-chinese` 的真实 BERTScore 为可选模型适配器：lang=`zh`、layer=8、rescale_with_baseline=True、idf=False、batch=64、use_fast_tokenizer=False。
- `zhrag-rag-quest-eval-v1`：无法回答哨兵为 `无法推断`；报告论文全问题分母与历史代码条件分母，不把空条件集合伪造为 0。

Table 8 只作为历史来源锚点：历史值来自 [arXiv 2401.17043v3](https://arxiv.org/pdf/2401.17043v3) 的 Table 8（HTML 锚点：[https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8](https://ar5iv.labs.arxiv.org/html/2401.17043#S4.T8)，PDF 第 26 页；副本 SHA-256 `2e4ae0cb708fdca9d96bcf8d1c0713dae121195a0a31b78e3132e9ef4fa7db8a`）。 上游固定为 `https://github.com/IAAR-Shanghai/CRUD_RAG@1aace383994e1f68efa12cf2a8e2dadfb4102ceb`，许可状态仍为无 LICENSE；其聚合值不构成本项目数值等价或实现复用许可。
<!-- END M9A-GENERATION-METRICS -->
<!-- BEGIN M9B-GENERATION-STATUS -->
**M9b1：CRUD-RAG 生成评测编排合同**

M9b1 冻结合同 `m9b1-known-context-v1`，当前 profile 是 known-context：生成请求直接使用每条 case 的已知 context，暂不包含 retrieval stage。支持的 task 为 `event_summary`, `questanswer_1doc`。这不是 CRUD-RAG Table 8 reproduction，也不产生端到端检索质量结论。

**已交付的离线边界**：
- 输入、prompt、hash、cache row、question/reference bank 与 finalizer 均为纯内存合同；输入 schema=`zhrag-crud-generation-v1`，cache row schema=`zhrag-crud-generation-cache-row-v1`。
- generation、QG、reference QA、prediction QA 和 semantic stage 按依赖图断点续跑；每 case 最多 8 个问题，prompt estimator 超过 30,000 个估算 token 即拒绝截断。
- status/dry-run 是只读检查，不读取 `.env`，不导入 provider 或可选模型，不创建 cache、artifact directory 或 lock；付费 chat 与模型加载必须分别显式开启 guard。
- cache sidecar 重新认证 input manifest、prompt、model/endpoint、served model、父级 fingerprint 和完整性；final report 只能由完整、认证过的 cache 离线重建。
- 完成后的 provenance hardening 统一拒绝 Windows 路径别名与保留设备名，固定 artifact root 为项目内 `indexes/crud/generation/v1/`，拒绝运行/参考路径上的 symlink/junction，并要求 provider 显式返回且全程匹配 requested model；chat profile 域与公开 provenance envelope 已轮换到 v2，旧 v1 sidecar/samples/report 不会被新 runner 续跑、原地改写或发布。

**artifact 与隐私边界**：
- 文本型 cache、question/reference bank 和中间答案只允许存在于 `indexes/crud/generation/v1/` 的 gitignored 本地树；不提交 corpus、source、question、answer、prediction、embedding、provider payload 或 BERTScore 权重。
- 发布的 `numeric_samples.json` 与 `report.json` 只含匿名 sequence/cluster、task、metric、allowlisted fingerprint 和统计计数，不含任何原文或可回溯的 case/question identity。
- generation 指标按 case 统计，RAGQuestEval 按 question 统计，多问题 case 以 parent case 作 cluster；单 profile 只发布 descriptive cluster-bootstrap 95% CI，不做假设检验。

**RAGQuestEval 口径**：问题只从 exact ground-truth reference 生成；reference QA 与 prediction QA 分别使用 reference 和 evaluated prediction 作为 context，source article 不进入 QG 或任一 QA context。空的 conditional denominator 保持为 null/0，而不是伪造为零分。

**真实实验状态**：
尚无通过 `numeric_samples.json` 重算认证并同步到文档的 M9b2 真实结果；
本 marker 不发布任何真实生成、QG、QA、BERTScore 或 RAGQuestEval 数字。
真实实验继续受显式授权、append-only cache、固定 case 上限与 ignored-artifact 边界约束。

**入口**：
```bash
uv run python scripts/run_crud_generation.py --status
uv run python scripts/run_crud_generation.py --dry-run
# 付费 chat：显式加 --allow-paid-provider，并按 stage 的 prerequisite 顺序执行
# BERTScore：另需显式加 --allow-model-download；finalize 全程离线
```
<!-- END M9B-GENERATION-STATUS -->

### 6.3 LLM-judge 设计

- **区分两条 judge 链路**：当前 TiDB pooled qrels 的生成、验证与相关性判断都由已配置的 `gpt-5.6-sol` 完成，必须标为同模型 self-agreement 与合成标签；M9 的生成侧系统对比仍要求判官 ≠ 被评估生成模型（例如 Qwen 生成 → DeepSeek/Kimi 评判）。
- **RAGQuestEval 分成 M9a 与 M9b**：M9a 只对已经存在的答案 pair 做纯内存评分，固定 `无法推断` 的有限归一化、near-sentinel 分类、多重集 token overlap，以及论文全问题分母与历史代码条件分母；不生成问题、不提交 `quest_gt`。M9b 才负责付费 QG/QA 编排，所有 generated question/answer 与 `quest_gt` 只留 gitignored 本地 artifact。
- **无法回答的哨兵字符串是 `无法推断`**。先做 Unicode NFKC、空白、平衡外层引号和句末标点的有限归一化，再区分 exact/normalized/near；例如「根据材料无法推断具体日期」是 near、仍按可回答文本处理，并在报告中单独计数。
- README 里写明 judge 模型 + temperature + 日期。judge drift 会无声地作废跨轮次对比。

### 6.4 如果各臂打平怎么办

这是需要**提前决定**的分支，不能等测完再想：

1. **先看 CI 是否重叠**。✅ **这条已经用上了**：dense 单臂 vs BM25 的 95% CI 大幅重叠（+2.12pp，CI [−0.88, +5.12]pp），McNemar p = 0.199 —— 已如实报告「在本语料上 **dense 单臂**未带来统计显著提升」。这是第二个诚实的负结果，和 98.0%→75.9% 那个同样值钱。**注意它没有连坐 hybrid**：RRF 对 BM25 是 +4.00pp、p = 1.6e-03，显著。打平的是单臂，不是融合。
2. **切分难度桶再看。** ✅ **已于 2026-08-20 跑完**：按实际 gold 数而非任务名分成 809 / 802 / 783 条。结果不是笼统的「dense 随 arity 增大而全面领先」，而是**指标特异**：跨 arity 可比的 `hit@1` 上，dense vs BM25 为 +1.98pp / +0.37pp / +4.34pp，配对 95% CI 分别为 [−0.99,+4.94] / [−3.37,+4.36] / [+0.64,+8.17]pp，12 项二元 family 经 Holm 后 p=1.000 / 1.000 / 0.176，均不显著；`ALL-gold@10` 上，arity=2 为 **+6.48pp**（95% CI [+4.24, +8.73]pp，Holm p=2.81e-07），arity=3 为 **+15.71pp**（[+12.90, +18.52]pp，Holm p=1.66e-27）。**dense 强在把整套证据捞进 top-10，不强在把任一证据排第一。** 同时，1doc 上选出的 RRF k=10/depth=100 到 arity=3 的 ALL-gold@10 相对 dense **−5.11pp**（95% CI [−7.02, −3.19]pp，6 项 Holm p=1.67e-06），而 hit@1 −0.77pp（95% CI [−3.58,+2.04]pp，Holm p=1.000）不显著；所以退化不能泛化成「融合整体伤害多文档查询」，但一套融合配置通吃全部 arity 也不再受数据支持。复现：`scripts/compare_dense_bm25.py` section 5。
3. **换看 nDCG@10 和 MRR@10**。R@1 打平不代表排序质量打平。
4. **绝不通过换评测集来制造差异**。若换了，必须两个集都报。

---

## 7. 里程碑路线图（业余时间，1 天 ≈ 3 小时有效工时）

已建成：`io_utils.py`、`tokens.py`、`lexical/{analyzers,bm25}.py`、`chunking/markdown.py`、`eval/metrics.py` + 4 个测试文件（~559 行）。

| # | 里程碑 | 天 | 交付物（artifact） | 数字（number） |
|---|---|---|---|---|
| **M0** | 仓库卫生 + CI | **1.0** | `.github/workflows/ci.yml`（ubuntu + windows 双 leg，windows 不设 PYTHONUTF8）；ruff 加 PLW1514；删除 `_research_*.py` / `_enc_test.txt`；`DATA_LICENSE.md` | CI 绿；ruff 0 error；测试通过率 100% |
| **M1** | 检索核心 Protocol 与显式组合根 | **1.5** | 现有 `retrieval/online.py`、`store/base.py`、`providers/*`、`lexical/*` 中的窄 Protocol，以及 `OnlineSettings` 和显式 adapter/组合根；没有已交付的 `registry.py`、`config.py` 或 `experiments/*.yaml` | BM25、dense、sparse、fusion、rerank 可注入并分别测试；配置语义由 frozen settings 约束 |
| **M2** | **Milvus store + provider 客户端 ✅ 2026-08-22** | **1.0** | ✅ `store/{base,milvus}.py`（vendor-neutral Protocol + 惰性导入的 pymilvus 适配器）；✅ `providers/{http,embedding,rerank,cache}.py`（One Hub/OpenAI-compatible transport、7 次长退避、严格响应校验、断点缓存与 provenance sidecar） | ✅ 适配器有 fake-client 契约测试（默认环境不 import pymilvus）；✅ 真实 Milvus Lite 集成通过 `scripts/verify_milvus_store.py`（schema 幂等、完整行 upsert、dense/sparse 两臂、fetch 定序、alias 切换、close 后重开）；`pymilvus==3.0.1` 收进可选 extra |
<!-- BEGIN M3-TIDB-EVAL-STATUS -->
| **M3** | **TiDB 全量索引 + 合成 pooled qrels + 离线质量 ✅ 2026-08-26** | **1.5** | ✅ `ingest.py` + `scripts/build_index.py` + `scripts/query_index.py`；✅ `scripts/build_tidb_{queries,pool,qrels}.py`；✅ `evaluate_tidb_retrieval.py` | 450 篇 evergreen → **1,832 chunks**；**490 pairs / 980 queries / 24,525 pooled candidates**。overall binary nDCG@10：BM25 0.701 / dense 0.793 / RRF 0.796 / rerank 0.914；以 245 个 source cluster 为重采样单位的 95% CI + 两个预声明 4-test 双尾 source-cluster bootstrap/Holm family。无上游人工 gold；本地 exact RRF 不代表服务端 hybrid_search |
<!-- END M3-TIDB-EVAL-STATUS -->
<!-- BEGIN H-HYBRID-MRL1024-M4 -->
| **M4** | 混合检索 + RRF | ~~1.0~~ **0.5** | 客户端 char-bigram → SPARSE_FLOAT_VECTOR（IP）；`hybrid_search` + RRFRanker | **离线部分已完成 2026-08-19，分层于 2026-08-20 补齐**（`scripts/compare_dense_bm25.py` + `retrieval/fusion.py`）：1doc hybrid **79.9 / 0.881** vs BM25 75.9（p=1.6e-03，显著）vs dense 78.0（Holm p=0.231，不显著）；多证据上 dense 的完整证据召回更强，而同一 RRF 在 arity=3 ALL@10 比 dense 低 5.11pp。在线链路已用客户端精确 RRF 落地（`retrieval/online.py`，两臂各 100 → 本地 RRF k=10/depth=100）；H（A+dense-1024）已独立离线运行：1doc R@1 **79.6% [76.8%, 82.4%]** / MRR@10 **0.878 [0.860, 0.894]**，同次重建的 G R@1 **79.9% [77.0%, 82.6%]**。H−G 只作双尾 difference test，I/J 仍绑定 G；M4 剩把融合搬进 Milvus 服务端并复现数字（服务端 tie 顺序与本地 doc-id tie-break 不保证一致，属优化路径而非默认精确路径） |
<!-- END H-HYBRID-MRL1024-M4 -->
| **M5** | **Rerank + 深度消融：离线部分 ✅ 2026-08-21** | **1.5** | ✅ `providers/rerank.py` + `providers/cache.py` + `eval/rerank.py` + `scripts/evaluate_rerank.py`；✅ 在线 stage 已接入 `retrieval/online.py`（**请求深度 100 / 应用深度 50** 分开建模） | 冻结 G（dense-4096 hybrid）统一评分 239,400 对：arity=1 `hit@1` +6.06pp（Holm p=0.0003），arity=3 `ALL@10` +7.15pp@50 / +7.41pp@100；arity=2 无显著增益；top-100 未显著胜 top-50。4B 对照因 provider 无模型而取消 |
| **M6** | ~~MRL 消融（原创）~~ **✅ 已完成 2026-08-19** | ~~1.0~~ **0.3** | `scripts/probe_mrl_quality.py`：4096→64 七档 + 逐维方差 + 配对检验 | 存储 93.1 MB → 23.3 MB（1024 维）；**1024 维 −0.50pp 不显著（p=0.684），仅 64 维显著劣化 −4.25pp（Holm p=0.001†；蒙特卡洛地板标记）** |
<!-- BEGIN M7-CHUNK-SWEEP -->
| **M7** | **TiDB chunk sweep（source-level known-item）✅** | **0.5** | 256/400/800；dense-4096 + char-bigram BM25 + exact RRF k=10；RRF 先于最终 arm 的 source collapse；不启用 rerank/chat | `tidb-chunk-t256-h384-v1` 2,802 chunks；origin-source MRR@10 0.870 [0.846, 0.893]<br>`tidb-chunk-t400-h600-v1` 1,832 chunks；origin-source MRR@10 0.862 [0.839, 0.884]<br>`tidb-chunk-t800-h1200-v1` 1,019 chunks；origin-source MRR@10 0.850 [0.824, 0.874]<br>主问题是 `400-origin exploratory known-item source retrieval sweep`，不是 passage-level answer relevance |
<!-- END M7-CHUNK-SWEEP -->
<!-- BEGIN M8-ROADMAP -->
| **M8** | **服务层 + HTTP 延迟/QPS ✅ 2026-08-28** | **1.5** | ✅ FastAPI + 单文件静态前端（阶段耗时条）+ `scripts/serve.py`；✅ `scripts/bench.py` + numeric-samples 认证 + `sync_m8_docs.py` | profile `tidb-docs-exact-rrf10-cached-query-no-rerank-v1`（cache-backed），正式请求 980/980 成功，HTTP p50/p95/p99 **197.4/228.3/244.4 ms**，**4.97 QPS**，并发 1；本机结果，不是生产 SLA，不与其他 profile 混写 |
<!-- END M8-ROADMAP -->
<!-- BEGIN M9B-GENERATION-ROADMAP -->
| **M9** | **生成侧评估（M9a ✅ / M9b1 编排合同 ✅；真实实验待授权）** | **2.0** | **M9a**：`eval/metrics_gen.py`、`eval/quest_eval.py`、合成测试、aggregate-only Table 8 evidence、独立文档同步；**M9b1**：纯内存合同、离线优先 CLI、QG/QA/生成 stage DAG（event_summary / questanswer_1doc）与 text-free finalizer | M9b1 已冻结 cache/provenance/privacy 合同；真实 provider 实验与结果数字不在本阶段发布 |
<!-- END M9B-GENERATION-ROADMAP -->
| **M10** | **TiDB 第二后端：离线合同已实现 / live smoke 待外部实例** | **1.0** | ✅ `store/tidb.py`：原生 VECTOR/HNSW dense、精确 sparse postings、alias registry 与合成合同测试；✅ README「为什么 TiDB 的文档没有跑在 TiDB 上」；⏳ TiDB Cloud schema/index、`EXPLAIN annIndex`、两臂排序、round-trip、publish/reconnect 真机验证 | 只有 live smoke 通过后才声称同一份 Protocol 双后端跑通 |
| **M11** | **检索可观测性 + Bad Case 归因 ✅ 2026-09-02** | **1.0** | ✅ 脱敏 trace start/end 合同与 FastAPI 生命周期接入；✅ 离线认证 frozen TiDB runs/qrels 并发布 aggregate-only `bad_case_report.json`；不依赖 provider/Phoenix/OTEL。运维 trace 只保留脱敏计数、耗时与 fingerprint，不含 query 或文档内容 | 按系统汇总 recall failure / ranking failure / success；HTTP 无生成阶段，generation 固定 `not_evaluated` |
| **M12** | 公网部署 + README 定稿 | **1.0** | Zilliz Cloud Free 集群 + 公网 demo 链接；README 首屏定稿 | 端到端在线可点 |

> M11 已完成 provider-free retrieval/ranking 可观测性与离线 aggregate-only 归因；不声称 Phoenix/OTEL UI 或公网部署已验证。报告数字只从 gitignored `indexes/tidb/eval/bad_case_report.json` 读取，未经专用同步器认证不写入 README。generation=`not_evaluated`（当前 HTTP 服务没有 generation stage）。

<!-- BEGIN M8-STATUS -->
> M8 在线 pipeline、FastAPI/静态前端与 HTTP benchmark 已完成。当前认证 headline 只绑定 `tidb-docs-exact-rrf10-cached-query-no-rerank-v1`：p95 **228.3 ms** / **4.97 QPS**；provider-included 与 cache-backed profile 必须分表，不能混成一个性能数字。
<!-- END M8-STATUS -->

<!-- BEGIN M8-RESUME-EVIDENCE -->
> M8 复现证据：`scripts/bench.py` 已对 cache-backed HTTP profile `tidb-docs-exact-rrf10-cached-query-no-rerank-v1` 完成 980 次正式请求；原始 numeric samples 与聚合报告均在 gitignored `indexes/tidb/eval/`，由 `sync_m8_docs.py` 校验 SHA-256 并离线重算。
<!-- END M8-RESUME-EVIDENCE -->

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
│   ├── lexical/
│   │   ├── analyzers.py            # ✅ char_ngram / jieba_words / union
│   │   ├── bm25.py                 # ✅ Okapi BM25，可插拔 analyzer
│   │   └── sparse.py               # ✅ BM25 权重 -> 稀疏向量
│   ├── chunking/
│   │   └── markdown.py             # ✅ 两阶段 header-aware
│   ├── retrieval/
│   │   ├── fusion.py               # ✅ RRF（可加权、可指定融合深度）
│   │   ├── online.py               # ✅ 在线检索编排
│   │   └── adapters.py             # ✅ provider 与在线 Protocol 的接缝
│   ├── store/
│   │   ├── base.py                 # ✅ VectorStore / ChunkRecord Protocol
│   │   ├── milvus.py               # ✅ Lite / Standalone / Zilliz 适配器
│   │   └── tidb.py                 # ✅ 离线 TiDB Vector/HNSW + sparse postings 适配器
│   ├── providers/
│   │   ├── http.py                 # ✅ One Hub JSON transport：UA / 长退避 / Retry-After
│   │   ├── embedding.py            # ✅ embedding prompt / 批缓存 / provenance sidecar
│   │   ├── rerank.py               # ✅ Qwen3 rerank 请求与严格响应校验
│   │   ├── chat.py                 # ✅ LLM chat completion 与严格响应校验
│   │   └── cache.py                # ✅ gitignored 配对分数缓存 + provenance sidecar
│   ├── eval/
│   │   ├── metrics.py              # ✅ R@k / MRR / nDCG / ALL-gold / bootstrap
│   │   ├── retrieval.py            # ✅ BM25/dense run 与共享逐查询指标
│   │   ├── rerank.py               # ✅ 窗口语义 / 指纹 / 配对检验族
│   │   ├── tidb_quality.py         # ✅ pair-aware TiDB 指标 / CI / bootstrap-Holm / 分层
│   │   ├── metrics_gen.py          # ✅ provider-free 生成指标合同与 lazy BERTScore
│   │   ├── quest_eval.py           # ✅ provider-free RAGQuestEval 与显式 denominator
│   │   ├── hybrid_mrl1024.py       # ✅ H 的离线 MRL 评估
│   │   ├── tidb_chunk_sweep_evaluation.py # ✅ M7 离线 chunk sweep
│   │   └── tidb_bad_cases.py       # ✅ M11 aggregate-only retrieval attribution
│   ├── service/
│   │   ├── app.py                  # ✅ FastAPI
│   │   ├── observability.py        # ✅ provider-free 脱敏 trace contract
│   │   └── static/index.html       # ✅ 单文件静态前端 + 阶段耗时条
│   └── ingest.py                   # ✅ 幂等：path 稳定键、sha256 变更检测、文档级 delta
├── docs/
│   ├── evidence/
│   │   └── crud_rag_table8_v3.json # ✅ 仅含 Table 8 aggregate-only 历史证据
│   ├── why-not-tidb.md             # 「为什么 TiDB 的文档没有跑在 TiDB 上」
│   ├── why-not-pgvector.md         # 4096 > 4000 的 96 维之差
│   └── bad-cases.md                # 归因表
├── scripts/                        # 评测、构建、服务与文档同步脚本
│   ├── corpus_stats.py             # ✅
│   ├── compare_dense_bm25.py       # ✅ dense/BM25/RRF 与 arity 分层
│   ├── evaluate_rerank.py          # ✅ top-100 断点评分 + top-50/100 离线分析
│   ├── evaluate_tidb_retrieval.py  # ✅ 冻结 runs/qrels 的 pair-aware 离线质量报告
│   ├── build_index.py              # ✅ manifest → chunk → index 发布
│   ├── bench.py                    # ✅ p50/p95/p99 + QPS
│   ├── sync_m9a_docs.py            # ✅ aggregate-only 证据与生成指标合同同步
│   ├── run_crud_generation.py      # ✅ M9b1 known-context 生成/QG/QA/semantic 编排
│   ├── sync_m9b_docs.py            # ✅ M9b1 合同状态与路线图同步
│   ├── evaluate_tidb_bad_cases.py  # ✅ M11 离线 aggregate-only 归因
│   ├── verify_tidb_store.py         # ⏳ opt-in TiDB Cloud synthetic smoke
│   └── sync_quality_gate_docs.py   # ✅ 独立 JUnit 质量门禁同步
├── tests/                          # ✅ provider-free、离线评测与集成回归测试
```

**语料目录（`crud-rag-subset/`、`tidb-rag-curated/`）保持在 `.gitignore` 里，一个字节都不提交。**

---

## 9. 成本预估

| 项目 | 用量 | 单价 | 成本 |
|---|---|---|---|
| Embedding 全量一遍（两个语料） | 4.88M tokens | ¥0.28/M | **¥1.37**（≈ $0.19） |
| ~~Embedding × 15 遍（MRL 消融 + 重建）~~ | ~~73M tokens~~ | — | **¥0**（实测 `dimensions` 即前缀切片，全部维度档共用一次嵌入，见 §13） |
| Rerank top-100 已完成 sweep | 2,394 queries × 100 docs = 239,400 pairs | One Hub 实际价格未核实 | **不写金额**；不能套用 SiliconFlow 价格 |
| top-50 深度消融 | 复用上述前 50 个配对分数 | — | **¥0 额外调用** |
| 4B vs 8B | 当前 relay 无 4B | — | **取消** |
| RAGQuestEval — QG（M9b，本地 ignored artifact） | ~2,400 题 × ~800 tok | 按 LLM 计费 | **≈ ¥30**（⚠️ 估算，不提交） |
| RAGQuestEval — QA（M9b，中档判官，本地 ignored artifact） | ~2,400 × N 题 × ~600 tok | 按 LLM 计费 | **≈ ¥20 / 配置**（⚠️ 估算，不提交） |
| 生成侧 LLM 输出（4 个任务 × ~2,000 行 × ~250 tok） | ~2M tokens 输出 | 按 LLM 计费 | **≈ ¥40**（⚠️ 估算） |
| BERTScore / BLEU / ROUGE | 本地 CPU | — | **¥0** |
| Milvus Lite / Standalone (WSL2) | 本地 | — | **¥0** |
| Zilliz Cloud Free | 5 GB / 2.5M vCU/月 | 免费 | **¥0** |
| GitHub Actions（公开仓库） | — | 免费 | **¥0** |
| M11 脱敏 trace contract + 离线归因 | 本地、provider-free | — | **¥0** |
| **总计（含全部消融，保守）** | | | **≈ ¥200–250（约 $28–35）** |

存储侧：1,832 TiDB chunks @ 4096 维 float32 ≈ **30.0 MB**；MRL-1024 ≈ **7.5 MB**；全部落在 Zilliz Free 的 5 GB 里，**约 175 倍余量**。

**成本与数据边界三条铁律**：① 重排分数按 `(qid, docid)` 缓存并由 sidecar 绑定 model/input provenance，
但它们是语料派生物，**只留 gitignored 本地目录、绝不提交**；② QG/QA/生成文本、`quest_gt` 与
对应 cache 同样只留 gitignored 本地目录，不因 M9b 完成而改变许可边界；③ CI 只跑无需 API key、
无需派生缓存的单元测试，付费指标只允许手动触发并断点续跑。

---

## 10. 明确不做的事

| 不做 | 一句话理由 |
|---|---|
| **不用 LangChain / LlamaIndex 做地基** | 当前固定检索流程使用窄 Protocol 即可表达，优先减少运行时依赖和评估中的隐式行为 |
| **不上 DSPy** | prompt optimizer 是消融矩阵里的**不受控混杂因素**：分高的配置可能只是被优化得更狠。要用就单独放到生成阶段实验、检索侧冻结 |
| **不上 GraphRAG / Self-RAG / CRAG** | 每一个都能单独吃掉 5 周，而且和「检索评估」这条主线正交。README 提一句「已知但本项目未覆盖，见 Roadmap」即可 |
| **不做微调（embedding / reranker / LLM）** | 需要标注数据与 GPU 时长，且会让「我的评估集有效吗」这条主线被稀释 |
| **不用 pgvector** | 索引维度 4096 > halfvec 上限 4000，物理上做不到原生索引。写成 `docs/why-not-pgvector.md` 反而是加分项 |
| **不把 TiDB 做主存储** | Vector Search 仍是 public preview，HNSW 依赖 TiFlash；Full-Text Search 仍 early-stage/region-gated 且 tokenizer 语义不同；TiDB adapter 只有离线合同，live smoke 尚未完成 |
| **不用 Qdrant Cloud 免费层挂公网链接** | 在本次验证范围内**没有找到一手官方页面支持“闲置 1 周挂起、4 周删除”这一精确说法**，因此不把它写成事实。若使用 Qdrant Cloud，部署前必须按当前计划条款核对；否则使用可重建的 Milvus/Zilliz 或自托管方案 |
| **不用 RAGAS 做主力评估框架** | 2026-02-24 起无提交、559 open issues，而 deepeval/langfuse/opik/phoenix 当天都在发版；且 `adapt()` 默认 `adapt_instruction=False`，中文提示词是半英半中 |
| **不开数据库内置中文分词器** | Milvus `chinese` analyzer = jieba `cut_for_search` = 你实测最差的 73.4%，比 char bigram 低 2.5 分 |
| **不提交任何语料字节（含「小样本」）** | CRUD_RAG **根本没有 LICENSE 文件**（Apache badge 只是 README 里的 shields.io 图片，GitHub API 报 `license: None`），8 万篇新闻无出处无授权声明；TiDB 文档是 CC BY-SA 3.0，你的 chunk 输出属于 Adaptation |
| **不用 Docker（本阶段）** | 本机没装；Milvus Lite 纯 Python；M11 只交付 provider-free trace contract 与离线归因，不把 Phoenix UI 作为已验证依赖。真需要 Standalone 时进 WSL2 |
| **不做多租户 / RBAC / 审计** | 没有任何编排框架白送这些，它们来自数据库和应用层；且 Dify 的许可证明确禁止未授权的多租户运营 |
| **不做移动端 / 复杂前端** | 优先展示阶段耗时和检索证据，控制前端维护成本 |

---

## 11. 技术摘要与证据边界

### 实现与实验摘要

**① 中文 RAG 评估框架与语料重建。** 发现 CRUD-RAG 官方 500 篇子集已饱和——40 行纯标准库字符 bigram BM25 即达 **R@1 98.0%、MRR@10 0.990**，任何检索配置均近满分、消融表无区分度；定位根因为语料规模不足与问句-证据表层重叠。从原始数据去重扩展至 **5,681 篇**干扰语料后 R@1 降至 **75.9%**、MRR@10 **0.857**，释放 **22 个百分点**可优化空间，并将主指标由已饱和的 R@5 改为 **R@1 / MRR@10 / nDCG@10**；全部指标附 95% bootstrap 置信区间，配置间比较对连续指标用**配对 bootstrap**、对二元指标用**精确 McNemar 检验**，全表经 **Holm-Bonferroni** 多重比较校正，并同时报逐查询**胜/负/平**计数。

**② 中文词法检索方案实测选型。** 对比 jieba 精确模式（R@1 **74.8%**）、jieba 搜索模式（**73.4%**）与字符 bigram（**75.9%**）；jieba+bigram 并集在 1doc 上达 76.4% 但汇总的 ALL-gold@10 反而略低（85.3% vs 85.4%）、索引构建耗时 **3.1 倍**，最终选定字符 bigram。进一步实测**向量数据库内置 jieba analyzer 默认即为搜索模式**，遂将词法臂移出数据库，以客户端预计算 BM25 权重作为 SPARSE_FLOAT_VECTOR（`metric_type=IP`）喂入，数据库承担两臂检索，当前在线路径在客户端执行 exact RRF。

**③ 面向技术文档的两阶段分块策略。** 针对 500 篇 TiDB 中文文档（**3,309** 个代码块 / **5,262** 行表格），纯标题切分导致 **63.5%** 分块 <100 tokens、最大块 **16,111** tokens；改为「标题切分 → 掩码代码块与表格 → 按 target=400 合并小块 / 拆分大块」，得 **1,832** 块，p50 **371** / p90 **734**，欠长块降至 **5.0%**，代码块破损 **0** 例；并给出 256/400/800 的分块尺寸-召回曲线。

**④ 检索栈与成本/性能工程。** 基于 Qwen3-Embedding-8B 构建 dense + 字符 bigram BM25 双臂检索与 RRF 融合，在 5,681 篇语料 / 800 条单证据查询上 **R@1 由 BM25 基线 75.9% 提升至 79.9%（+4.00pp，65 胜 33 负，McNemar 精确检验 p = 1.6e-03）**；在全部 2,394 条查询上对冻结 hybrid 的 top-100 候选统一调用 Qwen3-Reranker-8B，离线消融显示 top-50 已使单证据 `hit@1` **+6.06pp**（95% CI [+3.34,+8.78]，Holm p=0.0003）、三证据 `ALL@10` **+7.15pp**（[+4.98,+9.32]，p=1.32e-09），而 top-100 无显著额外收益，因此部署候选选 top-50；<!-- BEGIN M8-RESUME-PERFORMANCE -->当前认证的 cache-backed HTTP profile `tidb-docs-exact-rrf10-cached-query-no-rerank-v1`（并发 1，不含 provider 墙钟）端到端 **p95 228.3 ms / 4.97 QPS**。<!-- END M8-RESUME-PERFORMANCE -->本项目记录 Qwen3-Embedding-8B 的 **MRL 降维质量曲线**：4096→1024 维存储由 **93.1 MB 降至 23.3 MB（−75%）**，R@1 **78.0%→77.5%（−0.50pp，配对 bootstrap p=0.684，Holm 校正后不显著）**；降至 128 维（−97% 存储）仍无显著损失，**64 维起显著劣化（−4.25pp，Holm p=0.001†；蒙特卡洛地板标记）**——官方技术报告未发布此数据。

**⑤ 用配对检验推翻自己的点估计，并据此改路线。** 8B 稠密检索相对 40 行纯标准库 BM25 名义领先 2.1pp，配对检验后判定**不显著**（95% CI **[−0.88, +5.12]pp**，McNemar 精确 p = **0.199**）；进一步用 R@1 列联表（both 538 / 仅 BM25 69 / 仅 dense 86 / 都不中 107，φ = 0.455，**并集 oracle 上限 86.6%**）判定两臂**互补而非冗余**，据此把主线从「换更强的单臂」改为「融合」，离线 RRF 兑现 **75.9% → 79.9%（p = 1.6e-03）**。同一批实验还显示**等权 RRF 相对 dense 单臂不显著（39 胜 24 负，Holm p=0.231），只有加权 0.3/0.7 显著（16 胜 4 负，p=0.047）**——赢在少破坏，不在多修好。

> ④ 中的 HTTP 性能句由 `M8-RESUME-PERFORMANCE` 区域从认证 numeric samples 生成；不要手写或跨 profile 混用。M4 的融合、M5 的离线 rerank 深度消融、M6 的 MRL、M8 的在线 pipeline/本机 HTTP benchmark，以及 M11 的 provider-free trace contract 与离线 retrieval attribution 均已完成；①②③⑤ 每一个数字也都已实测。M11 不包含生成阶段，generation 状态固定为 `not_evaluated`。
>
> ⚠️ **不要把「dense 打赢 BM25」写进任何一条要点。** 实测 +2.12pp、95% CI [−0.88, +5.12]pp、
> McNemar p = 0.199 —— **在本语料上不显著**。可以写的是融合后对 BM25 的 +4.00pp（显著），
> 以及「融合相对 dense 单臂的 +1.87pp 同样不显著（Holm p = 0.231）」。
>
> ⚠️ **不要把降维无损归因于 MRL 训练。** 实测跨文档逐维方差首尾比 1.047（平的），
> 没有信息前置聚集的证据。可写的是行为（掉多少 pp），不是机制。机制解释应作为待验证假设，而不是实测结论：
> 「更可能是向量内在维度远低于 4096，前缀截断近似随机投影；JL 界在 n=5,681 时约 200 维，
> 与实测 128–256 的拐点吻合。」


---

## 12. 踩坑清单

| 坑 | 后果 | 修法 |
|---|---|---|
| **Windows 默认 cp936** | 裸 `open()` 读项目自己的中文文件直接 `UnicodeDecodeError`；`print()` 中文抛 `UnicodeEncodeError`。文件系统编码却是 utf-8（不对称） | 全项目走 `io_utils.py`；ruff 开 **PLW1514** 禁裸 `open()`；CI 加 `windows-latest` leg 且**故意不设** `PYTHONUTF8`/`PYTHONIOENCODING`（设了这条腿就废了）。⚠️ 注意 GH Actions 的 windows runner 大概率是 cp1252 不是 cp936，它验证的是「非 UTF-8 默认」而非你的具体 codepage |
| **`rouge_score` 默认 tokenizer 删光中文** | `tokenize('近千家展商参与了UDE2023博览会')` → `['ude2023']`；好改写打 1.0000、无关句打 0.0000，全靠共享拉丁子串。**产出一整张看起来能发表的噪声表** | 必须传 `tokenizer=lambda t: list(jieba.cut(t))`；上线前用一对「好改写 / 无关句」做判别力自检 |
| **sacrebleu 默认 `13a`；`zh` 是字符级不是词级** | 数字内部自洽但与 CRUD-RAG 表格不可比 | M9a 已冻结独立 token-level 合同；sacrebleu 不再是 canonical runtime dependency，若做外部稳健性对照须单独命名并隔离 |
| **CRUD-RAG 的 `bertScore` 不是 BERTScore** | 是 `text2vec-base-chinese` 句向量余弦；把你的真 BERTScore 和它放同列 = 无意义对比 | 两列分开命名并注明 |
| **CRUD-RAG 默认除掉 BLEU 的 brevity penalty** | 他们的「bleu」不是标准 BLEU，短生成不受惩罚 | M9a 同时报标准 BP 与明确命名的 no-BP 兼容列 |
| **论文 Eq.(2) 与代码的 precision 分母不一致** | 代码版（两次过滤后 `np.mean`）系统性偏高 | M9a 同时报告论文全问题分母与历史代码条件分母 |
| **`@catch_all_exceptions` 返回 `None` + `or 0.0`** | 崩掉的 ROUGE 变成 0.0 混进头条均值（BLEU 因返回 5 元组反而会 TypeError——失败模式还不一致） | 独立实现显式失败计数，任一行失败拒绝出均值 |
| **`Query:` 后面那个空格** | 模型卡自相矛盾（Python helper 无空格 / TEI curl 有空格）。选错 = 全库向量与线上查询前缀不一致，**索引完之后改不了，只能重嵌入** | 以 `config_sentence_transformers.json` 为准（**无空格**），定义成常量，永不改 |
| **文档侧误加 instruct 前缀** | 模型是非对称的，document prompt 是空串。两边都加会静默掉几个点 R@1，**不报错** | 文档原文入库 |
| **`padding_side` 不是 `'left'`（自建推理时）** | 右 padding + `hidden[:, -1]` → 短样本拿到 pad token 向量，全错且无异常 | 用模型卡的 `last_token_pool()`（按 attention_mask 分支） |
| **DeepInfra `normalize` 默认 false** | 未归一化的 4096 维向量进 cosine 索引 = 排序错误、无报错。SiliconFlow 干脆没文档说归一化默认值 | 第一次响应就 `np.linalg.norm()` 自检 |
| **只发布集合、不发布词表** | 查询向量的 term index 只在建库那份词表下有意义；换一份就打到恰好占位的词上，**排序静默出错**。而词表默认只活在构建进程的内存里 | 发布路径把 vocabulary+IDF 写成 `sparse_index.json`，查询端先比对 `state.json` 的 fingerprint 再启动；**只有发布成功才写**，否则查询端会加载到没有在线集合与之对应的词表 |
| **把多个 `os.replace()` 当成 bundle transaction，或检查 writer lock 后无锁读取** | 第二个文件替换失败会留下新旧混合 bundle；检查与读取之间也可能被另一个 publisher 换包，使报告认证基于不同版本的输入 | canonical writer 先拿各自 operation lock，再只在最终发布阶段拿 `eval/.artifacts.lock`；evaluator / docs sync 在 load → 重算认证 → publish 全程持有共享锁。`replace_files()` 遇 Python 异常会逆序回滚，report/state marker 最后发布；这只是 exception safety，**不声称进程强杀下的 crash-atomic transaction** |
| **给稀疏字段挂 `FunctionType.BM25`** | Milvus 会重新分词并**覆盖你预计算的向量** | 用 `SPARSE_INVERTED_INDEX` + `metric_type="IP"`，**不挂** BM25 function。Qdrant 上的对称坑：权重里已含 IDF 就别开 `idf` modifier，否则双重计算 |
| **Milvus Lite 的 BM25 IDF 是 segment 局部的** | Lite 的分数在 Standalone/Zilliz 上复现不出来，指标静默失真 | 走客户端稀疏向量（本方案已规避）；若非要用服务端 BM25，出指标的那一遍只在 Standalone 跑 |
| **Milvus Lite 对 data_dir 加文件锁，单进程** | 并行 eval worker 会死锁或报错 | 每个 worker 独立 data_dir，或 eval 跑 WSL2 Standalone |
| **Milvus Lite 在 Windows + Python 3.13 未经上游 CI 验证** | 上游 CI 只覆盖 Windows+3.10 和 Linux+3.10~3.13 | **第一件事**做 10 行冒烟测试（见 §13），5 分钟去掉整个方案的最大风险 |
| **把 SiliconFlow 限流/价格套到 One Hub** | 会把 relay 的 429 误判成配额，并给出无法核实的成本与墙钟承诺 | 只报告实测：429 是上游负载饱和；显式 UA；7 次长退避并尊重 Retry-After；实际价格单独核对 |
| **重排输入远大于 embedding** | 消融矩阵会反复付同一笔钱 | 统一评分 top-100 一次，top-50/100 离线切窗；分数缓存保持 gitignored，绝不提交 |
| **`无法推断` 是精确字符串比较** | 判官回「无法推断。」被算作「答上了」，静默抬高 recall | 归一化后再比，并记录近似哨兵率 |
| **`bert-score` 0.3.13 停在 2023-02-20** | 整套栈里最可能装不上 Python 3.13 的包 | **设计指标表之前先装**；必要时 pin transformers |
| **CRUD_RAG requirements 装不上** | pin 了 `llama_index==0.9.32` / `langchain==0.1.4` / `pymilvus==2.3.3`，全是 namespace 拆分前版本；上游也没有代码许可证 | **不要安装、复制或移植上游代码**；M9a 只依据公开指标定义做独立实现，并保留 aggregate-only 来源证据 |
| **`corpus_manifest.jsonl` 的 `id` 是内容哈希** | `id == git_blob_sha1` 500/500，内容一变 id 就变，不是稳定身份 | 稳定键 = `path`（500/500 唯一）；变更检测 = `sha256`；chunk id = `hash(path, ordinal, chunk_text)` 保证重跑是 upsert 不是重复插入 |
| **gRPC 遵循 `HTTP_PROXY`，Milvus Lite 走的正是回环 gRPC** | 导出了代理的 shell 里，本机 Lite 连接被路由到代理，报 `code=2, illegal connection params or server unavailable`——**读起来像服务器没起来，其实 TCP 能连、握手被劫**。实测 `GRPC_ENABLE_HTTP_PROXY=0` 单独设**无效**，起作用的是绕行列表 | opt-in 校验脚本自己把 `127.0.0.1,localhost` 加进 `no_proxy`/`NO_PROXY`（见 `scripts/verify_milvus_store.py`），不要在库代码里偷改进程环境 |
| **拿 Milvus 的 delete 计数当业务删除数** | Lite 对不存在的主键写 tombstone 并返回 `len(pks)`，删不存在的行不是错误 | 存储层只报「请求了几行、服务端确认了」；`{added, updated, deleted}` 一律来自 manifest diff |
| **把 `code=100` 当成「alias 还没发布」** | 100 被复用于多种「对象不存在」，若传输层故障恰好带上它，就会把一次连接失败读成「首次发布」，然后覆盖一个从未校验过的 collection | 只认消息里明说不存在的措辞（`not exist` / `not found`），码值不单独作数 |
| **合并相邻小节时丢掉后续标题** | 「甲」「乙」两个兄弟小节合成一块，只有「甲」的路径留在 metadata，「乙」的标题在生成永久 chunk id 与向量之前就消失 | 只把**共同前缀**放进 metadata，各自剩余层级物化进被索引正文（`_materialize_sections`）；改这条会改 chunk 数，README 数字须重出 |
| **README 里「企业级」不能只靠功能清单** | 已有 M8 本机 HTTP 延迟/QPS 证据；M11 现在补齐 provider-free trace contract 与离线 retrieval attribution，但仍缺公网 SLA、provider-included profile、Phoenix/OTEL UI 与生成链路验证 | 保持本机 profile 的边界声明；M12 公网部署后另测生产路径；不要把 M11 写成已完成的外部 tracing 平台 |

---

## 13. 验证事项清单

**🔴 阻塞级（做别的之前先做）**

- [x] **Milvus Lite 3.2.0 在 Windows + Python 3.13.5 上的冒烟测试**：已在隔离 `.venv-verify-milvus` 中安装 `pymilvus==3.0.1` + `milvus-lite==3.2.0`，并验证 4096-dim dense、sparse inverted index、`hybrid_search` + `RRFRanker`。结果：**PASS**。提交版脚本：`scripts/smoke_milvus_lite.py`。两个 Windows 特有坑已固化进脚本：① `pymilvus==3.0.1` 的 `[milvus-lite]` extra marker 明确排除了 `win32`，所以 Windows 上必须显式安装 `milvus-lite==3.2.0`；② `MilvusClient.close()` 返回后 `LOCK` 仍可能被 native 后台线程短暂持有，故用 worker subprocess 作为可靠的清理边界。
- [x] **`bert-score` 0.3.13 能否在 Python 3.13.5 + 当前 transformers 上装并跑通**：隔离 `.venv-verify-bert` 安装成功，`bert-score==0.3.13` + `transformers==5.15.0` + `torch==2.13.0` 可 import；`BERTScorer(lang='zh', rescale_with_baseline=True)` 端到端评分也通过。模型下载需可访问 Hugging Face（本机直连超时，使用 `HF_ENDPOINT=https://hf-mirror.com` 通过）。注意 distribution metadata 是 0.3.13，但模块内 `bert_score.__version__` 仍写 0.3.12，这是上游版本字符串不一致，不影响运行。
- [x] **返回的 embedding 是否已 L2 归一化**（无文档）：**已验证 = 是**。6 条长度从 1 字到 270 字的输入，L2 范数全部落在 `1.0 ± 7e-08`（跨输入极差 `6.98e-08`），是 float32 舍入量级而非真实偏差。**索引前无需客户端归一化，COSINE 与 IP 等价。** 复现：`uv run python scripts/verify_embedding_api.py`。
- [x] **relay 的 403 是本机代理造成的，不是 key / UA / 语料内容（2026-09-02 实测）。** `--generate` 的第一次真实调用返回 `HTTP 403`，响应体是 Cloudflare 的 HTML 拦截页（`Server: cloudflare`、`CF-RAY: a3469000189e2847`、标题「访问已被拦截」），**没有** `error code 1010`；未鉴权的 `GET /` 与 `/v1/models` 同样 403，裸 `ping` payload 同样 403，项目 UA / 浏览器 UA / `curl` UA 结果一致——所以既不是鉴权也不是 WAF 对语料内容的规则。根因是 shell 里的 `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:55430`：该代理出口 IP `81.31.232.59` 被 Cloudflare 封禁。**把这两个环境变量从进程环境里清掉后，同一个未鉴权 GET 立即变成 `HTTP 401`（即已到达 relay），真实付费调用随后全部成功。** ⚠️ 排查陷阱：`urllib.request.build_opener(ProxyHandler({}))` 在本机**没有**真正绕过，仍返回 403，据此会误判成「直连也被封」；有效做法是在子进程环境里 `env -u HTTP_PROXY -u HTTPS_PROXY ...`。这与 Milvus Lite 那条相反——那里是要把 `127.0.0.1,localhost` 加进 `no_proxy`，这里是整个 relay 都不能走代理。

**🟡 影响架构决策**

- [x] **`dimensions` 截断是「切片+重归一化」还是训练过的 MRL head**：**已验证 = 切片**。判据不能用绝对阈值，因为该 provider 自身不可重现——同一段文本分两次请求，余弦只有 `0.999931`。以此为噪声底，客户端切片与原生 `dimensions=n` 的余弦在 512 / 1024 / 2048 三档分别是 `0.999932 / 0.999932 / 0.999934`，**全部落在噪声底之上**：切片与原生的差异，不大于同一次调用重跑一遍的差异。排序检验同样通过（20 条近义句，top-1/3/5/10 集合重合 100%，分数扰动仅为相邻名次间隔的 7%）。**结论：只存一份 4096 维向量，MRL 消融（M6）纯客户端切片完成，零额外 API 调用、零额外存储。** 注意这只说明消融**怎么跑**，不说明 1024 维的质量——那正是 M6 要测的。
- [x] **Zilliz Cloud Free 长期闲置是否会被删除/挂起**：官方 [pricing](https://zilliz.com/pricing) 明确 Free 为 5 GB / 2.5M vCUs per month / up to 5 collections；pricing FAQ 明确“suspended”状态会停止 vector database costs，但 storage costs continue until deletion。官方 [limits](https://docs.zilliz.com/docs/limits) 没有写“闲置 N 天自动挂起/删除”，也没有 Free 专属的闲置删除倒计时。官方 [Terms and Conditions](https://zilliz.com/terms-and-conditions) 则保留较宽的服务终止权，并写明终止后“不承担保留已终止集群或备份快照中数据的义务”。结论：**没有证据支持“闲置 4 周必删”，但也不能承诺永久保留**；不要把 Zilliz Free 当唯一的服务可用性保障，部署前保留本地/可重建 artifact，并在控制台确认具体 region/plan 条款。
- [ ] **Zilliz Cloud Free 跑的是 Milvus 2.6.x 还是 3.0.x。** 3.0.0 于 2026-07-29 GA，托管云通常滞后。建好集群后从版本端点确认，不要信营销页。
- [x] **embedding 是否可逐位重现**（新增项，实测发现）：**否**。同一段文本分两次请求，余弦 `0.999931`（≈ 每分量 1e-3 量级）；同一批次内重复则**通常**逐位相同，但三次观测中有一次不是。成因是批式 GPU 推理的规约顺序随 batch 形状/位置变化，不是错误。三条后果：① `(text, model)` 缓存依然正确，但**单元测试只能带容差比较，绝不能断言相等**；② 任何比 `1 - 6.9e-05` 更接近的两个向量在本 provider 上不可区分——这也是判定「切片 vs 训练头」必须以它为基准的原因；③ 上报向量相关数字时要标注它，否则复现者会以为自己配错了。
- [ ] **TiDB Cloud Starter 免费实例长期闲置是否归档/删除。** `serverless-limitations.md` / `serverless-faqs.md` 里 grep 不到任何相关条款——可能确实没有，也可能在文档仓库外的控制台页面。
- [ ] **`pingcap/docs-cn` 的 pinned commit `26f202bc` 是在 TiDB v7.0 之前还是之后。** README 说 CC BY-SA 3.0 是「自 TiDB v7.0 起」适用，确认你的快照落在该声明覆盖范围内。
- [ ] **就 CRUD_RAG 的许可开一个 issue**（无 LICENSE 文件 + 8 万篇新闻无出处）。一个礼貌的 issue 零成本、可能拿到明确答复，且在 402 star 的仓库上本身就是可见的尽职信号。

**🟢 影响指标可信度**

- [x] **你现有的 `paired_bootstrap_test` 实现是否统计正确**：**已于 2026-08-19 审计，实现本身成立。** 三个问题的答案：重采样的是**配对的 query**（逐查询分差作为一个单元重采样，不是对两臂各自自举）；**单尾**（`treatment > baseline`，所以 p=0.60 意味着「没有证据说 treatment 赢」，不是「baseline 赢」）；Holm-Bonferroni 已实现且已用于 MRL 全表。审计另修了两处并新增一处：
  ① **蒙特卡洛分辨率地板**——估计量是 `(count+1)/(resamples+1)`，10,000 次重采样下最小可表示的 p 是 `9.999e-05`。0/10,000 个零分布样本达到观测值时，只能说 add-one **估计停在地板**，不能把它写成“真实 p `< floor`”。经 Holm 后还必须传播地板 provenance。当前统一用 `†` 标记（MRL 的 64 维为 `0.001†`），并明确它不是 `<` 上界；`bootstrap_p_floor()` 与 `holm_floor_flags()` 负责这一口径。
  ② **`observed <= 0` 直接返回 1.0 的短路**——保守、不产生假阳性，但会把一族真值各异的 p 压成同一个 1.0 再喂给 Holm。已改走通用路径；**副作用是 MRL 表 512 维那行的 p 从 1.000 变为 0.684**，README 与本文档已按硬规则 3 重出。
  ③ 二元指标新增 **`mcnemar_exact`**（精确、无下界、无种子、跨机器同值）与 **`win_loss_tie`**。dense vs BM25 的判定即由前者给出。⚠️ 注意 `recall_at_k(..., 1)` 在 2docs/3docs 上不是二元的（会返回 0.5 / 1/3 / 2/3），McNemar 会拒绝它——这是特性不是缺陷，混 arity 的列联表会把「部分得分变化」计成胜负。
- [x] **CRUD-RAG 论文 Table 8 的 baseline 数字**：已对照 arXiv `2401.17043v3` 的 HTML `S4.T8` 与 PDF 第 26 页核对实际引用的四行；固定证据保存在 `docs/evidence/crud_rag_table8_v3.json`，并记录 PDF SHA-256 `2e4ae0cb708fdca9d96bcf8d1c0713dae121195a0a31b78e3132e9ef4fa7db8a`。该文件只含 aggregate-only facts，不含语料或答案；数值仍不能当作本项目实现的 golden test。
- [ ] **GitHub Actions `windows-latest` Python 的实际默认 codepage**。预期是 cp1252（美式 locale）而非你的 cp936——它复现的是「非 UTF-8 默认」，不是你的具体环境。要精确复现 cp936 得强制 locale 或在某个 job 里设 `PYTHONIOENCODING=gbk`。
- [x] **生成指标 canonical 路径是否需要 `rouge-chinese` / `sacrebleu`**：M9a 已冻结 provider-free 的独立 token-level BLEU-4 与 ROUGE-L 合同，并用合成测试固定 clipping、brevity penalty、LCS 和 fail-closed 语义；`rouge-chinese` 与 `sacrebleu` 不再是 runtime 依赖。若未来做第三方数值对照，必须另列 provenance，不能与 canonical 列混用。
- [ ] **DeepInfra 的 Qwen3-Embedding-4B 标价 $0.020/M、比 8B 的 $0.010/M 贵一倍**，这个反常价格可能是促销或过期数据。做预算前在实时页面确认。
- [ ] **英文 instruction 与中文 instruction 在你的中文语料上到底哪个好。** Qwen 基于训练数据来源推荐英文，但那是通用建议不是在 TiDB 文档上的实测。两次跑，同一评测，又一行诚实消融。
- [x] **hybrid 到底能不能在你的 5,681 语料上打赢 char-bigram BM25 的 75.9%。** **能，且显著。** 离线 RRF 融合 dense-4096 与 BM25 两条 run：**R@1 79.9% / MRR@10 0.881**，对 BM25 **+4.00pp**（65 胜 33 负，McNemar 精确 p = **1.6e-03**）。同时**修正了本条此前的一个错误结论**：dense 单臂并没有「赢」——+2.12pp、95% CI **[−0.88, +5.12]pp**、p = **0.199**，**不显著**。当初「两臂接近不等于融合无用」的判断被证实了：列联表 538 / 69 / 86 / 107，φ = 0.455，并集 oracle 上限 **86.6%**；且一臂 rank-1 落空时 gold 在另一臂里 83–88% 落在前 3、掉出 top-100 的是 0.0%。三条工程结论：**融合深度 10 与 100 的逐查询 R@1 逐位相同**（0/800 条 top-1 改变）；k 从 60 调到 10 只动 0.1pp；**唯一有效的旋钮是权重**（0.3/0.7 是全表唯一显著优于 dense 单臂的配置，16 胜 4 负，Holm p=0.047）。复现：`uv run python scripts/compare_dense_bm25.py`（不联网）。⚠️ 四个融合配置是在同一批 800 条上选出又汇报的，最好那行是上界不是泛化估计。
- [x] **2docs/3docs 分层结果。** 已于 2026-08-20 完成。按实际 gold 数分组（809 / 802 / 783）后，dense vs BM25 的 `hit@1` 为 +1.98 / +0.37 / +4.34pp，配对 95% CI 为 [−0.99,+4.94] / [−3.37,+4.36] / [+0.64,+8.17]pp，12 项二元 family 经 Holm 后 p=1.000 / 1.000 / 0.176，均不显著；但 `ALL-gold@10` 在 arity=2/3 分别 **+6.48pp**（95% CI [+4.24,+8.73]，Holm p=2.81e-07）与 **+15.71pp**（[+12.90,+18.52]，p=1.66e-27）。结论是 **dense 强在找齐证据，不强在把任一证据排第一**。1doc 选出的 RRF k=10/depth=100 在 arity=3 的 ALL@10 又比 dense **低 5.11pp**（[−7.02,−3.19]，6 项 Holm p=1.67e-06），而 hit@1 为 −0.77pp（[−3.58,+2.04]，Holm p=1.000），差异不显著；一套 RRF 配置不能直接外推。复现：`uv run python scripts/compare_dense_bm25.py`（不联网）。
- [x] **rerank 能否在 hybrid 的 79.9% 之上再拿到显著增量。** **已于 2026-08-21 在冻结的 G（dense-4096 + char-bigram BM25，等权 RRF k=10/depth=100）上完成。** 对全部 2,394 条 query 的 top-100 候选统一打分一次（239,400 个 pair），再离线比较 top-50/100。arity=1 的 `hit@1` 79.7%→85.8%，**+6.06pp**（95% CI [+3.34,+8.78]，91 胜 42 负，12 项 Holm p=**0.0003**）；arity=3 的 `ALL@10` 79.3%→86.5%@50 / 86.7%@100，分别 **+7.15pp**（[+4.98,+9.32]，p=**1.32e-09**）与 **+7.41pp**（[+5.24,+9.58]，p=**4.93e-10**）；arity=2 没有 efficacy 项通过校正。top-100 对 top-50 的四个 depth family 中没有显著结果，三个 arity 的 `hit@1` 完全相同，故当前部署证据选择 **top-50**。这不是 dense-1024 结果；H 必须另跑、另存 fingerprint。另有未量化 stationarity 限制：评分跨多个时间段续跑，query 顺序与 arity 相关，而 relay 不暴露后端 revision；未观察到漂移，但跨 arity 解读依赖端点稳定。复现分析：`uv run python scripts/evaluate_rerank.py --analyze --resamples 100000`（完整本地缓存下不联网、不读 key）。
<!-- BEGIN H-HYBRID-MRL1024-CHECKLIST -->
- [x] **H：dense-1024 + char-bigram BM25 的独立等权 exact RRF 基线。** 已在 5,681 篇完整文档 / 2,394 条 query 上从同一 4096 维 cache 双侧前缀切片并重归一化，零 API、无 rerank；1doc nDCG@10 **0.908 [0.895, 0.921]**。H−G 两个 retention family 共 2/12 项通过 Holm；未拒绝项只写“未检测到差异”，不写“无损/等价”。复现：`scripts/evaluate_h_hybrid_mrl1024.py`。
<!-- END H-HYBRID-MRL1024-CHECKLIST -->
<!-- BEGIN M8-CHECKLIST -->
- [x] **M8：FastAPI 服务与 HTTP 性能基准。** `/api/search` 只编排现有 `OnlineRetriever`，有严格输入、脱敏错误、metadata allowlist、fail-fast 并发 admission 与单文件静态 UI；benchmark 排除 warm-up，报告 p50/p95/p99、成功 QPS、错误率/status 与八阶段耗时。当前认证 HTTP p95 **228.3 ms** / **4.97 QPS**。同步器从无文本 numeric samples 重算 aggregate；不调用 chat completion。
<!-- END M8-CHECKLIST -->
- [x] **M9b1 完成后的 provenance hardening（不回退、不重算 M9b1 完成状态）。** 已修复交付后审计发现的三项跨机器/多 profile 风险：① run/reference profile 统一为小写 1–64 字符 canonical slug，并用 `ntpath.isreserved()` 跨平台拒绝结尾句点、`CON` / `COM1` 等 Windows 别名；② `run_crud_generation.py` 与结果文档同步器把 `--artifacts` 固定到项目内 gitignored 的 `indexes/crud/generation/v1/`，并在 status、显式 run/reference 选择、读写或 provider import 前拒绝 artifact 树中现有的 symlink/junction；③ chat provider 不再把缺失/空白 `response.model` 回落为 requested model，generation/QG/reference QA/prediction QA 的 status、resume 与 completion 路径均认证每条 `served_model`。核心生成/cache row contract、prompt、stage DAG 与统计口径未改；chat profile fingerprint 域以及 text-free samples/report provenance envelope 有意升至 v2，公开 artifact 携带固定 `chat_profile_contract_sha256`。旧 v1 sidecar/samples/report 保留在磁盘但不能由新 runner 原地续跑、迁移或发布，实验必须换新 slug 重启。M9b2 仍须在完整 text-free numeric samples/report 重算认证后才能发布数字。
- [ ] **RAGAS / DeepEval 内置指标提示词在中文上的校准度。** 研究只验证了管道（`base_url` 支持、`adapt_instruction` 语义、DeepSeek/Kimi 类），**零中文评测**。人工标 ~50 行，先测判官与你的一致率。

**部署与生态待验证事项**

- [ ] **Langfuse / Opik / Confident AI 的免费额度**（研究阶段定价页被限流）；**Opik 自托管是否强制 Docker/K8s**（未核实，按 Langfuse 类比推断）。
- [ ] **Elasticsearch 若仍在对比范围内**：`dense_vector` 的 `dims` 上限**恰好是 4096**（零余量，换更宽的模型就直接废），且 `infinilabs/analysis-ik` 的 GitHub releases 最新一条停在 2024-05-06，未确认存在兼容 ES 9.5 的 IK 构建。