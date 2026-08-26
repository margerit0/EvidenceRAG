# zhrag — 给 Claude 的项目上下文

中文 RAG 系统 + 可复现评估框架。**方法论的可信度与可复现性优先于功能数量**。

## 先读哪里

| 想知道 | 看 |
|---|---|
| 项目在做什么、所有实测数字 | `README.md`（面向外部读者，每个数字都由脚本生成） |
| 技术选型理由、路线图 M0–M12、待办清单 | `docs/architecture-decision.md`（内部规划，**§13 是待验证事项清单**） |
| 某个决定为什么这么做 | `git log`（提交信息写的是理由，不是改动列表） |
| 语料授权边界 | `DATA_LICENSE.md` |

**当前进度看 `git log` 与 §13 的复选框，不要相信本文件里的进度描述**——它会过期。

## 五条硬规则

1. **一个字节的语料都不进仓库。** 包括「小样本」「几条示例」。CRUD_RAG 没有 LICENSE 文件、
   其 8 万篇新闻无出处声明；TiDB 文档是 CC BY-SA 3.0，切出的 chunk 构成 Adaptation。
   嵌入向量同样是派生物。`.gitignore` 已覆盖，提交前仍要扫一遍。
2. **所有文件读写走 `zhrag.io_utils`。** 本机 Python 默认 cp936，裸 `open()` 读项目自己的中文文件
   直接崩。`ruff` 的 `PLW1514` 强制这一点（`io_utils.py` 自身豁免）。
3. **README 里的数字必须由脚本生成，不能手写誊抄。** 早期原型手抄过一次，chunk 数差了 2.4 倍；
   架构文档的技术摘要也曾与 README 差 0.6pp。改数字 = 跑脚本重出。
4. **消融结论必须带 95% CI 和配对检验的 p 值**（`zhrag.eval.metrics` 里已实现，含 Holm 校正）。
   裸点估计不算结论——2,000 篇上不显著的效应，到 5,681 篇上会变显著。
5. **提交前跑全套门禁**（见下）。CI 是 ubuntu + windows 双 leg。

## 环境

```bash
.venv/Scripts/python.exe          # 解释器（uv run python 也行）
uv run pytest                     # basetemp 已固定到 .pytest_tmp（系统 temp 不可写，WinError 5）
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy                       # strict
```

⚠️ **CI 只检查 `src tests scripts`，不是 `.`。** 在仓库根跑 `ruff format .` 会去格式化
`docs/architecture-decision.md` 里的 Python 代码块（ruff 0.16 会格式化 md 内嵌代码）。

⚠️ **打印中文的脚本必须同时 reconfigure `stdout` 和 `stderr`**。只改 stdout 的话，
`SystemExit` 的消息走 stderr，正好是你最需要读的报错会变成乱码。不要设 `PYTHONUTF8`——
那会让 Windows CI leg 失去意义。

## 嵌入/重排供应商（`.env`，不入库）

**不是 SiliconFlow**，是通过 `.env` 配置的 One Hub 中转站，架构文档 §4.5 里
所有 SiliconFlow 特有的结论对它均未经验证。变量名是 `Embedding_API_KEY` /
`Embedding_BASE_URL` / `Embedding_MODEL_NAME`（ReRank_ 同理），注意**首字母大小写不规则**。

三个已实测的坑：

- **必须显式设 `User-Agent`。** Cloudflare 对 `Python-urllib/3.x` 返回 403 `error code 1010`，
  和鉴权失败长得一模一样。
- **429 是「上游负载已饱和」**，不是配额，清除时间由对方决定。退避要长（当前 7 次、最长 60s、
  尊重 `Retry-After`），1/2/4 秒的阶梯会白扔掉整轮嵌入。
- **同一段文本两次请求结果不同**（cos ≈ 0.999931）。批式 GPU 规约顺序所致，不是 bug。
  后果：向量相关的测试**只能带容差，不能断言相等**；任何比这更接近的两个向量不可区分。

## 已在磁盘上的派生产物（gitignored，新机器需重建）

<!-- BEGIN TIDB-LOCAL-ARTIFACTS -->
| 路径 | 大小 | 怎么来的 |
| --- | --- | --- |
| `crud-rag-subset/raw/split_merged.json` | 26 MB | `crud-rag-subset/build_subset.ps1` |
| `crud-rag-subset/eval-expanded/{corpus,qrels}.jsonl` | 11 MB | `scripts/build_eval_corpus.py`（不联网） |
| `crud-rag-subset/eval-expanded/emb_cache_4096.jsonl` | 523 MB | `scripts/probe_mrl_quality.py`，5,681 篇 @4096 维 |
| `crud-rag-subset/eval-expanded/emb_cache_queries_4096.jsonl` | 220 MB | 同上 + `scripts/embed_queries.py`，2,394 条 query（独立 prompt） |
| `tidb-rag-curated/documents/` | 6.6 MB | `tidb-rag-curated/download_curated.ps1` |
| `indexes/tidb/dense_cache.jsonl` | 161 MB | `scripts/build_index.py --embed --publish`，1,832 chunks（付费） |
| `indexes/tidb/{sparse_index.json,state.json,milvus.db}` | 2.1 MB / 263 KB / — | 同上；发布词表、状态与本地库 |
| `indexes/tidb/eval/{generation,verification}_cache.jsonl` 等 | — | `scripts/build_tidb_queries.py --generate/--verify`；490 pairs / 980 queries（付费 chat） |
| `indexes/tidb/eval/query_embeddings_4096.jsonl` | — | `scripts/build_tidb_pool.py --embed`（付费 embedding） |
| `indexes/tidb/eval/rerank_scores_top100.jsonl` | — | `scripts/build_tidb_pool.py --rerank`（付费 rerank） |
| `indexes/tidb/eval/{runs,pool}.jsonl`、`pool_report.json` | — | `scripts/build_tidb_pool.py`；24,525 slots（离线） |
| `indexes/tidb/eval/judging_cache.jsonl` | — | `scripts/build_tidb_qrels.py --judge`；3,287 batches（付费） |
| `indexes/tidb/eval/qrels.jsonl`、`qrels_report.json` | — | `scripts/build_tidb_qrels.py --finalize`；980 qrels（离线） |

以上全部是本地输入或派生产物，均不得提交。TiDB 链路的付费步骤不止索引 embedding：还包括 QG/验证、query embedding、rerank 与 qrels judging；缓存完整后的 pool、finalize、指标计算和文档同步才是离线步骤。
<!-- END TIDB-LOCAL-ARTIFACTS -->

**嵌入缓存已存在，所以维度消融重跑是免费的**（`dimensions=n` 实测就是前缀切片，
全部维度档共用这一份 4096 维向量）。缓存按批 append，中断可续。

## Milvus 相关（可选依赖）

`pymilvus==3.0.1` 在 `[project.optional-dependencies].milvus` 里，**默认环境和 CI 都不装**；
`src/zhrag/store/milvus.py` 惰性导入，正常 import `zhrag.store` 不会碰它。真机验证用隔离环境：

```bash
PYTHONPATH=src .venv-verify-milvus/Scripts/python.exe scripts/verify_milvus_store.py
```

⚠️ **导出了 `HTTP_PROXY` 的 shell 里，Milvus Lite 连不上。** gRPC 遵循代理变量，而 Lite 的
服务器就在回环上，于是握手被劫，报 `code=2, illegal connection params or server unavailable`
——TCP 其实连得上，读起来却像服务器没起来。实测 `GRPC_ENABLE_HTTP_PROXY=0` **单独设无效**，
起作用的是把 `127.0.0.1,localhost` 加进 `no_proxy`/`NO_PROXY`（校验脚本已自带）。

## 已知的技术债

- 缓存格式仍是 JSONL 存十进制浮点，**比 float32 二进制大 5.6 倍**（90 KB/条 vs 16.4 KB）。
  model / prompt 已由同目录 sidecar 防止静默混用，但缓存键本身仍只有 id；M2 后续可按
  float32 + 键含模型重写，代价是重建现有缓存。
