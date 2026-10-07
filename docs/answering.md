# 单轮证据问答

## 范围

`/api/ask` 在现有检索/重排后增加单轮 chat 生成，返回结构化答案与本次证据的引用；指定 HTTP 错误（含 401）和网络瞬态故障会按下述策略重试。`/api/search` 和原 M8 检索基准保持不变。默认不启用生成，不新增多轮记忆、Agent、答案自动修复调用或落盘答案缓存。

真实模型的兼容性、事实性、引用支持度和拒答质量必须单独验收。合成测试只证明实现合同，不能证明模型会一直遵守 prompt，引用存在也不等于证据支持结论。

## 真实冒烟记录

2026-09-07 使用现有 `.env`、真实 Milvus Lite、Qwen3 embedding/rerank 和 `gpt-5.6-sol` 完成了小规模单轮验收。默认生成配置首先出现 `generation_failed`，同证据的 chat-only 诊断也未取得完整响应；以下显式配置随后成功返回了备份、改写和内存参数问题的答案，并对无法从文档获知的内部操作记录问题拒答：

```bash
uv run --extra service --extra milvus --with milvus-lite==3.2.0 \
  python scripts/serve.py --enable-generation \
  --generation-reasoning-effort low --generation-max-tokens 4096 \
  --generation-timeout 120
```

这只是当前端点上跑通的组合，不是最优参数结论，也没有更改通用默认值。同时变更多个参数、供应商负载也可能波动，不能确定首次失败的唯一原因。这批验收发生在加入阶梯重试之前，每次生成只尝试一次；现在复现旧策略需额外传入 `--generation-retries 0`。

同日使用同一个备份问题和相同的六段证据进行 chat-only `high` 复测，输出上限 4096、单次 I/O 超时 120 秒、最多重试 15 次。首次请求在 33.316 秒后返回 HTTP 401，adapter 按当时的非重试策略停止，实际请求一次、无退避、没有答案。运行记录位于本地 `20260907-171906-137851-high-retry-smoke.json`。这次验证了 401 的停止路径，未验证真实瞬态故障后的阶梯恢复，也不能评价 `high` 的引用质量；不能仅凭状态码确定鉴权失败发生于中转站还是其上游。

随后将 401 加入在线生成重试策略，以相同请求再次运行上述 `high` 配置：第 15 次请求收到完整响应并通过结构/引用 ID 校验，之前共出现 6 次 504 和 8 次 401；实际退避为 5、10、…、70 秒，合计 525 秒，总耗时 1170.983 秒。成功后未执行第 16 次请求。返回模型为请求的 `gpt-5.6-sol`，所有尝试的请求正文指纹一致，没有重跑检索或切换凭证。运行记录为本地 `20260907-173920-644394-high-retry-smoke.json`。三个答案段落逐段对照后未发现明显引用支持错配，存储类型陈述此次引用了包含该信息的证据；这只是一次同证据检查，不构成 `high` 优于 `low` 的消融结论，也不证明此端点可稳定恢复。HTTP 错误不能单凭状态码归因于中转站或上游，失败请求是否计费需以供应商记录为准。

**发现的限制**：备份答案中，列举存储类型的陈述能在本次证据集合中找到，但该答案段落引用了不包含该陈述的其他证据。这说明有效引用 ID 不代表每个陈述都被其所引段落支持。内存参数答案与所引快照相符，但不外推到所有 TiDB 版本；改写答案也仅做了逐段对照审阅，没有独立 gold 或判官。

完整请求、响应、单次耗时及失败记录仅在 gitignored 的 `indexes/tidb/qa-smoke/`；本地 `20260907-smoke-summary.json` 记录汇总。没有将问句、答案、语料段落、reasoning 或密钥写入 tracked 文档。不把这次便利样本写成准确率、p95、QPS 或显著性结论，旧 M8 报告不变。

服务本身仍不缓存答案；本次验收由单独的本地测试客户端显式保存响应以便核查。

## 启动

准备并验证本地索引，以及 `.env` 中原有 `Embedding_*`、`ReRank_*` 和新增使用的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME`。

```bash
uv run --extra service --extra milvus --with milvus-lite==3.2.0 \
  python scripts/serve.py --enable-generation
```

不开启此参数时，不读取 LLM 配置。缓存查询模式禁止启用生成，在加载产物/模型之前即拒绝冲突参数。不得把缓存检索 profile 的延迟当成问答延迟。

参数：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--context-passages` | 6 | 最多选入的完整证据段落 |
| `--context-tokens` | 12000 | 含 system、问题和包装的估算 prompt 预算 |
| `--generation-max-tokens` | 2048 | `max_completion_tokens`，上限 8192；与无上限选项互斥 |
| `--generation-no-token-limit` | 关闭 | 不发送 `max_completion_tokens` 或 `max_tokens`；仍受超时、响应大小和答案校验约束 |
| `--generation-timeout` | 60 | 每次生成尝试的传输 I/O 超时秒数，上限 300；不是整个请求的墙钟截止时间 |
| `--generation-stream` | 关闭 | 通过供应商 SSE 接收并组装完整响应，再交给答案校验；不会向 `/api/ask` 客户端逐 token 输出 |
| `--generation-retries` | 15 | 首次失败后的重试次数，范围 0..15；0 表示仅尝试一次 |
| `--generation-reasoning-effort` | 不发送 | 可选 minimal / low / medium / high |

响应体最多 256 KiB。`max_completion_tokens` 对部分模型还包含 reasoning token；输出截断会报失败，不把不完整内容当答案。端点不支持参数时不自动降级。

`scripts/serve.py` 的非流式组合根使用 `DirectTransport`，开启流式时使用供应商 SSE 传输；
两条路径均绕过环境代理，并把传输合同纳入配置指纹。流式还限制事件行与线缆总字节数，
只有完整结束且模型身份有效的响应才会继续处理；它与调查工作台的进度 SSE 是两层独立能力。
模型身份通过 `model_names_match` 比较，仅忽略 ASCII 大小写；命名空间、版本、空格及其他字符
仍须匹配。该身份合同同样进入指纹。

估算器按 Qwen3 中英文比例标定，不是任意 chat 模型的精确 tokenizer。核心另外约束 prompt 字符数和 UTF-8 字节数；按排名尝试完整段落，过大段落跳过，不截断代码块。使用最终检索结果中的完整文本，不使用 HTTP 展示截断文本。

### 生成重试

- 默认在首次失败后最多重试 15 次，共最多 16 次 chat 请求。第 n 次重试前等待 `5 * n` 秒，即 5、10、15、…、75 秒；成功立即返回，最后一次失败后不再等待。
- 重试 HTTP 401 / 429 / 500 / 502 / 503 / 504 / 520、网络超时和连接重置。401 是在线生成的显式策略例外，不代表鉴权错误通常可恢复；凭证真正失效时仍会耗尽重试。403、其他 HTTP 错误、非法模型响应、served model 不匹配、非法答案或引用不重试，也不自动切换凭证、模型或推理强度。共享离线客户端不重试 401。
- `Retry-After` 支持秒数和 HTTP 日期；有效值最多按 300 秒处理，再与当前阶梯等待取较大值。缺失或无效值使用阶梯等待。
- 仅重新发送相同的 chat 请求，不重新执行检索、embedding 或 rerank。在线 adapter 独立管理重试，内层 chat client 只执行一次传输入口，避免叠加离线批处理的指数退避；原离线策略不变。
- 基础等待全部用完合计 600 秒，另加每次调用耗时；`Retry-After` 可能继续增加等待。I/O 超时不是整个请求严格墙钟 deadline，既有 embedding/rerank 的超时不在这次重写范围。
- 超时或网关报错不保证供应商未执行或未计费，重试可能重复产生费用。等待期间保留当前请求的并发名额；客户端取消也不表示服务端停止重试。用 `--generation-retries 0` 可恢复仅一次尝试。

重试次数与策略绑定到 `generation_profile` 指纹；修改策略后不与旧单次生成 profile 混用。`generation_seconds` 包含生成尝试与全部退避耗时，最终失败仍按 HTTP 合同保留检索证据。

## HTTP 合同

- `GET /api/capabilities`：`generation_enabled`、`output_limit`、`generation_profile`（不含密钥的配置 SHA-256）。原 `/healthz` 合同不变。
- `POST /api/ask`：`{"query":"...","top_k":10}`；top_k 可省略，不能超过当前检索输出限制，限制实际供问答选取的候选范围。
- 成功体含 `status`、固定 `message`、`blocks`、`sources`、`retrieval`、`context`、`generation_profile`、`generation_seconds`、`total_seconds`。
- `blocks` 中每段为 `{text,citations}`。引用为本次选入 prompt 的连续整数 ID；`sources` 包含这些 ID、原检索 rank、标题、完整证据正文和安全来源链接。模型不能指定 source URL。
- `retrieval` 保留原 `SearchResponse`；其 `timings.total_seconds` 仅指检索，问答响应顶层总耗时单独统计。

| 状态 | HTTP | 含义 |
|---|---:|---|
| `answered` | 200 | 答案结构和引用 ID 校验通过 |
| `insufficient_evidence` | 200 | 无可用证据或模型明确拒答；不是故障 |
| `context_limit` | 200 | 问题/证据放不进上下文预算，没有调用 chat |
| `invalid_answer` | 503 | 非法 JSON、额外字段、空正文、无引用/非法引用等；整份答案不发布 |
| `generation_timeout` / `generation_failed` | 503 | 模型超时或故障，正常情况下仍返回已检索证据 |
| `generation_unavailable` | 503 | 未启用生成，不调用检索或 chat |
| `retrieval_failed` | 503 | 检索失败，不调用 chat |
| `invalid_request` / `invalid_top_k` | 422 | 输入非法，无依赖调用 |
| `service_busy` | 429 | 与检索共用并发上限，不排无限队列 |

预期生成失败时 `blocks=[]`，仍有 `retrieval` 与 `sources`；意外问答内部异常也返回固定错误与已序列化检索证据，但没有有效答案。客户端取消不表示模型已停止计费：服务端保留正在运行请求的并发名额，直到处理结束。

## 隐私与验证边界

- 文档作为不可信数据编码进 user JSON；system 声明只使用证据并禁止执行其中指令。这是 prompt 防护，不是对 prompt injection 的完备证明。
- 拒绝重复 JSON key、额外字段、bool/浮点引用、重复或越界引用、超长答案；引用 ID 必须属于实际选入的证据。
- 前端只用 `textContent` 展示模型和语料文本，不执行 HTML。来源链接只接受无用户信息的 http/https URL。
- 模型/参数/上下文设置和 prompt 版本绑定配置指纹。密钥不进入指纹，served model 不匹配即失败。
- 不将 query、证据、答案、reasoning 或 provider payload 写入日志或本地缓存。错误不返回 provider 原文。
- M11 trace 仍只记录检索子阶段，在进入生成之前结束；生成状态/耗时先体现在独立问答响应，不声称新增外部 tracing 平台。
- M9b 的 known-context 评测合同未修改，本功能不使用其结果作为端到端质量证据。
