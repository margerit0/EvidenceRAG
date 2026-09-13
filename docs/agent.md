# 文档调查与对照运行

文档调查在现有检索器之上增加有限步数的动作循环：搜索候选、读取完整段落、补充检索、
请求必要信息或输出带引用的答案。模型提出结构化 JSON 动作，Python 白名单验证后执行。
当前工具是文档检索与读取；执行计划解析、日志诊断和持久多轮会话仍在后续计划中。

实现入口为 `src/zhrag/agent.py`。进度与交接见 [迭代计划](agent-iteration.md)。

## 启用服务

复用原有索引、Milvus 和 embedding/rerank 配置，另需 `LLM_*` 配置。
显式启用调查后，每次请求可能产生多轮 chat、embedding 和 rerank 调用。

```powershell
uv run --extra service --extra milvus --with milvus-lite==3.2.0 python scripts/serve.py --enable-agent --generation-reasoning-effort low --generation-max-tokens 4096
```

进入 `http://127.0.0.1:8000`，选择“文档调查”。要同时保留单轮问答入口，增加
`--enable-generation`。`--enable-agent` 不能与仅支持限定查询的 `--query-cache` 同用。
这些启动参数不是 Agent 真实效果验收记录。

接口：`POST /api/investigate`，请求体仅接受 `{"query":"需要调查的问题"}`。
`GET /api/capabilities` 增加 `agent_enabled` 与 `agent_profile`。
原 `/api/search` 与 `/api/ask` 分别保留独立检索、单轮问答语义。

响应包含：

- `status`、固定提示 `message`，以及必要时的 `clarification`。
- `blocks` 中每段正文及引用编号，`sources` 中本次读取的完整证据与安全来源链接。
- `events` 中动作、结果码、候选/证据编号和相对开始时间；它是执行记录，不含内部思维。
- `usage` 中模型决策、搜索、读取次数和累计估算输入 token 数。
- `total_seconds` 与绑定模型配置、索引身份、提示词和预算的 `agent_profile`。

页面在请求完成后显示执行记录，尚未实现 SSE 实时事件流。追问没有保存会话状态；
用户需把补充信息与原问题一起重新提交。

## 工具与证据约束

`search_docs` 复用原 `OnlineRetriever`，候选仅展示短摘要。
`read_passage` 读取本次搜索缓存中的完整段落；不是任意文件读取，也不会抓取外部 URL。
同一段落在一次调查内只有一个稳定编号，只有已读取的段落可以成为答案引用。
过大的段落整段跳过，不裁切其中的代码块和表格。

动作结构、引用 ID、答案长度和 JSON 格式均在程序中验证。
检索得到的内容被作为不可信数据传递，模型不能增加工具或直接执行 SQL、shell 命令。
这些限制不构成模型事实性或 prompt injection 防护的完整证明。
**引用合法仍不等于引用支持结论**；需要对照评测和人工审阅。

| 状态 | HTTP | 语义 |
| --- | --- | --- |
| `answered` | 200 | 答案结构和引用校验通过 |
| `clarification_needed` | 200 | 需要必要的用户信息 |
| `insufficient_evidence` | 200 | 文档不足以支持答案 |
| `budget_exhausted` / `cancelled` | 200 | 预算用完或调查停止；不返回未完成答案 |
| `invalid_action` / `invalid_answer` | 503 | 动作或答案校验失败 |
| `generation_failed` / `generation_timeout` | 503 | 模型故障 |
| `retrieval_failed` | 503 | 所有已尝试检索均失败后无法继续；不冒充无答案 |
| `agent_unavailable` | 503 | 未启用调查，不调用依赖 |
| `service_busy` | 429 | 达到与原接口共用的并发上限 |
| `invalid_request` | 422 | 输入不符合合同 |

## 执行预算

默认最多 10 次模型决策、3 次搜索、6 次读取；相同查询和已读取证据不重复调用。
通过 `--agent-max-steps`、`--agent-max-searches`、`--agent-max-seconds` 配置主要预算，
`--context-passages` 与 `--context-tokens` 配置读取数量及每轮输入估算 token 上限。
模型每次输出受 `--generation-max-tokens` 约束；累计估算输入上限由 `AgentSettings` 配置。

Agent chat 固定不重试，避免继承单轮问答历史上的长重试阶梯；检索依赖的重试语义保持原样。
180 秒默认总时长在依赖调用前后检查，**不是可强杀阻塞请求的严格墙钟 deadline**。
取消会在当前依赖调用返回后的边界停止后续操作；并发名额在工作实际结束时释放。
取消不保证供应商停止执行或计费。使用的 token 估算器不是任意模型的精确 tokenizer，
`usage` 中的计数也不能直接换算成供应商账单。

调查结果默认不落盘。原 M11 运维 trace 仍保持检索合同，没有新增外部 tracing 平台。

## 本地任务集

任务、答案、证据与审核材料位于 gitignored 的 `indexes/agent_eval/`。
初始化命令从独立编写的场景模板产生**未审核草稿**，不读取语料、不调用模型，已有文件拒绝覆盖。

```powershell
.venv/Scripts/python.exe scripts/agent_tasks.py --initialize
.venv/Scripts/python.exe scripts/agent_tasks.py
.venv/Scripts/python.exe scripts/agent_tasks.py --require-reviewed
```

最后一个命令会拒绝未审核草稿，这是预期行为。审核应结合固定文档快照，填写正确的
`expected_status`、逐条 `acceptance_criteria`、支持答案的 `reference_sources`、
`reviewer`、`snapshot`，完成后才将 `reviewed` 设为 true。
现有模板只是场景起点，不是已经核对答案的标注集。

`source_group` 应覆盖可能共享证据的问题，同组只能位于一个 split；同问题或同任务 ID
不能重复。场景模板的临时主题分组需在审核时根据真实证据重新确认。
最终测试集冻结前不得用它调整提示词或参数。

## 三种方法的对照入口

| 方法 | 流程 |
| --- | --- |
| `single_rag` | 原问题检索一次，再按原问答合同生成 |
| `fixed_workflow` | 模型一次性给出有限查询计划，顺序检索，按各查询排名轮流合并去重后生成 |
| `document_agent` | 模型看到工具返回值后动态搜索、读取、追问或结束 |

三者共用检索器、生成模型配置、上下文上限与预算配置；每种方法另有独立配置指纹。
固定流程不会根据检索结果追加查询，单轮/固定流程沿用只回答或拒答的旧问答合同。
因此“追问是否恰当”用于能力对比，不能当成对相同交互策略的纯检索消融。
固定方法与自适应方法的提示词、证据选择也不同，整体差异不能归因给单一因素。

默认只预览本地执行清单，未传 `--run` 不加载服务商配置，不连接 Milvus，不调用模型：

```powershell
.venv/Scripts/python.exe scripts/compare_agent.py --allow-drafts --split dev --limit 3
```

真实运行必须显式传 `--run` 与一个新 `--run-id`，需要服务可用的依赖环境，会产生费用：

```powershell
uv run --extra service --extra milvus --with milvus-lite==3.2.0 python scripts/compare_agent.py --allow-drafts --split dev --limit 3 --run --run-id dev-smoke-v1
```

完成审核后去掉 `--allow-drafts`；测试集始终禁止草稿模式。
输出保存到 `indexes/agent_eval/runs/<run-id>/`，既有目录拒绝覆盖，也不自动恢复部分实验。
任务顺序固定，各任务的方法执行顺序循环轮换，以减少方法与供应商时间段完全重合的影响。
这不消除模型随机性或供应商漂移；严格比较还需要重复试验与人工审阅。

`manifest.json` 保存任务集/选中任务指纹和完成状态；`trials.jsonl` 逐条保存结果及空审核字段。
`complete=true` 只代表选中调用流程结束，允许其中有失败试次，**不代表质量合格**。
人工检查任务成功、关键陈述支持度、追问/拒答是否合适，再汇总配对指标。
当前未实现语义自动打分、真实账单成本汇总或统计结果发布，不会把 HTTP/生成成功当成准确率。
