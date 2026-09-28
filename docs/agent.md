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
| `retrieval_failed` | 503 | 所有已尝试检索均失败后无法继续；不冒充无答案。无论模型以 `abstain` 还是 `answerable:false` 结束，均返回此状态 |
| `agent_unavailable` | 503 | 未启用调查，不调用依赖 |
| `service_busy` | 429 | 达到与原接口共用的并发上限 |
| `invalid_request` | 422 | 输入不符合合同 |

## 执行预算

默认最多 10 次模型决策、3 次搜索、6 次读取；相同查询和已读取证据不重复调用。
通过 `--agent-max-steps`、`--agent-max-searches`、`--agent-max-seconds` 配置主要预算，
`--context-passages` 与 `--context-tokens` 配置读取数量及每轮输入估算 token 上限。
模型每次输出受 `--generation-max-tokens` 约束；累计估算输入上限由 `AgentSettings` 配置。

Agent chat 默认不重试（`--agent-generation-retries 0`），避免继承单轮问答历史上的长重试阶梯；
可显式设为 0–5 次，失败后按 5、10、15、20、25 秒线性等待，只重试既有的瞬态状态码（含 429/5xx/401）
与超时/连接重置，403 不重试。重试次数与传输合同一起进入方法配置指纹。检索依赖的重试语义保持原样。
180 秒默认总时长在依赖调用前后检查，**不是可强杀阻塞请求的严格墙钟 deadline**。
取消会在当前依赖调用返回后的边界停止后续操作；并发名额在工作实际结束时释放。
取消不保证供应商停止执行或计费。使用的 token 估算器不是任意模型的精确 tokenizer，
`usage` 中的计数也不能直接换算成供应商账单。

服务组合根中 chat、embedding、rerank 均使用 `zhrag.providers.direct.DirectTransport`：
直连目标主机 443 端口、不读取环境变量或系统代理、不跟随重定向、保留 TLS 证书校验，
成功响应限制 256 KB（chat）/ 2 MB（embedding、rerank）；非 2xx 响应始终以其 HTTP 状态抛出，
只保留 16 KB 错误正文用于诊断。Milvus 回环连接另在 `no_proxy`/`NO_PROXY` 中加入 loopback。
远程端点必须是 HTTPS，仅本机地址允许 HTTP。

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

### 语料落地的草稿装配（v2）

`indexes/agent_eval/v2/` 下保留第二版任务集的原始**未审核草稿**，审核后版本另存于
`indexes/agent_eval/v2/reviewed/`，不覆盖原稿。两者都按主题分组、逐条绑定本地文档快照。
起草规范在 `indexes/agent_eval/v2/drafts/AUTHORING_PROMPT.md`，每个主题一个 JSON 草稿文件，
每条草稿除任务字段外还带 `evidence_quotes`（逐字引文）与 `author_notes`（起草理由、干扰项、
unanswerable 的检索关键词）。装配命令完全离线：

```powershell
.venv/Scripts/python.exe scripts/assemble_agent_tasks.py --snapshot "pingcap/docs-cn@26f202b"
.venv/Scripts/python.exe scripts/assemble_agent_tasks.py --snapshot "pingcap/docs-cn@26f202b" --write --split backup-restore=dev --split slow-query-tuning=test ...
```

装配器只证明三件机械事实：每段引文（去掉链接、强调、HTML 版本标记与 Hugo shortcode 后）
在所引本地文档中逐字存在；每个 `reference_sources` 都在已发布索引的 450 篇文档内；
每个来源组只落在一个 split。任何一条不通过就不写出任何文件。写出的 `tasks.jsonl` 全部
`reviewed=false`、`reviewer` 为空；引文与起草说明单独写入 `draft_evidence.jsonl` 供审核者核对。
**引文逐字存在不等于验收条件正确、问题分类恰当或 unanswerable 确实无答案**；这些仍需逐条语义审核。
用户可以委托模型完成审核，但 `reviewer` 必须如实记录实际审核者及“模型审核、非人工审核”，
完成复核后才能把 `reviewed` 设为 true。该字段表示审核状态，不能认证人工身份；模型起草或模型
审核均不计入人工审核数量。本轮按用户委托完成模型审核，人工审核仍为零。

<!-- BEGIN AGENT-V2-REVIEW -->
2026-09-26 模型审核写出结果（由本地聚合摘要重算；快照 `pingcap/docs-cn@26f202b`）：

| 范围 | 任务 | simple | multi_document | clarification | unanswerable | 来源组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 79 | 31 | 20 | 18 | 10 | 10 |
| dev | 48 | 19 | 12 | 11 | 6 | 6 |
| test（已冻结） | 31 | 12 | 8 | 7 | 4 | 4 |

模型审核 79 条，人工审核 0 条；2 accept / 77 revise / 0 reject。314 段引文通过规范化逐字校验，所有引用均在已发布索引内。
所有任务均通过 `validate_tasks(..., require_reviewed=True)`；原始任务与证据 packet 逐条对齐。

- 任务集 canonical SHA-256：`090f7408661a334b8b800470b282f3a9ff237f9f857f5d734298c883cf54749e`
- `tasks.jsonl` 文件 SHA-256：`a593ac04cd4386e5c4b771bbe5ce3088e6cabbb767c773f2cfaf7cfcab5b0de8`
- 冻结 test canonical SHA-256：`d5974735b36efa3e1ea02e73819d91553d82ca33b9d83d8ded91996825100a8c`
<!-- END AGENT-V2-REVIEW -->

本轮审核要求只把问题明确询问、有原文依据的内容作为必答项；多篇文档提供相同信息时不据此
认定为跨文档任务。追问的验收只针对问题本身，拒答按无正文的 `insufficient_evidence` 合同检查。
无答案判断重新搜索了整个本地语料。审核记录、原始 packet、修订草稿及输入指纹保存在本地，
最终还独立核对了参考来源和引文不跨来源组重复归属。

`source_group` 应覆盖可能共享证据的问题，同组只能位于一个 split；同问题或同任务 ID
不能重复。场景模板的临时主题分组需在审核时根据真实证据重新确认。
审核完成后冻结最终测试集及指纹，测试集及其结果均不得用于调整提示词或参数。

对已完成的本地试次，先准备独立审核文件：

```powershell
.venv/Scripts/python.exe scripts/review_agent.py --run-dir indexes/agent_eval/runs/<run-id> --prepare
```

审核者阅读 `review-packet.txt`，只在 `reviews.jsonl` 填写标签；原始 `trials.jsonl` 不修改。
审核完成且任务本身 `reviewed=true` 后，使用：

```powershell
.venv/Scripts/python.exe scripts/review_agent.py --run-dir indexes/agent_eval/runs/<run-id> --report
.venv/Scripts/python.exe scripts/review_agent.py --run-dir indexes/agent_eval/runs/<run-id> --check
```

`quality_report.json` 是仅含聚合数据的报告：任务成功率按全部任务计算，回答内的陈述支持度
单独计算；每个方法对的差异使用来源组 bootstrap 百分位 CI，配对 p 值使用**整来源组交换方法标签的
双侧置换检验**：非零差值组数 ≤16 时精确枚举全部 2^k 种符号组合，否则用带种子的 Monte Carlo，
再做 Holm 校正。只有两个独立来源组时精确 p 值最小为 0.5，不会因增加重采样次数变得显著；
来源组少于 20 个时报告标记 `ci_small_group_caution`，bootstrap CI 不是小样本显著性的补救。
报告合同为 `document-investigation-quality-v2`。
报告发布前会校验任务对是否完整、试次原文指纹、方法 profile、每条审核记录和任务审核状态。
故障试次、拒答试次和缺证据试次都必须保留并审核，不能通过删除分母改善结果。

`criteria_met` 对应任务的每条验收条件，必须逐项填布尔值；`task_success` 需同时满足全部条件、
动作符合 `expected_status`、答案陈述全部受到所引证据支持，以及适用时的恰当追问/拒答。
不一致的标签会被拒绝。只填写 reviewed 或只看 HTTP 成功不能通过审核。
统计推断至少需要两个来源组；来源组很少、试次单次执行时仍需谨慎解读，不作强结论。
模型填写的试次标签必须标注为模型判官结果，不能计入人工审核数量；报告认证只能核对完整性、
指纹与标签自洽性，不能把模型标签升级为人工结论。
报告里的引用支持比例只针对已回答内容，拒答不计为“完全支持”。
审核材料只隐藏显式方法标签，答案风格仍可能暴露方法；哈希验证完整性，不能认证审核者身份。

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

`manifest.json` 保存任务集/选中任务指纹、实际方法 profile、完整试次摘要和完成状态；
`trials.jsonl` 逐条保存不可修改的结果，历史空审核字段仍作为固定占位。
实际审核写入独立的 `reviews.jsonl`，逐条绑定 task+result 指纹。
`complete=true` 只代表选中调用流程结束，允许其中有失败试次，**不代表质量合格**。
人工检查任务成功、关键陈述支持度、追问/拒答是否合适，再汇总配对指标。
当前提供统计报告发布与重算检查，但未实现语义自动打分或真实账单成本汇总，
不会把 HTTP/生成成功当成准确率。

本轮比较产物合同升级为 v2，新增完整试次摘要及实际方法 profile。
审核器拒绝原 v1 运行产物，不通过人工补哈希或原地改名把旧实验升级成新合同。
已有旧产物保留本地；需要审核时使用新 run-id 重跑。

冒烟可用 `--methods document_agent --limit 1 --max-steps 6 --max-searches 1 --max-reads 3`
限制工作量；`--max-seconds`、`--generation-timeout`、`--generation-max-tokens`、`--generation-retries`
分别设置调用边界时长检查、单次 chat I/O 超时、输出上限与 chat 重试次数（0–5，默认 0）。
`--max-seconds` 的有效范围是 `(0, 600]`；本轮使用 `600`、chat 超时 `90` 秒和 `5` 次重试。
实际检索仍采用现有 embedding/rerank 重试策略，因此这些参数不是按金额或严格墙钟终止的费用上限。

## 2026-09-28 开发校准与回归入口

`indexes/agent_eval/v2/calibrated-20260928/` 是独立的 dev 验收校准版本；原审核产物与
旧试次保留。复核依据是同一语料快照中明确允许的场景，不按某个方法的输出修改金标准。
原验收将绕过检查限于不需要备份的表，校准后接受原文支持的导入完成后补做快照方案，
仍要求说明绕过检查不会补齐日志覆盖。原审核者和本次模型复核者均写入版本记录。

<!-- BEGIN AGENT-V2-CALIBRATION -->
2026-09-28 校准版本：79 条模型审核任务，dev 48 条 / test 31 条；只修改 1 条 dev 任务的一个验收条件及复核者信息。314 段引文复核通过，人工审核 0 条。

- 新任务集 SHA-256：`da30f5f8c4e1ab362ee51640052872093d45f3c56f8c464776956409f4c30564`。
- 继承的冻结 test SHA-256：`d5974735b36efa3e1ea02e73819d91553d82ca33b9d83d8ded91996825100a8c`；任务逐条不变。
- 原审核目录和旧冒烟目录共 26 个文件通过字节指纹保护检查。
- 本轮全套 pytest：**1512 passed**；ruff check、ruff format --check、mypy 均通过。
- 已准备 dev 冒烟 9 试次与全量 dev 144 试次的离线清单，用户已选择先跑 dev 冒烟；全量 dev 尚未执行。

本轮 `v2-cal-smoke-20260928-high` 已完成：3 条 dev、3 个来源组、9 个试次，全部完成模型审核，人工审核仍为 0。
请求模型：`grok-4.7`；思考等级：`high`。返回的 42 个证据条目（23 个唯一片段、14 篇文档）全部与本地快照重建文本精确一致。

| 方法 | 返回状态 | 验收通过 | 获引用支持的陈述 | 平均耗时（含失败） |
| --- | --- | ---: | ---: | ---: |
| `single_rag` | answered 3 | 2/3 | 50/50 | 141.0s |
| `fixed_workflow` | answered 3 | 1/3 | 42/43 | 163.9s |
| `document_agent` | answered 2、clarification_needed 1 | 2/3 | 26/27 | 210.9s |

成功 chat / embedding / rerank 请求：27 / 21 / 21；HTTP 错误计数：`{'504': 4}`；其他请求异常：`{'TimeoutError': 3}`。
成功 chat 响应的供应商 usage：prompt 111,559 / completion 36,109 token；失败请求计费未知，embedding/rerank 金额未知，不能据此给出总费用。
首次沙箱尝试 `v2-cal-smoke-20260928` 在嵌入阶段连续连接失败后中止，未产生完整试次；其原始记录单独保留。正式受控网络运行的全部试次均保留，失败不从分母删除。
旧 key/model 的 `v2-cal-smoke-20260928-direct` 在用户更新配置时停止，六个已完成试次和第七个中断记录保留；不与新配置混合计算质量。新运行的 chat 请求逐次校验 requested model 与 reasoning_effort=high，密钥不记录。
质量报告按来源组重算 95% CI、配对整组置换检验与 Holm 校正。这里只是同一小组 dev 场景上的开发回归，传输重试影响耗时；本次同时更换 key/model 和思考等级，不能据此归因于提示词或推断完整 dev/test 效果。
逐条审核发现两条将‘如果’升级为‘只有’的必要条件误读；Agent 的跨文档答案还遗漏了机制解释。Agent 在追问场景正确询问了实际模式；基线仍返回分支答案，未执行追问。
预算在依赖调用边界检查，重试中的请求可能使实际耗时超过 max_seconds；迟到的答案不发布。未回答的试次没有答案陈述，0/0 不表示引用完全可靠。
试次 SHA-256：`d805592c05ec344f539a9b3ef5dde62318c567ad06f455d24d9a16cbab40c1c0`。
<!-- END AGENT-V2-CALIBRATION -->

Agent 提示词增加问题覆盖、证据读取与必要追问要求；代码仍只验证动作/引用结构，
不自动证明语义支持。单轮 RAG 与固定流程的提示词和动作合同未改，仍不支持追问。
新版本已完成上表的有限真实回归，仍有条件误读与机制解释遗漏；旧报告保持原验收口径。

默认离线预览：

```powershell
.venv/Scripts/python.exe indexes/agent_eval/v2/review/run_calibrated_dev.py --scope smoke --run-id v2-cal-smoke-20260928-high --reasoning-effort high
.venv/Scripts/python.exe indexes/agent_eval/v2/review/run_calibrated_dev.py --scope full-dev --run-id v2-cal-full-dev-20260928-high --reasoning-effort high
```

确认规模后加 `--run` 才执行付费调用；任务、提示词与关键代码必须匹配已保存计划。
通用 `scripts/compare_agent.py` 新增 `--generation-reasoning-effort`，可选
`minimal/low/medium/high`，默认 `low`；清单与服务组合根使用同一值。本轮按用户更新的
`.env` 使用新模型和 `high`，逐次核对发出的 chat 请求字段，不记录密钥。
试次/诊断目录必须不存在，过程只在进程内设置死代理端口并使用直连传输。
日志仅记录调用类别、状态、耗时和供应商 usage；模型答案仍在 gitignored 试次中。
真实费用需供应商定价或账单，时间和 token 参数不是严格金额上限。

## 当前真实运行记录

<!-- BEGIN AGENT-V2-SMOKE -->
2026-09-26 完成 `v2-dev-smoke-20260926-direct`：3 条已审核 dev 任务、3 个来源组、9 个试次，全部保留并完成独立模型审核。人工审核仍为 0。

| 方法 | 返回状态 | 按本轮固定条件通过 | 回答内获引用支持的陈述 | 平均耗时 |
| --- | --- | ---: | ---: | ---: |
| `single_rag` | answered 3 | 1/3 | 39/39 | 45.5s |
| `fixed_workflow` | answered 3 | 1/3 | 45/45 | 77.9s |
| `document_agent` | answered 3 | 1/3 | 35/36 | 61.2s |

进程导出指向已确认未监听端口的 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` 后，chat / embedding / rerank 分别完成 23 / 18 / 18 次成功请求，验证了本次直连路径。
成功 chat 响应汇总的供应商 usage 为 prompt 50,759 / completion 15,114 token。这不包含 embedding/rerank 的货币费用，也不能代替供应商账单。
本轮未观察到出站 HTTP 错误；无据推算具体金额。

完整 `quality_report.json` 已按来源组生成 95% CI、配对整组置换检验与 Holm 校正，并通过重算检查。
这里只有三个来源组、每题每方法一次执行，且追问能力在方法间不同；上述计数仅描述冒烟，不能据此声称准确率提升或推断完整 dev/test 表现。
逐条复核区分了缺少必答信息、缺少所引证据支持和未执行追问三类问题。另发现一项前批 dev 验收限定比语料允许的场景更窄，已写入本地 `rubric_observations.json`：本次保留原任务、试次与评分口径，不把有据的其他方案判成事实错误；全量 dev 对照前需在新任务版本中校准该项。
试次 SHA-256：`e91aac1bae18a554a17720894cbfc7e261c17c6d7b4776bc9122a7eafd875f99`。
<!-- END AGENT-V2-SMOKE -->

本轮完整审核任务在 `indexes/agent_eval/v2/reviewed/tasks.jsonl`；冒烟使用其中逐条不变的子集
`reviewed/smoke/tasks.jsonl`，`selection.json` 保存父任务集与子集指纹。样本在付费调用前选择，
未使用 test，也不依据运行结果删改失败试次。原始试次、独立标签、诊断和质量报告继续保留在
Git 忽略范围内。全量 dev/test 不包含在本次九试次冒烟范围内。

后续小节为历史运行记录。

2026-09-14 对一个开发任务进行了有限调用的真实冒烟，模型在首次 chat 调用即失败，
没有进入检索。使用受控网络权限复测未恢复；两次最小连通性诊断确认 HTTP 403，
移除进程代理环境变量后结果相同。尚不能定位具体拦截层，不能据此评价调查策略质量。
完整状态、边界和本地产物位置记录于 [迭代计划](agent-iteration.md#当前工作位置)。
本轮没有生成可用于质量结论的真实答案，也没有填写人工审核标签。

同日后续最小连通性诊断已恢复：明确直连目标 HTTPS 443、不使用环境或系统代理，
首次请求约 4.27 秒返回 HTTP 200 和 `{"ok":true}`，模型标识与结束状态校验通过。
诊断已配置 5/10/15/20/25 秒的五次重试，但首次成功即停止，实际没有重试。
该结果不能替代完整 Agent 任务冒烟，也不能证明先前 403 的具体原因已确定。

2026-09-21 使用直连传输完成了一个开发任务（`draft-01-simple`，未审核草稿）的完整冒烟，
运行目录 `indexes/agent_eval/runs/dev-agent-smoke-20260921-direct-5`。进程故意导出了指向
死端口的 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`，chat、embedding、rerank 全部请求仍成功，
证明服务组合根不再经过代理。结果：`answered`，4 次模型决策、1 次搜索、2 次读取，
4 段回答分别引用 2 条已读取证据，总耗时约 317 秒；估算输入 6,925 token。
供应商 usage 汇总 4 次成功 chat：prompt 7,532 / completion 1,518 token，另有 2 次 504 请求的
费用未知。同一冒烟中两次 chat 返回 Cloudflare **HTTP 504（约 61 秒后返回、约 850 KB HTML）**，
均在等待 5 秒后的首次重试成功，实际验证了阶梯的第一级。先前 4 次同日尝试分别因
150 秒预算不足、45 秒超时、以及旧直连实现把超大 504 错误页当作"响应超限"（不可重试）而失败；
后者已修复为先按状态抛出、只保留 16 KB 错误正文，传输合同升为 v2。
诊断文件仅含状态码、字节数、耗时、usage 与动作名，不含提示词、答案或密钥。

这一条记录只证明直连、重试与完整调查路径在当时可用；单个未审核草稿任务不能作为
Agent 质量结论。该运行的 `reviews.jsonl` 已由模型审核者（Claude）填写：12 条陈述全部被所引
段落支持，证据已按 SHA-256 与本地文档快照逐行核对。这是模型判官标签而非人工审核，
任务本身仍是草稿，报告按合同不会发布。
