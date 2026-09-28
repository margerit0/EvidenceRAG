# 文档调查 Agent 迭代计划与会话交接

更新日期：2026-09-28。工作分支：`feat/document-investigation-agent`。

本文件是本轮迭代的进度与决策入口。跨会话先读 `CLAUDE.md`、本文件，再运行
`git status --short --branch` 和 `git log -5 --oneline`；代码、测试和真实运行记录优先于文档描述。
不要在 main 上开发；尚未验证的能力不能写成已完成的项目成果。

## 新会话从这里继续（2026-09-28）

本轮最后追加的授权是一次 Grok 流式、无输出 token 上限复核；已执行且未重试。
Grok 流式连接可用，但没有收到完整最终 JSON。DeepSeek 无显式上限的独立诊断在
四次 504 后第五次成功；其最终内容与思考已分离，本次供应商报告的 completion_tokens
未超过 4096。不能宣称已经实际验证超过 4096 token，也不能把成功归因给单一参数。

<!-- BEGIN AGENT-REVIEW-GROK-STREAM -->
用户授权的单次 Grok 流式诊断：`grok-4.7` / `high`，相同复核输入，stream=true，未发送 max_tokens/max_completion_tokens，只发一次且不重试。

HTTP 200，首个 SSE 事件 1.64s；共 38 个事件，最后一个在 20.21s，思考片段累计 1384 字符，最终 content 0 字符。未保存或展示思考正文。

结果 `stream_failed` / `TimeoutError`，总耗时 110.2s；完成标记 False，finish_reason=None，usage=`{}`。未收到最终 JSON，不能标为复核成功。

流式连接能够建立并返回中间数据，但之后停流触发 90 秒读取超时；本次未出现 HTTP 504，仍未取得完整答案。没有收到 usage，实际费用未知。

本次同时使用流式和无显式 token 上限，只有一条试次，不能据此单独归因或宣称已解决 Grok 故障。保留原始数值事件记录，不自动追加付费请求。
<!-- END AGENT-REVIEW-GROK-STREAM -->

下一步若继续 Grok，应先检查约 20 秒后停止出流的供应商链路和 90 秒空闲超时；本轮
不自动增加超时或再付费。GLM 与 DeepSeek 已分别取得单条有效复核，仍需新的同期
Agent 对照计划才能比较整体行为，不能用旧模型任务失败率替代新配置测试。

最新用户要求取消本轮请求中的 4096 输出限制，并询问两模型是否做了同一任务。
已核对两次系统提示词、问题、两段草稿、证据和生成参数的指纹：除模型名外一致。
GLM 最终正文只有 647 字符且为完整 JSON，思考放在独立字段；DeepSeek 将大量分析放入
正文，输出达到 4096 token 后 JSON 未完成。旧诊断未保存思考字段长度或嵌套 usage
明细，不能把 742/4096 当作跨模型思考总量比较；输入 token 不同也可能来自分词器差异。

已增加 `ChatAnswerGenerator(max_output_tokens=None)` 和服务/对照 CLI 的
`--generation-no-token-limit`，不发送任何输出 token 限制参数，计划和指纹以 null
标识。本轮后续诊断采用此选项，不沿用旧的 4096 计划。供应商自身默认额度仍可能生效，
省略参数不保证无限输出。原有模型调用、旧计划和结果没有改写；无显式上限的 DeepSeek
诊断另存 `answer-review-deepseek-no-cap/`，保持相同输入与其他生成设置。
代码和离线回归验证与真实效果分开记录；全套 pytest 1584 passed，ruff check/format、
mypy 均通过，JUnit 在 `indexes/agent_eval/v2/output-limit/pytest-full.xml`。
输入与输出对照摘要在 `indexes/agent_eval/v2/output-limit/input_output_comparison.json`。

<!-- BEGIN AGENT-REVIEW-NO-CAP -->
无显式 token 上限诊断 `v2-review-deepseek-no-cap-20260928`：同一 DeepSeek 模型、输入、`high`、90 秒超时及最多十次总尝试，仅移除请求的输出 token 上限参数；运行时断言未发送 max_completion_tokens 或 max_tokens。

结果 `review_response_valid`，尝试 5 次，总耗时 321.5s；HTTP 错误 `{'504': 4}`，其他异常 `{}`。

成功响应 usage：`{'completion_tokens': 3354, 'prompt_tokens': 1792, 'total_tokens': 5146}`；观察到 completion_tokens 超过 4096：否；有效复核 JSON：有。失败调用的计费未知。

成功响应最终正文 985 字符，独立思考字段 11223 字符，finish_reason=stop；usage 思考分项 `{'accepted_prediction_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}`。若非空思考字段却报告 reasoning_tokens=0，不能用该分项推断实际思考用量。

省略请求上限不取消供应商默认额度或模型最大输出长度，也不能解决中转自身的等待超时；本次真实结果与离线选项验证分开记录。
<!-- END AGENT-REVIEW-NO-CAP -->

### GLM 同输入诊断（历史）

最新用户要求沿用 DeepSeek 的 key 和地址，只把模型名改为 `z-ai/glm-5.3` 测试。
已完成同一输入、`high`、4096 输出上限的一次复核调用，正常返回完整 JSON 并通过校验。
本轮仅补充本地诊断和交接文档，产品代码仍为 `2275144`；继续在
`feat/document-investigation-agent`，未修改 main、原 `.env` 或旧实验记录。

<!-- BEGIN AGENT-REVIEW-GLM -->
同输入诊断 `v2-review-glm53-20260928`：沿用 DeepSeek 的独立凭证文件与地址，仅请求模型改为 `z-ai/glm-5.3`；保持 `high`、4096 token 上限、90 秒超时和最多十次总尝试。

首次请求 37.7s 返回 2xx，返回模型匹配，finish_reason=`stop`；共 1 次，无重试。供应商 usage：`{'completion_tokens': 742, 'prompt_tokens': 1782, 'total_tokens': 2524}`。

最终正文为完整 JSON（647 字符），通过复核结构与引用边界校验；四项覆盖、两个段落支持度及条件范围均通过。供应商使用独立 reasoning_content 字段，正文未混入非 JSON 文本；本次输出未触及上限。

独立模型检查与原题和所引 FAQ 一致，人工审核仍为零。此结果只验证该配置能正常完成这一条复核调用，不代表完整 Agent 对照成功或复核具有泛化提升。

后续同类诊断可沿用最多十次总尝试、成功即停；本次首次即成功，不能据此证明额外重试的收益。完整 Agent 六试次需另建新模型计划。旧 Grok/DeepSeek 结果、原 .env 和冻结 test 保留。
<!-- END AGENT-REVIEW-GLM -->

下一步可准备 GLM 同期 off/on 六试次，验证完整调查、复核与追问流程；不要把本次
单条复核通过当成整体收益，复核开关继续默认关闭。DeepSeek 的输出格式问题单独保留，
若继续排查，应另建实验并只改变一项设置，不与 GLM 结果混合归因。

### DeepSeek 诊断与此前交接

最新用户要求停止旧模型诊断，立即改用提供的 `deepseek-v4.1-flash` 配置。
已终止旧进程并保存独立中断记录，原记录中有九次完整 504，不能写成十次全失败。
新配置只执行同一简单题答案的单次复核诊断（同一输入指纹、`high`、4096 输出 token）；
没有覆盖原 `.env`，凭证位于 Git 忽略的独立文件，报告不记录密钥。
本轮实现、重试修正、测试及全部结果将按用户授权在本分支提交，实际提交以 Git log 为准。
完整门禁通过；尚未证明语义复核有效，开关继续默认关闭，不扩大 dev/test。

<!-- BEGIN AGENT-REVIEW-DEEPSEEK -->
新配置诊断 `v2-review-deepseek-20260928`：请求 `deepseek-v4.1-flash` / `high`；同一个已保存简单题答案、证据与复核提示词，输入指纹与旧十次诊断相同。

首次请求在 36.1s 返回 2xx，返回模型匹配；共 1 次，没有发生传输重试。结果 `invalid_review`，有效复核响应：无。

成功响应 usage：`{'completion_tokens': 4096, 'prompt_tokens': 1792, 'total_tokens': 5888}`。正文不是完整 JSON，包含非 JSON 说明文字，输出达到 4096 token 上限且末尾 JSON 不完整；没有提取片段或绕过结构校验。

这一条证明新配置能够返回响应，但未跑通现有复核合同；不能作为整条 Agent 任务成功或语义复核通过，也不能据此采用十次默认策略。尚未用新模型重跑六试次对照。

新凭证仅存于 Git 忽略的独立环境文件，原 .env 未修改；原 grok 配对结果和中断记录保持原样。下一步先验证新供应商在 high + JSON 输出模式下能否可靠返回完整最终 JSON，再决定是否接入 Agent 复核。
<!-- END AGENT-REVIEW-DEEPSEEK -->

下一步先定位新供应商的结构化输出兼容性：发送最小固定 JSON 合同，保持 `high`，
检查是否把非最终文本放进 `content`、是否遵守 `response_format`、是否正确报告截断。
这项诊断须另存新计划与结果，不增加上限掩盖当前失败；确认返回完整最终 JSON 后，
再用新模型准备同期 off/on 六试次，而非与 grok 历史结果直接相减归因。
继续保留失败并独立审核；十次默认策略的条件尚未满足，没有改成无限或十次通用重试。

### 旧模型十次上限诊断（用户中止）

用户新增授权：六次仍失败时先做一次最多十次总尝试的诊断，成功后后续实验再采用十次。
十次按首次 + 9 次重试执行，不修改正在执行的六试次配置，另存为
`indexes/agent_eval/v2/answer-review-ten-attempts/`；只复核已保存的简单题答案及证据。
复核原始草稿未保存，故不是失败请求的逐字重放。结果由
`indexes/agent_eval/v2/review/report_review_ten_attempts.py --check` 认证。

<!-- BEGIN AGENT-REVIEW-TEN-ATTEMPTS -->
单次复核诊断 `v2-review-ten-attempts-20260928`：`grok-4.7` / `high`，最多 10 次总尝试（首次 + 9 次重试），已记录 9 次完整请求。用户中止，最后未记录的在途请求及总耗时未知。

结果 `user_interrupted`；HTTP 错误 `{'504': 9}`，其他异常 `{}`；有效复核响应：无。

成功响应 usage：`{}`；失败尝试是否计费及金额未知。

只复核已保存的 off 简单题答案及其核验过的证据，没有新调查、embedding 或 rerank；失败的 on 草稿未持久化，这不是原请求逐字重放。此诊断独立于六试次配对报告。

用户要求不等十次，旧模型诊断已停止并转向新配置；不能记录为“十次全部失败”，旧配置未满足采用十次的条件。
<!-- END AGENT-REVIEW-TEN-ATTEMPTS -->

用户指出后台常见失败后重试成功，现已核实首次配对实验并非没有重试：唯一复核请求
尝试六次（504 ×4、连接重置 ×1、URLError ×1），之后四个试次的 URLError 未被旧策略
识别，首错即停。旧日志没有 reason，不能确定这五次 URLError 的具体根因，也不能据此
证明后台迟到成功。已补齐临时 DNS、远端断开、不完整响应和暂时性 socket 错误的有限
重试，永久 DNS、证书/其他 TLS、权限/配置错误仍停止。次数及 5/10/15/20/25 秒等待不变，
成功立即返回，策略版本进入生成器指纹。

重跑使用 `v2-answer-review-retry-20260928-{off,on}`，仍为原三条 dev 各一次，固定
`grok-4.7` / `high` 和原预算。新诊断提供本地 request_seq、attempt、UTC 时间、reason 类型
及数字错误码；不写异常正文、请求正文或密钥。旧试次、旧计划和已认证执行代码快照均保留，
新报告入口为 `indexes/agent_eval/v2/review/report_answer_review_retry_pair.py --check`。
旧报告继续通过 `report_answer_review_pair.py --check` 校验原实验。完整 pytest、ruff
check/format、mypy 已通过，当前 JUnit 位于 `indexes/agent_eval/v2/answer-review-retry/`。

<!-- BEGIN AGENT-ANSWER-REVIEW-RETRY -->
答案复核实验（`v2-answer-review-retry-20260928`）：`grok-4.7` / `high`，3 条 dev、off/on 共 6 试次，均已独立模型审核；人工审核 0。
全套离线门禁：1581 tests passed，ruff check / format / mypy 通过。

| 配置 | 状态 | 验收通过 | 引用支持陈述 | 平均模型调用 | 平均耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| review off | answered 2、clarification_needed 1 | 2/3 | 27/28 | 6.0 | 155.6s |
| review on | clarification_needed 1、generation_failed 2 | 1/3 | 0/0 | 7.7 | 549.2s |

配对成功率差（on − off）：-0.333，95% CI [-1.000, 0.000]；整组置换 p=1.000，Holm p=1.000。
成功请求：`{'chat': 39, 'embedding': 18, 'rerank': 18}`；HTTP 错误：`{'504': 15}`；其他请求异常：`{}`。
成功 chat usage：prompt 174,208 / completion 42,074；失败请求及 embedding/rerank 金额未知。
每题每配置只运行一次，且题目已用于开发；模型审核不计人工审核。此小样本不足以证明泛化提升或抵消供应商耗时波动。
未达到扩量门槛：保留全部失败记录，复核开关继续默认关闭，暂不扩大 dev。
成功返回的复核响应：0；发生重试的请求：3，其中最终成功：1。错误类别：`{'http': 15}`。
新运行两侧均使用 transient-network-v2，不能把与旧运行的差异单独归因于重试修正。旧六试次和失败记录保持原样。
<!-- END AGENT-ANSWER-REVIEW-RETRY -->

### 首次配对实验与实现

用户要求按计划继续执行。已完成离线失败清单、[答案复核设计](agent-answer-review.md)、
默认关闭的 `--agent-review-answers` 开关与回归测试。复核与补救共享既有预算；开启时
答案先作为草稿，首次未通过可补救一次，修订草稿再次未通过则停止；追问/拒答不额外复核。
同一生成器配置下关闭时保留 `3503741` 的 Agent profile 和行为。后续传输重试升级会
同时改变 off/on 的生成器及 Agent 指纹。此能力仍为实验，不能视为语义正确性证明。

本轮实现基于 `fb3f19d`，按用户既有提交委托保存阶段成果，实际提交以 `git log` 为准。
定向和完整离线门禁已通过，JUnit 位于
`indexes/agent_eval/v2/answer-review/`，README 已由完整报告同步。
本轮按已接受计划限定为六个 dev Agent 试次：三题各 off/on 一次，按题交错执行。
`pair_plan.json` 绑定任务、预算和代码指纹，固定 `grok-4.7` / `high`；运行入口为
`indexes/agent_eval/v2/review/run_answer_review_pair.py`，默认离线，`--run` 才付费。
两个新 run-id 为 `v2-answer-review-20260928-off` 与 `v2-answer-review-20260928-on`，
六试次已执行并独立审核；旧任务、冻结 test、试次和评分保持原样。复核阶段出现连续
504 和连接重置，后续 URLError 根因未被原日志记录，本轮没有成功复核响应，未达到扩量门槛。

后续已按用户反馈修正重试并准备新的同模型、任务及预算六试次计划，进度见文首。
复核开关继续默认关闭，分层样本和全量 dev 暂不执行。
本地运行器的最终发布曾把内存 tuple 送入要求 JSON list 的校验器；已改为从磁盘读取
canonical JSON 后认证，恢复记录与实际执行脚本快照保存到同一实验目录，原始试次未改。
旧计划绑定的是归档代码，不得原地覆盖或重用旧 run-id。离线复核入口：

```powershell
.venv/Scripts/python.exe indexes/agent_eval/v2/review/report_answer_review_pair.py --check
.venv/Scripts/python.exe indexes/agent_eval/v2/review/calibrate_dev.py
```

<!-- BEGIN AGENT-ANSWER-REVIEW -->
答案复核实验（`v2-answer-review-20260928`）：`grok-4.7` / `high`，3 条 dev、off/on 共 6 试次，均已独立模型审核；人工审核 0。
全套离线门禁：1552 tests passed，ruff check / format / mypy 通过。

| 配置 | 状态 | 验收通过 | 引用支持陈述 | 平均模型调用 | 平均耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| review off | answered 1、generation_failed 2 | 1/3 | 9/9 | 2.0 | 66.9s |
| review on | generation_failed 3 | 0/3 | 0/0 | 2.7 | 185.3s |

配对成功率差（on − off）：-0.333，95% CI [-1.000, 0.000]；整组置换 p=1.000，Holm p=1.000。
成功请求：`{'chat': 9, 'embedding': 5, 'rerank': 5}`；HTTP 错误：`{'504': 4}`；其他请求异常：`{'TimeoutError': 1, 'ConnectionResetError': 1, 'URLError': 5}`。
成功 chat usage：prompt 33,360 / completion 6,746；失败请求及 embedding/rerank 金额未知。
每题每配置只运行一次，且题目已用于开发；模型审核不计人工审核。此小样本不足以证明泛化提升或抵消供应商耗时波动。
本轮没有成功返回的复核响应，后续连接故障同时影响 off/on；不能把这些任务失败解释为语义复核有效或无效。
最终清单发布的 tuple/list 类型兼容问题已离线修复：重新读取磁盘 JSON 认证，保留原元数据与实际执行脚本快照，六条原始试次未改写。
未达到扩量门槛：保留全部失败记录，复核开关继续默认关闭，暂不扩大 dev。
<!-- END AGENT-ANSWER-REVIEW -->

### 本轮计划的起点（历史）

已按用户授权将本轮实现、测试及审核记录提交为 **`3503741`**
（`feat: honor agent reasoning effort and record reviewed dev runs`）。本节行动计划另作
文档提交；实际 HEAD 看 `git log`。继续使用 `feat/document-investigation-agent`，
main 保持 `0668fa9`，没有推送或合并。提交前全套门禁及本地报告认证通过，记录见下节。

当前阶段已具备可复现基线，但还没有证明 Agent 的整体收益。下一轮先解决已观察到的
证据遗漏和条件误读，维持简单题与必要追问的表现；不直接启动全量 dev。
本次委托是提交与规划，没有新增付费实验。模型审核仍不等于人工审核。

### 下一轮行动顺序与完成条件

| 顺序 | 行动 | 交付与完成条件 |
| --- | --- | --- |
| 1 | 离线复盘失败路径 | 从已认证 dev 试次、`claim_audit.jsonl` 和返回证据建立错误清单，逐个区分未读取机制证据、已读却遗漏、把充分条件升级为必要条件；每项绑定试次和证据指纹。实际材料留在 `indexes/`。 |
| 2 | 设计并实现有预算的答案复核 | 先以显式实验开关接入：对拟发布答案检查问题覆盖和条件/例外的证据依据，缺证据则利用剩余额度补查；将草稿与可发布答案分开。设置、复核提示词和流程版本进入配置指纹。 |
| 3 | 完成离线控制流验证 | 使用原创合成材料检查复核发现缺口后的补查/修订、非法引用、复核失败、预算耗尽和取消；简单题和追问不回退。所有复核调用计入现有步数、时间和 token 预算，不能新增不受控循环。 |
| 4 | 做同配置定向 dev 回归 | 准备当前基线与新流程在相同 dev 题目上的配对运行清单，固定模型、思考等级、索引、任务和预算；新 run-id 分开保存不同流程配置，独立审核全部输出并报告额外调用与耗时。 |
| 5 | 根据回归结果分阶段扩量 | 已知错误消除且简单题/追问未回退后，再准备覆盖四类问题及各 dev 来源组的分层样本；先审阅计划与费用范围，再考虑全量 dev。最终方法确定后才安排冻结 test。 |

步骤 1 的入口是 `indexes/agent_eval/runs/v2-cal-smoke-20260928-high/`：
先读 `review_decisions.json`、`claim_audit.jsonl` 和 `evidence_integrity.json`，再对照
`trials.jsonl` 中实际读取的证据。不能只凭检索次数断定机制段落曾被读到，不能用未引用的
文档替答案补证据。开发任务以 `calibrated-20260928/tasks.jsonl` 为准，旧评分保持原样。

步骤 2 先落一个最小设计，再改 `src/zhrag/agent.py`；确需调整组合根和运行清单时同步
修改 `scripts/serve.py`、`scripts/compare_agent.py` 与配置指纹。复核应返回可核对的
问题覆盖项、条件/例外与证据 ID，不输出内部推理过程；最多一次补救循环，具体调用预留
在设计中写清。仍由 Python 校验结构和引用边界，模型的“复核通过”不是语义正确性证明，
也不能用是否出现“如果/只有”等关键词代替语义判断。现有追问终态不强制生成答案草稿。

步骤 3 的合成回归只证明编排遵守复核结果与预算，真实条件理解必须在步骤 4 独立审核。
拒绝把语料句子复制进单元测试。完成修改后跑 pytest、ruff check、ruff format --check、
mypy，并按新的完整 JUnit 报告同步 README；再生成与实际代码指纹匹配的实验计划。

步骤 4 优先复用本轮简单、跨文档和追问的三个 dev 场景：基线/新流程各执行一次，
建议总计六个 Agent 试次。只改变复核流程，沿用已确认的 `grok-4.7` / `high` 与同一
预算，先核对 `.env` 的模型配置而不记录密钥。当前九试次历史报告可用于定位问题，
不能代替与新流程同期运行的严格对照。单次小样本只作开发诊断，不发布泛化或显著提升结论。

进入步骤 5 的条件：跨文档题补齐有据的机制解释，不再把充分条件当唯一前提；简单题
保持正确，缺少决定性信息时仍能追问；结构、预算和失败处理全部通过。任何条件未满足，
继续针对性修正并保留失败记录，不能通过删题、改评分或切换模型掩盖退化。
可先规划十二条 dev（四类各三条，覆盖六个来源组）与三种方法的对照，再决定是否执行
全量 dev。上述六试次、分层样本及全量运行均为后续提案，须在新调用前明确规模和预算。
若再次更换 key/model 或思考等级，另开实验配置，不把结果归因给唯一流程变更。

### 本轮已提交成果与验证记录

本节记录基于 `e52a969` 完成的 dev 验收校准、Agent 提示词修正，以及对照 CLI 的
思考等级参数和测试。此前文档改动一并保留，README 测试数量由脚本同步。
用户已授权判断并提交本轮成果；实际提交以 `git log` 为准，没有推送或合并。

校准版本另存 `indexes/agent_eval/v2/calibrated-20260928/`。只修正冒烟发现的旧 dev
验收限定，并记录原审核者、复核者、前后条件与源文依据；原任务、证据、冻结记录及旧试次
保持原样。模型审核不计人工审核。FAQ 仅作为校准佐证，正式任务的引用和来源组不变。

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

失败分析：一次调查追加搜索后没有读取新候选，就用已读段落推导了更强的禁令；另一次
已经读取包含模式差异的证据，却以分支建议代替必要追问。现在提示词明确要求核对用户
问题的各个部分、读取所用证据、避免将建议升级为禁令或保证，并在具体操作选择依赖
未知环境信息时追问。一般性的方案比较仍可直接回答，已提供的信息不重复询问。
这项提示词修正已完成有限真实回归并暴露剩余问题，不表示已建立语义校验器或已证明质量提升。

运行入口默认只做离线检查与预览；`--run` 才调用付费接口。已生成的两份计划绑定任务、
提示词及关键代码指纹；运行前必须匹配，使用新 run-id，拒绝覆盖旧目录。
基线的问答与动作合同未改，单轮 RAG / 固定流程仍无追问动作，因此不能把整体对比
解释成单一因素消融。新任务条件不得用于回写旧试次的评分。

```powershell
.venv/Scripts/python.exe indexes/agent_eval/v2/review/calibrate_dev.py
.venv/Scripts/python.exe indexes/agent_eval/v2/review/run_calibrated_dev.py --scope smoke --run-id v2-cal-smoke-20260928-high --reasoning-effort high
.venv/Scripts/python.exe indexes/agent_eval/v2/review/run_calibrated_dev.py --scope full-dev --run-id v2-cal-full-dev-20260928-high --reasoning-effort high
.venv/Scripts/python.exe indexes/agent_eval/v2/review/sync_calibrated_docs.py --check
```

用户已选择先跑九试次 dev 回归。首次沙箱尝试在嵌入阶段连接失败后中止，记录保留；
旧配置的 `v2-cal-smoke-20260928-direct` 在用户修改 `.env` 的 key/model 后停止，已有
六个试次和第七个中断记录保留。用户要求使用新设置并改为 `high`；当前运行改为
`v2-cal-smoke-20260928-high`，请求逐次校验模型名与思考等级，密钥不记录。
本轮九试次、逐条模型审核、证据重建校验和质量报告均已完成。仍存在条件误读与机制
解释遗漏，下一步应先针对这两类问题改进 dev 上的证据覆盖与判断，再决定是否扩大；
不能自动启动全量 dev。最终 test 不用于调参。
人工审核、页面视觉验收仍为后续事项。本轮按用户委托在当前功能分支保存提交。

## 2026-09-26 交接（历史）

本节取代下面的历史交接。当前 HEAD 为 `e52a969`；本轮完成任务审核、文档和有限付费冒烟，
没有提交、推送或合并。仍在 `feat/document-investigation-agent` 分支。

用户委托模型续审剩余五组。前批四十条审核记录保持其原审核者，本批三十九条记录实际模型审核者；
**模型审核不计入人工审核，人工审核仍为零**。原始 `v2/tasks.jsonl`、`draft_evidence.jsonl`
和 `drafts/` 保留为未审核出处；审核后的任务、证据、聚合摘要及冻结记录写到
`indexes/agent_eval/v2/reviewed/`。审核结论和复核脚本位于同级 `review/`，所有这些产物均被 Git 忽略。

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

本轮统一口径：验收只要求问题直接询问且原文支持的事实，接受等价表述；跨文档题核查文档间的
重复内容；追问题只检查决定性信息与不预设结论；无答案题重新搜索全部本地语料，并按
`insufficient_evidence` 且无答案段落验收。语料中冲突的数值不作为唯一金标准。
最终额外检查了引文来源的组归属，清除一处冗余跨组引用；装配器自身并不证明语义正确或来源组独立。

本次付费范围为三个不同来源组的 dev 任务、三个方法，共九个试次。样本在调用前选定，覆盖简单、
跨文档、追问三类；副本在 `reviewed/smoke/tasks.jsonl`，`selection.json` 绑定完整任务集指纹。
这使三个方法能共享同一输入，并避免默认 `--limit 3` 只取到同组简单题。
test 已冻结，未参与本轮运行或调参。全量 dev/test 仍须另行确认规模与预算。

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

复核入口（本地脚本不调用付费接口，除非显式加 `--run`）：

```powershell
.venv/Scripts/python.exe indexes/agent_eval/v2/review/finalize_review.py
.venv/Scripts/python.exe indexes/agent_eval/v2/review/run_reviewed_smoke.py
```

`finalize_review.py` 默认只读重算任务、引文、出处指纹与冻结信息；不要对已冻结任务重新执行
`--write`，不要用 test 结果改任务、提示词或参数。后续只在新 run-id 下安排已授权的实验。
只有用户明确要求时才提交。页面视觉验收、人工审核和更大规模的对照实验仍是后续工作。

环境复核：2026-09-26 C 盘可用约 21.2 GiB，旧的极低空间告警已不代表当前状态。
日志与报告继续写到 D 盘 `indexes/`。本轮定向离线测试为 **96 passed**；首次运行因既有
`.pytest_tmp` 权限失败，按环境权限流程重跑后全部通过。没有设置 `PYTHONUTF8`。

## 2026-09-21 第二次交接（历史）

本节保留当时进度，最新状态以上节为准。不要 reset、clean、重新创建分支或重做已提交功能。

- 工作目录：`D:\rag`，分支：`feat/document-investigation-agent`；main 保持 `0668fa9`；没有合并或推送。
- 本会话新增并提交：语料落地任务草稿装配器（`src/zhrag/eval/agent_task_drafts.py`、
  `scripts/assemble_agent_tasks.py`、`tests/test_agent_task_drafts.py`）与 `docs/agent.md` 的说明。
- 提交前全套门禁：**pytest 1510 passed**、ruff check、ruff format --check、mypy 均通过；
  README 测试数量已由 `scripts/sync_quality_gate_docs.py` 重出并 `--check` 通过。
- 当时 C 盘空间不足，pytest 全量运行曾因写不下临时输出出现 4 个假失败，单独重跑全部通过。
  当前空间情况见最新交接；日志继续写到 D 盘 gitignored 位置（`.pytest_tmp` 启动时会被清空）。

本会话完成的任务集扩充（全部位于 `indexes/agent_eval/v2/`，已忽略，不入库）：

| 项目 | 结果 |
| --- | --- |
| 起草方式 | 10 个模型起草 agent，各领一个主题与限定文档清单，按 `drafts/AUTHORING_PROMPT.md` 起草；语料仅为本地 `tidb-rag-curated` 快照（pingcap/docs-cn@26f202b）中已入索引的 450 篇 |
| 规模 | **79 条未审核草稿**：simple 30 / multi_document 20 / clarification 19 / unanswerable 10；10 个来源组，dev 48（6 组）/ test 31（4 组） |
| 机械核验 | 346 段引文全部在所引文档中逐字存在（装配器规范化后）；所有 `reference_sources` 均在索引内；没有文档被两个来源组同时引用 |
| 装配修正 | 首轮 13 处失配中 12 处为规范化缺口（链接方括号残留、`<span>` 版本标记、`+` 列表、Hugo `{{< copyable >}}`），已扩展规范化器并加测试；1 处（`mem-04` 第 3 段引文）为起草者省略句子，已按文档逐字补全并在 `author_notes` 标注 |
| split 分配 | backup-restore 放 dev（此前冒烟已用过该主题）；test = slow-query-tuning、transactions-locks、ai-vector-search、dev-guide-sql |
| 状态 | `reviewed=false`、`reviewer` 为空、`snapshot="pingcap/docs-cn@26f202b"`；`compare_agent.py --tasks indexes/agent_eval/v2/tasks.jsonl --allow-drafts --split dev` 可离线预览；test split 按合同拒绝草稿 |

起草 agent 在 `author_notes` 里留下的审核提示（人工审核时优先看）：

- 语料缺 `system-variables.md`、`temporary-tables.md`、`upgrade-tidb-using-tiup.md` 等被大量链接的文档，
  多条 unanswerable 以此为基础；审核者需确认这些确实不该由 Agent 凭先验知识补答。
- 语料自身不一致处已被刻意避开或标注：事务大小限制（两篇文档数值冲突）、`--ratelimit` 口径、
  `min-blob-size` 单位写法、`max-down-time` 拼写、`noatime` 是否必选。
- 若干问题故意含干扰信息（错误版本归因、超出 GC 窗口的时间），用于区分"附和用户"与"依据文档纠正"。
- `imp-06` 的引文是 HTML 表格片段，须对照源码而非渲染表。
- `txn-07` 的 clarification 只依赖单篇文档；若审核者要求跨文档可替换。

v1 的 24 条模板草稿保持原样，`dev-agent-smoke-20260921-direct-5` 的任务集指纹仍有效。

下一会话按这个顺序继续：

1. 阅读 `CLAUDE.md` 和本文，核对 `git status --short --branch` / `git log -3 --oneline`。
2. 当时计划由人审核 v2；之后用户明确委托模型审核，执行记录见最新交接。
   必须区分模型审核与人工审核，完成语义复核后才能标记 `reviewed=true`，最终 test 不用于调参。
3. 大规模付费对照试验仍未授权，须先确认预算。参考经验值：`--max-seconds 600`（上限就是 600）、
   `--generation-timeout 90`、`--generation-retries ≥ 2`；单次 chat 决策 20–60 秒、偶发 61 秒 504。
   79 任务 × 3 方法 ≈ 237 试次，按上次冒烟每试次 4–6 次 chat 估算调用量，费用需以供应商账单为准。
4. 当时小规模 dev 冒烟尚待授权；本轮已获授权的范围见最新交接。
5. 页面视觉验收仍为独立待办。

## 2026-09-21 第一次交接（已由上节取代，保留为历史）

本会话的验证与修复已提交为 `33b4a99`；此后仅 `docs/agent-iteration.md` 与 `docs/agent.md`
有一次审核记录更新，可能仍未提交（只含文档，直接提交即可）。
不要 reset、clean、重新创建分支或重做已提交功能。

- 工作目录：`D:\rag`，分支：`feat/document-investigation-agent`。
- HEAD：`33b4a99`（直连传输、可选重试、故障状态、置换检验、文档），父提交 `608515a`；
  main 保持 `0668fa9`；没有合并或推送。
- 提交前全套门禁：**pytest 1498 passed**、ruff check、ruff format --check、mypy 均通过；
  README 测试数量已由 `scripts/sync_quality_gate_docs.py` 重出并 `--check` 通过。
- 已完成一个开发任务的真实完整冒烟（`answered`，带引用），见下文与 `docs/agent.md`。
- 语料、任务、试次、诊断产物均在 `indexes/` 忽略边界内，未进入提交。

本会话验证并修正的内容：

| 项目 | 结果 |
| --- | --- |
| 直连传输 `src/zhrag/providers/direct.py` | 单元测试通过；真实运行在进程导出死代理端口 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` 时 chat/embedding/rerank 全部成功，证明不经代理。**修复**：原实现先读正文再判状态，Cloudflare 504 的约 850 KB HTML 错误页被当成"响应超限"（ValueError，不可重试）；现改为非 2xx 先按状态抛出、只保留 16 KB 错误正文。合同升为 `http-client-direct-no-redirect-v2` |
| `--generation-retries` / `--agent-generation-retries` | 新增回归：范围 0–5 校验、CLI 透传、指纹随重试次数变化、`_build_agent` 在读密钥前校验预算。真实冒烟两次 504 各在等待 5 秒后重试成功 |
| 检索故障状态 | `abstain` 与 `answerable:false` 均返回 `retrieval_failed`；新增 HTTP 层回归确认 503 |
| 整组置换检验 | 新增测试通过；**修复**统计标签含 `group-` 前缀导致隐私泄漏断言误报，标签改为 `two-sided-label-permutation-within-source-groups` 等 |
| `_direct_loopback` | 新增测试：保留既有 `no_proxy` 条目、去重、不改动 `HTTPS_PROXY`；Windows 下 `os.environ` 大小写不敏感，两种拼写共用一个键 |
| 传输指纹 | `transport_contract` 进入生成配置指纹，空值拒绝；新增测试 |

真实冒烟记录（均在 `indexes/agent_eval/runs/` 与 `indexes/agent_eval/diagnostics/`，已忽略）：

| run-id | 参数 | 结果 | 原因 |
| --- | --- | --- | --- |
| `dev-agent-smoke-20260921-direct` | 150s 预算 / 45s 超时 / 2048 tok | `budget_exhausted`，1 次决策 | 首个 chat 决策耗时超过预算（相同请求在诊断中 7 秒返回，供应商延迟波动大） |
| `-direct-2` | 420s / 90s / 2048 | `generation_failed`，4 次决策、1 次搜索、2 次读取 | 最后一次 chat 收到超大 504 页，旧直连实现判为超限 |
| `-direct-3`、`-direct-4` | 420s / 90s / 4096 | `generation_failed` | 同上，带逐调用诊断确认 `provider response exceeded size limit` |
| `-direct-5` | 600s / 90s / 4096 / 5 次重试 | **`answered`**：4 决策、1 搜索、2 读取、4 段引用 2 条证据、317 秒 | 传输修复后；两次 504 各重试一次成功 |

诊断文件 `20260921-chat-body-composition.json` 记录同一提示词三次直连：两次 3–5 秒返回 200，
第三次约 61 秒后返回 504、`Content-Length` 847,662、HTML；这是修复依据。
所有诊断只含状态码、字节数、耗时、usage 与动作名。

下一会话按这个顺序继续：

1. 阅读 `CLAUDE.md` 和本文，核对 `git status --short --branch` / `git log -3 --oneline`。
2. `dev-agent-smoke-20260921-direct-5/reviews.jsonl` 已由**模型审核者**（Claude）填写：
   12 条陈述全部被所引段落支持，`task_success=true`；证据段落已按 SHA-256 与本地
   `tidb-rag-curated` 快照（commit `26f202b`）逐行核对，仅 markdown 链接被切块器去除。
   这属于模型判官标签，不计入人工审核数量；任务本身仍是 `reviewed=false` 的草稿、验收条件为空，
   `--report` 按合同拒绝发布。需要人工确认时，改写 reviewer 字段并复核。
3. 扩充并审核任务集（目标 50–100 条）；大规模付费对照试验仍未授权，须先确认预算。
   基于本次观测：单次 chat 决策 20–60 秒、偶发 61 秒 504，建议正式运行 `--max-seconds 600`（允许范围 (0, 600]）、
   `--generation-timeout 90`、`--generation-retries ≥ 2`；这些是经验值，不是合同。
4. 页面视觉验收仍为独立待办。

可先运行的定向测试：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_direct_transport.py tests/test_agent.py tests/test_agent_service.py tests/test_agent_review.py tests/test_review_agent_script.py tests/test_answering_provider.py tests/test_serve.py tests/test_compare_agent_script.py --tb=short
```

本机 `.pytest_tmp` 可能需要受控权限；按环境权限流程运行，不更改 `PYTHONUTF8` 或随意删除目录。

## 2026-09-17 交接（已由上节取代，保留为历史）

当时未提交修复仅通过静态检查，新增测试未执行、真实完整任务未复测。两处修复的可复现原因：

1. 所有检索报错后，模型用 `answer` 搭配 `answerable:false` 结束，会绕过 `abstain` 的故障判断，
   返回 `insufficient_evidence`。修复后应为 `retrieval_failed`，HTTP 对应 503。
2. 只有两个独立来源组、组内差值一致时，旧中心化 bootstrap 可产生 p≈0.0001 并宣称显著。
   新精确整组置换在该例给出双侧 p=0.5；不要把 bootstrap CI 当成小样本显著性的补救。

用户要求的重试语义：首次失败后等待 5、10、15、20、25 秒，最多五次重试，成功立即停止。
正式 adapter 只重试既有状态码/瞬态故障，403 不在其中；默认 0 次重试。

## 用户目标与授权

- 项目从技术文档检索与问答扩展到可验证的文档调查 Agent。
- 用户认可从 RAG 扩展成有工具、有调查过程、有对照评测的领域 Agent。
- 用户要求创建新 Git 分支开展工作，不在 main 修改。
- 用户要求维护本文件，用于跨会话同步进度与记忆。
- 文档使用工程目标、实现决策与验证记录表述。
- 用户选择暂不使用浏览器自动化工具，继续完成代码与离线验证。
- 当前先交付可验证的文档调查 MVP，再按证据决定是否接 SQL 执行计划与日志工具。
- 2026-09-26 已获准完成剩余 v2 模型审核、写出、文档及有限付费测试；当前执行范围为
  三条 dev 任务 × 三方法。此前单任务授权记录保留为历史，全量 dev/test 尚未确认规模。

## 起点与选择理由

从 main 的 `0668fa9` 创建新分支；创建时工作区干净。
已有混合检索、重排、增量索引、统计评测、FastAPI、单轮带引用问答和静态页面。
现有 `/api/ask` 是固定检索后生成；引用检查验证格式和 ID，不证明语义支持。
既有 TiDB 评测包含同模型生成的合成标签；M8 性能 profile 不含在线模型调用。
这两类结果均不能改写成 Agent 成功率或端到端响应速度。

主要增量：根据工具结果选择下一步、补充检索、请求必要信息、控制执行预算，
并通过相同任务集比较单轮 RAG、固定工作流和自适应 Agent。
前端负责让这些能力可演示；优先复用已有页面。

## 阶段计划与完成条件

### A：有限步骤的文档调查 MVP

- [x] 创建并切换到 `feat/document-investigation-agent`。
- [x] 新建本计划及跨会话入口。
- [x] 编写纯编排核心 `src/zhrag/agent.py`，注入检索器、模型和时钟。
- [x] 受限动作：`search_docs`、`read_passage`、`answer`、`clarify`、`abstain`。
- [x] 搜索展示候选摘要，读取后才允许引用完整证据，引用 ID 在一次调查内稳定。
- [x] 调用步数、搜索次数、读取次数、上下文和累计估算输入 token 预算。
- [x] 重复搜索抑制、白名单动作校验、固定故障状态与执行事件。
- [x] `/api/investigate` 接入与 `--enable-agent` 显式开关。
- [x] 复用服务并发门禁；客户端中断后在下一依赖边界停止。
- [x] 页面增加文档调查模式、执行记录、证据和停止按钮。
- [x] 核心回归：补充搜索、追问、无证据、非法引用、工具错误、重复动作、预算与取消。
- [x] HTTP 回归：开关、输入约束、共享并发、故障脱敏、取消后释放时机。
- [x] 页面 JavaScript 语法检查；合成依赖验证调查与 HTTP 流程。
- [ ] 页面视觉验收与浏览器交互演示（本轮按用户要求暂跳过）。
- [x] pytest、ruff check、ruff format --check、mypy 全套门禁通过。
- [x] README 入口与 [用户操作说明](agent.md) 完成。

完成条件：不需密钥或语料就能测试控制逻辑；在现有服务组合根中可显式启用；
能演示补充搜索、追问和拒答；不将合成测试写成真实任务效果。

### B：对照评测与人工任务集

- [x] 建立任务格式、审核状态、开发集/测试集与来源分组约束。
- [x] 提供本地任务草稿初始化及校验入口；任务正文、证据和模型产物留在 gitignored 目录。
- [ ] 起步目标为 50–100 个经人工审阅的任务；草稿不计入人工审核数量。
      2026-09-26：v2 的 79 条已完成用户委托的模型审核，0 条人工审核；原草稿仍保留。
- [x] 完成用户委托的 v2 模型审核、单独写出及出处/引文/组归属校验。
- [x] 草稿覆盖简单文档问答、跨章节整合、信息不足、无答案；诊断任务待真实工具接入后补充。
- [x] 实现同任务对照单轮 RAG、固定拆解检索流程和自适应 Agent 的运行入口与合成测试。
- [x] 记录调用次数、估算输入 token 和耗时，预留任务成功/支持度/追问拒答的人工审核字段。
- [x] 添加独立审核准备、完整性校验、聚类 bootstrap CI、配对检验和 Holm 校正入口。
- [x] 对照 CLI 可配置模型决策、搜索、读取、时长和输出上限；执行前清单展示预算。
- [ ] 人工审核完成后汇总任务质量；实际成本需供应商 usage/账单数据，不用估算输入冒充费用。
- [x] 模型/提示词/索引/任务集指纹及每个 run 独立产物目录。
- [x] 模型审核后冻结 test 及其指纹；禁止在最终测试集反向调参。
- [ ] 人工审核标签与模型判官区分；对提升结论使用配对比较及 95% CI。
- [x] 完成有限 dev 付费冒烟及独立模型审核，数字从认证产物生成。
- [x] 全量 dev 前另存新任务版本，校准冒烟发现的一项旧 dev 验收限定。
- [x] 执行修正后 dev 真实回归，并在用户更换 key/model 后以 high 重跑、完成模型审核。
- [ ] 解决回归暴露的条件误读与机制解释遗漏；全量 dev/test 规模待确认。

### C：领域诊断工具与演示完善

- [ ] 基于错误案例判断是否增加执行计划/日志解析工具。
- [ ] 首先对用户提供的材料或本地测试环境作只读分析，工具必须真实执行。
- [ ] 报告区分已确认事实、可能原因、待验证假设和下一步检查建议。
- [ ] 三个可复现演示及短录屏：简单回答、补充调查、信息不足。
- [ ] 公网 demo 与费用限制在本地验证之后单独安排。
- [ ] 按实测结果更新项目能力与效果说明。

## 已选定的实现边界

1. **单 Agent + 显式 Python 编排。** 暂不引入框架或多 Agent；现有 Protocol 足以表达。
   模型输出 JSON 动作，Python 校验后调度；不是供应商原生 function calling API。
2. **只读文档工具。** 搜索复用现有 `OnlineRetriever`；读取工具打开本次搜索缓存的完整段落。
   不声称实现全文章节读取、SQL 执行、日志诊断或任意网页读取。
3. **每次调查状态独立。** 第一版追问返回给用户，用户将补充信息与原问题一起重提；
   尚无多轮 session、持久记忆或 checkpoint 恢复。
4. **预算如实命名。** `max_steps` 限制模型决策次数；搜索/读取单独限额。
   token 数是现有估算器计算的输入预算，不是供应商真实 usage 或账单；输出由模型请求上限约束。
5. **时间/取消在调用边界检查。** 阻塞中的 embedding、rerank、chat 请求不能强杀；
   不能承诺严格墙钟 deadline 或取消立即停止计费。
6. **Agent chat 默认无重试，可显式开启 0–5 次。** 已提交版本固定 `max_retries=0`；当前改动增加
   `--agent-generation-retries` / `--generation-retries`，按 5/10/15/20/25 秒线性等待，只重试既有
   瞬态状态码与超时/连接重置。重试次数与传输合同均进入配置指纹。既有 `/api/ask` 的历史重试策略
   独立保留；检索依赖仍有原来的重试语义。
7. **所有出站请求直连。** 服务组合根中 chat、embedding、rerank 使用 `DirectTransport`：不读环境/系统
   代理、不跟随重定向、保留 TLS 校验；非 2xx 先按状态抛出。Milvus 回环连接加入 `no_proxy`。
8. **事件不包含内部思维或原始 provider 错误。** 返回结构化动作、状态、计数、证据 ID 和时间；
   证据正文与答案仅在响应/请求内存中，不自动落盘，不改变旧检索 trace 合同。
9. **结构校验不是事实性证明。** 尚未实现陈述级语义支持校验；须由 B 阶段评测和人工审核补齐。

## 当前工作位置

以下记录截至已提交的 `608515a` 及本会话已验证的未提交改动；最新状态以文首交接节为准。
`tests/test_ask_service.py` 已适配新增 capabilities 字段。
2026-09-21 当前工作区完整 pytest：**1498 passed**；ruff check、ruff format --check、mypy 均通过。
测试存在 Starlette/httpx 弃用提示，不影响通过结果；当前没有为此变更依赖。
页面内联 JavaScript 经 `node --check` 通过。浏览器视觉/真实交互验收未执行。

已运行 `scripts/agent_tasks.py --initialize`：本地生成 **24 条草稿，0 条已审核**；
位于 `indexes/agent_eval/v1/tasks.jsonl`，`git check-ignore` 已确认忽略。
已运行 `scripts/compare_agent.py --allow-drafts`，默认仅输出执行清单，未进行付费调用。
完整 CLI 的写入、完成状态与拒绝覆盖行为通过合成依赖验证。新增 `scripts/review_agent.py`
可独立准备审核、检查原始试次完整性并生成仅含聚合数据的质量报告；当前尚无真实审核标签。
已使用真实 `.env` 尝试一个开发任务的两次独立冒烟（默认环境及受控网络权限），
两次均在首次 chat 返回 `generation_failed`，搜索次数为零。随后两次单请求、
小输出上限的诊断确认 **HTTP 403**，包括移除进程代理环境变量的对照。
没有修改系统代理或凭据，也不能仅凭 403 断定是鉴权失败、供应商封禁还是代理/网关拦截。
已经停止重复调用；这批记录只证明失败路径，不构成 Agent 任务效果或成本结果。

本地运行目录：

- `indexes/agent_eval/runs/dev-agent-smoke-20260914/`：首次失败记录。
- `indexes/agent_eval/runs/dev-agent-smoke-20260914-network/`：受控权限下的失败记录，
  已成功运行审核器 `--prepare`，保留空审核标签与 `review-packet.txt`。
- `indexes/agent_eval/diagnostics/20260914-transport{,-direct}.json`：仅含状态码与错误类别，
  不含密钥、端点地址或响应原文；`-direct` 仅表示已移除进程代理环境变量，不保证网络链路无代理。

比较产物合同升级为 v2：manifest 绑定全量试次摘要和实际方法 profile；审核材料独立于原始
试次。v1 产物不会被静默迁移。报告要求所有任务及所有试次均审核，至少两个来源组，
并将故障保留在任务成功率分母；摘要与统计可以使用 `--check` 完整重算核对。

后续按用户要求配置了首次失败后等待 5/10/15/20/25 秒、最多五次重试的诊断。
使用 `http.client.HTTPSConnection` 直连目标 443 端口，首次请求即返回 HTTP 200，
`finish_reason=stop`、served model 匹配，响应为 `{"ok":true}`，耗时约 4.27 秒。
本次实际只调用一次，没有触发退避，不能声称验证了阶梯重试后的恢复能力。
记录位于 `indexes/agent_eval/diagnostics/20260914-102248-https-linear-retry.json`。
这证明最小直连请求当时可用，完整文档调查与带证据回答仍待重测；
服务组合根的直连接入已写入未提交改动，尚未完成回归与真实验收。
用户要求后续请求不走代理端口，执行时必须显式保证，不应仅删除环境变量后假定直连。

下一步：提交本会话已验证的改动；人工核对 `dev-agent-smoke-20260921-direct-5` 的答案与
逐项验收条件，扩充任务集。大规模真实试验需明确预算；不得把合成回归或单个草稿任务冒烟作为模型质量结果。
页面视觉验收保留为独立待办。代码地图与命令详见 [操作说明](agent.md)。

后续会话还应注意：固定/单轮方法沿用旧问答合同，没有追问动作；三方法是整体流程对比，
不能将差异归因给唯一因素。所有检索调用失败后的终止为 `retrieval_failed`，不是无答案。

## 检查与运行入口

在仓库根运行，使用已有 `.venv/Scripts/python.exe`；不设置 `PYTHONUTF8`。

```powershell
git status --short --branch
.venv/Scripts/python.exe -m pytest
.venv/Scripts/ruff.exe check src tests scripts
.venv/Scripts/ruff.exe format --check src tests scripts
.venv/Scripts/mypy.exe
```

真实运行入口（依赖已构建索引、Milvus 与 `.env` 中服务商配置，会产生调用费用）：

```powershell
uv run --extra service --extra milvus --with milvus-lite==3.2.0 python scripts/serve.py --enable-agent --generation-reasoning-effort low --generation-max-tokens 4096
```

`--enable-agent` 不要求同时启用 `/api/ask`；需要并排演示单轮问答时增加 `--enable-generation`。
Agent 不兼容限定查询的 `--query-cache` 模式。

## 会话记录

### 2026-09-13

- 用户确认迭代方向并要求新分支。沙箱限制 `.git` 写入，获准创建并切换新分支；main 未提交改动。
- 完成核心、服务/UI、任务集合同与三方法对照入口。
- 用户追加跨会话记忆要求，建立本文；持续维护本文件，不仅在最终交付时更新。
- 修复回归暴露的动作 Unicode 校验、检索配置指纹字段、不可变证据 metadata 序列化问题。
- 全套门禁通过；main 仍指向 `0668fa9`，本轮未合并或推送。
- pytest 临时目录在当前沙箱下不可访问，离线回归已获受控权限运行；没有更改编码/CI 策略。
- 浏览器工具报告缺少会话认证；按用户要求跳过，未修改工具或应用认证设置。

### 2026-09-14

- 完成审核与统计报告闭环：独立 review 文件、完整配对校验、原文哈希、方法配置一致性，
  来源组聚类 CI、双侧配对 bootstrap 和 Holm 校正；报告只含聚合信息。
- 增加 `--prepare` / `--report` / `--check` 工作流；审核材料隐藏显式方法标签，
  但不声称严格双盲或已认证人工身份。
- 对照脚本增加可配置预算；v2 运行不可通过删除失败试次或改写原文后直接生成报告。
- 真实冒烟和连通性诊断均遇到 HTTP 403，未进入检索；失败记录在本地保留。
- 完整门禁 1476 passed；真实任务人工审核与视觉验收仍待完成。
- 最小 HTTPS 直连诊断首次请求成功，未实际触发用户指定的阶梯重试；完整 Agent 冒烟仍待完成。

### 2026-09-17

- 用户准备转到新会话，已核对当前分支、提交与工作区并更新文首交接节。
- 直连、可选重试、故障状态与整组置换检验修复仍未提交；静态门禁已通过，新增测试待执行。
- 本次仅整理交接，不启动真实模型调用、不将旧完整测试数量当作当前代码的验证结果。

### 2026-09-21

- 执行上次交接的定向测试：1 处失败为质量报告统计标签含 `group-` 前缀，触发隐私断言；改标签修复。
- 补齐交接清单中缺失的回归：传输合同指纹、`--agent-generation-retries` 范围与透传、
  `_direct_loopback`、HTTP 层 `answerable:false` 检索故障映射；`_build_agent` 改为读密钥前校验预算。
- 真实冒烟 5 次（同一草稿任务）。前 4 次失败依次归因于预算不足、超时和直连实现把超大 504
  错误页判为响应超限；诊断确认供应商偶发约 61 秒后返回 850 KB HTML 的 504。
- 修复直连传输：非 2xx 先按状态抛出、错误正文截断 16 KB，合同升 v2。第 5 次冒烟 `answered`，
  两次 504 各在等待 5 秒后重试成功；全程导出死代理端口证明不经代理。
- 全套门禁 1498 passed，README 测试数由同步器重出；已为该运行准备审核材料，未填写标签。
- 用户要求后复跑全部门禁并在本功能分支提交一次；没有修改系统代理或凭据，未合并或推送。
- 用户要求由模型完成该运行的审核：按 SHA-256 定位本地源文档核对两条证据，拆出 12 条陈述逐条比对，
  全部受支持，`task_success=true` 写入 `reviews.jsonl`，reviewer 明确标注为模型审核者。
  草稿任务 `reviewed=false` 且验收条件为空，`--report` 按合同拒绝；这不是人工审核结果。

### 2026-09-21（第二会话）

- 按交接顺序继续：v1 的 24 条模板草稿与语料无关，无法支撑验收条件；改为在 `indexes/agent_eval/v2/`
  建语料落地的任务集，v1 保持不动以保留既有冒烟的任务集指纹。
- 10 个起草 agent 并行起草 79 条，附逐字引文与起草说明；新增离线装配器核验引文逐字存在、
  来源在索引内、来源组不跨 split，任何失配即拒绝写出。
- 首轮 13 处失配：12 处为 markdown/HTML/Hugo 装饰造成的规范化缺口，扩展规范化器并加回归；
  1 处为引文省略句子，按文档补全并标注。
- 全套门禁 1510 passed；README 测试数由同步器重出。没有调用付费接口，没有填写任何人工审核标签。
- 发现本机 C 盘接近满盘导致 pytest 假失败与 shell 输出截断；未做清理，记入交接。
