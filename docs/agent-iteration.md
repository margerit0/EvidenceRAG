# 文档调查 Agent 迭代计划与会话交接

更新日期：2026-09-28。工作分支：`feat/document-investigation-agent`。

本文件是本轮迭代的进度与决策入口。跨会话先读 `CLAUDE.md`、本文件，再运行
`git status --short --branch` 和 `git log -5 --oneline`；代码、测试和真实运行记录优先于文档描述。
不要在 main 上开发；尚未验证的能力不能写成已完成的项目成果。

## 新会话从这里继续（2026-09-28）

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
