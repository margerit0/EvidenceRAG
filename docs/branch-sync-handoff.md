# 分支同步与新会话交接

核对日期：2026-10-07。工作区 `D:\rag`，本轮核对基线为本地 `main` / `9278053`。
已从该基线创建 `codex/branch-sync-review`，完成三个旧文件的语义审查，**没有需要迁移的缺口**。
没有合并旧历史，没有删除或改写旧工作树中的文件。
开始新会话时先重新检查 Git 状态，不能把这份快照当成实时状态。

## 当前分支状态

| 分支                                                         | 核对结果                                                | 后续处理                                     |
| ------------------------------------------------------------ | ------------------------------------------------------- | -------------------------------------------- |
| `main`、`codex/session-context-handoff`                      | 开始核对时同为 `9278053`，主工作区干净                  | 本轮审查从此基线开始                         |
| `feat/frontend-design`                                      | `d910a74`，已是 main 的祖先                            | 前端阶段已交付，不重复合并                   |
| `codex/branch-sync-review`                                  | 本轮从 `9278053` 新建                                  | 保存审查结论与后续演示交付                   |
| `docs/readme-portfolio`、`feat/document-investigation-agent` | 均为 `5365993`，已是 main 的祖先                        | 已纳入主线，无需把历史分支指针全部移到 main  |
| `feat/tidb-eval-qrels`                                       | 旧公开历史整理分支，存在与 main 同主题但不同 SHA 的提交 | 按内容核对，不能把独有提交数直接当成缺失功能 |
| `worktree-agent-*`                                           | 旧阶段分支；部分仍挂载工作树                            | 先核对未提交文件与主线，保留原目录和修改     |

本地 `main` 开始核对时领先本地远端跟踪引用 `origin/main` 25 个提交；本次没有 fetch 或 push，
不能据此判断服务器上的实时状态。本文后续提交也会使本地计数变化，以 Git 为准。

旧 `feat/tidb-eval-qrels` 的 `8f3b472` 与 main 历史中的 `23cfa70` 标题相同。
直接比较两棵树，仅 `.gitignore`、`AGENTS.md`、`CLAUDE.md` 与架构文档有差异；
其余大量同主题提交的 SHA 差异不等于新增实现。保留这条证据，避免重复合并历史。

## 挂载工作树的未提交文件

以下路径均相对于 `D:\rag\.claude\worktrees\`。使用 Git blob 内容比对，
只读检查当前 main 与 main 历史；“未找到相同内容”不等于需要迁移，仍需语义审查。

| 工作树                                               | 文件与结果                                                                                                             | 下一步                                                                                 |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `agent-aaf19e07a9f7383db`                            | `src/zhrag/providers/chat.py` 的内容已在 main 历史中；`tests/test_chat_provider.py` 与当前 main 一致                   | 不用旧版本覆盖当前实现                                                                 |
| 同上                                                 | 未跟踪的 `src/zhrag/providers/answering.py`、`tests/test_answering_provider.py` 已完成语义审查 | 有效能力已覆盖；旧指纹与重试合同已被后续实现替代，不迁移                             |
| `agent-aca6621a650a2cf9d`                            | 修改过的 `src/zhrag/service/static/index.html` 已完成语义审查 | 问答与检索功能已覆盖；旧视觉已被后续静态页及 React 工作台替代，不迁移 |
| `agent-ae7de2958c5a56acf`                            | 五个文档同步脚本及对应五个测试均与当前 main 一致                                                                       | 已同步，保留旧工作树即可，不重复迁移                                                   |
| `agent-a391dd1cb27df5a3d`、`agent-a46bfd4efdba0d284` | 工作树干净                                                                                                             | 本次无需处理                                                                           |

已确认一致的五组脚本/测试为 `sync_h_hybrid_mrl1024_docs`、`sync_m9a_docs`、
`sync_m9b_docs`、`sync_m9b_results_docs`、`sync_tidb_chunk_sweep_docs`。

## 三个旧文件的审查结论

比较对象是旧工作树磁盘上的实际内容（包括未跟踪文件），不是仅比较分支 HEAD。
以下主线路径均以 `9278053` 为准；`git diff --no-index -- <旧文件> <主线文件>` 返回 1
仅表示有差异。没有以 blob 不同或独有提交数推断缺失功能。

| 旧文件 / 关注点 | 主线证据与判定 |
| --- | --- |
| `providers/answering.py`：响应大小、超时、错误脱敏、JSON 输出、输出 token 限制 | 同路径保留 `_bounded_urlopen`、`_is_timeout`、`ChatAnswerGenerator.generate`；仍限制读取 `MAX_RESPONSE_BYTES + 1`，以固定 `GenerationError` 返回失败，不暴露 provider 正文。**主线已覆盖**。 |
| 同文件：一次尝试与批处理隔离 | `ChatClient(retries=1)` 仍保留；在线 `_retrying_transport` 单独控制瞬态重试，可用 `max_retries=0` 禁用。直接传输、预算中断与流式合同是后续增量。**旧默认不重试合同已被替代**，不能覆盖新版。 |
| 同文件：模型及配置身份 | 主线使用 `model_names_match`（仅忽略 ASCII 大小写），指纹绑定公开运行参数、重试/传输/模型身份合同且排除密钥。**旧 exact 模型比较与只绑定 endpoint/model 的指纹已被替代**。 |
| `tests/test_answering_provider.py`：旧用例逐项比对 | 成功请求次数、2048 token 请求、默认无 reasoning 字段、模型漂移脱敏、直接/包装超时、无效参数、超大响应与 timeout 传递均保留；单次超时测试显式设 `max_retries=0`。**主线已覆盖**。 |
| 同测试：`test_profile_does_not_bind_secret_or_runtime_knobs` | 主线 `test_profile_binds_generation_parameters_but_not_the_key` 验证参数变化导致指纹变化、密钥变化不影响指纹；另有重试、传输及模型合同回归。旧参数无关断言会破坏现行可复现性要求，**不迁移**。 |
| `service/static/index.html`：问答/检索切换、capabilities、证据和引用、错误响应 | 同路径保留 `normaliseCapabilities`、`showAskData`、`showSources`、`addCitationButton`、`isStructuredAnswer`、中止控制与过期响应过滤；安全文本用 `textContent`、外链限制 http(s) 且使用 `noopener noreferrer`。**主线已覆盖**。 |
| 同静态页：耗时、文案、布局 | 主线进一步区分 `Retrieval total` / `Ask total`，使用“本次选入的证据”，并补齐来源链接、能力加载禁用、文档调查入口。旧 panel 样式不恢复；React `/workbench/` 沿用冰川流线、RAG 品牌和现有模型入口。**后续实现替代**。 |

历史依据：`be8dfaa` 已交付单轮 grounded answering；后续 `33b4a99`、`2275144`、
`4533dd0`、`596f20b`、`8feb052`、`48483f9` 演进传输、重试、流式及模型身份合同。
静态页的调查与能力控制另见 `4b3113e`、`3e619cb`。本轮三个文件的“仍有价值的缺口”
与“待确认”均为空；结论只覆盖本次记录的实际内容，未来新增修改仍须重新审查。

旧文件 SHA-256（原始磁盘字节，方便后续识别新增修改）：

| 文件（工作树同上） | SHA-256 |
| --- | --- |
| `providers/answering.py` | `3AFD4AC176AE503ADA3C7FB028DA02EBB622C50EF224EB2FE75973735574244E` |
| `tests/test_answering_provider.py` | `C44D5A2ECEE10F3D2355812252248DF8BF14CE59A861C2E83F98F042E35F317C` |
| `service/static/index.html` | `BEDBB956C9699C815C85921585E243145C20A885FB0EBFAC54C172A9F43A8426` |

另重新按 Git blob 核对五组同步脚本/测试与 main 一致；旧 chat 测试与 main 一致，
旧 chat 实现仍可在 main 历史找到。五个旧工作树保留原 HEAD 与未提交状态。
本阶段没有产品代码变化，复用 [既有完整门禁](frontend-next-session.md#验证与产物)；
对本轮文档执行相对链接存在性、Git 差异和空白检查。

## 后续执行顺序

1. 读取 `CLAUDE.md`、本文和 [前端与下一步交接](frontend-next-session.md)，运行
   `git status --short --branch`、`git log -5 --oneline`、`git branch -vv` 与
   `git worktree list --porcelain`。再次核对三个有修改的旧工作树，保留新出现的修改。
2. `codex/branch-sync-review` 已创建且三个文件已判定，无需重复迁移或重建同名分支。
   若旧工作树出现新内容，仅审查新增差异；不按分支数量盲目 merge，不整文件覆盖新版实现，
   不清理旧工作树。
3. 提交前执行 `CLAUDE.md` 的门禁，核对暂存文件；不提交本地语料、
   真实回答、凭证、索引或旧工作树目录。没有产品代码变化时复用已有代码验证记录，
   对文档与 Git 配置做相应检查。
4. 差异核对已完成，继续精简交接中的第 1 项：可复现演示、生产构建预览与短录屏。
   冰川流线、RAG 品牌、输入框右下角且在发送按钮左侧的 DeepSeek / GLM 选择均保留。
   离线审核报告与失败回放仍为后续计划，不重做已完成的 v2 任务审核。

当前没有新的付费调用或远端发布任务。前后端启动方式、验证结果和已知限制继续以
[精简交接](frontend-next-session.md)为准；新会话不要复制密钥或粘贴整段历史聊天。
