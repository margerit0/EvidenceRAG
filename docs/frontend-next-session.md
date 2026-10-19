# 下一会话开发交接

更新：2026-10-20。工作区 `D:\rag`，本轮修复来源分支为 `codex/fix-agent-graph-edges`；
实际分支与远端同步状态以 Git 为准。
`feat/frontend-design` 保留为已完成的阶段分支；此前文档与分支审查使用
`codex/branch-sync-review`。后续先核对实际分支，不重复创建。

新会话先读取 `CLAUDE.md` 和本文，再核对 `git status --short --branch`、
`git log -3 --oneline`。以实际代码和最新验证记录为准，保留新出现的未提交修改。
本轮分支审查已完成（`83e55b9`），演示步骤与生产预览说明见 [演示指南](workbench-demo.md)。
用户已明确短录屏由其自行完成，不启动录制任务，不安装录制组件。
本次收尾包含图连线修复、恢复记录与后续计划；用户已要求提交后合并 `main` 并推送
`origin`。具体提交与同步状态以 Git 为准；未扩展正式模型评测。

## 本轮收尾验证（2026-10-20）

保留 React Flow 受控节点的尺寸测量状态，使更新执行进度、连续查看决策及开合
检查器时五条连接线持续存在。代码审查与截图复核完成，完整离线门禁通过：
Python 1,789 项、前端单测 22 项、生产构建上的浏览器回归 42 项；ruff check /
format、mypy、Prettier 与 diff 空白检查均通过。两种动效的连续采样均未丢线，
既有检查器动画时序回归本轮也通过。

完整浏览器测试由独立 4173 预览服务自动执行 TypeScript/Vite 生产构建，使用
原创合成演示与 API 拦截，不调用模型。Vite 单测首次被沙箱阻止写临时配置，获准
在沙箱外运行后 22 项通过。pytest 保留一条既有 Starlette/httpx 弃用警告；构建有
依赖中的 PURE 注释处理提示，均未导致失败。JUnit、浏览器报告及截图位于
`.research_tmp/graph-closeout-20261020/`，不提交。后续开发从离线审核与报告开始。

## 最新续接：模型调用恢复（2026-10-20）

已读取原聊天 `01a11c59-9d95-7871-bd29-7b604ebb9d6e`。用户最后明确继续使用
`ai.hybgzs.com`，请求不走代理。按当前本机配置复查，直连 TLS 校验和认证后的
`/v1/models` 均成功，DeepSeek / GLM 都在模型列表中；10 月 9 日记录的证书故障
本次未复现，不能据此反推当时的根因或恢复时间。网站首页直连返回 403，API 返回
200，首页可访问性和模型 API 可用性需分别判断。

本轮恢复前 8000 / 5173 均未监听。已恢复前后端，保持原 high / 4096 / 90 秒 /
600 秒 / 5 次 Agent 重试及两个模型配置。供应商请求继续使用保留 TLS 校验的
`DirectTransport`。首次启动还发现本机 Milvus 的 SOCKS 代理环境警告，现通过
本地启动器清除服务子进程继承的代理变量并设置回环 bypass；未修改系统代理或凭证。
当前本机恢复入口：

```powershell
.venv/Scripts/python.exe .research_tmp/generation-failure-20261020/serve_direct.py
```

启动器及日志位于 Git 忽略目录，脚本只适用于这台已有索引和配置的工作区。
前端仍为 `http://127.0.0.1:5173/workbench/`。后端健康检查、能力接口及前端代理均为
200，清理后的后端 stderr 为空。重启前仍需检查端口与进程，避免重复启动。

验证：两个模型各完成一次最小 JSON 生成，均首次成功、模型身份匹配且
`finish_reason=stop`（DeepSeek 10.4 秒，GLM 9.0 秒）。随后经前端代理执行一条
DeepSeek 完整调查，26.0 秒返回 `answered`，4 次模型调用、1 次搜索、2 次读取；
17 个 SSE 进度事件序号连续，两个返回来源均与本地快照重建片段精确一致，全部引用
属于已读取证据。该次验证没有独立语义质量审核，不代表全量质量或长期稳定性结论。
GLM 本轮只验证最小生成，没有追加完整调查。未运行旧 v2 审核或冻结 test。

图连线代码保留旧会话修复；在当前开发服务上重新执行普通/减少动效两项浏览器回归，
均通过。首次测试因沙箱拒绝创建输出目录而未运行用例，获准在沙箱外重跑后通过。
上述恢复阶段未改产品源码或前端测试；提交前完整门禁见上文。诊断、调查响应、引用核验及浏览器
截图均在 `.research_tmp/generation-failure-20261020/`，不得提交私有响应或派生产物。

## 历史修复记录

2026-10-07 文档一致性修正已接续实施：修复 README 双图测试和开发依赖命令，
补齐消融摘要的离线生成入口，校正统计解释、查询前缀、生成参数和历史状态。
修正范围与本轮验证见[文档一致性记录](documentation-consistency.md)。
README 原有暂存修改作为基线保留，新增修正留在工作区；Git 状态以实际检查为准。

2026-10-08 决策详情修复：检查器在每次“Agent 决策”标题下显示已解析的动作与参数，
覆盖搜索词、读取的证据编号、追问、提交答案和停止作答。SSE 与最终响应均保留详情；
等待、失败和旧服务缺字段有明确说明。模型输入、提示词与配置指纹保持原合同。
本轮提交一并包含输入框清空逻辑及对应 E2E 修正，下面的完整验证已覆盖这些改动。
后端已按当前进程原有参数重启，两个模型均可用；未发起真实模型调用。

2026-10-09 Agent graph 连线修复：受控节点在更新执行进度或查看决策时未保留
React Flow 的 `measured`，导致连线锚点被清空。组件现在通过 `onNodesChange`
保存尺寸测量，并在后续节点更新中保留；查看历史调用仍只改变检查标记。
浏览器回归持续采样五条连线，覆盖运行中反复切换决策、检查器开合及完成后的查看；
既有几何断言补上节点与连线数量，避免空集合被误判为通过。

2026-10-09 后续 `generation_failed` 排查：当时服务的模型地址在直连 TLS 握手时返回
不受信任的自签名证书，Python 校验码 18，Windows curl 返回 60 /
`SEC_E_UNTRUSTED_ROOT`。证书处于有效期内，本机 hosts 没有对应域名覆盖。
本地 `/healthz` 与能力接口正常只说明本地服务可用，不能证明供应商可调用。
当时记录为 TLS 阻断，尚未恢复真实模型调用；2026-10-20 的恢复验证见上文。
服务仍读取 `.research_tmp/frontend-model-selector-20261006/service.env`，启动参数
为 high / 4096 / 90 秒单次超时 / 600 秒调查预算 / 5 次重试 / 非供应商流式。
本轮仅查询模型列表及核对 TLS，未发送生成请求。脱敏记录位于
`.research_tmp/generation-failure-20261009/tls-diagnostic.json`。

## 当前已完成

- 前端为 A「冰川流线」设计，采用用户选定的冰白玻璃配色，浅深主题均支持。
  保留流程图、检查器、引用阅读、调用导航和下拉菜单的原有动效。
- 页面品牌、浏览器标题及无障碍标签已从 `zhRAG` 改为 `RAG`；内部包名和本地偏好
  存储键保留，已有主题与模型选择记忆继续有效。
- README 增加工作台介绍、无需密钥的演示入口及用户提供的两张截图：连接服务首页、
  原创合成演示的调查详情。品牌与 README 改动已提交为 `7bd01d3`；冰川流线与模型
  选择功能提交为 `48483f9`，无需重复实现。
- “连接服务”模式下，可在输入框右下角、发送按钮左侧选择 DeepSeek / GLM。
  入口是紧凑文字下拉，菜单向上展开；顶部仅显示运行模式和连接状态。
- 模型选择作用于下一次调查，运行中和答案入场动画期间禁用；选择偏好会保存。
  答案、运行历史和检查器记录该次模型，切换选择不改写旧记录。
- 后端通过 `--agent-model` 配置附加模型。两个调查接口接受可选 `model`，只允许
  服务器提供的选项；各模型共用检索器和并发预算，每次请求固定模型，不自动回退。
  `/api/ask` 仍使用默认模型。地址和凭证只留在服务端。
- 模型身份比较仅忽略 ASCII 大小写；命名空间、版本、空格及不同模型仍需严格区分。
  身份合同和流式合同均已更新并进入配置指纹。

## 下一步计划

后续工作已细化为 [Agent 后续任务计划](agent-next-plan.md)：当前修复已完成提交前门禁，
后续正式化离线审核/报告、失败回放和运行诊断，最后准备定向质量验证。后续条目为计划，
不表示相关实现或付费步骤已经启动。

分支同步已完成，依据见 [分支同步审查](branch-sync-handoff.md#三个旧文件的审查结论)。
三个旧文件的有效能力均已覆盖或被后续合同替代，没有迁移产品代码；旧工作树及未提交
修改保留。下次只需检查是否新增差异，再按以下顺序继续开发。

1. **可复现演示说明已补齐，录制交由用户。** [演示指南](workbench-demo.md)复用现有
   `answered`、`clarification_needed`、`insufficient_evidence`，包含生产构建预览、
   三条路径的步骤和预期终态，README 可直接找到入口。演示仅依赖前端公开源码与 npm
   依赖，不依赖 `.research_tmp`、密钥或本地索引。沿用已确认的界面与模型入口。
   本轮未执行录制；中途准备但未运行的录制脚本、配置与 CI 改动均撤回。
2. **再正式化离线审核与报告。** 先盘点并复用 `scripts/review_agent.py`、
   `src/zhrag/eval/agent_review.py` 和既有本地报告器，只补通用工具的缺口。
   保留多来源统计合同；单来源结果明确使用描述性报告。验收：旧结果可离线重算，
   缺档或指纹不符时失败，旧文件与旧评分不变；不重做已完成的 v2 任务审核。
3. **随后建立离线失败回放。** 复用保存的最终响应和原创合成 SSE，分别回放解析错误、
   协议错误和预算终态。验收：无网络仍能复现已记录失败；摘要漂移可检测，缺少正文或
   请求状态时标为不可回放；输出使用新目录，不补写历史响应。

后两项延续 [Agent 离线工程计划](agent-iteration.md)，进入各项前先核对已有实现。
完整调查与独立质量验收仍需区分；新的真实模型评测、公网部署和费用限制另行安排。
后续离线工程计划不自动追加付费调查、提示词修改、全量 dev/test 运行或公网发布。

## 本机运行

- 前端：`http://127.0.0.1:5173/workbench/`，Vite 代理到 `127.0.0.1:8000`。
- 前端恢复：在 `D:\rag\frontend` 执行 `npm run dev`。
- 后端本地配置：`.research_tmp/frontend-model-selector-20261006/service.env`，默认
  `deepseek-v4.1-flash`，附加 `z-ai/glm-5.3`，两者共用已配置的地址和密钥。
  原 `.env` 未改写；不要只按它启动后就认为两个模型均已启用。
- 后端恢复命令见 `docs/frontend.md` 的“选择调查模型”；也可从仓库根运行：

```powershell
.venv/Scripts/python.exe scripts/serve.py --env .research_tmp/frontend-model-selector-20261006/service.env --enable-generation --enable-agent --agent-model z-ai/glm-5.3 --generation-reasoning-effort high --generation-max-tokens 4096 --generation-timeout 90 --agent-max-seconds 600 --agent-generation-retries 5
```

恢复前先检查端口与对应进程，不重复启动或停止其他任务的服务。上述本地启动器、配置、
索引和运行产物均被 Git 忽略，新机器不会随代码获得。不要打印或复制凭证到文档、日志、
提交或新会话提示词。

2026-10-08 重启保留本轮开始时的实际参数：非流式、high 推理强度、4096 输出上限、
90 秒单次超时、600 秒调查预算，最多 5 次瞬态重试；前端执行进度仍为 SSE。
旧 `serve_live.py` 启动器使用 low / 1 次重试，不能用它代替上面的当前配置。
DeepSeek 流式曾出现缺少正常 `stop` 的响应，
不能为了跑通而取消完整性校验；放宽大小写并不等于放宽结束条件。

## 验证与产物

2026-10-09 连线修复：改动前新增的两种动效回归均捕获五条连线同时消失；修复后
两项连续采样均通过。前端单测 22 项、TypeScript/Vite 生产构建与 Prettier 通过。
浏览器首轮 41/42 项通过，1440px 普通动效用例的动画速度时序断言失败；该次采样中
四个节点、五条连线及画布边界均正常，保留原时序断言单独复验通过。
诊断样本及专项连线截图位于 `.research_tmp/agent-graph-edges-20261009/`。
本轮只修改前端与交接文档，没有执行 Python 全套或真实模型调用；下列 Python
门禁仍是对应日期的历史记录。

2026-10-08 决策详情：Python 全套 1,789 项、前端单测 22 项、生产构建上的浏览器回归
40 项通过；ruff check / format、mypy、Prettier 与 diff 空白检查通过。浏览器验证覆盖
实时与批量响应、多轮动作切换、追问、停止作答、失败、旧服务兼容以及长文本安全显示，
已查看浅色桌面与深色手机截图。JUnit 与本轮后端日志位于
`.research_tmp/agent-decision-details-20261008/`；截图位于 `frontend/test-results/`。
本地 `/healthz`、能力接口和 `5173/workbench/` 已恢复，未调用付费 API。

2026-10-07 分支核对后续：生产构建（TypeScript / Vite）通过，独立生产预览上的
三条既有浏览器回归通过（完整调查及引用/同类调用导航、需要补充信息、证据不足）。
测试拦截 API，使用仓库内原创合成脚本；没有录制视频或新增产品代码。
新增 [演示指南](workbench-demo.md) 与 README 入口，相对链接、Git 差异及空白检查通过。
旧工作树 HEAD、未提交清单及所有已记录修改文件的 SHA-256 与开始时一致。
本轮为文档变更，按同步交接约定复用下列完整代码门禁，不重复运行 Python 全套。

GLM 和 DeepSeek 此前各完成一次真实 RAG 调查，两份引用均通过原文重建校验。
这只是单次链路验证，不是模型质量或速度比较，也不代表全量评测完成。
模型选择功能的自动化验证使用原创合成响应，没有新增付费调查。

- GLM 真实结果：`.research_tmp/frontend-live-20261006/`。
- DeepSeek 真实结果：`.research_tmp/frontend-deepseek-20261006/`。
- 当前本机配置与截图：`.research_tmp/frontend-model-selector-20261006/`，
  `model-selector-composer-light.jpg` 为最终模型入口位置。
- README 当前图片为 `docs/images/workbench-connected.png` 与
  `docs/images/workbench-investigation-detail.png`；旧 `workbench-connected.jpg`
  作为此前素材保留，不再被 README 引用。公开图只包含空态或原创合成演示。
- 2026-10-07 本地前后端已恢复，首屏与 `/healthz` 均为 HTTP 200，代理的能力接口
  返回 DeepSeek / GLM 两种模型；服务是否仍在运行须在新会话重新检查端口。
- 前端收尾阶段的历史门禁：Python 1,767 项、前端单测 17 项、浏览器回归 31 项通过；
  TypeScript/Vite 构建、Prettier、ruff check / format、mypy 与 diff 空白检查通过。
  命令、首次浏览器启动失败及复验说明见
  [2026-10-07 收尾记录](frontend-motion-handoff.md#品牌readme-与阶段收尾2026-10-07)。
- 本轮文档与同步器门禁：Python 1,779 项通过；ruff check / format、mypy 通过。
  新 JUnit 位于 `.research_tmp/docs-consistency-20261007/pytest.xml`。
  本轮未改前端实现，前端验证仍引用上面的历史记录，不视为本轮重新执行。

## 实现入口

- `frontend/src/App.tsx`、`components/ui.tsx`：模型入口、运行状态、历史和交互。
- `frontend/src/glacier.css`、`styles.css`、`assets/glacier.svg`：主题、布局、原创背景。
- `frontend/src/lib/api.ts`、`model.ts`：请求模型参数、响应校验和类型。
- `scripts/serve.py`、`src/zhrag/service/app.py`：模型配置、能力接口、请求路由。
- `src/zhrag/providers/model_identity.py`、`answering.py`、`streaming.py`：模型身份与传输。
- 新增回归：`tests/test_agent_model_selection.py`、`frontend/e2e/model-selection.spec.ts`。

需要完整过程时再读 `docs/frontend.md`、`docs/frontend-motion-handoff.md`、`docs/agent.md`。
前端任务不自动扩展到旧 v2 评审或全量付费评测。修改后按范围验证，提交前执行
`CLAUDE.md` 要求的完整门禁；真实语料、向量、引文与凭证均不入库。
