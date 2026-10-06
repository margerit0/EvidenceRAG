# 下一会话开发交接

更新：2026-10-07。工作区 `D:\rag`，本阶段交付基线为 `main`。
`feat/frontend-design` 保留为已完成的阶段分支；下一轮从最新 `main` 新建 `codex/` 分支。

新会话先读取 `CLAUDE.md` 和本文，再核对 `git status --short --branch`、
`git log -3 --oneline`。以实际代码和最新验证记录为准，保留新出现的未提交修改。
本次仅完成提交、下一步规划与合并；下文计划尚未实施，不自动启动新开发或付费调用。

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

新会话若要先同步各分支，先读 [分支同步盘点](branch-sync-handoff.md)。旧工作树中
有三个文件尚需判断是否包含主线缺失改动；其余已核对一致的内容不重复迁移。
先完成该盘点中的差异审查，再按以下顺序继续开发。

1. **先完成可复现演示交付。** 建议分支 `codex/workbench-demo-guide`。复用现有
   `answered`、`clarification_needed`、`insufficient_evidence` 合成场景及已通过的
   浏览器回归，整理“完整调查、需要补充信息、证据不足”的演示步骤和短录屏。
   补充生产构建的本地预览说明，验证仅依赖公开仓库文件即可演示，不依赖
   `.research_tmp`、密钥或本地索引。验收：三条演示路径与实际状态一致，素材明确标注
   合成数据，README 可直接找到入口；沿用已确认的界面与模型入口，不重新设计。
2. **再正式化离线审核与报告。** 先盘点并复用 `scripts/review_agent.py`、
   `src/zhrag/eval/agent_review.py` 和既有本地报告器，只补通用工具的缺口。
   保留多来源统计合同；单来源结果明确使用描述性报告。验收：旧结果可离线重算，
   缺档或指纹不符时失败，旧文件与旧评分不变；不重做已完成的 v2 任务审核。
3. **随后建立离线失败回放。** 复用保存的最终响应和原创合成 SSE，分别回放解析错误、
   协议错误和预算终态。验收：无网络仍能复现已记录失败；摘要漂移可检测，缺少正文或
   请求状态时标为不可回放；输出使用新目录，不补写历史响应。

后两项延续 [Agent 离线工程计划](agent-iteration.md)，进入各项前先核对已有实现。
完整调查与独立质量验收仍需区分；新的真实模型评测、公网部署和费用限制另行安排。
本轮没有新增付费调查、提示词修改、全量 dev/test 运行或公网发布。

## 本机运行

- 前端：`http://127.0.0.1:5173/workbench/`，Vite 代理到 `127.0.0.1:8000`。
- 前端恢复：在 `D:\rag\frontend` 执行 `npm run dev`。
- 后端本地配置：`.research_tmp/frontend-model-selector-20261006/service.env`，默认
  `deepseek-v4.1-flash`，附加 `z-ai/glm-5.3`，两者共用已配置的地址和密钥。
  原 `.env` 未改写；不要只按它启动后就认为两个模型均已启用。
- 后端恢复命令见 `docs/frontend.md` 的“选择调查模型”；也可从仓库根运行：

```powershell
.venv/Scripts/python.exe .research_tmp/frontend-model-selector-20261006/serve_live.py
```

恢复前先检查端口与对应进程，不重复启动或停止其他任务的服务。上述本地启动器、配置、
索引和运行产物均被 Git 忽略，新机器不会随代码获得。不要打印或复制凭证到文档、日志、
提交或新会话提示词。

当前模型请求使用非流式、低推理强度、4096 输出上限、90 秒单次超时、600 秒调查预算，
最多 1 次瞬态重试；前端执行进度仍为 SSE。DeepSeek 流式曾出现缺少正常 `stop` 的响应，
不能为了跑通而取消完整性校验；放宽大小写并不等于放宽结束条件。

## 验证与产物

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
- 本次提交门禁：Python 1,767 项、前端单测 17 项、浏览器回归 31 项通过；
  TypeScript/Vite 构建、Prettier、ruff check / format、mypy 与 diff 空白检查通过。
  命令、首次浏览器启动失败及复验说明见
  [2026-10-07 收尾记录](frontend-motion-handoff.md#品牌readme-与阶段收尾2026-10-07)。

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
