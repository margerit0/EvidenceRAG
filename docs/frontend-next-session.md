# 下一会话开发交接

更新：2026-10-06。工作区 `D:\rag`，当前分支 `feat/frontend-design`。

新会话先读取 `CLAUDE.md` 和本文，再核对 `git status --short --branch`、
`git log -3 --oneline`。以实际代码和最新验证记录为准，保留新出现的未提交修改。
下一步目标尚未指定，由用户在新会话补充。

## 当前已完成

- 前端为 A「冰川流线」设计，采用用户选定的冰白玻璃配色，浅深主题均支持。
  保留流程图、检查器、引用阅读、调用导航和下拉菜单的原有动效。
- “连接服务”模式下，可在输入框右下角、发送按钮左侧选择 DeepSeek / GLM。
  入口是紧凑文字下拉，菜单向上展开；顶部仅显示运行模式和连接状态。
- 模型选择作用于下一次调查，运行中和答案入场动画期间禁用；选择偏好会保存。
  答案、运行历史和检查器记录该次模型，切换选择不改写旧记录。
- 后端通过 `--agent-model` 配置附加模型。两个调查接口接受可选 `model`，只允许
  服务器提供的选项；各模型共用检索器和并发预算，每次请求固定模型，不自动回退。
  `/api/ask` 仍使用默认模型。地址和凭证只留在服务端。
- 模型身份比较仅忽略 ASCII 大小写；命名空间、版本、空格及不同模型仍需严格区分。
  身份合同和流式合同均已更新并进入配置指纹。

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
- 最终提交门禁记录见 `docs/frontend-motion-handoff.md` 的 2026-10-06 提交门禁段。

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
