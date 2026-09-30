# React 调查工作台

实现位于 `frontend/`，工作分支为 `feat/frontend-design`。用于作品演示与本地调试，
以问题、执行图、逐次调用和证据详情为主。默认运行原创合成演示，未连接任何模型。

## 本地启动

需要 Node.js 22.12+ 和 npm。项目 `.npmrc` 固定兼容 peer 解析模式，避免 npm 10
解析可选 peer 时的 `edgesOut` 异常；依赖版本与完整性由 `package-lock.json` 锁定。
没有在项目中配置代理或镜像地址。

```powershell
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173/workbench/`。Vite 将 `/api` 转发到
`http://127.0.0.1:8000`。先选“模拟演示”即可使用，不需要凭证、语料、Milvus 或后端。

生产构建：

```powershell
npm run build
```

从源码仓库启动现有 FastAPI 服务后，自动在 `/workbench/` 挂载 `frontend/dist`。
原 `/` 静态页面保留。构建目录未提交；部署应先构建，再启动后端。
嵌入式/安装包使用方式可向 `create_app(frontend_dir=...)` 显式传入构建目录。
前端配置不包含模型密钥，后端原有启用和预算边界继续生效。

## 页面与状态

- 左侧问题及答案，中间执行图与时间线，右侧步骤、证据和运行详情。
- 固定图表示可能的动作路径；时间线用独立调用编号记录重复搜索和多次决策。
  图节点展示该动作类型的最近一次调用，时间线可以查看更早的具体调用。
- 时间线与步骤标题按动作类型显示“第 N 次”，与模型轮次和全局调用序号分别标注。
  查看较早调用时，图节点以虚线及“正在查看第 N 次”同步标记选中调用；“最近”计数、
  节点执行状态和连线仍保留实际进度。检查器提供跳回最新调用的入口。
- 状态区分进行中、回答、追问、证据不足、预算耗尽、超时和停止。
  格式/引用校验通过不等于结论独立验收通过。
- 原创合成脚本不分析输入问题，模拟数据、事件和耗时始终明确标识。
- 浅深主题仅将主题偏好保存到本地；最近十次运行只存在页面内存，刷新即清空。
  不保存真实问答、语料或证据到 localStorage，也不上传外部观测平台。
- 仅已读取证据进入引用详情；搜索候选编号不冒充完整证据。
  完整段落与最终答案一起返回，当前不提供逐字答案流。
- 窄屏按问题、流程、详情顺序纵向排列；动效响应 `prefers-reduced-motion`。
- 侧栏下移的视口（1180px 及以下）点击引用直接打开证据阅读层，手机端采用底部抽屉；
  桌面证据提供“展开阅读”。关闭后恢复触发元素焦点，不改变原阅读位置。
- 画布顶部固定展示当前动作与该动作调用次数；执行节点使用品牌底色，检查选中节点使用虚线外框。
  异常检查器突出中文终止原因，并提供已读取证据入口；“调用已返回”不代表调查成功。

## 组件与视觉

React + TypeScript + Vite；React Flow 展示流程，Lucide 提供图标，Motion 处理局部
状态切换。`components/ui.tsx` 采用 shadcn/ui 的 Button/CVA、Radix Tabs、Select、
Tooltip 和 Dialog 组合方式，并适配本项目语义色与尺寸；不依赖整套后台模板。
使用本地打包的 Inter Latin 字体及系统中文字体，不在运行时请求字体 CDN。

冷灰连续分区、青绿色品牌色；成功、需关注和错误使用独立语义状态色。
避免卡片嵌套、无关 KPI 和大面积装饰动效。流程图及证据关系本身构成视觉主体。
组件库许可证随安装依赖提供；上游入口：
[shadcn/ui](https://ui.shadcn.com/)、[Radix](https://www.radix-ui.com/)、
[React Flow](https://reactflow.dev/)、[Motion](https://motion.dev/)。

## 实时接口

`GET /api/capabilities` 新增 `agent_streaming`。Agent 启用时，客户端可请求：

```http
POST /api/investigate/stream
Content-Type: application/json
Accept: text/event-stream

{"query":"需要调查的问题"}
```

一次 POST 只创建一次运行，不自动重试、不支持断线重连或跨页面历史查询。

| SSE 事件 | 含义 |
| --- | --- |
| `run_started` | 本次请求的随机 `run_id` |
| `progress` | 连续 `seq`、独立 `invocation_id`、模型轮次、动作、开始/完成、相对时间、结果码和证据编号 |
| `result` | `run_id` 与 `response`；后者复用 `/api/investigate` 的完整响应合同 |
| `failure` | 流基础设施故障的固定错误码，不含内部异常正文 |

HTTP 头发送前的未启用、参数错误和并发繁忙仍返回 503/422/429 JSON。
流开始后的模型或工具失败由 `result.response.status` 表示，不通过 HTTP 200 推断成功。
没有最终结果的断流显示“连接已中断”，不会自动重发可能产生费用的请求。

进度观察与模型使用的 `AgentEvent` 分离；不改变提示词、模型输入、预算、配置指纹或
既有 JSON 接口响应。事件只含公共动作、编号和计数，不包含模型原始输出或内部思考。
每请求队列上限 256；溢出请求合作取消并发送失败事件，不伪造完整轨迹。

停止会中断浏览器接收并请求后端合作取消。真实运行显示“已请求停止”，而不是宣称
供应商已停止；正在执行的阻塞调用仍受自身 timeout 约束，实际结束后才释放并发名额。
这不保证供应商停止计费。模拟播放则可以立即停止。

对于旧服务，客户端回退到 `/api/investigate`，仅展示完成后返回的记录。
旧 `elapsed_seconds` 是相对运行开始的时间，不能当成单个操作耗时。

## 验证

```powershell
cd frontend
npm test
npm run build
npm run format:check
npm run test:e2e
```

本机 E2E 使用已安装的 Chrome；CI 使用 Playwright Chromium。浏览器测试仅使用原创
合成脚本和拦截后的合成 API 响应，不调用真实模型。后端的事件顺序、取消、溢出、
共享并发和原有输出兼容性由离线 pytest 验证。

真实 chat、embedding、rerank 与付费试次仍暂停。本工作台的视觉/交互验收不替代
Agent 的真实任务效果验收；持久会话、跨会话回放、工作流编辑器不在本版范围内。

2026-09-29 本机验证：Python 全套 1,735 项、前端单元测试 13 项、浏览器场景 9 项
通过；生产构建额外检查主题、窄屏、停止和合成 SSE 接入。TypeScript 构建、
ruff check / format、mypy、Prettier 和 diff 空白检查通过。浅深主题与窄屏截图已检查。
Python 仍有既存 Starlette/httpx 弃用提醒；构建有上游 Zod 注释标记提醒，无构建失败。
