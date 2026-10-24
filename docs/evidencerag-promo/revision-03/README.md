# EvidenceRAG · 原比例工作台宣传短片

基于用户已明确确认的第三版分镜完成连续动画，并补齐前端原有的执行流程图动效。保留浏览器 100% CSS 尺度、原生 Agent 调度图和三栏检查器联动。

## 交付

- [带声音 MP4](final-motion/EvidenceRAG-12s.mp4)
- [可编辑源码 ZIP](final-motion/EvidenceRAG-source.zip)（本地打包后生成，Git 中直接保存编辑源码）
- [最终 MP4 五帧检查拼图](final-motion/check-sheet.png)
- [从成片截取的流程图动效局部](final-motion/graph-motion-detail.gif)
- [完整检查记录](VERIFY.md)
- [逐帧 HTML 动画](movie.html)：空格播放，1–5 定位关键帧，I 定位检查器点击；HTML 只预览画面，MP4 包含配乐。
- [已确认分镜拼图](preview/storyboard-overview.png)

## 画面与节奏

| 时间 | 动作 |
| --- | --- |
| 0–2 秒 | 问题预填；1.5 秒点击运行演示 |
| 2–5 秒 | 搜索、读取、补充搜索、再次读取自动推进；4.5 秒点击读取记录，展开步骤检查器 |
| 5–7 秒 | 两条带 [1]、[2] 的答案出现，右栏保留单次调用详情 |
| 7–10 秒 | 点击 [2]，同一检查器切到证据；7.5–10 秒静止阅读 |
| 10–12 秒 | 工作台收拢为产品卡片；11–12 秒稳定显示品牌 |

原生图保留 Agent 决策、搜索文档、读取证据、校验与输出四类节点及五条有向连线。补充搜索再次使用搜索节点，四次工具调用顺序在时间线中呈现。打开检查器后的三栏宽度为 484 / 650 / 500 px，与确认分镜一致。

流程图沿用 `ExecutionGraph.tsx` 的第二条 SVG 信号路径和 `styles.css` 的原始效果：连线短线按 1.4 秒周期向前流动，运行图标按 1.3 秒周期旋转；节点高亮用 220ms 渐变，连线颜色与粗细用 200ms 渐变，状态图标用 180ms 入场。运行开始、读取后返回 Agent 和答案校验均有对应动效；完成后停止旋转与流动。箭头颜色同步执行状态。

标题 29px、问题 16px、答案与引用 13px；1920 × 1000 网页视口居中放入 1920 × 1080 画布。全程保留「合成演示 · 流程已压缩」。问题和引文来自 `frontend/src/lib/demo.ts` 的原创合成资料。

## 已验证规格

| 项目 | 结果 |
| --- | --- |
| 视频 | 1920 × 1080，30 fps，360 帧，12.000 秒 |
| 编码 | H.264 / AAC，yuv420p，BT.709 |
| 音频 | 48 kHz，双声道，12.000 秒；原创代码配乐，120 BPM / 24 拍 |
| 点击 | 第 45 / 135 / 210 帧，画面与音效同拍 |
| 原文 / 品牌停顿 | 2.5 秒 / 1.0 秒，停顿区间逐帧像素一致 |
| 完整播放 | 本机 Chrome 1 倍速，音轨启用，360 帧 / 0 丢帧 |
| 4 倍采样音频峰值 | -5.68 dBFS，0 个削波样本 |
| 编码后音轨偏移 | 0 个采样 |

## 编辑与复现

在项目根目录执行：

```powershell
Set-Location D:\rag
.\.venv\Scripts\python.exe docs/evidencerag-promo/revision-03/render_movie.py
.\.venv\Scripts\python.exe docs/evidencerag-promo/revision-03/verify_movie.py
.\.venv\Scripts\python.exe docs/evidencerag-promo/revision-03/package_delivery.py
```

第一条逐帧渲染、合成配乐并编码 MP4；第二条检查参数、解码抽取五帧、生成拼图并完整播放；第三条从检查结果生成说明与源码包。`render_movie.py --samples` 只渲染过渡检查帧；`--encode-only` 复用已有 PNG 序列重新合成配乐和编码。

| 文件 | 编辑内容 |
| --- | --- |
| `motion.js` | 统一帧时间轴、检查器展开、引用切换、光标与收尾 |
| `graph_motion.js` | 按帧定位前端原生连线、加载图标、状态渐变和选中虚线 |
| `storyboard.js` / `frame.html` | 已确认的页面结构、原文、品牌卡片 |
| `native-overrides.css` | 在原生界面上进行的视频精简与高亮 |
| `reference/native.json` | 本地原创合成演示的 DOM / CSS 快照 |
| `demo_graph.mjs` | 从合成事件计算节点调用计数与五条连线状态 |
| `synthesize_audio.py` | 原创和弦、旋律、点击与落位音效 |
| `timing-plan.json` | 拍点、镜头顺序与停顿 |
| `render_movie.py` / `render_video.mjs` | 构建 HTML、逐帧 PNG、几何检查、编码 |
| `verify_movie.py` / `playback.mjs` | MP4、音轨、解码帧与完整播放检查 |

每帧由整数帧号驱动，可在控制台 `await window.movieReady; window.renderMovieFrame(255)` 定位。停顿区间不重复修改 DOM，避免 SVG 重绘带来的细微像素波动。文字保持正常字号，通过状态、选中标记和面板展开引导视线。

图动效采用暂停后逐帧设置时间的 Web Animations，复用前端关键帧、周期和缓动。自动运行的 CSS 时钟仍关闭，避免截图速度改变动画节奏。`verify_graph_motion.py` 同时核对逐帧的信号偏移、旋转时间和最终 MP4 的局部像素变化，确保成片里确实有连续动效。

复用项目 Python 环境（NumPy 与 `zhrag.io_utils`）、Node.js、`frontend/node_modules` 中的 TypeScript / Playwright 与 Inter。FFmpeg、FFprobe 在 PATH；渲染使用 Playwright Chromium，完整播放检查使用本机 Chrome。

新检出先按项目说明配置 Python，然后执行 `npm --prefix frontend ci` 和 `node frontend/node_modules/playwright/cli.js install chromium`。中文使用本机 Microsoft YaHei；系统字体未复制。Inter 等沿用原交付的许可文件。

Git 只保存本方案的编辑源码、自包含 `movie.html`、已确认分镜总览、成片、五帧拼图、动效局部和验证记录。源码 ZIP、PCM 音轨、单张抽帧与中间 PNG 序列由上述命令本地生成并忽略。旧视频草稿仍作为本地历史产物保留，不进入本次提交。

源码 ZIP 按仓库相对路径组织，解压到项目根目录可保持引用路径。数据提取由本目录的 `media_data.py` / `extract_demo.mjs` 完成，复现不依赖任何旧版宣传片脚本或 anatomy 设计文件。

渲染浏览器离线运行，未读取密钥、真实语料或调用模型。此轮只改宣传媒体源码与交付说明，未改产品代码。
