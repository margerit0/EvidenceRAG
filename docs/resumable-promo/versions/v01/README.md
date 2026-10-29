# v01 · 成片

EvidenceRAG「可暂停、可恢复的文档调查」宣传片 v01。分镜预览已确认（见下），按确认的方案制作了连续动画、代码配乐与 MP4。

- 成片：[final/evidencerag-resumable-v01.mp4](final/evidencerag-resumable-v01.mp4)
  ——43.0 秒，1920×1080，30 fps 恒定帧率（1290 帧），H.264 High + AAC-LC 48 kHz 立体声，8,749,342 字节
- SHA-256：`21210c048016742c6788bdd0d1f92ebb7d9515a487a406e6d8dba1bd85a7a1e6`
- 检查拼图（均解码自成片）：[final/check-sheet.png](final/check-sheet.png)（18 个关键帧）、
  [final/transitions.png](final/transitions.png)（12 次运镜的中点与品牌转场 4 帧）
- 检查记录：[final/verify-media.json](final/verify-media.json)、[final/verify-final.json](final/verify-final.json)、
  [final/verify-audio.json](final/verify-audio.json)、[final/playback.json](final/playback.json)；
  编码命令 [final/evidencerag-resumable-v01.mp4.encode.json](final/evidencerag-resumable-v01.mp4.encode.json)，
  抽帧记录 [final/extraction.json](final/extraction.json)，配乐参数 [final/audio-score.json](final/audio-score.json)
- 时间轴：[timeline.json](timeline.json) → [plan.json](plan.json)（帧号、拍点、停顿与音频样本数都由它派生）
- 已确认的分镜总览：[preview/overview.jpg](preview/overview.jpg)。预览页、关键帧和无声小样只留在本地，由上级目录的
  `.gitignore` 排除；本地预览页页首的确认记录是归档时手工加的，重跑 [source/build_preview.py](source/build_preview.py) 会覆盖它

## 确认记录

- 确认对象：v01 分镜预览，timeline `15eddbec556f49938a8aab56a7a8011c2923f709b920f10effce297bfa9e78f8`，
  plan `4075604c0d597dd041836a4cd29e0da1e24f25c11aeea63a7ffb01091dade6f6`；11 个镜头，推荐 43 秒。
- 2026-10-29 11:04 UTC，会话 `9b985e4c-fa86-4fa9-9b30-445eb9390802`，用户原话：
  「读取 cb0dc5a2-af2a-44a7-8c83-b80738103483 会话的内容，继续任务，当前的分镜不错可以继续生成视频，」
  （`cb0dc5a2…` 是提出需求并交付预览的会话。）
- 标语：确认的是“当前的分镜”，没有另选，沿用预览关键帧中的第一条「停下来，也能接着查。」。
- 成片与预览共用同一份 plan.json，布局、功能和镜头顺序都没有改变；改动只有下面的方案内修复。

## 相对预览的方案内修复

| 预览中的问题 | 修复 | 成片中的证据 |
| --- | --- | --- |
| 19–23 秒「主动暂停」等局部取景的上沿切过页面副标题与图标 | [source/scene.mjs](source/scene.mjs) `cameraClearance`：每个局部取景在 ±48 CSS px 内上下微调，或放宽 2–8%，让上下边落在离文字至少 2 px 的空隙里；左右边仍按预览裁到相邻栏目 | 18 处取景中 10 处做了调整，相机静止时上下边切字 0 帧（`render.edgeCuts`）；F585、F660 副标题完整 |
| 35.3 秒光标停在答案出现的位置，盖住答案正文 | 第二次「继续调查」后光标停到状态行右端空白处（`motion.park['resume-2']`），答案出现在它下方 | F1059；光标没有停在任何文字上（`render.cursorOverText`） |
| 输入问题和补充时文本插入符不按片内时间闪烁 | [source/film-runtime.js](source/film-runtime.js) 画虚拟插入符，按片内时钟 500 ms 闪烁 | F195–254、F450–494（`render.caret`） |
| 检查脚本把折叠起来的「已提交的补充信息」当成可见文字 | `checkVisibility()` 只统计实际绘制的元素（检查用，不改画面；18 处取景结果不变） | 文案检查区分已绘制与折叠内容 |

## 配乐与音效

[source/score.py](source/score.py) 用代码合成无歌词配乐（F 大调，120 BPM），音符落在 plan.json 的节拍网格上。
音乐状态跟随片中状态：片头 → 已连接 → 运行 → 等待补充 → 运行 → 正在暂停 → 已暂停 → 运行 → 服务中断 → 已暂停 → 运行 → 已生成答案 → 品牌。
暂停时节奏层在 1.6 秒内撤出；服务中断期间只留低通处理的长音与慢速脉冲，重连后打开。

音效：10 次点击、18 个按键、4 个提示音（等待补充、已暂停、重连、已生成答案）、服务关闭的下滑音与闷响、开始调查前的上扬气声、检查器展开的气声、品牌收尾的低音拨弦。
每个音效都从对应事件帧的第一个样本开始。随机种子固定为 20261029；用 numpy 2.5.2 生成的 WAV SHA-256 是
`b4badb40ff4fb8fa6b4b149a7e14eeb6d1568c37480d47eab5a98e85424eb781`。
整体响度 −17.8 LUFS，响度范围 3.2 LU，真峰值 −3.1 dBTP。

## 画面从哪来

- **界面**：`frontend/` 在提交 `e794086` 的生产构建（`vite build` 输出到 Git 忽略的缓存目录），未改产品源码。
- **数据**：[source/record_backend.py](source/record_backend.py) 在本机回环端口启动仓库自带的原创合成后端
  （`create_demo_app`，模型 `synthetic-demo`，每次模型调用固定延迟 2 秒），经真实 HTTP 走完
  “创建 → 追问 → 补充 → 暂停 → 继续 → 服务正常关闭并在同一数据库重启 → 继续 → 完成”，
  把快照、事件与命令响应录进 [source/api-recording.json](source/api-recording.json)。不读 `.env`，不调用任何供应商。
- **回放**：[source/film-runtime.js](source/film-runtime.js) 通过 Playwright `addInitScript` 注入，提供虚拟时钟、
  录制接口与 SSE 回放、CSS 动画与过渡的逐帧定位，以及叠加层（光标、字幕、“合成演示 · 流程已压缩”、品牌卡）。
- **操作**：[source/render.mjs](source/render.mjs) 在时间轴帧上发出真实的鼠标与键盘事件，点击前检查命中目标，
  点击后按 [source/scene.mjs](source/scene.mjs) 核对界面结果；相机按 16:9 裁切，直接渲染为 1920×1080。
- 片中显示的耗时是合成后端的真实记录，但流程已压缩，不代表模型或产品的响应速度。
- **许可**：品牌卡图标是产品所用 Lucide `layers-2` 的路径，复制在 [source/scene.mjs](source/scene.mjs) 中，
  许可见 [../../licenses/Lucide-ISC.txt](../../licenses/Lucide-ISC.txt)。字体文件都没有复制：拉丁字体 Inter 来自 frontend 构建，中文用本机 Microsoft YaHei 渲染。

## 复现（仓库根目录，Windows / Git Bash）

依赖：
- Node 22，以及 `frontend/node_modules` 里的 Vite 与 playwright-core 1.56.1（渲染用其 Chromium headless shell 141）
- 系统 Chrome：H.264/AAC 播放检查需要品牌版，路径见下
- ffmpeg / ffprobe 7.1
- Python 3.13 + Pillow：`media.py`、`verify_final.py`
- 项目 `.venv`：`score.py` 与 `verify_audio.py` 需要 numpy，`record_backend.py` 需要 `zhrag`
- 字体 Microsoft YaHei（`C:/Windows/Fonts/msyh.ttc`）

`timeline.py` 与 `media.py` 来自 product-demo-video skill 的 `scripts/`，不在仓库内；下面的命令假定它装在 `~/.claude/skills/product-demo-video/`。
各脚本拒绝覆盖已有输出，重跑时换新路径。

```bash
# 0. 只在需要重录时运行：会生成新的调查 ID 与时间戳，画面随之变化；--db 必须是新文件
PYTHONPATH=src NO_PROXY=127.0.0.1,localhost .venv/Scripts/python.exe docs/resumable-promo/versions/v01/source/record_backend.py --out docs/resumable-promo/versions/v01/source/api-recording.json --db .research_tmp/resumable-promo/v01/record.sqlite3

# 1. 编译时间轴（plan.json 带自校验哈希；已确认的方案无需重编）
python ~/.claude/skills/product-demo-video/scripts/timeline.py docs/resumable-promo/versions/v01/timeline.json --out docs/resumable-promo/versions/v01/plan.json

# 2. 构建 frontend 并逐帧渲染全片（约 6 分钟，PNG 约 0.8 GB）
node docs/resumable-promo/versions/v01/source/render.mjs --plan docs/resumable-promo/versions/v01/plan.json --recording docs/resumable-promo/versions/v01/source/api-recording.json --app .research_tmp/resumable-promo/v01/app-dist --out .research_tmp/resumable-promo/v01/frames-final4 --build

# 3. 另起浏览器、按非顺序帧号独立渲染 22 帧，作确定性对照
node docs/resumable-promo/versions/v01/source/render.mjs --plan docs/resumable-promo/versions/v01/plan.json --recording docs/resumable-promo/versions/v01/source/api-recording.json --app .research_tmp/resumable-promo/v01/app-dist --out .research_tmp/resumable-promo/v01/frames-final4-subset --list 45,171,200,240,336,348,405,480,493,585,660,700,759,819,903,988,1000,1024,1059,1110,1170,1272

# 4. 配乐（同时写出 .json 与 .cues.json 旁注）
.venv/Scripts/python.exe docs/resumable-promo/versions/v01/source/score.py --plan docs/resumable-promo/versions/v01/plan.json --out .research_tmp/resumable-promo/v01/audio/score-final.wav

# 5. 编码、探测与抽帧
python ~/.claude/skills/product-demo-video/scripts/media.py encode --plan docs/resumable-promo/versions/v01/plan.json --frames .research_tmp/resumable-promo/v01/frames-final4 --audio .research_tmp/resumable-promo/v01/audio/score-final.wav --out docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4
python ~/.claude/skills/product-demo-video/scripts/media.py verify --plan docs/resumable-promo/versions/v01/plan.json --video docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4 --out .research_tmp/resumable-promo/v01/verify-media.json
python ~/.claude/skills/product-demo-video/scripts/media.py extract --plan docs/resumable-promo/versions/v01/plan.json --video docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4 --font C:/Windows/Fonts/msyh.ttc --out .research_tmp/resumable-promo/v01/extract-final

# 6. 成片检查：渲染日志、文案、像素与逐帧解码；音轨；Chrome 播放
python docs/resumable-promo/versions/v01/source/verify_final.py --version-dir docs/resumable-promo/versions/v01 --frames .research_tmp/resumable-promo/v01/frames-final4 --subset .research_tmp/resumable-promo/v01/frames-final4-subset --video docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4 --repo . --font C:/Windows/Fonts/msyh.ttc --sheet .research_tmp/resumable-promo/v01/transitions.png --out .research_tmp/resumable-promo/v01/verify-final.json
.venv/Scripts/python.exe docs/resumable-promo/versions/v01/source/verify_audio.py --plan docs/resumable-promo/versions/v01/plan.json --wav .research_tmp/resumable-promo/v01/audio/score-final.wav --video docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4 --out .research_tmp/resumable-promo/v01/verify-audio.json
node docs/resumable-promo/versions/v01/source/playback.mjs --video docs/resumable-promo/versions/v01/final/evidencerag-resumable-v01.mp4 --frames .research_tmp/resumable-promo/v01/frames-final4 --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe" --at 0,171,585,819,1059,1170,1272 --out .research_tmp/resumable-promo/v01/playback.json
```

第 6 步的输出按原名复制到 `final/`（`playback.json` 来自正式播放那次运行）。浏览器进程之间的抗锯齿差异最多 4/255，
所以重新渲染得到的 MP4 SHA-256 会不同，状态、布局与文字不变。中间帧、构建产物、WAV 与日志都在 Git 忽略的
`.research_tmp/resumable-promo/v01/`，可以整体删除后按上面的命令重建。

## 验证记录

自动检查、程序播放与目视检查分开记录；它们都不能替代人工完整观看与听音。

| 类别 | 项目 | 结果 |
| --- | --- | --- |
| 自动 | 成片规格 | 通过：1920×1080、30 fps 恒定帧率、1290 帧、43.0 秒，H.264 High + AAC-LC 48 kHz 立体声；帧时间戳连续（0 → 42.966667 秒），音视频轨同时起始（`verify-media.json`） |
| 自动 | 渲染与交互 | 通过：页面无报错、无控制台警告；10 次点击都在计划帧、落在 15 帧拍点上，命中可用的顶层目标；8 项点击后的界面断言通过；相机静止时上下边切字 0 帧（切字的 96 帧都在 12 次运镜过程中） |
| 自动 | 文案 | 通过：11 个探测帧上的可见文字与录制的合成后端、frontend 源码及已确认时间轴逐字一致，都已绘制且在取景内；折叠的补充记录确认没有绘制；17 条界面文案逐字存在于对应的 frontend 文件；答案两句、两条原文与追问逐字存在于 `src/zhrag/service/investigation_demo.py`（相邻的字符串字面量拼接后比较）；点击引用 2 后检查器显示「证据 02」「回退演练」与完整原文，与录制的来源逐字一致 |
| 自动 | 确定性 | 通过：另起浏览器、按非顺序帧号渲染的 22 帧与全片相比，没有像素差超过 15 级，最大差 4/255（抗锯齿） |
| 自动 | 停顿 | 通过：8 个停顿区间内，变化的只有镜头首尾字幕的淡入淡出，以及片头 F0–5 页面载入时产品自身的过渡（流程节点状态图标淡入，即 styles.css `node-status-enter` 180 ms；画布缩放控件的「−」按钮由浅变深）。从动效真正结束算起，原文稳定 69 帧（要求 ≥ 45），品牌稳定 45 帧（要求 ≥ 30） |
| 自动 | 连续动效 | 通过：运行中同一状态的两帧（F336/F348、F988/F1000）在 PNG 与成片中都有变化；运行结束后的「已生成答案」停顿内画面静止 |
| 自动 | 成片逐帧 | 通过：1290 帧逐帧对比各自的 PNG，PSNR 最低 40.18 dB、平均 43.8 dB，没有一帧更接近相邻帧；“合成演示 · 流程已压缩”在全部 1290 帧中都可见，包括转场与品牌帧 |
| 自动 | 成片停顿 | 通过：8 个稳定区间里，解码帧相对区间首帧超过 15 级的像素最多 9 个（阈值是一个 8×8 变换块，64 像素）。来源一是片头 F12 字幕刚停稳时的编码残差，F13 起回落；来源二是连接镜头 F166 单帧 8×1 像素的编码噪点 |
| 自动 | 音轨 | 通过：36 个带帧号的音效都落在计划帧的样本上；AAC 与源 WAV 的互相关偏移为 0 个样本（r = 0.998），各类音效起点相对 WAV 偏差 ≤ 0.125 ms；削波 0，真峰值 −3.1 dBTP |
| 程序播放 | Chrome 156 无头模式，1 倍速 | 通过：43.08 秒播完，1290 帧中丢帧 0，帧回调逐帧连续（1289 次；第 0 帧在加载时已显示）；每次回调都有音频信号，Web Audio 电平包络与文件音轨在 0 ms 处对齐（r = 0.9997）。声音经零增益节点静音 |
| 程序播放 | 色彩 | 7 帧对比：Chrome 解码与 PNG 的 PSNR 为 43.0–45.8 dB。平均偏差 R −0.9 至 −1.9、G −0.3 至 −1.2、B +0.03 至 +0.7 级（0–255）。原因见下方“已知局限” |
| 目视 | 成片画面 | 已看：`check-sheet.png` 18 帧、`transitions.png` 16 帧；停顿首尾、F12–13 与 F166 的局部放大；Chrome 解码帧与 PNG 并排对比 |
| 目视 | 引用与结论（人工对读） | 已对读：答案第一句与引用 1「版本准备」原文“甲版到乙版须先在隔离环境验证兼容性”一致；第二句几乎逐字取自引用 2「回退演练」原文。片中只打开引用 2。这只说明结论能对照原文，不是独立的质量评测 |
| 未做 | 人工完整观看与主观听音 | 没有在真实显示器上连续观看，也没有听过音轨。闪烁、节奏和音量平衡的主观判断有待人工观看与听音 |

另：早期两次播放检查报告过 58 和 49 帧丢弃。帧回调多处一次前进 2–3 帧，最多一次前进 7 帧，静止镜头中也有。
当时脚本加了 `--mute-audio`，分析器直接接到输出。之后用同一文件重跑了三种组合：当时的接法加 `--mute-audio`、现在的零增益接法加该开关、现在的接法不加。
三次都是 0 丢帧，早期的丢帧没能复现，原因未查明。几次运行用的是同一个文件，所以丢帧数的差别来自运行环境，而不是文件。
表中的 0 丢帧只说明这台机器上能以 1 倍速完整播放，不保证其他设备。这几次运行的记录在 Git 忽略的
`.research_tmp/resumable-promo/v01/playback-diagnostics/`。

## 已知局限

- **左右边裁切**：局部取景的左右边会切到相邻栏目，例如控制区镜头右侧的执行流程栏，两栏镜头边缘的「服务已连接」「第 5 次」。这与确认的预览一致；上下边已保证不切字。
- **色彩标注**：码流没有写入色彩空间标注。ffmpeg 按 BT.601 转换，Chrome 按 BT.709 解释，实测偏差与这一推算一致：
  浅色界面约 1–2 级，强调蓝约 4 级。修正需要在编码时写入色彩标注，而 skill 的 `media.py` 目前不支持。
- **光标掠过占位文字**：F432–434 光标起身前往补充输入框时，从底部输入框的占位文字「描述你想调查的问题…」上掠过 3 帧。这发生在追问镜头停顿的最后，字幕正在淡出；光标没有停在任何文字上。
- **耗时已压缩**：片中耗时来自合成后端（每次模型调用固定 2 秒），流程经过压缩，不能当作产品速度。引用可追溯也不等于结论通过了独立的质量评测。
