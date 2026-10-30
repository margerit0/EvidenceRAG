# v02 · 成片

在 v01 成片（43 秒）的基础上加强检查器介绍。0–36 秒沿用 v01 的镜头、录制与相机；36–48.5 秒新增检查器段，展示步骤、证据、运行三个页签与同类调用导航；品牌收尾移到 48.5–51.5 秒。

- 成片：[final/evidencerag-resumable-v02.mp4](final/evidencerag-resumable-v02.mp4)
  ——51.5 秒，1920×1080，30 fps 恒定帧率（1545 帧），H.264 High + AAC-LC 48 kHz 立体声，10,037,316 字节
- SHA-256：`c1b0d444cd0e6c1bf9265435cdd07caeff2292074a310049414576eeec216e6d`
- 检查拼图（均解码自成片）：[final/check-sheet.png](final/check-sheet.png)（21 个关键帧）、[final/transitions.png](final/transitions.png)
- 检查记录：[final/verify-media.json](final/verify-media.json)、[final/verify-final.json](final/verify-final.json)、
  [final/verify-audio.json](final/verify-audio.json)、[final/playback.json](final/playback.json)；
  编码命令 [final/evidencerag-resumable-v02.mp4.encode.json](final/evidencerag-resumable-v02.mp4.encode.json)，
  抽帧记录 [final/extraction.json](final/extraction.json)，配乐参数 [final/audio-score.json](final/audio-score.json)
- 时间轴：[timeline.json](timeline.json) → [plan.json](plan.json)
- 已确认的分镜总览：[preview/overview.jpg](preview/overview.jpg)；预览页、关键帧与无声小样只留在本地

## 确认记录

- 需求（2026-10-30）：用户指出 README 视频对检查器的介绍少了一点，并选择含运行页签、约 51.5 秒的方案。
- 确认对象：v02 分镜预览，timeline `a2f8eb76dceaf4b4d74915a632368192987cc16be0de594339c0ac523c444989`，
  plan `e017c1b068b20cbefaf80b8d07023c3531427d9d49941948d126b851e32b9dc5`；14 个镜头，51.5 秒。
- 2026-10-30 用户原话：「The v02 preview is confirmed; make the full video and soundtrack」。
- 成片与预览共用同一份 plan.json；改动只有下面的方案内修复。

## 相对 v01 的变化

| 时间 | 镜头 | 画面 |
| --- | --- | --- |
| 36.0–40.0s | 步骤检查器（新增） | 点击流程图“读取证据”节点，检查器打开“步骤”页：读取证据 · 第 2 次，调用 ID、调用状态、模型轮次、耗时、运行后时间、返回码 |
| 40.0–42.5s | 同类调用切换（新增） | 点击“上一次”：第 2 / 2 次 → 第 1 / 2 次，流程图节点同步显示“正在查看第 1 次” |
| 42.5–46.5s | 证据页原文（原“引用核对”） | 点击“证据”页签，列出本次实际读取的 2 份段落；选择“回退演练”显示完整原文 |
| 46.5–48.5s | 运行概览（新增） | 点击“运行”页签：模型调用 6、搜索 / 读取 2 / 2、总耗时、运行编号 |

- 检查器镜头的字幕居中于执行流程栏（`captionX`），不遮挡检查器标题。
- 步骤镜头从流程图节点进入：运行完成后时间线只显示最后 3 条，读取记录不在可见区域内。
- 读取调用没有“决策内容”区（只有 Agent 决策调用有），步骤镜头展示的是调用属性。时间轴中 39.2 秒关键帧的标注写作“决策内容与调用属性”，并不准确；它只是检查拼图的标注，不出现在画面里。为了不改动已确认的 plan 哈希，没有改它。
- 配乐：检查器展开的气声从“点击引用 2”移到“点击读取证据节点”；新增的 5 次点击各有点击音；“已生成答案”的音乐状态延续到品牌收尾。

## 相对预览的方案内修复

| 问题 | 修复 | 证据 |
| --- | --- | --- |
| 点击“证据”页签后，光标在 F1298–1299 停在“证据 2”字样上 | 点击点改为页签右侧空白处（`targets['tab-evidence'].at`） | `verify-final.json` 中 `cursorFlagged` 为空 |
| 同类调用、运行两段停顿开始时，光标还在让位 | 两段停顿从镜头内 1.0 秒改为 1.2 秒开始（plan 帧与事件不变） | 停顿区间静止，`holdsPng` 全部通过 |

## 复现（仓库根目录，Windows / Git Bash）

依赖与 v01 相同（见 [../v01/README.md](../v01/README.md)）；界面使用 `frontend/` 在 `e794086` 的生产构建（v02 制作时 frontend 自该提交起未改），数据沿用 v01 的录制 [source/api-recording.json](source/api-recording.json)。

```bash
# 1. 编译时间轴
python ~/.claude/skills/product-demo-video/scripts/timeline.py docs/resumable-promo/versions/v02/timeline.json --out docs/resumable-promo/versions/v02/plan.json

# 2. 逐帧渲染全片（首次运行加 --build），以及作确定性对照的 26 帧
node docs/resumable-promo/versions/v02/source/render.mjs --plan docs/resumable-promo/versions/v02/plan.json --recording docs/resumable-promo/versions/v02/source/api-recording.json --app .research_tmp/resumable-promo/v01/app-dist --out .research_tmp/resumable-promo/v02/frames-final
node docs/resumable-promo/versions/v02/source/render.mjs --plan docs/resumable-promo/versions/v02/plan.json --recording docs/resumable-promo/versions/v02/source/api-recording.json --app .research_tmp/resumable-promo/v01/app-dist --out .research_tmp/resumable-promo/v02/frames-final-subset --list 45,171,200,240,336,348,405,480,493,585,660,700,759,819,903,988,1000,1024,1059,1110,1176,1254,1300,1365,1443,1527

# 3. 配乐
.venv/Scripts/python.exe docs/resumable-promo/versions/v02/source/score.py --plan docs/resumable-promo/versions/v02/plan.json --out .research_tmp/resumable-promo/v02/audio/score-final.wav

# 4. 编码、探测与抽帧
python ~/.claude/skills/product-demo-video/scripts/media.py encode --plan docs/resumable-promo/versions/v02/plan.json --frames .research_tmp/resumable-promo/v02/frames-final --audio .research_tmp/resumable-promo/v02/audio/score-final.wav --out docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4
python ~/.claude/skills/product-demo-video/scripts/media.py verify --plan docs/resumable-promo/versions/v02/plan.json --video docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4 --out .research_tmp/resumable-promo/v02/verify-media.json
python ~/.claude/skills/product-demo-video/scripts/media.py extract --plan docs/resumable-promo/versions/v02/plan.json --video docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4 --font C:/Windows/Fonts/msyh.ttc --out .research_tmp/resumable-promo/v02/extract-final

# 5. 成片检查
python docs/resumable-promo/versions/v02/source/verify_final.py --version-dir docs/resumable-promo/versions/v02 --frames .research_tmp/resumable-promo/v02/frames-final --subset .research_tmp/resumable-promo/v02/frames-final-subset --video docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4 --repo . --font C:/Windows/Fonts/msyh.ttc --sheet .research_tmp/resumable-promo/v02/transitions.png --out .research_tmp/resumable-promo/v02/verify-final.json
.venv/Scripts/python.exe docs/resumable-promo/versions/v02/source/verify_audio.py --plan docs/resumable-promo/versions/v02/plan.json --wav .research_tmp/resumable-promo/v02/audio/score-final.wav --video docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4 --out .research_tmp/resumable-promo/v02/verify-audio.json
node docs/resumable-promo/versions/v02/source/playback.mjs --video docs/resumable-promo/versions/v02/final/evidencerag-resumable-v02.mp4 --frames .research_tmp/resumable-promo/v02/frames-final --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe" --at 0,171,585,819,1059,1176,1254,1365,1443,1527 --out .research_tmp/resumable-promo/v02/playback.json
```

各脚本拒绝覆盖已有输出，重跑时换新路径。第 5 步的输出按原名复制到 `final/`。

## 验证记录

| 类别 | 项目 | 结果 |
| --- | --- | --- |
| 自动 | 成片规格 | 通过：1920×1080、30 fps 恒定帧率、1545 帧、51.5 秒，H.264 High + AAC-LC（`verify-media.json`） |
| 自动 | 渲染与交互 | 通过：页面无报错；14 次点击都在计划帧、落在 15 帧拍点上，命中可用的顶层目标；12 项点击后的界面断言通过；光标没有停在文字上 |
| 自动 | 文案 | 通过：步骤页（读取证据 · 第 2 / 1 次、调用 ID `call-10` / `call-4`，来自录制进度）、证据页（证据 02「回退演练」与完整原文，来自录制来源）、运行页文案逐字一致，都已绘制且在取景内 |
| 自动 | 确定性 | 通过：另起浏览器、按非顺序帧号渲染的 26 帧与全片相比，最大差 4/255（抗锯齿） |
| 自动 | 停顿 | 通过：11 个停顿区间；步骤详情与证据原文各稳定 54 帧（要求 ≥ 45），品牌稳定 45 帧（要求 ≥ 30） |
| 自动 | 成片逐帧 | 通过：1545 帧 PSNR 最低 40.18 dB、平均 43.81 dB；“合成演示 · 流程已压缩”在全部帧中可见 |
| 自动 | 音轨 | 通过：音效落在计划帧的样本上，AAC 与 WAV 对齐；削波 0，整体 −17.6 LUFS，响度范围 3.9 LU，真峰值 −3.0 dBTP |
| 程序播放 | Chrome 156 无头模式，1 倍速 | 通过：51.53 秒播完，1545 帧丢帧 0，帧回调逐帧连续；音频包络在 0 ms 处对齐（r = 0.9996）；10 帧解码与 PNG 的 PSNR 43.0–45.8 dB |
| 目视 | 成片画面 | 已看检查拼图 21 帧，以及检查器段的预览帧 |
| 未做 | 人工完整观看与主观听音 | 没有在真实显示器上连续观看，也没有听过音轨 |

## 已知局限

与 v01 相同：码流没有写入色彩标注（Chrome 中浅色界面偏差约 1–2 级）；局部取景的左右边会切到相邻栏目；F432–434 光标掠过占位文字。片中耗时来自合成后端且流程已压缩，不代表产品速度；引用可追溯也不等于结论通过了独立的质量评测。
