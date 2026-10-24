"""Package the editable revision and write documentation from verified results."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import zipfile

from zhrag.io_utils import read_bytes, read_json, read_text, write_json, write_text

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[1]
FINAL = HERE / "final-motion"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    report = read_json(FINAL / "technical-checks.json")
    playback = read_json(FINAL / "playback-checks.json")
    assert report["status"] == "passed"
    assert report["sha256"] == hashlib.sha256(read_bytes(FINAL / "EvidenceRAG-12s.mp4")).hexdigest()
    audio = report["audio"]
    timing = {
        "fps": 30, "frame_count": 360, "seconds": 12, "bpm": 120, "beats": 24,
        "events": [
            {"frame": f, "seconds": f / 30, "action": action}
            for f, action in [
                (45, "点击运行演示"), (60, "第一次搜索"), (75, "第一次读取"),
                (90, "返回 Agent 准备补充搜索"), (105, "补充搜索"), (120, "第二次读取"),
                (135, "点击第二次读取记录，展开步骤检查器"), (150, "第一条答案"),
                (165, "第二条答案"), (210, "点击引用 [2]，切到证据检查器"),
                (225, "开始原文稳定阅读"), (300, "工作台收拢为产品卡片"),
                (315, "品牌文字进入"), (330, "品牌稳定停留"),
            ]
        ],
        "holds": report["holds"],
        "question": "示例集群升级前，需要检查哪些配置与回退条件？",
        "source2": "升级演练 / 回退条件",
    }
    assert all(e["frame"] % 15 == 0 for e in timing["events"])
    write_json(HERE / "timing-plan.json", timing)
    readme = f"""# EvidenceRAG · 原比例工作台宣传短片

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
| 原文 / 品牌停顿 | {report['holds']['source_seconds']:.1f} 秒 / {report['holds']['brand_seconds']:.1f} 秒，停顿区间逐帧像素一致 |
| 完整播放 | 本机 Chrome 1 倍速，音轨启用，{playback['totalVideoFrames']} 帧 / {playback['droppedVideoFrames']} 丢帧 |
| 4 倍采样音频峰值 | {audio['true_peak_dbfs_4x']:.2f} dBFS，{audio['clipped_samples']} 个削波样本 |
| 编码后音轨偏移 | {audio['alignment_lag_samples']} 个采样 |

## 编辑与复现

在项目根目录执行：

```powershell
Set-Location D:\\rag
.\\.venv\\Scripts\\python.exe docs/evidencerag-promo/revision-03/render_movie.py
.\\.venv\\Scripts\\python.exe docs/evidencerag-promo/revision-03/verify_movie.py
.\\.venv\\Scripts\\python.exe docs/evidencerag-promo/revision-03/package_delivery.py
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
"""
    write_text(HERE / "README.md", readme)
    verify = f"""# 成片检查记录 · 浏览器原比例与原生 Agent 图

对象：`final-motion/EvidenceRAG-12s.mp4`

SHA-256：`{report['sha256']}`

- 最终 MP4 解码抽取第 45 / 105 / 195 / 255 / 345 帧，即 1.5 / 3.5 / 6.5 / 8.5 / 11.5 秒；已目视检查主问题、功能标签、两条摘要、引用、原文与品牌。
- 卡片、正文和高亮无异常越界；中文未见缺字。两条答案摘要逐帧按完整文字核对，所有画面均无 Unicode 替换字符。[2] 对应「升级演练 / 回退条件」，高亮逐字匹配 `demo.ts` 中的合成原文。
- 第 0–299 帧逐帧检查原生图的四类节点、五条路径、箭头和连线两端，检查器展开期间未丢线或裁切。落位后三栏保持 484 / 650 / 500 px。
- 已恢复前端连线信号、运行图标旋转、节点和连线渐变、状态图标入场。90 帧存在沿真实路径流动的短线，105 帧存在运行图标旋转。第 84 / 89 帧的最终 MP4 局部像素核验确认信号与旋转均真正进入成片，详见 `final-motion/graph-motion-checks.json`。
- 第 45 / 135 / 210 帧光标分别命中运行演示、第二次读取时间线记录与 [2]；对应音效也在这些帧开始。
- 4.5 秒点击后右栏显示「读取证据 · 第 2 次」；7 秒点击引用后，同一检查器切到证据。全部关键事件落在 15 帧拍点。
- 第 225–299 帧原文画面像素一致，稳定 {report['holds']['source_seconds']:.1f} 秒；第 330–359 帧品牌画面像素一致，稳定 {report['holds']['brand_seconds']:.1f} 秒。
- 编码视频 360 帧、12 秒；帧时间戳连续，最大偏差 {report['pts_max_error_seconds']:.9f} 秒。帧间亮度变化较大的位置位于检查器展开和收尾转场，未见异常黑白闪帧。
- 本机 Chrome 以 1 倍速、音轨启用完整播放至结束，经过 {playback['wallMs']/1000:.3f} 秒，{playback['totalVideoFrames']} 帧、{playback['droppedVideoFrames']} 丢帧。每半秒留存实际播放画面，记录视频呈现和音轨信号。
- AAC 解码 {audio['samples']} 个双声道采样，与 WAV 互相关偏移 {audio['alignment_lag_samples']} 个采样；4 倍采样峰值 {audio['true_peak_dbfs_4x']:.2f} dBFS，削波样本 {audio['clipped_samples']}。整体响度 {audio['integrated_lufs']:.1f} LUFS；最后 100 ms RMS 为 {audio['tail_100ms_rms_dbfs']:.1f} dBFS，尾音自然衰减。
- 过渡检查后修正阅读停顿的 DOM 重绘波动和品牌背景叠加，并重新渲染完整成片。
- 修正渲染器 stdin 在字节块边界切断 UTF-8 中文的问题，使用流式文本解码；完成重新渲染、逐帧文字核验和最终 MP4 抽帧复核。
- 另以非顺序帧号独立重渲 7 帧，状态和布局一致；浏览器抗锯齿存在微小像素差异，详见 `final-motion/repro-checks.json`。成片内要求的两个静止区间仍逐帧像素完全一致。

检查证据：`final-motion/technical-checks.json`、`final-motion/playback-checks.json`、`final-motion/check-sheet.png`、`final-motion/render-report.json`、`final-motion/audio-score.json`。`final-motion/playback-contact.png` 为本地生成的完整播放检查图。

本次没有运行真实模型、旧 v2 审核或后端评测。产品代码未修改。
"""
    write_text(HERE / "VERIFY.md", verify)
    root_readme = """# EvidenceRAG 产品宣传短片

本目录保存已确认的浏览器原比例视频方案，展示提问、Agent 搜索与读取、带引用答案和检查器原文阅读。流程图保留前端原有的连线流动、运行旋转与状态渐变。所有资料来自仓库原创合成演示。

- [播放成片](revision-03/final-motion/EvidenceRAG-12s.mp4)
- [五帧检查拼图](revision-03/final-motion/check-sheet.png)
- [流程图动效局部](revision-03/final-motion/graph-motion-detail.gif)
- [编辑入口与复现命令](revision-03/README.md)
- [成片检查记录](revision-03/VERIFY.md)

Git 保存最终源码与交付资料。源码 ZIP、音轨、中间渲染帧和旧版视频属于本地生成产物，已由本目录 `.gitignore` 排除。
"""
    write_text(BASE / "README.md", root_readme)
    source_names = [
        "README.md", "VERIFY.md", "timing-plan.json", "movie.html", "frame.html",
        "storyboard.js", "native-overrides.css", "graph_motion.js", "motion.js",
        "media_data.py", "extract_demo.mjs", "demo_graph.mjs", "render_movie.py",
        "render_video.mjs", "synthesize_audio.py", "verify_movie.py", "verify_graph_motion.py",
        "playback.mjs", "render_sheet.mjs", "overview.html", "render_preview.py", "render.mjs",
        "capture_frontend.py", "capture_frontend.mjs", "package_delivery.py",
    ]
    selected = [HERE / name for name in source_names]
    selected += [HERE / "reference/native.json"]
    selected += [HERE / "preview/storyboard-overview.png", BASE / ".gitignore", BASE / "README.md"]
    selected += list((BASE / "licenses").glob("*"))
    selected += [REPO / "frontend/src/lib/demo.ts", REPO / "frontend/src/assets/glacier.svg", REPO / "frontend/src/components/ExecutionGraph.tsx", REPO / "frontend/src/styles.css", REPO / "frontend/src/glacier.css"]
    selected += [REPO / "src/zhrag/io_utils.py", REPO / "src/zhrag/__init__.py"]
    selected += [FINAL / "technical-checks.json", FINAL / "playback-checks.json", FINAL / "audio-score.json", FINAL / "encode-command.json", FINAL / "check-sheet.png", FINAL / "repro-checks.json", FINAL / "graph-motion-checks.json", FINAL / "graph-motion-detail.gif"]
    selected += [FINAL / "render-report.json", FINAL / "source-checks.json"]
    package = FINAL / "EvidenceRAG-source.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in sorted(set(selected)):
            if p.is_file():
                archive.writestr(p.relative_to(REPO).as_posix(), read_bytes(p))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert all(not any(part in name for part in ["/.env", "indexes/", "crud-rag-subset/", "tidb-rag-curated/"]) for name in names)
    manifest = {"revision": "03-native-motion", "approved": True, "files": {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(read_bytes(p)).hexdigest()}
        for p in [FINAL / "EvidenceRAG-12s.mp4", package, FINAL / "check-sheet.png"]
    }, "source_archive_entries": len(names), "technical_checks": "passed"}
    write_json(FINAL / "delivery-manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
