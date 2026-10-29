"""Build the storyboard preview page for one promo version.

Reads plan.json, brief.json and the render log, copies keyframes as JPEG and
writes preview/index.html. Python 3.10+ with Pillow.

    python build_preview.py --version-dir docs/resumable-promo/versions/v01 \
        --frames <rendered-frames-dir> --clips <dir-with-clip-mp4s>
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from fractions import Fraction
from pathlib import Path

from PIL import Image

ACTIONS = {
    "open-mode": "点击“运行模式”",
    "pick-live": "选择“连接服务”",
    "focus-query": "点击输入框",
    "type-query": "逐字输入问题",
    "start-run": "点击“开始调查”",
    "waiting-input": "系统：等待补充信息",
    "focus-supplement": "点击补充输入框",
    "type-supplement": "逐字输入补充",
    "submit-supplement": "点击“提交补充并继续”",
    "pause": "点击“暂停调查”",
    "paused": "系统：已暂停",
    "resume-1": "点击“继续调查”",
    "service-stop": "系统：服务正常关闭，进度连接断开",
    "reconnected": "系统：服务重启，前端自动重连",
    "resume-2": "点击“继续调查”",
    "answered": "系统：已生成答案",
    "open-citation": "点击引用 2，检查器展开",
}
CAMERA = {
    "full": "完整界面",
    "mode": "运行模式区域",
    "conversation": "问题与答案栏",
    "execution": "执行流程栏",
    "controls": "调查控制区",
    "work": "问题 + 执行两栏",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    args = parser.parse_args()
    version = args.version_dir
    plan = read_json(version / "plan.json")
    brief = read_json(version.parent.parent / "brief.json")
    preview = version / "preview"
    (preview / "keyframes").mkdir(parents=True, exist_ok=True)
    fps = Fraction(plan["video"]["fps"])
    esc = html.escape

    labels: dict[int, list[str]] = {}
    for item in plan["keyframes"]:
        labels.setdefault(item["frame"], []).append(item["label"])
    for frame in labels:
        source = args.frames / f"frame-{frame:06d}.png"
        target = preview / "keyframes" / f"frame-{frame:06d}.jpg"
        with Image.open(source) as image:
            image.convert("RGB").save(target, quality=92, optimize=True)

    rows = []
    for shot in plan["shots"]:
        start = shot["startFrame"] / float(fps)
        end = shot["endFrameExclusive"] / float(fps)
        events = [
            f"{e['seconds']:.1f}s {ACTIONS.get(e['id'], e['id'])}"
            for e in plan["events"]
            if e["shot"] == shot["id"]
        ]
        holds = [
            f"{h['startFrame'] / float(fps):.1f}–{h['endFrameExclusive'] / float(fps):.1f}s {h['label']}"
            for h in plan["holds"]
            if h["shot"] == shot["id"]
        ]
        camera = " → ".join(CAMERA.get(c[1], c[1]) for c in shot["data"].get("camera", []))
        frames = [k for k in plan["keyframes"] if k["shot"] == shot["id"]]
        thumbs = "".join(
            f'<figure><img src="keyframes/frame-{k["frame"]:06d}.jpg" loading="lazy" alt="{esc(k["label"])}">'
            f'<figcaption>{k["frame"] / float(fps):.2f}s · F{k["frame"]} · {esc(k["label"])}</figcaption></figure>'
            for k in frames
        )
        rows.append(
            f"<section class=shot><header><span class=time>{start:.1f}–{end:.1f}s</span>"
            f"<h3>{esc(shot['label'])}</h3><span class=dur>{shot['durationSeconds']:.1f} 秒</span></header>"
            f"<p class=caption>字幕：{esc(shot['data'].get('caption') or '（无，品牌收尾）')}</p>"
            f"<p><b>动作</b>：{esc('；'.join(events) or '无用户操作，系统自动推进或静止阅读')}</p>"
            f"<p><b>取景</b>：{esc(camera)}</p>"
            f"<p><b>稳定阅读</b>：{esc('；'.join(holds) or '—')}</p>"
            f"<div class=thumbs>{thumbs}</div></section>"
        )

    clips = []
    for clip in plan["previewClips"]:
        name = f"clip-{clip['id']}.mp4"
        if not (args.clips / name).exists():
            continue
        shutil.copy2(args.clips / name, preview / name)
        clips.append(
            f'<figure><video src="{name}" controls muted playsinline preload="metadata"></video>'
            f"<figcaption>{esc(clip['id'])} · {clip['startFrame'] / float(fps):.1f}–"
            f"{clip['endFrameExclusive'] / float(fps):.1f}s（无声小样）</figcaption></figure>"
        )

    coverage = brief["coverage"]
    names = {"overview": "产品全貌", "configuration": "关键配置", "coreOperation": "核心操作", "result": "结果"}
    decision = {"show": "展示", "omit": "省略", "not_applicable": "不适用"}
    cover_rows = "".join(
        f"<tr><td>{names[key]}</td><td>{decision[coverage[key]['decision']]}</td>"
        f"<td>{esc(coverage[key]['evidence'])}</td><td>{esc(coverage[key]['reason'])}</td></tr>"
        for key in names
    )
    omitted = "".join(f"<li>{esc(o['item'])}：{esc(o['reason'])}</li>" for o in coverage["omitted"])
    taglines = "".join(f"<li>{esc(t)}</li>" for t in brief["tagline"]["candidates"])
    total = plan["video"]["durationSeconds"]
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v01 分镜预览 · EvidenceRAG 可恢复调查</title>
<style>
:root{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;color:#263845;background:#e6eff4}}
body{{margin:0;padding:28px}} main{{max-width:1240px;margin:auto}}
h1{{font-size:26px;margin:0}} h2{{font-size:19px;margin:30px 0 12px}} h3{{font-size:17px;margin:0}}
.pill{{display:inline-block;padding:3px 12px;border-radius:999px;background:#fff3d6;color:#7a4f0f;font-size:13px}}
.muted{{color:#536b7c;font-size:13px}} p{{margin:6px 0;font-size:14px;line-height:1.7}}
section.shot,.box{{background:#ffffffb8;border:1px solid #fff;border-radius:18px;padding:16px 18px;margin:14px 0}}
section.shot header{{display:flex;gap:14px;align-items:baseline}} .time{{font-variant-numeric:tabular-nums;color:#527b96;font-weight:600}}
.dur{{margin-left:auto;color:#536b7c;font-size:13px}} .caption{{color:#2f4a5c}}
.thumbs{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:10px}}
figure{{margin:0}} img,video{{width:100%;border-radius:10px;display:block;background:#cfdde6}}
figcaption{{font-size:12px;color:#536b7c;margin-top:4px}}
.clips{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td:nth-child(-n+2),th:nth-child(-n+2){{white-space:nowrap}} th,td{{text-align:left;padding:7px 8px;border-top:1px solid #46628226;vertical-align:top}}
ul{{margin:6px 0 6px 20px;font-size:14px;line-height:1.7}}
</style></head><body><main>
<h1>EvidenceRAG · 可暂停、可恢复的文档调查 — 分镜预览 v01</h1>
<p><span class=pill>等待确认：内容覆盖、推荐时长、镜头安排与标语</span></p>
<p class=muted>时间轴 {plan['timelineSha256'][:12]} · plan {plan['planSha256'][:12]} · {len(plan['shots'])} 个镜头 · 推荐 {total:.0f} 秒 · 1920×1080 @ {fps} fps。
画面全部来自 frontend 当前提交的生产构建，配合原创合成后端（synthetic-demo）的逐次录制回放；未调用模型、未用真实语料。</p>

<h2>推荐时长：约 {total:.0f} 秒</h2>
<div class=box><p>{esc(brief['spec']['durationReason'])}。离散操作（点击、状态切换、面板展开）对齐 120 BPM 的节拍（每拍 15 帧），系统自动推进的步骤不由光标点击。
全程右上角保留“合成演示 · 流程已压缩”；片中显示的耗时是合成后端（每次模型调用固定延迟 2 秒）的真实记录，不代表模型速度。</p></div>

<h2>内容覆盖</h2>
<div class=box><table><tr><th>检查项</th><th>决定</th><th>片中证据</th><th>理由</th></tr>{cover_rows}</table>
<p><b>未纳入</b>：</p><ul>{omitted}</ul></div>

<h2>标语候选（品牌收尾）</h2>
<div class=box><ul>{taglines}</ul><p class=muted>当前关键帧使用第一条；可改为任一候选或你自己的文案。</p></div>

<h2>分镜</h2>
{''.join(rows)}

<h2>可播放小样（无声）</h2>
<div class=clips>{''.join(clips)}</div>
<p class=muted>小样由同一份逐帧渲染编码，包含关键配置菜单、运行中的连续动效（节点旋转图标与连线信号流）及运行结束后的停止、暂停/继续、服务重启重连、引用面板展开与品牌收尾。正式配乐与完整成片在确认后制作。</p>

<h2>总览</h2>
<figure><img src="overview.jpg" alt="分镜总览"><figcaption>按关键帧排列，标注时间、帧号、镜头、动作、取景与稳定区间。</figcaption></figure>
</main></body></html>
"""
    (preview / "index.html").write_text(page, encoding="utf-8")
    print(preview / "index.html")


if __name__ == "__main__":
    main()
