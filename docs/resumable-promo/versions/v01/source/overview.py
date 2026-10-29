"""Storyboard overview sheet for the resumable-investigation promo preview.

Lays out every rendered keyframe with its time, frame, shot, action, transition
and stable-hold interval, read from plan.json. Python 3.10+ with Pillow.

    python overview.py --plan ../plan.json --frames <keyframe-dir> --font <cjk.ttf> --out sheet.png
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ACTION_NAMES = {
    "open-mode": "点击“运行模式”",
    "pick-live": "选择“连接服务”",
    "focus-query": "点击输入框",
    "type-query": "输入问题",
    "start-run": "点击“开始调查”",
    "waiting-input": "系统：等待补充信息",
    "focus-supplement": "点击补充输入框",
    "type-supplement": "输入补充",
    "submit-supplement": "点击“提交补充并继续”",
    "pause": "点击“暂停调查”",
    "paused": "系统：已暂停",
    "resume-1": "点击“继续调查”",
    "service-stop": "系统：服务关闭，连接断开",
    "reconnected": "系统：服务重启，自动重连",
    "resume-2": "点击“继续调查”",
    "answered": "系统：已生成答案",
    "open-citation": "点击引用 2",
}


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        if draw.textlength(current + char, font=font) > width:
            lines.append(current)
            current = char
        else:
            current += char
    return [*lines, current][:3]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--font", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    fps = Fraction(plan["video"]["fps"])
    title_font = ImageFont.truetype(args.font, 30)
    font = ImageFont.truetype(args.font, 19)
    small = ImageFont.truetype(args.font, 16)

    shots = plan["shots"]
    by_frame: dict[int, list[str]] = {}
    for item in plan["keyframes"]:
        by_frame.setdefault(item["frame"], []).append(item["label"])
    frames = sorted(by_frame)

    def shot_of(frame: int) -> dict:
        return next(s for s in shots if s["startFrame"] <= frame < s["endFrameExclusive"])

    cell_w, image_h = 560, 315
    text_h = 150
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    margin = 22
    head = 92
    sheet = Image.new(
        "RGB",
        (columns * (cell_w + margin) + margin, head + rows * (image_h + text_h + margin) + margin),
        "#eef3f6",
    )
    draw = ImageDraw.Draw(sheet)
    total = plan["video"]["durationSeconds"]
    draw.text((margin, 18), f"{plan['title']}", font=title_font, fill="#1f3442")
    draw.text(
        (margin, 58),
        f"{len(shots)} 个镜头 · {total:.1f} 秒 · {plan['video']['width']}×{plan['video']['height']} @ {fps} fps"
        f" · 时间轴 {plan['timelineSha256'][:12]} · plan {plan['planSha256'][:12]}",
        font=small,
        fill="#415a6c",
    )
    for index, frame in enumerate(frames):
        x = margin + (index % columns) * (cell_w + margin)
        y = head + (index // columns) * (image_h + text_h + margin)
        path = args.frames / f"frame-{frame:06d}.png"
        with Image.open(path) as source:
            sheet.paste(ImageOps.fit(source.convert("RGB"), (cell_w, image_h)), (x, y))
        shot = shot_of(frame)
        seconds = float(Fraction(frame, 1) / fps)
        draw.rectangle((x, y + image_h, x + cell_w, y + image_h + text_h), fill="#ffffff")
        draw.text(
            (x + 12, y + image_h + 8),
            f"{seconds:5.2f}s · F{frame} · {shot['label']}",
            font=font,
            fill="#1f3442",
        )
        draw.text(
            (x + 12, y + image_h + 36), " / ".join(by_frame[frame]), font=small, fill="#2f4a5c"
        )
        events = [
            ACTION_NAMES.get(e["id"], e["id"]) + f" @{e['seconds']:.1f}s"
            for e in plan["events"]
            if e["shot"] == shot["id"]
        ]
        holds = [
            f"稳定 {h['startFrame'] / float(fps):.1f}–{h['endFrameExclusive'] / float(fps):.1f}s"
            for h in plan["holds"]
            if h["shot"] == shot["id"]
        ]
        camera = " → ".join(str(c[1]) for c in shot["data"].get("camera", []))
        line = "；".join(events) if events else "无用户操作"
        y_text = y + image_h + 60
        for text, color in ((f"动作：{line}", "#415a6c"), (f"取景：{camera}  {'；'.join(holds)}", "#536b7c")):
            for wrapped in wrap(draw, text, small, cell_w - 24):
                draw.text((x + 12, y_text), wrapped, font=small, fill=color)
                y_text += 21
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
