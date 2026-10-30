"""Checks of the final promo against its render, the recorded synthetic backend
and the encoded MP4. Python 3.10+ with Pillow; ffmpeg on PATH.

1. Render log: clicks on beat frames hit enabled targets and produce the
   expected UI, no page errors, frame edges never slice text while the camera
   is still, the cursor never rests on text it is not clicking and never
   crosses body text during a hold (other hold crossings are listed for review).
2. Copy: text read from the live DOM at probe frames equals the recording, the
   frontend source and the confirmed timeline; every compared element is
   painted and inside the camera. Expected UI strings must occur verbatim in
   the cited frontend file, so they are not invented here.
3. Pixels: an independent subset render matches the full render; inside each
   hold nothing moves but a caption fading at its shot's edge or documented
   native motion, and the stable stretch (counted from when motion really
   ends) meets the checklist minimum; running states move; every decoded MP4
   frame matches its own PNG (not a neighbour), stays still through each
   stable stretch and carries the "合成演示 · 流程已压缩" label.

    python verify_final.py --version-dir docs/resumable-promo/versions/v02 \
        --frames <full-render> --subset <subset-render> --video <final.mp4> \
        --repo . --font C:/Windows/Fonts/msyh.ttc --sheet <transitions.png> \
        --out <verify-final.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

LABEL = "合成演示 · 流程已压缩"
# (label, frame a, frame b): two frames of one running state with a still camera.
MOTION = [
    ("运行中：Agent 决策第 3 次（执行流程取景静止）", 336, 348),
    ("运行中：Agent 决策第 6 次（两栏取景静止）", 988, 1000),
]
# Expected UI copy that is not data: each string must occur in this file.
UI_COPY = {
    "连接服务": "frontend/src/App.tsx",
    "服务已连接": "frontend/src/App.tsx",
    "等待补充信息": "frontend/src/lib/model.ts",
    "正在暂停": "frontend/src/lib/model.ts",
    "已暂停": "frontend/src/lib/model.ts",
    "调查进行中": "frontend/src/lib/model.ts",
    "已生成答案": "frontend/src/lib/model.ts",
    "补充本次调查需要的信息…": "frontend/src/components/InvestigationControls.tsx",
    "等待当前调用返回并保存结果后暂停。": "frontend/src/components/InvestigationControls.tsx",
    "进度连接已断开，正在重连。调查保留在服务端。": "frontend/src/components/InvestigationControls.tsx",
    "服务关闭前已保存检查点，可以继续原调查。": "frontend/src/components/InvestigationControls.tsx",
    "已提交的补充信息（": "frontend/src/components/InvestigationControls.tsx",
    "继续调查": "frontend/src/components/InvestigationControls.tsx",
    "取消调查": "frontend/src/components/InvestigationControls.tsx",
    "引用可追溯，结论仍需结合原文核对。": "frontend/src/App.tsx",
    "查看引用 ": "frontend/src/App.tsx",
    "本次运行": "frontend/src/App.tsx",
    "计数与 token 为运行诊断信息，不等于供应商账单。": "frontend/src/App.tsx",
    "调用 ID · ": "frontend/src/App.tsx",
    "证据 ": "frontend/src/App.tsx",
}
DEMO_SOURCE = "src/zhrag/service/investigation_demo.py"
# Cursor arrow (tip at the logged point) and click ripple, in output pixels,
# with room for the drop shadow.
CURSOR_MASK = (-6, -6, 30, 38)
RIPPLE_MASK = 40
# Product motion inside a hold that is the product's own, checked by eye:
# (label, first frame, last frame). The film opens on a fresh page load.
NATIVE = [("页面载入：流程节点状态图标淡入（node-status-enter 180ms），缩放控件「−」按钮由浅变深", 0, 5)]
# Captions fade in over a shot's first 12 frames and out over its last 7
# (scene.mjs overlayState). Their band is the union of the probed caption
# boxes grown by the drop shadow (0 10px 34px) and the fade-in lift, in output
# pixels: left, top, right, bottom.
CAPTION_FADE = (12, 7)
CAPTION_MARGIN = (40, 34, 40, 50)
# Stable reading the checklist requires, in frames, by shot.
MIN_STABLE = {"inspector-step": 45, "evidence": 45, "brand": 30}
# Lossy coding can disturb a few isolated pixels of a still picture for a
# frame; motion or flicker changes whole regions. A decoded stable stretch
# passes when no frame has more pixels over 15 levels off the stretch's first
# frame than one 8×8 transform block holds.
CODING_NOISE_PIXELS = 64


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def png(directory: Path, frame: int) -> Image.Image:
    with Image.open(directory / f"frame-{frame:06d}.png") as image:
        return image.convert("RGB")


def delta(a: Image.Image, b: Image.Image) -> Image.Image:
    """Largest per-channel difference of each pixel."""
    red, green, blue = ImageChops.difference(a, b).split()
    return ImageChops.lighter(ImageChops.lighter(red, green), blue)


def summary(diff: Image.Image) -> dict:
    histogram = diff.histogram()
    return {
        "changedPixels": sum(histogram[1:]),
        "changedPixelsOver15": sum(histogram[16:]),
        "maxDelta": max((level for level, count in enumerate(histogram) if count), default=0),
        "bbox": diff.getbbox(),
    }


def psnr(a: Image.Image, b: Image.Image) -> float:
    rms = ImageStat.Stat(ImageChops.difference(a, b)).rms
    mse = sum(value * value for value in rms) / len(rms)
    return math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)


def mask(diff: Image.Image, overlay: dict, frames: list[int]) -> Image.Image:
    """Blank the cursor and click ripple of the given frames."""
    draw = ImageDraw.Draw(diff)
    for frame in frames:
        state = overlay.get(str(frame)) or {}
        if state.get("cursor"):
            x, y, _ = state["cursor"]
            draw.rectangle((x + CURSOR_MASK[0], y + CURSOR_MASK[1], x + CURSOR_MASK[2], y + CURSOR_MASK[3]), fill=0)
        if state.get("ripple"):
            x, y = state["ripple"]
            draw.rectangle((x - RIPPLE_MASK, y - RIPPLE_MASK, x + RIPPLE_MASK, y + RIPPLE_MASK), fill=0)
    return diff


def to_output(box: dict, clip: dict, width: int) -> tuple[float, float, float, float]:
    scale = width / clip["width"]
    return (
        (box["x"] - clip["x"]) * scale,
        (box["y"] - clip["y"]) * scale,
        (box["x"] + box["width"] - clip["x"]) * scale,
        (box["y"] + box["height"] - clip["y"]) * scale,
    )


def runs(frames: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for frame in sorted(frames):
        if result and result[-1][1] == frame - 1:
            result[-1][1] = frame
        else:
            result.append([frame, frame])
    return result


# ------------------------------------------------------------ render log
def camera_moves(plan: dict) -> list[tuple[int, int]]:
    fps = float(plan["video"]["fps"])
    moves = []
    for shot in plan["shots"]:
        for cue in shot["data"].get("camera", []):
            frames = cue[2] if len(cue) > 2 else 0
            if frames:
                start = shot["startFrame"] + round(cue[0] * fps)
                moves.append((start, start + frames))
    return moves


def check_render(plan: dict, log: dict, manifest: dict) -> dict:
    report: dict = {}
    events = {event["id"]: event for event in plan["events"]}
    per_beat = plan["beats"][1]["frame"] - plan["beats"][0]["frame"]
    report["beatGrid"] = {
        "framesPerBeat": per_beat,
        "offBeat": [event["id"] for event in plan["events"] if event["frame"] % per_beat],
    }
    report["clicks"] = [
        {"id": click["id"], "frame": click["frame"], "planned": events[click["id"]]["frame"], "hit": click["hit"],
         "disabled": click["disabled"], "top": click.get("top")}
        for click in log["clicks"]
    ]
    planned = sorted(event["id"] for event in plan["events"] if event["action"] == "click")
    report["checks"] = [{key: check[key] for key in ("id", "frame", "selector", "text", "ok")} for check in log["checks"]]
    report["pageErrors"] = log["pageErrors"]
    report["console"] = log["console"]
    moves = camera_moves(plan)
    cut_frames = sorted(int(frame) for frame in log["edgeCuts"])
    still_cuts = [frame for frame in cut_frames if not any(a <= frame <= b for a, b in moves)]
    report["edgeCuts"] = {
        "framesWithCuts": len(cut_frames),
        "duringCameraMoves": len(cut_frames) - len(still_cuts),
        "stillFramesWithCuts": still_cuts,
        "cameraMoves": moves,
    }
    report["framing"] = [
        {"frame": item["frame"], "region": item["name"], "fittedY": round(item["fitted"]["y"], 2),
         "chosen": {key: round(item["chosen"][key], 2) for key in ("x", "y", "width")}, "grow": item.get("grow"),
         "fittedCuts": len(item.get("fittedCuts", [])), "cuts": item["cuts"]}
        for item in log["framing"]
    ]
    caret = {int(frame): state for frame, state in log["caret"].items()}
    blinks = []
    for frame in sorted(caret):
        visible = caret[frame]["visible"]
        if blinks and blinks[-1]["visible"] == visible and blinks[-1]["end"] == frame - 1:
            blinks[-1]["end"] = frame
        else:
            blinks.append({"start": frame, "end": frame, "visible": visible})
    report["caret"] = blinks
    report["cursorOverText"] = cursor_over_text(plan, log, manifest)
    report["ok"] = (
        not report["beatGrid"]["offBeat"]
        and sorted(click["id"] for click in log["clicks"]) == planned
        and all(c["hit"] and not c["disabled"] and c["frame"] == c["planned"] for c in report["clicks"])
        and all(check["ok"] for check in report["checks"])
        and not report["pageErrors"]
        and not report["console"]
        and not still_cuts
        and report["cursorOverText"]["ok"]
    )
    return report


def cursor_over_text(plan: dict, log: dict, manifest: dict) -> dict:
    """Every frame where the arrow covers text is on its click target or in
    motion, and no hold has it over body text (fields and icons are reviewed)."""
    width = manifest["size"]["width"]
    overlay, cameras = log["overlay"], manifest["camera"]

    def page(frame: int) -> tuple[float, float] | None:
        state = overlay.get(str(frame))
        if not state or not state["cursor"]:
            return None
        clip = cameras[str(frame)]
        x, y, _ = state["cursor"]
        return clip["x"] + x * clip["width"] / width, clip["y"] + y * clip["width"] / width

    holds = [(hold["startFrame"], hold["endFrameExclusive"]) for hold in plan["holds"]]
    covered = {int(frame): labels for frame, labels in log["cursorInk"].items()}
    result = []
    for start, end in runs(list(covered)):
        for frame in range(start, end + 1):
            here = page(frame)
            on_target = any(
                click["frame"] - 7 <= frame <= click["frame"] + 7
                and here and math.dist(here, (click["x"], click["y"])) <= 1.5
                for click in log["clicks"]
            )
            moving = any(
                (other := page(frame + step)) and here and math.dist(here, other) > 0.5 for step in (-1, 1)
            )
            kind = "clickTarget" if on_target else "moving" if moving else "resting"
            in_hold = any(a <= frame < b for a, b in holds)
            if result and result[-1]["end"] == frame - 1 and result[-1]["kind"] == kind and result[-1]["inHold"] == in_hold:
                result[-1]["end"] = frame
                result[-1]["text"] = sorted(set(result[-1]["text"]) | set(covered[frame]))
            else:
                result.append({"start": frame, "end": frame, "kind": kind, "inHold": in_hold, "text": sorted(set(covered[frame]))})
    bad = [item for item in result if item["kind"] == "resting"
           or (item["inHold"] and any(label.startswith("text:") for label in item["text"]))]
    review = [item for item in result if item["inHold"] and item not in bad]
    return {"runs": result, "flagged": bad, "reviewInHold": review, "ok": not bad}


# ------------------------------------------------------------ copy
def read_calls(final: dict) -> list[str]:
    """Invocation ids of the recorded read_passage calls, in order."""
    ids: list[str] = []
    for item in final["progress"]:
        if item["action"] == "read_passage" and item["invocation_id"] not in ids:
            ids.append(item["invocation_id"])
    return ids


def expected_copy(plan: dict, recording: dict, brief: dict) -> tuple[dict, dict, int]:
    snapshots = recording["snapshots"]
    final = snapshots[-1]["value"]
    response = final["response"]
    sources = {source["citation_id"]: source for source in response["sources"]}
    waiting = next(item["value"] for item in snapshots if item["value"]["status"] == "waiting_input")
    target = next(event["target"] for event in plan["events"] if event["id"] == "pick-source")
    clicked = int(target.rsplit("-", 1)[1])
    shown = sources[clicked]
    answer = [block["text"] for block in response["blocks"]]
    citations = [f"查看引用 {cid}" for block in response["blocks"] for cid in block["citations"]]
    reads = read_calls(final)
    history = [f"已提交的补充信息（{len(final['supplements'])}）"]
    folded = [waiting["clarification"], f"你：{recording['supplement']}"]
    visible = {
        171: {"mode": ["连接服务"], "connection": ["服务已连接"]},
        240: {"query": [recording["question"]]},
        405: {"status": ["等待补充信息"], "question": [waiting["clarification"]], "placeholder": ["补充本次调查需要的信息…"]},
        480: {"supplement": [recording["supplement"]]},
        585: {"status": ["正在暂停"], "controls": ["等待当前调用返回并保存结果后暂停。"], "history": history},
        660: {"status": ["已暂停"], "history": history, "buttons": ["继续调查", "取消调查"]},
        819: {"status": ["调查进行中"], "controls": ["进度连接已断开，正在重连。调查保留在服务端。"], "history": history},
        903: {"status": ["已暂停"], "controls": ["服务关闭前已保存检查点，可以继续原调查。"], "history": history},
        1059: {"status": ["已生成答案"], "answer": answer, "citations": citations},
        # Step inspector: the latest read, then the earlier one; ids and
        # counts come from the recorded progress (two reads in this run).
        1176: {"stepTitle": ["读取证据 · 第 2 次"], "position": ["第 2 / 2 次"], "stepId": [f"调用 ID · {reads[1]}"]},
        1254: {"stepTitle": ["读取证据 · 第 1 次"], "position": ["第 1 / 2 次"], "stepId": [f"调用 ID · {reads[0]}"]},
        1365: {
            "pressed": [f"{clicked}{shown['title']}"],
            "evidenceKicker": [f"证据 {clicked:02d}"],
            "evidenceTitle": [shown["title"]],
            "evidenceText": [shown["text"]],
        },
        1443: {"runTitle": ["本次运行"], "runNote": ["计数与 token 为运行诊断信息，不等于供应商账单。"]},
        1527: {"brandName": [brief["product"]["name"]], "tagline": [brief["tagline"]["candidates"][0]]},
    }
    # Collapsed supplement history: present in the DOM, not painted.
    hidden = {585: {"controls": folded}, 819: {"controls": folded}, 903: {"controls": folded}}
    for frame, names in visible.items():
        shot = next(s for s in plan["shots"] if s["startFrame"] <= frame < s["endFrameExclusive"])
        names["label"] = [LABEL]
        if shot["data"].get("caption"):
            names["caption"] = [shot["data"]["caption"]]
        else:
            hidden.setdefault(frame, {})["caption"] = [""]
    return visible, hidden, clicked


def check_copy(plan: dict, log: dict, manifest: dict, recording: dict, brief: dict, repo: Path) -> dict:
    visible, hidden, clicked = expected_copy(plan, recording, brief)
    width = manifest["size"]["width"]
    report: dict = {"frames": {}, "provenance": {}, "clickedCitation": clicked}
    ok = True
    for frame, names in visible.items():
        probes = log["probes"].get(str(frame))
        clip = manifest["camera"][str(frame)]
        if probes is None:
            report["frames"][str(frame)] = {"error": "no probe"}
            ok = False
            continue
        entry = {}
        for name in sorted(set(names) | set(hidden.get(frame, {}))):
            elements = probes.get(name, [])
            painted = [item for item in elements if item["visible"]]
            unpainted = [item for item in elements if not item["visible"]]
            outside = []
            for item in painted:
                x0, y0, x1, y1 = to_output(item["box"], clip, width)
                if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > manifest["size"]["height"] + 1:
                    outside.append(item["text"])
            result = {
                "shown": [item["text"] for item in painted],
                "expected": names.get(name, []),
                "notPainted": [item["text"] for item in unpainted],
                "expectedNotPainted": hidden.get(frame, {}).get(name, []),
                "outsideCamera": outside,
            }
            result["ok"] = (
                result["shown"] == result["expected"]
                and result["notPainted"] == result["expectedNotPainted"]
                and not outside
            )
            ok &= result["ok"]
            entry[name] = result
        report["frames"][str(frame)] = entry
    # Copy expected from the frontend must exist there verbatim.
    for text, file in UI_COPY.items():
        found = text in (repo / file).read_text(encoding="utf-8")
        report["provenance"][text] = {"file": file, "found": found}
        ok &= found
    # The recording replays the synthetic backend; its documents, question and
    # answer text are the backend's own literals (adjacent literals joined).
    demo = re.sub(r'"\s*\n\s*"', "", (repo / DEMO_SOURCE).read_text(encoding="utf-8"))
    final = recording["snapshots"][-1]["value"]["response"]
    literals = [source["title"] for source in final["sources"]] + [source["text"] for source in final["sources"]]
    literals += [block["text"] for block in final["blocks"]]
    literals.append(next(s["value"]["clarification"] for s in recording["snapshots"] if s["value"]["status"] == "waiting_input"))
    report["demoSource"] = {"file": DEMO_SOURCE, "missing": [text for text in literals if text not in demo]}
    ok &= not report["demoSource"]["missing"]
    report["ok"] = ok
    return report


# ------------------------------------------------------------ pixels
def caption_band(log: dict, manifest: dict) -> tuple[int, int, int, int]:
    width = manifest["size"]["width"]
    boxes = [to_output(item["box"], manifest["camera"][frame], width)
             for frame, probes in log["probes"].items() for item in probes["caption"] if item["text"]]
    return (math.floor(min(box[0] for box in boxes)) - CAPTION_MARGIN[0],
            math.floor(min(box[1] for box in boxes)) - CAPTION_MARGIN[1],
            math.ceil(max(box[2] for box in boxes)) + CAPTION_MARGIN[2],
            math.ceil(max(box[3] for box in boxes)) + CAPTION_MARGIN[3])


def caption_fades(plan: dict) -> set[int]:
    fades: set[int] = set()
    for shot in plan["shots"]:
        if shot["data"].get("caption"):
            a, b = shot["startFrame"], shot["endFrameExclusive"]
            fades |= set(range(a, a + CAPTION_FADE[0])) | set(range(b - CAPTION_FADE[1], b))
    return fades


def check_pngs(plan: dict, log: dict, full: Path, subset: Path) -> dict:
    report: dict = {"determinism": [], "motion": [], "holds": []}
    manifest = read_json(subset / "render-manifest.json")
    for frame in manifest["frames"]:
        result = summary(delta(png(full, frame), png(subset, frame)))
        result["frame"] = frame
        # Anti-aliasing may differ by a level or two between browser processes.
        result["ok"] = result["changedPixelsOver15"] == 0
        report["determinism"].append(result)
    full_manifest = read_json(full / "render-manifest.json")
    cameras = full_manifest["camera"]
    for label, a, b in MOTION:
        result = summary(delta(png(full, a), png(full, b)))
        result.update(label=label, frames=[a, b], sameCamera=cameras[str(a)] == cameras[str(b)])
        result["ok"] = result["sameCamera"] and result["changedPixelsOver15"] > 0
        report["motion"].append(result)
    # Each hold frame against the hold's middle, cursor and ripple blanked: it
    # is still, or only its caption band differs while that caption fades, or
    # it is documented native motion. The stable stretch is the run of still
    # frames around the middle.
    fps = float(plan["video"]["fps"])
    band = caption_band(log, full_manifest)
    fades = caption_fades(plan)
    native = {frame for _, a, b in NATIVE for frame in range(a, b + 1)}
    report["captionBand"] = band
    for hold in plan["holds"]:
        a, b = hold["startFrame"], hold["endFrameExclusive"]
        middle = (a + b) // 2
        reference = png(full, middle)
        kinds: dict[int, str] = {}
        worst = 0
        for frame in range(a, b):
            diff = mask(delta(png(full, frame), reference), log["overlay"], [middle, frame])
            if summary(diff)["changedPixelsOver15"] == 0:
                kinds[frame] = "still"
                worst = max(worst, summary(diff)["maxDelta"])
                continue
            ImageDraw.Draw(diff).rectangle(band, fill=0)
            if frame in fades and summary(diff)["changedPixelsOver15"] == 0:
                kinds[frame] = "captionFade"
            elif frame in native:
                kinds[frame] = "native"
            else:
                kinds[frame] = "unexpected"
        start = end = middle
        while start - 1 >= a and kinds[start - 1] == "still":
            start -= 1
        while end + 1 < b and kinds[end + 1] == "still":
            end += 1
        of = {kind: runs([frame for frame, value in kinds.items() if value == kind])
              for kind in ("captionFade", "native", "unexpected")}
        stable = end - start + 1
        minimum = MIN_STABLE.get(hold["shot"])
        report["holds"].append({
            "shot": hold["shot"], "label": hold["label"], "frames": [a, b - 1], "seconds": round((b - a) / fps, 2),
            "stable": [start, end], "stableFrames": stable, "stableSeconds": round(stable / fps, 2),
            "stableMaxDelta": worst, "minStableFrames": minimum, **of,
            "ok": not of["unexpected"] and (minimum is None or stable >= minimum),
        })
    report["ok"] = all(item["ok"] for key in ("determinism", "motion", "holds") for item in report[key])
    return report


def label_template(log: dict, manifest: dict, full: Path) -> tuple[tuple[int, int, int, int], list, list]:
    """Crop box of the label (the same output pixels in every probe) and its text pixels."""
    width = manifest["size"]["width"]
    boxes = []
    for frame, probes in log["probes"].items():
        for item in probes["label"]:
            boxes.append(to_output(item["box"], manifest["camera"][frame], width))
    spread = max(max(abs(box[i] - boxes[0][i]) for box in boxes) for i in range(4))
    crop = (math.floor(min(b[0] for b in boxes)) - 3, math.floor(min(b[1] for b in boxes)) - 3,
            math.ceil(max(b[2] for b in boxes)) + 3, math.ceil(max(b[3] for b in boxes)) + 3)
    reference = png(full, int(min(log["probes"], key=int))).crop(crop).convert("L")
    text = [index for index, value in enumerate(reference.tobytes()) if value < 120]
    return crop, text, [round(spread, 2)]


def check_video(plan: dict, log: dict, manifest: dict, full: Path, video: Path, digest: str, font: Path, sheet: Path,
                stable: list[tuple[int, int, str]]) -> dict:
    width, height = manifest["size"]["width"], manifest["size"]["height"]
    total = plan["video"]["totalFrames"]
    crop, text_pixels, spread = label_template(log, manifest, full)
    # Through each stable stretch found on the PNGs, decoded frames stay within
    # coding noise of the stretch's first decoded frame (cursor blanked).
    hold_reference: dict[int, Image.Image] = {}
    hold_worst = {a: {"maxDelta": 0, "framesOver15": []} for a, _, _ in stable}
    motion_frames = {frame for _, a, b in MOTION for frame in (a, b)}
    keep: dict[int, Image.Image] = {}
    moves = camera_moves(plan)
    brand = next(s for s in plan["shots"] if s["id"] == "brand")["startFrame"]
    sheet_frames = sorted({a + (b - a) // 2 for a, b in moves} | {brand + 4, brand + 10, brand + 18, brand + 30})
    thumbs: dict[int, Image.Image] = {}
    psnrs: list[tuple[float, int]] = []
    misaligned = []
    label_failures = []
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = width * height * 3
    previous = None
    current = png(full, 0)
    frame = 0
    try:
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size or frame >= total:
                raise ValueError(f"unexpected decoded data at frame {frame}")
            decoded = Image.frombytes("RGB", (width, height), data)
            following = png(full, frame + 1) if frame + 1 < total else None
            same = psnr(decoded, current)
            psnrs.append((same, frame))
            # Matches its own PNG better than either neighbour that visibly differs.
            for neighbour in (previous, following):
                if neighbour is not None and psnr(neighbour, current) < 45 and psnr(decoded, neighbour) >= same:
                    misaligned.append(frame)
            # The label: its text pixels stay dark in every frame.
            luma = decoded.crop(crop).convert("L").tobytes()
            dark = sum(luma[index] < 160 for index in text_pixels) / len(text_pixels)
            if dark < 0.95:
                label_failures.append({"frame": frame, "darkTextRatio": round(dark, 3)})
            for a, b, _ in stable:
                if frame == a:
                    hold_reference[a] = decoded
                elif a < frame <= b:
                    result = summary(mask(delta(decoded, hold_reference[a]), log["overlay"], [a, frame]))
                    worst = hold_worst[a]
                    worst["maxDelta"] = max(worst["maxDelta"], result["maxDelta"])
                    if result["changedPixelsOver15"]:
                        worst["framesOver15"].append([frame, result["changedPixelsOver15"]])
            if frame in motion_frames:
                keep[frame] = decoded
            if frame in sheet_frames:
                thumbs[frame] = decoded.resize((480, 270), Image.LANCZOS)
            previous, current = current, following
            frame += 1
    finally:
        process.stdout.close()
        if process.wait():
            raise RuntimeError("ffmpeg could not decode the video")
    motion = []
    for label, a, b in MOTION:
        result = summary(delta(keep[a], keep[b]))
        result.update(label=label, frames=[a, b], ok=result["changedPixelsOver15"] > 0)
        motion.append(result)
    still = []
    for a, b, shot in stable:
        worst = hold_worst[a]
        largest = max((count for _, count in worst["framesOver15"]), default=0)
        still.append({"shot": shot, "stable": [a, b], **worst, "largestOver15": largest,
                      "ok": largest <= CODING_NOISE_PIXELS})
    values = [value for value, _ in psnrs]
    finite = [value for value in values if math.isfinite(value)]
    write_sheet(plan, thumbs, font, sheet, digest)
    report = {
        "decodedFrames": frame,
        "psnrAgainstOwnPng": {"min": round(min(finite), 2), "mean": round(sum(finite) / len(finite), 2),
                              "minFrame": min(psnrs)[1], "identicalFrames": len(values) - len(finite)},
        "misalignedFrames": sorted(set(misaligned)),
        "label": {"crop": crop, "probeSpreadPx": spread, "textPixels": len(text_pixels), "failures": label_failures},
        "motion": motion,
        "holds": still,
        "sheet": {"path": str(sheet), "frames": sheet_frames},
    }
    report["ok"] = (
        frame == total
        and not report["misalignedFrames"]
        and not label_failures
        and all(item["ok"] for item in motion + still)
    )
    return report


def write_sheet(plan: dict, thumbs: dict, font_path: Path, path: Path, digest: str) -> None:
    font = ImageFont.truetype(str(font_path), 17)
    frames = sorted(thumbs)
    columns = 4
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (16 + columns * 496, 52 + rows * 312), "#f5f3ef")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), f"运镜与收尾转场（解码自最终 MP4 · SHA-256 {digest[:16]}）", font=font, fill="#222222")
    fps = float(plan["video"]["fps"])
    for index, frame in enumerate(frames):
        x, y = 16 + (index % columns) * 496, 48 + (index // columns) * 312
        sheet.paste(thumbs[frame], (x, y))
        shot = next(s for s in plan["shots"] if s["startFrame"] <= frame < s["endFrameExclusive"])
        draw.text((x, y + 276), f"{frame / fps:.2f}s · F{frame} · {shot['label']}", font=font, fill="#222222")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.out, args.sheet):
        if output.exists():
            raise SystemExit(f"output exists; use a new file: {output}")
    version = args.version_dir
    plan = read_json(version / "plan.json")
    recording = read_json(version / "source" / "api-recording.json")
    brief = read_json(version.parent.parent / "brief.json")
    log = read_json(args.frames / "render-log.json")
    manifest = read_json(args.frames / "render-manifest.json")
    if manifest["planSha256"] != plan["planSha256"] or not manifest["complete"]:
        raise SystemExit("render does not match plan.json or is incomplete")
    digest = hashlib.sha256(args.video.read_bytes()).hexdigest()
    pngs = check_pngs(plan, log, args.frames, args.subset)
    stable = [(*hold["stable"], hold["shot"]) for hold in pngs["holds"]]
    report = {
        "planSha256": plan["planSha256"],
        "videoSha256": digest,
        "render": check_render(plan, log, manifest),
        "copy": check_copy(plan, log, manifest, recording, brief, args.repo),
        "pngs": pngs,
        "video": check_video(plan, log, manifest, args.frames, args.video, digest, args.font, args.sheet, stable),
    }
    report["passed"] = all(report[key]["ok"] for key in ("render", "copy", "pngs", "video"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        **{key: report[key]["ok"] for key in ("render", "copy", "pngs", "video")},
        "cursorFlagged": report["render"]["cursorOverText"]["flagged"],
        "cursorReviewInHold": report["render"]["cursorOverText"]["reviewInHold"],
        "stillEdgeCuts": report["render"]["edgeCuts"]["stillFramesWithCuts"],
        "copyFailures": [f"{frame}:{name}" for frame, names in report["copy"]["frames"].items()
                         for name, item in names.items() if not item.get("ok", False)],
        "determinismMaxDelta": max(item["maxDelta"] for item in report["pngs"]["determinism"]),
        "psnr": report["video"]["psnrAgainstOwnPng"],
        "misaligned": report["video"]["misalignedFrames"][:10],
        "labelFailures": report["video"]["label"]["failures"][:5],
        "holdsPng": [{key: h[key] for key in ("shot", "frames", "stable", "stableFrames", "captionFade", "native",
                                              "unexpected", "ok")} for h in report["pngs"]["holds"]],
        "holdsMp4": [(h["shot"], h["stable"], h["framesOver15"], h["maxDelta"]) for h in report["video"]["holds"]],
        "motion": [(m["frames"], m["changedPixelsOver15"]) for m in report["pngs"]["motion"] + report["video"]["motion"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
