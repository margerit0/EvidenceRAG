"""Numeric checks for the storyboard preview (Python 3.10+, Pillow).

1. Determinism: frames from an independent render that only captured a
   subset (fresh browser, replayed from frame 0) must match the full render.
2. Continuous motion: two frames inside one running state, with a still
   camera, must differ only in small native-animation regions.
3. Stopped motion: two frames in a paused/waiting state must be identical.
4. Clicks and expected UI results recorded by the renderer all passed.

    python verify_preview.py --full <frames-a> --subset <frames-b> --out verify.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

# (label, frame a, frame b) pairs chosen inside one backend state each.
MOTION = [
    ("运行中：Agent 决策第 3 次（执行流程取景静止）", 336, 348),
    ("运行中：Agent 决策第 6 次（两栏取景静止）", 988, 1000),
]
STILL = [
    ("等待补充信息（阅读停留）", 400, 425),
    ("已暂停（阅读停留）", 640, 680),
    ("服务重启后已暂停（阅读停留）", 885, 920),
    ("步骤检查器（阅读停留）", 1140, 1189),
    ("同类调用第 1 / 2 次（阅读停留）", 1236, 1264),
    ("证据原文（阅读停留）", 1340, 1384),
    ("运行概览（阅读停留）", 1431, 1444),
    ("品牌（稳定停留）", 1500, 1544),
]


def frame(directory: Path, number: int) -> Image.Image:
    return Image.open(directory / f"frame-{number:06d}.png").convert("RGB")


def difference(a: Image.Image, b: Image.Image) -> dict:
    diff = ImageChops.difference(a, b).convert("L")
    histogram = diff.histogram()
    changed = sum(histogram[1:])
    strong = sum(histogram[16:])
    return {"changedPixels": changed, "changedPixelsOver15": strong, "maxDelta": max(
        (i for i, count in enumerate(histogram) if count), default=0), "bbox": diff.getbbox()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"determinism": [], "motion": [], "still": []}
    subset = json.loads((args.subset / "render-manifest.json").read_text(encoding="utf-8"))["frames"]
    for number in subset:
        result = difference(frame(args.full, number), frame(args.subset, number))
        result["frame"] = number
        # Anti-aliasing may differ by a level between processes; state may not.
        result["ok"] = result["changedPixelsOver15"] == 0
        report["determinism"].append(result)
    for label, a, b in MOTION:
        result = difference(frame(args.full, a), frame(args.full, b))
        result.update(label=label, frames=[a, b], ok=result["changedPixelsOver15"] > 0)
        report["motion"].append(result)
    for label, a, b in STILL:
        result = difference(frame(args.full, a), frame(args.full, b))
        result.update(label=label, frames=[a, b], ok=result["changedPixelsOver15"] == 0)
        report["still"].append(result)
    log = json.loads((args.full / "render-log.json").read_text(encoding="utf-8"))
    report["clicks"] = [{k: c[k] for k in ("id", "frame", "hit", "top", "disabled")} for c in log["clicks"]]
    report["checks"] = [{k: c[k] for k in ("id", "frame", "selector", "text", "ok")} for c in log["checks"]]
    report["pageErrors"] = log["pageErrors"]
    report["console"] = log["console"]
    report["passed"] = (
        all(item["ok"] for key in ("determinism", "motion", "still") for item in report[key])
        and all(c["hit"] and not c["disabled"] for c in report["clicks"])
        and all(c["ok"] for c in report["checks"])
        and not report["pageErrors"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "determinism": [(d["frame"], d["changedPixelsOver15"]) for d in report["determinism"]],
                      "motion": [(m["frames"], m["changedPixelsOver15"], m["bbox"]) for m in report["motion"]],
                      "still": [(s["frames"], s["changedPixelsOver15"]) for s in report["still"]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
