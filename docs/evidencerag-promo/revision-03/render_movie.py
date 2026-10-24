"""Render the confirmed native-scale revision without network or model calls."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

from zhrag.io_utils import read_bytes, read_json, read_text, write_json, write_text
from media_data import node_data
from synthesize_audio import synthesize

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[1]
FINAL = HERE / "final-motion"
FRAMEDIR = HERE / "rendered-frames-motion"
SAMPLE = sorted(set([0, 10, 30, 37, 44, 45, 49, 55, 59, 60, 64, 69, 74, 75,
                    89, 90, 104, 105, 119, 120, 128, 134, 135, 136, 139, 143,
                    149, 150, 154, 164, 165, 171, 179, 180, 195, 203, 209, 210,
                    211, 215, 219, 224, 225, 255, 299, 300, 303, 307, 310, 314,
                    315, 320, 329, 330, 345, 359]))


def build_html() -> str:
    native = read_json(HERE / "reference/native.json")
    font = "data:font/woff2;base64," + base64.b64encode(read_bytes(
        REPO / "frontend/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"
    )).decode("ascii")
    demo = node_data()
    demo["glacierSvg"] = read_text(REPO / "frontend/src/assets/glacier.svg")
    graph = subprocess.run(["node", str(HERE / "demo_graph.mjs")],
        input=json.dumps({"demoTs": read_text(REPO / "frontend/src/lib/demo.ts")}, ensure_ascii=False),
        text=True, encoding="utf-8", capture_output=True, check=True)
    demo["graphStates"] = json.loads(graph.stdout)
    data = {"snapshots": native["snapshots"], "demo": demo,
            "css": native["css"].replace('/workbench/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2', font),
            "overrides": read_text(HERE / "native-overrides.css")}
    html = (read_text(HERE / "frame.html").replace("__INTER_FONT__", font)
        .replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
        .replace("/*__SCRIPT__*/", read_text(HERE / "storyboard.js") + "\n" + read_text(HERE / "graph_motion.js") + "\n" + read_text(HERE / "motion.js"))
        .replace("浏览器原比例分镜", "浏览器原比例动画 · 空格播放"))
    assert "\ufffd" not in html, "Unexpected replacement character in HTML source"
    write_text(HERE / "movie.html", html)
    return html


def render(frames: list[int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["node", str(HERE / "render_video.mjs")], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", cwd=REPO)
    proc.stdin.write(json.dumps({"html": build_html(), "frames": frames, "outputDir": str(output)}, ensure_ascii=False))
    proc.stdin.close()
    report = None
    for line in proc.stdout:
        message = json.loads(line)
        if message.get("done"):
            report = message
        else:
            print(f"Rendered {message['progress']}/{message['total']} (frame {message['frame']})", flush=True)
    stderr = proc.stderr.read()
    code = proc.wait()
    if stderr:
        print(stderr, file=sys.stderr)
    if code or report is None:
        raise RuntimeError(f"Chromium renderer failed: {code}")
    write_json(output / "render-report.json", report)
    assert not report["errors"], report["errors"]
    assert not report["bounds"], report["bounds"][:3]
    assert all(s["labelCount"] for s in report["samples"]), "Synthetic label disappeared"
    assert all(not s["hit"] or s["hit"]["inside"] for s in report["samples"]), "Cursor missed"
    assert all(not s["graph"] or s["graph"]["valid"] for s in report["samples"]), "Graph invalid"
    assert all(not s["replacementCharacters"] for s in report["samples"]), "Chinese text corrupted in transport"
    expected_answers = ["核对版本、配置与兼容性[1]", "验证升级演练与回退条件[2]"]
    assert all(s["answerTexts"] == expected_answers for s in report["samples"] if s["frame"] >= 150), "Answer text changed"
    return report


def encode() -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-framerate", "30",
        "-start_number", "0", "-i", str(FRAMEDIR / "frame-%04d.png"), "-i", str(FINAL / "soundtrack.wav"),
        "-map", "0:v:0", "-map", "1:a:0", "-frames:v", "360", "-t", "12",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
        "-color_range", "tv", "-pix_fmt", "yuv420p", "-r", "30", "-fps_mode", "cfr",
        "-video_track_timescale", "15360", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-metadata", "title=EvidenceRAG — 让答案有据可查",
        "-metadata", "comment=原创合成演示；流程已压缩；原生工作台与检查器。",
        str(FINAL / "EvidenceRAG-12s.mp4")]
    subprocess.run(cmd, check=True)
    write_json(FINAL / "encode-command.json", cmd)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    FINAL.mkdir(exist_ok=True)
    if args.samples:
        report = render(sorted(set(SAMPLE + [65, 70, 80, 84, 85, 95, 100, 110, 115, 125, 130, 140, 145, 155, 160])), HERE / "graph-motion-checks")
        print(json.dumps({"rendered": report["rendered"], "errors": report["errors"], "bounds": report["bounds"]}))
        return
    if not args.encode_only:
        result = render(list(range(360)), FRAMEDIR)
        write_json(FINAL / "render-report.json", result)
    else:
        write_json(FINAL / "render-report.json", read_json(FRAMEDIR / "render-report.json"))
    synthesize(FINAL)
    encode()
    print(str(FINAL / "EvidenceRAG-12s.mp4"), flush=True)


if __name__ == "__main__":
    main()
