"""Verify the encoded MP4 and produce the five required decoded check frames."""

from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from zhrag.io_utils import read_bytes, read_json, read_text, write_json, write_text

HERE = Path(__file__).resolve().parent
FINAL = HERE / "final-motion"
VIDEO = FINAL / "EvidenceRAG-12s.mp4"


def run(args: list[str]) -> bytes:
    p = subprocess.run(args, capture_output=True, check=True)
    return p.stdout


def decode_audio(path: Path, rate: int = 48000) -> np.ndarray:
    return np.frombuffer(
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-t",
                "12",
                "-vn",
                "-ar",
                str(rate),
                "-ac",
                "2",
                "-f",
                "f32le",
                "pipe:1",
            ]
        ),
        dtype="<f4",
    ).reshape(-1, 2)


def sheet(html: str, path: Path) -> None:
    subprocess.run(
        ["node", str(HERE / "render_sheet.mjs")],
        input=json.dumps({"html": html, "output": str(path)}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    probe = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,sample_rate,channels,duration,start_time,color_space,color_transfer,color_primaries",
                "-of",
                "json",
                str(VIDEO),
            ]
        )
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (
        v["width"],
        v["height"],
        v["codec_name"],
        v["avg_frame_rate"],
        int(v["nb_read_frames"]),
    ) == (1920, 1080, "h264", "30/1", 360)
    assert a["codec_name"] == "aac" and a["sample_rate"] == "48000" and a["channels"] == 2
    assert (
        float(probe["format"]["duration"]) == 12
        and float(a["duration"]) == 12
        and float(v["duration"]) == 12
    )
    assert float(a["start_time"]) == 0 and float(v["start_time"]) == 0
    timing = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(VIDEO),
            ]
        )
    )
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in timing["frames"]])
    assert len(pts) == 360 and np.max(np.abs(pts - np.arange(360) / 30)) < 1e-5
    decoded = decode_audio(VIDEO)
    reference = decode_audio(FINAL / "soundtrack.wav")
    oversampled = decode_audio(VIDEO, 192000)
    mono = decoded.mean(axis=1)
    ref = reference.mean(axis=1)
    nfft = 1 << (len(ref) * 2 - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(mono, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lag = int(np.argmax(corr))
    lag = lag if lag < nfft // 2 else lag - nfft
    peak = float(np.max(np.abs(decoded)))
    truepeak = float(np.max(np.abs(oversampled)))
    assert abs(lag) <= 16 and truepeak < 10 ** (-1 / 20)
    render = read_json(FINAL / "render-report.json")
    from verify_graph_motion import verify_graph_motion

    graph_motion = verify_graph_motion(VIDEO, render)
    hashes = {x["frame"]: x["hash"] for x in render["hashes"]}
    assert len(set(hashes[i] for i in range(225, 300))) == 1
    assert len(set(hashes[i] for i in range(330, 360))) == 1
    assert not render["errors"] and not render["bounds"]
    assert all(not s["replacementCharacters"] for s in render["samples"])
    assert all(s["answerTexts"] == ["核对版本、配置与兼容性[1]", "验证升级演练与回退条件[2]"] for s in render["samples"] if s["frame"] >= 150)
    assert all(not s["graph"] or s["graph"]["valid"] for s in render["samples"])
    assert all(s["columnWidths"] == [484, 650, 500] for s in render["samples"] if 149 <= s["frame"] < 300)
    assert all("步骤" in s["inspectorTab"] for s in render["samples"] if 149 <= s["frame"] < 210)
    assert all("证据" in s["inspectorTab"] for s in render["samples"] if 210 <= s["frame"] < 300)
    assert all(s["labelCount"] >= 1 for s in render["samples"])
    from media_data import node_data

    demo = node_data()
    expected = demo["sources"][1]["text"].split("\n\n")[1]
    assert all(s["sourceText"] == expected for s in render["samples"] if 225 <= s["frame"] < 300)
    clicks = [s for s in render["samples"] if s["hit"]]
    assert [s["frame"] for s in clicks] == [45, 135, 210] and all(s["hit"]["inside"] for s in clicks)
    score = read_json(FINAL / "audio-score.json")
    beats = [e for e in score["events"] if e["kind"] == "beat"]
    assert [e["sample"] for e in beats] == list(range(0, 576000, 24000))
    audio_clicks = [e for e in score["events"] if e["kind"] == "pointer-click"]
    assert [e["frame"] for e in audio_clicks] == [45, 135, 210]
    raw = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(VIDEO),
            "-an",
            "-vf",
            "scale=320:180,format=gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    thumb = np.frombuffer(raw, dtype=np.uint8).reshape(360, 180, 320)
    delta = np.abs(np.diff(thumb.astype(np.float32), axis=0)).mean(axis=(1, 2))
    assert np.min(thumb.mean(axis=(1, 2))) > 180
    extraction = "select=" + "+".join(rf"eq(n\,{f})" for f in [45, 105, 195, 255, 345])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(VIDEO),
            "-vf",
            extraction,
            "-fps_mode",
            "vfr",
            "-frames:v",
            "5",
            "-start_number",
            "1",
            str(FINAL / "check-%02d.png"),
        ],
        check=True,
    )
    loud = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(VIDEO),
            "-vn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    write_text(FINAL / "audio-loudness.txt", loud.stderr)
    loudness = re.findall(r"\bI:\s+([-\d.]+) LUFS", loud.stderr)
    report = {
        "file": str(VIDEO),
        "sha256": hashlib.sha256(read_bytes(VIDEO)).hexdigest(),
        "probe": probe,
        "pts_max_error_seconds": float(np.max(np.abs(pts - np.arange(360) / 30))),
        "audio": {
            "peak_dbfs": float(20 * np.log10(peak)),
            "true_peak_dbfs_4x": float(20 * np.log10(truepeak)),
            "clipped_samples": int(np.count_nonzero(np.abs(decoded) >= 1)),
            "samples": len(decoded),
            "alignment_lag_samples": lag,
            "alignment_lag_ms": lag / 48,
            "integrated_lufs": float(loudness[-1]) if loudness else None,
            "tail_100ms_rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(decoded[-4800:] ** 2)))),
        },
        "holds": {
            "source_frames": [225, 299],
            "source_seconds": 2.5,
            "brand_frames": [330, 359],
            "brand_seconds": 1.0,
        },
        "clicks": [{"frame": s["frame"], **s["hit"]} for s in clicks],
        "synthetic_label_every_frame": True,
        "flow_order": ["搜索文档", "读取证据", "补充搜索", "读取证据"],
        "source2_matches_demo": True,
        "native_graph_motion": graph_motion,
        "frame_delta": {
            "max_mean_absolute_gray_change": float(delta.max()),
            "largest_changes": [
                {"frame": int(i + 1), "change": float(delta[i])}
                for i in np.argsort(delta)[-10:][::-1]
            ],
        },
        "frame_times": [1.5, 3.5, 6.5, 8.5, 11.5],
        "layout_unexpected_issues": 0,
        "status": "encoded-file-checks-passed; playback review follows",
    }
    write_json(FINAL / "technical-checks.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ["audio", "holds", "frame_delta"]}, ensure_ascii=False, indent=2
        )
    )
    html = read_text(HERE / "overview.html")
    font = (
        HERE.parents[2]
        / "frontend/node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2"
    )
    html = html.replace(
        "__INTER_FONT__",
        "data:font/woff2;base64," + base64.b64encode(read_bytes(font)).decode("ascii"),
    )
    for i in range(1, 6):
        html = html.replace(
            f"__FRAME_{i}__",
            "data:image/png;base64,"
            + base64.b64encode(read_bytes(FINAL / f"check-{i:02d}.png")).decode("ascii"),
        )
    html = (
        html.replace("EvidenceRAG / 五镜分镜预览", "EvidenceRAG / 成片五帧检查")
        .replace(
            "沿用冰川流线界面 · 原创合成演示资料 · 真实 HTML / CSS / SVG 渲染",
            "从最终 MP4 解码抽取 · 原创合成演示资料 · 时间与帧号逐项核对",
        )
        .replace("第一阶段 · 待确认", "最终 MP4 · 成片检查")
    )
    a0 = html.index('<aside class="tile notes">')
    a1 = html.index("</aside>", a0) + len("</aside>")
    notes = f"""<aside class="tile notes"><h2>成片参数与检查</h2><div class="spec">1920 × 1080 · 30 fps · 12 s / 360 帧</div><h3>H.264 视频 / AAC 双声道音频</h3><div class="beat"><span>120 BPM</span>24 拍；关键事件每 15 帧对齐</div><div class="beat"><span>1.5 / 4.5 / 7 s</span>启动、检查器、引用点击命中</div><div class="beat"><span>7.5–10 s</span>原文稳定停留 2.5 秒</div><div class="beat"><span>11–12 s</span>品牌稳定停留 1 秒</div><div class="beat"><span>48 kHz</span>AAC 音轨时长 12 秒</div><div class="beat"><span>{report["audio"]["true_peak_dbfs_4x"]:.1f} dBFS</span>4 倍采样峰值；无削波</div><div class="pending">[2] 对应「升级演练 / 回退条件」<br>正文与现有原创合成资料一致。<br>全程保留压缩演示标识。</div></aside>"""
    html = html[:a0] + notes + html[a1:]
    html = html.replace(
        "引文逐字取自 frontend/src/lib/demo.ts 的原创合成资料",
        "抽帧来自 EvidenceRAG-12s.mp4；引文取自 demo.ts 原创合成资料",
    )
    write_text(FINAL / "check-sheet.html", html)
    sheet(html, FINAL / "check-sheet.png")
    print("Five decoded frames and check-sheet.png generated.", flush=True)
    payload = {"videoBase64": base64.b64encode(read_bytes(VIDEO)).decode("ascii")}
    proc = subprocess.run(
        ["node", str(HERE / "playback.mjs")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    playback = json.loads(proc.stdout)
    thumbs = playback.pop("thumbnails")
    assert (
        playback["ended"]
        and playback["currentTime"] == 12
        and playback["playbackRate"] == 1
        and not playback["muted"]
    )
    assert (
        playback["wallMs"] >= 11800
        and playback["totalVideoFrames"] == 360
        and playback["droppedVideoFrames"] == 0
    )
    assert max(x["rms"] for x in playback["audioAudits"]) > 0.01
    write_json(FINAL / "playback-checks.json", playback)
    gallery = (
        "<style>*{box-sizing:border-box}body{margin:0;padding:12px;background:#e6eff4;font-family:Arial;display:grid;grid-template-columns:repeat(4,1fr);gap:12px}article{background:#f5fafd}img{display:block;width:100%}p{margin:0;padding:7px;font-size:20px}</style>"
        + "".join(
            f'<article><img src="{t["data"]}"><p>{t["time"]:.3f} s · actual playback</p></article>'
            for t in thumbs
        )
    )
    sheet(gallery, FINAL / "playback-contact.png")
    report["status"] = "passed"
    report["playback"] = {
        "ended": True,
        "wall_ms": playback["wallMs"],
        "dropped_frames": playback["droppedVideoFrames"],
        "total_frames": playback["totalVideoFrames"],
        "unmuted_audio_active": True,
    }
    write_json(FINAL / "technical-checks.json", report)
    print(json.dumps(report["playback"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
