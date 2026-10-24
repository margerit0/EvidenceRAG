"""Check native motion on the DOM timeline and in the encoded video pixels."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np

from zhrag.io_utils import read_bytes, read_text, write_json


def write_graph_preview(video: Path, render: dict) -> None:
    canvas = next(s for s in render["samples"] if s["frame"] == 84)["graph"]["canvas"]
    x = round(canvas["x"] + canvas["w"] / 2 - 260)
    y = round(canvas["y"] - 4)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-ss", "2", "-t", "2.5", "-i", str(video),
        "-filter_complex", f"[0:v]crop=520:358:{x}:{y}:exact=1,fps=20,scale=1040:716:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[palette];[b][palette]paletteuse=dither=sierra2_4a",
        "-loop", "0", str(video.parent / "graph-motion-detail.gif"),
    ], capture_output=True, check=True)


def verify_graph_motion(video: Path, render: dict) -> dict:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    styles = read_text(repo / "frontend/src/styles.css")
    for contract in ["animation: graph-signal 1.4s linear infinite", "stroke-dasharray: 12 88", "animation: spin 1.3s linear infinite", "border-color 220ms ease-out", "stroke 200ms ease-out", "animation: node-status-enter 180ms ease-out"]:
        assert contract in styles, f"Frontend motion changed: {contract}"
    samples = {s["frame"]: s for s in render["samples"]}
    observed_edges = set()
    signal_frames = spinner_frames = 0
    for f in range(300):
        motion = samples[f]["graph"]["motion"]
        expected_edge = motion["flowingEdge"]
        assert len(motion["signals"]) == int(expected_edge is not None)
        assert len(motion["spinners"]) == int(motion["activeNode"] is not None)
        for signal in motion["signals"]:
            signal_frames += 1
            observed_edges.add(signal["edge"])
            assert signal["edge"] == expected_edge and signal["pathMatches"]
            assert signal["pathLength"] == "100" and signal["dasharray"] == "12px, 88px"
            assert signal["strokeWidth"] == 2.5 and signal["paused"]
            expected = 100 * (1 - (motion["ageMs"] % 1400) / 1400)
            assert abs(signal["offset"] - expected) < .001
            assert abs(signal["animationTime"] - motion["ageMs"]) < .001
        for spinner in motion["spinners"]:
            spinner_frames += 1
            assert spinner["node"] == motion["activeNode"] and spinner["paused"]
            assert abs(spinner["animationTime"] - motion["ageMs"]) < .001
    assert observed_edges == {"agent-search", "agent-read", "read-agent", "agent-finish"}
    assert signal_frames == 90 and spinner_frames == 105

    def decoded_change(kind: str) -> dict:
        # At these frames the layout, node fills, text, cursor and timeline are
        # stationary. Only the native line signal and loader have moving pixels.
        bounds = samples[84]["graph"]["motion"][kind][0]["bounds"]
        x, y = math.floor(bounds["x"]) - 5, math.floor(bounds["y"]) - 5
        w, h = math.ceil(bounds["right"]) - x + 5, math.ceil(bounds["bottom"]) - y + 5
        raw = subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf",
            rf"select=eq(n\,84)+eq(n\,89),crop={w}:{h}:{x}:{y}:exact=1,format=rgb24",
            "-fps_mode", "vfr", "-frames:v", "2", "-f", "rawvideo", "pipe:1",
        ], capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(2, h, w, 3)
        delta = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).max(axis=2)
        changed = int(np.count_nonzero(delta > 8))
        assert changed >= 12 and int(delta.max()) >= 20, (kind, changed, int(delta.max()))
        return {"frames": [84, 89], "crop": [x, y, w, h], "changed_pixels_over_8": changed, "maximum_channel_difference": int(delta.max())}

    result = {
        "status": "passed", "signal_frames": signal_frames, "spinner_frames": spinner_frames,
        "observed_flowing_edges": sorted(observed_edges),
        "native_timing_ms": {"signal": 1400, "spinner": 1300, "node_transition": 220, "edge_transition": 200, "status_enter": 180},
        "encoded_motion": {kind: decoded_change(kind) for kind in ["signals", "spinners"]},
        "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(read_bytes(p)).hexdigest() for p in [repo / "frontend/src/styles.css", repo / "frontend/src/components/ExecutionGraph.tsx"]},
        "note": "The existing frontend signal path and CSS keyframes are paused and sought from integer frame time; no wall-clock effects are relied on.",
    }
    write_graph_preview(video, render)
    write_json(video.parent / "graph-motion-checks.json", result)
    return result
