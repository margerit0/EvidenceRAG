"""Numeric checks of the final soundtrack. Python 3.10+ with numpy; ffmpeg on PATH.

Every effect cue the score logged must sit on the sample of the planned frame
it answers. The MP4's AAC track is then decoded and compared with the source
WAV: global offset (cross-correlation), coding noise, clipping, and the onset
of every effect cue measured the same way in both, so any shift the encode
introduced shows up as a difference. The onsets themselves are band-envelope
estimates; where music shares the band they can cross early, which is why the
placement check above is the one that ties sounds to frames. Loudness comes
from ffmpeg's EBU R128 meter (integrated, range, true peak), with momentary
loudness averaged per musical state.

    .venv/Scripts/python.exe verify_audio.py --plan ../plan.json \
        --wav <score.wav> --video <final.mp4> --out <verify-audio.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

MAX_LAG = 4800  # samples searched either way for the global offset
# Pass band (Hz) where each cue's sound starts; music that lands on the same
# beat starts on the same sample, which is the point of the beat grid.
BANDS = {
    "click": (1800, 2800),
    "key": (2800, 4200),
    "chime": (600, 2000),
    "power-down": (300, 900),
    "brand": (60, 400),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        rate, channels = stream.getframerate(), stream.getnchannels()
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    return data.reshape(-1, channels).astype(np.float64) / 32768, rate


def decode(video: Path, rate: int) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-f", "f32le",
               "-acodec", "pcm_f32le", "-ar", str(rate), "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)


def offset(source: np.ndarray, decoded: np.ndarray) -> tuple[int, float]:
    """Lag k with decoded[n + k] ≈ source[n], and the normalized correlation there."""
    x, y = source.mean(axis=1), decoded.mean(axis=1)
    size = 1 << (len(x) + len(y)).bit_length()
    corr = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size)), size)
    lags = np.concatenate([np.arange(0, MAX_LAG + 1), np.arange(-MAX_LAG, 0)])
    values = np.concatenate([corr[: MAX_LAG + 1], corr[-MAX_LAG:]])
    best = int(np.argmax(values))
    lag = int(lags[best])
    overlap_x = x[max(0, -lag) : len(x) - max(0, lag)]
    overlap_y = y[max(0, lag) : max(0, lag) + len(overlap_x)]
    return lag, float(values[best] / (np.linalg.norm(overlap_x) * np.linalg.norm(overlap_y)))


def onset_ms(signal: np.ndarray, cue: int, rate: int, band: tuple[int, int]) -> float | None:
    """First crossing of 30% between the pre-cue level and the peak, band-limited."""
    before, after = round(0.08 * rate), round(0.12 * rate)
    if cue - before < 0 or cue + after > len(signal):
        return None
    segment = signal[cue - before : cue + after]
    spectrum = np.fft.rfft(segment)
    frequencies = np.fft.rfftfreq(len(segment), 1 / rate)
    spectrum[(frequencies < band[0]) | (frequencies > band[1])] = 0
    filtered = np.fft.irfft(spectrum, len(segment))
    width = round(0.001 * rate)
    envelope = np.convolve(np.abs(filtered), np.ones(width) / width, mode="same")
    base = float(np.median(envelope[before - round(0.06 * rate) : before - round(0.01 * rate)]))
    peak = float(envelope[before - round(0.005 * rate) : before + round(0.06 * rate)].max())
    if peak <= base * 1.5:
        return None
    level = base + 0.3 * (peak - base)
    start = before - round(0.02 * rate)
    crossing = start + int(np.argmax(envelope[start:] >= level))
    return round((crossing - before) * 1000 / rate, 3)


def placement(plan: dict, cues: list[dict], rate: int) -> dict:
    """Cues logged with a frame: on that frame's sample, and that frame is the
    planned event (a typed key, the brand shot's first frame)."""
    fps = float(plan["video"]["fps"])
    events = {event["id"]: event for event in plan["events"]}
    shots = {shot["id"]: shot for shot in plan["shots"]}
    framed = [cue for cue in cues if "frame" in cue]
    wrong = []
    for cue in framed:
        event = events.get(cue["id"])
        if cue["kind"] == "key":
            planned = cue["frame"] in [event["frame"] + 1 + i * event["everyFrames"] for i in range(len(event["text"]))]
        elif event:
            planned = cue["frame"] == event["frame"] and cue["sample"] == event["sample"]
        else:
            planned = cue["frame"] == shots[cue["id"]]["startFrame"]
        if not planned or cue["sample"] != round(cue["frame"] * rate / fps):
            wrong.append(cue)
    kinds: dict[str, int] = {}
    for cue in framed:
        kinds[cue["kind"]] = kinds.get(cue["kind"], 0) + 1
    return {"checked": len(framed), "byKind": kinds, "wrong": wrong}


def loudness(video: Path) -> dict:
    command = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-map", "0:a:0",
               "-af", "ebur128=peak=true:framelog=info", "-f", "null", "-"]
    log = subprocess.run(command, capture_output=True, check=True).stderr.decode("utf-8", errors="replace")
    momentary = [(float(t), float(m)) for t, m in re.findall(r"t:\s*([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.]+|-inf)", log)
                 if m != "-inf"]
    summary = log[log.rfind("Summary:"):]
    pick = lambda pattern: float(re.search(pattern, summary, re.S).group(1))  # noqa: E731
    return {
        "integratedLufs": pick(r"I:\s*(-?[\d.]+) LUFS"),
        "loudnessRangeLu": pick(r"LRA:\s*(-?[\d.]+) LU"),
        "truePeakDbtp": pick(r"True peak:\s*Peak:\s*(-?[\d.]+) dBFS"),
        "momentary": momentary,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists; use a new file: {args.out}")
    plan = read_json(args.plan)
    sidecar = read_json(Path(str(args.wav) + ".json"))
    cues = read_json(Path(str(args.wav) + ".cues.json"))
    wav_sha = hashlib.sha256(args.wav.read_bytes()).hexdigest()
    source, rate = read_wav(args.wav)
    decoded = decode(args.video, rate)
    count = plan["audio"]["sampleCount"]
    report: dict = {
        "planSha256": plan["planSha256"],
        "videoSha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "wavSha256": wav_sha,
        "sidecarMatches": sidecar["planSha256"] == plan["planSha256"] == cues["planSha256"]
        and sidecar["audioSha256"] == wav_sha == cues["audioSha256"],
        "sampleRate": rate,
        "sourceSamples": len(source),
        "decodedSamples": len(decoded),
    }
    report["cuePlacement"] = placement(plan, cues["cues"], rate)
    lag, correlation = offset(source, decoded)
    aligned = decoded[lag : lag + len(source)] if lag >= 0 else np.vstack([np.zeros((-lag, 2)), decoded])[: len(source)]
    noise = aligned - source
    rms = lambda signal: float(np.sqrt(np.mean(signal**2)))  # noqa: E731
    report["alignment"] = {
        "lagSamples": lag,
        "lagMs": round(lag * 1000 / rate, 3),
        "correlation": round(correlation, 6),
        "codingNoiseDb": round(20 * math.log10(rms(noise) / rms(source)), 2),
    }
    report["levels"] = {
        "decodedPeakDbfs": round(20 * math.log10(float(np.abs(decoded).max())), 2),
        "sourcePeakDbfs": round(20 * math.log10(float(np.abs(source).max())), 2),
        "clippedSamples": int((np.abs(decoded) >= 1.0).sum()),
        "paddingSamples": len(decoded) - count,
    }
    mono_source, mono_decoded = source.mean(axis=1), aligned.mean(axis=1)
    measured = []
    for cue in cues["cues"]:
        band = BANDS.get(cue["kind"])
        if not band:
            continue
        in_wav = onset_ms(mono_source, cue["sample"], rate, band)
        in_mp4 = onset_ms(mono_decoded, cue["sample"], rate, band)
        measured.append({"kind": cue["kind"], "id": cue.get("id"), "frame": cue.get("frame"), "sample": cue["sample"],
                         "onsetWavMs": in_wav, "onsetMp4Ms": in_mp4,
                         "differenceMs": None if in_wav is None or in_mp4 is None else round(in_mp4 - in_wav, 3)})
    by_kind = {}
    for kind in BANDS:
        items = [item for item in measured if item["kind"] == kind]
        found = [item for item in items if item["differenceMs"] is not None]
        by_kind[kind] = {
            "cues": len(items),
            "measured": len(found),
            "wavOnsetMs": [min((i["onsetWavMs"] for i in found), default=None), max((i["onsetWavMs"] for i in found), default=None)],
            "maxAbsDifferenceMs": max((abs(i["differenceMs"]) for i in found), default=None),
        }
    report["cueOnsets"] = {"byKind": by_kind, "cues": measured}
    meter = loudness(args.video)
    states = cues["states"]
    per_state: dict[str, list[float]] = {}
    for t, value in meter["momentary"]:
        sample = t * rate
        name = [state["state"] for state in states if state["sample"] <= sample][-1]
        per_state.setdefault(name, []).append(value)
    report["loudness"] = {
        key: meter[key] for key in ("integratedLufs", "loudnessRangeLu", "truePeakDbtp")
    } | {"momentaryByStateLufs": {name: round(10 * math.log10(np.mean([10 ** (v / 10) for v in values])), 1)
                                  for name, values in per_state.items()}}
    report["checks"] = {
        "sidecar": report["sidecarMatches"],
        "cuesOnPlannedSamples": report["cuePlacement"]["checked"] > 0 and not report["cuePlacement"]["wrong"],
        "noOffset": lag == 0,
        "matchesSource": correlation > 0.99,
        "sampleCount": 0 <= report["levels"]["paddingSamples"] <= 1024,
        "noClipping": report["levels"]["clippedSamples"] == 0,
        "truePeakBelow1Dbtp": meter["truePeakDbtp"] < -1.0,
        "cueOnsetsUnchanged": all(v["measured"] == v["cues"] and v["maxAbsDifferenceMs"] <= 1.0 for v in by_kind.values()),
    }
    report["passed"] = all(report["checks"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "alignment": report["alignment"],
                      "placement": {k: report["cuePlacement"][k] for k in ("checked", "byKind")},
                      "levels": report["levels"], "onsets": by_kind,
                      "loudness": {k: v for k, v in report["loudness"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
