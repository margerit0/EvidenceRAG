"""Original deterministic, lyric-free 120 BPM score and quiet UI effects."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from zhrag.io_utils import write_json

SR = 48_000
SECONDS = 12
BPM = 120
BEAT = 60 / BPM
SEED = 10232026
HERE = Path(__file__).resolve().parent


def synthesize(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    mix = np.zeros((SR * SECONDS, 2), dtype=np.float64)
    events = []

    def add(
        signal: np.ndarray, when: float, gain: float, pan: float = 0, kind: str = "note"
    ) -> None:
        start = round(when * SR)
        length = min(len(signal), len(mix) - start)
        if length <= 0:
            return
        angle = (pan + 1) * np.pi / 4
        mix[start : start + length, 0] += signal[:length] * gain * np.cos(angle)
        mix[start : start + length, 1] += signal[:length] * gain * np.sin(angle)
        events.append({"time": when, "sample": start, "frame": round(when * 30), "kind": kind})

    def hz(midi: int) -> float:
        return 440 * 2 ** ((midi - 69) / 12)

    def key(midi: int, duration: float = 0.95) -> np.ndarray:
        t = np.arange(round(SR * duration)) / SR
        freq = hz(midi)
        env = (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.30) * np.minimum(1, (duration - t) / 0.06)
        return env * (
            np.sin(
                2 * np.pi * freq * t + 0.20 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t / 0.12)
            )
            + 0.13 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.035 * np.sin(2 * np.pi * 3 * freq * t)
        )

    # Six bars, each four quarter notes. Dadd9 / Bm7 / Gmaj9 / Aadd9 / Em7 / D6/9.
    chords = [
        [50, 57, 61, 64, 66],
        [47, 54, 57, 62],
        [43, 50, 54, 57, 59],
        [45, 52, 59, 61],
        [40, 47, 55, 59, 62],
        [50, 57, 59, 64, 66],
    ]
    melody = [
        74,
        None,
        78,
        None,
        73,
        None,
        74,
        78,
        71,
        None,
        74,
        78,
        73,
        None,
        76,
        81,
        71,
        None,
        74,
        None,
        74,
        78,
        81,
        None,
    ]
    for bar, chord in enumerate(chords):
        when = bar * 2
        duration = 2.5 if bar < 5 else 2
        t = np.arange(round(SR * duration)) / SR
        env = np.minimum(1, t / 0.24) * np.minimum(1, (duration - t) / 0.7)
        for j, note in enumerate(chord):
            tone = np.sin(2 * np.pi * hz(note) * t) + 0.08 * np.sin(2 * np.pi * hz(note) * 2 * t)
            add(
                tone * env,
                when,
                0.042 / np.sqrt(len(chord)),
                (j / (len(chord) - 1) - 0.5) * 0.7,
                "chord",
            )
        bass_t = np.arange(round(SR * 1.65)) / SR
        bass_env = (1 - np.exp(-bass_t / 0.012)) * np.exp(-bass_t / 0.52)
        add(np.sin(2 * np.pi * hz(chord[0] - 12) * bass_t) * bass_env, when, 0.09, 0, "bass")
    for beat in range(24):
        when = beat * BEAT
        t = np.arange(round(SR * 0.035)) / SR
        noise = rng.normal(0, 1, len(t))
        click = (
            np.concatenate(([0], np.diff(noise)))
            * np.exp(-t / 0.006)
            * np.sin(np.pi * np.minimum(1, t / 0.002))
        )
        add(click, when, 0.0045 if beat % 4 else 0.008, 0.12 if beat % 2 else -0.12, "beat")
        if melody[beat] is not None:
            add(
                key(melody[beat]),
                when,
                0.075 if beat < 20 else 0.095,
                (-0.2, 0.18, 0)[beat % 3],
                "key",
            )
    # Clicks and functional changes share the exact frame/sample origin with the picture.
    for when in [1.5, 4.5, 7.0]:
        t = np.arange(round(SR * 0.06)) / SR
        click = (
            (np.sin(2 * np.pi * 1250 * t) + 0.28 * np.sin(2 * np.pi * 2410 * t))
            * np.exp(-t / 0.010)
            * (1 - np.exp(-t / 0.001))
        )
        add(click, when, 0.055, 0, "pointer-click")
    for when, note in [(2, 81), (2.5, 83), (3.5, 85), (4, 86), (4.5, 81)]:
        add(key(note, 0.35), when, 0.018, 0, "flow-state")
    for when in [5, 7.5, 10.5]:
        t = np.arange(round(SR * 0.13)) / SR
        freq = 160 + 240 * np.exp(-t / 0.018)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        thud = np.sin(phase) * np.exp(-t / 0.025) * (1 - np.exp(-t / 0.002))
        add(thud, when, 0.029, 0, "panel-land")
    # Short, low-level taps add space without a wash of reverb.
    dry = mix.copy()
    for delay, gain in [(0.125, 0.075), (0.25, 0.035)]:
        n = round(delay * SR)
        mix[n:] += dry[:-n, ::-1] * gain
    fadein = np.minimum(1, np.arange(len(mix)) / (SR * 0.025))
    fadeout = np.minimum(1, (len(mix) - 1 - np.arange(len(mix))) / (SR * 0.8))
    mix *= np.maximum(0, fadein * fadeout)[:, None]
    gain = 0.52 / float(np.max(np.abs(mix)))
    mix *= gain
    path = output / "soundtrack.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-c:a",
            "pcm_s24le",
            str(path),
        ],
        input=mix.astype("<f4").tobytes(),
        capture_output=True,
        check=True,
    )
    report = {
        "bpm": BPM,
        "beats": 24,
        "duration_seconds": SECONDS,
        "sample_rate": SR,
        "samples": len(mix),
        "channels": 2,
        "seed": SEED,
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(mix)))),
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(mix**2)))),
        "first_sample": mix[0].tolist(),
        "last_sample": mix[-1].tolist(),
        "score_chords_midi": chords,
        "score_melody_midi": melody,
        "events": events,
        "path": str(path),
    }
    write_json(output / "audio-score.json", report)
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    r = synthesize(HERE / "final")
    print(
        json.dumps(
            {k: r[k] for k in ["bpm", "beats", "samples", "peak_dbfs", "rms_dbfs", "path"]},
            ensure_ascii=False,
        )
    )
