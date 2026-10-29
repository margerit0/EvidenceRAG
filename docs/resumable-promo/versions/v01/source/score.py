"""Original lyric-free score and UI effects for the resumable-investigation promo.

Everything is read from plan.json: musical states follow the timeline's events,
notes sit on its beat grid and every effect starts on its event's sample, so
picture and sound share one clock. Python 3.10+ with numpy (the repository's
.venv has it); deterministic for a given numpy build.

    .venv/Scripts/python.exe score.py --plan ../plan.json --out <new .wav>

Writes a 48 kHz stereo 16-bit WAV, `<wav>.json` (plan and audio SHA-256, the
sidecar media.py checks before encoding) and `<wav>.cues.json` (every effect
and state change with its sample, for sync checks on the final MP4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np

SEED = 20261029

# Musical state from a shot start or event on; ids come from timeline.json.
STATES = [
    ("shot", "overview", "intro"),
    ("event", "pick-live", "connected"),
    ("event", "start-run", "running"),
    ("event", "waiting-input", "waiting"),
    ("event", "submit-supplement", "running"),
    ("event", "pause", "pausing"),
    ("event", "paused", "paused"),
    ("event", "resume-1", "running"),
    ("event", "service-stop", "outage"),
    ("event", "reconnected", "paused"),
    ("event", "resume-2", "running"),
    ("event", "answered", "answered"),
    ("shot", "brand", "brand"),
]

# Layer levels per state. Pausing thins the groove while the current call
# finishes; paused is the calmest state; the outage keeps only a muffled pad
# and a slow pulse.
MIX = {
    "intro": {"pad": 0.7, "bell": 0.8},
    "connected": {"pad": 0.65, "bass": 0.35, "arp": 0.55, "bell": 0.5},
    "running": {"pad": 0.45, "bass": 0.8, "kick": 0.8, "hat": 0.7, "arp": 0.8},
    "waiting": {"pad": 0.5, "bass": 0.2, "hat": 0.25, "arp": 0.35, "bell": 0.6},
    "pausing": {"pad": 0.5, "bass": 0.4, "hat": 0.35, "arp": 0.55},
    "paused": {"pad": 0.45, "arp": 0.2, "bell": 0.6},
    "outage": {"pad": 0.6, "pulse": 0.9},
    "answered": {"pad": 0.6, "bass": 0.55, "hat": 0.5, "arp": 0.7, "bell": 0.4},
    "brand": {"pad": 0.85, "bass": 0.45, "bell": 0.7},
}
# Per-note gains before the state levels above.
VOICE = {"pad": 0.035, "bass": 0.22, "push": 0.1, "kick": 0.24, "hat": 0.075, "arp": 0.11, "bell": 0.08}
LAYERS = ("pad", "bass", "kick", "hat", "arp", "bell", "pulse")
RISE_SECONDS = 0.04  # a joining layer reaches its level by its first note
FALL_SECONDS = {"pausing": 1.6, "outage": 0.3}  # by entered state; default below
DEFAULT_FALL = 0.45

# F major. Chords: pad voicing, bass root, arpeggio/bell tones (MIDI).
CHORDS = {
    "Fmaj9": ([57, 60, 64, 67], 41, [65, 69, 72, 76, 79]),
    "Dm9": ([53, 57, 60, 64], 38, [62, 65, 69, 72, 76]),
    "Bbmaj9": ([57, 60, 62, 65], 34, [58, 62, 65, 69, 72]),
    "Cadd9": ([55, 60, 62, 64], 36, [60, 64, 67, 72, 74]),
    "C11": ([58, 62, 65, 69], 36, [65, 67, 70, 72, 77]),
    "Dsus": ([50, 57, 62, 64], 38, [62, 64, 69, 74, 76]),
    "Fadd9": ([53, 57, 60, 67, 69], 41, [65, 69, 72, 77, 79]),
}
# Harmony: a state either cycles bar by bar from its entry or holds one chord.
HARMONY = {
    "intro": ["Fmaj9", "Bbmaj9"],
    "connected": ["Fmaj9", "Bbmaj9"],
    "running": ["Fmaj9", "Dm9", "Bbmaj9", "Cadd9"],
    "pausing": ["Fmaj9", "Dm9", "Bbmaj9", "Cadd9"],
    "waiting": ["C11"],
    "paused": ["C11"],
    "outage": ["Dsus"],
    "answered": ["Fmaj9", "Bbmaj9", "Fmaj9", "Cadd9"],
    "brand": ["Fadd9"],
}
CONTINUES = {"pausing": "running"}  # keeps counting bars from the running entry
ARP_PATTERN = [0, 2, 4, 3, 1, 3, 4, 2]  # tone index per eighth note in a bar
BELL_PATTERN = [(0.0, 4), (1.5, 2), (3.0, 3)]  # (beat in bar, tone index)


def digest(data: dict) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    body = {key: value for key, value in plan.items() if key != "planSha256"}
    if plan.get("schemaVersion") != 1 or plan.get("planSha256") != digest(body):
        raise ValueError("invalid or manually changed plan.json; recompile timeline.json")
    return plan


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


class Score:
    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.rate = plan["audio"]["sampleRate"]
        self.count = plan["audio"]["sampleCount"]
        self.fps = float(plan["video"]["fps"])
        self.beat = round(self.rate * 60 / plan["audio"]["bpm"])
        self.rng = np.random.default_rng(SEED)
        self.buses = {name: np.zeros((self.count, 2)) for name in ("music", "send", "sfx")}
        self.layers = {name: np.zeros((self.count, 2)) for name in LAYERS}
        self.cues: list[dict] = []
        self.events = {event["id"]: event for event in plan["events"]}
        self.shots = {shot["id"]: shot for shot in plan["shots"]}
        self.transitions = self._transitions()

    # ------------------------------------------------------------ timeline
    def frame_sample(self, frame: int) -> int:
        return round(frame * self.rate / self.fps)

    def _transitions(self) -> list[tuple[int, str]]:
        result = []
        for kind, key, state in STATES:
            if kind == "event":
                sample = self.events[key]["sample"]
            else:
                sample = self.frame_sample(self.shots[key]["startFrame"])
            result.append((sample, state))
        if [sample for sample, _ in result] != sorted(sample for sample, _ in result):
            raise ValueError("musical states are out of timeline order")
        return result

    def state_at(self, sample: int) -> tuple[str, int]:
        """State at a sample and the sample its chord cycle counts from."""
        state, entry = self.transitions[0][1], 0
        for at, name in self.transitions:
            if at > sample:
                break
            if CONTINUES.get(name) != state:
                entry = at
            state = name
        return state, entry

    def chord_at(self, sample: int) -> str:
        state, entry = self.state_at(sample)
        cycle = HARMONY[state]
        bars = (sample // (4 * self.beat)) - (entry // (4 * self.beat))
        return cycle[bars % len(cycle)]

    def wants(self, layer: str, sample: int) -> bool:
        """New notes of a layer start only while the current state uses it."""
        return MIX[self.state_at(sample)[0]].get(layer, 0.0) > 0

    def gain_envelopes(self) -> dict[str, np.ndarray]:
        """Piecewise-linear level per layer. A layer joining reaches its level
        on the state's first sample (its first note keeps its attack); a layer
        leaving fades from there while its last notes ring out."""
        samples = np.arange(self.count)
        envelopes = {}
        for layer in LAYERS:
            level = MIX[self.transitions[0][1]].get(layer, 0.0)
            xs, ys = [0], [level]
            for index, (at, state) in enumerate(self.transitions[1:], start=1):
                target = MIX[state].get(layer, 0.0)
                if target == level:
                    continue
                limit = self.transitions[index + 1][0] if index + 1 < len(self.transitions) else self.count
                if target > level:
                    begin, end = at - round(RISE_SECONDS * self.rate), at
                else:
                    begin = at
                    end = min(limit, at + round(FALL_SECONDS.get(state, DEFAULT_FALL) * self.rate))
                begin = max(begin, xs[-1] + 1)
                end = max(end, begin + 1)
                xs += [begin, end]
                ys += [level, target]
                level = target
            if xs[-1] < self.count - 1:
                xs.append(self.count - 1)
                ys.append(level)
            envelopes[layer] = np.interp(samples, xs, ys)
        return envelopes

    def cue(self, kind: str, sample: int, **extra) -> None:
        self.cues.append({"kind": kind, "sample": int(sample), "seconds": round(sample / self.rate, 6), **extra})

    # ------------------------------------------------------------ mixing
    def place(self, target: np.ndarray, start: int, signal: np.ndarray, gain: float, pan: float = 0.0) -> None:
        if signal.ndim == 1:
            angle = (pan + 1) * math.pi / 4
            signal = np.stack([signal * math.cos(angle), signal * math.sin(angle)], axis=1) * math.sqrt(2)
        begin = max(0, start)
        end = min(self.count, start + len(signal))
        if end > begin:
            target[begin:end] += signal[begin - start : end - start] * gain

    # ------------------------------------------------------------ voices
    def time(self, seconds: float) -> np.ndarray:
        return np.arange(round(seconds * self.rate)) / self.rate

    def pad_voice(self, midi: int, hold: float, attack: float, release: float) -> np.ndarray:
        t = self.time(hold + release)
        env = np.minimum(1, t / attack)
        tail = t > hold
        env[tail] *= 0.5 + 0.5 * np.cos(np.pi * np.minimum(1, (t[tail] - hold) / release))
        out = np.zeros((len(t), 2))
        for channel, cents in enumerate((-4.0, 4.0)):
            frequency = hz(midi) * 2 ** (cents / 1200)
            phases = self.rng.uniform(0, 2 * np.pi, 5)
            wave_ = sum(np.sin(2 * np.pi * k * frequency * t + phases[k - 1]) / k**1.7 for k in range(1, 6))
            drift = 1 + 0.05 * np.sin(2 * np.pi * (0.19 + 0.04 * channel) * t + phases[0])
            out[:, channel] = wave_ * drift
        return out * env[:, None]

    def pluck(self, midi: float, seconds: float, decay: float, bright: float = 0.35) -> np.ndarray:
        t = self.time(seconds)
        frequency = hz(midi)
        attack = 1 - np.exp(-t / 0.0025)
        tone = (
            np.sin(2 * np.pi * frequency * t) * np.exp(-t / decay)
            + bright * np.sin(2 * np.pi * 2 * frequency * t) * np.exp(-t / (decay * 0.35))
            + 0.12 * bright * np.sin(2 * np.pi * 3.01 * frequency * t) * np.exp(-t / (decay * 0.2))
        )
        fade = np.minimum(1, (len(t) - 1 - np.arange(len(t))) / (0.02 * self.rate))
        return tone * attack * fade

    def bell(self, midi: float, seconds: float = 1.6) -> np.ndarray:
        t = self.time(seconds)
        frequency = hz(midi)
        tone = (
            np.sin(2 * np.pi * frequency * t) * np.exp(-t / 0.9)
            + 0.35 * np.sin(2 * np.pi * 2.0 * frequency * t) * np.exp(-t / 0.35)
            + 0.12 * np.sin(2 * np.pi * 4.07 * frequency * t) * np.exp(-t / 0.12)
        )
        fade = np.minimum(1, (len(t) - 1 - np.arange(len(t))) / (0.05 * self.rate))
        return tone * (1 - np.exp(-t / 0.002)) * fade

    def bass_note(self, midi: int) -> np.ndarray:
        t = self.time(0.46)
        frequency = hz(midi)
        tone = np.sin(2 * np.pi * frequency * t) + 0.28 * np.sin(4 * np.pi * frequency * t) + 0.06 * np.sin(6 * np.pi * frequency * t)
        env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.3) * np.minimum(1, (0.46 - t) / 0.05)
        return tone * env

    def kick(self) -> np.ndarray:
        t = self.time(0.36)
        frequency = 46 + 78 * np.exp(-t / 0.032)
        body = np.sin(2 * np.pi * np.cumsum(frequency) / self.rate) * np.exp(-t / 0.15)
        click = self.rng.standard_normal(len(t)) * np.exp(-t / 0.0015) * 0.18
        return (body + click) * (1 - np.exp(-t / 0.0008)) * np.minimum(1, (0.36 - t) / 0.03)

    def hat(self) -> np.ndarray:
        t = self.time(0.07)
        noise = np.diff(self.rng.standard_normal(len(t) + 1))
        return noise * np.exp(-t / 0.018) * (1 - np.exp(-t / 0.0006))

    def noise_swell(self, seconds: float, smooth: int, rise: bool) -> np.ndarray:
        t = self.time(seconds)
        noise = np.convolve(self.rng.standard_normal(len(t)), np.ones(smooth) / smooth, mode="same")
        shape = (t / seconds) ** 2 if rise else np.sin(np.pi * np.minimum(1, t / seconds)) ** 2
        return noise * shape / (np.abs(noise).max() or 1)

    def tick(self, pitch: float, seconds: float, decay: float) -> np.ndarray:
        t = self.time(seconds)
        noise = np.convolve(np.diff(self.rng.standard_normal(len(t) + 1)), np.ones(3) / 3, mode="same")
        blip = np.sin(2 * np.pi * pitch * t) * np.exp(-t / decay) + 0.4 * np.sin(np.pi * pitch * t) * np.exp(-t / (decay * 2))
        shape = (1 - np.exp(-t / 0.0005)) * np.minimum(1, (seconds - t) / 0.004)
        return (0.3 * noise * np.exp(-t / 0.004) + blip) * shape

    # ------------------------------------------------------------ parts
    def write_music(self) -> None:
        beats = [beat["sample"] for beat in self.plan["beats"]]
        bar = 4 * self.beat
        # Pad: one voicing per chord span, overlapping releases as crossfades.
        spans: list[list] = []
        for beat in beats:
            chord = self.chord_at(beat)
            state, _ = self.state_at(beat)
            if spans and spans[-1][0] == chord and spans[-1][3] == state:
                continue
            spans.append([chord, beat, None, state])
        for index, span in enumerate(spans):
            span[2] = spans[index + 1][1] if index + 1 < len(spans) else self.count
        for chord, start, end, state in spans:
            hold = (end - start) / self.rate
            attack = 0.9 if state == "brand" else 0.32
            release = 2.4 if state == "brand" else 0.75
            for midi in CHORDS[chord][0]:
                voice = self.pad_voice(midi, hold, attack, release)
                self.place(self.layers["pad"], start, voice, VOICE["pad"])
            self.cue("chord", start, chord=chord, state=state)
        envelopes = self.gain_envelopes()
        wants = self.wants

        def velocity() -> float:
            return float(self.rng.uniform(0.9, 1.06))

        eighth = self.beat // 2
        for index, beat in enumerate(beats):
            chord = CHORDS[self.chord_at(beat)]
            in_bar = index % 4
            state, entry = self.state_at(beat)
            # Bass on beats 1 and 3, a lighter push on the last eighth.
            if wants("bass", beat):
                if in_bar in (0, 2):
                    self.place(self.layers["bass"], beat, self.bass_note(chord[1]), VOICE["bass"] * velocity())
                if in_bar == 3 and wants("bass", beat + eighth):
                    self.place(self.layers["bass"], beat + eighth, self.bass_note(chord[1] + 12), VOICE["push"] * velocity())
            # Kick on 1 and 3, plus an accent where the groove enters.
            if wants("kick", beat) and (in_bar in (0, 2) or beat == entry):
                self.place(self.layers["kick"], beat, self.kick(), VOICE["kick"] * velocity())
            if wants("hat", beat + eighth):
                self.place(self.layers["hat"], beat + eighth, self.hat(), VOICE["hat"] * velocity(), pan=0.3)
            # Arpeggio: eighths while working, quarter notes when calm.
            if wants("arp", beat):
                steps = (0, 1) if state in ("running", "pausing", "answered") else (0,)
                for step in steps:
                    at = beat + step * eighth
                    if not wants("arp", at):
                        continue
                    tone = chord[2][ARP_PATTERN[(in_bar * 2 + step) % len(ARP_PATTERN)]]
                    pan = -0.35 if (in_bar * 2 + step) % 2 == 0 else 0.35
                    self.place(self.layers["arp"], at, self.pluck(tone, 0.6, 0.22), VOICE["arp"] * velocity(), pan=pan)
            # Slow pulse while the service is down (one heartbeat per two beats).
            if wants("pulse", beat) and index % 2 == 0:
                for offset, gain in ((0, 0.3), (round(0.2 * self.rate), 0.18)):
                    self.place(self.layers["pulse"], beat + offset, self.pluck(31, 0.35, 0.09, bright=0.1), gain)
        # Bell figure in calm states, on the bar grid.
        for start in range(0, self.count, bar):
            for beat_offset, tone_index in BELL_PATTERN:
                at = start + round(beat_offset * self.beat)
                if at < self.count and wants("bell", at):
                    tone = CHORDS[self.chord_at(at)][2][tone_index] + 12
                    self.place(self.layers["bell"], at, self.bell(tone), VOICE["bell"], pan=0.15)
        for layer in LAYERS:
            self.layers[layer] *= envelopes[layer][:, None]
        music = sum(self.layers[layer] for layer in LAYERS)
        self.buses["music"] += music
        self.buses["send"] += 0.7 * self.layers["pad"] + self.layers["arp"] + self.layers["bell"]

    def write_effects(self) -> None:
        sfx, send = self.buses["sfx"], self.buses["send"]
        for event in self.plan["events"]:
            at = event["sample"]
            if event["action"] == "click":
                self.place(sfx, at, self.tick(2300, 0.05, 0.006), 0.13)
                self.cue("click", at, id=event["id"], frame=event["frame"])
            elif event["action"] == "type":
                for index, _ in enumerate(event["text"]):
                    frame = event["frame"] + 1 + index * event["everyFrames"]
                    sample = self.frame_sample(frame)
                    pitch = float(self.rng.uniform(3000, 3700))
                    self.place(sfx, sample, self.tick(pitch, 0.03, 0.003), 0.045, pan=float(self.rng.uniform(-0.2, 0.2)))
                    self.cue("key", sample, id=event["id"], frame=frame)
        events = self.events

        def chime(name: str, notes: list[tuple[float, float]], gain: float, pan: float = 0.0) -> None:
            at = events[name]["sample"]
            for offset, midi in notes:
                start = at + round(offset * self.rate)
                voice = self.bell(midi, 1.4)
                self.place(sfx, start, voice, gain, pan)
                self.place(send, start, voice, gain * 0.8, pan)
            self.cue("chime", at, id=name, frame=events[name]["frame"])

        # Before the run starts: a short rising air into the first downbeat.
        start = events["start-run"]["sample"]
        swell = self.noise_swell(0.5, 18, rise=True)
        self.place(sfx, start - len(swell), swell, 0.05)
        self.cue("swell-end", start, id="start-run", frame=events["start-run"]["frame"])
        chime("waiting-input", [(0.0, 81), (0.12, 84)], 0.06)
        chime("paused", [(0.0, 84), (0.14, 79)], 0.05)
        # Service stops: a falling tone and a soft thump; reconnect: a rising chime.
        stop = events["service-stop"]["sample"]
        t = self.time(0.6)
        frequency = 110 + 540 * np.exp(-t / 0.13)
        fall = np.sin(2 * np.pi * np.cumsum(frequency) / self.rate) * np.exp(-t / 0.22) * (1 - np.exp(-t / 0.003))
        self.place(sfx, stop, fall * np.minimum(1, (0.6 - t) / 0.05), 0.09)
        self.place(sfx, stop, self.pluck(33, 0.4, 0.12, bright=0.05), 0.12)
        self.cue("power-down", stop, id="service-stop", frame=events["service-stop"]["frame"])
        chime("reconnected", [(0.0, 77), (0.08, 84), (0.16, 89)], 0.055)
        chime("answered", [(0.0, 77), (0.06, 81), (0.12, 84), (0.18, 88), (0.24, 91)], 0.045)
        # The inspector slides in from the right when the citation opens.
        opened = events["open-citation"]["sample"]
        swoosh = self.noise_swell(0.5, 26, rise=False)
        stereo = np.stack([swoosh * np.linspace(0.9, 0.3, len(swoosh)), swoosh * np.linspace(0.3, 0.9, len(swoosh))], axis=1)
        self.place(sfx, opened, stereo, 0.05)
        self.cue("swoosh", opened, id="open-citation", frame=events["open-citation"]["frame"])
        brand = self.frame_sample(self.shots["brand"]["startFrame"])
        self.place(sfx, brand, self.pluck(41, 1.8, 0.7, bright=0.15), 0.14)
        self.cue("brand", brand, id="brand", frame=self.shots["brand"]["startFrame"])

    # ------------------------------------------------------------ processing
    def outage_filter(self, signal: np.ndarray) -> np.ndarray:
        """Close a one-pole low-pass over the outage and open it on reconnect."""
        down = self.events["service-stop"]["sample"]
        up = self.events["reconnected"]["sample"]
        begin = max(0, down - round(0.05 * self.rate))
        end = min(self.count, up + round(0.8 * self.rate))
        t = np.arange(begin, end)
        closing = np.clip((t - down) / (0.4 * self.rate), 0, 1)
        opening = np.clip((t - up) / (0.6 * self.rate), 0, 1)
        depth = closing * (1 - opening)
        cutoff = np.exp(np.log(18000) + (np.log(420) - np.log(18000)) * depth)
        alpha = 1 - np.exp(-2 * np.pi * cutoff / self.rate)
        out = signal.copy()
        state = signal[begin].copy()
        block = signal[begin:end]
        filtered = np.empty_like(block)
        for index in range(len(block)):
            state += alpha[index] * (block[index] - state)
            filtered[index] = state
        fade = np.minimum(1, np.minimum(np.arange(len(block)), np.arange(len(block))[::-1]) / (0.01 * self.rate))
        out[begin:end] = block + (filtered - block) * fade[:, None]
        return out

    def reverb(self, send: np.ndarray) -> np.ndarray:
        length = round(2.4 * self.rate)
        t = np.arange(length) / self.rate
        response = np.zeros((length, 2))
        for channel in range(2):
            noise = np.convolve(self.rng.standard_normal(length), np.ones(6) / 6, mode="same")
            response[:, channel] = noise * np.exp(-t / 0.55) * (1 - np.exp(-t / 0.012))
        response /= np.sqrt((response**2).sum(axis=0))
        size = 1 << (len(send) + length - 1).bit_length()
        wet = np.empty_like(send)
        for channel in range(2):
            spectrum = np.fft.rfft(send[:, channel], size) * np.fft.rfft(response[:, channel], size)
            wet[:, channel] = np.fft.irfft(spectrum, size)[: len(send)]
        return wet

    def high_pass(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Zero-phase high-pass (removes DC and sub-bass rumble)."""
        size = 1 << len(signal).bit_length()
        frequencies = np.fft.rfftfreq(size, 1 / self.rate)
        response = frequencies**4 / (frequencies**4 + cutoff**4)
        out = np.empty_like(signal)
        for channel in range(2):
            out[:, channel] = np.fft.irfft(np.fft.rfft(signal[:, channel], size) * response, size)[: len(signal)]
        return out

    def render(self) -> tuple[np.ndarray, dict]:
        self.write_music()
        self.write_effects()
        music = self.outage_filter(self.buses["music"])
        wet = self.outage_filter(self.reverb(self.buses["send"]))
        mix = self.high_pass(music + 0.32 * wet + self.buses["sfx"], 28.0)
        # Ease in from silence; fade the final chord's tail out by the last sample.
        t = np.arange(self.count) / self.rate
        end = self.count / self.rate
        envelope = np.minimum(1, t / 0.03) * np.where(t > end - 1.6, 0.5 + 0.5 * np.cos(np.pi * np.clip((t - (end - 1.6)) / 1.6, 0, 1)), 1)
        mix *= envelope[:, None]
        peak = float(np.abs(mix).max())
        target = 10 ** (-3.0 / 20)
        mix *= target / peak
        stats = {"peakBeforeNormalize": peak, "normalizedPeakDbfs": -3.0, "layerRms": {}}
        scale = target / peak
        for layer in LAYERS:
            active = np.abs(self.layers[layer]).max(axis=1) > 1e-9
            if active.any():
                rms = float(np.sqrt((self.layers[layer][active] ** 2).mean())) * scale
                stats["layerRms"][layer] = round(20 * math.log10(rms), 2) if rms > 0 else None
        sfx_peak = float(np.abs(self.buses["sfx"]).max()) * scale
        stats["sfxPeakDbfs"] = round(20 * math.log10(sfx_peak), 2)
        return mix, stats


def write_wav(path: Path, mix: np.ndarray, rate: int, rng: np.random.Generator) -> None:
    dither = (rng.random(mix.shape) - rng.random(mix.shape)) / 32768
    pcm = np.clip(np.round((mix + dither) * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists; use a new file: {args.out}")
    plan = load_plan(args.plan)
    score = Score(plan)
    mix, stats = score.render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(args.out, mix, score.rate, score.rng)
    audio_sha = file_digest(args.out)
    sidecar = {"planSha256": plan["planSha256"], "sampleRate": score.rate, "sampleCount": score.count,
               "channels": 2, "audioSha256": audio_sha, "seed": SEED, "numpy": np.__version__, **stats}
    Path(str(args.out) + ".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cues = {"planSha256": plan["planSha256"], "audioSha256": audio_sha, "sampleRate": score.rate,
            "states": [{"sample": sample, "state": state} for sample, state in score.transitions],
            "cues": sorted(score.cues, key=lambda cue: cue["sample"])}
    Path(str(args.out) + ".cues.json").write_text(json.dumps(cues, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "audioSha256": audio_sha, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
