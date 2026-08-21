"""Append-only, provenance-guarded cache for rerank pair scores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zhrag.io_utils import append_jsonl, read_json, read_jsonl, write_json

__all__ = [
    "PairScore",
    "append_pair_scores",
    "load_pair_score_provenance",
    "load_pair_scores",
    "prepare_pair_score_cache",
    "validate_pair_score_cache",
]


@dataclass(frozen=True, slots=True)
class PairScore:
    """A text-free score for one query/document pair."""

    query_id: str
    doc_id: str
    score: float


def prepare_pair_score_cache(cache: Path, provenance: Mapping[str, object]) -> None:
    """Create or validate the sidecar that makes cached scores reusable."""
    expected = dict(provenance)
    sidecar = _sidecar(cache)
    has_scores = cache.exists() and cache.stat().st_size > 0
    if sidecar.exists():
        raw = read_json(sidecar)
        if not isinstance(raw, dict):
            raise SystemExit(f"! malformed rerank cache metadata: {sidecar}")
        if raw != expected:
            if not has_scores:
                write_json(sidecar, expected, indent=2)
                return
            raise SystemExit(
                f"! rerank cache metadata drift: {sidecar}\n  cached={raw}\n  current={expected}"
            )
        return
    if has_scores:
        raise SystemExit(f"! refusing to adopt rerank cache without provenance: {cache}")
    write_json(sidecar, expected, indent=2)


def load_pair_score_provenance(cache: Path) -> dict[str, object]:
    """Read an existing score sidecar without creating or modifying it."""
    sidecar = _sidecar(cache)
    if not sidecar.exists():
        raise SystemExit(f"! rerank cache provenance is absent: {sidecar}")
    raw = read_json(sidecar)
    if not isinstance(raw, dict):
        raise SystemExit(f"! malformed rerank cache metadata: {sidecar}")
    return raw


def validate_pair_score_cache(cache: Path, provenance: Mapping[str, object]) -> None:
    """Validate an existing cache sidecar without changing filesystem state."""
    recorded = load_pair_score_provenance(cache)
    expected = dict(provenance)
    if recorded != expected:
        raise SystemExit(
            f"! rerank cache metadata drift: {_sidecar(cache)}\n"
            f"  cached={recorded}\n  current={expected}"
        )


def load_pair_scores(cache: Path) -> dict[tuple[str, str], float]:
    """Load append-only scores with last-write-wins semantics."""
    if not cache.exists():
        return {}
    scores: dict[tuple[str, str], float] = {}
    for row in read_jsonl(cache):
        if not isinstance(row, dict):
            raise SystemExit(f"! malformed row in {cache}: expected an object")
        query_id = _string_field(row, "query_id", cache)
        doc_id = _string_field(row, "doc_id", cache)
        raw_score = row.get("score")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise SystemExit(f"! malformed score in {cache}: {raw_score!r}")
        score = float(raw_score)
        if not math.isfinite(score):
            raise SystemExit(f"! non-finite score in {cache}: {raw_score!r}")
        scores[(query_id, doc_id)] = score
    return scores


def append_pair_scores(cache: Path, rows: Iterable[PairScore]) -> None:
    """Checkpoint scores as UTF-8 JSONL without query or document text."""
    payloads: list[dict[str, Any]] = []
    for row in rows:
        if not math.isfinite(row.score):
            raise ValueError("rerank scores must be finite")
        payloads.append(
            {
                "query_id": row.query_id,
                "doc_id": row.doc_id,
                "score": row.score,
            }
        )
    append_jsonl(cache, payloads)


def _sidecar(cache: Path) -> Path:
    return Path(f"{cache}.meta.json")


def _string_field(row: Mapping[str, Any], name: str, cache: Path) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"! malformed {name} in {cache}: {value!r}")
    return value
