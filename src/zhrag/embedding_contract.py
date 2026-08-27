"""Provider-neutral contracts for reproducible embedding caches.

Offline evaluators need to authenticate the vectors they read without importing
an HTTP client or loading ``.env``.  Keep the prompt and sidecar reader here;
``zhrag.providers.embedding`` re-exports the existing public names for callers
that also need to create vectors.
"""

from __future__ import annotations

from pathlib import Path

from zhrag.io_utils import read_json

__all__ = [
    "CRUD_EMBEDDING_MODEL",
    "DOCUMENT_PROMPT",
    "QUERY_PROMPT",
    "embedding_provenance_path",
    "load_embedding_provenance",
    "validate_embedding_cache",
]

CRUD_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DOCUMENT_PROMPT = ""

#: Asymmetric by design. Qwen3's document prompt is the empty string; adding a
#: prefix to both sides silently costs several points of R@1 and raises nothing.
#: The instruction is English even though the corpus is Chinese -- Qwen's own
#: advice, because the training-time instructions were English.
QUERY_PROMPT = (
    "Instruct: Given a Chinese question, retrieve the news passage that answers it\nQuery:"
)


def embedding_provenance_path(cache: str | Path) -> Path:
    """Return the sidecar path that records a cache's model and prompt."""
    path = Path(cache)
    return path.with_suffix(path.suffix + ".meta.json")


def load_embedding_provenance(cache: str | Path) -> dict[str, object]:
    """Read an embedding cache's model/prompt sidecar without modifying it."""
    meta = embedding_provenance_path(cache)
    if not meta.exists():
        raise SystemExit(f"! embedding cache provenance is absent: {meta}")
    try:
        recorded = read_json(meta)
    except ValueError as exc:
        raise SystemExit(f"! {meta.name} is not readable JSON: {exc}") from exc
    if not isinstance(recorded, dict):
        raise SystemExit(f"! {meta.name} should hold a JSON object, found {type(recorded)}.")
    return recorded


def validate_embedding_cache(cache: str | Path, *, model: str, prompt: str) -> None:
    """Validate an existing embedding cache without adopting or changing it."""
    path = Path(cache)
    recorded = load_embedding_provenance(path)
    expected = {"model": model, "prompt": prompt}
    drift = [
        (key, recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) != value
    ]
    if drift:
        detail = "; ".join(f"{key}: cache has {was!r}, expected {now!r}" for key, was, now in drift)
        raise SystemExit(
            f"! {path.name} was written under different settings.\n"
            f"  {detail}\n"
            "  Read-only evaluation cannot adopt or rewrite embedding provenance."
        )
