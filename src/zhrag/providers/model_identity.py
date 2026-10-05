"""Model identity comparison shared by streamed and complete chat replies."""

from __future__ import annotations

from string import ascii_lowercase, ascii_uppercase

MODEL_IDENTITY_CONTRACT = "ascii-case-insensitive-v1"
_ASCII_LOWER = str.maketrans(ascii_uppercase, ascii_lowercase)


def model_names_match(requested: str, served: object) -> bool:
    """Ignore ASCII case only; keep namespaces, versions and whitespace significant."""
    return (
        isinstance(served, str)
        and bool(requested)
        and requested.translate(_ASCII_LOWER) == served.translate(_ASCII_LOWER)
    )
