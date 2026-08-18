"""Approximate token counting for mixed Chinese/English text.

A real tokenizer is the right answer, but the ingestion pipeline needs a token
estimate *before* any network call, and pulling a tokenizer just to size chunks
is a heavy dependency for a heuristic. So we approximate with a two-component
model instead of the usual single chars-per-token constant, because a flat ratio
is badly wrong on this corpus: the TiDB docs mix Chinese prose (dense, ~1.5
chars/token) with English identifiers, SQL keywords and CLI flags (sparse, ~4
chars/token) inside the same paragraph.

Replace :func:`estimate_tokens` with the provider's real tokenizer once you are
willing to take the dependency; :func:`zhrag.chunking` only needs a callable.
"""

from __future__ import annotations

import re

__all__ = ["CJK_CHARS_PER_TOKEN", "LATIN_CHARS_PER_TOKEN", "cjk_ratio", "estimate_tokens"]

# Measured against the actual Qwen3 tokenizer (151k vocab), which is what
# Qwen3-Embedding-8B uses, over this project's two corpora:
#
#   tokenizer            zh      zh+en      en     512 tok in chars (zh/mixed/en)
#   Qwen3 (151k)      1.570      2.275   4.491        804 / 1165 / 2299
#   cl100k_base       0.849      1.959   4.491        435 / 1003 / 2299
#   o200k_base        1.331      2.305   4.491        682 / 1180 / 2299
#   BGE-M3 / XLM-R    1.524      2.076   3.456        780 / 1063 / 1770
#
# Note how badly cl100k does on Chinese (0.85 chars/token): sizing chunks with a
# GPT-3.5-era tokenizer would under-fill every Chinese chunk by nearly half.
CJK_CHARS_PER_TOKEN = 1.57
LATIN_CHARS_PER_TOKEN = 4.49

_CJK = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"  # Han + ext A + compat
    r"　-〿＀-￯]"  # CJK punctuation, full-width forms
)


def cjk_ratio(text: str) -> float:
    """Fraction of characters that are CJK ideographs or full-width punctuation."""
    if not text:
        return 0.0
    return len(_CJK.findall(text)) / len(text)


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed Chinese/English text.

    Splits characters into CJK and non-CJK buckets and applies a separate ratio
    to each. Accurate enough to size chunks; not accurate enough to bill against.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return max(1, round(cjk / CJK_CHARS_PER_TOKEN + other / LATIN_CHARS_PER_TOKEN))
