"""Lexical analyzers for Chinese BM25.

The default is character n-grams, not word segmentation. That is a measured
decision, not a stylistic one. On the 5,681-document CRUD-RAG pool with 500
queries (see ``scripts/run_lexical_sweep.py``):

    analyzer                 vocab      R@1     MRR@10   build    query
    char bigram            351,779    75.8%     0.856     1.7s    7.3ms
    jieba precise           85,365    73.8%     0.844     9.7s   14.5ms
    jieba cut_for_search    91,246    72.8%     0.838    10.8s   15.6ms
    jieba + char bigram    393,862    76.0%     0.859    12.4s   19.5ms

Character bigrams beat word segmentation on every retrieval metric while
building 6x faster and querying 2x faster with zero dependencies. The union
analyzer buys +0.2pp R@1 for 7x the build cost, which does not pay for itself.

``jieba`` is kept only so the comparison row above stays reproducible. It is an
optional extra: it has had no release since 0.42.1 (2020) and emits
``SyntaxWarning`` on Python 3.13.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["Analyzer", "char_ngram", "jieba_words", "union"]

type Analyzer = Callable[[str], list[str]]

_WS = re.compile(r"\s+")


def char_ngram(n: int = 2) -> Analyzer:
    """Character n-gram analyzer. ``n=2`` is the measured optimum for Chinese news.

    Whitespace is stripped rather than treated as a boundary: Chinese has no
    inter-word spaces, so preserving it would only create n-grams that straddle
    the incidental line wraps in the source text.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    def analyze(text: str) -> list[str]:
        s = _WS.sub("", text)
        return [s[i : i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])

    return analyze


def jieba_words(*, for_search: bool = False) -> Analyzer:
    """Word-segmentation analyzer. Requires the optional ``jieba`` extra.

    Retained for the ablation table only -- it loses to :func:`char_ngram`.
    """
    try:
        # Imported lazily: jieba is an optional extra and importing it costs
        # ~1s of dictionary loading that the default char-ngram path never pays.
        import jieba  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "jieba is an optional extra kept only for the lexical ablation row; "
            "install it with `uv pip install jieba`"
        ) from exc

    jieba.setLogLevel(60)
    cut = jieba.cut_for_search if for_search else jieba.lcut

    def analyze(text: str) -> list[str]:
        return [w for w in cut(text) if w and not w.isspace()]

    return analyze


def union(*analyzers: Analyzer) -> Analyzer:
    """Concatenate several analyzers' output into one term list."""

    def analyze(text: str) -> list[str]:
        out: list[str] = []
        for a in analyzers:
            out.extend(a(text))
        return out

    return analyze
