"""Okapi BM25 over a pluggable analyzer.

This is deliberately hand-rolled rather than imported. It is ~80 lines, it is
the *strong* baseline every other configuration in the ablation table must
beat, and owning the implementation means the ablation can swap the analyzer
and the (k1, b) parameters as configuration rather than as a fork.

Scoring follows the standard Okapi BM25 with the ``log(1 + (N - df + 0.5) /
(df + 0.5))`` IDF variant, which is non-negative and so avoids the pathological
negative weights plain BM25 assigns to terms appearing in over half the corpus.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from zhrag.lexical.analyzers import Analyzer, char_ngram

__all__ = ["BM25", "BM25Params"]


@dataclass(frozen=True, slots=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


@dataclass
class BM25:
    """An in-memory BM25 index.

    At this project's scale (<20k chunks) an in-memory posting list is both
    faster and simpler than a service. If the corpus grows past a few hundred
    thousand documents, swap this for the vector store's native sparse index --
    the :meth:`search` signature is what the retriever depends on.
    """

    analyzer: Analyzer = field(default_factory=lambda: char_ngram(2))
    params: BM25Params = field(default_factory=BM25Params)

    _doc_ids: list[str] = field(default_factory=list, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _postings: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avgdl: float = field(default=0.0, repr=False)

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> BM25:
        """Build the index. Replaces any previously indexed content."""
        if len(doc_ids) != len(texts):
            raise ValueError(f"doc_ids ({len(doc_ids)}) and texts ({len(texts)}) differ in length")
        if not doc_ids:
            raise ValueError("cannot index an empty corpus")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids):
            raise ValueError("doc_ids must be non-empty strings")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")

        order = sorted(range(len(doc_ids)), key=lambda index: doc_ids[index])
        self._doc_ids = [doc_ids[index] for index in order]
        self._postings = defaultdict(list)
        self._lengths = []

        for i, index in enumerate(order):
            tf = Counter(self.analyzer(texts[index]))
            self._lengths.append(sum(tf.values()))
            for term, freq in tf.items():
                self._postings[term].append((i, freq))

        n = len(self._doc_ids)
        self._avgdl = sum(self._lengths) / n or 1.0
        self._idf = {
            term: math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self._postings.items()
        }
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(doc_id, score)`` pairs, highest score first."""
        if not self._doc_ids:
            raise RuntimeError("index() must be called before search()")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        k1, b = self.params.k1, self.params.b
        scores: dict[int, float] = defaultdict(float)
        # set() matters: a term repeated in the query must not double-count.
        for term in set(self.analyzer(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf[term]
            for doc, freq in posting:
                norm = 1 - b + b * self._lengths[doc] / self._avgdl
                scores[doc] += idf * (freq * (k1 + 1)) / (freq + k1 * norm)

        top = sorted(scores.items(), key=lambda kv: (-kv[1], self._doc_ids[kv[0]]))[:k]
        return [(self._doc_ids[doc], score) for doc, score in top]

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def __len__(self) -> int:
        return len(self._doc_ids)
