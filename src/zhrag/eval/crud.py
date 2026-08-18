"""Load the CRUD-RAG benchmark from its raw distribution file.

The subset shipped by ``build_subset.ps1`` indexes 500 documents and asks 500
single-evidence questions. That configuration is **retrieval-saturated**: a
character-bigram BM25 with no embeddings, no reranking and no tuning scores
R@1 98.6% / R@3 100% / MRR@10 0.993 on it. Every ablation arm lands on the same
number, so the benchmark cannot rank two retrieval configurations -- which is
the one thing this project needs it to do.

The cause is corpus size plus construction: each question was written *from* its
evidence document, so entity names, figures and dates overlap verbatim and
lexical matching alone resolves them.

The fix needs no download. ``raw/split_merged.json`` already contains all six
CRUD tasks (7,661 records), and harvesting every distinct news body from them
yields 5,681 unique documents. Re-measured over that pool with all 800
single-evidence queries (regenerate via ``scripts/run_lexical_sweep.py``):

    corpus     R@1     R@5   MRR@10
       800   98.0%  100.0%    0.990   <- gold floor: all evidence, no distractors
     1,000   96.2%  100.0%    0.981
     2,000   90.4%  100.0%    0.949
     4,000   80.6%   99.5%    0.890
     5,681   75.9%   99.2%    0.857

R@1 recovers 22 points of headroom. R@5 stays saturated at 99.2% even at full
size and must not be used as a headline metric.

Multi-evidence tasks are harder still:

    task                  n     R@1   MRR@10   ALL-gold@10
    questanswer_1doc    800   75.9%    0.857         99.9%
    questanswer_2docs   797   35.4%    0.816         87.3%
    questanswer_3docs   797   23.0%    0.799         68.9%

so 2docs and 3docs are where reranking and query decomposition have room to show
a measurable effect.

**R@1 is not comparable across those three rows.**
:func:`zhrag.eval.metrics.recall_at_k` returns the *fraction* of gold retrieved,
so at k=1 a 3-gold query can score at most 1/3. The 23.0% is 69% of its 33.3%
ceiling, not a collapse. Compare arities with
:func:`zhrag.eval.metrics.all_gold_at_k`, or hold arity fixed and compare
systems within a row.

Everything here is a pure function over the parsed JSON. I/O lives in
``scripts/build_eval_corpus.py`` so this module stays testable against small
in-memory fixtures.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIN_DOCUMENT_CHARS",
    "QA_TASKS",
    "Query",
    "build_queries",
    "document_id",
    "harvest_documents",
    "sample_corpus",
]

#: The three question-answering tasks, in ascending order of difficulty.
QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")

#: Fragments shorter than this are headlines or leads, not retrievable documents.
MIN_DOCUMENT_CHARS = 200

#: Record fields that hold a genuine news body, as an explicit allowlist.
#:
#: An allowlist rather than a prefix match, because ``hallu_modified`` also
#: carries ``hallucinatedContinuation`` and ``hallucinatedMod`` -- text that was
#: deliberately fabricated for the hallucination-detection task. Letting those
#: into a retrieval corpus would seed it with plausible Chinese news prose
#: asserting false facts, and any generator that retrieved one would be graded
#: against a gold answer it now had every reason to contradict.
#:
#: Included:
#:   news1/news2/news3   evidence articles from the three QA tasks
#:   text                full articles from ``event_summary``
#:   newsRemainder       article bodies from ``hallu_modified`` (1,228 documents,
#:                       verified distinct from the rest of the pool)
#:
#: Excluded on purpose:
#:   newsBeginning       lead fragment, below the length floor anyway
#:   beginning/continuing  ``continuing_writing`` halves, which would duplicate
#:                       articles already reachable through other tasks
#:   hallucinated*       fabricated text, see above
_BODY_FIELDS = frozenset({"news1", "news2", "news3", "text", "newsRemainder"})


def document_id(text: str) -> str:
    """Stable content-addressed id for a document.

    A content hash is the *right* key here and the wrong key for a living
    corpus. This is an immutable benchmark snapshot in which deduplication is
    the entire point -- the same article reached through ``questanswer_2docs``
    and through ``event_summary`` is one document and must collapse to one id.
    Contrast ``tidb-rag-curated``, where documents get edited upstream and a
    content hash would orphan every chunk on each revision; there the stable key
    is ``path``.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Query:
    """One evaluation question and the documents that answer it."""

    query_id: str
    question: str
    answer: str
    gold_doc_ids: tuple[str, ...]
    task: str

    def __post_init__(self) -> None:
        if not self.gold_doc_ids:
            raise ValueError(f"{self.query_id}: a query needs at least one gold document")


def harvest_documents(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    """Collect every distinct news body across all CRUD tasks.

    Scans all six tasks, not just the QA ones. ``event_summary`` and
    ``hallu_modified`` contribute 2,611 articles that are never anyone's
    evidence, which makes them ideal distractors: plausible same-domain Chinese
    news that no question is asking about. Restricting the harvest to the QA
    tasks would leave only the 3,070 gold documents and every query's answer
    would be in the corpus by construction.
    """
    pool: dict[str, str] = {}
    for records in raw.values():
        for record in records:
            for field, value in record.items():
                if field not in _BODY_FIELDS or not isinstance(value, str):
                    continue
                body = value.strip()
                if len(body) >= MIN_DOCUMENT_CHARS:
                    pool.setdefault(document_id(body), body)
    return pool


def build_queries(
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tasks: Iterable[str] = QA_TASKS,
    pool: Mapping[str, str] | None = None,
) -> list[Query]:
    """Build the query set with gold labels resolved to document ids.

    ``questanswer_2docs`` and ``questanswer_3docs`` carry two and three evidence
    documents respectively; the arity is read from the record rather than
    assumed from the task name, so a malformed record is dropped instead of
    silently producing a query with fewer gold documents than it should have.

    Passing ``pool`` restricts gold documents to those actually present in the
    corpus. A query whose evidence is missing is skipped rather than kept with
    an unreachable label -- otherwise it would depress every arm's recall by a
    constant and look like a retrieval failure.
    """
    queries: list[Query] = []
    for task in tasks:
        for record in raw.get(task, []):
            question = record.get("questions")
            answer = record.get("answers")
            record_id = record.get("ID")
            if not isinstance(question, str) or not question.strip() or not record_id:
                continue

            gold: list[str] = []
            for index in range(1, 4):
                body = record.get(f"news{index}")
                if isinstance(body, str) and len(body.strip()) >= MIN_DOCUMENT_CHARS:
                    gold.append(document_id(body))
            if not gold:
                continue
            if pool is not None and any(doc_id not in pool for doc_id in gold):
                continue

            queries.append(
                Query(
                    query_id=f"{task}:{record_id}",
                    question=question.strip(),
                    answer=answer.strip() if isinstance(answer, str) else "",
                    gold_doc_ids=tuple(dict.fromkeys(gold)),
                    task=task,
                )
            )
    return queries


def sample_corpus(
    pool: Mapping[str, str],
    queries: Sequence[Query],
    *,
    size: int | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Take a corpus of ``size`` documents that always retains every gold document.

    Used to reproduce the saturation curve. Distractors are drawn with a seeded
    shuffle so the 500 / 2,000 / 5,681 rows are comparable across runs and
    across machines; sampling gold documents out would change what is being
    measured rather than how hard it is.

    ``size=None`` returns the whole pool. A ``size`` below the gold count is an
    error, because silently returning more documents than asked for would make
    the corpus-size column of the ablation table a lie.
    """
    gold = {doc_id for query in queries for doc_id in query.gold_doc_ids}
    missing = gold - set(pool)
    if missing:
        raise ValueError(f"{len(missing)} gold documents are absent from the pool")
    if size is None:
        return dict(pool)
    if size < len(gold):
        raise ValueError(f"size {size} is below the {len(gold)} gold documents that must be kept")

    distractors = sorted(set(pool) - gold)
    random.Random(seed).shuffle(distractors)
    keep = list(gold) + distractors[: size - len(gold)]
    return {doc_id: pool[doc_id] for doc_id in keep}
