"""Assemble corpus-grounded task drafts and verify their quotes against local documents.

Drafts are authored offline (by people or by a model) as JSON arrays. Assembly proves
three mechanical facts before a draft becomes an unreviewed task row: every quoted
evidence string appears verbatim in the referenced local document, every referenced
source is part of the published index, and split assignment follows one plan per
source group. It never marks a task reviewed; that stays a human decision.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from zhrag.eval.agent_tasks import CATEGORIES, EXPECTED_STATUSES, validate_tasks
from zhrag.io_utils import read_text

DRAFT_FIELDS = frozenset(
    {
        "task_id",
        "category",
        "source_group",
        "question",
        "expected_status",
        "acceptance_criteria",
        "reference_sources",
        "evidence_quotes",
        "author_notes",
    }
)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SHORTCODE = re.compile(r"\{\{<[^}]{0,80}>\}\}")
_TAG = re.compile(r"<[^<>\n]{1,80}>")
_NOISE = re.compile(r"[\s*`_>#|\-\[\]+()]+")


def normalize_for_match(text: str) -> str:
    """Collapse markdown decoration so a quote survives link stripping and re-wrapping."""
    stripped = _TAG.sub("", _SHORTCODE.sub("", _LINK.sub(r"\1", text)))
    return _NOISE.sub("", stripped).casefold()


@dataclass(frozen=True, slots=True)
class DraftProblem:
    task_id: str
    message: str


def _quote_found(quote: str, document: str) -> bool:
    needle = normalize_for_match(quote)
    return bool(needle) and needle in normalize_for_match(document)


def _check_sources(
    task_id: str, draft: Mapping[str, object], sources: list[object], indexed: frozenset[str]
) -> list[DraftProblem]:
    problems = [
        DraftProblem(task_id, f"source not in published index: {source}")
        for source in sources
        if source not in indexed
    ]
    if draft["expected_status"] == "answered" and not sources:
        problems.append(DraftProblem(task_id, "answered task without reference_sources"))
    if draft["category"] == "multi_document" and len(set(sources)) < 2:
        problems.append(DraftProblem(task_id, "multi_document task needs two distinct sources"))
    return problems


def _check_quotes(
    task_id: str, quotes: list[object], *, corpus_root: Path, local_paths: Mapping[str, str]
) -> tuple[list[DraftProblem], set[str]]:
    problems: list[DraftProblem] = []
    cited: set[str] = set()
    documents: dict[str, str] = {}
    for index, item in enumerate(quotes):
        if not isinstance(item, Mapping) or set(item) != {"source", "quote"}:
            problems.append(DraftProblem(task_id, f"quote {index} malformed"))
            continue
        source, quote = str(item["source"]), str(item["quote"])
        cited.add(source)
        if source not in local_paths:
            problems.append(DraftProblem(task_id, f"quote {index} cites unknown source {source}"))
            continue
        if source not in documents:
            documents[source] = read_text(corpus_root / local_paths[source])
        if not _quote_found(quote, documents[source]):
            problems.append(DraftProblem(task_id, f"quote {index} not found verbatim in {source}"))
    return problems, cited


def verify_draft(
    draft: Mapping[str, object],
    *,
    corpus_root: Path,
    local_paths: Mapping[str, str],
    indexed_sources: frozenset[str],
) -> list[DraftProblem]:
    """Return mechanical problems; an empty list is not an endorsement of correctness."""
    task_id = str(draft.get("task_id", "<missing>"))
    if set(draft) != DRAFT_FIELDS:
        return [DraftProblem(task_id, "draft schema mismatch")]
    problems: list[DraftProblem] = []
    if draft["category"] not in CATEGORIES or draft["expected_status"] not in EXPECTED_STATUSES:
        problems.append(DraftProblem(task_id, "invalid category or expected_status"))
    sources = draft["reference_sources"]
    quotes = draft["evidence_quotes"]
    if not isinstance(sources, list) or not isinstance(quotes, list):
        return [*problems, DraftProblem(task_id, "sources and quotes must be lists")]
    problems.extend(_check_sources(task_id, draft, sources, indexed_sources))
    quote_problems, cited = _check_quotes(
        task_id, quotes, corpus_root=corpus_root, local_paths=local_paths
    )
    problems.extend(quote_problems)
    if draft["expected_status"] == "answered":
        problems.extend(
            DraftProblem(task_id, f"reference source has no quote: {source}")
            for source in sources
            if source not in cited
        )
    return problems


def _list(draft: Mapping[str, object], name: str) -> list[object]:
    value = draft[name]
    if not isinstance(value, list):
        raise ValueError(f"draft field {name} must be a list: {draft['task_id']}")
    return list(value)


def assemble_tasks(
    drafts: Iterable[Mapping[str, object]],
    *,
    split_plan: Mapping[str, str],
    snapshot: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Turn verified drafts into unreviewed task rows plus a separate evidence sidecar."""
    tasks: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    for draft in drafts:
        group = str(draft["source_group"])
        if group not in split_plan:
            raise ValueError(f"source group without split assignment: {group}")
        tasks.append(
            {
                "task_id": draft["task_id"],
                "category": draft["category"],
                "split": split_plan[group],
                "source_group": group,
                "question": draft["question"],
                "expected_status": draft["expected_status"],
                "acceptance_criteria": _list(draft, "acceptance_criteria"),
                "reference_sources": _list(draft, "reference_sources"),
                "reviewed": False,
                "reviewer": "",
                "snapshot": snapshot,
            }
        )
        evidence.append(
            {
                "task_id": draft["task_id"],
                "evidence_quotes": _list(draft, "evidence_quotes"),
                "author_notes": draft["author_notes"],
            }
        )
    validate_tasks(tasks)
    return tasks, evidence
