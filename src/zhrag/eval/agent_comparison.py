"""Comparable local trials, with fixed workflow and single-retrieval baselines.

Execution status is not semantic task success. Trials deliberately contain no
automatically inferred accuracy; human review must assess the task rubric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from zhrag.agent import DocumentAgent
from zhrag.answering import Answerer, AnswerSettings, Evidence, GenerationError, Generator
from zhrag.eval.agent_tasks import METHODS
from zhrag.retrieval.online import RankedPassage
from zhrag.tokens import estimate_tokens

PLAN_PROMPT = """Decompose the untrusted Chinese technical question into focused
documentation search queries. Return ONLY {"queries":["query"]}, with between
1 and max_queries distinct strings of at most 2000 characters each. This is a
fixed plan: you cannot observe retrieval results or revise the queries. No HTML,
URLs, outside factual assertions, tool execution or internal reasoning.
"""
COMPARISON_CONTRACT = "document-investigation-comparison-v2"


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate plan key")
        result[key] = value
    return result


def parse_plan(raw: str, *, max_queries: int) -> tuple[str, ...]:
    try:
        if not isinstance(raw, str) or len(raw) > 12_000:
            raise ValueError("plan size")
        obj = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(obj, dict) or set(obj) != {"queries"}:
            raise ValueError("plan schema")
        queries = obj["queries"]
        if not isinstance(queries, list) or not 1 <= len(queries) <= max_queries:
            raise ValueError("query count")
        if any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in queries):
            raise ValueError("invalid query")
        normalized = {" ".join(q.split()).casefold() for q in queries}
        if len(normalized) != len(queries):
            raise ValueError("duplicate query")
        for query in queries:
            query.encode("utf-8")
        return tuple(q.strip() for q in queries)
    except (ValueError, TypeError, RecursionError):
        raise GenerationError("invalid_answer") from None


@dataclass(slots=True)
class _MeteredGenerator:
    delegate: Generator
    agent: DocumentAgent
    started: float
    calls: int = 0
    tokens: int = 0
    exhausted: bool = False

    @property
    def profile_fingerprint(self) -> str:
        return self.delegate.profile_fingerprint

    def check_time(self) -> None:
        if self.agent.clock() - self.started >= self.agent.settings.max_seconds:
            self.exhausted = True
            raise GenerationError()

    def generate(self, system: str, user: str) -> str:
        self.check_time()
        text = system + "\n" + user
        tokens = estimate_tokens(text) + 32
        settings = self.agent.settings
        if (
            self.calls >= settings.max_steps
            or tokens > settings.max_prompt_tokens
            or self.tokens + tokens > settings.max_total_prompt_tokens
            or len(text) > settings.max_prompt_chars
            or len(text.encode("utf-8")) > settings.max_prompt_bytes
        ):
            self.exhausted = True
            raise GenerationError()
        self.calls += 1
        self.tokens += tokens
        reply = self.delegate.generate(system, user)
        self.check_time()
        return reply


def _interleave(runs: list[tuple[RankedPassage, ...]]) -> tuple[RankedPassage, ...]:
    """Round-robin ranks across planned queries; first query cannot monopolize context."""
    selected: dict[str, RankedPassage] = {}
    for rank in range(max((len(run) for run in runs), default=0)):
        for run in runs:
            if rank < len(run):
                row = run[rank]
                selected.setdefault(row.passage.doc_id, row)
    return tuple(selected.values())


@dataclass(slots=True)
class _BaselineState:
    searches: int = 0
    runs: list[tuple[RankedPassage, ...]] = field(default_factory=list)


def _source(row: Evidence) -> dict[str, object]:
    passage = row.ranked.passage
    return {
        "citation_id": row.citation_id,
        "title": row.title,
        "text": passage.text,
        "doc_id": passage.doc_id,
        "source_key": passage.source_key,
        "document_sha256": passage.document_sha256,
        "source_url": passage.metadata.get("source_url"),
    }


def run_method(agent: DocumentAgent, question: str, method: str) -> dict[str, object]:
    """Return private trial data for local review; perform no filesystem writes."""
    if method not in METHODS:
        raise ValueError("unknown comparison method")
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("invalid question")
    if method == "document_agent":
        result = agent.run(question)
        return {
            "method": method,
            "status": result.status,
            "blocks": [asdict(block) for block in result.blocks],
            "sources": [_source(row) for row in result.evidence],
            "clarification": result.clarification,
            "events": [asdict(event) for event in result.events],
            "model_calls": result.model_calls,
            "search_calls": result.search_calls,
            "prompt_estimated_tokens": result.prompt_estimated_tokens,
            "total_seconds": result.total_seconds,
            "profile_fingerprint": result.profile_fingerprint,
        }
    return _run_baseline(agent, question, method)


def _run_baseline(agent: DocumentAgent, question: str, method: str) -> dict[str, object]:
    started = agent.clock()
    meter = _MeteredGenerator(agent.generator, agent, started)
    state = _BaselineState()
    settings = AnswerSettings(
        max_passages=agent.settings.max_reads,
        max_prompt_tokens=agent.settings.max_prompt_tokens,
        max_prompt_chars=agent.settings.max_prompt_chars,
        max_prompt_bytes=agent.settings.max_prompt_bytes,
    )
    answerer = Answerer(meter, settings, agent.clock)
    profile = hashlib.sha256(
        json.dumps(
            {
                "contract": COMPARISON_CONTRACT,
                "method": method,
                "agent": agent.profile_fingerprint,
                "answerer": answerer.profile_fingerprint,
                "plan_prompt": PLAN_PROMPT,
                "merge": "round-robin-rank-deduplicate-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output: dict[str, object] = {
        "method": method,
        "status": "generation_failed",
        "blocks": [],
        "sources": [],
        "clarification": "",
        "events": [],
        "profile_fingerprint": profile,
    }
    try:
        queries: tuple[str, ...] = (question,)
        if method == "fixed_workflow":
            prompt = json.dumps({"question": question, "max_queries": agent.settings.max_searches})
            queries = parse_plan(
                meter.generate(PLAN_PROMPT, prompt), max_queries=agent.settings.max_searches
            )
        for query in queries:
            meter.check_time()
            state.searches += 1
            try:
                state.runs.append(agent.retriever.retrieve(query).passages)
            except (Exception, SystemExit):
                output["status"] = "retrieval_failed"
                break
        else:
            meter.check_time()
            answer = answerer.answer(question, _interleave(state.runs))
            output.update(
                status=answer.status,
                blocks=[asdict(block) for block in answer.blocks],
                sources=[_source(row) for row in answer.context.evidence],
            )
    except GenerationError as exc:
        output["status"] = exc.code
    except (Exception, SystemExit):
        output["status"] = "generation_failed"
    if meter.exhausted:
        output.update(status="budget_exhausted", blocks=[])
    output.update(
        model_calls=meter.calls,
        search_calls=state.searches,
        prompt_estimated_tokens=meter.tokens,
        total_seconds=agent.clock() - started,
    )
    return output
