"""Build a TiDB retrieval evaluation set from the indexed chunks themselves.

The deployed TiDB index has never been measured. CRUD-RAG supplies the only
R@1/MRR numbers this project has, and they were measured on Chinese news with a
different instruction, so nothing about them transfers to technical
documentation. Without a ruler here, the chunk-size sweep, the generation-side
evaluation and any future tuning all reduce to eyeballing screenshots.

There is no public Chinese TiDB retrieval benchmark, so the set is synthesised
from the corpus. That construction has one well-known failure mode, and this
project has already been bitten by it once: a question written *from* its
evidence document reuses that document's rare strings, so lexical matching alone
resolves it and the benchmark saturates. Three things are done about it here,
and none of them is "hope":

* **A paraphrase twin.** Every sampled chunk yields two questions with the same
  gold: a direct one and a rewrite that avoids the chunk's distinctive strings
  wherever a natural synonym exists. The two sets are aligned, so the effect of
  lexical overlap on any system becomes a paired contrast rather than a caveat.
* **A measured overlap statistic.** :func:`bigram_containment` reports how much
  of a query is literally present in its gold chunk, using the *same* character
  bigrams the BM25 arm tokenizes with. Reporting metrics stratified by it turns
  the leakage from an unquantified worry into a number.
* **A separate verifier pass.** Generation and verification are separate calls.
  The configured model is used for both by default and reported as same-model
  self-agreement; an explicitly selected second model provides independent review.
  The verifier also checks that the direct/paraphrase pair has one intent.

What is deliberately *not* done: candidates are never filtered by whether the
retriever finds them. Dropping the queries a system misses manufactures a
perfect score, and it is the single easiest way to produce a benchmark that
proves whatever the author already believed.

Everything here is a pure function over in-memory values. HTTP, caching and the
paid loop live in ``scripts/build_tidb_queries.py`` so the sampling, prompting,
parsing and validation rules can be tested without a key or a corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from zhrag.lexical.analyzers import char_ngram

__all__ = [
    "GENERATION_CACHE_KEY_SCHEMA",
    "GENERATION_INSTRUCTIONS",
    "MAX_ANSWER_CHARS",
    "MAX_QUESTION_CHARS",
    "MIN_QUESTION_CHARS",
    "QGEN_SCHEMA",
    "QUESTION_TYPES",
    "SELF_REFERENCE_PHRASES",
    "VARIANTS",
    "VERIFICATION_CACHE_KEY_SCHEMA",
    "VERIFICATION_INSTRUCTIONS",
    "EvalChunk",
    "GeneratedQuery",
    "GenerationOutcome",
    "Verdict",
    "bigram_containment",
    "dedupe_questions",
    "generation_instructions_fingerprint",
    "generation_prompt",
    "instructions_fingerprint",
    "parse_generation",
    "parse_verdict",
    "query_id",
    "retain_complete_pairs",
    "stratified_sample",
    "theme_counts",
    "verification_cache_id",
    "verification_instructions_fingerprint",
    "verification_prompt",
]

QGEN_SCHEMA = "zhrag-tidb-qgen-v1"
GENERATION_CACHE_KEY_SCHEMA = "chunk-id-v1"
VERIFICATION_CACHE_KEY_SCHEMA = "candidate-content-v1"

#: ``direct`` copies the chunk's phrasing freely; ``paraphrase`` avoids it where
#: a natural synonym exists. Both carry the same gold, which is what makes the
#: pair usable as a controlled contrast rather than two unrelated query sets.
VARIANTS = ("direct", "paraphrase")

QUESTION_TYPES = frozenset({"concept", "howto", "config", "troubleshoot", "factoid"})

#: Real queries are short. The upper bound also rejects the failure mode where a
#: model restates the whole passage as a "question"; the lower bound rejects
#: fragments too generic to have a single answer.
MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 60
MAX_ANSWER_CHARS = 120

#: Phrases that point at the passage the question was written from. A query
#: containing one is unanswerable to anybody who has not already been handed the
#: gold document, which is exactly the situation a retrieval benchmark simulates.
SELF_REFERENCE_PHRASES = (
    "本文",
    "本节",
    "本章",
    "本页",
    "上文",
    "上述",
    "上表",
    "下表",
    "如上",
    "如下",
    "该章节",
    "该文档",
    "该表",
    "该片段",
    "这段",
    "这个表格",
    "文档中",
    "此处",
    "文中",
)

_ANALYZE = char_ngram(2)
_QUERY_ID_SCHEMA = "zhrag-tidb-query-id-v1"
_VERIFICATION_CACHE_ID_SCHEMA = "zhrag-tidb-verification-cache-id-v1"
_SAMPLE_SCHEMA = "zhrag-tidb-sample-v1"
_WHITESPACE = re.compile(r"\s+")

GENERATION_INSTRUCTIONS = """你在为 TiDB 中文文档的**检索评测集**撰写查询。

给你一段文档片段。请写出一个真实用户会提出的问题，且该问题的答案就在这段片段里。

要求：
1. 中文，一句话，不超过 40 个字。真实用户不会写长段落。
2. 问题必须自足：读者不看这段片段也能明白在问什么。
   禁止出现"本文""上述""该章节""文档中""这个表格"等指代当前片段的说法。
3. 必须足够具体，使得只有讲这个主题的文档才能回答。
   避免"TiDB 是什么""怎么优化性能"这类泛问。
4. 可以使用产品术语（如 BR、TiKV、系统变量名、参数名），真实用户就是这么问的。
5. 问题里不能已经包含答案。

同时给出一个**改写版**（paraphrase）：语义完全相同，但尽量不照抄片段里的稀有字符串；如果某个术语没有自然的替代说法（例如系统变量名），可以保留。改写版要像另一个用户用自己的话问同一件事。

还要给出一个**简短答案**：直接依据片段作答，不超过 60 字，不要复述问题。

如果这段片段不适合出题（例如只是链接列表、目录、版本号清单、纯 front matter，
或信息量太低），把 usable 设为 false 并说明原因，其余字段留空字符串。

只输出 JSON，字段：
{
  "usable": true/false,
  "reason": "",
  "question": "",
  "paraphrase": "",
  "answer": "",
  "question_type": "concept|howto|config|troubleshoot|factoid"
}"""

VERIFICATION_INSTRUCTIONS = """你在审核一份 TiDB 中文文档检索评测集的候选问题对。

给你一段文档片段、一个直接问题、一个改写问题和一个候选答案。请逐项严格判断：

1. answerable：仅凭这段片段，能否完整回答两个问题？任一问题只沾边但没有答案，判 false。
2. self_contained：两个问题是否都不依赖任何未给出的上下文？
   任一问题出现"本文""上述""该章节"等指代某段文字的说法，判 false。
3. specific：两个问题是否都足够具体，只有讲这个主题的文档才能回答？
   如果随便换一篇 TiDB 文档也能大致回答（例如"TiDB 有什么优势"），判 false。
4. leaks_answer：任一问题本身是否已经把答案说出来了？
5. answer_supported：候选答案是否被片段支持，且没有编造片段里没有的内容？
6. same_intent：改写问题是否与直接问题询问完全相同的事实，能由同一个答案完整回答？
   仅仅来自同一个片段或主题相近不算同义，判 false。

不要考虑检索系统能否找到这段片段，那不是你的任务。

只输出 JSON：
{
  "answerable": true/false,
  "self_contained": true/false,
  "specific": true/false,
  "leaks_answer": true/false,
  "answer_supported": true/false,
  "same_intent": true/false,
  "note": "不超过30字的理由"
}"""


def _fingerprint(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def generation_instructions_fingerprint() -> str:
    """Hash the generator contract without coupling it to verifier edits."""
    return _fingerprint(QGEN_SCHEMA, GENERATION_INSTRUCTIONS)


def verification_instructions_fingerprint() -> str:
    """Hash the verifier contract without coupling it to generator edits."""
    return _fingerprint(QGEN_SCHEMA, VERIFICATION_INSTRUCTIONS)


def instructions_fingerprint() -> str:
    """Hash both prompt contracts for whole-pipeline identity."""
    return _fingerprint(QGEN_SCHEMA, GENERATION_INSTRUCTIONS, VERIFICATION_INSTRUCTIONS)


@dataclass(frozen=True, slots=True)
class EvalChunk:
    """One indexed chunk, carrying the strata a sample must be balanced over."""

    chunk_id: str
    source_key: str
    ordinal: int
    collection: str
    theme: str
    text: str
    approx_tokens: int

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.source_key:
            raise ValueError("chunk_id and source_key must be non-empty")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.text.strip():
            raise ValueError(f"{self.chunk_id}: chunk text is blank")


@dataclass(frozen=True, slots=True)
class GeneratedQuery:
    """One candidate query and the chunk it was written from."""

    query_id: str
    chunk_id: str
    variant: str
    question: str
    answer: str
    question_type: str

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"unknown question_type {self.question_type!r}")


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """Either a validated pair of queries, or why the chunk produced none."""

    chunk_id: str
    queries: tuple[GeneratedQuery, ...] = ()
    rejection: str | None = None

    def __post_init__(self) -> None:
        if bool(self.queries) == bool(self.rejection):
            raise ValueError("an outcome carries either queries or a rejection, not both")


@dataclass(frozen=True, slots=True)
class Verdict:
    """A separate verifier pass's per-axis judgement of one candidate pair."""

    answerable: bool
    self_contained: bool
    specific: bool
    leaks_answer: bool
    answer_supported: bool
    same_intent: bool
    note: str = ""

    @property
    def keep(self) -> bool:
        return (
            self.answerable
            and self.self_contained
            and self.specific
            and self.answer_supported
            and self.same_intent
            and not self.leaks_answer
        )

    def failures(self) -> tuple[str, ...]:
        """Name the axes that rejected this query, for a reportable breakdown."""
        failed = [
            name
            for name, ok in (
                ("answerable", self.answerable),
                ("self_contained", self.self_contained),
                ("specific", self.specific),
                ("answer_supported", self.answer_supported),
                ("same_intent", self.same_intent),
                ("leaks_answer", not self.leaks_answer),
            )
            if not ok
        ]
        return tuple(failed)


def query_id(chunk_id: str, variant: str) -> str:
    """Derive a stable public query id from its gold chunk and variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return f"{variant}:{_fingerprint(_QUERY_ID_SCHEMA, chunk_id, variant)[:16]}"


def verification_cache_id(direct: GeneratedQuery, paraphrase: GeneratedQuery) -> str:
    """Bind one verifier reply to the exact candidate pair and answer."""
    if direct.chunk_id != paraphrase.chunk_id:
        raise ValueError("verification pair must share one chunk")
    if direct.variant != "direct" or paraphrase.variant != "paraphrase":
        raise ValueError("verification pair must be ordered direct, paraphrase")
    if direct.answer != paraphrase.answer:
        raise ValueError("verification pair must share one answer")
    return _fingerprint(
        _VERIFICATION_CACHE_ID_SCHEMA,
        direct.query_id,
        paraphrase.query_id,
        direct.chunk_id,
        direct.question,
        paraphrase.question,
        direct.answer,
    )


def _sample_key(seed: str, chunk_id: str) -> str:
    digest = hashlib.sha256()
    for value in (_SAMPLE_SCHEMA, seed, chunk_id):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def stratified_sample(
    chunks: Sequence[EvalChunk],
    *,
    size: int,
    seed: str,
) -> tuple[EvalChunk, ...]:
    """Draw ``size`` chunks, allocated across themes in proportion to the corpus.

    Deterministic by construction: the draw order inside a stratum is a hash of
    the seed and the chunk id, never :mod:`random`. A sample that shifted between
    runs would silently change which queries a cached generation belongs to.

    Allocation uses largest remainder, so a theme holding 1.6% of the corpus gets
    a whole seat rather than being rounded out of the benchmark entirely.
    """
    if size < 1:
        raise ValueError("size must be positive")
    if not chunks:
        raise ValueError("cannot sample an empty corpus")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("chunk ids must be unique")
    total = len(chunks)
    if size > total:
        raise ValueError(f"cannot sample {size} chunks from a corpus of {total}")

    strata: dict[str, list[EvalChunk]] = {}
    for chunk in chunks:
        strata.setdefault(chunk.theme, []).append(chunk)

    exact = {theme: size * len(rows) / total for theme, rows in strata.items()}
    quota = {theme: min(int(value), len(strata[theme])) for theme, value in exact.items()}
    remaining = size - sum(quota.values())
    # Largest remainder, with the theme name as a deterministic tie-break.
    order = sorted(strata, key=lambda theme: (-(exact[theme] - int(exact[theme])), theme))
    while remaining > 0:
        progressed = False
        for theme in order:
            if remaining == 0:
                break
            if quota[theme] < len(strata[theme]):
                quota[theme] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # pragma: no cover - guarded by the size check above
            raise ValueError("not enough chunks to fill the requested sample")

    picked: list[EvalChunk] = []
    for theme in sorted(strata):
        rows = sorted(strata[theme], key=lambda chunk: _sample_key(seed, chunk.chunk_id))
        picked.extend(rows[: quota[theme]])
    return tuple(sorted(picked, key=lambda chunk: (chunk.source_key, chunk.ordinal)))


def generation_prompt(chunk: EvalChunk) -> str:
    """Frame one chunk for the generator, delimited so instructions cannot blend."""
    return f"文档片段：\n<<<\n{chunk.text}\n>>>"


def verification_prompt(
    chunk: EvalChunk,
    direct: GeneratedQuery,
    paraphrase: GeneratedQuery,
) -> str:
    """Frame one candidate pair for joint quality and equivalence verification."""
    if direct.chunk_id != chunk.chunk_id or paraphrase.chunk_id != chunk.chunk_id:
        raise ValueError("verification pair does not belong to the supplied chunk")
    if direct.variant != "direct" or paraphrase.variant != "paraphrase":
        raise ValueError("verification pair must be ordered direct, paraphrase")
    if direct.answer != paraphrase.answer:
        raise ValueError("verification pair must share one answer")
    return (
        f"文档片段：\n<<<\n{chunk.text}\n>>>\n\n"
        f"直接问题：{direct.question}\n"
        f"改写问题：{paraphrase.question}\n"
        f"候选答案：{direct.answer}"
    )


def _clean(value: object) -> str:
    return _WHITESPACE.sub(" ", value.strip()) if isinstance(value, str) else ""


def _reject_question(question: str) -> str | None:
    if not question:
        return "invalid:empty question"
    if len(question) < MIN_QUESTION_CHARS:
        return f"invalid:question shorter than {MIN_QUESTION_CHARS} characters"
    if len(question) > MAX_QUESTION_CHARS:
        return f"invalid:question longer than {MAX_QUESTION_CHARS} characters"
    for phrase in SELF_REFERENCE_PHRASES:
        if phrase in question:
            return f"invalid:self-reference {phrase!r}"
    return None


def parse_generation(chunk: EvalChunk, raw: str) -> GenerationOutcome:
    """Validate a generator reply into a direct/paraphrase pair, or reject it.

    Rejection is a first-class outcome, not an exception: the share of chunks a
    generator declines is itself a corpus statistic worth reporting, and a
    benchmark that hid it would overstate how much of the index it covers.
    """
    rejection: str | None = None
    payload: object = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        rejection = f"invalid:malformed JSON ({exc.msg})"

    if rejection is None and not isinstance(payload, dict):
        rejection = "invalid:reply is not a JSON object"

    question = paraphrase = answer = question_type = ""
    if rejection is None:
        assert isinstance(payload, dict)
        if payload.get("usable") is not True:
            reason = _clean(payload.get("reason")) or "no reason given"
            rejection = f"declined:{reason}"
        else:
            question = _clean(payload.get("question"))
            paraphrase = _clean(payload.get("paraphrase"))
            answer = _clean(payload.get("answer"))
            question_type = _clean(payload.get("question_type"))
            for candidate in (question, paraphrase):
                problem = _reject_question(candidate)
                if problem:
                    rejection = problem
                    break
            if rejection is None and question == paraphrase:
                rejection = "invalid:paraphrase repeats the question"
            if rejection is None and not answer:
                rejection = "invalid:empty answer"
            if rejection is None and len(answer) > MAX_ANSWER_CHARS:
                rejection = f"invalid:answer longer than {MAX_ANSWER_CHARS} characters"
            if rejection is None and question_type not in QUESTION_TYPES:
                rejection = f"invalid:unknown question_type {question_type!r}"

    if rejection is not None:
        return GenerationOutcome(chunk.chunk_id, rejection=rejection)

    return GenerationOutcome(
        chunk.chunk_id,
        queries=tuple(
            GeneratedQuery(
                query_id=query_id(chunk.chunk_id, variant),
                chunk_id=chunk.chunk_id,
                variant=variant,
                question=text,
                answer=answer,
                question_type=question_type,
            )
            for variant, text in zip(VARIANTS, (question, paraphrase), strict=True)
        ),
    )


def parse_verdict(raw: str) -> Verdict:
    """Validate a verifier reply. A missing or non-boolean axis is a failure."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed verifier JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("verifier reply is not a JSON object")

    axes = {}
    for name in (
        "answerable",
        "self_contained",
        "specific",
        "leaks_answer",
        "answer_supported",
        "same_intent",
    ):
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"verifier axis {name!r} is {value!r}, not a boolean")
        axes[name] = value
    return Verdict(note=_clean(payload.get("note"))[:60], **axes)


def bigram_containment(query: str, text: str) -> float:
    """Fraction of the query's character bigrams that occur literally in ``text``.

    Deliberately the *same* bigrams :class:`zhrag.lexical.BM25` indexes with. A
    leakage statistic computed over a different tokenization would describe some
    other retriever than the one it is meant to caveat.

    Containment rather than Jaccard: the question is "how much of this query was
    copied", and a long document would drive any symmetric measure to zero
    regardless of how much copying happened.
    """
    query_grams = set(_ANALYZE(query))
    if not query_grams:
        return 0.0
    return len(query_grams & set(_ANALYZE(text))) / len(query_grams)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def retain_complete_pairs(
    queries: Iterable[GeneratedQuery],
    *,
    required_chunk_ids: Collection[str] | None = None,
) -> tuple[tuple[GeneratedQuery, ...], tuple[str, ...]]:
    """Keep only chunks whose direct and paraphrase queries both survived.

    The optional set is useful after verification: it prevents a stale or
    malformed cache row from turning a controlled pair into an orphan query.
    """
    rows = tuple(queries)
    by_chunk: dict[str, dict[str, GeneratedQuery]] = {}
    for query in rows:
        by_chunk.setdefault(query.chunk_id, {})[query.variant] = query
    eligible = set(by_chunk) if required_chunk_ids is None else set(required_chunk_ids)
    complete = {
        chunk_id
        for chunk_id, group in by_chunk.items()
        if chunk_id in eligible and all(variant in group for variant in VARIANTS)
    }
    kept = tuple(query for query in rows if query.chunk_id in complete)
    dropped = tuple(sorted(eligible - complete))
    return kept, dropped


def dedupe_questions(
    queries: Iterable[GeneratedQuery],
    *,
    threshold: float = 0.85,
) -> tuple[tuple[GeneratedQuery, ...], tuple[tuple[str, str], ...]]:
    """Drop cross-pair near-duplicates while keeping each twin pair aligned.

    Neighbouring chunks of one document repeat headings and boilerplate, so two
    of them can yield the same question with two different golds. Left in, such a
    pair is unanswerable *as specified*: whichever chunk the system returns, one
    of the two queries counts it wrong, and the benchmark charges a system for
    being right.

    Every surface form is compared across chunks, including direct/paraphrase
    collisions. A collision drops the whole later chunk so pairing is preserved.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be within (0, 1]")
    rows = sorted(queries, key=lambda query: (query.chunk_id, query.variant))
    by_chunk: dict[str, list[GeneratedQuery]] = {}
    for query in rows:
        by_chunk.setdefault(query.chunk_id, []).append(query)

    kept: list[tuple[str, frozenset[str]]] = []
    dropped: list[tuple[str, str]] = []
    survivors: list[GeneratedQuery] = []
    for chunk_id, group in by_chunk.items():
        variants = {query.variant for query in group}
        if variants != set(VARIANTS):
            raise ValueError(f"{chunk_id}: a complete query pair is required to deduplicate")
        grams = [frozenset(_ANALYZE(query.question)) for query in group]
        collision = next(
            (
                other
                for other, other_grams in kept
                if any(_jaccard(value, other_grams) >= threshold for value in grams)
            ),
            None,
        )
        if collision is not None:
            dropped.append((chunk_id, collision))
            continue
        kept.extend((chunk_id, value) for value in grams)
        survivors.extend(group)
    return tuple(survivors), tuple(dropped)


def theme_counts(chunks: Iterable[EvalChunk]) -> Mapping[str, int]:
    """Count chunks per theme, for reporting a sample against its population."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.theme] = counts.get(chunk.theme, 0) + 1
    return dict(sorted(counts.items()))
