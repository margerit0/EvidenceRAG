"""Pool retrieval candidates and grade them into multi-gold TiDB qrels.

``qgen.py`` produces one gold chunk per query: the chunk the question was written
from. That label is *correct but incomplete*. A 1,832-chunk documentation corpus
repeats itself -- the same system variable is described in a reference page, a
task page and an FAQ -- so a retriever that returns a different chunk answering
the same question is scored as a miss. Reported without correction, that
understates every system by an unknown amount, and understates the *best* system
most, because a stronger ranker is likelier to surface the alternative phrasing.

The standard correction is pooling (TREC): take the union of the top ranks of
every system under test, judge each pooled candidate, and treat anything outside
the pool as non-relevant. Three properties of that method matter here and are
enforced rather than assumed:

* **The pool must not be built from one system.** Judging only what BM25 returned
  would make BM25 look complete and every other arm look noisy. ``build_pool``
  takes a mapping of named runs and records which systems surfaced each
  candidate, so the report can show the contribution of each.
* **Judging order must not encode retrieval rank.** If candidates were judged in
  fused-rank order, any leniency drift within a batch would flow straight into
  the top ranks of the systems being measured. :func:`judging_order` therefore
  sorts by a keyed digest of the candidate id: deterministic, reproducible, and
  uncorrelated with any run.
* **Unjudged is not the same as irrelevant.** The qrels row keeps
  ``judged_doc_ids`` so a later metric can report how much of a run was judged
  at all instead of silently scoring unjudged documents as misses.

One judging call covers a direct/paraphrase pair. ``qgen`` verified that they
ask for the same fact and share one answer; both surfaces still appear in the
judging prompt, because semantic equivalence does not guarantee that every
alternative chunk answers an underspecified rewrite as well as the direct form.
The shared call therefore keeps the paired gold set honest while halving a pass
whose cost is dominated by the number of calls, not their size.

The generating chunk is judged along with everything else. It is *retained* as
gold regardless of the verdict, because two earlier passes already established
that it answers the question; but the judge's grade for it is recorded, and the
disagreement rate over ~500 controls is the only cheap estimate of judge error
this pipeline can produce.

Pure functions over in-memory values. HTTP, caching and the paid loop live in
``scripts/build_tidb_qrels.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "GRADE_LABELS",
    "JUDGING_CACHE_KEY_SCHEMA",
    "JUDGING_INSTRUCTIONS",
    "MAX_GRADE",
    "POOL_SCHEMA",
    "JudgedQuery",
    "PooledQuery",
    "batched",
    "build_pool",
    "judging_cache_id",
    "judging_input_fingerprint",
    "judging_instructions_fingerprint",
    "judging_order",
    "judging_prompt",
    "parse_judgements",
    "pool_contribution",
    "pool_fingerprint",
    "qrels_rows",
]

POOL_SCHEMA = "zhrag-tidb-pool-v1"
JUDGING_CACHE_KEY_SCHEMA = "batch-content-v1"

_JUDGING_CACHE_ID_SCHEMA = "zhrag-tidb-judging-cache-id-v1"
_ORDER_SCHEMA = "zhrag-tidb-pool-order-v1"
_POOL_FINGERPRINT_SCHEMA = "zhrag-tidb-pool-fingerprint-v1"

#: A fully relevant chunk answers the question on its own. Anything less is
#: recorded but is not gold: a partial chunk in the gold set would let a system
#: that retrieves context pages outscore one that retrieves answers.
MAX_GRADE = 2

GRADE_LABELS: Mapping[int, str] = {
    2: "完全回答：仅凭该片段就能回答问题",
    1: "部分相关：包含必要背景或部分信息，但单独不足以回答",
    0: "不相关",
}

JUDGING_INSTRUCTIONS = """你在为 TiDB 中文文档检索评测集标注相关性。

给你一对语义等价的问题、一个参考答案，以及若干个候选文档片段。
请为**每一个**候选片段独立打分；分数必须对两个问题都成立：

- 2 = 完全回答：仅凭这个片段就能回答该问题。
- 1 = 部分相关：包含必要的背景、前置条件或部分信息，但单独不足以回答该问题。
- 0 = 不相关：与该问题没有实质关系。

判断准则：

1. 只看片段本身能否完整回答两个问题，不要考虑它在检索结果里排第几，
   也不要考虑参考答案是从哪个片段写出来的。
2. 参考答案只用来明确问题问的是什么事实，不是评分标准；
   如果某个片段用不同的措辞给出了同一个事实，同样算 2。
3. 同一个问题可以有多个 2。不要因为已经给过一个 2 就压低其余片段。
4. 只是提到了问题里的产品名、参数名或术语，但没有回答问题，算 0。
5. 片段被截断、只剩标题或只剩链接列表时，算 0。

必须为每个给定编号各输出一条，不要增加或遗漏编号。

只输出 JSON：
{
  "judgements": [
    {"id": 1, "grade": 0},
    {"id": 2, "grade": 2}
  ]
}"""


def _fingerprint(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def judging_instructions_fingerprint() -> str:
    """Hash the judging contract so a prompt edit invalidates its cache."""
    return _fingerprint(POOL_SCHEMA, JUDGING_INSTRUCTIONS)


@dataclass(frozen=True, slots=True)
class PooledQuery:
    """One judging unit: a verified query pair and the candidates it pooled."""

    chunk_id: str
    query_ids: tuple[str, ...]
    questions: tuple[str, ...]
    answer: str
    candidates: tuple[str, ...]
    contributors: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("a pooled query needs its generating chunk id")
        if not self.query_ids:
            raise ValueError(f"{self.chunk_id}: a pooled query needs at least one query id")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError(f"{self.chunk_id}: duplicate query ids")
        if len(self.questions) != len(self.query_ids):
            raise ValueError(f"{self.chunk_id}: questions and query ids differ in length")
        if len(self.query_ids) != 2:
            raise ValueError(f"{self.chunk_id}: judging requires one direct/paraphrase pair")
        variants = {query_id.partition(":")[0] for query_id in self.query_ids}
        if variants != {"direct", "paraphrase"}:
            raise ValueError(f"{self.chunk_id}: query ids are not a direct/paraphrase pair")
        if any(not question.strip() for question in self.questions) or not self.answer.strip():
            raise ValueError(f"{self.chunk_id}: questions and answer must be non-empty")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"{self.chunk_id}: duplicate pooled candidates")
        candidate_set = set(self.candidates)
        if set(self.contributors) != candidate_set:
            raise ValueError(f"{self.chunk_id}: contributor keys differ from candidates")
        for doc_id, systems in self.contributors.items():
            if len(set(systems)) != len(systems):
                raise ValueError(f"{doc_id}: duplicate contributor systems")
        object.__setattr__(self, "contributors", MappingProxyType(dict(self.contributors)))
        if self.chunk_id not in self.candidates:
            # The generating chunk doubles as the control that measures judge
            # error; a pool without it silently drops that measurement.
            raise ValueError(f"{self.chunk_id}: the generating chunk must be pooled")


@dataclass(frozen=True, slots=True)
class JudgedQuery:
    """One pooled query after every candidate carries a grade."""

    chunk_id: str
    query_ids: tuple[str, ...]
    grades: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("judged query ids must be non-empty and unique")
        variants = {query_id.partition(":")[0] for query_id in self.query_ids}
        if len(self.query_ids) != 2 or variants != {"direct", "paraphrase"}:
            raise ValueError("judged query ids must be one direct/paraphrase pair")
        for doc_id, grade in self.grades.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("judged document ids must be non-empty strings")
            if isinstance(grade, bool) or not isinstance(grade, int):
                raise ValueError(f"{doc_id}: grade must be an integer")
            if grade not in GRADE_LABELS:
                raise ValueError(f"{doc_id}: grade {grade} is outside {sorted(GRADE_LABELS)}")
        object.__setattr__(self, "grades", MappingProxyType(dict(self.grades)))
        if self.chunk_id not in self.grades:
            raise ValueError(f"{self.chunk_id}: the generating chunk was not judged")

    @property
    def gold_doc_ids(self) -> tuple[str, ...]:
        """Fully relevant chunks, with the verified generating chunk retained."""
        gold = {doc_id for doc_id, grade in self.grades.items() if grade == MAX_GRADE}
        gold.add(self.chunk_id)
        return tuple(sorted(gold))

    @property
    def partial_doc_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                doc_id
                for doc_id, grade in self.grades.items()
                if grade == 1 and doc_id != self.chunk_id
            )
        )

    @property
    def judged_doc_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.grades))

    @property
    def generator_chunk_grade(self) -> int:
        return self.grades[self.chunk_id]


def build_pool(
    runs: Mapping[str, Sequence[str]],
    *,
    depth: int,
    required: Sequence[str] = (),
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Union the top ``depth`` of every named run, recording each contributor.

    ``required`` ids join the pool even when no run returned them, which is how
    the generating chunk stays present as a control. The returned order is the
    stable union order; :func:`judging_order` decides the order a judge sees.
    """
    if not runs:
        raise ValueError("pooling needs at least one run")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")

    best_rank: dict[str, int] = {}
    contributors: dict[str, list[str]] = {}
    for system in sorted(runs):
        seen: set[str] = set()
        for rank, doc_id in enumerate(runs[system][:depth], start=1):
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{system}: run contains a non-string document id")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            contributors.setdefault(doc_id, []).append(system)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    for doc_id in required:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("required contains a non-string document id")
        contributors.setdefault(doc_id, [])
        best_rank.setdefault(doc_id, depth + 1)

    ordered = tuple(sorted(best_rank, key=lambda doc_id: (best_rank[doc_id], doc_id)))
    return ordered, {doc_id: tuple(systems) for doc_id, systems in sorted(contributors.items())}


def pool_fingerprint(units: Sequence[PooledQuery]) -> str:
    """Hash the exact pair/question/answer/candidate pool without chunk text."""
    if not units:
        raise ValueError("pool must be non-empty")
    digest = hashlib.sha256()
    values = [_POOL_FINGERPRINT_SCHEMA]
    seen_query_ids: set[str] = set()
    for unit in units:
        if seen_query_ids & set(unit.query_ids):
            raise ValueError("pool repeats a query id")
        seen_query_ids.update(unit.query_ids)
        values.extend((unit.chunk_id, *unit.query_ids, *unit.questions, unit.answer))
        for doc_id in unit.candidates:
            values.append(doc_id)
            values.extend(unit.contributors[doc_id])
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def pool_contribution(
    contributors: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """Count how many pooled candidates each system contributed.

    A system whose contribution is near zero is not adding judgements, which
    means the pool is effectively single-system and the qrels inherit its blind
    spots. Reported so that stays visible instead of implied.
    """
    counts: dict[str, int] = {}
    for systems in contributors.values():
        for system in systems:
            counts[system] = counts.get(system, 0) + 1
    counts["unique_candidates"] = len(contributors)
    counts["exclusive_to_one_system"] = sum(
        1 for systems in contributors.values() if len(systems) == 1
    )
    return dict(sorted(counts.items()))


def judging_order(candidates: Sequence[str], *, seed: str) -> tuple[str, ...]:
    """Order candidates by a keyed digest, decorrelating position from rank.

    Deterministic for a given seed, so a resumed run rebuilds the same batches
    and reuses the same cache keys.
    """
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates contain duplicates")
    return tuple(sorted(candidates, key=lambda doc_id: _fingerprint(_ORDER_SCHEMA, seed, doc_id)))


def batched(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """Split candidates into fixed-size judging batches, preserving order."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch size must be a positive integer")
    return [tuple(items[start : start + size]) for start in range(0, len(items), size)]


def judging_cache_id(
    questions: Sequence[str],
    answer: str,
    batch: Sequence[str],
) -> str:
    """Bind a cached verdict to the exact pair, answer and batch it judged.

    The direct and paraphrase surfaces are both included. Judging only one
    surface and copying its gold set to the other assumes their answerability is
    identical for every *alternative* chunk, which ``same_intent`` does not
    establish: one phrasing may require a named product or version that the other
    omits. A candidate is fully relevant only when it answers both verified
    surfaces, so both are visible to the judge and bound into the cache key.
    """
    if not questions or any(
        not isinstance(question, str) or not question.strip() for question in questions
    ):
        raise ValueError("judging questions must be non-empty strings")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("judging answer must be a non-empty string")
    if not batch:
        raise ValueError("a judging batch must be non-empty")
    if len(set(batch)) != len(batch):
        raise ValueError("a judging batch must not repeat a candidate")
    return _fingerprint(_JUDGING_CACHE_ID_SCHEMA, *questions, answer, *batch)


def judging_input_fingerprint(
    units: Sequence[PooledQuery],
    *,
    corpus: Mapping[str, str],
    seed: str,
    batch_size: int,
) -> str:
    """Hash the exact text-bearing judging inputs and batch boundaries.

    Pool and cache ids intentionally omit passage text, so stable chunk ids alone
    cannot prove that a cached judgement still describes the current documents.
    """
    if not seed:
        raise ValueError("judging input seed must be non-empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("judging input batch size must be positive")
    digest = hashlib.sha256()

    def update(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    update("zhrag-tidb-judging-input-v1")
    update(seed)
    update(str(batch_size))
    for unit in units:
        update(unit.chunk_id)
        for value in (*unit.query_ids, *unit.questions, unit.answer):
            update(value)
        ordered = judging_order(unit.candidates, seed=f"{seed}:{unit.chunk_id}")
        for batch in batched(ordered, batch_size):
            for doc_id in batch:
                text = corpus.get(doc_id)
                if text is None:
                    raise KeyError(f"no text for pooled candidate {doc_id!r}")
                update(doc_id)
                update(text)
    return digest.hexdigest()


def judging_prompt(
    questions: Sequence[str],
    answer: str,
    batch: Sequence[str],
    texts: Mapping[str, str],
) -> str:
    """Render one batch as numbered passages.

    Candidates are numbered rather than named: a chunk id is a 64-character
    digest that carries no information a judge can use, costs tokens in both
    directions, and invites the model to echo a mistyped one.
    """
    if not questions or any(
        not isinstance(question, str) or not question.strip() for question in questions
    ):
        raise ValueError("questions must be non-empty strings")
    if not answer.strip():
        raise ValueError("answer must be non-empty")
    if not batch:
        raise ValueError("a judging batch must be non-empty")
    blocks = []
    for number, doc_id in enumerate(batch, start=1):
        text = texts.get(doc_id)
        if text is None:
            raise KeyError(f"no text for pooled candidate {doc_id!r}")
        blocks.append(f"[{number}]\n<<<\n{text}\n>>>")
    joined = "\n\n".join(blocks)
    rendered_questions = "\n".join(
        f"问题{number}：{question}" for number, question in enumerate(questions, start=1)
    )
    return (
        f"{rendered_questions}\n参考答案：{answer}\n\n候选片段（共 {len(batch)} 个）：\n\n{joined}"
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _json_payload(reply: str) -> object:
    """Decode one JSON value, tolerating only a single surrounding code fence.

    A greedy ``{.*}`` extractor accepts arbitrary prose and can swallow two JSON
    objects into one malformed region. Chat JSON mode should already return a
    bare object, so fail closed on anything else; the fence exception handles a
    common harmless model formatting slip without guessing where JSON starts.
    """
    text = reply.strip()
    fence = _JSON_FENCE.fullmatch(text)
    if fence is not None:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc


def parse_judgements(reply: str, batch: Sequence[str]) -> dict[str, int]:
    """Map a judging reply back onto chunk ids, failing closed on any mismatch.

    A short reply that grades three of eight passages is the expensive failure
    here: the five it skipped would become implicit zeroes and depress every
    system that retrieved them. Every requested number must come back exactly
    once, or the batch is rejected and re-requested.
    """
    payload = _json_payload(reply)
    if not isinstance(payload, dict):
        raise ValueError("reply is not a JSON object")
    rows = payload.get("judgements")
    if not isinstance(rows, list):
        raise ValueError("reply has no judgements array")

    grades: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("judgement is not an object")
        number = row.get("id")
        grade = row.get("grade")
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError("judgement id must be an integer")
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise ValueError(f"judgement {number}: grade must be an integer")
        if grade not in GRADE_LABELS:
            raise ValueError(f"judgement {number}: grade {grade} is outside {sorted(GRADE_LABELS)}")
        if number in grades:
            raise ValueError(f"judgement {number}: repeated")
        grades[number] = grade

    expected = set(range(1, len(batch) + 1))
    if set(grades) != expected:
        missing = sorted(expected - set(grades))
        extra = sorted(set(grades) - expected)
        raise ValueError(f"judgements do not cover the batch: missing={missing} extra={extra}")
    return {doc_id: grades[number] for number, doc_id in enumerate(batch, start=1)}


def qrels_rows(
    judged: Iterable[JudgedQuery],
    *,
    queries: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Expand judged pairs into one qrels row per query variant.

    Both variants of a pair inherit one judgement set. They keep their own
    question text and their own leakage statistic, so the paired direct-versus-
    paraphrase contrast stays intact.
    """
    rows: list[dict[str, object]] = []
    for unit in judged:
        gold = list(unit.gold_doc_ids)
        partial = list(unit.partial_doc_ids)
        judged_ids = list(unit.judged_doc_ids)
        for query_id in unit.query_ids:
            source = queries.get(query_id)
            if source is None:
                raise KeyError(f"no generated query for {query_id!r}")
            row = dict(source)
            row["gold_doc_ids"] = gold
            row["partial_doc_ids"] = partial
            row["judged_doc_ids"] = judged_ids
            row["generating_chunk_id"] = unit.chunk_id
            row["generating_chunk_grade"] = unit.generator_chunk_grade
            rows.append(row)
    rows.sort(key=lambda row: str(row["query_id"]))
    return rows
