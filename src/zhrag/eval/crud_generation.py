"""Offline-first contracts for the CRUD-RAG generation experiment.

The module deliberately has no file, environment, provider, or optional-model
access. It turns the two supported CRUD records into immutable cases, frames the
three prompt stages, validates append-only cache rows against exact universes,
and rebuilds aggregate-only numeric artifacts from complete in-memory inputs.

RAGQuestEval follows the CRUD-RAG paper's one-sided construction: questions are
generated only from the ground-truth reference, then answered against both that
reference and the evaluated prediction. The source article is a generation input
and must never enter either scoring context.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal, cast

from zhrag.eval.metrics import BootstrapCI, clustered_bootstrap_ci
from zhrag.eval.metrics_gen import (
    BERTSCORE_METRIC,
    CRUD_BLEU_METRIC,
    ROUGE_L_METRIC,
    STANDARD_BLEU_METRIC,
    GenerationExample,
    SemanticProvenance,
    Tokenizer,
    TokenizerProvenance,
    evaluate_generation,
)
from zhrag.eval.quest_eval import (
    UNANSWERABLE_SENTINEL,
    QuestAnswerPair,
    evaluate_quest_answers,
)
from zhrag.tokens import estimate_tokens

__all__ = [
    "CACHE_ROW_SCHEMA",
    "CHAT_PROFILE_CONTRACT_SHA256",
    "CHAT_PROFILE_SCHEMA",
    "GENERATION_CASE_KEY_SCHEMA",
    "GENERATION_EXPERIMENT_SCHEMA",
    "GENERATION_REPORT_SCHEMA",
    "GENERATION_SAMPLES_SCHEMA",
    "M9B_CONTRACT_VERSION",
    "M9B_TASKS",
    "MAX_INPUT_ESTIMATED_TOKENS",
    "MAX_QUESTIONS_PER_CASE",
    "PUBLIC_INPUT_FINGERPRINT_KEYS",
    "QUESTION_BANK_SCHEMA",
    "REFERENCE_BANK_SCHEMA",
    "GeneratedQuestion",
    "GenerationCacheRow",
    "GenerationCase",
    "GenerationInputManifest",
    "QACacheRow",
    "QGCacheRow",
    "QuestionBank",
    "ReferenceBank",
    "SemanticCacheRow",
    "TaskName",
    "build_generation_cases",
    "build_generation_report",
    "build_input_manifest",
    "build_numeric_samples",
    "build_quest_answer_pairs",
    "build_question_bank",
    "build_reference_bank",
    "cache_rows_fingerprint",
    "canonical_fingerprint",
    "dataset_snapshot_sha256",
    "generation_cache_id",
    "generation_instructions_fingerprint",
    "generation_system_prompt",
    "generation_user_prompt",
    "numeric_samples_sha256",
    "parse_generation_cache_row",
    "parse_generation_input_manifest",
    "parse_generation_response",
    "parse_qa_cache_row",
    "parse_qa_response",
    "parse_qg_cache_row",
    "parse_question_bank",
    "parse_questions_response",
    "parse_reference_bank",
    "parse_semantic_cache_row",
    "prediction_qa_cache_id",
    "qa_instructions_fingerprint",
    "qa_system_prompt",
    "qa_user_prompt",
    "qg_cache_id",
    "qg_instructions_fingerprint",
    "qg_system_prompt",
    "qg_user_prompt",
    "question_id",
    "reference_qa_cache_id",
    "semantic_cache_id",
    "validate_generation_report",
    "validate_generation_rows",
    "validate_numeric_samples",
    "validate_prediction_qa_rows",
    "validate_prompt_budget",
    "validate_qg_rows",
    "validate_reference_qa_rows",
    "validate_semantic_rows",
]

M9B_CONTRACT_VERSION = "m9b1-known-context-v1"
GENERATION_EXPERIMENT_SCHEMA = "zhrag-crud-generation-v1"
GENERATION_CASE_KEY_SCHEMA = "zhrag-crud-generation-case-key-v1"
CACHE_ROW_SCHEMA = "zhrag-crud-generation-cache-row-v1"
QUESTION_BANK_SCHEMA = "zhrag-crud-question-bank-v1"
REFERENCE_BANK_SCHEMA = "zhrag-crud-reference-bank-v1"
GENERATION_SAMPLES_SCHEMA = "zhrag-crud-generation-samples-v2"
GENERATION_REPORT_SCHEMA = "zhrag-crud-generation-report-v2"
CHAT_PROFILE_SCHEMA = "zhrag-crud-chat-profile-v2"
CHAT_PROFILE_CONTRACT_SHA256 = hashlib.sha256(
    b"zhrag-crud-chat-profile-contract-v1\0" + CHAT_PROFILE_SCHEMA.encode("ascii")
).hexdigest()

type TaskName = Literal["event_summary", "questanswer_1doc"]
M9B_TASKS: tuple[TaskName, ...] = ("event_summary", "questanswer_1doc")
type QALane = Literal["reference", "prediction"]
type SentinelMatch = Literal["none", "exact", "normalized", "near"]

MAX_QUESTIONS_PER_CASE = 8
# This is an estimator guard, not a tokenizer proof. It deliberately leaves room
# for output inside providers whose advertised context window is 32k tokens.
MAX_INPUT_ESTIMATED_TOKENS = 30_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 0

_TASK_REQUESTS: dict[TaskName, str] = {
    "event_summary": "请根据给定新闻正文生成简明、准确的中文摘要。",
    "questanswer_1doc": "请仅根据给定新闻正文回答问题。",
}

_SUMMARY_GENERATION_INSTRUCTIONS = """你在执行中文新闻摘要任务。
只使用用户提供的新闻正文，不补充正文之外的事实。输出简明、连贯的摘要。
只输出 JSON 对象，且只能有一个字段：{"text": "摘要"}。"""

_QA_GENERATION_INSTRUCTIONS = f"""你在执行中文单文档问答任务。
只使用用户提供的新闻正文回答问题；若正文无法支持答案，回答“{UNANSWERABLE_SENTINEL}”。
只输出 JSON 对象，且只能有一个字段：{{"answer": "答案"}}。"""

_QG_INSTRUCTIONS = f"""你在为 RAGQuestEval 从 ground-truth reference 生成问题。
问题必须只依据 reference 中的关键信息，答案应能从 reference 中得到。
不要使用外部知识，不要询问写作风格，不要输出答案。
生成 1 到 {MAX_QUESTIONS_PER_CASE} 个互不重复的中文问题。
只输出 JSON 对象，且只能有一个字段：{{"questions": ["问题"]}}。"""

_QA_JUDGE_INSTRUCTIONS = f"""你在回答 RAGQuestEval 的一个问题。
只能依据用户给出的 context；context 不足以回答时，必须精确输出“{UNANSWERABLE_SENTINEL}”。
回答应简短，不解释判断过程。只输出 JSON 对象，且只能有一个字段：{{"answer": "答案"}}。"""

_WHITESPACE = re.compile(r"\s+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_PUBLIC_TASKS = set(M9B_TASKS)
_GENERATION_METRIC_NAMES = (
    STANDARD_BLEU_METRIC,
    CRUD_BLEU_METRIC,
    "mean_sentence_rouge_l_precision",
    "mean_sentence_rouge_l_recall",
    ROUGE_L_METRIC,
    "mean_bertscore_precision_zh_rescaled",
    "mean_bertscore_recall_zh_rescaled",
    BERTSCORE_METRIC,
)
_RAG_METRIC_NAMES = (
    "paper_recall_all_questions",
    "paper_precision_all_questions",
    "code_recall_reference_answerable",
    "code_precision_generated_answerable",
)
PUBLIC_INPUT_FINGERPRINT_KEYS = frozenset(
    {
        "chat_profile_contract_sha256",
        "dataset_snapshot_sha256",
        "input_manifest_sha256",
        "generation_model_profile_sha256",
        "qg_model_profile_sha256",
        "qa_model_profile_sha256",
        "semantic_profile_sha256",
        "generation_cache_sha256",
        "question_bank_sha256",
        "reference_bank_sha256",
        "prediction_qa_cache_sha256",
        "semantic_cache_sha256",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answers",
        "chunk",
        "chunk_id",
        "content",
        "context",
        "doc_id",
        "document_id",
        "embedding",
        "passage",
        "payload",
        "prediction",
        "provider_payload",
        "question",
        "questions",
        "query",
        "query_id",
        "reference",
        "rerank_score",
        "source",
        "source_id",
        "text",
        "vector",
    }
)
_LIMITATIONS = (
    "known-context generation profile; no retrieval stage is evaluated",
    "not a reproduction of CRUD-RAG Table 8 and not directly comparable to its rows",
    "questions are generated only from each exact ground-truth reference",
    "single-profile descriptive confidence intervals; no ablation hypothesis tests",
)


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _clean_text(value: object, *, label: str) -> str:
    return _require_text(value, label=label).strip()


def _clean_question(value: object, *, label: str) -> str:
    return _WHITESPACE.sub(" ", _clean_text(value, label=label))


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, label: str, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if unit_interval and not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    actual = set(cast(Mapping[str, object], value))
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ValueError(f"{label} keys drift: missing={missing}, extra={extra}")
    return cast(Mapping[str, Any], value)


def _json_object(raw: str, *, keys: set[str], label: str) -> Mapping[str, Any]:
    _require_text(raw, label=f"{label} response")
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} response is malformed JSON: {exc.msg}") from exc
    return _exact_mapping(parsed, keys=keys, label=f"{label} response")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_fingerprint(schema: str, *values: str) -> str:
    """Length-frame strings before hashing so concatenation is unambiguous."""

    _require_text(schema, label="fingerprint schema")
    digest = hashlib.sha256()
    for value in (schema, *values):
        _require_text(value, label="fingerprint value")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def dataset_snapshot_sha256(raw_bytes: bytes) -> str:
    if not isinstance(raw_bytes, bytes) or not raw_bytes:
        raise ValueError("dataset snapshot bytes must be non-empty")
    return hashlib.sha256(raw_bytes).hexdigest()


def generation_instructions_fingerprint(task: TaskName) -> str:
    return canonical_fingerprint(
        "zhrag-crud-generation-instructions-v1",
        task,
        generation_system_prompt(task),
    )


def qg_instructions_fingerprint() -> str:
    return canonical_fingerprint("zhrag-crud-qg-instructions-v1", _QG_INSTRUCTIONS)


def qa_instructions_fingerprint() -> str:
    return canonical_fingerprint("zhrag-crud-qa-instructions-v1", _QA_JUDGE_INSTRUCTIONS)


@dataclass(frozen=True, slots=True)
class GenerationCase:
    """One local, text-bearing generation example."""

    case_key: str
    source_id: str
    task: TaskName
    generation_context: str
    user_request: str
    reference: str

    def __post_init__(self) -> None:
        _require_sha256(self.case_key, label="case_key")
        _require_text(self.source_id, label=f"{self.case_key}: source_id")
        if self.task not in M9B_TASKS:
            raise ValueError(f"unsupported generation task {self.task!r}")
        for name in ("generation_context", "user_request", "reference"):
            _require_text(getattr(self, name), label=f"{self.case_key}: {name}")


@dataclass(frozen=True, slots=True)
class GenerationInputManifest:
    schema: str
    contract: str
    dataset_snapshot_sha256: str
    tasks: tuple[TaskName, ...]
    case_count: int
    ordered_case_fingerprint: str
    generation_universe_fingerprint: str
    reference_universe_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != GENERATION_EXPERIMENT_SCHEMA or self.contract != M9B_CONTRACT_VERSION:
            raise ValueError("generation input manifest schema drift")
        _require_sha256(self.dataset_snapshot_sha256, label="dataset_snapshot_sha256")
        if self.tasks != M9B_TASKS:
            raise ValueError("generation input manifest task order drift")
        _require_int(self.case_count, label="case_count", minimum=2)
        for name in (
            "ordered_case_fingerprint",
            "generation_universe_fingerprint",
            "reference_universe_fingerprint",
        ):
            _require_sha256(getattr(self, name), label=name)

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        result["tasks"] = list(self.tasks)
        return result


def parse_generation_input_manifest(raw: object) -> GenerationInputManifest:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "contract",
            "dataset_snapshot_sha256",
            "tasks",
            "case_count",
            "ordered_case_fingerprint",
            "generation_universe_fingerprint",
            "reference_universe_fingerprint",
        },
        label="generation input manifest",
    )
    tasks = row["tasks"]
    if tasks != list(M9B_TASKS):
        raise ValueError("generation input manifest task order drift")
    return GenerationInputManifest(
        schema=row["schema"],
        contract=row["contract"],
        dataset_snapshot_sha256=row["dataset_snapshot_sha256"],
        tasks=M9B_TASKS,
        case_count=row["case_count"],
        ordered_case_fingerprint=row["ordered_case_fingerprint"],
        generation_universe_fingerprint=row["generation_universe_fingerprint"],
        reference_universe_fingerprint=row["reference_universe_fingerprint"],
    )


def _case_key(task: TaskName, source_id: str) -> str:
    return canonical_fingerprint(GENERATION_CASE_KEY_SCHEMA, task, source_id)


def build_generation_cases(
    raw: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[GenerationCase, ...]:
    """Strictly load the two M9b tasks without carrying unrelated record fields."""

    if not isinstance(raw, Mapping):
        raise ValueError("CRUD generation input must be a mapping")
    cases: list[GenerationCase] = []
    for task in M9B_TASKS:
        records = raw.get(task)
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
            raise ValueError(f"{task} must be a non-empty record sequence")
        task_name = task
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"{task}[{index}] must be an object")
            source_id = _clean_text(record.get("ID"), label=f"{task}[{index}].ID")
            if task_name == "event_summary":
                context = _clean_text(record.get("text"), label=f"{task}[{index}].text")
                reference = _clean_text(
                    record.get("summary"),
                    label=f"{task}[{index}].summary",
                )
                request = _TASK_REQUESTS[task_name]
            else:
                context = _clean_text(record.get("news1"), label=f"{task}[{index}].news1")
                reference = _clean_text(
                    record.get("answers"),
                    label=f"{task}[{index}].answers",
                )
                question = _clean_question(
                    record.get("questions"),
                    label=f"{task}[{index}].questions",
                )
                request = f"{_TASK_REQUESTS[task_name]}\n问题：{question}"
            cases.append(
                GenerationCase(
                    case_key=_case_key(task_name, source_id),
                    source_id=source_id,
                    task=task_name,
                    generation_context=context,
                    user_request=request,
                    reference=reference,
                )
            )
    keys = [case.case_key for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("generation case keys must be unique")
    return tuple(cases)


def build_input_manifest(
    cases: Sequence[GenerationCase],
    *,
    dataset_sha256: str,
) -> GenerationInputManifest:
    checked = _checked_cases(cases)
    snapshot = _require_sha256(dataset_sha256, label="dataset_sha256")
    tasks = tuple(dict.fromkeys(case.task for case in checked))
    if tasks != M9B_TASKS:
        raise ValueError("generation cases must contain both tasks in frozen order")
    return GenerationInputManifest(
        schema=GENERATION_EXPERIMENT_SCHEMA,
        contract=M9B_CONTRACT_VERSION,
        dataset_snapshot_sha256=snapshot,
        tasks=M9B_TASKS,
        case_count=len(checked),
        ordered_case_fingerprint=canonical_fingerprint(
            "zhrag-crud-ordered-cases-v1",
            *(case.case_key for case in checked),
        ),
        generation_universe_fingerprint=canonical_fingerprint(
            "zhrag-crud-generation-universe-v1",
            *(
                canonical_fingerprint(
                    "zhrag-crud-generation-input-v1",
                    case.case_key,
                    case.task,
                    case.generation_context,
                    case.user_request,
                )
                for case in checked
            ),
        ),
        reference_universe_fingerprint=canonical_fingerprint(
            "zhrag-crud-reference-universe-v1",
            *(
                canonical_fingerprint(
                    "zhrag-crud-reference-input-v1",
                    case.case_key,
                    case.task,
                    case.reference,
                )
                for case in checked
            ),
        ),
    )


def _checked_cases(cases: Sequence[GenerationCase]) -> tuple[GenerationCase, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence) or not cases:
        raise ValueError("generation cases must be a non-empty sequence")
    checked: list[GenerationCase] = []
    for index, case in enumerate(cases):
        if not isinstance(case, GenerationCase):
            raise ValueError(f"cases[{index}] must be a GenerationCase")
        checked.append(case)
    if len({case.case_key for case in checked}) != len(checked):
        raise ValueError("generation case keys must be unique")
    return tuple(checked)


def generation_system_prompt(task: TaskName) -> str:
    if task == "event_summary":
        return _SUMMARY_GENERATION_INSTRUCTIONS
    if task == "questanswer_1doc":
        return _QA_GENERATION_INSTRUCTIONS
    raise ValueError(f"unsupported generation task {task!r}")


def generation_user_prompt(case: GenerationCase) -> str:
    return (
        "generation_context:\n<<<\n"
        f"{case.generation_context}\n"
        ">>>\n\n"
        "user_request:\n<<<\n"
        f"{case.user_request}\n"
        ">>>"
    )


def qg_system_prompt() -> str:
    return _QG_INSTRUCTIONS


def qg_user_prompt(case: GenerationCase) -> str:
    """Frame only the reference; source context and prediction are unavailable."""

    return f"ground_truth_reference:\n<<<\n{case.reference}\n>>>"


def qa_system_prompt() -> str:
    return _QA_JUDGE_INSTRUCTIONS


def qa_user_prompt(question: str, context: str) -> str:
    checked_question = _clean_question(question, label="QA question")
    checked_context = _clean_text(context, label="QA context")
    return f"question:\n<<<\n{checked_question}\n>>>\n\ncontext:\n<<<\n{checked_context}\n>>>"


def validate_prompt_budget(
    system: str,
    user: str,
    *,
    maximum: int = MAX_INPUT_ESTIMATED_TOKENS,
) -> int:
    _require_text(system, label="system prompt")
    _require_text(user, label="user prompt")
    _require_int(maximum, label="maximum prompt tokens", minimum=1)
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > maximum:
        raise ValueError(
            f"estimated prompt length {estimated} exceeds frozen input cap {maximum}; "
            "refusing to truncate"
        )
    return estimated


def parse_generation_response(task: TaskName, raw: str) -> str:
    key = "text" if task == "event_summary" else "answer"
    if task not in M9B_TASKS:
        raise ValueError(f"unsupported generation task {task!r}")
    payload = _json_object(raw, keys={key}, label=f"{task} generation")
    return _clean_text(payload[key], label=f"{task} generation.{key}")


def parse_questions_response(raw: str) -> tuple[str, ...]:
    payload = _json_object(raw, keys={"questions"}, label="question generation")
    values = payload["questions"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("question generation.questions must be a list")
    if not 1 <= len(values) <= MAX_QUESTIONS_PER_CASE:
        raise ValueError(f"question generation must return 1..{MAX_QUESTIONS_PER_CASE} questions")
    questions = tuple(
        _clean_question(value, label=f"question generation.questions[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(questions)) != len(questions):
        raise ValueError("question generation questions must be unique after whitespace cleanup")
    return questions


def parse_qa_response(raw: str) -> str:
    payload = _json_object(raw, keys={"answer"}, label="QA judge")
    return _clean_text(payload["answer"], label="QA judge.answer")


def _usage_values(
    prompt_tokens: object,
    completion_tokens: object,
    reasoning_tokens: object,
    *,
    label: str,
) -> tuple[int | None, int | None, int | None]:
    values = (prompt_tokens, completion_tokens, reasoning_tokens)
    if values == (None, None, None):
        return None, None, None
    if any(value is None for value in values):
        raise ValueError(f"{label} token usage must be complete or entirely unknown")
    return cast(
        tuple[int, int, int],
        tuple(_require_int(value, label=f"{label} token usage", minimum=0) for value in values),
    )


def generation_cache_id(case: GenerationCase, model_profile_fingerprint: str) -> str:
    profile = _require_sha256(model_profile_fingerprint, label="generation model profile")
    return canonical_fingerprint(
        "zhrag-crud-generation-cache-key-v1",
        case.case_key,
        case.task,
        case.generation_context,
        case.user_request,
        generation_instructions_fingerprint(case.task),
        profile,
    )


def qg_cache_id(case: GenerationCase, model_profile_fingerprint: str) -> str:
    profile = _require_sha256(model_profile_fingerprint, label="QG model profile")
    return canonical_fingerprint(
        "zhrag-crud-qg-cache-key-v1",
        case.case_key,
        case.task,
        case.reference,
        qg_instructions_fingerprint(),
        profile,
    )


def question_id(case: GenerationCase, ordinal: int, question: str) -> str:
    _require_int(ordinal, label="question ordinal", minimum=0)
    checked = _clean_question(question, label="question")
    return canonical_fingerprint(
        "zhrag-crud-question-id-v1",
        case.case_key,
        case.reference,
        qg_instructions_fingerprint(),
        str(ordinal),
        checked,
    )


@dataclass(frozen=True, slots=True)
class GeneratedQuestion:
    question_id: str
    case_key: str
    ordinal: int
    text: str

    def __post_init__(self) -> None:
        _require_sha256(self.question_id, label="question_id")
        _require_sha256(self.case_key, label="question case_key")
        _require_int(self.ordinal, label="question ordinal", minimum=0)
        _require_text(self.text, label="question text")

    def as_json(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "case_key": self.case_key,
            "ordinal": self.ordinal,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class GenerationCacheRow:
    schema: str
    cache_id: str
    case_key: str
    task: TaskName
    prediction: str
    served_model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.schema != CACHE_ROW_SCHEMA:
            raise ValueError("generation cache row schema drift")
        _require_sha256(self.cache_id, label="generation cache_id")
        _require_sha256(self.case_key, label="generation case_key")
        if self.task not in M9B_TASKS:
            raise ValueError(f"unsupported generation cache task {self.task!r}")
        _require_text(self.prediction, label="generation prediction")
        _require_text(self.served_model, label="generation served_model")
        _usage_values(
            self.prompt_tokens,
            self.completion_tokens,
            self.reasoning_tokens,
            label="generation",
        )

    def as_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QGCacheRow:
    schema: str
    cache_id: str
    case_key: str
    questions: tuple[GeneratedQuestion, ...]
    served_model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.schema != CACHE_ROW_SCHEMA:
            raise ValueError("QG cache row schema drift")
        _require_sha256(self.cache_id, label="QG cache_id")
        _require_sha256(self.case_key, label="QG case_key")
        if not 1 <= len(self.questions) <= MAX_QUESTIONS_PER_CASE:
            raise ValueError("QG row question count is outside the frozen range")
        if any(question.case_key != self.case_key for question in self.questions):
            raise ValueError("QG row questions must belong to its case")
        if tuple(question.ordinal for question in self.questions) != tuple(
            range(len(self.questions))
        ):
            raise ValueError("QG row question ordinals must be contiguous")
        if len({question.question_id for question in self.questions}) != len(self.questions):
            raise ValueError("QG row question ids must be unique")
        if len({question.text for question in self.questions}) != len(self.questions):
            raise ValueError("QG row question texts must be unique")
        _require_text(self.served_model, label="QG served_model")
        _usage_values(
            self.prompt_tokens,
            self.completion_tokens,
            self.reasoning_tokens,
            label="QG",
        )

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        result["questions"] = [question.as_json() for question in self.questions]
        return result


@dataclass(frozen=True, slots=True)
class QACacheRow:
    schema: str
    cache_id: str
    case_key: str
    question_id: str
    lane: QALane
    answer: str
    served_model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.schema != CACHE_ROW_SCHEMA:
            raise ValueError("QA cache row schema drift")
        for name in ("cache_id", "case_key", "question_id"):
            _require_sha256(getattr(self, name), label=f"QA {name}")
        if self.lane not in ("reference", "prediction"):
            raise ValueError(f"unknown QA lane {self.lane!r}")
        _require_text(self.answer, label="QA answer")
        _require_text(self.served_model, label="QA served_model")
        _usage_values(
            self.prompt_tokens,
            self.completion_tokens,
            self.reasoning_tokens,
            label="QA",
        )

    def as_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticCacheRow:
    schema: str
    cache_id: str
    case_key: str
    precision: float
    recall: float
    f1: float

    def __post_init__(self) -> None:
        if self.schema != CACHE_ROW_SCHEMA:
            raise ValueError("semantic cache row schema drift")
        _require_sha256(self.cache_id, label="semantic cache_id")
        _require_sha256(self.case_key, label="semantic case_key")
        for name in ("precision", "recall", "f1"):
            _finite(getattr(self, name), label=f"semantic {name}")

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def _parse_usage(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int | None, int | None, int | None]:
    return _usage_values(
        row["prompt_tokens"],
        row["completion_tokens"],
        row["reasoning_tokens"],
        label=label,
    )


def parse_generation_cache_row(raw: object) -> GenerationCacheRow:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "cache_id",
            "case_key",
            "task",
            "prediction",
            "served_model",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        },
        label="generation cache row",
    )
    usage = _parse_usage(row, label="generation")
    return GenerationCacheRow(
        schema=row["schema"],
        cache_id=row["cache_id"],
        case_key=row["case_key"],
        task=row["task"],
        prediction=row["prediction"],
        served_model=row["served_model"],
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        reasoning_tokens=usage[2],
    )


def _parse_question(raw: object, *, label: str) -> GeneratedQuestion:
    row = _exact_mapping(
        raw,
        keys={"question_id", "case_key", "ordinal", "text"},
        label=label,
    )
    return GeneratedQuestion(
        question_id=row["question_id"],
        case_key=row["case_key"],
        ordinal=row["ordinal"],
        text=row["text"],
    )


def parse_qg_cache_row(raw: object) -> QGCacheRow:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "cache_id",
            "case_key",
            "questions",
            "served_model",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        },
        label="QG cache row",
    )
    raw_questions = row["questions"]
    if isinstance(raw_questions, (str, bytes)) or not isinstance(raw_questions, Sequence):
        raise ValueError("QG cache row questions must be a list")
    questions = tuple(
        _parse_question(value, label=f"QG cache question {index}")
        for index, value in enumerate(raw_questions)
    )
    usage = _parse_usage(row, label="QG")
    return QGCacheRow(
        schema=row["schema"],
        cache_id=row["cache_id"],
        case_key=row["case_key"],
        questions=questions,
        served_model=row["served_model"],
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        reasoning_tokens=usage[2],
    )


def parse_qa_cache_row(raw: object) -> QACacheRow:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "cache_id",
            "case_key",
            "question_id",
            "lane",
            "answer",
            "served_model",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
        },
        label="QA cache row",
    )
    usage = _parse_usage(row, label="QA")
    return QACacheRow(
        schema=row["schema"],
        cache_id=row["cache_id"],
        case_key=row["case_key"],
        question_id=row["question_id"],
        lane=row["lane"],
        answer=row["answer"],
        served_model=row["served_model"],
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        reasoning_tokens=usage[2],
    )


def parse_semantic_cache_row(raw: object) -> SemanticCacheRow:
    row = _exact_mapping(
        raw,
        keys={"schema", "cache_id", "case_key", "precision", "recall", "f1"},
        label="semantic cache row",
    )
    return SemanticCacheRow(
        schema=row["schema"],
        cache_id=row["cache_id"],
        case_key=row["case_key"],
        precision=row["precision"],
        recall=row["recall"],
        f1=row["f1"],
    )


def _validate_universe[RowT](
    rows: Sequence[RowT],
    expected_ids: Sequence[str],
    *,
    row_id: Callable[[RowT], str],
    label: str,
    require_complete: bool,
) -> dict[str, RowT]:
    expected = tuple(expected_ids)
    if len(set(expected)) != len(expected):
        raise ValueError(f"{label} expected ids must be unique")
    expected_set = set(expected)
    result: dict[str, RowT] = {}
    for row in rows:
        key = row_id(row)
        if key in result:
            raise ValueError(f"{label} contains duplicate cache id {key}")
        if key not in expected_set:
            raise ValueError(f"{label} contains unknown cache id {key}")
        result[key] = row
    if require_complete and set(result) != expected_set:
        missing = len(expected_set - set(result))
        raise ValueError(f"{label} is incomplete: {missing} expected rows are missing")
    return result


def validate_generation_rows(
    cases: Sequence[GenerationCase],
    rows: Sequence[GenerationCacheRow],
    *,
    model_profile_fingerprint: str,
    require_complete: bool = False,
) -> dict[str, GenerationCacheRow]:
    checked = _checked_cases(cases)
    expected = [generation_cache_id(case, model_profile_fingerprint) for case in checked]
    by_id = _validate_universe(
        rows,
        expected,
        row_id=lambda row: row.cache_id,
        label="generation cache",
        require_complete=require_complete,
    )
    case_by_id = dict(zip(expected, checked, strict=True))
    for cache_id, row in by_id.items():
        case = case_by_id[cache_id]
        if row.case_key != case.case_key or row.task != case.task:
            raise ValueError("generation cache row identity drift")
    return by_id


def validate_qg_rows(
    cases: Sequence[GenerationCase],
    rows: Sequence[QGCacheRow],
    *,
    model_profile_fingerprint: str,
    require_complete: bool = False,
) -> dict[str, QGCacheRow]:
    checked = _checked_cases(cases)
    expected = [qg_cache_id(case, model_profile_fingerprint) for case in checked]
    by_id = _validate_universe(
        rows,
        expected,
        row_id=lambda row: row.cache_id,
        label="QG cache",
        require_complete=require_complete,
    )
    case_by_id = dict(zip(expected, checked, strict=True))
    for cache_id, row in by_id.items():
        case = case_by_id[cache_id]
        if row.case_key != case.case_key:
            raise ValueError("QG cache row case identity drift")
        for question in row.questions:
            if question.question_id != question_id(case, question.ordinal, question.text):
                raise ValueError("QG cache question identity drift")
    return by_id


def _ordered_questions(
    cases: Sequence[GenerationCase],
    qg_rows: Sequence[QGCacheRow],
    *,
    qg_model_profile_fingerprint: str,
) -> tuple[GeneratedQuestion, ...]:
    checked = _checked_cases(cases)
    by_id = validate_qg_rows(
        checked,
        qg_rows,
        model_profile_fingerprint=qg_model_profile_fingerprint,
        require_complete=True,
    )
    return tuple(
        question
        for case in checked
        for question in by_id[qg_cache_id(case, qg_model_profile_fingerprint)].questions
    )


def reference_qa_cache_id(
    case: GenerationCase,
    question: GeneratedQuestion,
    model_profile_fingerprint: str,
) -> str:
    if question.case_key != case.case_key:
        raise ValueError("reference QA question belongs to another case")
    profile = _require_sha256(model_profile_fingerprint, label="QA model profile")
    return canonical_fingerprint(
        "zhrag-crud-reference-qa-cache-key-v1",
        case.case_key,
        question.question_id,
        question.text,
        case.reference,
        qa_instructions_fingerprint(),
        profile,
    )


def prediction_qa_cache_id(
    case: GenerationCase,
    question: GeneratedQuestion,
    prediction: str,
    model_profile_fingerprint: str,
) -> str:
    if question.case_key != case.case_key:
        raise ValueError("prediction QA question belongs to another case")
    profile = _require_sha256(model_profile_fingerprint, label="QA model profile")
    return canonical_fingerprint(
        "zhrag-crud-prediction-qa-cache-key-v1",
        case.case_key,
        question.question_id,
        question.text,
        _clean_text(prediction, label="prediction QA context"),
        qa_instructions_fingerprint(),
        profile,
    )


def _validate_qa_rows(
    cases: Sequence[GenerationCase],
    questions: Sequence[GeneratedQuestion],
    rows: Sequence[QACacheRow],
    *,
    lane: QALane,
    model_profile_fingerprint: str,
    predictions: Mapping[str, str] | None,
    require_complete: bool,
) -> dict[str, QACacheRow]:
    checked = _checked_cases(cases)
    case_by_key = {case.case_key: case for case in checked}
    expected: list[str] = []
    identity: dict[str, tuple[GenerationCase, GeneratedQuestion]] = {}
    for question in questions:
        case = case_by_key.get(question.case_key)
        if case is None:
            raise ValueError("QA question belongs to an unknown case")
        if lane == "reference":
            cache_id = reference_qa_cache_id(case, question, model_profile_fingerprint)
        else:
            if predictions is None or case.case_key not in predictions:
                raise ValueError("prediction QA requires every exact case prediction")
            cache_id = prediction_qa_cache_id(
                case,
                question,
                predictions[case.case_key],
                model_profile_fingerprint,
            )
        expected.append(cache_id)
        identity[cache_id] = (case, question)
    by_id = _validate_universe(
        rows,
        expected,
        row_id=lambda row: row.cache_id,
        label=f"{lane} QA cache",
        require_complete=require_complete,
    )
    for cache_id, row in by_id.items():
        case, question = identity[cache_id]
        if (
            row.lane != lane
            or row.case_key != case.case_key
            or row.question_id != question.question_id
        ):
            raise ValueError(f"{lane} QA cache row identity drift")
    return by_id


def validate_reference_qa_rows(
    cases: Sequence[GenerationCase],
    questions: Sequence[GeneratedQuestion],
    rows: Sequence[QACacheRow],
    *,
    model_profile_fingerprint: str,
    require_complete: bool = False,
) -> dict[str, QACacheRow]:
    return _validate_qa_rows(
        cases,
        questions,
        rows,
        lane="reference",
        model_profile_fingerprint=model_profile_fingerprint,
        predictions=None,
        require_complete=require_complete,
    )


def validate_prediction_qa_rows(
    cases: Sequence[GenerationCase],
    questions: Sequence[GeneratedQuestion],
    rows: Sequence[QACacheRow],
    *,
    model_profile_fingerprint: str,
    predictions: Mapping[str, str],
    require_complete: bool = False,
) -> dict[str, QACacheRow]:
    return _validate_qa_rows(
        cases,
        questions,
        rows,
        lane="prediction",
        model_profile_fingerprint=model_profile_fingerprint,
        predictions=predictions,
        require_complete=require_complete,
    )


def semantic_cache_id(
    case: GenerationCase,
    prediction: str,
    semantic_profile_fingerprint: str,
) -> str:
    profile = _require_sha256(semantic_profile_fingerprint, label="semantic profile")
    return canonical_fingerprint(
        "zhrag-crud-semantic-cache-key-v1",
        case.case_key,
        _clean_text(prediction, label="semantic prediction"),
        case.reference,
        profile,
    )


def validate_semantic_rows(
    cases: Sequence[GenerationCase],
    rows: Sequence[SemanticCacheRow],
    *,
    predictions: Mapping[str, str],
    semantic_profile_fingerprint: str,
    require_complete: bool = False,
) -> dict[str, SemanticCacheRow]:
    checked = _checked_cases(cases)
    expected: list[str] = []
    case_by_id: dict[str, GenerationCase] = {}
    for case in checked:
        if case.case_key not in predictions:
            raise ValueError("semantic cache requires every exact case prediction")
        cache_id = semantic_cache_id(
            case,
            predictions[case.case_key],
            semantic_profile_fingerprint,
        )
        expected.append(cache_id)
        case_by_id[cache_id] = case
    by_id = _validate_universe(
        rows,
        expected,
        row_id=lambda row: row.cache_id,
        label="semantic cache",
        require_complete=require_complete,
    )
    for cache_id, row in by_id.items():
        if row.case_key != case_by_id[cache_id].case_key:
            raise ValueError("semantic cache row case identity drift")
    return by_id


def cache_rows_fingerprint(rows: Sequence[object]) -> str:
    if not rows:
        raise ValueError("cannot fingerprint an empty cache")
    payloads: list[object] = []
    for row in rows:
        as_json = getattr(row, "as_json", None)
        if not callable(as_json):
            raise ValueError("cache rows must expose as_json()")
        payloads.append(cast(Callable[[], object], as_json)())
    ordered_payloads = sorted(
        payloads,
        key=lambda payload: str(cast(Mapping[str, object], payload).get("cache_id", "")),
    )
    return hashlib.sha256(
        b"zhrag-crud-cache-content-v1\0" + _canonical_json(ordered_payloads)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class QuestionBank:
    schema: str
    reference_universe_fingerprint: str
    qg_model_profile_fingerprint: str
    qg_cache_fingerprint: str
    question_universe_fingerprint: str
    cases: int
    questions: int

    def __post_init__(self) -> None:
        if self.schema != QUESTION_BANK_SCHEMA:
            raise ValueError("question bank schema drift")
        for name in (
            "reference_universe_fingerprint",
            "qg_model_profile_fingerprint",
            "qg_cache_fingerprint",
            "question_universe_fingerprint",
        ):
            _require_sha256(getattr(self, name), label=f"question bank {name}")
        _require_int(self.cases, label="question bank cases", minimum=1)
        _require_int(self.questions, label="question bank questions", minimum=1)

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def parse_question_bank(raw: object) -> QuestionBank:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "reference_universe_fingerprint",
            "qg_model_profile_fingerprint",
            "qg_cache_fingerprint",
            "question_universe_fingerprint",
            "cases",
            "questions",
        },
        label="question bank",
    )
    return QuestionBank(
        schema=row["schema"],
        reference_universe_fingerprint=row["reference_universe_fingerprint"],
        qg_model_profile_fingerprint=row["qg_model_profile_fingerprint"],
        qg_cache_fingerprint=row["qg_cache_fingerprint"],
        question_universe_fingerprint=row["question_universe_fingerprint"],
        cases=row["cases"],
        questions=row["questions"],
    )


def build_question_bank(
    cases: Sequence[GenerationCase],
    qg_rows: Sequence[QGCacheRow],
    *,
    reference_universe_fingerprint: str,
    qg_model_profile_fingerprint: str,
) -> QuestionBank:
    checked = _checked_cases(cases)
    reference_fp = _require_sha256(
        reference_universe_fingerprint,
        label="reference universe fingerprint",
    )
    profile = _require_sha256(qg_model_profile_fingerprint, label="QG model profile")
    questions = _ordered_questions(
        checked,
        qg_rows,
        qg_model_profile_fingerprint=profile,
    )
    return QuestionBank(
        schema=QUESTION_BANK_SCHEMA,
        reference_universe_fingerprint=reference_fp,
        qg_model_profile_fingerprint=profile,
        qg_cache_fingerprint=cache_rows_fingerprint(qg_rows),
        question_universe_fingerprint=canonical_fingerprint(
            "zhrag-crud-question-universe-v1",
            *(
                canonical_fingerprint(
                    "zhrag-crud-question-row-v1",
                    question.question_id,
                    question.case_key,
                    str(question.ordinal),
                    question.text,
                )
                for question in questions
            ),
        ),
        cases=len(checked),
        questions=len(questions),
    )


@dataclass(frozen=True, slots=True)
class ReferenceBank:
    schema: str
    question_universe_fingerprint: str
    qa_model_profile_fingerprint: str
    reference_qa_cache_fingerprint: str
    reference_answer_universe_fingerprint: str
    questions: int

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_BANK_SCHEMA:
            raise ValueError("reference bank schema drift")
        for name in (
            "question_universe_fingerprint",
            "qa_model_profile_fingerprint",
            "reference_qa_cache_fingerprint",
            "reference_answer_universe_fingerprint",
        ):
            _require_sha256(getattr(self, name), label=f"reference bank {name}")
        _require_int(self.questions, label="reference bank questions", minimum=1)

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def parse_reference_bank(raw: object) -> ReferenceBank:
    row = _exact_mapping(
        raw,
        keys={
            "schema",
            "question_universe_fingerprint",
            "qa_model_profile_fingerprint",
            "reference_qa_cache_fingerprint",
            "reference_answer_universe_fingerprint",
            "questions",
        },
        label="reference bank",
    )
    return ReferenceBank(
        schema=row["schema"],
        question_universe_fingerprint=row["question_universe_fingerprint"],
        qa_model_profile_fingerprint=row["qa_model_profile_fingerprint"],
        reference_qa_cache_fingerprint=row["reference_qa_cache_fingerprint"],
        reference_answer_universe_fingerprint=row["reference_answer_universe_fingerprint"],
        questions=row["questions"],
    )


def build_reference_bank(
    cases: Sequence[GenerationCase],
    questions: Sequence[GeneratedQuestion],
    rows: Sequence[QACacheRow],
    *,
    question_universe_fingerprint: str,
    qa_model_profile_fingerprint: str,
) -> ReferenceBank:
    question_fp = _require_sha256(
        question_universe_fingerprint,
        label="question universe fingerprint",
    )
    profile = _require_sha256(qa_model_profile_fingerprint, label="QA model profile")
    by_id = validate_reference_qa_rows(
        cases,
        questions,
        rows,
        model_profile_fingerprint=profile,
        require_complete=True,
    )
    ordered_rows = tuple(by_id[key] for key in sorted(by_id))
    return ReferenceBank(
        schema=REFERENCE_BANK_SCHEMA,
        question_universe_fingerprint=question_fp,
        qa_model_profile_fingerprint=profile,
        reference_qa_cache_fingerprint=cache_rows_fingerprint(ordered_rows),
        reference_answer_universe_fingerprint=canonical_fingerprint(
            "zhrag-crud-reference-answer-universe-v1",
            *(
                canonical_fingerprint(
                    "zhrag-crud-reference-answer-row-v1",
                    row.question_id,
                    row.answer,
                )
                for row in ordered_rows
            ),
        ),
        questions=len(questions),
    )


def build_quest_answer_pairs(
    questions: Sequence[GeneratedQuestion],
    reference_rows: Sequence[QACacheRow],
    prediction_rows: Sequence[QACacheRow],
) -> tuple[QuestAnswerPair, ...]:
    reference: dict[str, QACacheRow] = {}
    generated: dict[str, QACacheRow] = {}
    for rows, lane, target in (
        (reference_rows, "reference", reference),
        (prediction_rows, "prediction", generated),
    ):
        for row in rows:
            if row.lane != lane:
                raise ValueError(f"{lane} pair input contains the wrong QA lane")
            if row.question_id in target:
                raise ValueError(f"{lane} pair input contains duplicate question ids")
            target[row.question_id] = row
    expected = [question.question_id for question in questions]
    if len(set(expected)) != len(expected):
        raise ValueError("Quest question inputs contain duplicate question ids")
    if set(reference) != set(expected) or set(generated) != set(expected):
        raise ValueError("Quest answer rows must exactly cover the question universe")
    for question in questions:
        if (
            reference[question.question_id].case_key != question.case_key
            or generated[question.question_id].case_key != question.case_key
        ):
            raise ValueError("Quest answer row case identity drift")
    return tuple(
        QuestAnswerPair(
            question_id=question.question_id,
            reference_answer=reference[question.question_id].answer,
            generated_answer=generated[question.question_id].answer,
        )
        for question in questions
    )


def _tokenizer_json(provenance: TokenizerProvenance) -> dict[str, object]:
    return asdict(provenance)


def _semantic_json(provenance: SemanticProvenance) -> dict[str, object]:
    return asdict(provenance)


def _validate_fingerprint_map(
    raw: object,
    *,
    label: str,
    expected_keys: frozenset[str] | None = None,
) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} must be a non-empty mapping")
    if expected_keys is not None and set(raw) != expected_keys:
        raise ValueError(f"{label} keys drift")
    result: dict[str, str] = {}
    for key, value in raw.items():
        if type(key) is not str or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = _require_sha256(value, label=f"{label}.{key}")
    return dict(sorted(result.items()))


def build_numeric_samples(  # noqa: PLR0912
    cases: Sequence[GenerationCase],
    *,
    generation_rows: Sequence[GenerationCacheRow],
    semantic_rows: Sequence[SemanticCacheRow],
    questions: Sequence[GeneratedQuestion],
    reference_qa_rows: Sequence[QACacheRow],
    prediction_qa_rows: Sequence[QACacheRow],
    tokenizer: Tokenizer,
    semantic_provenance: SemanticProvenance,
    input_fingerprints: Mapping[str, str],
) -> dict[str, object]:
    """Build text-free rows from complete, already-authenticated local caches."""

    checked = _checked_cases(cases)
    generation_by_case: dict[str, GenerationCacheRow] = {}
    for generation_row in generation_rows:
        if generation_row.case_key in generation_by_case:
            raise ValueError("generation rows contain duplicate case keys")
        generation_by_case[generation_row.case_key] = generation_row
    if set(generation_by_case) != {case.case_key for case in checked}:
        raise ValueError("generation rows must exactly cover every case")
    for case in checked:
        generation_row = generation_by_case[case.case_key]
        if generation_row.task != case.task:
            raise ValueError("generation row task differs from its case")

    semantic_by_case: dict[str, SemanticCacheRow] = {}
    for semantic_input_row in semantic_rows:
        if semantic_input_row.case_key in semantic_by_case:
            raise ValueError("semantic rows contain duplicate case keys")
        semantic_by_case[semantic_input_row.case_key] = semantic_input_row
    if set(semantic_by_case) != {case.case_key for case in checked}:
        raise ValueError("semantic rows must exactly cover every case")

    case_sequence = {case.case_key: index for index, case in enumerate(checked)}
    case_rows: list[dict[str, object]] = []
    for task in M9B_TASKS:
        task_cases = [case for case in checked if case.task == task]
        examples = tuple(
            GenerationExample(
                sample_id=case.case_key,
                prediction=generation_by_case[case.case_key].prediction,
                reference=case.reference,
            )
            for case in task_cases
        )
        lexical = evaluate_generation(examples, tokenizer)
        lexical_by_id = {row.sample_id: row for row in lexical.lexical_scores}
        for case in task_cases:
            lexical_row = lexical_by_id[case.case_key]
            semantic_row = semantic_by_case[case.case_key]
            case_rows.append(
                {
                    "sequence": case_sequence[case.case_key],
                    "cluster": case_sequence[case.case_key],
                    "task": case.task,
                    "metrics": {
                        STANDARD_BLEU_METRIC: lexical_row.bleu.score,
                        CRUD_BLEU_METRIC: lexical_row.bleu.score_without_brevity_penalty,
                        "mean_sentence_rouge_l_precision": lexical_row.rouge_l.precision,
                        "mean_sentence_rouge_l_recall": lexical_row.rouge_l.recall,
                        ROUGE_L_METRIC: lexical_row.rouge_l.f1,
                        "mean_bertscore_precision_zh_rescaled": semantic_row.precision,
                        "mean_bertscore_recall_zh_rescaled": semantic_row.recall,
                        BERTSCORE_METRIC: semantic_row.f1,
                    },
                }
            )
    case_rows.sort(key=lambda row: cast(int, row["sequence"]))

    case_by_key = {case.case_key: case for case in checked}
    question_by_id: dict[str, GeneratedQuestion] = {}
    for question in questions:
        if question.question_id in question_by_id:
            raise ValueError("questions contain duplicate ids")
        if question.case_key not in case_by_key:
            raise ValueError("question belongs to an unknown case")
        question_by_id[question.question_id] = question
    pairs = build_quest_answer_pairs(questions, reference_qa_rows, prediction_qa_rows)
    pair_by_id = {pair.question_id: pair for pair in pairs}
    question_rows: list[dict[str, object]] = []
    sequence = 0
    for task in M9B_TASKS:
        task_questions = [
            question for question in questions if case_by_key[question.case_key].task == task
        ]
        if not task_questions:
            raise ValueError(f"{task} must have at least one generated question")
        report = evaluate_quest_answers(
            tuple(pair_by_id[question.question_id] for question in task_questions),
            tokenizer,
        )
        score_by_id = {score.question_id: score for score in report.scores}
        for question in task_questions:
            score = score_by_id[question.question_id]
            question_rows.append(
                {
                    "sequence": sequence,
                    "cluster": case_sequence[question.case_key],
                    "task": task,
                    "reference_match": score.reference_sentinel.match,
                    "generated_match": score.generated_sentinel.match,
                    "token_f1": None if score.token_overlap is None else score.token_overlap.f1,
                }
            )
            sequence += 1

    artifact: dict[str, object] = {
        "schema": GENERATION_SAMPLES_SCHEMA,
        "tasks": list(M9B_TASKS),
        "generation_metrics": list(_GENERATION_METRIC_NAMES),
        "ragquest_metrics": list(_RAG_METRIC_NAMES),
        "case_count": len(checked),
        "question_count": len(questions),
        "input_fingerprints": _validate_fingerprint_map(
            input_fingerprints,
            label="input fingerprints",
            expected_keys=PUBLIC_INPUT_FINGERPRINT_KEYS,
        ),
        "tokenizer": _tokenizer_json(tokenizer.provenance),
        "semantic_provenance": _semantic_json(semantic_provenance),
        "case_rows": case_rows,
        "question_rows": question_rows,
    }
    validate_numeric_samples(artifact)
    return artifact


def _parse_tokenizer_provenance(raw: object) -> TokenizerProvenance:
    row = _exact_mapping(
        raw,
        keys={"name", "package", "version", "mode", "hmm", "user_dictionary"},
        label="tokenizer provenance",
    )
    return TokenizerProvenance(
        name=row["name"],
        package=row["package"],
        version=row["version"],
        mode=row["mode"],
        hmm=row["hmm"],
        user_dictionary=row["user_dictionary"],
    )


def _parse_semantic_provenance(raw: object) -> SemanticProvenance:
    row = _exact_mapping(
        raw,
        keys={
            "metric",
            "package",
            "distribution_version",
            "module_version",
            "model",
            "lang",
            "num_layers",
            "rescale_with_baseline",
            "idf",
            "batch_size",
            "use_fast_tokenizer",
        },
        label="semantic provenance",
    )
    return SemanticProvenance(
        metric=row["metric"],
        package=row["package"],
        distribution_version=row["distribution_version"],
        module_version=row["module_version"],
        model=row["model"],
        lang=row["lang"],
        num_layers=row["num_layers"],
        rescale_with_baseline=row["rescale_with_baseline"],
        idf=row["idf"],
        batch_size=row["batch_size"],
        use_fast_tokenizer=row["use_fast_tokenizer"],
    )


def _walk_forbidden(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: public keys must be strings")
            if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"{path}.{key}: forbidden raw field")
            _walk_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")


def validate_numeric_samples(raw: object) -> None:  # noqa: PLR0912, PLR0915
    root = _exact_mapping(
        raw,
        keys={
            "schema",
            "tasks",
            "generation_metrics",
            "ragquest_metrics",
            "case_count",
            "question_count",
            "input_fingerprints",
            "tokenizer",
            "semantic_provenance",
            "case_rows",
            "question_rows",
        },
        label="generation numeric samples",
    )
    if root["schema"] != GENERATION_SAMPLES_SCHEMA:
        raise ValueError("generation numeric sample schema drift")
    if root["tasks"] != list(M9B_TASKS):
        raise ValueError("generation numeric sample task order drift")
    if root["generation_metrics"] != list(_GENERATION_METRIC_NAMES):
        raise ValueError("generation metric order drift")
    if root["ragquest_metrics"] != list(_RAG_METRIC_NAMES):
        raise ValueError("RAGQuest metric order drift")
    case_count = _require_int(root["case_count"], label="case_count", minimum=2)
    question_count = _require_int(root["question_count"], label="question_count", minimum=2)
    input_fingerprints = _validate_fingerprint_map(
        root["input_fingerprints"],
        label="input fingerprints",
        expected_keys=PUBLIC_INPUT_FINGERPRINT_KEYS,
    )
    if input_fingerprints["chat_profile_contract_sha256"] != CHAT_PROFILE_CONTRACT_SHA256:
        raise ValueError("chat profile contract fingerprint drift")
    _parse_tokenizer_provenance(root["tokenizer"])
    _parse_semantic_provenance(root["semantic_provenance"])

    case_rows = root["case_rows"]
    if not isinstance(case_rows, list) or len(case_rows) != case_count:
        raise ValueError("case rows must exactly cover case_count")
    task_by_cluster: dict[int, str] = {}
    task_case_counts = {task: 0 for task in M9B_TASKS}
    for index, raw_row in enumerate(case_rows):
        row = _exact_mapping(
            raw_row,
            keys={"sequence", "cluster", "task", "metrics"},
            label=f"case row {index}",
        )
        if row["sequence"] != index or row["cluster"] != index:
            raise ValueError("case row sequence/cluster must be dense and aligned")
        task = row["task"]
        if task not in _SAFE_PUBLIC_TASKS:
            raise ValueError("case row has an unknown task")
        task_case_counts[cast(TaskName, task)] += 1
        task_by_cluster[index] = task
        metrics = _exact_mapping(
            row["metrics"],
            keys=set(_GENERATION_METRIC_NAMES),
            label=f"case row {index} metrics",
        )
        for name, value in metrics.items():
            _finite(
                value,
                label=f"case row {index}.{name}",
                unit_interval=name
                not in {
                    "mean_bertscore_precision_zh_rescaled",
                    "mean_bertscore_recall_zh_rescaled",
                    BERTSCORE_METRIC,
                },
            )
    if any(count < 1 for count in task_case_counts.values()):
        raise ValueError("case rows must cover both tasks")

    question_rows = root["question_rows"]
    if not isinstance(question_rows, list) or len(question_rows) != question_count:
        raise ValueError("question rows must exactly cover question_count")
    task_question_counts = {task: 0 for task in M9B_TASKS}
    matches = {"none", "exact", "normalized", "near"}
    for index, raw_row in enumerate(question_rows):
        row = _exact_mapping(
            raw_row,
            keys={
                "sequence",
                "cluster",
                "task",
                "reference_match",
                "generated_match",
                "token_f1",
            },
            label=f"question row {index}",
        )
        if row["sequence"] != index:
            raise ValueError("question row sequences must be dense")
        cluster = _require_int(row["cluster"], label="question cluster", minimum=0)
        if cluster >= case_count:
            raise ValueError("question row cluster is outside the case universe")
        task = row["task"]
        if task not in _SAFE_PUBLIC_TASKS or task_by_cluster[cluster] != task:
            raise ValueError("question task must match its case cluster")
        task_question_counts[cast(TaskName, task)] += 1
        reference_match = row["reference_match"]
        generated_match = row["generated_match"]
        if reference_match not in matches or generated_match not in matches:
            raise ValueError("question sentinel match is invalid")
        both_answerable = reference_match not in {"exact", "normalized"} and (
            generated_match not in {"exact", "normalized"}
        )
        if both_answerable:
            _finite(row["token_f1"], label="question token_f1", unit_interval=True)
        elif row["token_f1"] is not None:
            raise ValueError("unanswerable question rows must have null token_f1")
    if any(count < 1 for count in task_question_counts.values()):
        raise ValueError("question rows must cover both tasks")
    _walk_forbidden(raw)


def numeric_samples_sha256(samples: Mapping[str, object]) -> str:
    validate_numeric_samples(samples)
    return hashlib.sha256(
        b"zhrag-crud-generation-samples-hash-v1\0" + _canonical_json(samples)
    ).hexdigest()


def _seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256()
    for value in ("zhrag-crud-generation-bootstrap-seed-v1", str(base_seed), namespace):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def _ci_json(interval: BootstrapCI, *, observations: int) -> dict[str, object]:
    return {
        "mean": interval.mean,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "observations": observations,
        "clusters": interval.n,
    }


def _cluster_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    clusters: Sequence[int],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    if not (len(numerators) == len(denominators) == len(clusters)) or not numerators:
        raise ValueError("ratio inputs must be non-empty and aligned")
    grouped: dict[int, tuple[float, float]] = {}
    for numerator, denominator, cluster in zip(
        numerators,
        denominators,
        clusters,
        strict=True,
    ):
        n = _finite(numerator, label="ratio numerator")
        d = _finite(denominator, label="ratio denominator")
        if n < 0.0 or d < 0.0 or n > d:
            raise ValueError("ratio contributions must satisfy 0 <= numerator <= denominator")
        old_n, old_d = grouped.get(cluster, (0.0, 0.0))
        grouped[cluster] = old_n + n, old_d + d
    total_n = math.fsum(numerators)
    total_d = math.fsum(denominators)
    if total_d == 0.0:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "observations": 0,
            "clusters": len(grouped),
            "denominator": 0,
            "valid_resamples": 0,
        }
    rng = random.Random(seed)
    groups = tuple(grouped.values())
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.choices(groups, k=len(groups))
        denominator = math.fsum(item[1] for item in sampled)
        if denominator > 0.0:
            values.append(math.fsum(item[0] for item in sampled) / denominator)
    if not values:
        raise ValueError("conditional bootstrap produced no defined replicates")
    values.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": total_n / total_d,
        "ci_low": values[int(alpha * len(values))],
        "ci_high": values[min(int((1.0 - alpha) * len(values)), len(values) - 1)],
        "observations": int(total_d),
        "clusters": len(grouped),
        "denominator": int(total_d),
        "valid_resamples": len(values),
    }


def _task_rows(samples: Mapping[str, Any], name: str, task: TaskName) -> list[Mapping[str, Any]]:
    raw = samples[name]
    if not isinstance(raw, list):
        raise AssertionError("validated sample rows are not a list")
    return [cast(Mapping[str, Any], row) for row in raw if row["task"] == task]


def build_generation_report(
    samples: Mapping[str, object],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Rebuild every aggregate solely from authenticated, text-free samples."""

    validate_numeric_samples(samples)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    _require_int(resamples, label="resamples", minimum=1)
    _require_int(seed, label="seed", minimum=0)
    typed = cast(Mapping[str, Any], samples)
    task_reports: dict[str, object] = {}
    for task in M9B_TASKS:
        case_rows = _task_rows(typed, "case_rows", task)
        case_clusters = [int(row["cluster"]) for row in case_rows]
        generation: dict[str, object] = {}
        for metric in _GENERATION_METRIC_NAMES:
            values = [float(row["metrics"][metric]) for row in case_rows]
            generation[metric] = _ci_json(
                clustered_bootstrap_ci(
                    values,
                    case_clusters,
                    confidence=confidence,
                    resamples=resamples,
                    seed=_seed(seed, f"{task}/generation/{metric}"),
                ),
                observations=len(values),
            )

        question_rows = _task_rows(typed, "question_rows", task)
        clusters = [int(row["cluster"]) for row in question_rows]
        generated_answerable = [
            0.0 if row["generated_match"] in {"exact", "normalized"} else 1.0
            for row in question_rows
        ]
        token_f1 = [
            0.0 if row["token_f1"] is None else float(row["token_f1"]) for row in question_rows
        ]
        reference_answerable = [
            0.0 if row["reference_match"] in {"exact", "normalized"} else 1.0
            for row in question_rows
        ]
        both_answerable = [
            left * right
            for left, right in zip(reference_answerable, generated_answerable, strict=True)
        ]
        ragquest: dict[str, object] = {}
        for metric, values in (
            ("paper_recall_all_questions", generated_answerable),
            ("paper_precision_all_questions", token_f1),
        ):
            ragquest[metric] = _ci_json(
                clustered_bootstrap_ci(
                    values,
                    clusters,
                    confidence=confidence,
                    resamples=resamples,
                    seed=_seed(seed, f"{task}/ragquest/{metric}"),
                ),
                observations=len(values),
            )
            cast(dict[str, object], ragquest[metric])["denominator"] = len(values)
        ragquest["code_recall_reference_answerable"] = _cluster_ratio_ci(
            both_answerable,
            reference_answerable,
            clusters,
            confidence=confidence,
            resamples=resamples,
            seed=_seed(seed, f"{task}/ragquest/code-recall"),
        )
        ragquest["code_precision_generated_answerable"] = _cluster_ratio_ci(
            token_f1,
            both_answerable,
            clusters,
            confidence=confidence,
            resamples=resamples,
            seed=_seed(seed, f"{task}/ragquest/code-precision"),
        )
        sentinel_counts = {
            f"reference_{match}": sum(row["reference_match"] == match for row in question_rows)
            for match in ("exact", "normalized", "near")
        }
        sentinel_counts.update(
            {
                f"generated_{match}": sum(row["generated_match"] == match for row in question_rows)
                for match in ("exact", "normalized", "near")
            }
        )
        task_reports[task] = {
            "case_count": len(case_rows),
            "question_count": len(question_rows),
            "generation": generation,
            "ragquest": ragquest,
            "sentinel_counts": sentinel_counts,
        }

    inputs = _validate_fingerprint_map(
        typed["input_fingerprints"],
        label="report input fingerprints",
        expected_keys=PUBLIC_INPUT_FINGERPRINT_KEYS,
    )
    if inputs["chat_profile_contract_sha256"] != CHAT_PROFILE_CONTRACT_SHA256:
        raise ValueError("chat profile contract fingerprint drift")
    report: dict[str, object] = {
        "schema": GENERATION_REPORT_SCHEMA,
        "contract": M9B_CONTRACT_VERSION,
        "design": {
            "profile": "known-context generation; retrieval excluded",
            "tasks": list(M9B_TASKS),
            "confidence": confidence,
            "resamples": resamples,
            "seed": seed,
            "bootstrap": "case-cluster percentile bootstrap",
            "hypothesis_tests": "none; single-profile descriptive estimates",
        },
        "inputs": inputs,
        "samples": {
            "schema": GENERATION_SAMPLES_SCHEMA,
            "sha256": numeric_samples_sha256(samples),
        },
        "tokenizer_fingerprint": hashlib.sha256(
            b"zhrag-crud-tokenizer-provenance-v1\0" + _canonical_json(typed["tokenizer"])
        ).hexdigest(),
        "semantic_provenance_fingerprint": hashlib.sha256(
            b"zhrag-crud-semantic-provenance-v1\0" + _canonical_json(typed["semantic_provenance"])
        ).hexdigest(),
        "tasks": task_reports,
        "limitations": list(_LIMITATIONS),
    }
    validate_generation_report(report)
    return report


def _validate_estimate(
    raw: object,
    *,
    label: str,
    allow_none: bool = False,
    unit_interval: bool = False,
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an estimate object")
    base = {"mean", "ci_low", "ci_high", "observations", "clusters"}
    optional = {"denominator", "valid_resamples"}
    if not base <= set(raw) or set(raw) - base - optional:
        raise ValueError(f"{label} estimate keys drift")
    mean = raw["mean"]
    low = raw["ci_low"]
    high = raw["ci_high"]
    if allow_none and mean is low is high is None:
        if raw.get("denominator") != 0 or raw.get("valid_resamples") != 0:
            raise ValueError(f"{label} empty conditional estimate metadata drift")
    else:
        _finite(mean, label=f"{label}.mean", unit_interval=unit_interval)
        checked_low = _finite(low, label=f"{label}.ci_low", unit_interval=unit_interval)
        checked_high = _finite(high, label=f"{label}.ci_high", unit_interval=unit_interval)
        if checked_low > checked_high:
            raise ValueError(f"{label} CI bounds are reversed")
    _require_int(raw["observations"], label=f"{label}.observations", minimum=0)
    _require_int(raw["clusters"], label=f"{label}.clusters", minimum=1)
    if "denominator" in raw:
        _require_int(raw["denominator"], label=f"{label}.denominator", minimum=0)
    if "valid_resamples" in raw:
        _require_int(raw["valid_resamples"], label=f"{label}.valid_resamples", minimum=0)


def validate_generation_report(raw: object) -> None:
    root = _exact_mapping(
        raw,
        keys={
            "schema",
            "contract",
            "design",
            "inputs",
            "samples",
            "tokenizer_fingerprint",
            "semantic_provenance_fingerprint",
            "tasks",
            "limitations",
        },
        label="generation report",
    )
    if root["schema"] != GENERATION_REPORT_SCHEMA or root["contract"] != M9B_CONTRACT_VERSION:
        raise ValueError("generation report schema drift")
    design = _exact_mapping(
        root["design"],
        keys={
            "profile",
            "tasks",
            "confidence",
            "resamples",
            "seed",
            "bootstrap",
            "hypothesis_tests",
        },
        label="generation report design",
    )
    if (
        design["profile"] != "known-context generation; retrieval excluded"
        or design["tasks"] != list(M9B_TASKS)
        or design["bootstrap"] != "case-cluster percentile bootstrap"
        or design["hypothesis_tests"] != "none; single-profile descriptive estimates"
    ):
        raise ValueError("generation report design drift")
    confidence = _finite(design["confidence"], label="report confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("report confidence must be in (0, 1)")
    _require_int(design["resamples"], label="report resamples", minimum=1)
    _require_int(design["seed"], label="report seed", minimum=0)
    inputs = _validate_fingerprint_map(
        root["inputs"],
        label="report inputs",
        expected_keys=PUBLIC_INPUT_FINGERPRINT_KEYS,
    )
    if inputs["chat_profile_contract_sha256"] != CHAT_PROFILE_CONTRACT_SHA256:
        raise ValueError("chat profile contract fingerprint drift")
    sample_ref = _exact_mapping(
        root["samples"],
        keys={"schema", "sha256"},
        label="generation report samples",
    )
    if sample_ref["schema"] != GENERATION_SAMPLES_SCHEMA:
        raise ValueError("generation report sample schema drift")
    _require_sha256(sample_ref["sha256"], label="generation report sample sha256")
    _require_sha256(root["tokenizer_fingerprint"], label="tokenizer fingerprint")
    _require_sha256(
        root["semantic_provenance_fingerprint"],
        label="semantic provenance fingerprint",
    )
    tasks = _exact_mapping(root["tasks"], keys=set(M9B_TASKS), label="generation report tasks")
    for task in M9B_TASKS:
        task_row = _exact_mapping(
            tasks[task],
            keys={
                "case_count",
                "question_count",
                "generation",
                "ragquest",
                "sentinel_counts",
            },
            label=f"generation report task {task}",
        )
        _require_int(task_row["case_count"], label=f"{task}.case_count", minimum=1)
        _require_int(
            task_row["question_count"],
            label=f"{task}.question_count",
            minimum=1,
        )
        generation = _exact_mapping(
            task_row["generation"],
            keys=set(_GENERATION_METRIC_NAMES),
            label=f"{task}.generation",
        )
        for metric, estimate in generation.items():
            _validate_estimate(
                estimate,
                label=f"{task}.generation.{metric}",
                unit_interval=metric
                not in {
                    "mean_bertscore_precision_zh_rescaled",
                    "mean_bertscore_recall_zh_rescaled",
                    BERTSCORE_METRIC,
                },
            )
        ragquest = _exact_mapping(
            task_row["ragquest"],
            keys=set(_RAG_METRIC_NAMES),
            label=f"{task}.ragquest",
        )
        for metric, estimate in ragquest.items():
            _validate_estimate(
                estimate,
                label=f"{task}.ragquest.{metric}",
                allow_none=metric.startswith("code_"),
                unit_interval=True,
            )
        sentinel = _exact_mapping(
            task_row["sentinel_counts"],
            keys={
                "reference_exact",
                "reference_normalized",
                "reference_near",
                "generated_exact",
                "generated_normalized",
                "generated_near",
            },
            label=f"{task}.sentinel_counts",
        )
        for name, value in sentinel.items():
            _require_int(value, label=f"{task}.{name}", minimum=0)
    if root["limitations"] != list(_LIMITATIONS):
        raise ValueError("generation report limitations drift")
    _walk_forbidden(raw)
