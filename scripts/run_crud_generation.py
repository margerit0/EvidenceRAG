"""Offline-first orchestration for the CRUD-RAG M9b1 generation profile.

Running this script without an action is read-only status inspection.  Chat
actions require ``--allow-paid-provider`` and semantic scoring additionally
requires ``--allow-model-download``.  Text-bearing artifacts stay in the
ignored experiment tree; published samples and reports are aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from zhrag.eval.crud_generation import (
    CACHE_ROW_SCHEMA,
    M9B_CONTRACT_VERSION,
    M9B_TASKS,
    GeneratedQuestion,
    GenerationCacheRow,
    GenerationCase,
    GenerationInputManifest,
    QACacheRow,
    QGCacheRow,
    QuestionBank,
    ReferenceBank,
    SemanticCacheRow,
    build_generation_cases,
    build_generation_report,
    build_input_manifest,
    build_numeric_samples,
    build_question_bank,
    build_reference_bank,
    cache_rows_fingerprint,
    canonical_fingerprint,
    dataset_snapshot_sha256,
    generation_cache_id,
    generation_instructions_fingerprint,
    generation_system_prompt,
    generation_user_prompt,
    parse_generation_cache_row,
    parse_generation_input_manifest,
    parse_generation_response,
    parse_qa_cache_row,
    parse_qa_response,
    parse_qg_cache_row,
    parse_question_bank,
    parse_questions_response,
    parse_reference_bank,
    parse_semantic_cache_row,
    prediction_qa_cache_id,
    qa_instructions_fingerprint,
    qa_system_prompt,
    qa_user_prompt,
    qg_cache_id,
    qg_instructions_fingerprint,
    qg_system_prompt,
    qg_user_prompt,
    question_id,
    reference_qa_cache_id,
    semantic_cache_id,
    validate_generation_report,
    validate_generation_rows,
    validate_numeric_samples,
    validate_prediction_qa_rows,
    validate_prompt_budget,
    validate_qg_rows,
    validate_reference_qa_rows,
    validate_semantic_rows,
)
from zhrag.eval.metrics_gen import SemanticProvenance, Tokenizer
from zhrag.io_utils import (
    append_jsonl,
    exclusive_lock,
    read_bytes,
    read_json,
    read_jsonl,
    replace_files,
    write_json,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "crud-rag-subset" / "raw" / "split_merged.json"
DEFAULT_ARTIFACTS = ROOT / "indexes" / "crud" / "generation" / "v1"
DEFAULT_ENV = ROOT / ".env"
ARTIFACT_LOCK = ".artifacts.lock"
INPUT_MANIFEST = "input_manifest.json"
QUESTION_BANK = "question_bank.json"
REFERENCE_BANK = "reference_bank.json"
NUMERIC_SAMPLES = "numeric_samples.json"
REPORT = "report.json"
GENERATION_CACHE = "generation_cache.jsonl"
QG_CACHE = "qg_cache.jsonl"
REFERENCE_QA_CACHE = "reference_qa_cache.jsonl"
PREDICTION_QA_CACHE = "prediction_qa_cache.jsonl"
SEMANTIC_CACHE = "semantic_scores.jsonl"

CACHE_META_SCHEMA = "zhrag-crud-generation-cache-meta-v1"
CACHE_UNIVERSE_SCHEMA = "zhrag-crud-cache-universe-v1"
GENERATION_CACHE_KEY_SCHEMA = "zhrag-crud-generation-cache-key-v1"
QG_CACHE_KEY_SCHEMA = "zhrag-crud-qg-cache-key-v1"
REFERENCE_QA_CACHE_KEY_SCHEMA = "zhrag-crud-reference-qa-cache-key-v1"
PREDICTION_QA_CACHE_KEY_SCHEMA = "zhrag-crud-prediction-qa-cache-key-v1"
SEMANTIC_CACHE_KEY_SCHEMA = "zhrag-crud-semantic-cache-key-v1"

_SLUG = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_META_KEYS = frozenset(
    {
        "schema",
        "contract",
        "stage",
        "cache_key_schema",
        "input_manifest_sha256",
        "universe_fingerprint",
        "expected_ids_fingerprint",
        "expected_count",
        "prompt_fingerprint",
        "model_profile_sha256",
        "model",
        "endpoint",
        "reasoning_effort",
        "json_object",
        "retries",
        "parent_fingerprints",
        "semantic_provenance",
        "complete",
        "cache_sha256",
    }
)
_STATIC_META_KEYS = _META_KEYS - {"complete", "cache_sha256"}
_META_STAGES = frozenset({"generation", "qg", "reference_qa", "prediction_qa", "semantic"})
_SEMANTIC_FIELDS = frozenset(
    {
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
    }
)


type ActionName = Literal[
    "generate",
    "generate_questions",
    "finalize_questions",
    "answer_reference",
    "finalize_reference_bank",
    "answer_prediction",
    "score_semantic",
    "finalize",
]

_ACTION_FLAGS: tuple[tuple[str, ActionName], ...] = (
    ("generate", "generate"),
    ("generate_questions", "generate_questions"),
    ("finalize_questions", "finalize_questions"),
    ("answer_reference", "answer_reference"),
    ("finalize_reference_bank", "finalize_reference_bank"),
    ("answer_prediction", "answer_prediction"),
    ("score_semantic", "score_semantic"),
    ("finalize", "finalize"),
)
_PAID_ACTIONS = frozenset(
    {"generate", "generate_questions", "answer_reference", "answer_prediction"}
)
_RUN_ACTIONS = frozenset({"generate", "answer_prediction", "score_semantic", "finalize"})
_REFERENCE_ACTIONS = frozenset(
    {
        "generate_questions",
        "finalize_questions",
        "answer_reference",
        "finalize_reference_bank",
        "answer_prediction",
        "finalize",
    }
)
_BOTH_PROFILE_ACTIONS = frozenset({"answer_prediction", "finalize"})


class ChatReplyLike(Protocol):
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


class ChatClientLike(Protocol):
    retries: int

    def complete(self, system: str, user: str, *, json_object: bool = True) -> ChatReplyLike: ...


class SemanticScorerLike(Protocol):
    @property
    def provenance(self) -> SemanticProvenance: ...

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> tuple[Sequence[object], Sequence[object], Sequence[object]]: ...


@dataclass(frozen=True, slots=True)
class InputState:
    cases: tuple[GenerationCase, ...]
    manifest: GenerationInputManifest
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class StageState:
    meta: dict[str, object]
    rows: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class QuestionState:
    qg: StageState
    questions: tuple[GeneratedQuestion, ...]
    bank: QuestionBank
    bank_sha256: str


@dataclass(frozen=True, slots=True)
class ReferenceState:
    question: QuestionState
    qa: StageState
    bank: ReferenceBank
    bank_sha256: str


@dataclass(frozen=True, slots=True)
class ChatProfile:
    client: ChatClientLike
    model: str
    endpoint: str
    reasoning_effort: str
    retries: int
    profile_sha256: str


@dataclass
class CallBudget:
    limit: int | None
    used: int = 0

    def take(self) -> bool:
        if self.limit is not None and self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass(frozen=True, slots=True)
class CacheDefinition:
    stage: str
    cache_name: str
    key_schema: str
    parser: Callable[[object], object]


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    for flag, _action_name in _ACTION_FLAGS:
        actions.add_argument(f"--{flag.replace('_', '-')}", action="store_true")
    actions.add_argument("--status", action="store_true", help="read-only status inspection")
    actions.add_argument("--dry-run", action="store_true", help="alias for read-only status")
    parser.add_argument("--input", "--dataset", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--reference-profile", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="place the selected run under runs/canonical instead of runs/trials",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--qg-model", default=None)
    parser.add_argument("--qa-model", default=None)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-paid-provider", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args(argv)


def _action(args: argparse.Namespace) -> ActionName | None:
    selected = [name for attribute, name in _ACTION_FLAGS if getattr(args, attribute)]
    return selected[0] if selected else None


def _slug(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise SystemExit(
            f"! {label} must be a safe 1..64-character ASCII slug using letters, digits, . _ -"
        )
    return value


def _validate_args(args: argparse.Namespace, action: ActionName | None) -> None:  # noqa: PLR0912
    if args.max_calls is not None and args.max_calls < 0:
        raise SystemExit("! --max-calls must be non-negative")
    if args.resamples < 1:
        raise SystemExit("! --resamples must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("! --confidence must be in (0, 1)")
    if args.seed < 0:
        raise SystemExit("! --seed must be non-negative")
    if action is None:
        if args.max_calls is not None:
            raise SystemExit("! --max-calls requires a paid chat action")
        if args.allow_paid_provider or args.allow_model_download:
            raise SystemExit("! provider/model guards require an explicit action")
        if args.canonical and args.run_id is None:
            raise SystemExit("! --canonical requires --run-id")
        return
    if action in _PAID_ACTIONS and not args.allow_paid_provider:
        raise SystemExit(
            f"! --{action.replace('_', '-')} sends paid chat requests; "
            "repeat with --allow-paid-provider"
        )
    if action == "score_semantic" and not args.allow_model_download:
        raise SystemExit(
            "! --score-semantic loads BERTScore weights; repeat with --allow-model-download"
        )
    if action not in _PAID_ACTIONS and args.max_calls is not None:
        raise SystemExit("! --max-calls applies only to paid chat actions")
    if action != "score_semantic" and args.allow_model_download:
        raise SystemExit("! --allow-model-download applies only to --score-semantic")
    if action not in _PAID_ACTIONS and args.allow_paid_provider:
        raise SystemExit("! --allow-paid-provider applies only to paid chat actions")
    if action in _RUN_ACTIONS:
        if args.run_id is None:
            raise SystemExit(f"! --{action.replace('_', '-')} requires --run-id")
        _slug(args.run_id, label="--run-id")
    elif args.run_id is not None:
        raise SystemExit(f"! --run-id is not used by --{action.replace('_', '-')}")
    if action in _REFERENCE_ACTIONS:
        if args.reference_profile is None:
            raise SystemExit(f"! --{action.replace('_', '-')} requires --reference-profile")
        _slug(args.reference_profile, label="--reference-profile")
    elif args.reference_profile is not None:
        raise SystemExit(f"! --reference-profile is not used by --{action.replace('_', '-')}")
    if action in _BOTH_PROFILE_ACTIONS and args.run_id is None:
        raise SystemExit(f"! --{action.replace('_', '-')} requires --run-id")
    if args.canonical and action not in _RUN_ACTIONS:
        raise SystemExit("! --canonical applies only to run artifacts")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _json_sha256(value: object, *, schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8") + b"\0" + _canonical_json(value)).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    result = float(cast(Any, value))
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _exact_mapping(value: object, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} keys drift: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return cast(Mapping[str, Any], value)


def _normalise_endpoint(raw: object) -> str:
    value = _text(raw, label="endpoint")
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in {"http", "https", "local"} or not parts.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) or local URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError("endpoint must not contain query parameters or a fragment")
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc, path, "", ""))


def _manifest_digest(manifest: GenerationInputManifest) -> str:
    return _json_sha256(
        manifest.as_json(),
        schema="zhrag-crud-generation-input-manifest-hash-v1",
    )


def _load_input(path: Path) -> InputState:
    try:
        raw_bytes = read_bytes(path)
        raw = read_json(path)
        if not isinstance(raw, Mapping):
            raise ValueError("dataset root must be an object")
        cases = build_generation_cases(cast(Mapping[str, Sequence[Mapping[str, object]]], raw))
        manifest = build_input_manifest(
            cases,
            dataset_sha256=dataset_snapshot_sha256(raw_bytes),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid CRUD generation input {path}: {exc}") from exc
    return InputState(cases, manifest, _manifest_digest(manifest))


def _run_root(args: argparse.Namespace) -> Path:
    if args.run_id is None:
        raise AssertionError("run id is required after argument validation")
    group = "canonical" if args.canonical else "trials"
    run_id = _slug(args.run_id, label="--run-id")
    return Path(args.artifacts) / "runs" / group / run_id


def _reference_root(args: argparse.Namespace) -> Path:
    if args.reference_profile is None:
        raise AssertionError("reference profile is required after argument validation")
    profile = _slug(args.reference_profile, label="--reference-profile")
    return Path(args.artifacts) / "reference_banks" / profile


def _manifest_path(root: Path) -> Path:
    return root / INPUT_MANIFEST


def _has_entries(root: Path) -> bool:
    return root.is_dir() and any(root.iterdir())


def _publish_new_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = Path(f"{path}.initial.tmp")
    try:
        write_json(temporary, dict(value))
        replace_files(((temporary, path),))
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_manifest(root: Path, state: InputState) -> None:
    path = _manifest_path(root)
    if path.exists():
        try:
            recorded = parse_generation_input_manifest(read_json(path))
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"! invalid input manifest marker {path}: {exc}") from exc
        if recorded.as_json() != state.manifest.as_json():
            raise SystemExit(f"! input manifest drift at {path}; use a new profile/root")
        return
    if _has_entries(root):
        raise SystemExit(f"! refusing to adopt artifacts without input manifest: {root}")
    _publish_new_json(path, state.manifest.as_json())


def _require_manifest(root: Path, state: InputState) -> None:
    path = _manifest_path(root)
    if not path.exists():
        raise SystemExit(f"! input manifest marker is absent: {path}")
    try:
        recorded = parse_generation_input_manifest(read_json(path))
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid input manifest marker {path}: {exc}") from exc
    if recorded.as_json() != state.manifest.as_json():
        raise SystemExit(f"! input manifest drift at {path}; use a new profile/root")


def _profile_fingerprint(
    *,
    model: str,
    endpoint: str,
    reasoning_effort: str,
    retries: int,
    json_object: bool,
) -> str:
    return canonical_fingerprint(
        "zhrag-crud-chat-profile-v1",
        _text(model, label="model"),
        _normalise_endpoint(endpoint),
        _text(reasoning_effort, label="reasoning effort"),
        str(_int(retries, label="retries")),
        str(json_object),
    )


def _chat_profile_fingerprint_from_meta(meta: Mapping[str, object]) -> str:
    return _profile_fingerprint(
        model=cast(str, meta["model"]),
        endpoint=cast(str, meta["endpoint"]),
        reasoning_effort=cast(str, meta["reasoning_effort"]),
        retries=cast(int, meta["retries"]),
        json_object=cast(bool, meta["json_object"]),
    )


def _load_chat_profile(
    args: argparse.Namespace,
    stage: Literal["generation", "qg", "qa"],
) -> ChatProfile:
    # Dynamic imports keep status and all offline finalizers provider-free.
    chat_module = importlib.import_module("zhrag.providers.chat")
    embedding_module = importlib.import_module("zhrag.providers.embedding")
    env = embedding_module.load_env(args.env)
    override_name = {"generation": "generator_model", "qg": "qg_model", "qa": "qa_model"}[stage]
    model = getattr(args, override_name) or args.model
    config = chat_module.ChatConfig.from_env(env, model=model)
    effort = _text(args.reasoning_effort, label="reasoning effort")
    client = chat_module.ChatClient(config=config, reasoning_effort=effort)
    endpoint = _normalise_endpoint(config.endpoint)
    retries = _int(getattr(client, "retries", 7), label="retries")
    return ChatProfile(
        client=cast(ChatClientLike, client),
        model=_text(config.model, label="model"),
        endpoint=endpoint,
        reasoning_effort=effort,
        retries=retries,
        profile_sha256=_profile_fingerprint(
            model=config.model,
            endpoint=endpoint,
            reasoning_effort=effort,
            retries=retries,
            json_object=True,
        ),
    )


def _semantic_provenance(raw: object) -> SemanticProvenance:
    row = _exact_mapping(raw, keys=_SEMANTIC_FIELDS, label="semantic provenance")
    try:
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid semantic provenance: {exc}") from exc


def _semantic_profile_fingerprint(provenance: SemanticProvenance) -> str:
    return _json_sha256(asdict(provenance), schema="zhrag-crud-semantic-profile-v1")


def _prompt_fingerprint(stage: str) -> str:
    if stage == "generation":
        return canonical_fingerprint(
            "zhrag-crud-generation-prompt-profile-v1",
            *(generation_instructions_fingerprint(task) for task in M9B_TASKS),
            "strict-json-single-field-parser-v1",
        )
    if stage == "qg":
        return canonical_fingerprint(
            "zhrag-crud-qg-prompt-profile-v1",
            qg_instructions_fingerprint(),
            "strict-json-question-list-parser-v1",
        )
    if stage in {"reference_qa", "prediction_qa"}:
        return canonical_fingerprint(
            "zhrag-crud-qa-prompt-profile-v1",
            qa_instructions_fingerprint(),
            "strict-json-answer-parser-v1",
        )
    raise ValueError(f"unsupported prompt stage {stage!r}")


def _cache_definition(stage: str) -> CacheDefinition:
    definitions = {
        "generation": CacheDefinition(
            "generation", GENERATION_CACHE, GENERATION_CACHE_KEY_SCHEMA, parse_generation_cache_row
        ),
        "qg": CacheDefinition("qg", QG_CACHE, QG_CACHE_KEY_SCHEMA, parse_qg_cache_row),
        "reference_qa": CacheDefinition(
            "reference_qa",
            REFERENCE_QA_CACHE,
            REFERENCE_QA_CACHE_KEY_SCHEMA,
            parse_qa_cache_row,
        ),
        "prediction_qa": CacheDefinition(
            "prediction_qa",
            PREDICTION_QA_CACHE,
            PREDICTION_QA_CACHE_KEY_SCHEMA,
            parse_qa_cache_row,
        ),
        "semantic": CacheDefinition(
            "semantic", SEMANTIC_CACHE, SEMANTIC_CACHE_KEY_SCHEMA, parse_semantic_cache_row
        ),
    }
    try:
        return definitions[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported cache stage {stage!r}") from exc


def _expected_meta(
    *,
    state: InputState,
    definition: CacheDefinition,
    universe_fingerprint: str,
    expected_ids: Sequence[str],
    prompt_fingerprint: str,
    model_profile_sha256: str,
    model: str,
    endpoint: str,
    reasoning_effort: str,
    retries: int,
    json_object: bool,
    parent_fingerprints: Mapping[str, str] | None = None,
    semantic_provenance: SemanticProvenance | None = None,
) -> dict[str, object]:
    ids = tuple(expected_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{definition.stage} expected cache ids must be unique and non-empty")
    parents = parent_fingerprints or {}
    checked_parents = {
        _text(key, label="parent fingerprint key"): _digest(
            value, label=f"parent fingerprint {key}"
        )
        for key, value in parents.items()
    }
    if definition.stage == "semantic" and semantic_provenance is None:
        raise ValueError("semantic metadata requires semantic provenance")
    if definition.stage != "semantic" and semantic_provenance is not None:
        raise ValueError("only semantic metadata may contain semantic provenance")
    return {
        "schema": CACHE_META_SCHEMA,
        "contract": M9B_CONTRACT_VERSION,
        "stage": definition.stage,
        "cache_key_schema": definition.key_schema,
        "input_manifest_sha256": _digest(state.manifest_sha256, label="input manifest fingerprint"),
        "universe_fingerprint": _digest(universe_fingerprint, label="cache universe fingerprint"),
        "expected_ids_fingerprint": canonical_fingerprint(CACHE_UNIVERSE_SCHEMA, *ids),
        "expected_count": len(ids),
        "prompt_fingerprint": _digest(prompt_fingerprint, label="prompt fingerprint"),
        "model_profile_sha256": _digest(model_profile_sha256, label="model profile fingerprint"),
        "model": _text(model, label="model"),
        "endpoint": _normalise_endpoint(endpoint),
        "reasoning_effort": _text(reasoning_effort, label="reasoning effort"),
        "json_object": json_object,
        "retries": _int(retries, label="retries"),
        "parent_fingerprints": dict(sorted(checked_parents.items())),
        "semantic_provenance": (
            None if semantic_provenance is None else asdict(semantic_provenance)
        ),
        "complete": False,
        "cache_sha256": None,
    }


def _parse_meta(raw: object, path: Path) -> dict[str, object]:  # noqa: PLR0912
    row = _exact_mapping(raw, keys=_META_KEYS, label=f"cache metadata {path}")
    if row["schema"] != CACHE_META_SCHEMA or row["contract"] != M9B_CONTRACT_VERSION:
        raise ValueError(f"cache metadata contract drift at {path}")
    stage = row["stage"]
    if stage not in _META_STAGES:
        raise ValueError(f"cache metadata stage drift at {path}")
    for name in (
        "cache_key_schema",
        "input_manifest_sha256",
        "universe_fingerprint",
        "expected_ids_fingerprint",
        "prompt_fingerprint",
        "model_profile_sha256",
        "model",
        "endpoint",
        "reasoning_effort",
    ):
        _text(row[name], label=f"cache metadata {name}")
    for name in (
        "input_manifest_sha256",
        "universe_fingerprint",
        "expected_ids_fingerprint",
        "prompt_fingerprint",
        "model_profile_sha256",
    ):
        _digest(row[name], label=f"cache metadata {name}")
    _normalise_endpoint(row["endpoint"])
    _int(row["expected_count"], label="cache metadata expected_count", minimum=1)
    _int(row["retries"], label="cache metadata retries", minimum=0)
    if type(row["json_object"]) is not bool:
        raise ValueError("cache metadata json_object must be boolean")
    parent = row["parent_fingerprints"]
    if not isinstance(parent, Mapping) or any(not isinstance(key, str) for key in parent):
        raise ValueError("cache metadata parent_fingerprints must be an object")
    for key, value in parent.items():
        _text(key, label="parent fingerprint key")
        _digest(value, label=f"parent fingerprint {key}")
    if stage == "semantic":
        provenance = _semantic_provenance(row["semantic_provenance"])
        if _semantic_profile_fingerprint(provenance) != row["model_profile_sha256"]:
            raise ValueError("semantic cache metadata profile drift")
    else:
        if row["semantic_provenance"] is not None:
            raise ValueError("non-semantic cache metadata contains semantic provenance")
        try:
            profile_fingerprint = _chat_profile_fingerprint_from_meta(row)
        except (TypeError, ValueError) as exc:
            raise ValueError("chat cache metadata profile is invalid") from exc
        if profile_fingerprint != row["model_profile_sha256"]:
            raise ValueError("chat cache metadata profile drift")
    if type(row["complete"]) is not bool:
        raise ValueError("cache metadata complete must be boolean")
    cache_sha = row["cache_sha256"]
    if row["complete"]:
        _digest(cache_sha, label="complete cache sha256")
    elif cache_sha is not None:
        raise ValueError("partial cache metadata cannot contain cache_sha256")
    return dict(row)


def _compare_static_meta(recorded: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key in _STATIC_META_KEYS:
        if recorded.get(key) != expected.get(key):
            raise SystemExit(f"! cache metadata drift in field {key}; use a new profile/root")


def _read_rows(path: Path, parser: Callable[[object], object]) -> tuple[object, ...]:
    if not path.exists() or path.stat().st_size == 0:
        return ()
    try:
        return tuple(parser(raw) for raw in read_jsonl(path))
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid cache {path}: {exc}") from exc


def _cache_fingerprint(rows: Sequence[object]) -> str:
    try:
        return cache_rows_fingerprint(rows)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"! cannot fingerprint cache rows: {exc}") from exc


def _prepare_cache(
    root: Path,
    definition: CacheDefinition,
    expected: Mapping[str, object],
) -> tuple[dict[str, object], tuple[object, ...]]:
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if meta_path.exists():
        try:
            recorded = _parse_meta(read_json(meta_path), meta_path)
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"! invalid cache metadata {meta_path}: {exc}") from exc
        _compare_static_meta(recorded, expected)
        rows = _read_rows(cache, definition.parser)
        if recorded["complete"] and (
            not rows or _cache_fingerprint(rows) != recorded["cache_sha256"]
        ):
            raise SystemExit(f"! complete cache content drift: {cache}")
        return recorded, rows
    if cache.exists() and cache.stat().st_size > 0:
        raise SystemExit(f"! refusing to adopt non-empty cache without provenance: {cache}")
    _publish_new_json(meta_path, expected)
    return dict(expected), ()


def _finish_cache(
    root: Path,
    definition: CacheDefinition,
    recorded: Mapping[str, object],
    rows: Sequence[object],
) -> dict[str, object]:
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    completed = dict(recorded)
    if recorded["complete"]:
        if _cache_fingerprint(rows) != recorded["cache_sha256"]:
            raise SystemExit(f"! immutable cache content drift: {cache}")
        return completed
    if not rows:
        raise SystemExit(f"! cannot complete an empty cache: {cache}")
    completed["complete"] = True
    completed["cache_sha256"] = _cache_fingerprint(rows)
    temporary = Path(f"{meta_path}.complete.tmp")
    try:
        write_json(temporary, completed)
        replace_files(((temporary, meta_path),))
    finally:
        temporary.unlink(missing_ok=True)
    return completed


def _generation_expected(state: InputState, profile: ChatProfile) -> dict[str, object]:
    definition = _cache_definition("generation")
    return _expected_meta(
        state=state,
        definition=definition,
        universe_fingerprint=state.manifest.generation_universe_fingerprint,
        expected_ids=[generation_cache_id(case, profile.profile_sha256) for case in state.cases],
        prompt_fingerprint=_prompt_fingerprint("generation"),
        model_profile_sha256=profile.profile_sha256,
        model=profile.model,
        endpoint=profile.endpoint,
        reasoning_effort=profile.reasoning_effort,
        retries=profile.retries,
        json_object=True,
    )


def _qg_expected(state: InputState, profile: ChatProfile) -> dict[str, object]:
    definition = _cache_definition("qg")
    return _expected_meta(
        state=state,
        definition=definition,
        universe_fingerprint=state.manifest.reference_universe_fingerprint,
        expected_ids=[qg_cache_id(case, profile.profile_sha256) for case in state.cases],
        prompt_fingerprint=_prompt_fingerprint("qg"),
        model_profile_sha256=profile.profile_sha256,
        model=profile.model,
        endpoint=profile.endpoint,
        reasoning_effort=profile.reasoning_effort,
        retries=profile.retries,
        json_object=True,
    )


def _validate_served_models(
    rows: Sequence[object],
    meta: Mapping[str, object],
    *,
    label: str,
) -> None:
    expected = _text(meta["model"], label=f"{label} requested model")
    for row in rows:
        if getattr(row, "served_model", None) != expected:
            raise SystemExit(f"! {label} served model drift; cache row is not authenticated")


def _validate_generation_state(
    state: InputState,
    rows: Sequence[object],
    meta: Mapping[str, object],
) -> tuple[GenerationCacheRow, ...]:
    typed = tuple(cast(GenerationCacheRow, row) for row in rows)
    try:
        validate_generation_rows(
            state.cases,
            typed,
            model_profile_fingerprint=cast(str, meta["model_profile_sha256"]),
            require_complete=bool(meta["complete"]),
        )
    except ValueError as exc:
        raise SystemExit(f"! generation cache validation failed: {exc}") from exc
    _validate_served_models(typed, meta, label="generation cache")
    return typed


def _validate_qg_state(
    state: InputState,
    rows: Sequence[object],
    meta: Mapping[str, object],
) -> tuple[QGCacheRow, ...]:
    typed = tuple(cast(QGCacheRow, row) for row in rows)
    try:
        validate_qg_rows(
            state.cases,
            typed,
            model_profile_fingerprint=cast(str, meta["model_profile_sha256"]),
            require_complete=bool(meta["complete"]),
        )
    except ValueError as exc:
        raise SystemExit(f"! QG cache validation failed: {exc}") from exc
    _validate_served_models(typed, meta, label="QG cache")
    return typed


def _load_generation_state(
    state: InputState,
    root: Path,
    *,
    require_complete: bool,
) -> StageState:
    _require_manifest(root, state)
    definition = _cache_definition("generation")
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"! generation cache metadata is missing under {root}")
    try:
        preliminary = _parse_meta(read_json(meta_path), meta_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid generation cache metadata: {exc}") from exc
    expected = _expected_meta_from_recorded(
        state=state,
        definition=definition,
        meta=preliminary,
        universe_fingerprint=state.manifest.generation_universe_fingerprint,
        expected_ids=[
            generation_cache_id(case, cast(str, preliminary["model_profile_sha256"]))
            for case in state.cases
        ],
        prompt_fingerprint=_prompt_fingerprint("generation"),
    )
    _compare_static_meta(preliminary, expected)
    rows = _read_rows(cache, definition.parser)
    typed = _validate_generation_state(state, rows, preliminary)
    if preliminary["complete"] and _cache_fingerprint(typed) != preliminary["cache_sha256"]:
        raise SystemExit(f"! immutable generation cache content drift: {cache}")
    if require_complete and not preliminary["complete"]:
        raise SystemExit("! generation cache is incomplete; finish --generate first")
    return StageState(dict(preliminary), typed)


def _load_qg_state(state: InputState, root: Path, *, require_complete: bool) -> StageState:
    _require_manifest(root, state)
    definition = _cache_definition("qg")
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"! QG cache metadata is missing under {root}")
    try:
        preliminary = _parse_meta(read_json(meta_path), meta_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid QG cache metadata: {exc}") from exc
    profile = cast(str, preliminary["model_profile_sha256"])
    expected = _expected_meta_from_recorded(
        state=state,
        definition=definition,
        meta=preliminary,
        universe_fingerprint=state.manifest.reference_universe_fingerprint,
        expected_ids=[qg_cache_id(case, profile) for case in state.cases],
        prompt_fingerprint=_prompt_fingerprint("qg"),
    )
    _compare_static_meta(preliminary, expected)
    rows = _read_rows(cache, definition.parser)
    typed = _validate_qg_state(state, rows, preliminary)
    if preliminary["complete"] and _cache_fingerprint(typed) != preliminary["cache_sha256"]:
        raise SystemExit(f"! immutable QG cache content drift: {cache}")
    if require_complete and not preliminary["complete"]:
        raise SystemExit("! QG cache is incomplete; finish --generate-questions first")
    return StageState(dict(preliminary), typed)


def _expected_meta_from_recorded(
    *,
    state: InputState,
    definition: CacheDefinition,
    meta: Mapping[str, object],
    universe_fingerprint: str,
    expected_ids: Sequence[str],
    prompt_fingerprint: str,
    parent_fingerprints: Mapping[str, str] | None = None,
    semantic_provenance: SemanticProvenance | None = None,
) -> dict[str, object]:
    return _expected_meta(
        state=state,
        definition=definition,
        universe_fingerprint=universe_fingerprint,
        expected_ids=expected_ids,
        prompt_fingerprint=prompt_fingerprint,
        model_profile_sha256=cast(str, meta["model_profile_sha256"]),
        model=cast(str, meta["model"]),
        endpoint=cast(str, meta["endpoint"]),
        reasoning_effort=cast(str, meta["reasoning_effort"]),
        retries=cast(int, meta["retries"]),
        json_object=cast(bool, meta["json_object"]),
        parent_fingerprints=parent_fingerprints,
        semantic_provenance=semantic_provenance,
    )


def _ordered_questions(
    state: InputState,
    qg: StageState,
) -> tuple[GeneratedQuestion, ...]:
    typed = tuple(cast(QGCacheRow, row) for row in qg.rows)
    try:
        by_id = validate_qg_rows(
            state.cases,
            typed,
            model_profile_fingerprint=cast(str, qg.meta["model_profile_sha256"]),
            require_complete=True,
        )
    except ValueError as exc:
        raise SystemExit(f"! cannot order finalized questions: {exc}") from exc
    profile = cast(str, qg.meta["model_profile_sha256"])
    return tuple(
        question for case in state.cases for question in by_id[qg_cache_id(case, profile)].questions
    )


def _bank_hash(value: Mapping[str, object], *, schema: str) -> str:
    return _json_sha256(value, schema=schema)


def _build_question_state(
    state: InputState,
    root: Path,
    *,
    require_marker: bool,
) -> QuestionState:
    qg = _load_qg_state(state, root, require_complete=True)
    questions = _ordered_questions(state, qg)
    profile = cast(str, qg.meta["model_profile_sha256"])
    try:
        bank = build_question_bank(
            state.cases,
            tuple(cast(QGCacheRow, row) for row in qg.rows),
            reference_universe_fingerprint=state.manifest.reference_universe_fingerprint,
            qg_model_profile_fingerprint=profile,
        )
    except ValueError as exc:
        raise SystemExit(f"! cannot build question bank: {exc}") from exc
    marker = root / QUESTION_BANK
    if require_marker:
        if not marker.exists():
            raise SystemExit(f"! question bank marker is missing: {marker}")
        try:
            recorded = parse_question_bank(read_json(marker))
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"! invalid question bank marker: {exc}") from exc
        if recorded.as_json() != bank.as_json():
            raise SystemExit(f"! question bank marker drift: {marker}")
    return QuestionState(
        qg=qg,
        questions=questions,
        bank=bank,
        bank_sha256=_bank_hash(bank.as_json(), schema="zhrag-crud-question-bank-marker-v1"),
    )


def _reference_qa_expected(
    state: InputState,
    question: QuestionState,
    meta: Mapping[str, object],
) -> dict[str, object]:
    profile = cast(str, meta["model_profile_sha256"])
    expected_ids = [
        reference_qa_cache_id(case, generated_question, profile)
        for case in state.cases
        for generated_question in question.questions
        if generated_question.case_key == case.case_key
    ]
    return _expected_meta_from_recorded(
        state=state,
        definition=_cache_definition("reference_qa"),
        meta=meta,
        universe_fingerprint=question.bank.question_universe_fingerprint,
        expected_ids=expected_ids,
        prompt_fingerprint=_prompt_fingerprint("reference_qa"),
        parent_fingerprints={"question_bank_sha256": question.bank_sha256},
    )


def _load_reference_qa_state(
    state: InputState,
    root: Path,
    question: QuestionState,
    *,
    require_complete: bool,
) -> StageState:
    _require_manifest(root, state)
    definition = _cache_definition("reference_qa")
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"! reference QA cache metadata is missing under {root}")
    try:
        preliminary = _parse_meta(read_json(meta_path), meta_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid reference QA cache metadata: {exc}") from exc
    expected = _reference_qa_expected(state, question, preliminary)
    _compare_static_meta(preliminary, expected)
    rows = _read_rows(cache, definition.parser)
    typed = tuple(cast(QACacheRow, row) for row in rows)
    try:
        validate_reference_qa_rows(
            state.cases,
            question.questions,
            typed,
            model_profile_fingerprint=cast(str, preliminary["model_profile_sha256"]),
            require_complete=bool(preliminary["complete"]),
        )
    except ValueError as exc:
        raise SystemExit(f"! reference QA cache validation failed: {exc}") from exc
    _validate_served_models(typed, preliminary, label="reference QA cache")
    if preliminary["complete"] and _cache_fingerprint(typed) != preliminary["cache_sha256"]:
        raise SystemExit(f"! immutable reference QA cache content drift: {cache}")
    if require_complete and not preliminary["complete"]:
        raise SystemExit("! reference QA cache is incomplete; finish --answer-reference first")
    return StageState(dict(preliminary), typed)


def _build_reference_state(
    state: InputState,
    root: Path,
    *,
    require_marker: bool,
) -> ReferenceState:
    question = _build_question_state(state, root, require_marker=True)
    qa = _load_reference_qa_state(state, root, question, require_complete=True)
    profile = cast(str, qa.meta["model_profile_sha256"])
    try:
        bank = build_reference_bank(
            state.cases,
            question.questions,
            tuple(cast(QACacheRow, row) for row in qa.rows),
            question_universe_fingerprint=question.bank.question_universe_fingerprint,
            qa_model_profile_fingerprint=profile,
        )
    except ValueError as exc:
        raise SystemExit(f"! cannot build reference bank: {exc}") from exc
    marker = root / REFERENCE_BANK
    if require_marker:
        if not marker.exists():
            raise SystemExit(f"! reference bank marker is missing: {marker}")
        try:
            recorded = parse_reference_bank(read_json(marker))
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"! invalid reference bank marker: {exc}") from exc
        if recorded.as_json() != bank.as_json():
            raise SystemExit(f"! reference bank marker drift: {marker}")
    return ReferenceState(
        question=question,
        qa=qa,
        bank=bank,
        bank_sha256=_bank_hash(bank.as_json(), schema="zhrag-crud-reference-bank-marker-v1"),
    )


def _prediction_expected(
    state: InputState,
    reference: ReferenceState,
    generation: StageState,
    meta: Mapping[str, object],
) -> dict[str, object]:
    profile = cast(str, meta["model_profile_sha256"])
    predictions = _prediction_map(generation)
    expected_ids = [
        prediction_qa_cache_id(case, generated_question, predictions[case.case_key], profile)
        for case in state.cases
        for generated_question in reference.question.questions
        if generated_question.case_key == case.case_key
    ]
    return _expected_meta_from_recorded(
        state=state,
        definition=_cache_definition("prediction_qa"),
        meta=meta,
        universe_fingerprint=reference.question.bank.question_universe_fingerprint,
        expected_ids=expected_ids,
        prompt_fingerprint=_prompt_fingerprint("prediction_qa"),
        parent_fingerprints={
            "question_bank_sha256": reference.question.bank_sha256,
            "reference_bank_sha256": reference.bank_sha256,
            "generation_cache_sha256": _cache_fingerprint(generation.rows),
        },
    )


def _prediction_map(generation: StageState) -> dict[str, str]:
    return {
        cast(GenerationCacheRow, row).case_key: cast(GenerationCacheRow, row).prediction
        for row in generation.rows
    }


def _load_prediction_qa_state(
    state: InputState,
    root: Path,
    reference: ReferenceState,
    generation: StageState,
    *,
    require_complete: bool,
) -> StageState:
    _require_manifest(root, state)
    definition = _cache_definition("prediction_qa")
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"! prediction QA cache metadata is missing under {root}")
    try:
        preliminary = _parse_meta(read_json(meta_path), meta_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid prediction QA cache metadata: {exc}") from exc
    expected = _prediction_expected(state, reference, generation, preliminary)
    _compare_static_meta(preliminary, expected)
    rows = _read_rows(cache, definition.parser)
    typed = tuple(cast(QACacheRow, row) for row in rows)
    try:
        validate_prediction_qa_rows(
            state.cases,
            reference.question.questions,
            typed,
            model_profile_fingerprint=cast(str, preliminary["model_profile_sha256"]),
            predictions=_prediction_map(generation),
            require_complete=bool(preliminary["complete"]),
        )
    except ValueError as exc:
        raise SystemExit(f"! prediction QA cache validation failed: {exc}") from exc
    _validate_served_models(typed, preliminary, label="prediction QA cache")
    if preliminary["complete"] and _cache_fingerprint(typed) != preliminary["cache_sha256"]:
        raise SystemExit(f"! immutable prediction QA cache content drift: {cache}")
    if require_complete and not preliminary["complete"]:
        raise SystemExit("! prediction QA cache is incomplete; finish --answer-prediction first")
    return StageState(dict(preliminary), typed)


def _semantic_expected(
    state: InputState,
    generation: StageState,
    meta: Mapping[str, object],
    provenance: SemanticProvenance,
) -> dict[str, object]:
    profile = cast(str, meta["model_profile_sha256"])
    predictions = _prediction_map(generation)
    definition = _cache_definition("semantic")
    return _expected_meta_from_recorded(
        state=state,
        definition=definition,
        meta=meta,
        universe_fingerprint=state.manifest.generation_universe_fingerprint,
        expected_ids=[
            semantic_cache_id(case, predictions[case.case_key], profile) for case in state.cases
        ],
        prompt_fingerprint=_json_sha256(
            asdict(provenance), schema="zhrag-crud-semantic-settings-v1"
        ),
        parent_fingerprints={"generation_cache_sha256": _cache_fingerprint(generation.rows)},
        semantic_provenance=provenance,
    )


def _validate_semantic_state(
    state: InputState,
    generation: StageState,
    rows: Sequence[object],
    meta: Mapping[str, object],
) -> tuple[SemanticCacheRow, ...]:
    typed = tuple(cast(SemanticCacheRow, row) for row in rows)
    try:
        validate_semantic_rows(
            state.cases,
            typed,
            predictions=_prediction_map(generation),
            semantic_profile_fingerprint=cast(str, meta["model_profile_sha256"]),
            require_complete=bool(meta["complete"]),
        )
    except ValueError as exc:
        raise SystemExit(f"! semantic cache validation failed: {exc}") from exc
    return typed


def _load_semantic_state(
    state: InputState,
    root: Path,
    generation: StageState,
    *,
    require_complete: bool,
) -> tuple[StageState, SemanticProvenance]:
    _require_manifest(root, state)
    definition = _cache_definition("semantic")
    cache = root / definition.cache_name
    meta_path = Path(f"{cache}.meta.json")
    if not meta_path.exists():
        raise SystemExit(f"! semantic cache metadata is missing under {root}")
    try:
        preliminary = _parse_meta(read_json(meta_path), meta_path)
        provenance = _semantic_provenance(preliminary["semantic_provenance"])
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! invalid semantic cache metadata: {exc}") from exc
    expected = _semantic_expected(state, generation, preliminary, provenance)
    _compare_static_meta(preliminary, expected)
    rows = _read_rows(cache, definition.parser)
    typed = _validate_semantic_state(state, generation, rows, preliminary)
    if preliminary["complete"] and _cache_fingerprint(typed) != preliminary["cache_sha256"]:
        raise SystemExit(f"! immutable semantic cache content drift: {cache}")
    if require_complete and not preliminary["complete"]:
        raise SystemExit("! semantic cache is incomplete; finish --score-semantic first")
    return StageState(dict(preliminary), typed), provenance


def _publish_marker(
    path: Path,
    value: Mapping[str, object],
    parser: Callable[[object], object],
    *,
    label: str,
) -> bool:
    if path.exists():
        try:
            recorded = parser(read_json(path))
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(f"! invalid {label} marker {path}: {exc}") from exc
        recorded_json = recorded.as_json() if hasattr(recorded, "as_json") else recorded
        if recorded_json != dict(value):
            raise SystemExit(f"! immutable {label} marker drift: {path}")
        return False
    temporary = Path(f"{path}.tmp")
    try:
        write_json(temporary, dict(value))
        replace_files(((temporary, path),))
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _reply_usage(reply: ChatReplyLike) -> tuple[int | None, int | None, int | None]:
    return (
        getattr(reply, "prompt_tokens", None),
        getattr(reply, "completion_tokens", None),
        getattr(reply, "reasoning_tokens", None),
    )


def _call_chat(
    profile: ChatProfile,
    system: str,
    user: str,
    budget: CallBudget,
    *,
    label: str,
) -> ChatReplyLike | None:
    if not budget.take():
        return None
    try:
        reply = profile.client.complete(system, user, json_object=True)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"! {label} provider call failed; no cache row appended") from exc
    if getattr(reply, "model", None) != profile.model:
        raise SystemExit(f"! {label} provider returned an unexpected model; no cache row appended")
    return reply


def _require_same_qa_profile(
    reference: ReferenceState,
    profile: ChatProfile,
) -> None:
    reference_profile = reference.qa.meta["model_profile_sha256"]
    if profile.profile_sha256 != reference_profile:
        raise SystemExit("! prediction QA profile differs from finalized reference QA profile")


def _append_row(path: Path, row: object) -> None:
    as_json = getattr(row, "as_json", None)
    if not callable(as_json):
        raise ValueError("cache row must expose as_json()")
    payload = as_json()
    if not isinstance(payload, dict):
        raise ValueError("cache row JSON must be an object")
    append_jsonl(path, [payload])


def _run_generate(args: argparse.Namespace, state: InputState) -> int:
    root = _run_root(args)
    if (root / REPORT).exists():
        raise SystemExit(f"! run is finalized and immutable: {root / REPORT}")
    profile = _load_chat_profile(args, "generation")
    root.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(root, state)
    definition = _cache_definition("generation")
    expected = _generation_expected(state, profile)
    with exclusive_lock(root / ".generation.lock"):
        recorded, old_rows = _prepare_cache(root, definition, expected)
        typed_old = _validate_generation_state(state, old_rows, recorded)
        if recorded["complete"]:
            print(f"generation: complete ({len(typed_old):,}/{len(state.cases):,})")
            return 0
        existing = {row.cache_id for row in typed_old}
        budget = CallBudget(args.max_calls)
        for case in state.cases:
            cache_id = generation_cache_id(case, profile.profile_sha256)
            if cache_id in existing:
                continue
            system = generation_system_prompt(case.task)
            user = generation_user_prompt(case)
            try:
                validate_prompt_budget(system, user)
                reply = _call_chat(profile, system, user, budget, label="generation")
            except ValueError as exc:
                raise SystemExit(
                    f"! generation item rejected; no cache row appended: {exc}"
                ) from exc
            if reply is None:
                print(
                    f"generation: paused after {budget.used:,} call(s); resume with the same run id"
                )
                return 0
            try:
                prediction = parse_generation_response(case.task, reply.content)
                usage = _reply_usage(reply)
                row = GenerationCacheRow(
                    schema=CACHE_ROW_SCHEMA,
                    cache_id=cache_id,
                    case_key=case.case_key,
                    task=case.task,
                    prediction=prediction,
                    served_model=reply.model,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    reasoning_tokens=usage[2],
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"! invalid generation response; no cache row appended: {exc}"
                ) from exc
            _append_row(root / GENERATION_CACHE, row)
            existing.add(cache_id)
        rows = _read_rows(root / GENERATION_CACHE, definition.parser)
        complete = _validate_generation_state(state, rows, {**recorded, "complete": True})
        _finish_cache(root, definition, recorded, complete)
    print(f"generation: complete ({len(complete):,}/{len(state.cases):,})")
    return 0


def _run_generate_questions(args: argparse.Namespace, state: InputState) -> int:
    root = _reference_root(args)
    if (root / QUESTION_BANK).exists():
        _build_question_state(state, root, require_marker=True)
        print("QG: question bank is already finalized and immutable")
        return 0
    profile = _load_chat_profile(args, "qg")
    root.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(root, state)
    definition = _cache_definition("qg")
    expected = _qg_expected(state, profile)
    with exclusive_lock(root / ".qg.lock"):
        recorded, old_rows = _prepare_cache(root, definition, expected)
        typed_old = _validate_qg_state(state, old_rows, recorded)
        if recorded["complete"]:
            print(
                f"QG: cache complete ({len(typed_old):,}/{len(state.cases):,}); "
                "run --finalize-questions next"
            )
            return 0
        existing = {row.cache_id for row in typed_old}
        budget = CallBudget(args.max_calls)
        for case in state.cases:
            cache_id = qg_cache_id(case, profile.profile_sha256)
            if cache_id in existing:
                continue
            system = qg_system_prompt()
            user = qg_user_prompt(case)
            try:
                validate_prompt_budget(system, user)
                reply = _call_chat(profile, system, user, budget, label="QG")
            except ValueError as exc:
                raise SystemExit(f"! QG item rejected; no cache row appended: {exc}") from exc
            if reply is None:
                print(
                    f"QG: paused after {budget.used:,} call(s); "
                    "resume with the same reference profile"
                )
                return 0
            try:
                texts = parse_questions_response(reply.content)
                questions = tuple(
                    GeneratedQuestion(
                        question_id=question_id(case, ordinal, text),
                        case_key=case.case_key,
                        ordinal=ordinal,
                        text=text,
                    )
                    for ordinal, text in enumerate(texts)
                )
                usage = _reply_usage(reply)
                row = QGCacheRow(
                    schema=CACHE_ROW_SCHEMA,
                    cache_id=cache_id,
                    case_key=case.case_key,
                    questions=questions,
                    served_model=reply.model,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    reasoning_tokens=usage[2],
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"! invalid QG response; no cache row appended: {exc}") from exc
            _append_row(root / QG_CACHE, row)
            existing.add(cache_id)
        rows = _read_rows(root / QG_CACHE, definition.parser)
        complete = _validate_qg_state(state, rows, {**recorded, "complete": True})
        _finish_cache(root, definition, recorded, complete)
    print(
        f"QG: cache complete ({len(complete):,}/{len(state.cases):,}); "
        "run --finalize-questions next"
    )
    return 0


def _run_finalize_questions(args: argparse.Namespace, state: InputState) -> int:
    root = _reference_root(args)
    if not root.is_dir():
        raise SystemExit(f"! reference profile directory is missing: {root}")
    with exclusive_lock(args.artifacts / ARTIFACT_LOCK):
        question = _build_question_state(state, root, require_marker=False)
        changed = _publish_marker(
            root / QUESTION_BANK,
            question.bank.as_json(),
            parse_question_bank,
            label="question bank",
        )
    status = "published" if changed else "already published"
    print(f"questions: {status} ({question.bank.questions:,} question(s))")
    return 0


def _qa_expected(
    state: InputState,
    question: QuestionState,
    profile: ChatProfile,
    *,
    lane: Literal["reference_qa", "prediction_qa"],
    parents: Mapping[str, str],
    predictions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    definition = _cache_definition(lane)
    if lane == "reference_qa":
        ids = [
            reference_qa_cache_id(case, generated_question, profile.profile_sha256)
            for case in state.cases
            for generated_question in question.questions
            if generated_question.case_key == case.case_key
        ]
    else:
        if predictions is None:
            raise ValueError("prediction QA expected metadata requires predictions")
        ids = [
            prediction_qa_cache_id(
                case,
                generated_question,
                predictions[case.case_key],
                profile.profile_sha256,
            )
            for case in state.cases
            for generated_question in question.questions
            if generated_question.case_key == case.case_key
        ]
    return _expected_meta(
        state=state,
        definition=definition,
        universe_fingerprint=question.bank.question_universe_fingerprint,
        expected_ids=ids,
        prompt_fingerprint=_prompt_fingerprint(lane),
        model_profile_sha256=profile.profile_sha256,
        model=profile.model,
        endpoint=profile.endpoint,
        reasoning_effort=profile.reasoning_effort,
        retries=profile.retries,
        json_object=True,
        parent_fingerprints=parents,
    )


def _run_answer_reference(args: argparse.Namespace, state: InputState) -> int:
    root = _reference_root(args)
    question = _build_question_state(state, root, require_marker=True)
    if (root / REFERENCE_BANK).exists():
        _build_reference_state(state, root, require_marker=True)
        print("reference QA: reference bank is already finalized and immutable")
        return 0
    profile = _load_chat_profile(args, "qa")
    definition = _cache_definition("reference_qa")
    expected = _qa_expected(
        state,
        question,
        profile,
        lane="reference_qa",
        parents={"question_bank_sha256": question.bank_sha256},
    )
    with exclusive_lock(root / ".reference-qa.lock"):
        recorded, old_rows = _prepare_cache(root, definition, expected)
        typed_old = tuple(cast(QACacheRow, row) for row in old_rows)
        try:
            validate_reference_qa_rows(
                state.cases,
                question.questions,
                typed_old,
                model_profile_fingerprint=profile.profile_sha256,
                require_complete=bool(recorded["complete"]),
            )
        except ValueError as exc:
            raise SystemExit(f"! reference QA cache validation failed: {exc}") from exc
        if recorded["complete"]:
            print(
                f"reference QA: complete ({len(typed_old):,}/{len(question.questions):,}); "
                "run --finalize-reference-bank next"
            )
            return 0
        existing = {row.cache_id for row in typed_old}
        budget = CallBudget(args.max_calls)
        case_by_key = {case.case_key: case for case in state.cases}
        for generated_question in question.questions:
            case = case_by_key[generated_question.case_key]
            cache_id = reference_qa_cache_id(case, generated_question, profile.profile_sha256)
            if cache_id in existing:
                continue
            system = qa_system_prompt()
            user = qa_user_prompt(generated_question.text, case.reference)
            try:
                validate_prompt_budget(system, user)
                reply = _call_chat(profile, system, user, budget, label="reference QA")
            except ValueError as exc:
                raise SystemExit(
                    f"! reference QA item rejected; no cache row appended: {exc}"
                ) from exc
            if reply is None:
                print(
                    f"reference QA: paused after {budget.used:,} call(s); "
                    "resume with the same reference profile"
                )
                return 0
            try:
                answer = parse_qa_response(reply.content)
                usage = _reply_usage(reply)
                row = QACacheRow(
                    schema=CACHE_ROW_SCHEMA,
                    cache_id=cache_id,
                    case_key=case.case_key,
                    question_id=generated_question.question_id,
                    lane="reference",
                    answer=answer,
                    served_model=reply.model,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    reasoning_tokens=usage[2],
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"! invalid reference QA response; no cache row appended: {exc}"
                ) from exc
            _append_row(root / REFERENCE_QA_CACHE, row)
            existing.add(cache_id)
        rows = _read_rows(root / REFERENCE_QA_CACHE, definition.parser)
        typed = tuple(cast(QACacheRow, row) for row in rows)
        try:
            validate_reference_qa_rows(
                state.cases,
                question.questions,
                typed,
                model_profile_fingerprint=profile.profile_sha256,
                require_complete=True,
            )
        except ValueError as exc:
            raise SystemExit(f"! reference QA cache validation failed: {exc}") from exc
        _finish_cache(root, definition, recorded, typed)
    print(
        f"reference QA: complete ({len(typed):,}/{len(question.questions):,}); "
        "run --finalize-reference-bank next"
    )
    return 0


def _run_finalize_reference_bank(args: argparse.Namespace, state: InputState) -> int:
    root = _reference_root(args)
    if not root.is_dir():
        raise SystemExit(f"! reference profile directory is missing: {root}")
    with exclusive_lock(args.artifacts / ARTIFACT_LOCK):
        reference = _build_reference_state(state, root, require_marker=False)
        changed = _publish_marker(
            root / REFERENCE_BANK,
            reference.bank.as_json(),
            parse_reference_bank,
            label="reference bank",
        )
    status = "published" if changed else "already published"
    print(f"reference bank: {status} ({reference.bank.questions:,} question(s))")
    return 0


def _run_answer_prediction(args: argparse.Namespace, state: InputState) -> int:  # noqa: PLR0915
    run_root = _run_root(args)
    reference_root = _reference_root(args)
    generation = _load_generation_state(state, run_root, require_complete=True)
    if (run_root / REPORT).exists():
        raise SystemExit(f"! run is finalized and immutable: {run_root / REPORT}")
    reference = _build_reference_state(state, reference_root, require_marker=True)
    profile = _load_chat_profile(args, "qa")
    _require_same_qa_profile(reference, profile)
    predictions = _prediction_map(generation)
    definition = _cache_definition("prediction_qa")
    expected = _qa_expected(
        state,
        reference.question,
        profile,
        lane="prediction_qa",
        parents={
            "question_bank_sha256": reference.question.bank_sha256,
            "reference_bank_sha256": reference.bank_sha256,
            "generation_cache_sha256": _cache_fingerprint(generation.rows),
        },
        predictions=predictions,
    )
    with exclusive_lock(run_root / ".prediction-qa.lock"):
        recorded, old_rows = _prepare_cache(run_root, definition, expected)
        typed_old = tuple(cast(QACacheRow, row) for row in old_rows)
        try:
            validate_prediction_qa_rows(
                state.cases,
                reference.question.questions,
                typed_old,
                model_profile_fingerprint=profile.profile_sha256,
                predictions=predictions,
                require_complete=bool(recorded["complete"]),
            )
        except ValueError as exc:
            raise SystemExit(f"! prediction QA cache validation failed: {exc}") from exc
        if recorded["complete"]:
            print(
                f"prediction QA: complete ({len(typed_old):,}/"
                f"{len(reference.question.questions):,})"
            )
            return 0
        existing = {row.cache_id for row in typed_old}
        budget = CallBudget(args.max_calls)
        case_by_key = {case.case_key: case for case in state.cases}
        for generated_question in reference.question.questions:
            case = case_by_key[generated_question.case_key]
            prediction = predictions[case.case_key]
            cache_id = prediction_qa_cache_id(
                case, generated_question, prediction, profile.profile_sha256
            )
            if cache_id in existing:
                continue
            system = qa_system_prompt()
            user = qa_user_prompt(generated_question.text, prediction)
            try:
                validate_prompt_budget(system, user)
                reply = _call_chat(profile, system, user, budget, label="prediction QA")
            except ValueError as exc:
                raise SystemExit(
                    f"! prediction QA item rejected; no cache row appended: {exc}"
                ) from exc
            if reply is None:
                print(
                    f"prediction QA: paused after {budget.used:,} call(s); "
                    "resume with the same run id"
                )
                return 0
            try:
                answer = parse_qa_response(reply.content)
                usage = _reply_usage(reply)
                row = QACacheRow(
                    schema=CACHE_ROW_SCHEMA,
                    cache_id=cache_id,
                    case_key=case.case_key,
                    question_id=generated_question.question_id,
                    lane="prediction",
                    answer=answer,
                    served_model=reply.model,
                    prompt_tokens=usage[0],
                    completion_tokens=usage[1],
                    reasoning_tokens=usage[2],
                )
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"! invalid prediction QA response; no cache row appended: {exc}"
                ) from exc
            _append_row(run_root / PREDICTION_QA_CACHE, row)
            existing.add(cache_id)
        rows = _read_rows(run_root / PREDICTION_QA_CACHE, definition.parser)
        typed = tuple(cast(QACacheRow, row) for row in rows)
        try:
            validate_prediction_qa_rows(
                state.cases,
                reference.question.questions,
                typed,
                model_profile_fingerprint=profile.profile_sha256,
                predictions=predictions,
                require_complete=True,
            )
        except ValueError as exc:
            raise SystemExit(f"! prediction QA cache validation failed: {exc}") from exc
        _finish_cache(run_root, definition, recorded, typed)
    print(f"prediction QA: complete ({len(typed):,}/{len(reference.question.questions):,})")
    return 0


def _load_semantic_adapter() -> SemanticScorerLike:
    module = importlib.import_module("zhrag.eval.metrics_gen")
    adapter = module.BertScoreAdapter.load_default()
    return cast(SemanticScorerLike, adapter)


def _run_score_semantic(args: argparse.Namespace, state: InputState) -> int:
    root = _run_root(args)
    generation = _load_generation_state(state, root, require_complete=True)
    if (root / REPORT).exists():
        raise SystemExit(f"! run is finalized and immutable: {root / REPORT}")
    semantic_meta = root / f"{SEMANTIC_CACHE}.meta.json"
    if semantic_meta.exists():
        semantic, _provenance = _load_semantic_state(
            state,
            root,
            generation,
            require_complete=False,
        )
        if semantic.meta["complete"]:
            print(f"semantic: complete ({len(semantic.rows):,}/{len(state.cases):,})")
            return 0

    scorer = _load_semantic_adapter()
    provenance = scorer.provenance
    profile_sha256 = _semantic_profile_fingerprint(provenance)
    predictions = _prediction_map(generation)
    definition = _cache_definition("semantic")
    expected = _expected_meta(
        state=state,
        definition=definition,
        universe_fingerprint=state.manifest.generation_universe_fingerprint,
        expected_ids=[
            semantic_cache_id(case, predictions[case.case_key], profile_sha256)
            for case in state.cases
        ],
        prompt_fingerprint=_json_sha256(
            asdict(provenance), schema="zhrag-crud-semantic-settings-v1"
        ),
        model_profile_sha256=profile_sha256,
        model=provenance.model,
        endpoint="local://bert-score",
        reasoning_effort="none",
        retries=0,
        json_object=False,
        parent_fingerprints={"generation_cache_sha256": _cache_fingerprint(generation.rows)},
        semantic_provenance=provenance,
    )
    with exclusive_lock(root / ".semantic.lock"):
        recorded, old_rows = _prepare_cache(root, definition, expected)
        typed_old = _validate_semantic_state(state, generation, old_rows, recorded)
        if recorded["complete"]:
            print(f"semantic: complete ({len(typed_old):,}/{len(state.cases):,})")
            return 0
        existing = {row.cache_id for row in typed_old}
        pending = [
            case
            for case in state.cases
            if semantic_cache_id(case, predictions[case.case_key], profile_sha256) not in existing
        ]
        batch_size = max(1, provenance.batch_size)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                columns = scorer.score(
                    [predictions[case.case_key] for case in batch],
                    [case.reference for case in batch],
                )
                if not isinstance(columns, tuple) or len(columns) != 3:
                    raise ValueError("semantic scorer must return precision, recall and F1")
                values = tuple(tuple(column) for column in columns)
                if any(len(column) != len(batch) for column in values):
                    raise ValueError("semantic scorer returned a wrong row count")
                rows = tuple(
                    SemanticCacheRow(
                        schema=CACHE_ROW_SCHEMA,
                        cache_id=semantic_cache_id(
                            case,
                            predictions[case.case_key],
                            profile_sha256,
                        ),
                        case_key=case.case_key,
                        precision=_finite(values[0][index], label="semantic precision"),
                        recall=_finite(values[1][index], label="semantic recall"),
                        f1=_finite(values[2][index], label="semantic F1"),
                    )
                    for index, case in enumerate(batch)
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise SystemExit(
                    f"! invalid semantic score batch; no rows appended: {exc}"
                ) from exc
            append_jsonl(root / SEMANTIC_CACHE, [row.as_json() for row in rows])
            existing.update(row.cache_id for row in rows)
        semantic_values = tuple(
            cast(SemanticCacheRow, row)
            for row in _read_rows(root / SEMANTIC_CACHE, definition.parser)
        )
        typed = _validate_semantic_state(
            state,
            generation,
            semantic_values,
            {**recorded, "complete": True},
        )
        _finish_cache(root, definition, recorded, typed)
    print(f"semantic: complete ({len(typed):,}/{len(state.cases):,})")
    return 0


def _new_tokenizer() -> Tokenizer:
    module = importlib.import_module("zhrag.eval.metrics_gen")
    return cast(Tokenizer, module.JiebaTokenizer())


def _build_final_artifacts(
    args: argparse.Namespace,
    state: InputState,
    run_root: Path,
    reference_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    generation = _load_generation_state(state, run_root, require_complete=True)
    reference = _build_reference_state(state, reference_root, require_marker=True)
    prediction = _load_prediction_qa_state(
        state,
        run_root,
        reference,
        generation,
        require_complete=True,
    )
    semantic, semantic_provenance = _load_semantic_state(
        state,
        run_root,
        generation,
        require_complete=True,
    )
    reference_profile = cast(str, reference.qa.meta["model_profile_sha256"])
    prediction_profile = cast(str, prediction.meta["model_profile_sha256"])
    if reference_profile != prediction_profile:
        raise SystemExit("! reference and prediction QA model profiles differ")
    fingerprints = {
        "dataset_snapshot_sha256": state.manifest.dataset_snapshot_sha256,
        "input_manifest_sha256": state.manifest_sha256,
        "generation_model_profile_sha256": cast(str, generation.meta["model_profile_sha256"]),
        "qg_model_profile_sha256": cast(str, reference.question.qg.meta["model_profile_sha256"]),
        "qa_model_profile_sha256": reference_profile,
        "semantic_profile_sha256": cast(str, semantic.meta["model_profile_sha256"]),
        "generation_cache_sha256": _cache_fingerprint(generation.rows),
        "question_bank_sha256": reference.question.bank_sha256,
        "reference_bank_sha256": reference.bank_sha256,
        "prediction_qa_cache_sha256": _cache_fingerprint(prediction.rows),
        "semantic_cache_sha256": _cache_fingerprint(semantic.rows),
    }
    samples = build_numeric_samples(
        state.cases,
        generation_rows=tuple(cast(GenerationCacheRow, row) for row in generation.rows),
        semantic_rows=tuple(cast(SemanticCacheRow, row) for row in semantic.rows),
        questions=reference.question.questions,
        reference_qa_rows=tuple(cast(QACacheRow, row) for row in reference.qa.rows),
        prediction_qa_rows=tuple(cast(QACacheRow, row) for row in prediction.rows),
        tokenizer=_new_tokenizer(),
        semantic_provenance=semantic_provenance,
        input_fingerprints=fingerprints,
    )
    validate_numeric_samples(samples)
    report = build_generation_report(
        samples,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
    )
    validate_generation_report(report)
    return samples, report


def _run_finalize(args: argparse.Namespace, state: InputState) -> int:
    run_root = _run_root(args)
    reference_root = _reference_root(args)
    if not run_root.is_dir():
        raise SystemExit(f"! run directory is missing: {run_root}")
    if not reference_root.is_dir():
        raise SystemExit(f"! reference profile directory is missing: {reference_root}")
    samples_path = run_root / NUMERIC_SAMPLES
    report_path = run_root / REPORT
    if samples_path.exists() != report_path.exists():
        raise SystemExit("! final report bundle is incomplete; refusing to overwrite it")
    with exclusive_lock(args.artifacts / ARTIFACT_LOCK):
        samples, report = _build_final_artifacts(args, state, run_root, reference_root)
        if samples_path.exists():
            try:
                old_samples = read_json(samples_path)
                old_report = read_json(report_path)
                validate_numeric_samples(old_samples)
                validate_generation_report(old_report)
            except (OSError, TypeError, ValueError) as exc:
                raise SystemExit(f"! existing final report bundle is invalid: {exc}") from exc
            if old_samples != samples or old_report != report:
                raise SystemExit("! immutable final report drift; use a new run id")
            print(f"finalize: already published ({report_path})")
            return 0
        staged_samples = Path(f"{samples_path}.tmp")
        staged_report = Path(f"{report_path}.tmp")
        try:
            write_json(staged_samples, samples)
            write_json(staged_report, report)
            # replace_files applies these in order, so report.json is published last.
            replace_files(((staged_samples, samples_path), (staged_report, report_path)))
        finally:
            staged_samples.unlink(missing_ok=True)
            staged_report.unlink(missing_ok=True)
    print(f"finalize: published text-free samples and report under {run_root}")
    return 0


def _cache_status(path: Path) -> str:  # noqa: PLR0911
    meta_path = Path(f"{path}.meta.json")
    cache_exists = path.exists()
    meta_exists = meta_path.exists()
    if not meta_exists:
        return "invalid" if cache_exists else "missing"
    try:
        meta = _parse_meta(read_json(meta_path), meta_path)
        definition = _cache_definition(cast(str, meta["stage"]))
        if not cache_exists:
            if meta["complete"]:
                return "invalid"
            return f"partial 0/{cast(int, meta['expected_count']):,}"
        rows = _read_rows(path, definition.parser)
        count = len(rows)
        expected = cast(int, meta["expected_count"])
        if meta["complete"]:
            if not rows or _cache_fingerprint(rows) != meta["cache_sha256"]:
                return "invalid"
            return f"complete {count:,}/{expected:,}"
        return f"partial {count:,}/{expected:,}"
    except (OSError, TypeError, ValueError, SystemExit):
        return "invalid"


def _marker_status(path: Path, parser: Callable[[object], object] | None = None) -> str:
    if not path.exists():
        return "missing"
    try:
        raw = read_json(path)
        if parser is not None:
            parser(raw)
    except (OSError, TypeError, ValueError):
        return "invalid"
    return "present"


def _status_root(root: Path, state: InputState | None, *, kind: str) -> None:
    print(f"{kind}: {root}")
    if not root.is_dir():
        print("  state: missing")
        return
    if state is None:
        print("  state: input unavailable; artifact details withheld")
        return
    try:
        _require_manifest(root, state)
    except SystemExit:
        print("  input: invalid")
        return
    print(f"  input: present ({len(state.cases):,} cases)")
    if kind == "reference bank":
        qg_status = _cache_status(root / QG_CACHE)
        print(f"  QG: {qg_status}")
        qg_complete = qg_status.startswith("complete ")
        question_status = _marker_status(root / QUESTION_BANK, parse_question_bank)
        if not qg_complete:
            print("  downstream: waiting for complete QG; question universe withheld")
            return
        print(f"  question bank: {question_status}")
        if question_status != "present":
            print("  downstream: waiting for finalized question bank")
            return
        print(f"  reference QA: {_cache_status(root / REFERENCE_QA_CACHE)}")
        print(f"  reference bank: {_marker_status(root / REFERENCE_BANK, parse_reference_bank)}")
        return
    generation_status = _cache_status(root / GENERATION_CACHE)
    print(f"  generation: {generation_status}")
    if generation_status.startswith("complete "):
        print(f"  prediction QA: {_cache_status(root / PREDICTION_QA_CACHE)}")
        print(f"  semantic: {_cache_status(root / SEMANTIC_CACHE)}")
        print(f"  final report: {_marker_status(root / REPORT, validate_generation_report)}")
    else:
        print("  downstream: waiting for complete generation; question/answer details withheld")


def _status(args: argparse.Namespace) -> int:  # noqa: PLR0912
    try:
        state: InputState | None = _load_input(args.input_path)
    except SystemExit:
        state = None
        print(f"input: missing or invalid ({args.input_path})")
    case_count = None if state is None else len(state.cases)
    if case_count is not None:
        print(f"input: present ({case_count:,} cases; status is read-only)")

    reference_roots: list[Path] = []
    if args.reference_profile is not None:
        reference_roots.append(_reference_root(args))
    else:
        parent = args.artifacts / "reference_banks"
        if parent.is_dir():
            reference_roots.extend(sorted(path for path in parent.iterdir() if path.is_dir()))
    if reference_roots:
        for root in reference_roots:
            _status_root(root, state, kind="reference bank")
    else:
        print("reference banks: none")

    run_roots: list[Path] = []
    if args.run_id is not None:
        run_roots.append(_run_root(args))
    else:
        for group in ("trials", "canonical"):
            parent = args.artifacts / "runs" / group
            if parent.is_dir():
                run_roots.extend(sorted(path for path in parent.iterdir() if path.is_dir()))
    if run_roots:
        for root in run_roots:
            _status_root(root, state, kind="run")
    else:
        print("runs: none")
    if args.dry_run:
        print("dry-run: no provider, model, cache, directory, or lock action performed")
    return 0


def _dispatch(args: argparse.Namespace, state: InputState, action: ActionName) -> int:
    handlers: dict[ActionName, Callable[[argparse.Namespace, InputState], int]] = {
        "generate": _run_generate,
        "generate_questions": _run_generate_questions,
        "finalize_questions": _run_finalize_questions,
        "answer_reference": _run_answer_reference,
        "finalize_reference_bank": _run_finalize_reference_bank,
        "answer_prediction": _run_answer_prediction,
        "score_semantic": _run_score_semantic,
        "finalize": _run_finalize,
    }
    return handlers[action](args, state)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    action = _action(args)
    _validate_args(args, action)
    if action is None:
        return _status(args)
    state = _load_input(args.input_path)
    return _dispatch(args, state, action)


if __name__ == "__main__":
    raise SystemExit(main())
