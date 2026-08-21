"""Strict client for the One Hub ``/v1/rerank`` API."""

from __future__ import annotations

import math
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zhrag.providers.http import JsonClient
from zhrag.tokens import estimate_tokens

__all__ = [
    "DEFAULT_RERANK_INSTRUCTION",
    "RerankClient",
    "RerankConfig",
    "RerankResult",
    "estimate_rerank_tokens",
    "resolve_rerank_url",
]

DEFAULT_RERANK_INSTRUCTION = (
    "Given a Chinese question, retrieve news passages that contain the evidence "
    "needed to answer it."
)

_RERANK_SYSTEM = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Rerank endpoint credentials and model identity."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RerankConfig:
        missing = [
            name
            for name in ("ReRank_API_KEY", "ReRank_BASE_URL", "ReRank_MODEL_NAME")
            if not env.get(name)
        ]
        if missing:
            raise ValueError(f".env is missing or empty for: {', '.join(missing)}")
        return cls(
            api_key=env["ReRank_API_KEY"],
            base_url=env["ReRank_BASE_URL"].rstrip("/"),
            model=env["ReRank_MODEL_NAME"],
        )

    @property
    def endpoint(self) -> str:
        return resolve_rerank_url(self.base_url)


def resolve_rerank_url(base_url: str) -> str:
    """Append the rerank suffix without corrupting URL query parameters."""
    parts = urllib.parse.urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/rerank"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = f"{path}/rerank"
    else:
        endpoint_path = f"{path}/v1/rerank"
    return urllib.parse.urlunsplit(parts._replace(path=endpoint_path))


def estimate_rerank_tokens(
    query: str,
    documents: Sequence[str],
    *,
    instruction: str = DEFAULT_RERANK_INSTRUCTION,
) -> int:
    """Approximate the provider's one scaffold-per-document prompt tokens."""
    return sum(
        estimate_tokens(
            f"{_RERANK_SYSTEM}<Instruct>: {instruction}\n<Query>: {query}\n"
            f"<Document>: {document}{_RERANK_SUFFIX}"
        )
        for document in documents
    )


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Validated scores in request order plus reported token usage."""

    scores: tuple[float, ...]
    prompt_tokens: int | None


def _validated_documents(documents: Sequence[str]) -> list[str]:
    if isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of strings, not a scalar string")
    sent = list(documents)
    if not sent:
        raise ValueError("documents must be non-empty")
    if any(not isinstance(document, str) for document in sent):
        raise TypeError("every document must be a string")
    return sent


def _scores_in_request_order(raw_results: object, expected_count: int) -> tuple[float, ...]:
    if not isinstance(raw_results, list):
        raise SystemExit("! malformed rerank response: results is not a list")
    if len(raw_results) != expected_count:
        raise SystemExit(
            "! malformed rerank response: "
            f"got {len(raw_results)} results for {expected_count} documents"
        )

    by_index: dict[int, float] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            raise SystemExit("! malformed rerank response: result is not an object")
        index = raw.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise SystemExit("! malformed rerank response: index is not an integer")
        if not 0 <= index < expected_count or index in by_index:
            raise SystemExit(
                f"! malformed rerank response: duplicate or out-of-range index {index}"
            )
        raw_score = raw.get("relevance_score")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise SystemExit("! malformed rerank response: relevance_score is not numeric")
        try:
            score = float(raw_score)
        except OverflowError as exc:
            raise SystemExit("! malformed rerank response: relevance_score is not finite") from exc
        if not math.isfinite(score):
            raise SystemExit("! malformed rerank response: relevance_score is not finite")
        by_index[index] = score

    if set(by_index) != set(range(expected_count)):
        raise SystemExit("! malformed rerank response: result indices are incomplete")
    return tuple(by_index[i] for i in range(expected_count))


def _prompt_tokens(response: Mapping[str, Any]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    raw_tokens = usage.get("prompt_tokens")
    if isinstance(raw_tokens, int) and not isinstance(raw_tokens, bool) and raw_tokens >= 0:
        return raw_tokens
    return None


@dataclass(frozen=True, slots=True)
class RerankClient:
    """Call and validate reranking without retaining echoed document text."""

    config: RerankConfig
    http: JsonClient

    @classmethod
    def create(cls, config: RerankConfig, *, retries: int = 7) -> RerankClient:
        return cls(
            config=config,
            http=JsonClient(
                url=config.endpoint,
                key=config.api_key,
                retries=retries,
            ),
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> RerankResult:
        """Return one finite score per input document in its original order."""
        sent = _validated_documents(documents)
        response = self.http.post(
            {
                "model": self.config.model,
                "query": query,
                "documents": sent,
                "instruction": instruction,
                "top_n": len(sent),
                "return_documents": False,
            }
        )
        return RerankResult(
            scores=_scores_in_request_order(response.get("results"), len(sent)),
            prompt_tokens=_prompt_tokens(response),
        )
