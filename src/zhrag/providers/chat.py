"""Strict client for the One Hub ``/v1/chat/completions`` API.

This is the third provider surface in the project, after embedding and rerank,
and it exists for one job: building and judging a TiDB evaluation set. It shares
:class:`zhrag.providers.http.JsonClient`, so the explicit User-Agent, the 403
diagnostics and the long upstream-saturation retry ladder are the same ones the
other two providers were tuned against.

Two response conditions are treated as failures rather than data:

* ``finish_reason == "length"`` means the model was cut off mid-answer. Every
  caller here asks for JSON, and truncated JSON either fails to parse or -- far
  worse -- parses into a shorter object that silently loses fields.
* An empty ``content`` with a present ``reasoning_content`` means the model spent
  its whole budget thinking. Returning ``""`` would look like a refusal.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from zhrag.providers.http import JsonClient, Transport

__all__ = [
    "REASONING_EFFORTS",
    "ChatClient",
    "ChatConfig",
    "ChatReply",
    "resolve_chat_url",
]

#: Accepted values for the OpenAI-style ``reasoning_effort`` knob. Sending an
#: unlisted value is rejected locally: the relay forwards unknown fields
#: verbatim, and an upstream that ignores a typo would quietly answer at its own
#: default effort while the run's provenance records what we asked for.
REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


def resolve_chat_url(base_url: str) -> str:
    """Append the chat-completions suffix without corrupting query parameters."""
    parts = urllib.parse.urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = f"{path}/chat/completions"
    else:
        endpoint_path = f"{path}/v1/chat/completions"
    return urllib.parse.urlunsplit(parts._replace(path=endpoint_path))


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Chat endpoint credentials and requested model identity."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, model: str | None = None) -> ChatConfig:
        """Read the ``LLM_*`` variables, optionally overriding the model.

        The override exists so one run can use two different models -- a
        generator and an independent verifier -- without a second credential.
        """
        missing = [
            name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME") if not env.get(name)
        ]
        if missing:
            raise ValueError(f".env is missing or empty for: {', '.join(missing)}")
        return cls(
            api_key=env["LLM_API_KEY"],
            base_url=env["LLM_BASE_URL"].rstrip("/"),
            model=model or env["LLM_MODEL_NAME"],
        )

    @property
    def endpoint(self) -> str:
        return resolve_chat_url(self.base_url)


@dataclass(frozen=True, slots=True)
class ChatReply:
    """One completed assistant turn, with token counts when the relay reports them."""

    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None


def _flush_print(message: str) -> None:
    print(message, flush=True)


def _int_field(usage: Mapping[str, Any], name: str) -> int | None:
    value = usage.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True, slots=True)
class ChatClient:
    """A retrying client for one OpenAI-compatible chat-completions endpoint."""

    config: ChatConfig
    reasoning_effort: str | None = "high"
    retries: int = 7
    transport: Transport | None = None
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = _flush_print

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}, "
                f"got {self.reasoning_effort!r}"
            )

    def _client(self) -> JsonClient:
        kwargs: dict[str, Any] = {
            "url": self.config.endpoint,
            "key": self.config.api_key,
            "retries": self.retries,
            "sleep": self.sleep,
            "log": self.log,
        }
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return JsonClient(**kwargs)

    def complete(self, system: str, user: str, *, json_object: bool = True) -> ChatReply:
        """Send one two-message turn and return the assistant's text."""
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system message must be a non-empty string")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user message must be a non-empty string")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        raw = self._client().post(payload)
        return self._reply(raw)

    def _reply(self, raw: Mapping[str, Any]) -> ChatReply:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SystemExit(f"! chat response has no choices: {sorted(raw)}")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise SystemExit("! chat response choice is not an object")

        finish = choice.get("finish_reason")
        if finish != "stop":
            raise SystemExit(
                f"! chat completion stopped early (finish_reason={finish!r}); "
                "a truncated answer must not be parsed as data"
            )

        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise SystemExit("! chat response has no message object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SystemExit("! chat completion returned empty content")

        usage = raw.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        model = raw.get("model")
        return ChatReply(
            content=content,
            model=model if isinstance(model, str) and model else self.config.model,
            prompt_tokens=_int_field(usage, "prompt_tokens"),
            completion_tokens=_int_field(usage, "completion_tokens"),
            reasoning_tokens=_int_field(details, "reasoning_tokens"),
        )
