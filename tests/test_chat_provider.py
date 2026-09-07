from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from zhrag.io_utils import append_jsonl, read_json, write_text
from zhrag.providers.cache import (
    load_cache_provenance,
    prepare_cache_sidecar,
    validate_cache_sidecar,
)
from zhrag.providers.chat import ChatClient, ChatConfig, resolve_chat_url


class Recorder:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> bytes:
        self.requests.append(request)
        return self.body


class TestChatConfig:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            ("https://host", "https://host/v1/chat/completions"),
            ("https://host/", "https://host/v1/chat/completions"),
            ("https://host/v1", "https://host/v1/chat/completions"),
            ("https://host/v1/chat/completions", "https://host/v1/chat/completions"),
        ],
    )
    def test_resolves_supported_base_url_forms(self, base: str, expected: str) -> None:
        assert resolve_chat_url(base) == expected

    def test_reads_llm_variables_and_allows_model_override(self) -> None:
        config = ChatConfig.from_env(
            {
                "LLM_API_KEY": "secret",
                "LLM_BASE_URL": "https://host/v1/",
                "LLM_MODEL_NAME": "generator",
            },
            model="verifier",
        )
        assert config.api_key == "secret"
        assert config.model == "verifier"
        assert config.endpoint == "https://host/v1/chat/completions"

    @pytest.mark.parametrize("missing", ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME"])
    def test_rejects_missing_variables(self, missing: str) -> None:
        env = {
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://host",
            "LLM_MODEL_NAME": "model",
        }
        del env[missing]
        with pytest.raises(ValueError, match=missing):
            ChatConfig.from_env(env)

    def test_rejects_unknown_reasoning_effort(self) -> None:
        config = ChatConfig("secret", "https://host", "model")
        with pytest.raises(ValueError, match="reasoning_effort"):
            ChatClient(config, reasoning_effort="turbo")


class TestChatClient:
    def test_posts_json_mode_and_extracts_usage(self) -> None:
        recorder = Recorder(
            json.dumps(
                {
                    "model": "actual-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"usable":true}'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "completion_tokens_details": {"reasoning_tokens": 5},
                    },
                }
            ).encode("utf-8")
        )
        client = ChatClient(
            ChatConfig("secret", "https://host", "model"),
            transport=recorder,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )
        reply = client.complete("system", "user")
        assert reply.content == '{"usable":true}'
        assert reply.model == "actual-model"
        assert (reply.prompt_tokens, reply.completion_tokens, reply.reasoning_tokens) == (12, 8, 5)

    def test_preserves_missing_usage_as_unknown(self) -> None:
        recorder = Recorder(
            json.dumps(
                {
                    "model": "model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "{}"},
                        }
                    ],
                }
            ).encode("utf-8")
        )
        client = ChatClient(
            ChatConfig("secret", "https://host", "model"),
            transport=recorder,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )
        reply = client.complete("system", "user")
        assert reply.model == "model"
        assert (reply.prompt_tokens, reply.completion_tokens, reply.reasoning_tokens) == (
            None,
            None,
            None,
        )

    @pytest.mark.parametrize("served_model", [None, "", 42])
    def test_rejects_missing_or_invalid_served_model(self, served_model: object) -> None:
        recorder = Recorder(
            json.dumps(
                {
                    "model": served_model,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "{}"},
                        }
                    ],
                }
            ).encode("utf-8")
        )
        client = ChatClient(
            ChatConfig("secret", "https://host", "model"),
            transport=recorder,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )
        with pytest.raises(SystemExit, match="served model"):
            client.complete("system", "user")

        request_data = recorder.requests[0].data
        assert isinstance(request_data, bytes)
        payload = json.loads(request_data)
        assert payload["reasoning_effort"] == "high"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

    @pytest.mark.parametrize(
        "response",
        [
            {"choices": []},
            {"choices": [{"finish_reason": "length", "message": {"content": "x"}}]},
            {"choices": [{"finish_reason": None, "message": {"content": "{}"}}]},
            {"choices": [{"finish_reason": "", "message": {"content": "{}"}}]},
            {"choices": [{"message": {"content": "{}"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        ],
    )
    def test_rejects_incomplete_responses(self, response: dict[str, Any]) -> None:
        recorder = Recorder(json.dumps(response).encode("utf-8"))
        client = ChatClient(
            ChatConfig("secret", "https://host", "model"),
            transport=recorder,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )
        with pytest.raises(SystemExit):
            client.complete("system", "user")

    def test_rejects_blank_messages_before_network(self) -> None:
        recorder = Recorder(b"{}")
        client = ChatClient(ChatConfig("secret", "https://host", "model"), transport=recorder)
        with pytest.raises(ValueError, match="system message"):
            client.complete(" ", "user")
        assert not recorder.requests

    def test_adds_optional_completion_token_cap(self) -> None:
        recorder = Recorder(
            json.dumps(
                {
                    "model": "model",
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "{}"}},
                    ],
                }
            ).encode("utf-8")
        )
        client = ChatClient(
            ChatConfig("secret", "https://host", "model"),
            transport=recorder,
            sleep=lambda _seconds: None,
            log=lambda _message: None,
        )

        client.complete("system", "user", max_output_tokens=2048)

        request_data = recorder.requests[0].data
        assert isinstance(request_data, bytes)
        assert json.loads(request_data)["max_completion_tokens"] == 2048

    @pytest.mark.parametrize("value", [0, -1, True, False, 1.5])
    def test_rejects_invalid_completion_token_cap(self, value: object) -> None:
        recorder = Recorder(b"{}")
        client = ChatClient(ChatConfig("secret", "https://host", "model"), transport=recorder)
        with pytest.raises(ValueError, match="max_output_tokens"):
            client.complete("system", "user", max_output_tokens=value)  # type: ignore[arg-type]
        assert not recorder.requests


class TestCacheSidecar:
    def test_creates_loads_and_validates_generic_provenance(self, tmp_path: Path) -> None:
        cache = tmp_path / "replies.jsonl"
        provenance = {"schema": "qgen", "model": "m", "stage": "generate"}
        prepare_cache_sidecar(cache, provenance, label="qgen cache")
        assert load_cache_provenance(cache, label="qgen cache") == provenance
        validate_cache_sidecar(cache, provenance, label="qgen cache")

    def test_refuses_drift_after_rows_exist(self, tmp_path: Path) -> None:
        cache = tmp_path / "replies.jsonl"
        prepare_cache_sidecar(cache, {"model": "old"}, label="qgen cache")
        append_jsonl(cache, [{"id": "q", "content": "{}"}])
        with pytest.raises(SystemExit, match="metadata drift"):
            prepare_cache_sidecar(cache, {"model": "new"}, label="qgen cache")

    def test_refuses_adopting_rows_without_provenance(self, tmp_path: Path) -> None:
        cache = tmp_path / "replies.jsonl"
        write_text(cache, '{"id":"q","content":"{}"}\n')
        with pytest.raises(SystemExit, match="without provenance"):
            prepare_cache_sidecar(cache, {"model": "m"}, label="qgen cache")

    def test_does_not_create_sidecar_during_read_only_validation(self, tmp_path: Path) -> None:
        cache = tmp_path / "missing.jsonl"
        with pytest.raises(SystemExit, match="provenance is absent"):
            validate_cache_sidecar(cache, {"model": "m"}, label="qgen cache")
        assert not Path(f"{cache}.meta.json").exists()

    def test_sidecar_is_exact_json(self, tmp_path: Path) -> None:
        cache = tmp_path / "replies.jsonl"
        prepare_cache_sidecar(cache, {"model": "m"}, label="qgen cache")
        assert read_json(Path(f"{cache}.meta.json")) == {"model": "m"}
