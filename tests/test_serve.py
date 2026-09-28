"""Composition-root tests without a live provider or Milvus process."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from test_service import FakeRetriever
from zhrag.ingest import IngestState, write_state
from zhrag.io_utils import write_json, write_jsonl, write_text
from zhrag.lexical import build_sparse_index, write_sparse_index
from zhrag.providers.direct import DIRECT_CONTRACT, DirectTransport

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "serve.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("serve_test_module", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/serve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


serve = load_script()


def test_agent_composition_is_explicit_and_uses_no_chat_retries(tmp_path: Path) -> None:
    fake = FakeRetriever()
    assert serve._build_agent(serve._parse_args([]), fake, index_identity="test") is None
    env = tmp_path / ".env"
    write_text(
        env, "LLM_API_KEY=synthetic\nLLM_BASE_URL=https://example.invalid\nLLM_MODEL_NAME=model\n"
    )
    args = serve._parse_args(["--enable-agent", "--env", str(env)])
    agent = serve._build_agent(args, fake, index_identity="test")
    assert agent.generator.max_retries == 0
    assert agent.generator.stream is False
    assert agent.settings.max_steps == 10
    assert agent.settings.review_answers is False
    assert agent.settings.plan_investigation is False
    assert agent.retriever is fake
    # Model calls never consult environment or system proxies; the contract is fingerprinted.
    assert isinstance(agent.generator.transport, DirectTransport)
    assert agent.generator.transport.timeout_seconds == 60.0
    assert agent.generator.transport_contract == DIRECT_CONTRACT
    assert DIRECT_CONTRACT != "urllib-default-v1"
    reviewed = serve._build_agent(
        serve._parse_args(["--enable-agent", "--env", str(env), "--agent-review-answers"]),
        fake,
        index_identity="test",
    )
    assert reviewed.settings.review_answers is True
    assert reviewed.profile_fingerprint != agent.profile_fingerprint
    planned = serve._build_agent(
        serve._parse_args(["--enable-agent", "--env", str(env), "--agent-plan-investigation"]),
        fake,
        index_identity="test",
    )
    assert planned.settings.plan_investigation is True
    assert planned.settings.review_answers is False
    assert planned.profile_fingerprint != agent.profile_fingerprint
    streamed = serve._build_agent(
        serve._parse_args(["--enable-agent", "--env", str(env), "--generation-stream"]),
        fake,
        index_identity="test",
    )
    assert streamed.generator.stream is True
    assert streamed.generator.transport is None
    assert streamed.profile_fingerprint != agent.profile_fingerprint


@pytest.mark.parametrize("retries", [5, 9])
def test_agent_retries_are_explicit_bounded_and_fingerprinted(tmp_path: Path, retries: int) -> None:
    fake = FakeRetriever()
    env = tmp_path / ".env"
    write_text(
        env, "LLM_API_KEY=synthetic\nLLM_BASE_URL=https://example.invalid\nLLM_MODEL_NAME=model\n"
    )
    base = ["--enable-agent", "--env", str(env)]
    default = serve._build_agent(serve._parse_args(base), fake, index_identity="test")
    ladder = serve._build_agent(
        serve._parse_args([*base, "--agent-generation-retries", str(retries)]),
        fake,
        index_identity="test",
    )
    assert ladder.generator.max_retries == retries
    assert ladder.generator.profile_fingerprint != default.generator.profile_fingerprint
    assert ladder.profile_fingerprint != default.profile_fingerprint
    for value in ("-1", "10"):
        with pytest.raises(ValueError, match="agent-generation-retries"):
            serve._build_agent(
                serve._parse_args([*base, "--agent-generation-retries", value]),
                fake,
                index_identity="test",
            )


def test_direct_loopback_extends_no_proxy_without_dropping_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.environ is case-insensitive on Windows, so both spellings may share one key.
    monkeypatch.setenv("no_proxy", "internal.example, localhost")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    serve._direct_loopback()
    for name in ("no_proxy", "NO_PROXY"):
        entries = os.environ[name].split(",")
        assert len(entries) == len(set(entries))
        assert entries[0] == "internal.example" and " " not in os.environ[name]
        assert {"localhost", "127.0.0.1", "::1"} <= set(entries)
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"  # not our decision to unset


def test_agent_cache_conflict_fails_before_artifact_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_path: Path) -> None:
        raise AssertionError("must not load")

    monkeypatch.setattr(serve, "_load_published_artifacts", fail)
    assert serve.main(["--enable-agent", "--query-cache", "cache.jsonl", "--no-rerank"]) == 1


class FakeStore:
    def __init__(
        self,
        *,
        target: str | None = "tidb_chunks_v1",
        dimensions: int = 4096,
        rows: int = 1,
    ) -> None:
        self.target = target
        self._dimensions = dimensions
        self.rows = rows
        self.ensure_calls = 0
        self.alias_calls: list[str] = []

    @property
    def dense_dimensions(self) -> int:
        return self._dimensions

    def ensure_collection(self) -> None:
        self.ensure_calls += 1

    def alias_target(self, alias: str) -> str | None:
        self.alias_calls.append(alias)
        return self.target

    def count(self) -> int:
        return self.rows

    def close(self) -> None:
        pass


def state(*, fingerprint: str = "f" * 64, collection: str = "tidb_chunks_v1") -> IngestState:
    return IngestState(
        scope="scope",
        chunker_fingerprint="chunker",
        embedding_profile=serve.DOCUMENT_EMBEDDING_PROFILE,
        sparse_fingerprint=fingerprint,
        collection_name=collection,
        documents={
            "source:a": {
                "document_sha256": "a" * 64,
                "metadata_fingerprint": "m",
                "chunk_ids": ["doc-a"],
            }
        },
    )


def write_artifacts(path: Path, *, drift: bool = False) -> None:
    sparse = build_sparse_index({"doc-a": "向量检索"}).index
    published = state(fingerprint="0" * 64 if drift else sparse.fingerprint)
    write_state(path / "state.json", published)
    write_sparse_index(path / "sparse_index.json", sparse)


class TestCleanImport:
    def test_import_does_not_load_provider_modules_or_touch_env(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text("not-valid-utf8-surrogate", encoding="utf-8")
        probe = (
            "import importlib.util, json, sys; "
            f"spec=importlib.util.spec_from_file_location('serve_probe',{str(SCRIPT)!r}); "
            "m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; "
            "spec.loader.exec_module(m); "
            "print(json.dumps(sorted(n for n in sys.modules "
            "if n.startswith('zhrag.providers'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []


class TestPublishedArtifacts:
    def test_missing_artifacts_fail_before_provider_composition(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="missing"):
            serve._load_published_artifacts(tmp_path)

    def test_sparse_fingerprint_drift_fails_closed(self, tmp_path: Path) -> None:
        write_artifacts(tmp_path, drift=True)
        with pytest.raises(ValueError, match="does not match"):
            serve._load_published_artifacts(tmp_path)

    def test_state_and_sparse_chunk_counts_must_agree(self, tmp_path: Path) -> None:
        sparse = build_sparse_index({"doc-a": "甲", "doc-b": "乙"}).index
        write_state(tmp_path / "state.json", state(fingerprint=sparse.fingerprint))
        write_sparse_index(tmp_path / "sparse_index.json", sparse)
        with pytest.raises(ValueError, match="chunk count"):
            serve._load_published_artifacts(tmp_path)

    def test_valid_bundle_returns_typed_contracts(self, tmp_path: Path) -> None:
        write_artifacts(tmp_path)
        loaded_state, sparse = serve._load_published_artifacts(tmp_path)
        assert loaded_state.collection_name == "tidb_chunks_v1"
        assert sparse.document_count == 1


class TestProfiles:
    def test_no_rerank_profile_is_not_named_as_paid_rerank(self) -> None:
        settings = serve._settings(no_rerank=True)
        assert settings.profile_name == "tidb-docs-exact-rrf10-no-rerank-v1"
        assert settings.rerank_profile == "disabled-identity-fused-order-v1"
        assert "rerank100to50" not in settings.profile_name

    def test_cache_backed_profile_is_distinct_and_provider_free(self) -> None:
        settings = serve._settings(no_rerank=True, cached_query=True)
        assert settings.profile_name == "tidb-docs-exact-rrf10-cached-query-no-rerank-v1"
        assert settings.embedding_profile.startswith("cached-")
        assert settings.rerank_profile == "disabled-identity-fused-order-v1"

    def test_cache_backed_embedding_cannot_enable_rerank(self) -> None:
        with pytest.raises(ValueError, match="requires no-rerank"):
            serve._settings(no_rerank=False, cached_query=True)

    def test_live_profile_remains_explicit(self) -> None:
        settings = serve._settings(no_rerank=False)
        assert settings.profile_name == "tidb-docs-exact-rrf10-rerank100to50-v1"
        assert settings.rerank_profile == "qwen3-reranker-8b-tidb-v1"

    def test_identity_reranker_preserves_request_order(self) -> None:
        assert serve._IdentityReranker().score("ignored", ["a", "b", "c"]) == (
            0.0,
            -1.0,
            -2.0,
        )


class TestCachedQueries:
    def test_loads_exact_fixture_text_with_valid_provenance(self, tmp_path: Path) -> None:
        cache = tmp_path / "query_embeddings.jsonl"
        fixture = tmp_path / "queries.jsonl"
        vector = [0.0] * serve.DENSE_WIDTH
        vector[0] = 1.0
        write_jsonl(fixture, [{"query_id": "q1", "question": "如何备份？"}])
        write_jsonl(cache, [{"doc_id": "q1", "embedding": vector}])
        write_json(
            Path(f"{cache}.meta.json"),
            {"model": serve.EMBEDDING_MODEL, "prompt": serve.QUERY_PROMPT},
        )

        encoder = serve._load_cached_queries(cache, fixture)
        assert encoder.encode("如何备份？") == tuple(vector)
        with pytest.raises(ValueError, match="absent"):
            encoder.encode("未缓存的问题")

    def test_rejects_incomplete_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "query_embeddings.jsonl"
        fixture = tmp_path / "queries.jsonl"
        write_jsonl(
            fixture,
            [
                {"query_id": "q1", "question": "问题一"},
                {"query_id": "q2", "question": "问题二"},
            ],
        )
        write_jsonl(
            cache,
            [{"doc_id": "q1", "embedding": [0.0] * serve.DENSE_WIDTH}],
        )
        write_json(
            Path(f"{cache}.meta.json"),
            {"model": serve.EMBEDDING_MODEL, "prompt": serve.QUERY_PROMPT},
        )

        with pytest.raises(ValueError, match="incomplete"):
            serve._load_cached_queries(cache, fixture)

    def test_rejects_provenance_drift(self, tmp_path: Path) -> None:
        cache = tmp_path / "query_embeddings.jsonl"
        fixture = tmp_path / "queries.jsonl"
        write_jsonl(fixture, [{"query_id": "q1", "question": "问题"}])
        write_jsonl(
            cache,
            [{"doc_id": "q1", "embedding": [0.0] * serve.DENSE_WIDTH}],
        )
        write_json(
            Path(f"{cache}.meta.json"),
            {"model": "wrong", "prompt": serve.QUERY_PROMPT},
        )

        with pytest.raises(ValueError, match="provenance"):
            serve._load_cached_queries(cache, fixture)


class TestStoreValidation:
    def test_alias_must_target_the_published_collection(self) -> None:
        store = FakeStore(target="tidb_chunks_old")
        with pytest.raises(ValueError, match="alias target"):
            serve._verify_store(cast(Any, store), alias="tidb_chunks", state=state())
        assert store.ensure_calls == 0
        assert store.alias_calls == ["tidb_chunks"]

    def test_store_dimension_must_match_profile(self) -> None:
        store = FakeStore(dimensions=1024)
        with pytest.raises(ValueError, match="dense width"):
            serve._verify_store(cast(Any, store), alias="tidb_chunks", state=state())
        assert store.ensure_calls == 0

    def test_store_row_count_must_match_published_state(self) -> None:
        store = FakeStore(rows=2)
        with pytest.raises(ValueError, match="row count"):
            serve._verify_store(cast(Any, store), alias="tidb_chunks", state=state())

    def test_matching_alias_passes(self) -> None:
        store = FakeStore()
        serve._verify_store(cast(Any, store), alias="tidb_chunks", state=state())
        assert store.ensure_calls == 1


class TestAnswerComposition:
    def test_no_output_cap_reaches_answer_and_agent_generators(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        write_text(
            env,
            "LLM_API_KEY=synthetic\nLLM_BASE_URL=https://example.invalid\nLLM_MODEL_NAME=model\n",
        )
        args = serve._parse_args(
            [
                "--enable-generation",
                "--enable-agent",
                "--env",
                str(env),
                "--generation-no-token-limit",
            ]
        )
        assert serve._build_answerer(args).generator.max_output_tokens is None
        assert (
            serve._build_agent(
                args, FakeRetriever(), index_identity="test"
            ).generator.max_output_tokens
            is None
        )
        with pytest.raises(SystemExit):
            serve._parse_args(["--generation-no-token-limit", "--generation-max-tokens", "4096"])

    def test_default_does_not_read_chat_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = serve._parse_args([])
        assert args.enable_generation is False
        assert args.generation_retries == 15
        assert serve._build_answerer(args) is None

    @pytest.mark.parametrize(
        "options,expected_retries",
        [([], 15), (["--generation-retries", "0"], 0), (["--generation-retries", "3"], 3)],
    )
    def test_explicit_generation_uses_configured_chat_without_network(
        self,
        tmp_path: Path,
        options: list[str],
        expected_retries: int,
    ) -> None:
        env = tmp_path / ".env"
        write_text(
            env,
            "LLM_API_KEY=synthetic\nLLM_BASE_URL=https://example.invalid\nLLM_MODEL_NAME=model\n",
        )
        args = serve._parse_args(["--enable-generation", "--env", str(env), *options])
        answerer = serve._build_answerer(args)
        assert answerer is not None
        assert answerer.generator.config.model == "model"
        assert answerer.generator.max_output_tokens == 2048
        assert answerer.generator.reasoning_effort is None
        assert answerer.generator.max_retries == expected_retries
        assert answerer.generator.transport_contract == DIRECT_CONTRACT

    def test_generation_and_query_cache_fail_before_loading_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_path: Path) -> None:
            raise AssertionError("must not read artifacts")

        monkeypatch.setattr(serve, "_load_published_artifacts", fail)
        assert (
            serve.main(["--enable-generation", "--query-cache", "cache.jsonl", "--no-rerank"]) == 1
        )

    @pytest.mark.parametrize(
        "option,value",
        [
            ("--generation-timeout", "nan"),
            ("--generation-timeout", "0"),
            ("--generation-retries", "-1"),
            ("--generation-retries", "16"),
            ("--generation-max-tokens", "8193"),
            ("--context-tokens", "0"),
            ("--context-passages", "21"),
        ],
    )
    def test_invalid_generation_options_fail_before_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        option: str,
        value: str,
    ) -> None:
        def fail(_path: Path) -> None:
            raise AssertionError("must not read artifacts")

        monkeypatch.setattr(serve, "_load_published_artifacts", fail)
        assert serve.main(["--enable-generation", option, value]) == 1


class TestCli:
    def test_main_sanitizes_missing_artifact_path(self, tmp_path: Path, capsys: Any) -> None:
        secret = tmp_path / "SENTINEL_PRIVATE_PATH"
        assert serve.main(["--artifacts", str(secret)]) == 1
        output = capsys.readouterr()
        assert "service startup failed (ValueError)" in output.out
        assert "SENTINEL_PRIVATE_PATH" not in output.out
        assert output.err == ""

    def test_cli_has_no_worker_option(self) -> None:
        args = serve._parse_args([])
        assert not hasattr(args, "workers")

    def test_query_cache_requires_no_rerank_before_artifacts_are_loaded(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        cache = tmp_path / "cache.jsonl"
        assert serve.main(["--query-cache", str(cache)]) == 1
        assert "ValueError" in capsys.readouterr().out

    def test_query_fixture_requires_query_cache(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        fixture = tmp_path / "queries.jsonl"
        assert serve.main(["--query-fixture", str(fixture)]) == 1
        assert "ValueError" in capsys.readouterr().out

    @pytest.mark.parametrize("port", [0, 65_536])
    def test_invalid_port_fails_before_store_creation(
        self,
        port: int,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        write_artifacts(tmp_path)
        assert serve.main(["--artifacts", str(tmp_path), "--port", str(port)]) == 1
        assert "ValueError" in capsys.readouterr().out
