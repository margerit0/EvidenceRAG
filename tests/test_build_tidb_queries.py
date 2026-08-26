from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.eval.qgen import EvalChunk, GeneratedQuery
from zhrag.ingest import IngestState, Scope, chunker_fingerprint, scope_fingerprint, write_state
from zhrag.io_utils import append_jsonl, read_json, read_jsonl, write_json, write_jsonl


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_tidb_queries.py"
    spec = importlib.util.spec_from_file_location("build_tidb_queries_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _chunk(chunk_id: str = "chunk") -> EvalChunk:
    return EvalChunk(
        chunk_id=chunk_id,
        source_key="pingcap/docs-cn:doc.md",
        ordinal=0,
        collection="core_ops",
        theme="deploy_config_tiup",
        text="标题：正文",
        approx_tokens=4,
    )


def _state(chunk_id: str = "chunk", *, scope: str | None = None) -> IngestState:
    return IngestState(
        scope=scope or scope_fingerprint(Scope.evergreen()),
        chunker_fingerprint=chunker_fingerprint(target_tokens=400, hard_max_tokens=600),
        embedding_profile="embedding",
        sparse_fingerprint="sparse",
        collection_name="tidb_chunks_v1",
        documents={
            "pingcap/docs-cn:doc.md": {
                "document_sha256": "a" * 64,
                "chunk_ids": [chunk_id],
            }
        },
    )


class TestPublishedIndexValidation:
    def test_accepts_the_exact_evergreen_chunk_set(self, tmp_path: Path) -> None:
        runner = _runner()
        write_state(tmp_path / "state.json", _state())
        runner._verify_against_index([_chunk()], tmp_path)

    def test_rejects_scope_drift(self, tmp_path: Path) -> None:
        runner = _runner()
        write_state(tmp_path / "state.json", _state(scope="wrong"))
        with pytest.raises(SystemExit, match="scope drift"):
            runner._verify_against_index([_chunk()], tmp_path)

    def test_rejects_extra_published_chunks(self, tmp_path: Path) -> None:
        runner = _runner()
        state = _state()
        write_state(
            tmp_path / "state.json",
            IngestState(
                scope=state.scope,
                chunker_fingerprint=state.chunker_fingerprint,
                embedding_profile=state.embedding_profile,
                sparse_fingerprint=state.sparse_fingerprint,
                collection_name=state.collection_name,
                documents={
                    "pingcap/docs-cn:doc.md": {
                        "document_sha256": "a" * 64,
                        "chunk_ids": ["chunk", "extra"],
                    }
                },
            ),
        )
        with pytest.raises(SystemExit, match="1 extra"):
            runner._verify_against_index([_chunk()], tmp_path)


class TestReplyCache:
    def test_loads_legacy_rows_but_reports_partial_usage(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [{"id": "a", "content": "{}", "model": "m"}])

        replies, usage, models = runner._read_cache(cache)

        assert replies == {"a": "{}"}
        assert models == {"m": 1}
        assert usage.as_dict() == {
            "calls": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "token_count_coverage": "partial: 0/1 calls carry token counts",
        }

    def test_last_write_wins_for_content_and_usage(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        append_jsonl(
            cache,
            [
                {
                    "id": "a",
                    "content": "old",
                    "model": "m",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reasoning_tokens": 3,
                },
                {
                    "id": "a",
                    "content": "new",
                    "model": "m",
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "reasoning_tokens": 6,
                },
            ],
        )

        replies, usage, models = runner._read_cache(cache)

        assert replies == {"a": "new"}
        assert models == {"m": 1}
        assert usage.as_dict() == {
            "calls": 1,
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "reasoning_tokens": 6,
            "token_count_coverage": "complete",
        }

    @pytest.mark.parametrize(
        "row",
        [
            {"content": "{}", "model": "m"},
            {"id": "a", "content": 1, "model": "m"},
            {"id": "a", "content": "{}", "model": ""},
            {
                "id": "a",
                "content": "{}",
                "model": "m",
                "prompt_tokens": 1,
            },
        ],
    )
    def test_rejects_malformed_rows(self, tmp_path: Path, row: dict[str, object]) -> None:
        runner = _runner()
        cache = tmp_path / "cache.jsonl"
        append_jsonl(cache, [row])
        with pytest.raises(SystemExit, match=r"malformed|incomplete"):
            runner._read_cache(cache)


class TestGenerationSidecarMigration:
    def test_migrates_only_the_exact_legacy_provenance(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "generation_cache.jsonl"
        append_jsonl(cache, [{"id": "a", "content": "{}", "model": "m"}])
        old = {
            "schema": "schema",
            "instructions": "combined",
            "reasoning_effort": "high",
            "model": "m",
            "stage": "generate",
        }
        new = {
            "schema": "schema",
            "cache_key": "chunk-id-v1",
            "instructions": "generation-only",
            "reasoning_effort": "high",
            "model": "m",
            "stage": "generate",
        }
        write_json(Path(f"{cache}.meta.json"), old)

        migrated = runner._migrate_generation_sidecar(
            cache,
            new,
            legacy_instructions="combined",
        )

        assert migrated
        assert read_json(Path(f"{cache}.meta.json")) == new

    def test_refuses_to_migrate_any_other_metadata(self, tmp_path: Path) -> None:
        runner = _runner()
        cache = tmp_path / "generation_cache.jsonl"
        append_jsonl(cache, [{"id": "a", "content": "{}", "model": "m"}])
        recorded = {
            "schema": "schema",
            "instructions": "other",
            "reasoning_effort": "high",
            "model": "m",
            "stage": "generate",
        }
        new = {
            "schema": "schema",
            "cache_key": "chunk-id-v1",
            "instructions": "generation-only",
            "reasoning_effort": "high",
            "model": "m",
            "stage": "generate",
        }
        sidecar = Path(f"{cache}.meta.json")
        write_json(sidecar, recorded)

        migrated = runner._migrate_generation_sidecar(
            cache,
            new,
            legacy_instructions="combined",
        )

        assert not migrated
        assert read_json(sidecar) == recorded


class TestOutputRouting:
    """A trial must never be able to land on the canonical query set."""

    def test_the_default_full_sample_is_canonical(self, tmp_path: Path) -> None:
        runner = _runner()
        args = runner._parse_args(["--artifacts", str(tmp_path)])
        assert runner._output_dir(args) == tmp_path / "eval"

    @pytest.mark.parametrize(
        ("argv", "suffix"),
        [
            (["--limit", "5"], "size-500-limit-5"),
            (["--size", "2"], "size-2-limit-all"),
            (["--size", "2", "--limit", "1"], "size-2-limit-1"),
        ],
    )
    def test_any_narrowed_run_lands_in_a_trial_directory(
        self,
        tmp_path: Path,
        argv: list[str],
        suffix: str,
    ) -> None:
        runner = _runner()
        args = runner._parse_args(["--artifacts", str(tmp_path), *argv])
        assert runner._output_dir(args) == tmp_path / "eval-trials" / suffix


class TestWrite:
    def _args(self, runner: ModuleType, tmp_path: Path) -> object:
        return runner._parse_args(["--artifacts", str(tmp_path)])

    def test_writes_both_variants_of_a_surviving_pair(self, tmp_path: Path) -> None:
        runner = _runner()
        chunk = _chunk()
        kept = [
            GeneratedQuery("d", chunk.chunk_id, "direct", "如何配置事务隔离级别", "答案", "config"),
            GeneratedQuery(
                "p", chunk.chunk_id, "paraphrase", "事务隔离级别怎样配置", "答案", "config"
            ),
        ]

        code = runner._write(
            self._args(runner, tmp_path),
            kept=kept,
            sample=[chunk],
            by_chunk={chunk.chunk_id: chunk},
            out=tmp_path,
            report_extra={},
        )

        assert code == 0
        rows = list(read_jsonl(tmp_path / "queries.jsonl"))
        assert [row["task"] for row in rows] == ["direct", "paraphrase"]
        assert all(row["gold_doc_ids"] == [chunk.chunk_id] for row in rows)
        assert read_json(tmp_path / "report.json")["complete_pairs"] == 1

    def test_a_zero_survivor_run_fails_without_truncating_the_last_good_set(
        self, tmp_path: Path
    ) -> None:
        # A failed paid run must be auditable *and* non-destructive: overwriting
        # the canonical file with an empty one would silently delete a query set
        # that cost hundreds of paid calls to build.
        chunk = _chunk()
        write_jsonl(tmp_path / "queries.jsonl", [{"query_id": "previous"}])
        runner = _runner()

        code = runner._write(
            self._args(runner, tmp_path),
            kept=[],
            sample=[chunk],
            by_chunk={chunk.chunk_id: chunk},
            out=tmp_path,
            report_extra={},
        )

        assert code == 1
        assert [row["query_id"] for row in read_jsonl(tmp_path / "queries.jsonl")] == ["previous"]
        report = read_json(tmp_path / "report.json")
        assert report["queries"] == 0
        assert report["complete_pairs"] == 0
