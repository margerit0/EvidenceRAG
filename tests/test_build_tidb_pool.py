from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from zhrag.eval.crud import Query
from zhrag.eval.qgen import EvalChunk
from zhrag.eval.tidb_runs import DENSE_LABEL, LEXICAL_LABEL, RERANK_LABEL, RRF_LABEL, TiDBRuns
from zhrag.io_utils import read_json, read_jsonl, read_text, write_text
from zhrag.providers import PairScore, RerankConfig, RerankResult, append_pair_scores


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_tidb_pool.py"
    spec = importlib.util.spec_from_file_location("build_tidb_pool_test_module", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _query(query_id: str, task: str, *, gold: str = "gold", answer: str = "答案") -> Query:
    return Query(
        query_id=query_id,
        question=f"{task}问题",
        answer=answer,
        gold_doc_ids=(gold,),
        task=task,
    )


def _experiment(runner: ModuleType, queries: tuple[Query, ...]) -> object:
    chunk = EvalChunk(
        chunk_id="gold",
        source_key="source",
        ordinal=0,
        collection="manual",
        theme="sql",
        text="正文",
        approx_tokens=2,
    )
    rows = tuple(
        {
            "query_id": query.query_id,
            "question": query.question,
            "answer": query.answer,
            "gold_doc_ids": list(query.gold_doc_ids),
            "task": query.task,
        }
        for query in queries
    )
    return runner.Experiment(
        chunks=(chunk,), corpus={"gold": "正文"}, query_rows=rows, queries=queries
    )


def _runs(query_ids: tuple[str, ...], rows: dict[str, tuple[tuple[str, ...], ...]]) -> TiDBRuns:
    return TiDBRuns(query_ids=query_ids, runs=rows)


def _rerank_experiment(runner: ModuleType, query_ids: tuple[str, ...]) -> object:
    queries = tuple(_query(query_id, "direct", gold="doc-0") for query_id in query_ids)
    corpus = {f"doc-{index}": f"正文 {index}" for index in range(runner.RERANK_REQUEST_DEPTH)}
    return runner.Experiment(chunks=(), corpus=corpus, query_rows=(), queries=queries)


def _rerank_args(tmp_path: Path, *, max_queries: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(artifacts=tmp_path, rerank=True, max_queries=max_queries)


def _complete_scores(
    runner: ModuleType,
    query_ids: Iterable[str],
    documents: tuple[str, ...],
) -> list[PairScore]:
    return [
        PairScore(query_id, doc_id, float(index))
        for query_id in query_ids
        for index, doc_id in enumerate(documents)
    ]


class _RerankClientStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(
        self,
        query: str,
        documents: list[str],
        *,
        instruction: str,
    ) -> RerankResult:
        self.calls.append((query, len(documents)))
        return RerankResult(tuple(float(index) for index in range(len(documents))), 0)


class TestRerankResume:
    def _patch_config(self, runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
        config = RerankConfig("secret", "https://example.test/v1", runner.RERANK_MODEL)
        monkeypatch.setattr(runner, "load_env", lambda _path: {})
        monkeypatch.setattr(runner.RerankConfig, "from_env", lambda _env: config)

    def test_reloads_cache_after_lock_and_skips_repaid_calls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_config(runner, monkeypatch)
        experiment = _rerank_experiment(runner, ("q1",))
        documents = tuple(experiment.corpus)
        fused = (documents,)
        cache = tmp_path / "eval" / runner.RERANK_CACHE.name
        client = _RerankClientStub()

        @contextmanager
        def fill_before_entry(_path: Path) -> Iterable[None]:
            runner.prepare_pair_score_cache(
                cache,
                runner._rerank_provenance(
                    experiment,
                    fused,
                    model=runner.RERANK_MODEL,
                    endpoint="https://example.test/v1/rerank",
                ),
            )
            append_pair_scores(cache, _complete_scores(runner, ("q1",), documents))
            yield

        monkeypatch.setattr(runner, "exclusive_lock", fill_before_entry)
        monkeypatch.setattr(runner.RerankClient, "create", lambda _config: client)

        scores, _ = runner._score_reranker(experiment, fused, _rerank_args(tmp_path))

        assert len(scores) == runner.RERANK_REQUEST_DEPTH
        assert client.calls == []

    def test_rejects_an_unexpected_pair_added_at_lock_boundary_before_provider_use(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_config(runner, monkeypatch)
        experiment = _rerank_experiment(runner, ("q1",))
        documents = tuple(experiment.corpus)
        fused = (documents,)
        cache = tmp_path / "eval" / runner.RERANK_CACHE.name
        client = _RerankClientStub()

        @contextmanager
        def inject_unexpected(_path: Path) -> Iterable[None]:
            runner.prepare_pair_score_cache(
                cache,
                runner._rerank_provenance(
                    experiment,
                    fused,
                    model=runner.RERANK_MODEL,
                    endpoint="https://example.test/v1/rerank",
                ),
            )
            append_pair_scores(cache, [PairScore("unknown", "doc-0", 0.5)])
            yield

        monkeypatch.setattr(runner, "exclusive_lock", inject_unexpected)
        monkeypatch.setattr(runner.RerankClient, "create", lambda _config: client)

        with pytest.raises(SystemExit, match="unexpected query/document pairs"):
            runner._score_reranker(experiment, fused, _rerank_args(tmp_path))
        assert client.calls == []

    def test_applies_max_queries_to_the_refreshed_missing_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_config(runner, monkeypatch)
        experiment = _rerank_experiment(runner, ("q1", "q2", "q3"))
        documents = tuple(experiment.corpus)
        fused = (documents, documents, documents)
        cache = tmp_path / "eval" / runner.RERANK_CACHE.name
        client = _RerankClientStub()

        @contextmanager
        def fill_first_query(_path: Path) -> Iterable[None]:
            runner.prepare_pair_score_cache(
                cache,
                runner._rerank_provenance(
                    experiment,
                    fused,
                    model=runner.RERANK_MODEL,
                    endpoint="https://example.test/v1/rerank",
                ),
            )
            append_pair_scores(cache, _complete_scores(runner, ("q1",), documents))
            yield

        monkeypatch.setattr(runner, "exclusive_lock", fill_first_query)
        monkeypatch.setattr(runner.RerankClient, "create", lambda _config: client)

        scores, _ = runner._score_reranker(
            experiment,
            fused,
            _rerank_args(tmp_path, max_queries=1),
        )

        assert client.calls == [("direct问题", runner.RERANK_REQUEST_DEPTH)]
        assert all(("q2", doc_id) in scores for doc_id in documents)
        assert all(("q3", doc_id) not in scores for doc_id in documents)

    def test_partial_query_is_rerequested_as_the_complete_top_100_batch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        self._patch_config(runner, monkeypatch)
        experiment = _rerank_experiment(runner, ("q1",))
        documents = tuple(experiment.corpus)
        fused = (documents,)
        cache = tmp_path / "eval" / runner.RERANK_CACHE.name
        provenance = runner._rerank_provenance(
            experiment,
            fused,
            model=runner.RERANK_MODEL,
            endpoint="https://example.test/v1/rerank",
        )
        runner.prepare_pair_score_cache(cache, provenance)
        append_pair_scores(cache, [PairScore("q1", documents[0], -1.0)])
        client = _RerankClientStub()
        monkeypatch.setattr(runner.RerankClient, "create", lambda _config: client)

        scores, _ = runner._score_reranker(experiment, fused, _rerank_args(tmp_path))

        assert client.calls == [("direct问题", runner.RERANK_REQUEST_DEPTH)]
        assert len(scores) == runner.RERANK_REQUEST_DEPTH
        assert scores[("q1", documents[0])] == 0.0


class TestPairedPools:
    def test_unions_both_surfaces_and_all_four_systems(self) -> None:
        runner = _runner()
        queries = (_query("direct:x", "direct"), _query("paraphrase:x", "paraphrase"))
        runs = _runs(
            tuple(query.query_id for query in queries),
            {
                LEXICAL_LABEL: (("l-direct", "shared"), ("l-para", "shared")),
                DENSE_LABEL: (("d-direct", "shared"), ("d-para", "shared")),
                RRF_LABEL: (("r-direct", "shared"), ("r-para", "shared")),
                RERANK_LABEL: (("rr-direct", "shared"), ("rr-para", "shared")),
            },
        )

        units, contribution, exclusive = runner._paired_pools(
            _experiment(runner, queries),
            runs,
            pool_depth=1,
        )

        assert len(units) == 1
        unit = units[0]
        assert set(unit.candidates) == {
            "gold",
            "l-direct",
            "l-para",
            "d-direct",
            "d-para",
            "r-direct",
            "r-para",
            "rr-direct",
            "rr-para",
        }
        assert unit.questions == ("direct问题", "paraphrase问题")
        assert contribution == {
            LEXICAL_LABEL: 2,
            DENSE_LABEL: 2,
            RRF_LABEL: 2,
            RERANK_LABEL: 2,
        }
        assert exclusive == {
            LEXICAL_LABEL: 2,
            DENSE_LABEL: 2,
            RRF_LABEL: 2,
            RERANK_LABEL: 2,
        }

    def test_does_not_retruncate_the_cross_surface_union(self) -> None:
        """Depth 2 per surface means up to 4, not 2, candidates per system."""
        runner = _runner()
        queries = (_query("direct:x", "direct"), _query("paraphrase:x", "paraphrase"))
        surface = (("a", "b"), ("c", "d"))
        runs = _runs(
            tuple(query.query_id for query in queries),
            {label: surface for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)},
        )
        units, _, _ = runner._paired_pools(_experiment(runner, queries), runs, pool_depth=2)
        assert set(units[0].candidates) == {"gold", "a", "b", "c", "d"}

    def test_pairs_real_qgen_order_by_generating_chunk_not_query_id_suffix(self) -> None:
        runner = _runner()
        # qgen sorts by the full public id: all direct rows precede all
        # paraphrases, and each surface has its own content-derived hash.
        queries = (
            _query("direct:hash-a", "direct", gold="chunk-a", answer="答案 A"),
            _query("direct:hash-b", "direct", gold="chunk-b", answer="答案 B"),
            _query(
                "paraphrase:other-b",
                "paraphrase",
                gold="chunk-b",
                answer="答案 B",
            ),
            _query(
                "paraphrase:other-a",
                "paraphrase",
                gold="chunk-a",
                answer="答案 A",
            ),
        )
        per_query = (("a-direct",), ("b-direct",), ("b-para",), ("a-para",))
        runs = _runs(
            tuple(query.query_id for query in queries),
            {label: per_query for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)},
        )

        units, _, _ = runner._paired_pools(_experiment(runner, queries), runs, pool_depth=1)

        assert [unit.chunk_id for unit in units] == ["chunk-a", "chunk-b"]
        assert units[0].query_ids == ("direct:hash-a", "paraphrase:other-a")
        assert set(units[0].candidates) == {"chunk-a", "a-direct", "a-para"}
        assert units[1].query_ids == ("direct:hash-b", "paraphrase:other-b")
        assert set(units[1].candidates) == {"chunk-b", "b-direct", "b-para"}

    def test_rejects_odd_or_incomplete_pairs(self) -> None:
        runner = _runner()
        one = (_query("direct:x", "direct"),)
        # TiDBRuns requires all four labels but permits one aligned query.
        rows = {label: (("a",),) for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)}
        with pytest.raises(SystemExit, match="odd"):
            runner._paired_pools(_experiment(runner, one), _runs(("direct:x",), rows), pool_depth=1)

        bad = (
            _query("direct:x", "direct", gold="chunk-a"),
            _query("paraphrase:y", "paraphrase", gold="chunk-b"),
        )
        rows2 = {label: (("a",), ("b",)) for label in rows}
        with pytest.raises(SystemExit, match="incomplete query pair"):
            runner._paired_pools(
                _experiment(runner, bad),
                _runs(tuple(query.query_id for query in bad), rows2),
                pool_depth=1,
            )

    def test_rejects_pair_answer_or_gold_drift(self) -> None:
        runner = _runner()
        bad = (
            _query("direct:x", "direct"),
            _query("paraphrase:x", "paraphrase", answer="另一个答案"),
        )
        rows = {
            label: (("a",), ("b",))
            for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)
        }
        with pytest.raises(SystemExit, match="share one gold and answer"):
            runner._paired_pools(
                _experiment(runner, bad),
                _runs(tuple(query.query_id for query in bad), rows),
                pool_depth=1,
            )


class TestArtifactBundle:
    def test_writes_a_self_contained_pool_and_publishes_report_last(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        queries = (_query("direct:a", "direct"), _query("paraphrase:b", "paraphrase"))
        runs = _runs(
            tuple(query.query_id for query in queries),
            {
                label: (("candidate",), ("candidate",))
                for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)
            },
        )
        targets: list[str] = []
        original_replace = runner.replace_files

        def record_replacements(staged: tuple[tuple[Path, Path], ...]) -> None:
            pairs = tuple(staged)
            targets.extend(Path(target).name for _source, target in pairs)
            original_replace(pairs)

        monkeypatch.setattr(runner, "replace_files", record_replacements)
        args = argparse.Namespace(artifacts=tmp_path)

        runner._write_runs(
            _experiment(runner, queries),
            runs,
            args,
            pool_depth=1,
            rerank_provenance={"model": "reranker"},
        )

        pool = list(read_jsonl(tmp_path / "eval" / "pool.jsonl"))
        assert pool[0]["schema"] == runner.POOL_SCHEMA
        assert pool[0]["questions"] == ["direct问题", "paraphrase问题"]
        assert pool[0]["answer"] == "答案"
        assert targets == ["runs.jsonl", "pool.jsonl", "pool_report.json"]
        assert read_json(tmp_path / "eval" / "pool_report.json")["pairs"] == 1

    def test_failed_staging_keeps_previous_bundle_and_cleans_temporary_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        runner = _runner()
        queries = (_query("direct:a", "direct"), _query("paraphrase:b", "paraphrase"))
        runs = _runs(
            tuple(query.query_id for query in queries),
            {
                label: (("candidate",), ("candidate",))
                for label in (LEXICAL_LABEL, DENSE_LABEL, RRF_LABEL, RERANK_LABEL)
            },
        )
        eval_root = tmp_path / "eval"
        for name in ("runs.jsonl", "pool.jsonl", "pool_report.json"):
            write_text(eval_root / name, f"old {name}")
        monkeypatch.setattr(
            runner,
            "write_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            runner._write_runs(
                _experiment(runner, queries),
                runs,
                argparse.Namespace(artifacts=tmp_path),
                pool_depth=1,
                rerank_provenance={"model": "reranker"},
            )

        for name in ("runs.jsonl", "pool.jsonl", "pool_report.json"):
            assert read_text(eval_root / name) == f"old {name}"
            assert not (eval_root / f"{name}.tmp").exists()


class TestArguments:
    def test_default_is_offline(self) -> None:
        runner = _runner()
        args = runner._parse_args([])
        assert not args.embed
        assert not args.rerank
        assert args.pool_depth == 20

    def test_output_paths_follow_the_artifact_root(self, tmp_path: Path) -> None:
        runner = _runner()
        args = argparse.Namespace(artifacts=tmp_path)
        assert args.artifacts / "eval" / runner.RUNS.name == tmp_path / "eval" / "runs.jsonl"
