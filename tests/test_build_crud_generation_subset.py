from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from zhrag.eval.crud_generation import build_generation_cases
from zhrag.io_utils import read_json, write_json, write_text


def _runner() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_crud_generation_subset.py"
    spec = importlib.util.spec_from_file_location("build_crud_generation_subset_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw(count: int = 6) -> dict[str, list[dict[str, str]]]:
    return {
        "event_summary": [
            {
                "ID": f"summary-{index}",
                "text": f"来源文本 {index} 甲 乙 丙",
                "summary": f"参考摘要 {index} 甲 乙",
                "unused_column": "MUST_NOT_ESCAPE",
            }
            for index in range(count)
        ],
        "questanswer_1doc": [
            {
                "ID": f"qa-{index}",
                "news1": f"来源新闻 {index} 春 夏 秋",
                "questions": f"原始问题 {index} 是什么？",
                "answers": f"参考答案 {index} 春 夏",
            }
            for index in range(count)
        ],
    }


def _argv(source: Path, target: Path, *, seed: int = 0, summary: int = 2, qa: int = 3) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(target),
        "--event-summary",
        str(summary),
        "--questanswer-1doc",
        str(qa),
        "--seed",
        str(seed),
    ]


class TestSampling:
    def test_draw_is_deterministic_and_sized_per_task(self, tmp_path: Path) -> None:
        runner = _runner()
        first = runner.sample_subset(_raw(), {"event_summary": 2, "questanswer_1doc": 3}, seed=0)
        second = runner.sample_subset(_raw(), {"event_summary": 2, "questanswer_1doc": 3}, seed=0)

        assert first == second
        assert len(first["event_summary"]) == 2
        assert len(first["questanswer_1doc"]) == 3

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        runner = _runner()
        sizes = {"event_summary": 2, "questanswer_1doc": 2}
        seeds = {
            seed: tuple(
                row["ID"]
                for row in runner.sample_subset(_raw(12), sizes, seed=seed)["event_summary"]
            )
            for seed in range(6)
        }
        assert len(set(seeds.values())) > 1

    def test_source_ordering_does_not_change_the_draw(self) -> None:
        """The upstream file's order must not leak into which cases get bought."""
        runner = _runner()
        sizes = {"event_summary": 2, "questanswer_1doc": 2}
        forward = _raw(8)
        reversed_source = {task: list(reversed(rows)) for task, rows in forward.items()}
        assert runner.sample_subset(forward, sizes, seed=3) == runner.sample_subset(
            reversed_source, sizes, seed=3
        )

    def test_only_contract_fields_are_copied(self) -> None:
        runner = _runner()
        subset = runner.sample_subset(_raw(), {"event_summary": 2, "questanswer_1doc": 2}, seed=0)
        assert all(set(row) == {"ID", "text", "summary"} for row in subset["event_summary"])
        assert all(
            set(row) == {"ID", "news1", "questions", "answers"}
            for row in subset["questanswer_1doc"]
        )

    def test_output_is_a_valid_generation_input(self) -> None:
        runner = _runner()
        subset = runner.sample_subset(_raw(), {"event_summary": 2, "questanswer_1doc": 3}, seed=0)
        assert len(build_generation_cases(subset)) == 5

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda raw: raw.pop("event_summary"), "non-empty list"),
            (lambda raw: raw.__setitem__("event_summary", []), "non-empty list"),
            (lambda raw: raw["event_summary"][0].__setitem__("summary", "  "), "non-blank"),
            (lambda raw: raw["event_summary"][0].pop("text"), "non-blank"),
            (
                lambda raw: raw["event_summary"].__setitem__(1, dict(raw["event_summary"][0])),
                "not unique",
            ),
        ],
    )
    def test_malformed_input_fails_closed(self, mutate: Any, message: str) -> None:
        runner = _runner()
        raw = _raw()
        mutate(raw)
        with pytest.raises(SystemExit, match=message):
            runner.sample_subset(raw, {"event_summary": 2, "questanswer_1doc": 2}, seed=0)

    def test_oversized_request_fails_closed(self) -> None:
        runner = _runner()
        with pytest.raises(SystemExit, match="asked for"):
            runner.sample_subset(_raw(3), {"event_summary": 99, "questanswer_1doc": 2}, seed=0)


class TestPublication:
    def test_writes_once_and_is_idempotent(self, tmp_path: Path) -> None:
        runner = _runner()
        source = tmp_path / "split.json"
        target = tmp_path / "subset.json"
        write_json(source, _raw())

        assert runner.main(_argv(source, target)) == 0
        first = read_json(target)
        assert runner.main(_argv(source, target)) == 0
        assert read_json(target) == first

    def test_refuses_to_overwrite_a_different_subset(self, tmp_path: Path) -> None:
        runner = _runner()
        source = tmp_path / "split.json"
        target = tmp_path / "subset.json"
        write_json(source, _raw(12))

        assert runner.main(_argv(source, target, seed=0)) == 0
        before = read_json(target)
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            runner.main(_argv(source, target, seed=7, summary=4, qa=4))
        assert read_json(target) == before
        assert not list(tmp_path.glob("*.subset.tmp"))

    def test_rejects_a_nonpositive_size_before_reading_input(self, tmp_path: Path) -> None:
        runner = _runner()
        missing = tmp_path / "absent.json"
        target = tmp_path / "subset.json"
        with pytest.raises(SystemExit, match="must be positive"):
            runner.main(_argv(missing, target, summary=0))
        assert not target.exists()

    def test_unreadable_input_fails_closed(self, tmp_path: Path) -> None:
        runner = _runner()
        source = tmp_path / "split.json"
        target = tmp_path / "subset.json"
        write_text(source, "{not json")
        with pytest.raises(SystemExit, match="unreadable dataset"):
            runner.main(_argv(source, target))
        assert not target.exists()

    def test_reports_provenance_without_printing_corpus_text(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        runner = _runner()
        source = tmp_path / "split.json"
        target = tmp_path / "subset.json"
        write_json(source, _raw())

        assert runner.main(_argv(source, target)) == 0
        output = capsys.readouterr().out
        assert "dataset_snapshot_sha256:" in output
        assert "cases: 5" in output
        for secret in ("来源文本", "参考摘要", "来源新闻", "参考答案", "MUST_NOT_ESCAPE"):
            assert secret not in output
