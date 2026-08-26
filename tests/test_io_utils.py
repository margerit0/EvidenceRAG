"""Tests for the UTF-8 I/O port and the token estimator.

The I/O tests exist because this project's development machine defaults to
cp936: a regression here fails loudly on Windows and silently passes on Linux
CI, so the round-trip assertions are deliberately explicit about encoding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zhrag.io_utils import (
    append_jsonl,
    exclusive_lock,
    read_json,
    read_jsonl,
    read_text,
    read_yaml,
    replace_files,
    write_json,
    write_jsonl,
    write_text,
)
from zhrag.tokens import cjk_ratio, estimate_tokens

CHINESE = "向量搜索：使用 `tiup cluster deploy` 部署 TiDB 集群。—— 全角标点／测试"


class TestExclusiveLock:
    def test_refuses_a_second_writer_and_cleans_up(self, tmp_path: Path) -> None:
        lock = tmp_path / "build.lock"
        with exclusive_lock(lock):
            assert lock.exists()
            with pytest.raises(SystemExit, match="another writer"), exclusive_lock(lock):
                raise AssertionError("unreachable")
        assert not lock.exists()

    def test_cleans_up_after_an_exception(self, tmp_path: Path) -> None:
        lock = tmp_path / "build.lock"
        with pytest.raises(RuntimeError, match="boom"), exclusive_lock(lock):
            raise RuntimeError("boom")
        assert not lock.exists()


class TestReplaceFiles:
    def test_replaces_a_bundle_in_order(self, tmp_path: Path) -> None:
        first = tmp_path / "first.tmp"
        marker = tmp_path / "marker.tmp"
        write_text(first, "new data")
        write_text(marker, "new marker")
        data_target = tmp_path / "data.jsonl"
        marker_target = tmp_path / "report.json"
        write_text(data_target, "old data")
        write_text(marker_target, "old marker")

        replace_files(((first, data_target), (marker, marker_target)))

        assert read_text(data_target) == "new data"
        assert read_text(marker_target) == "new marker"
        assert not first.exists()
        assert not marker.exists()

    def test_validates_every_staged_file_before_replacing_any(self, tmp_path: Path) -> None:
        present = tmp_path / "present.tmp"
        missing = tmp_path / "missing.tmp"
        first_target = tmp_path / "first.json"
        second_target = tmp_path / "second.json"
        write_text(present, "new")
        write_text(first_target, "old")

        with pytest.raises(FileNotFoundError, match=r"missing\.tmp"):
            replace_files(((present, first_target), (missing, second_target)))

        assert read_text(first_target) == "old"
        assert present.exists()


class TestAppendJsonl:
    """append_jsonl backs the incremental embedding cache; see io_utils."""

    def test_creates_the_file_when_absent(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "cache.jsonl"
        assert append_jsonl(p, [{"id": "a"}]) == 1
        assert [r["id"] for r in read_jsonl(p)] == ["a"]

    def test_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.jsonl"
        append_jsonl(p, [{"id": "a"}, {"id": "b"}])
        append_jsonl(p, [{"id": "c"}])
        assert [r["id"] for r in read_jsonl(p)] == ["a", "b", "c"]

    def test_duplicate_keys_resolve_last_write_wins(self, tmp_path: Path) -> None:
        # The cache flushes per batch and a resumed run may re-emit a key. The
        # documented contract is that the later row wins once a caller folds the
        # stream into a dict -- verify that rather than assume it.
        p = tmp_path / "cache.jsonl"
        append_jsonl(p, [{"id": "a", "v": 1}])
        append_jsonl(p, [{"id": "a", "v": 2}])
        assert {r["id"]: r["v"] for r in read_jsonl(p)} == {"a": 2}

    def test_line_endings_stay_lf(self, tmp_path: Path) -> None:
        # Same guarantee write_jsonl makes; an appending writer opens the file
        # separately and would re-introduce CRLF on Windows if newline= is lost.
        p = tmp_path / "cache.jsonl"
        append_jsonl(p, [{"id": "a"}])
        append_jsonl(p, [{"id": "b"}])
        assert b"\r\n" not in p.read_bytes()

    def test_chinese_is_not_escaped(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.jsonl"
        append_jsonl(p, [{"text": CHINESE}])
        assert CHINESE in p.read_text(encoding="utf-8")


class TestTextRoundTrip:
    def test_chinese_survives_a_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        write_text(p, CHINESE)
        assert read_text(p) == CHINESE

    def test_bytes_on_disk_are_utf8_not_locale_encoded(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        write_text(p, CHINESE)
        assert p.read_bytes().decode("utf-8") == CHINESE

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c.txt"
        write_text(p, "x")
        assert p.exists()

    def test_line_endings_stay_lf_on_windows(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.txt"
        write_text(p, "一\n二\n")
        assert b"\r\n" not in p.read_bytes()

    def test_undecodable_bytes_are_replaced_rather_than_raising(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.txt"
        p.write_bytes(b"\xff\xfe ok")
        assert "ok" in read_text(p)


class TestJson:
    def test_round_trips_chinese_unescaped(self, tmp_path: Path) -> None:
        p = tmp_path / "x.json"
        write_json(p, {"标题": "向量搜索", "n": 3})
        assert read_json(p) == {"标题": "向量搜索", "n": 3}
        # ensure_ascii=False keeps the committed artifact human-reviewable.
        assert "向量搜索" in p.read_bytes().decode("utf-8")

    def test_yaml_reads_as_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "x.yaml"
        write_text(p, "title: 向量搜索\ncount: 3\n")
        assert read_yaml(p) == {"title": "向量搜索", "count": 3}


class TestJsonl:
    def test_round_trips_and_reports_the_row_count(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        rows = [{"id": i, "text": f"文档 {i}"} for i in range(3)]
        assert write_jsonl(p, rows) == 3
        assert list(read_jsonl(p)) == rows

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        write_text(p, '{"a": 1}\n\n{"a": 2}\n')
        assert list(read_jsonl(p)) == [{"a": 1}, {"a": 2}]

    def test_malformed_line_reports_its_number(self, tmp_path: Path) -> None:
        p = tmp_path / "x.jsonl"
        write_text(p, '{"a": 1}\nnot json\n')
        with pytest.raises(ValueError, match=r":2: malformed JSONL"):
            list(read_jsonl(p))

    def test_is_lazy(self, tmp_path: Path) -> None:
        """Streaming matters: the raw CRUD-RAG split file is 26 MB."""
        p = tmp_path / "x.jsonl"
        write_text(p, '{"a": 1}\nnot json\n')
        next(iter(read_jsonl(p)))  # first row must not raise


class TestTokens:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("", 0.0), ("abc", 0.0), ("向量", 1.0)],
    )
    def test_cjk_ratio(self, text: str, expected: float) -> None:
        assert cjk_ratio(text) == pytest.approx(expected)

    def test_full_width_punctuation_counts_as_cjk(self) -> None:
        assert cjk_ratio("，。！") == pytest.approx(1.0)

    def test_empty_text_is_zero_tokens(self) -> None:
        assert estimate_tokens("") == 0

    def test_chinese_is_denser_than_latin_per_character(self) -> None:
        """A flat chars-per-token constant would get this backwards."""
        assert estimate_tokens("向量" * 50) > estimate_tokens("ab" * 50)

    def test_mixed_text_falls_between_the_pure_cases(self) -> None:
        n = 60
        mixed = estimate_tokens(("向量" + "ab") * (n // 2))
        assert estimate_tokens("ab" * n) < mixed < estimate_tokens("向量" * n)

    def test_scales_roughly_linearly(self) -> None:
        assert estimate_tokens("向量搜索" * 100) == pytest.approx(
            estimate_tokens("向量搜索" * 10) * 10, rel=0.05
        )
