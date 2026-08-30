"""Tests for header-aware Markdown chunking.

The invariants that matter for retrieval quality on this corpus: code fences and
tables survive intact, heading paths are carried down, small sections merge, and
oversized sections split.
"""

from __future__ import annotations

import pytest

from zhrag.chunking import (
    Chunk,
    chunk_markdown,
    normalize,
    parse_frontmatter,
    split_by_headings,
    split_trigger_exceedance_reason,
)

DOC = """---
title: Vector Search Overview
summary: 了解 TiDB 中的向量搜索功能。
aliases: ['/zh/tidb/stable/vector-search-overview/']
---

# 向量搜索概述

向量搜索提供语义相似性搜索解决方案。

## 概念 {#concepts}

### 向量嵌入 {#vector-embedding}

向量嵌入是一组数字序列。

## 用法

```sql
SELECT * FROM t ORDER BY VEC_COSINE_DISTANCE(v, '[1,2,3]') LIMIT 10;
```
"""


class TestFrontmatter:
    def test_extracts_mapping_and_strips_it_from_the_body(self) -> None:
        meta, body = parse_frontmatter(DOC)
        assert meta["title"] == "Vector Search Overview"
        assert meta["summary"].startswith("了解 TiDB")
        assert body.startswith("# 向量搜索概述")

    def test_document_without_frontmatter_is_returned_unchanged(self) -> None:
        meta, body = parse_frontmatter("# 标题\n正文")
        assert meta == {}
        assert body == "# 标题\n正文"

    def test_malformed_yaml_degrades_instead_of_raising(self) -> None:
        meta, body = parse_frontmatter("---\n: : :\n---\n正文")
        assert isinstance(meta, dict)
        assert isinstance(body, str)

    def test_unterminated_frontmatter_is_left_alone(self) -> None:
        meta, _ = parse_frontmatter("---\ntitle: x\n还没结束")
        assert meta == {}


class TestNormalize:
    def test_unwraps_template_placeholders(self) -> None:
        assert normalize("在 {{{ .starter }}} 上创建") == "在 starter 上创建"

    def test_keeps_link_text_and_drops_the_target(self) -> None:
        assert normalize("参考 [向量索引](/ai/reference/index.md) 文档") == "参考 向量索引 文档"

    def test_keeps_image_alt_text(self) -> None:
        assert normalize("![架构图](/media/x.png)") == "架构图"

    def test_link_stripping_can_be_disabled(self) -> None:
        assert "/ai/x.md" in normalize("[a](/ai/x.md)", strip_links=False)


class TestSplitTriggerDiagnostics:
    @staticmethod
    def _chars(text: str) -> int:
        return len(text)

    def test_returns_none_within_trigger(self) -> None:
        assert (
            split_trigger_exceedance_reason(
                "short",
                split_trigger_tokens=5,
                token_counter=self._chars,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        (
            ("```sql\nselect 1\n```", "protected_fence"),
            ("| a |\n|---|\n| 1 |", "protected_table"),
            ("```text\nx\n```\n\n| a |\n|---|", "protected_fence_and_table"),
            ("one-indivisible-paragraph", "indivisible_paragraph"),
            ("aaa\n\nbbb", "unexplained"),
        ),
    )
    def test_uses_chunker_grammar_for_exceedances(self, text: str, expected: str) -> None:
        assert (
            split_trigger_exceedance_reason(
                text,
                split_trigger_tokens=4,
                token_counter=self._chars,
            )
            == expected
        )

    def test_rejects_non_positive_trigger(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            split_trigger_exceedance_reason("text", split_trigger_tokens=0)


class TestHeadingSplit:
    def test_carries_the_full_heading_path(self) -> None:
        _, body = parse_frontmatter(DOC)
        paths = [s.heading_path for s in split_by_headings(body)]
        assert ("向量搜索概述", "概念", "向量嵌入") in paths

    def test_strips_anchor_suffixes_from_titles(self) -> None:
        sections = split_by_headings("## 概念 {#concepts}\n正文")
        assert sections[0].heading_path == ("概念",)

    def test_sections_without_body_text_are_dropped(self) -> None:
        sections = split_by_headings("# A\n\n## B\n\n内容")
        assert [s.heading_path for s in sections] == [("A", "B")]

    def test_deeper_heading_resets_only_the_tail(self) -> None:
        sections = split_by_headings("# A\nx\n## B\ny\n## C\nz")
        assert [s.heading_path for s in sections] == [("A",), ("A", "B"), ("A", "C")]


class TestChunking:
    def test_promotes_frontmatter_into_chunk_metadata(self) -> None:
        chunks = chunk_markdown(DOC)
        assert all(c.metadata["title"] == "Vector Search Overview" for c in chunks)

    def test_caller_metadata_is_attached(self) -> None:
        chunks = chunk_markdown(DOC, metadata={"collection": "dev_reference"})
        assert chunks[0].metadata["collection"] == "dev_reference"

    def test_ordinals_are_dense_and_ordered(self) -> None:
        chunks = chunk_markdown(DOC, target_tokens=20, hard_max_tokens=40)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_code_fences_are_never_split(self) -> None:
        # A tiny target would cut the SQL block in half if fences were not masked.
        chunks = chunk_markdown(DOC, target_tokens=10, hard_max_tokens=15)
        for c in chunks:
            assert c.text.count("```") % 2 == 0

    def test_tables_are_never_split(self) -> None:
        table = "# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
        chunks = chunk_markdown(table, target_tokens=5, hard_max_tokens=8)
        joined = [c for c in chunks if "| 1 | 2 |" in c.text]
        assert joined and "| 5 | 6 |" in joined[0].text

    def test_small_sibling_sections_are_merged(self) -> None:
        doc = "".join(f"## 小节 {i}\n简短内容。\n\n" for i in range(10))
        chunks = chunk_markdown(doc, target_tokens=200, hard_max_tokens=300)
        assert len(chunks) < 10

    def test_merged_siblings_keep_every_heading_in_contextual_text(self) -> None:
        doc = "# 顶层\n\n## 甲\n甲的正文。\n\n## 乙\n乙的正文。\n"
        [chunk] = chunk_markdown(doc, target_tokens=200, hard_max_tokens=300)

        assert chunk.heading_path == ("顶层",)
        assert chunk.contextual_text == ("顶层\n\n甲\n\n甲的正文。\n\n乙\n\n乙的正文。")

    def test_single_section_keeps_the_historical_materialization(self) -> None:
        [chunk] = chunk_markdown("# 顶层\n\n## 子节\n正文。")
        assert chunk.heading_path == ("顶层", "子节")
        assert chunk.text == "正文。"
        assert chunk.contextual_text == "顶层 > 子节\n\n正文。"

    def test_oversized_section_is_split_on_paragraph_boundaries(self) -> None:
        doc = "# 大\n\n" + "\n\n".join("这是一个很长的段落。" * 12 for _ in range(12))
        chunks = chunk_markdown(doc, target_tokens=100, hard_max_tokens=150)
        assert len(chunks) > 1

    def test_heading_path_is_preserved_through_splitting(self) -> None:
        doc = "# 顶层\n\n## 子节\n\n" + "\n\n".join("长段落内容。" * 20 for _ in range(8))
        chunks = chunk_markdown(doc, target_tokens=80, hard_max_tokens=120)
        assert all(c.heading_path[:1] == ("顶层",) for c in chunks)

    def test_contextual_text_prefixes_the_heading_path(self) -> None:
        c = Chunk("默认值为 3。", ("TiKV 配置", "raftstore"), 0, 6)
        assert c.contextual_text.startswith("TiKV 配置 > raftstore")

    def test_contextual_text_without_a_heading_is_the_bare_text(self) -> None:
        assert Chunk("正文", (), 0, 2).contextual_text == "正文"

    def test_empty_document_yields_no_chunks(self) -> None:
        assert chunk_markdown("") == []

    def test_rejects_incoherent_size_bounds(self) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            chunk_markdown(DOC, target_tokens=500, hard_max_tokens=100)

    def test_token_estimates_are_populated(self) -> None:
        assert all(c.approx_tokens > 0 for c in chunk_markdown(DOC))
