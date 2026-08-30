"""Header-aware Markdown chunking for mixed Chinese/English technical docs.

Splitting on Markdown headers alone is the usual advice and it does not survive
contact with this corpus. Measured over the 450 indexed TiDB documents, sizing
with :func:`zhrag.tokens.estimate_tokens` (regenerate via
``scripts/corpus_stats.py``):

    strategy                     n     p10   p50    p90     max   <100tok  broken
    headers only             5,504      14    65    318  16,111     63.5%       -
    two-stage target=400     1,832     165   371    734  10,914      5.0%       0
    two-stage target=512     1,486     215   476    890  16,111      3.2%       0
    two-stage target=700     1,154     224   645  1,165  16,111      2.7%       0

Nearly two thirds of header-only chunks fall under 100 tokens -- a bare heading
and one sentence -- because the corpus's h2/h3 sections have a median of just 332
characters. Those fragments embed to noise. At the other end a single section
runs to 16k tokens. So chunking runs in two stages:

1. Split on the header hierarchy, recording the full heading path.
2. Merge consecutive small sections up to ``target_tokens``, and split oversized
   sections on paragraph boundaries.

Fenced code blocks and pipe tables are masked before splitting and restored
afterwards, so they are never cut in half -- verified at 0 broken fences across
all three target sizes. Retrieval on this corpus depends on exact matches
against identifiers like ``tiup cluster deploy``, and a code block truncated
mid-token destroys that.

Residual: at target=400, 282 of 1,832 chunks (15.4%) still exceed
``hard_max_tokens`` because a single masked block is larger than the budget --
the biggest is a 16k-token generated table. They stay oversized by design;
Qwen3-Embedding-8B's 32k context accepts them, and splitting a table away from
its header row would cost more than the size does.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from zhrag.tokens import estimate_tokens

__all__ = [
    "Chunk",
    "Section",
    "chunk_markdown",
    "normalize",
    "parse_frontmatter",
    "split_by_headings",
    "split_trigger_exceedance_reason",
]

_FENCE = re.compile(r"```.*?```", re.S)
_TABLE = re.compile(r"(?:^\|.*\|[ \t]*$\n?){2,}", re.M)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_PLACEHOLDER = re.compile(r"\{\{\{\s*\.(\w+)\s*\}\}\}")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\((?:[^)]*)\)")
_PARA = re.compile(r"\n[ \t]*\n")
_SENTINEL = "\x00\x01{}\x00"
_SENTINEL_RE = re.compile(r"\x00\x01(\d+)\x00")


@dataclass(frozen=True, slots=True)
class Section:
    heading_path: tuple[str, ...]
    body: str


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    heading_path: tuple[str, ...]
    ordinal: int
    approx_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contextual_text(self) -> str:
        """Chunk text prefixed with its heading path.

        The prefix is what makes an isolated chunk interpretable: a fragment
        reading "默认值为 3" is meaningless until you know it sits under
        "TiKV 配置文件 > raftstore > apply-pool-size".
        """
        return f"{' > '.join(self.heading_path)}\n\n{self.text}" if self.heading_path else self.text


type SplitTriggerReason = Literal[
    "protected_fence",
    "protected_table",
    "protected_fence_and_table",
    "indivisible_paragraph",
    "unexplained",
]


def split_trigger_exceedance_reason(
    text: str,
    *,
    split_trigger_tokens: int,
    token_counter: Callable[[str], int] = estimate_tokens,
) -> SplitTriggerReason | None:
    """Classify why one emitted chunk exceeds the best-effort split trigger.

    ``hard_max_tokens`` in :func:`chunk_markdown` triggers paragraph splitting;
    it is not a strict output cap. Fenced code and pipe tables stay atomic, and
    an individual paragraph has no smaller safe boundary. This helper uses the
    same grammar as the chunker so reports can reconcile every exceedance without
    mislabelling expected preservation as a hard-limit violation.
    """
    if split_trigger_tokens < 1:
        raise ValueError("split_trigger_tokens must be positive")
    if token_counter(text) <= split_trigger_tokens:
        return None
    has_fence = _FENCE.search(text) is not None
    has_table = _TABLE.search(text) is not None
    if has_fence and has_table:
        return "protected_fence_and_table"
    if has_fence:
        return "protected_fence"
    if has_table:
        return "protected_table"
    if any(
        paragraph.strip() and token_counter(paragraph) > split_trigger_tokens
        for paragraph in _PARA.split(text)
    ):
        return "indivisible_paragraph"
    return "unexplained"


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from the body. All 500 TiDB docs carry one."""
    if not markdown.startswith("---"):
        return {}, markdown
    end = markdown.find("\n---", 3)
    if end < 0:
        return {}, markdown
    try:
        meta = yaml.safe_load(markdown[3:end]) or {}
    except yaml.YAMLError:
        return {}, markdown
    body = markdown[end + 4 :].lstrip("\n")
    return (meta if isinstance(meta, dict) else {}), body


def normalize(text: str, *, strip_links: bool = True) -> str:
    """Clean corpus-specific markup that would otherwise be embedded as noise.

    ``strip_links`` keeps the anchor text and drops the target: link *text*
    carries meaning ("向量搜索索引"), while a relative path like
    ``/ai/reference/vector-search-index.md`` contributes only tokens. The corpus
    has 5,015 such links.
    """
    text = _PLACEHOLDER.sub(lambda m: m.group(1), text)  # {{{ .starter }}} -> starter
    if strip_links:
        text = _IMAGE.sub(lambda m: m.group(1), text)
        text = _MD_LINK.sub(lambda m: m.group(1), text)
    return text


def _mask(text: str) -> tuple[str, list[str]]:
    """Replace code fences and tables with sentinels so splitting cannot cut them."""
    blocks: list[str] = []

    def take(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return _SENTINEL.format(len(blocks) - 1)

    return _TABLE.sub(take, _FENCE.sub(take, text)), blocks


def _unmask(text: str, blocks: list[str]) -> str:
    return _SENTINEL_RE.sub(lambda m: blocks[int(m.group(1))], text)


def _common_heading_path(sections: list[Section]) -> tuple[str, ...]:
    """Return the heading prefix shared by every section in one chunk."""
    if not sections:
        return ()
    common = list(sections[0].heading_path)
    for section in sections[1:]:
        shared = 0
        for left, right in zip(common, section.heading_path, strict=False):
            if left != right:
                break
            shared += 1
        del common[shared:]
        if not common:
            break
    return tuple(common)


def _materialize_sections(sections: list[Section]) -> tuple[tuple[str, ...], str]:
    """Preserve every branch heading when several small sections are merged.

    A chunk still carries the shared heading prefix as metadata. Each section's
    remaining suffix is embedded in the body, so merging siblings cannot discard
    every heading after the first one. A single section keeps the historical
    representation exactly: its full path stays in metadata and its body is not
    rewritten.
    """
    path = _common_heading_path(sections)
    parts: list[str] = []
    for section in sections:
        suffix = section.heading_path[len(path) :]
        if suffix:
            parts.append(f"{' > '.join(suffix)}\n\n{section.body}")
        else:
            parts.append(section.body)
    return path, "\n\n".join(parts)


def split_by_headings(body: str) -> list[Section]:
    """Split on ATX headings, carrying the full heading path down the tree."""
    sections: list[Section] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf and "".join(buf).strip():
            sections.append(Section(tuple(stack), "\n".join(buf).strip()))

    for line in body.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = _ANCHOR.sub("", m.group(2)).strip()
            stack = [*stack[: level - 1], title]
        else:
            buf.append(line)
    flush()
    return sections


def _split_paragraphs(body: str, target: int, count: Callable[[str], int]) -> list[list[str]]:
    """Greedily group paragraphs into batches of at most ``target`` tokens.

    A paragraph that exceeds ``target`` on its own still gets its own group --
    masked code blocks and tables are atomic by design, so there is nothing
    smaller to split them into.
    """
    groups: list[list[str]] = []
    acc: list[str] = []
    for para in _PARA.split(body):
        if acc and count("\n\n".join([*acc, para])) > target:
            groups.append(acc)
            acc = [para]
        else:
            acc.append(para)
    if acc:
        groups.append(acc)
    return groups


def chunk_markdown(
    markdown: str,
    *,
    target_tokens: int = 400,
    hard_max_tokens: int = 600,
    metadata: dict[str, Any] | None = None,
    token_counter: Callable[[str], int] = estimate_tokens,
    strip_links: bool = True,
) -> list[Chunk]:
    """Chunk one Markdown document.

    ``target_tokens=400`` is the chosen operating point for this corpus: it
    yields p50=371 / p90=734 tokens with 5.0% undersized chunks. 512 and 700
    trim waste only marginally further (3.2%, 2.7%) while pushing p90 to 890 and
    1,165, which costs generator context on every query for little recall gain.
    """
    if target_tokens < 1 or hard_max_tokens < target_tokens:
        raise ValueError(
            f"require 1 <= target_tokens <= hard_max_tokens, got {target_tokens}, {hard_max_tokens}"
        )

    front, body = parse_frontmatter(markdown)
    masked, blocks = _mask(normalize(body, strip_links=strip_links))

    base: dict[str, Any] = dict(metadata or {})
    for key in ("title", "summary"):
        if isinstance(front.get(key), str):
            base.setdefault(key, front[key])

    out: list[Chunk] = []
    pending: list[Section] = []

    def emit(sections: list[Section]) -> None:
        path, materialized = _materialize_sections(sections)
        text = _unmask(materialized.strip(), blocks)
        if text:
            out.append(Chunk(text, path, len(out), token_counter(text), dict(base)))

    for section in split_by_headings(masked):
        size = token_counter(section.body)

        if size > hard_max_tokens:
            # Oversized: flush what is pending, then split on paragraph breaks.
            if pending:
                emit(pending)
                pending = []
            for group in _split_paragraphs(section.body, target_tokens, token_counter):
                emit([Section(section.heading_path, "\n\n".join(group))])
            continue

        proposed = [*pending, section]
        _, materialized = _materialize_sections(proposed)
        if pending and token_counter(materialized) > target_tokens:
            emit(pending)
            pending = []

        pending.append(section)

    if pending:
        emit(pending)
    return out
