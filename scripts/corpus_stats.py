"""Regenerate the corpus statistics quoted in README.md.

The README's credibility depends on its numbers matching what the code actually
produces, so they are generated here rather than transcribed by hand. Run:

    uv run python scripts/corpus_stats.py

Every figure uses ``zhrag.tokens.estimate_tokens``, which is calibrated against
the real Qwen3 tokenizer. An earlier prototype used a flat 1.15 chars/token and
reported materially different chunk counts; sizing Chinese text with the wrong
tokenizer ratio is an easy way to publish numbers you cannot reproduce.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from pathlib import Path

from zhrag.chunking import chunk_markdown, normalize, parse_frontmatter, split_by_headings
from zhrag.io_utils import read_jsonl, read_text
from zhrag.tokens import cjk_ratio, estimate_tokens

ROOT = Path(__file__).resolve().parent.parent
TIDB = ROOT / "tidb-rag-curated"
CRUD = ROOT / "crud-rag-subset"
EXCLUDED_COLLECTION = "temporal_releases"


def _pct(sizes: list[int], f: float) -> int:
    return sorted(sizes)[min(int(len(sizes) * f), len(sizes) - 1)]


def _row(label: str, sizes: list[int], broken: int | None = None) -> str:
    tail = "" if broken is None else f" | broken fences {broken}"
    return (
        f"  {label:24s} n={len(sizes):>6,} | p10={_pct(sizes, 0.10):>4} "
        f"p50={_pct(sizes, 0.50):>4} p90={_pct(sizes, 0.90):>5} max={max(sizes):>6} | "
        f"<100tok {sum(1 for s in sizes if s < 100) / len(sizes):5.1%}{tail}"
    )


def tidb_stats() -> None:
    docs = [
        p
        for p in (TIDB / "documents").rglob("*.md")
        if p.relative_to(TIDB / "documents").parts[0] != EXCLUDED_COLLECTION
    ]
    if not docs:
        print(f"! no TiDB documents under {TIDB / 'documents'} -- run download_curated.ps1 first")
        return
    texts = [read_text(p) for p in docs]

    print(f"\n## TiDB corpus ({len(docs)} indexed docs, excluding {EXCLUDED_COLLECTION})\n")

    naive: list[int] = []
    for t in texts:
        _, body = parse_frontmatter(t)
        naive += [estimate_tokens(s.body) for s in split_by_headings(normalize(body))]
    print(_row("headers only", naive))

    for target, hard in ((400, 600), (512, 768), (700, 1000)):
        sizes: list[int] = []
        broken = 0
        for t in texts:
            for c in chunk_markdown(t, target_tokens=target, hard_max_tokens=hard):
                sizes.append(c.approx_tokens)
                broken += c.text.count("```") % 2
        print(_row(f"two-stage target={target}", sizes, broken))

    chosen = [c for t in texts for c in chunk_markdown(t)]
    tokens = sum(c.approx_tokens for c in chosen)
    atomic = sum(1 for c in chosen if c.approx_tokens > 600)
    print(f"\n  chosen config (target=400): {len(chosen):,} chunks, {tokens:,} tokens")
    print(f"  over hard_max, i.e. atomic code blocks/tables: {atomic} ({atomic / len(chosen):.1%})")
    for dims, label in ((4096, "native"), (1024, "MRL-1024")):
        print(f"  vectors @{dims:>4}d float32 ({label:8s}): {len(chosen) * dims * 4 / 1e6:6.1f} MB")

    manifest = TIDB / "corpus_manifest.jsonl"
    if manifest.exists():
        rows = list(read_jsonl(manifest))
        print(f"\n  manifest rows={len(rows)}  unique path={len({r['path'] for r in rows})}")
        same = sum(1 for r in rows if r["id"] == r["git_blob_sha1"])
        print(f"  id == git_blob_sha1: {same}/{len(rows)}")
        print(f"  category facet: {dict(Counter(r['category'] for r in rows).most_common(8))}")


def crud_stats() -> None:
    corpus = CRUD / "corpus" / "corpus.jsonl"
    if not corpus.exists():
        print(f"\n! no CRUD-RAG corpus at {corpus} -- run build_subset.ps1 first")
        return
    rows = list(read_jsonl(corpus))
    chars = [len(r.get("text", "")) for r in rows]
    tokens = [estimate_tokens(r.get("text", "")) for r in rows]
    print(f"\n## CRUD-RAG evaluation corpus ({len(rows)} docs)\n")
    print(
        f"  chars  min={min(chars)} p50={int(st.median(chars))} "
        f"max={max(chars)} total={sum(chars):,}"
    )
    print(f"  tokens p50={int(st.median(tokens))} total={sum(tokens):,}")
    print(f"  mean CJK ratio: {st.mean(cjk_ratio(r.get('text', '')) for r in rows):.1%}")
    fits = sum(1 for t in tokens if t <= 400) / len(tokens)
    print(f"  docs fitting in one 400-token chunk: {fits:.1%}")


if __name__ == "__main__":
    print("# Corpus statistics (regenerate with: uv run python scripts/corpus_stats.py)")
    tidb_stats()
    crud_stats()
