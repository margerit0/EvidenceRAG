"""Synchronize the README quality-gate marker from one JUnit report.

This synchronizer intentionally knows nothing about TiDB, CRUD-RAG, corpora,
queries or evaluation artifacts.  It reads only a JUnit report and owns only
``QUALITY-GATE-STATUS`` in README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhrag.io_utils import exclusive_lock, read_text, replace_files, write_text
from zhrag.quality_gate import JunitCounts, read_junit, require_clean

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DEFAULT_JUNIT = ROOT / ".pytest_tmp" / "pytest.xml"
DOCS_LOCK = ".docs.lock"
MARKER = "QUALITY-GATE-STATUS"


def _reconfigure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize README quality-gate status from a JUnit report."
    )
    parser.add_argument("--junit", type=Path, default=DEFAULT_JUNIT)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--check", action="store_true", help="fail if README is stale")
    return parser.parse_args(argv)


def load_counts(path: Path) -> JunitCounts:
    """Read a JUnit report and require zero failures, errors and skips."""

    try:
        return require_clean(read_junit(path))
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"! quality gate is not clean: {exc}") from exc


def render(counts: JunitCounts) -> str:
    """Render only the marker body; the surrounding README remains untouched."""

    if not isinstance(counts, JunitCounts):
        raise TypeError("counts must be JunitCounts")
    require_clean(counts)
    return (
        f"tests/                   {counts.tests:,} 个单元测试\n"
        "```\n\n"
        f"质量门禁（本行仅由 `pytest.xml` 生成）：`pytest` {counts.tests:,} passed。"
        "`ruff check` / `ruff format --check` / `mypy --strict` 是独立的提交前门禁，"
        "不由本报告认证。"
    )


def _replace_region(text: str, body: str, *, path: Path) -> str:
    start = f"<!-- BEGIN {MARKER} -->"
    end = f"<!-- END {MARKER} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"! {path}: marker pair {MARKER!r} must occur exactly once")
    prefix, remainder = text.split(start, 1)
    _old, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.rstrip()}\n{end}{suffix}"


def synchronize(args: argparse.Namespace) -> tuple[Path, ...]:
    """Validate JUnit, render README, and publish it atomically."""

    with exclusive_lock(args.readme.parent / DOCS_LOCK):
        counts = load_counts(args.junit)
        before = read_text(args.readme)
        after = _replace_region(before, render(counts), path=args.readme)
        if before == after:
            return ()
        if args.check:
            raise SystemExit(f"! generated documentation is stale: {args.readme}")

        temporary = args.readme.with_suffix(args.readme.suffix + ".sync.tmp")
        try:
            write_text(temporary, after)
            replace_files(((temporary, args.readme),))
        finally:
            temporary.unlink(missing_ok=True)
    return (args.readme,)


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    args = _parse_args(argv)
    changed = synchronize(args)
    if args.check:
        print("quality-gate documentation is synchronized")
    elif changed:
        print("synchronized quality-gate documentation")
    else:
        print("quality-gate documentation was already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
