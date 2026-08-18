"""The single file-I/O port for the project.

Every read and write in this codebase goes through this module. The reason is
specific and was verified on the development machine:

    >>> import sys, locale
    >>> sys.stdout.encoding, locale.getpreferredencoding(), sys.flags.utf8_mode
    ('gbk', 'cp936', 0)
    >>> open('tidb-rag-curated/README.md').read()
    UnicodeDecodeError: 'gbk' codec can't decode byte 0xad in position 9

PEP 686 (UTF-8 as the default encoding) did not land in 3.13, so on a
Simplified-Chinese Windows install ``open()`` still resolves to cp936. The
failure is asymmetric and therefore easy to misdiagnose:
``sys.getfilesystemencoding()`` is already ``utf-8``, so *paths* work and only
file *contents* blow up.

It is also asymmetric with CI: Linux runners default to UTF-8, so a Linux-only
matrix goes green while the author's own machine fails. The CI matrix keeps a
``windows-latest`` leg specifically to catch this class of bug.

``ruff``'s ``PLW1514`` rule bans bare ``open()`` everywhere except here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "read_json",
    "read_jsonl",
    "read_text",
    "read_yaml",
    "write_json",
    "write_jsonl",
    "write_text",
]

ENCODING = "utf-8"


def read_text(path: str | Path) -> str:
    """Read a text file as UTF-8, tolerating the occasional malformed byte."""
    return Path(path).read_text(encoding=ENCODING, errors="replace")


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text with LF endings, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding=ENCODING, newline="\n") as fh:
        fh.write(content)


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    # ensure_ascii=False keeps Chinese readable in the artifact, which matters
    # because eval results get committed and reviewed by humans.
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file. Blank lines are skipped; malformed lines raise."""
    with open(Path(path), encoding=ENCODING, errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSONL: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding=ENCODING, newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(read_text(path))
