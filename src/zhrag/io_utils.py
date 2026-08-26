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
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "append_jsonl",
    "exclusive_lock",
    "read_bytes",
    "read_json",
    "read_jsonl",
    "read_text",
    "read_yaml",
    "replace_files",
    "write_json",
    "write_jsonl",
    "write_text",
]

ENCODING = "utf-8"


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Hold a fail-fast, cross-process lock represented by an exclusive file.

    The lock file stores only the writer PID and is removed on normal exit or a
    Python exception. An unclean process death leaves a stale lock deliberately:
    automatically stealing it would make two writers possible when process liveness
    cannot be proved portably. The operator may inspect and remove a stale file.
    """
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"! another writer may be active; lock exists: {lock}") from exc
    try:
        with os.fdopen(fd, "w", encoding=ENCODING, newline="\n") as fh:
            fh.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def replace_files(staged: Iterable[tuple[str | Path, str | Path]]) -> None:
    """Replace a validated bundle of files, publishing its marker last.

    Callers write every staged file first, then pass data files followed by the
    bundle's report/manifest. ``os.replace`` is atomic for each file on the same
    filesystem; publishing the marker last means readers never mistake a partial
    replacement for a complete new bundle after a crash between replacements.
    """
    pairs = [(Path(source), Path(target)) for source, target in staged]
    if not pairs:
        raise ValueError("replacement bundle must be non-empty")
    sources = [source for source, _target in pairs]
    targets = [target for _source, target in pairs]
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise ValueError("replacement sources and targets must be unique")
    missing = [source for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"staged replacement is absent: {missing[0]}")
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)


def read_bytes(path: str | Path) -> bytes:
    """Read a file as raw bytes.

    :func:`read_text` decodes with ``errors="replace"``, which is right for
    display but destroys byte-level identity: a replaced byte hashes differently
    from the original. Content hashes and upstream checksum verification must
    therefore start here, not from decoded text.
    """
    return Path(path).read_bytes()


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


def write_json(path: str | Path, obj: Any, *, indent: int | None = 2) -> None:
    # ensure_ascii=False keeps Chinese readable in the artifact, which matters
    # because eval results get committed and reviewed by humans. Machine-only
    # artifacts (a 75k-term vocabulary) pass indent=None: one line per file is
    # several times smaller and nobody reads it by eye anyway.
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


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows to a JSONL file, creating it if absent.

    This exists so an incrementally-built cache costs O(new rows) per flush
    rather than O(all rows). Rewriting the whole file after every batch is
    trivially crash-safe and fine for small files, but an embedding cache holds
    thousands of 4,096-float rows: re-serialising all of them on each of ~115
    batches spends more time in :func:`json.dumps` than in the network call it
    is checkpointing.

    Readers must therefore tolerate duplicate keys. The convention in this
    codebase is last-write-wins, which :func:`read_jsonl` gives naturally when
    the caller builds a dict.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "a", encoding=ENCODING, newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(read_text(path))
