"""Pure parsing and validation for the repository's JUnit quality gate."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from zhrag.io_utils import read_text

__all__ = ["JunitCounts", "parse_junit", "read_junit", "require_clean"]


@dataclass(frozen=True, slots=True)
class JunitCounts:
    """Aggregated JUnit counters from one report."""

    tests: int
    failures: int
    errors: int
    skipped: int

    def __post_init__(self) -> None:
        for name in ("tests", "failures", "errors", "skipped"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"JUnit {name} must be a non-negative integer")

    @property
    def is_clean(self) -> bool:
        return self.tests > 0 and not any((self.failures, self.errors, self.skipped))


def _counter(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"JUnit {name} is not an integer") from exc
    if parsed < 0:
        raise ValueError(f"JUnit {name} must be non-negative")
    return parsed


def _suite_counts(suite: ET.Element) -> tuple[int, int, int, int]:
    return tuple(
        _counter(suite.get(name, "0"), name=name)
        for name in ("tests", "failures", "errors", "skipped")
    )  # type: ignore[return-value]


def parse_junit(xml: str) -> JunitCounts:
    """Parse a JUnit document and sum its top-level suites.

    Pytest emits a ``testsuites`` root with one or more ``testsuite`` children;
    some producers emit a single ``testsuite`` root.  Unknown roots and missing
    suites fail closed rather than producing a zero-test green status.
    """

    if type(xml) is not str or not xml.strip():
        raise ValueError("JUnit XML must be non-blank text")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"malformed JUnit XML: {exc}") from exc

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [child for child in root if child.tag == "testsuite"]
    else:
        raise ValueError(f"malformed JUnit XML root: {root.tag!r}")
    if not suites:
        raise ValueError("JUnit XML contains no testsuite")

    totals = [0, 0, 0, 0]
    for suite in suites:
        for index, value in enumerate(_suite_counts(suite)):
            totals[index] += value
    return JunitCounts(*totals)


def read_junit(path: str | Path) -> JunitCounts:
    """Read and parse a JUnit file through the project's UTF-8 I/O port."""

    try:
        return parse_junit(read_text(path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read JUnit report {path}: {exc}") from exc


def require_clean(counts: JunitCounts) -> JunitCounts:
    """Require at least one test and no failure, error or skipped case."""

    if not isinstance(counts, JunitCounts):
        raise TypeError("counts must be JunitCounts")
    if not counts.is_clean:
        raise ValueError(
            "quality gate is not clean: "
            f"tests={counts.tests}, failures={counts.failures}, "
            f"errors={counts.errors}, skipped={counts.skipped}"
        )
    return counts
