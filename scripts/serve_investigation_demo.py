"""Run the original synthetic persistent investigation demo, entirely offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from zhrag.service.investigation_demo import create_demo_app


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--db", type=Path, default=Path("indexes/investigations/demo.sqlite3"))
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 <= args.delay <= 10:
        parser.error("delay must be in [0, 10]")
    uvicorn.run(create_demo_app(args.db, delay=args.delay), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
