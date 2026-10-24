"""Read the repository's original synthetic demo without legacy preview helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from zhrag.io_utils import read_text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def node_data() -> dict:
    result = subprocess.run(
        ["node", str(HERE / "extract_demo.mjs")],
        input=json.dumps(
            {"demoTs": read_text(REPO / "frontend/src/lib/demo.ts")}, ensure_ascii=False
        ),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        cwd=REPO,
    )
    return json.loads(result.stdout)
