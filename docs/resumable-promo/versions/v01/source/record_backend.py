"""Record the original synthetic backend for the resumable-investigation promo.

Runs the repository's offline synthetic app (`create_demo_app`, the same app as
`scripts/serve_investigation_demo.py`) on a loopback port and drives the
documented flow over real HTTP: create -> clarification -> supplement -> user
pause -> resume -> graceful service shutdown while running -> restart on the
same database -> resume -> answer. Every distinct snapshot and every command
response is stored so the film can replay exactly what the product returned.

No credentials, corpus, provider client or paid call is involved.

    .venv/Scripts/python.exe docs/resumable-promo/versions/v01/source/record_backend.py \
        --out docs/resumable-promo/versions/v01/source/api-recording.json \
        --db .research_tmp/resumable-promo/v01/record.sqlite3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from zhrag.io_utils import write_json
from zhrag.service.investigation_demo import create_demo_app
from zhrag.service.server import create_server

QUESTION = "星桥集群是否可以升级？"
SUPPLEMENT = "甲版升级到乙版"
MODEL = "synthetic-demo"
# Loopback only. A shell HTTP proxy must never see these requests.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Service:
    def __init__(self, db: Path, port: int, delay: float):
        app = create_demo_app(db, delay=delay)
        self.server = create_server(app, host="127.0.0.1", port=port, log_level="warning")
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 30
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise SystemExit("synthetic service did not start")
            time.sleep(0.02)

    def stop(self) -> None:
        # Same path as Ctrl+C: InvestigationServer.shutdown signals the executor
        # before Uvicorn drains connections.
        self.server.should_exit = True
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise SystemExit("synthetic service did not stop")


class Recorder:
    def __init__(self, base: str):
        self.base = base
        self.started = time.monotonic()
        self.snapshots: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []
        self.lists: list[dict[str, Any]] = []
        self.marks: list[dict[str, Any]] = []
        self.capabilities: Any = None

    def clock(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with OPENER.open(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def mark(self, label: str) -> None:
        self.marks.append({"label": label, "t": self.clock()})

    def snapshot(self, run_id: str, label: str = "poll") -> dict[str, Any]:
        status, value = self.request("GET", f"/api/investigations/{run_id}")
        if status != 200:
            raise SystemExit(f"snapshot failed: {status} {value}")
        key = (value["cursor"], value["status"], value["version"], value["reason"])
        last = self.snapshots[-1]["value"] if self.snapshots else None
        if last is None or key != (last["cursor"], last["status"], last["version"], last["reason"]):
            self.snapshots.append({"t": self.clock(), "label": label, "value": value})
        return value

    def command(self, run: dict[str, Any], action: str, text: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"request_id": str(uuid.uuid4()), "version": run["version"]}
        if action == "supplement":
            body.update(wait_id=run["wait_id"], text=text)
        status, value = self.request("POST", f"/api/investigations/{run['run_id']}/{action}", body)
        self.commands.append(
            {"t": self.clock(), "action": action, "status": status, "request": body, "response": value}
        )
        if status != 200:
            raise SystemExit(f"{action} failed: {status} {value}")
        self.snapshots.append({"t": self.clock(), "label": f"{action}-response", "value": value})
        return value

    def until(self, run_id: str, done: Any, timeout: float = 60) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            value = self.snapshot(run_id)
            if done(value):
                return value
            if time.monotonic() > deadline:
                raise SystemExit(f"timed out waiting; last status {value['status']}")
            time.sleep(0.02)

    def listing(self, label: str) -> None:
        status, value = self.request("GET", "/api/investigations")
        if status != 200:
            raise SystemExit(f"list failed: {status}")
        self.lists.append({"t": self.clock(), "label": label, "value": value})


def in_flight(run: dict[str, Any], action: str, ordinal: int) -> bool:
    """True while the `ordinal`-th call of `action` has started but not completed."""
    calls: dict[str, dict[str, Any]] = {}
    for event in run["progress"]:
        calls.setdefault(event["invocation_id"], {"action": event["action"]})[event["phase"]] = event
    same = [call for call in calls.values() if call["action"] == action]
    return len(same) == ordinal and "started" in same[-1] and "completed" not in same[-1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.db.exists():
        raise SystemExit(f"database exists; choose a fresh path: {args.db}")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(f"http://127.0.0.1:{args.port}")

    first = Service(args.db, args.port, args.delay)
    first.start()
    recorder.mark("service-a-started")
    status, recorder.capabilities = recorder.request("GET", "/api/capabilities")
    if status != 200:
        raise SystemExit("capabilities failed")
    recorder.listing("empty")

    status, created = recorder.request(
        "POST",
        "/api/investigations",
        {"request_id": str(uuid.uuid4()), "query": QUESTION, "model": MODEL},
    )
    if status != 201:
        raise SystemExit(f"create failed: {status} {created}")
    recorder.commands.append({"t": recorder.clock(), "action": "create", "status": status,
                              "request": {"query": QUESTION, "model": MODEL}, "response": created})
    recorder.snapshots.append({"t": recorder.clock(), "label": "create-response", "value": created})
    run_id = created["run_id"]

    waiting = recorder.until(run_id, lambda run: run["status"] == "waiting_input")
    recorder.mark("waiting-input")
    resumed = recorder.command(waiting, "supplement", SUPPLEMENT)
    # Pause while the first post-supplement model decision is in flight.
    running = recorder.until(
        run_id, lambda run: run["status"] == "running" and in_flight(run, "decide", 4)
    )
    del resumed
    recorder.mark("user-pause")
    recorder.command(running, "pause")
    paused = recorder.until(run_id, lambda run: run["status"] == "paused")
    recorder.mark("paused")
    time.sleep(0.3)
    recorder.command(paused, "resume")
    # Stop the service while the next model decision is in flight.
    recorder.until(run_id, lambda run: run["status"] == "running" and in_flight(run, "decide", 5))
    recorder.mark("service-stop")
    first.stop()
    recorder.mark("service-a-stopped")

    second = Service(args.db, args.port, args.delay)
    second.start()
    recorder.mark("service-b-started")
    recorder.listing("after-restart")
    restored = recorder.snapshot(run_id, "after-restart")
    if restored["status"] != "paused" or restored["reason"] != "service_shutdown":
        raise SystemExit(f"unexpected restart state: {restored['status']} {restored['reason']}")
    recorder.command(restored, "resume")
    final = recorder.until(run_id, lambda run: run["status"] == "completed")
    recorder.mark("completed")
    recorder.listing("completed")
    events: list[dict[str, Any]] = []
    after = 0
    while True:
        status, page = recorder.request("GET", f"/api/investigations/{run_id}/events?after={after}")
        if status != 200:
            raise SystemExit("events failed")
        batch = page["events"]
        if not batch:
            break
        events.extend(batch)
        after = batch[-1]["seq"]
    second.stop()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", check=False
    ).stdout.strip()
    write_json(
        args.out,
        {
            "schema_version": 1,
            "source": "zhrag.service.investigation_demo.create_demo_app (original synthetic data)",
            "git_head": head,
            "delay_seconds": args.delay,
            "question": QUESTION,
            "supplement": SUPPLEMENT,
            "run_id": run_id,
            "capabilities": recorder.capabilities,
            "lists": recorder.lists,
            "commands": recorder.commands,
            "snapshots": recorder.snapshots,
            "events": events,
            "marks": recorder.marks,
            "final_usage": final["response"]["usage"],
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "snapshots": len(recorder.snapshots),
                "events": len(events),
                "usage": final["response"]["usage"],
                "statuses": [item["value"]["status"] for item in recorder.snapshots],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
