"""Versioned SQLite runs, checkpoints, durable events and fenced call attempts.

Only this store mutates lifecycle state. Transactions serialize commands across
processes; a lease plus monotonically increasing epoch fences late workers.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from zhrag.io_utils import sqlite_database

SCHEMA_VERSION = 1
TERMINAL = frozenset({"completed", "cancelled", "failed", "budget_exhausted", "unrecoverable"})
OWNED = frozenset({"running", "pausing", "cancelling", "resuming"})


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class RunConflict(Exception):
    def __init__(self, code: str, status: int = 409):
        self.code, self.status = code, status
        super().__init__(code)


class LeaseLost(Exception):
    """The caller must stop; it may not commit a late external result."""


class InvestigationStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time, lease: float = 15):
        self.path, self.clock, self.lease = path, clock, lease
        with self.transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, SCHEMA_VERSION}:
                raise ValueError("unsupported investigation database schema")
            for statement in (
                """CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, create_key TEXT UNIQUE NOT NULL, request TEXT NOT NULL,
                schema_version INTEGER NOT NULL, query TEXT NOT NULL, model TEXT NOT NULL,
                profile TEXT NOT NULL, snapshot TEXT NOT NULL, settings TEXT NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL,
                epoch INTEGER NOT NULL DEFAULT 0, owner TEXT, lease_until REAL,
                active_seconds REAL NOT NULL DEFAULT 0, charged_at REAL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL, wait_id TEXT,
                clarification TEXT NOT NULL DEFAULT '', supplements TEXT NOT NULL DEFAULT '[]',
                response TEXT, next_seq INTEGER NOT NULL DEFAULT 1)""",
                """CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL, boundary TEXT NOT NULL,
                payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1)""",
                """CREATE TABLE IF NOT EXISTS events (
                run_id TEXT REFERENCES runs(id) ON DELETE CASCADE, seq INTEGER,
                schema_version INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(run_id, seq))""",
                """CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL, epoch INTEGER NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL,
                result TEXT, error TEXT)""",
                """CREATE TABLE IF NOT EXISTS commands (
                run_id TEXT REFERENCES runs(id) ON DELETE CASCADE, request_id TEXT,
                request TEXT NOT NULL, PRIMARY KEY(run_id, request_id))""",
            ):
                db.execute(statement)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with sqlite_database(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _row(db: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise RunConflict("run_not_found", 404)
        assert isinstance(row, sqlite3.Row)
        return row

    def _event(self, db: sqlite3.Connection, run_id: str, kind: str, payload: object) -> None:
        seq = self._row(db, run_id)["next_seq"]
        db.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (run_id, seq, SCHEMA_VERSION, kind, json_text(payload)),
        )
        db.execute("UPDATE runs SET next_seq=next_seq+1 WHERE id=?", (run_id,))

    def _status(self, db: sqlite3.Connection, run_id: str, status: str, reason: str = "") -> None:
        db.execute(
            "UPDATE runs SET status=?, reason=?, version=version+1, updated_at=? WHERE id=?",
            (status, reason, self.clock(), run_id),
        )
        self._event(db, run_id, "state", {"status": status, "reason": reason})

    def create(
        self,
        key: str,
        query: str,
        model: str,
        *,
        profile: str,
        snapshot: str,
        settings: dict[str, Any],
        checkpoint: dict[str, Any],
    ) -> str:
        request = json_text({"query": query, "model": model})
        with self.transaction() as db:
            previous = db.execute(
                "SELECT id, request FROM runs WHERE create_key=?", (key,)
            ).fetchone()
            if previous:
                if previous["request"] != request:
                    raise RunConflict("idempotency_conflict")
                return str(previous["id"])
            run_id, now = uuid.uuid4().hex, self.clock()
            db.execute(
                """INSERT INTO runs
                (id,create_key,request,schema_version,query,model,profile,snapshot,settings,
                 status,version,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,'queued',1,?,?)""",
                (
                    run_id,
                    key,
                    request,
                    SCHEMA_VERSION,
                    query,
                    model,
                    profile,
                    snapshot,
                    json_text(settings),
                    now,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,1)",
                (run_id, SCHEMA_VERSION, checkpoint["phase"], json_text(checkpoint)),
            )
            self._event(db, run_id, "state", {"status": "queued", "reason": ""})
            return run_id

    def _charge(self, db: sqlite3.Connection, row: sqlite3.Row) -> None:
        if row["owner"]:
            # A dead process is charged conservatively up to lease expiry, never
            # for offline/user waiting time. A clock reversal cannot refund time.
            delta = max(0.0, min(self.clock(), row["lease_until"]) - row["charged_at"])
            db.execute(
                "UPDATE runs SET active_seconds=active_seconds+?, charged_at=? WHERE id=?",
                (delta, self.clock(), row["id"]),
            )

    def recover(self) -> None:
        with self.transaction() as db:
            rows = db.execute(
                "SELECT * FROM runs WHERE owner IS NOT NULL AND lease_until<=?", (self.clock(),)
            ).fetchall()
            for row in rows:
                self._charge(db, row)
                unknown = db.execute(
                    "SELECT 1 FROM attempts WHERE run_id=? AND status='started'", (row["id"],)
                ).fetchone()
                status, reason = "interrupted", "service_interrupted"
                if row["status"] == "cancelling":
                    status, reason = "cancelled", "cancelled"
                elif unknown:
                    status, reason = "unrecoverable", "call_outcome_unknown"
                elif row["status"] == "pausing":
                    status, reason = "paused", "service_interrupted"
                db.execute(
                    "UPDATE attempts SET status='unknown', ended_at=? "
                    "WHERE run_id=? AND status='started'",
                    (self.clock(), row["id"]),
                )
                db.execute(
                    "UPDATE runs SET owner=NULL, lease_until=NULL, epoch=epoch+1 WHERE id=?",
                    (row["id"],),
                )
                self._status(db, row["id"], status, reason)

    @staticmethod
    def _owned(row: sqlite3.Row, owner: str, epoch: int, now: float) -> None:
        if row["owner"] != owner or row["epoch"] != epoch or row["lease_until"] <= now:
            raise LeaseLost()

    def claim(self, run_id: str, owner: str, profile: str, snapshot: str) -> int | None:
        self.recover()
        with self.transaction() as db:
            row = self._row(db, run_id)
            if row["status"] not in {"queued", "resuming"} or row["owner"]:
                return None
            if row["profile"] != profile or row["snapshot"] != snapshot:
                self._status(db, run_id, "unrecoverable", "snapshot_or_model_mismatch")
                return None
            epoch = int(row["epoch"]) + 1
            now = self.clock()
            db.execute(
                "UPDATE runs SET owner=?, epoch=?, lease_until=?, charged_at=? WHERE id=?",
                (owner, epoch, now + self.lease, now, run_id),
            )
            self._status(db, run_id, "running")
            return epoch

    def heartbeat(self, run_id: str, owner: str, epoch: int) -> str:
        with self.transaction() as db:
            row = self._row(db, run_id)
            self._owned(row, owner, epoch, self.clock())
            self._charge(db, row)
            db.execute(
                "UPDATE runs SET lease_until=? WHERE id=?", (self.clock() + self.lease, run_id)
            )
            return str(row["status"])

    def save(
        self,
        run_id: str,
        owner: str,
        epoch: int,
        checkpoint: dict[str, Any],
        progress: list[dict[str, Any]],
        *,
        status: str | None = None,
        reason: str = "",
        response: dict[str, Any] | None = None,
        begin: str | None = None,
        complete: str | None = None,
        result: object = None,
        error: str | None = None,
        local_operation: str | None = None,
    ) -> str | None:
        with self.transaction() as db:
            row = self._row(db, run_id)
            self._owned(row, owner, epoch, self.clock())
            self._charge(db, row)
            # A control command takes effect only at a saved boundary. Do not
            # start a call once pause/cancel has won the transaction race.
            if begin and row["status"] in {"pausing", "cancelling"}:
                return None
            db.execute(
                "UPDATE checkpoints SET payload=?, boundary=?, revision=revision+1 WHERE run_id=?",
                (json_text(checkpoint), checkpoint["phase"], run_id),
            )
            for event in progress:
                self._event(db, run_id, "progress", event)
            attempt = None
            if begin:
                attempt = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?,?,'started',?,NULL,NULL,NULL)",
                    (attempt, run_id, SCHEMA_VERSION, epoch, begin, self.clock()),
                )
            if complete:
                changed = db.execute(
                    """UPDATE attempts SET status=?,ended_at=?,result=?,error=?
                    WHERE id=? AND run_id=? AND epoch=? AND status='started'""",
                    (
                        "failed" if error else "completed",
                        self.clock(),
                        json_text(result),
                        error,
                        complete,
                        run_id,
                        epoch,
                    ),
                ).rowcount
                if changed != 1:
                    raise LeaseLost()
            if local_operation:
                # Reading an already persisted candidate has no external side
                # effect. Commit the attempt and updated evidence atomically.
                db.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?,?,'completed',?,?,?,NULL)",
                    (
                        uuid.uuid4().hex,
                        run_id,
                        SCHEMA_VERSION,
                        epoch,
                        local_operation,
                        self.clock(),
                        self.clock(),
                        json_text(result),
                    ),
                )
            if response is not None:
                db.execute("UPDATE runs SET response=? WHERE id=?", (json_text(response), run_id))
            if status:
                if (
                    status == "unrecoverable"
                    and db.execute(
                        "SELECT 1 FROM attempts WHERE run_id=? AND status='started'", (run_id,)
                    ).fetchone()
                ):
                    reason = "call_outcome_unknown"
                    db.execute(
                        "UPDATE attempts SET status='unknown', ended_at=? "
                        "WHERE run_id=? AND status='started'",
                        (self.clock(), run_id),
                    )
                if row["status"] == "cancelling":
                    status, reason = "cancelled", "cancelled"
                    # No answer is published after cancellation wins the race.
                    db.execute("UPDATE runs SET response=NULL WHERE id=?", (run_id,))
                self._status(db, run_id, status, reason)
                if status == "waiting_input":
                    question = (response or {}).get("clarification", "")
                    db.execute(
                        "UPDATE runs SET wait_id=?,clarification=? WHERE id=?",
                        (uuid.uuid4().hex, question, run_id),
                    )
                db.execute("UPDATE runs SET owner=NULL,lease_until=NULL WHERE id=?", (run_id,))
            return attempt

    def command(
        self,
        run_id: str,
        request_id: str,
        action: str,
        *,
        version: int,
        wait_id: str | None = None,
        text: str | None = None,
    ) -> None:
        request = json_text(
            {"action": action, "version": version, "wait_id": wait_id, "text": text}
        )
        self.recover()
        with self.transaction() as db:
            row = self._row(db, run_id)
            prior = db.execute(
                "SELECT request FROM commands WHERE run_id=? AND request_id=?", (run_id, request_id)
            ).fetchone()
            if prior:
                if prior["request"] != request:
                    raise RunConflict("idempotency_conflict")
                return
            if row["version"] != version:
                raise RunConflict("stale_run_version")
            status = row["status"]
            if status in TERMINAL:
                raise RunConflict("run_is_terminal")
            if action == "supplement":
                if status != "waiting_input" or wait_id != row["wait_id"]:
                    raise RunConflict("stale_clarification")
                if not text or not text.strip() or len(text) > 2000:
                    raise RunConflict("invalid_supplement", 422)
                supplements = json.loads(row["supplements"])
                supplements.append(
                    {"wait_id": wait_id, "question": row["clarification"], "text": text}
                )
                db.execute(
                    "UPDATE runs SET supplements=?,wait_id=NULL,clarification='',response=NULL "
                    "WHERE id=?",
                    (json_text(supplements), run_id),
                )
                self._event(db, run_id, "supplement", supplements[-1])
                target = "resuming"
            elif action == "resume" and status in {"paused", "interrupted"}:
                target = "resuming"
            elif action == "pause" and status in {"running", "resuming", "queued", "pausing"}:
                target = "pausing" if row["owner"] else "paused"
            elif action == "cancel":
                target = "cancelling" if row["owner"] else "cancelled"
                db.execute(
                    "UPDATE runs SET response=NULL,wait_id=NULL,clarification='' WHERE id=?",
                    (run_id,),
                )
            else:
                raise RunConflict("invalid_run_state")
            self._status(db, run_id, target)
            db.execute("INSERT INTO commands VALUES (?,?,?)", (run_id, request_id, request))

    def get(self, run_id: str) -> dict[str, Any]:
        with sqlite_database(self.path) as db:
            db.execute("BEGIN")
            row = dict(self._row(db, run_id))
            cp = db.execute("SELECT * FROM checkpoints WHERE run_id=?", (run_id,)).fetchone()
            row["checkpoint"] = json.loads(cp["payload"])
            row["checkpoint_revision"] = cp["revision"]
            for key in ("settings", "supplements", "response"):
                row[key] = json.loads(row[key]) if row[key] is not None else None
            return row

    def list_runs(self, limit: int = 100) -> list[str]:
        with sqlite_database(self.path) as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT id FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            ]

    def pending(self) -> list[str]:
        with sqlite_database(self.path) as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT id FROM runs WHERE status IN ('queued','resuming') AND owner IS NULL "
                    "ORDER BY created_at"
                )
            ]

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with sqlite_database(self.path) as db:
            row = self._row(db, run_id)
            if after < 0 or after >= row["next_seq"]:
                raise RunConflict("invalid_event_cursor", 422)
            return [
                {
                    "run_id": run_id,
                    "seq": item["seq"],
                    "schema_version": item["schema_version"],
                    "kind": item["kind"],
                    "payload": json.loads(item["payload"]),
                }
                for item in db.execute(
                    "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 256",
                    (run_id, after),
                )
            ]

    def attempts(self, run_id: str) -> list[dict[str, Any]]:
        with sqlite_database(self.path) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,epoch,kind,status,started_at,ended_at,error FROM attempts "
                    "WHERE run_id=? ORDER BY started_at,id",
                    (run_id,),
                )
            ]

    def delete(self, run_id: str) -> None:
        with self.transaction() as db:
            if self._row(db, run_id)["status"] not in TERMINAL:
                raise RunConflict("cancel_before_delete")
            db.execute("DELETE FROM runs WHERE id=?", (run_id,))
        # Reclaim WAL pages too; backups/filesystem snapshots are outside this API.
        with sqlite_database(self.path) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()
