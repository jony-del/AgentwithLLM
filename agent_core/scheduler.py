"""User-level cron store and delivery router; never invokes a model itself."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import time
import uuid
from zoneinfo import ZoneInfo


class CronError(ValueError):
    pass


def _parse_field(raw: str, minimum: int, maximum: int) -> set[int]:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise CronError("empty cron field component")
        base, slash, step_raw = part.partition("/")
        step = int(step_raw) if slash else 1
        if step <= 0:
            raise CronError("cron step must be positive")
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            left, right = base.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(base)
        if start < minimum or end > maximum or start > end:
            raise CronError(f"cron value outside {minimum}-{maximum}")
        values.update(range(start, end + 1, step))
    return values


@dataclass(frozen=True, slots=True)
class CronExpression:
    minutes: set[int]
    hours: set[int]
    days: set[int]
    months: set[int]
    weekdays: set[int]

    @classmethod
    def parse(cls, value: str) -> "CronExpression":
        fields = value.split()
        if len(fields) != 5:
            raise CronError("cron must contain exactly five fields")
        return cls(
            _parse_field(fields[0], 0, 59), _parse_field(fields[1], 0, 23),
            _parse_field(fields[2], 1, 31), _parse_field(fields[3], 1, 12),
            _parse_field(fields[4], 0, 6),
        )

    def matches(self, value: datetime) -> bool:
        cron_weekday = (value.weekday() + 1) % 7
        return (
            value.minute in self.minutes and value.hour in self.hours and value.day in self.days
            and value.month in self.months and cron_weekday in self.weekdays
        )

    def next_after(self, timestamp: float, timezone: str) -> float:
        zone = ZoneInfo(timezone)
        current = datetime.fromtimestamp(timestamp, zone).replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(60 * 24 * 366 * 2):
            if self.matches(current):
                return current.timestamp()
            current += timedelta(minutes=1)
        raise CronError("cron has no occurrence within two years")


class SchedulerStore:
    def __init__(self, path: str | Path, *, max_jobs: int = 50, max_prompt_chars: int = 16_000) -> None:
        self.path = Path(path).expanduser()
        self.max_jobs = max_jobs
        self.max_prompt_chars = max_prompt_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner_session TEXT NOT NULL, owner_agent TEXT NOT NULL,
                    schedule TEXT NOT NULL, timezone TEXT NOT NULL, prompt TEXT NOT NULL,
                    persistent INTEGER NOT NULL, one_shot INTEGER NOT NULL DEFAULT 0,
                    next_run REAL NOT NULL, last_run REAL, missed_count INTEGER NOT NULL DEFAULT 0,
                    inflight INTEGER NOT NULL DEFAULT 0, coalesced INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS heartbeats (
                    owner_session TEXT NOT NULL, owner_agent TEXT NOT NULL, expires_at REAL NOT NULL,
                    PRIMARY KEY(owner_session, owner_agent)
                );
                CREATE TABLE IF NOT EXISTS deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, owner_session TEXT NOT NULL,
                    owner_agent TEXT NOT NULL, prompt TEXT NOT NULL, due_at REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                """
            )

    def create(
        self, *, owner_session: str, owner_agent: str, schedule: str, timezone: str,
        prompt: str, persistent: bool, one_shot: bool = False, now: float | None = None,
    ) -> dict[str, object]:
        if not prompt.strip() or len(prompt) > self.max_prompt_chars:
            raise CronError(f"prompt must contain 1-{self.max_prompt_chars} characters")
        expression = CronExpression.parse(schedule)
        now = time.time() if now is None else now
        next_run = expression.next_after(now, timezone)
        job_id = f"cron_{uuid.uuid4().hex[:12]}"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            if count >= self.max_jobs:
                db.execute("ROLLBACK")
                raise CronError(f"scheduler job limit reached ({self.max_jobs})")
            db.execute(
                "INSERT INTO jobs(id,owner_session,owner_agent,schedule,timezone,prompt,persistent,one_shot,next_run,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (job_id, owner_session, owner_agent, schedule, timezone, prompt,
                 int(persistent), int(one_shot), next_run, now),
            )
            db.execute("COMMIT")
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, object]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise CronError(f"unknown cron job: {job_id}")
        return dict(row)

    def list(self, *, owner_session: str | None = None, owner_agent: str | None = None) -> list[dict[str, object]]:
        query = "SELECT * FROM jobs"
        values: list[object] = []
        clauses: list[str] = []
        if owner_session is not None:
            clauses.append("owner_session=?")
            values.append(owner_session)
        if owner_agent is not None:
            clauses.append("owner_agent=?")
            values.append(owner_agent)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY next_run,id"
        with self._connect() as db:
            return [dict(row) for row in db.execute(query, values).fetchall()]

    def delete(self, job_id: str, *, owner_session: str, owner_agent: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM jobs WHERE id=? AND owner_session=? AND owner_agent=?",
                (job_id, owner_session, owner_agent),
            )
        return cursor.rowcount > 0

    def delete_session_jobs(self, owner_session: str, owner_agent: str) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM jobs WHERE owner_session=? AND owner_agent=? AND persistent=0",
                (owner_session, owner_agent),
            )
        return cursor.rowcount

    def heartbeat(
        self, owner_session: str, owner_agent: str, *, ttl: float = 60,
        now: float | None = None,
    ) -> builtins.list[dict[str, object]]:
        """Refresh liveness and enqueue at most one catch-up for each missed recurring job."""
        now = time.time() if now is None else now
        catchups: builtins.list[dict[str, object]] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO heartbeats(owner_session,owner_agent,expires_at) VALUES(?,?,?) "
                "ON CONFLICT(owner_session,owner_agent) DO UPDATE SET expires_at=excluded.expires_at",
                (owner_session, owner_agent, now + ttl),
            )
            missed = db.execute(
                "SELECT * FROM jobs WHERE owner_session=? AND owner_agent=? "
                "AND missed_count>0 AND inflight=0 AND one_shot=0 ORDER BY next_run",
                (owner_session, owner_agent),
            ).fetchall()
            for row in missed:
                job = dict(row)
                next_run = CronExpression.parse(str(job["schedule"])).next_after(
                    now, str(job["timezone"])
                )
                cursor = db.execute(
                    "INSERT INTO deliveries(job_id,owner_session,owner_agent,prompt,due_at,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (job["id"], owner_session, owner_agent, job["prompt"], now, now),
                )
                db.execute(
                    "UPDATE jobs SET inflight=1,last_run=?,next_run=?,missed_count=0 WHERE id=?",
                    (now, next_run, job["id"]),
                )
                catchups.append({
                    "delivery_id": cursor.lastrowid, "job_id": job["id"],
                    "prompt": job["prompt"], "catch_up": True,
                })
            db.execute("COMMIT")
        return catchups

    def route_due(self, *, now: float | None = None) -> builtins.list[dict[str, object]]:
        """Queue due prompts only for live agents; coalesce overlapping occurrences."""
        now = time.time() if now is None else now
        delivered: builtins.list[dict[str, object]] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            jobs = db.execute("SELECT * FROM jobs WHERE next_run<=? ORDER BY next_run", (now,)).fetchall()
            for row in jobs:
                job = dict(row)
                live = db.execute(
                    "SELECT 1 FROM heartbeats WHERE owner_session=? AND owner_agent=? AND expires_at>?",
                    (job["owner_session"], job["owner_agent"], now),
                ).fetchone() is not None
                expression = CronExpression.parse(str(job["schedule"]))
                next_run = expression.next_after(now, str(job["timezone"]))
                if job["inflight"]:
                    db.execute("UPDATE jobs SET coalesced=1,next_run=? WHERE id=?", (next_run, job["id"]))
                    continue
                if not live:
                    if job["one_shot"]:
                        db.execute(
                            "UPDATE jobs SET missed_count=1,next_run=? WHERE id=?",
                            (253402300799.0, job["id"]),
                        )
                    else:
                        db.execute(
                            "UPDATE jobs SET missed_count=missed_count+1,next_run=? WHERE id=?",
                            (next_run, job["id"]),
                        )
                    continue
                cursor = db.execute(
                    "INSERT INTO deliveries(job_id,owner_session,owner_agent,prompt,due_at,created_at) VALUES(?,?,?,?,?,?)",
                    (job["id"], job["owner_session"], job["owner_agent"], job["prompt"], job["next_run"], now),
                )
                db.execute(
                    "UPDATE jobs SET inflight=1,last_run=?,next_run=?,missed_count=0 WHERE id=?",
                    (now, 253402300799.0 if job["one_shot"] else next_run, job["id"]),
                )
                delivered.append({"delivery_id": cursor.lastrowid, "job_id": job["id"], "prompt": job["prompt"]})
            db.execute("COMMIT")
        return delivered

    def pending(
        self, owner_session: str, owner_agent: str
    ) -> builtins.list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM deliveries WHERE owner_session=? AND owner_agent=? AND state='pending' ORDER BY id",
                (owner_session, owner_agent),
            ).fetchall()
        return [dict(row) for row in rows]

    def missed_one_shots(
        self, owner_session: str, owner_agent: str
    ) -> builtins.list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE owner_session=? AND owner_agent=? "
                "AND one_shot=1 AND missed_count>0 AND inflight=0 ORDER BY created_at",
                (owner_session, owner_agent),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_missed_one_shot(self, job_id: str, *, deliver: bool, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE id=? AND one_shot=1 AND missed_count>0 AND inflight=0",
                (job_id,),
            ).fetchone()
            if row is None:
                db.execute("ROLLBACK")
                raise CronError(f"unknown missed one-shot job: {job_id}")
            if not deliver:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            else:
                db.execute(
                    "INSERT INTO deliveries(job_id,owner_session,owner_agent,prompt,due_at,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (job_id, row["owner_session"], row["owner_agent"], row["prompt"], now, now),
                )
                db.execute(
                    "UPDATE jobs SET inflight=1,last_run=?,missed_count=0 WHERE id=?", (now, job_id)
                )
            db.execute("COMMIT")

    def complete_delivery(self, delivery_id: int) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT job_id FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
            if row is not None:
                db.execute("UPDATE deliveries SET state='completed' WHERE id=?", (delivery_id,))
                one_shot = db.execute(
                    "SELECT one_shot FROM jobs WHERE id=?", (row["job_id"],)
                ).fetchone()
                if one_shot is not None and one_shot["one_shot"]:
                    db.execute("DELETE FROM jobs WHERE id=?", (row["job_id"],))
                else:
                    db.execute("UPDATE jobs SET inflight=0,coalesced=0 WHERE id=?", (row["job_id"],))
            db.execute("COMMIT")
