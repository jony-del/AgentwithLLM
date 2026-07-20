from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
import shutil
from pathlib import Path
from typing import Any, BinaryIO

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TASK_STATUSES = frozenset({"pending", "assigned", "in_progress", "blocked", "completed"})


class TeamError(ValueError):
    """A team operation could not be completed because the input/state is invalid."""


class TeamPermissionError(PermissionError):
    """The current agent is not allowed to make the requested team change."""


class FileLock:
    """Small cross-platform exclusive file lock.

    The lock is taken on a sidecar ``*.lock`` file, so writers can atomically replace
    the protected JSON file while holding the lock. This keeps the team subsystem
    dependency-free and works for the thread/process-level concurrency this project
    needs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: BinaryIO | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        self._file = file
        if file.seek(0, os.SEEK_END) == 0:
            file.write(b"\0")
            file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            # LK_LOCK retries once per second and gives up after ~10s with EDEADLK
            # ("Resource deadlock avoided") — under many concurrent workers that's
            # plain contention, not deadlock. Poll the non-blocking variant on a
            # short interval instead, waiting indefinitely like fcntl.flock.
            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.005)
        else:
            import importlib

            fcntl = importlib.import_module("fcntl")
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is None:
            return
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import importlib

            fcntl = importlib.import_module("fcntl")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None


class TeamStore:
    """File-backed team state shared by leader and teammate agents."""

    def __init__(self, root: str | Path = Path("runs") / "teams") -> None:
        self.root = Path(root)

    # --- public operations (async) ------------------------------------------
    # Each call offloads its blocking implementation (FileLock + JSON file IO) to
    # a worker thread so the event loop never blocks; the ``*.lock`` sidecar files
    # remain the real cross-thread/cross-process mutual exclusion.

    async def create_team(self, name: str, goal: str, leader_name: str = "leader") -> dict[str, Any]:
        return await asyncio.to_thread(self._create_team_sync, name, goal, leader_name)

    async def get_team(self, team_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_team_sync, team_id)

    async def add_member(self, team_id: str, name: str, role: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._add_member_sync, team_id, name, role)

    async def create_task(
        self,
        team_id: str,
        title: str,
        description: str,
        owner: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_task_sync, team_id, title, description, owner, priority)

    async def list_tasks(self, team_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_tasks_sync, team_id)

    async def get_task(self, team_id: str, task_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_task_sync, team_id, task_id)

    async def update_task(
        self,
        team_id: str,
        task_id: str,
        actor: str,
        *,
        status: str | None = None,
        owner: str | None = None,
        note: str | None = None,
        result: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        def call() -> tuple[dict[str, Any], str | None]:
            return self._update_task_sync(team_id, task_id, actor, status=status, owner=owner, note=note, result=result)

        return await asyncio.to_thread(call)

    async def send_message(
        self,
        team_id: str,
        from_name: str,
        to: str,
        content: str,
        *,
        task_id: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            return self._send_message_sync(team_id, from_name, to, content, task_id=task_id, kind=kind)

        return await asyncio.to_thread(call)

    async def read_inbox(self, team_id: str, agent_name: str, *, unread_only: bool = True) -> list[dict[str, Any]]:
        def call() -> list[dict[str, Any]]:
            return self._read_inbox_sync(team_id, agent_name, unread_only=unread_only)

        return await asyncio.to_thread(call)

    async def status(self, team_id: str, recent_events: int = 20) -> dict[str, Any]:
        return await asyncio.to_thread(self._status_sync, team_id, recent_events)

    async def read_events(self, team_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._read_events_sync, team_id, limit)

    async def delete_team(self, team_id: str) -> None:
        await asyncio.to_thread(self._delete_team_sync, team_id)

    # --- blocking implementations (worker-thread internals) ------------------

    def _create_team_sync(self, name: str, goal: str, leader_name: str = "leader") -> dict[str, Any]:
        name = name.strip()
        goal = goal.strip()
        leader_name = self._validate_agent_name(leader_name or "leader")
        if not name:
            raise TeamError("team name must not be empty")
        if not goal:
            raise TeamError("team goal must not be empty")

        team_id = f"team_{uuid.uuid4().hex[:12]}"
        inbox_dir = self._inbox_dir(team_id)
        inbox_dir.mkdir(parents=True, exist_ok=False)

        now = time.time()
        team = {
            "id": team_id,
            "name": name,
            "goal": goal,
            "leader": leader_name,
            "created_at": now,
            "updated_at": now,
            "members": {
                leader_name: {
                    "name": leader_name,
                    "role": "leader",
                    "created_at": now,
                    "updated_at": now,
                }
            },
        }
        tasks = {"team_id": team_id, "tasks": []}
        self._write_json(self._team_file(team_id), team)
        self._write_json(self._tasks_file(team_id), tasks)
        self._ensure_inbox(team_id, leader_name)
        self._append_event(team_id, "team_created", {"leader": leader_name, "goal": goal})
        return team

    def _get_team_sync(self, team_id: str) -> dict[str, Any]:
        path = self._team_file(team_id)
        if not path.exists():
            raise TeamError(f"unknown team: {team_id}")
        with FileLock(self._lock_file(path)):
            return self._read_json(path)

    def _add_member_sync(self, team_id: str, name: str, role: str) -> dict[str, Any]:
        name = self._validate_agent_name(name)
        role = role.strip()
        if not role:
            raise TeamError("member role must not be empty")
        path = self._team_file(team_id)
        now = time.time()
        with FileLock(self._lock_file(path)):
            team = self._read_json(path)
            members = team.setdefault("members", {})
            existing = members.get(name)
            if existing:
                existing["role"] = role
                existing["updated_at"] = now
            else:
                members[name] = {
                    "name": name,
                    "role": role,
                    "created_at": now,
                    "updated_at": now,
                }
            team["updated_at"] = now
            self._write_json(path, team)
        self._ensure_inbox(team_id, name)
        self._append_event(team_id, "member_added", {"name": name, "role": role})
        return team

    def _create_task_sync(
        self,
        team_id: str,
        title: str,
        description: str,
        owner: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        description = description.strip()
        if not title:
            raise TeamError("task title must not be empty")
        if not description:
            raise TeamError("task description must not be empty")
        if owner:
            owner = self._validate_known_member(team_id, owner)

        path = self._tasks_file(team_id)
        now = time.time()
        task: dict[str, Any] = {
            "id": f"task_{uuid.uuid4().hex[:10]}",
            "title": title,
            "description": description,
            "owner": owner,
            "status": "assigned" if owner else "pending",
            "priority": (priority or "normal").strip() or "normal",
            "created_at": now,
            "updated_at": now,
            "notes": [],
            "result": None,
        }
        with FileLock(self._lock_file(path)):
            data = self._read_json(path)
            data.setdefault("tasks", []).append(task)
            self._write_json(path, data)
        self._append_event(team_id, "task_created", {"task_id": task["id"], "owner": owner})
        return task

    def _list_tasks_sync(self, team_id: str) -> list[dict[str, Any]]:
        path = self._tasks_file(team_id)
        with FileLock(self._lock_file(path)):
            return list(self._read_json(path).get("tasks", []))

    def _get_task_sync(self, team_id: str, task_id: str) -> dict[str, Any]:
        for task in self._list_tasks_sync(team_id):
            if task.get("id") == task_id:
                return task
        raise TeamError(f"unknown task: {task_id}")

    def _update_task_sync(
        self,
        team_id: str,
        task_id: str,
        actor: str,
        *,
        status: str | None = None,
        owner: str | None = None,
        note: str | None = None,
        result: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        actor = self._validate_known_member(team_id, actor)
        if owner:
            owner = self._validate_known_member(team_id, owner)
        if status:
            status = status.strip()
            if status not in TASK_STATUSES:
                raise TeamError(f"invalid task status: {status}")

        team = self._get_team_sync(team_id)
        leader = team["leader"]
        path = self._tasks_file(team_id)
        assigned_to: str | None = None
        with FileLock(self._lock_file(path)):
            data = self._read_json(path)
            tasks = data.setdefault("tasks", [])
            task = next((item for item in tasks if item.get("id") == task_id), None)
            if task is None:
                raise TeamError(f"unknown task: {task_id}")

            current_owner = task.get("owner")
            if actor != leader:
                if current_owner not in (None, actor):
                    raise TeamPermissionError(f"{actor} cannot update task owned by {current_owner}")
                if current_owner is None and owner not in (None, actor):
                    raise TeamPermissionError(f"{actor} can only claim unowned tasks for itself")

            if current_owner is None and actor != leader and owner is None:
                owner = actor
            if owner is not None and owner != current_owner:
                task["owner"] = owner
                assigned_to = owner
                if status is None and task.get("status") == "pending":
                    task["status"] = "assigned"
            if status is not None:
                task["status"] = status
            if note:
                task.setdefault("notes", []).append({"at": time.time(), "by": actor, "content": note.strip()})
            if result is not None:
                task["result"] = result
            task["updated_at"] = time.time()
            updated = dict(task)
            self._write_json(path, data)

        self._append_event(
            team_id,
            "task_updated",
            {
                "task_id": task_id,
                "actor": actor,
                "status": updated.get("status"),
                "owner": updated.get("owner"),
            },
        )
        return updated, assigned_to

    def _send_message_sync(
        self,
        team_id: str,
        from_name: str,
        to: str,
        content: str,
        *,
        task_id: str | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        from_name = self._validate_known_member(team_id, from_name)
        to = self._validate_known_member(team_id, to)
        content = content.strip()
        if not content:
            raise TeamError("message content must not be empty")
        if task_id:
            self._get_task_sync(team_id, task_id)
        message = {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "team_id": team_id,
            "from": from_name,
            "to": to,
            "content": content,
            "task_id": task_id,
            "kind": (kind or "message").strip() or "message",
            "ts": time.time(),
        }
        inbox = self._inbox_file(team_id, to)
        with FileLock(self._lock_file(inbox)):
            inbox.parent.mkdir(parents=True, exist_ok=True)
            with inbox.open("a", encoding="utf-8") as file:
                file.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        self._append_event(team_id, "message_sent", {"message_id": message["id"], "from": from_name, "to": to})
        return message

    def _read_inbox_sync(self, team_id: str, agent_name: str, *, unread_only: bool = True) -> list[dict[str, Any]]:
        agent_name = self._validate_known_member(team_id, agent_name)
        inbox = self._inbox_file(team_id, agent_name)
        cursor = self._cursor_file(team_id, agent_name)
        with FileLock(self._lock_file(inbox)):
            messages = self._read_jsonl(inbox)
            if not unread_only:
                return messages
            start = self._read_cursor(cursor)
            unread = messages[start:]
            self._write_cursor(cursor, len(messages))
            return unread

    def _status_sync(self, team_id: str, recent_events: int = 20) -> dict[str, Any]:
        return {
            "team": self._get_team_sync(team_id),
            "tasks": self._list_tasks_sync(team_id),
            "recent_events": self._read_events_sync(team_id, limit=recent_events),
        }

    def _read_events_sync(self, team_id: str, limit: int = 20) -> list[dict[str, Any]]:
        path = self._events_file(team_id)
        with FileLock(self._lock_file(path)):
            events = self._read_jsonl(path)
        return events[-limit:]

    def _delete_team_sync(self, team_id: str) -> None:
        directory = self._team_dir(team_id).resolve()
        root = self.root.resolve()
        if directory.parent != root or not directory.is_dir():
            raise TeamError(f"unknown team: {team_id}")
        with FileLock(root / f".{team_id}.delete.lock"):
            shutil.rmtree(directory)

    # --- path helpers ------------------------------------------------------

    def _team_dir(self, team_id: str) -> Path:
        self._validate_id(team_id)
        return self.root / team_id

    def _team_file(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "team.json"

    def _tasks_file(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "tasks.json"

    def _events_file(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "events.jsonl"

    def _inbox_dir(self, team_id: str) -> Path:
        return self._team_dir(team_id) / "inbox"

    def _inbox_file(self, team_id: str, agent_name: str) -> Path:
        agent_name = self._validate_agent_name(agent_name)
        return self._inbox_dir(team_id) / f"{agent_name}.jsonl"

    def _cursor_file(self, team_id: str, agent_name: str) -> Path:
        agent_name = self._validate_agent_name(agent_name)
        return self._inbox_dir(team_id) / f"{agent_name}.cursor"

    @staticmethod
    def _lock_file(path: Path) -> Path:
        return path.with_name(path.name + ".lock")

    # --- validation / low-level IO ----------------------------------------

    @staticmethod
    def _validate_id(raw: str) -> str:
        value = str(raw).strip()
        if not _SAFE_NAME.fullmatch(value):
            raise TeamError(f"invalid id/name: {raw}")
        return value

    @classmethod
    def _validate_agent_name(cls, raw: str) -> str:
        return cls._validate_id(raw)

    def _validate_known_member(self, team_id: str, raw: str) -> str:
        name = self._validate_agent_name(raw)
        team = self._get_team_sync(team_id)
        if name not in team.get("members", {}):
            raise TeamError(f"unknown team member: {name}")
        return name

    def _ensure_inbox(self, team_id: str, agent_name: str) -> None:
        inbox = self._inbox_file(team_id, agent_name)
        inbox.parent.mkdir(parents=True, exist_ok=True)
        inbox.touch(exist_ok=True)
        self._cursor_file(team_id, agent_name).touch(exist_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise TeamError(f"missing team file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    @staticmethod
    def _read_cursor(path: Path) -> int:
        try:
            return int(path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            return 0

    @staticmethod
    def _write_cursor(path: Path, value: int) -> None:
        path.write_text(str(max(0, value)), encoding="utf-8")

    def _append_event(self, team_id: str, event: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), "event": event, **payload}
        path = self._events_file(team_id)
        with FileLock(self._lock_file(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
