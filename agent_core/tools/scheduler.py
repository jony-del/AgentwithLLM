from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.permission_types import PermissionContext, PermissionResult
from agent_core.scheduler import CronError, SchedulerStore
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


def _store(session: Any) -> SchedulerStore:
    if session.scheduler_store is None:
        config = session.tool_suite.scheduler
        session.scheduler_store = SchedulerStore(
            config.database_path(), max_jobs=config.max_jobs, max_prompt_chars=config.max_prompt_chars
        )
    return session.scheduler_store


class _CronTool(SessionAwareMixin, Tool):
    deferred = True

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("scheduler", "jobs", "write"),))


@builtin_tool
class CronCreateTool(_CronTool):
    name = "cron_create"
    description = "Create a five-field local-time cron prompt delivery for this existing agent session."
    input_schema = {
        "type": "object",
        "properties": {
            "schedule": {"type": "string"}, "timezone": {"type": "string"},
            "prompt": {"type": "string"}, "persistent": {"type": "boolean"},
            "one_shot": {"type": "boolean"},
        },
        "required": ["schedule", "timezone", "prompt"],
    }
    risk = ToolRisk.WRITE

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        if arguments.get("persistent"):
            return PermissionResult.ask(
                "persistent cron creation requires interactive confirmation",
                bypass_immune=True, classifier_approvable=False,
            )
        return PermissionResult.allow("session cron is bounded by this agent lifetime")

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        persistent = bool(arguments.get("persistent", False))
        if persistent and self.session.parent_agent_id is not None:
            return ToolResult(self.name, "Teammates cannot create durable cron jobs.", ok=False)
        try:
            store = _store(self.session)
            await asyncio.to_thread(
                store.heartbeat, self.session.session_id, self.session.agent_id, ttl=120
            )
            job = await asyncio.to_thread(
                store.create,
                owner_session=self.session.session_id, owner_agent=self.session.agent_id,
                schedule=str(arguments.get("schedule", "")), timezone=str(arguments.get("timezone", "")),
                prompt=str(arguments.get("prompt", "")), persistent=persistent,
                one_shot=bool(arguments.get("one_shot", False)),
            )
        except (CronError, OSError, ValueError, sqlite3.Error) as exc:
            return ToolResult(self.name, f"Cron creation failed: {exc}", ok=False)
        return ToolResult(self.name, json.dumps(job, indent=2, default=str), metadata={"job_id": job["id"]})


@builtin_tool
class CronListTool(_CronTool):
    name = "cron_list"
    description = "List cron jobs owned by this agent."
    input_schema = {"type": "object", "properties": {}, "required": []}
    risk = ToolRisk.READ

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        jobs = await asyncio.to_thread(
            _store(self.session).list, owner_session=self.session.session_id, owner_agent=self.session.agent_id
        )
        return ToolResult(self.name, json.dumps(jobs, indent=2, default=str), metadata={"count": len(jobs)})


@builtin_tool
class CronDeleteTool(_CronTool):
    name = "cron_delete"
    description = "Delete a cron job owned by this agent. Persistent deletion requires confirmation."
    input_schema = {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}
    risk = ToolRisk.WRITE

    async def check_permissions(self, arguments: dict[str, Any], context: PermissionContext) -> PermissionResult:
        try:
            job = _store(self.session).get(str(arguments.get("job_id", "")))
        except Exception:
            return PermissionResult.allow("unknown jobs cannot be deleted")
        if job.get("persistent"):
            return PermissionResult.ask(
                "persistent cron deletion requires interactive confirmation",
                bypass_immune=True, classifier_approvable=False,
            )
        return PermissionResult.allow("session cron deletion is local cleanup")

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        deleted = await asyncio.to_thread(
            _store(self.session).delete, str(arguments.get("job_id", "")),
            owner_session=self.session.session_id, owner_agent=self.session.agent_id,
        )
        return ToolResult(self.name, "Deleted." if deleted else "Cron job not found.", ok=deleted)
