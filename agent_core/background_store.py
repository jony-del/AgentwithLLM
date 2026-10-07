"""Private, versioned background records and notification outbox."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from agent_core.tools.transaction import JournalStorage, _open_state, _secure_mkdir
from agent_core.permission_audit import sanitize_log_payload


class BackgroundTaskStore:
    def __init__(self, storage: JournalStorage, *, isolated: bool = False) -> None:
        self.storage = storage
        self.path = (storage.run_root if isolated else storage.recovery_root) / "background-state.json"

    def load(self) -> dict[str, Any]:
        self.storage.validate(self.path)
        if not self.path.exists():
            return {}
        if self.path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("background state exceeds size limit")
        with _open_state(self.storage, self.path, "r") as handle:
            value = json.load(handle)
        if (not isinstance(value, dict) or value.get("v") != 1
                or value.get("session_id") != self.storage.session_id
                or value.get("project_id") != self.storage.project_id):
            raise ValueError("background state ownership/schema mismatch")
        records, events = value.get("records"), value.get("events")
        if not isinstance(records, list) or not isinstance(events, list) or len(records) > 128 or len(events) > 256:
            raise ValueError("invalid background state collections")
        for record in records:
            if (not isinstance(record, dict)
                    or any(not isinstance(record.get(key), str) for key in
                           ("task_id", "task_type", "owner_run", "description", "state", "result"))
                    or any(not isinstance(record.get(key), bool) for key in ("backgrounded", "required", "notified"))
                    or not isinstance(record.get("metadata"), dict)):
                raise ValueError("invalid background task record")
        ids = {record["task_id"] for record in records}
        if len(ids) != len(records):
            raise ValueError("duplicate background task id")
        by_id = {record["task_id"]: record for record in records}
        for record in records:
            if (record["task_type"] not in {"shell", "agent", "teammate"}
                    or record["state"] not in {"pending", "running", "completed", "failed", "stopped", "lost",
                        "timed_out", "deadline", "interrupted", "max_steps", "no_progress", "unverified", "blocked", "cancelled"}):
                raise ValueError("invalid background task kind/state")
        for event in events:
            if (not isinstance(event, dict)
                    or any(not isinstance(event.get(key), str) for key in
                           ("event_id", "task_id", "owner_run", "owner_agent", "event", "task_type", "state"))
                    or event["task_id"] not in ids):
                raise ValueError("invalid background notification")
            if event["owner_run"] != by_id[event["task_id"]]["owner_run"]:
                raise ValueError("background notification owner mismatch")
            if (event["event"] not in {"finished", "interactive_input"}
                    or event["event_id"] != f"{event['task_id']}:{event['event']}"
                    or event["task_type"] != by_id[event["task_id"]]["task_type"]):
                raise ValueError("invalid background notification identity/kind")
        if len({event["event_id"] for event in events}) != len(events):
            raise ValueError("duplicate background notification id")
        return value

    def save(self, records: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
        self.storage.validate(self.path)
        _secure_mkdir(self.path.parent)
        temporary = self.path.with_name(f".background-{uuid.uuid4().hex}.tmp")
        try:
            with _open_state(self.storage, temporary, "w") as handle:
                json.dump(sanitize_log_payload({"v": 1, "session_id": self.storage.session_id,
                           "project_id": self.storage.project_id,
                           "records": records, "events": events}), handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self.storage.validate(self.path)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def confirmed_events(self, transcript: Path, expected: dict[str, str]) -> set[str]:
        """Find durable receipts even before a compaction snapshot discarded the context.

        Stream only candidate message lines; do not reconstruct the whole transcript.
        Reuse transcript schema validation rather than trusting matching text in output.
        """
        from agent_core.execution import current_execution_scope
        from agent_core.transcript import _Accumulator

        confirmed: set[str] = set()
        reverse = {message_id: event_id for event_id, message_id in expected.items()}
        try:
            handle = transcript.open("rb")
        except FileNotFoundError:
            return confirmed
        scope = current_execution_scope()
        with handle:
            for raw in handle:
                if scope is not None:
                    scope.raise_if_cancelled()
                if b'"background_event_ids"' not in raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (ValueError, RecursionError):
                    continue
                if (not isinstance(entry, dict) or entry.get("type") != "message"
                        or entry.get("session_id") != self.storage.session_id):
                    continue
                probe = _Accumulator(self.storage.session_id)
                probe.feed(raw)
                if probe.diagnostics:
                    continue
                for message_id in probe.messages.keys() & reverse.keys():
                    message = probe.messages[message_id]
                    event_id = reverse[message_id]
                    event_ids = message.metadata.get("background_event_ids")
                    ingress = message.metadata.get("prompt_ingress")
                    if (message.role == "user" and isinstance(event_ids, list) and event_id in event_ids
                            and isinstance(ingress, dict) and ingress.get("source") == "background_task"):
                        confirmed.add(event_id)
                if len(confirmed) == len(expected):
                    break
        return confirmed
