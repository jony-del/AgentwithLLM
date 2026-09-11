"""Resumable session transcripts — append-only Message round-trip JSONL.

This is the persistence layer behind ``--resume`` / ``--continue`` / ``--fork-session``
and ``sessions list``. It is deliberately distinct from ``storage.JSONLRunLogger``: that
logger is a one-way *event* log for debugging (``runs/*.jsonl``) whose records cannot be
faithfully turned back into :class:`~agent_core.models.Message` objects; this module
stores the conversation itself so it can be reloaded and continued.

Layout (mirrors the reference project's ``~/.claude/projects/{cwd}/{sessionId}.jsonl``)::

    {root}/{sanitized_cwd}/{session_id}.jsonl                # a session transcript
    {root}/{sanitized_cwd}/{session_id}/subagents/agent-*.jsonl  # sidechains

Each line is one entry. ``{"type": "message", ...Message.to_dict(), session_id, cwd,
git_branch, ts}`` for conversation turns; ``{"type": <kind>, ...}`` for metadata
(``custom-title``, ``tag``).

Messages form a tree via ``uuid``/``parent_uuid``; a linear conversation is reconstructed
by following ``parent_uuid`` back from a leaf (the root's ``parent_uuid`` is ``None``).
The transcript is append-only: nothing already written is ever rewritten. Compaction is
persisted as ``compaction_snapshot`` records (schema v4, sha256-checksummed), and a resume
reconstructs the conversation from the last snapshot — the pre-boundary messages stay
on disk but are no longer loaded. Forking clones a chain under a fresh ``session_id``
with new, re-linked uuids, leaving the source file untouched.

Frozen contract (schema v4): ``compaction_snapshot`` is the only authoritative compaction
boundary record. ``relink`` records are legacy read-only compatibility — the load path
still applies them (last-wins), but no production code writes them any more; the read
support is scheduled for removal at schema v5.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .file_lock import FileLock
from .models import Message
from .permission_audit import sanitize_log_payload

# Entry "type" values that are NOT conversation messages.
_TITLE = "custom-title"
_TAG = "tag"
_RELINK = "relink"
_SESSION = "session"
_COMPACTION_SNAPSHOT = "compaction_snapshot"

# Schema version stamped on every transcript entry (the "v" field). Bump on breaking
# changes to the entry shape; loaders stay tolerant of records without it (pre-v1).
SCHEMA_VERSION = 4

# Above this file size, the resume load skips everything before the last compaction
# boundary (reading only post-boundary bytes + a cheap pre-boundary metadata rescue),
# mirroring the reference's SKIP_PRECOMPACT_THRESHOLD. Small files are read whole.
SKIP_PRECOMPACT_THRESHOLD = 5 * 1024 * 1024

# Byte signature of a compact-boundary message line (we control the serialization, so the
# ``json.dumps`` default ``": "`` separator is stable). Used to locate the last boundary
# without JSON-parsing every line.
_BOUNDARY_MARKER = b'"compact_boundary": true'
_SNAPSHOT_MARKER = b'"type": "compaction_snapshot"'


def sanitize_project(cwd: str | Path) -> str:
    """Turn an absolute cwd into a flat, filesystem-safe directory name.

    Each non-alphanumeric character maps to ``-`` individually (matching the reference's
    scheme), so ``E:\\ZNGZ\\Code_copy`` becomes ``E--ZNGZ-Code-copy`` — distinct cwds
    never collide, and the same cwd always resolves to the same project dir.
    """
    resolved = os.path.normcase(str(Path(cwd).resolve()))
    slug = "-".join(filter(None, re.split(r"[^A-Za-z0-9]+", resolved)))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96] or 'project'}-{digest}"


def _legacy_sanitize_project(cwd: str | Path) -> str:
    resolved = str(Path(cwd).resolve())
    return "".join(c if c.isalnum() else "-" for c in resolved)


def project_dir(root: str | Path, cwd: str | Path) -> Path:
    """The per-project directory under ``root`` for a given working directory."""
    base = Path(root).expanduser()
    return base / sanitize_project(cwd)


def new_session_id() -> str:
    return uuid.uuid4().hex


def _git_branch(cwd: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    branch = (out.stdout or "").strip()
    return branch or None


class TranscriptStore:
    """Append-only writer for one session file (best-effort; never fails a run).

    The session directory is created lazily on first successful write. Concurrent appends
    from worker threads (the same agent's overlapping ``to_thread`` offloads) are
    serialized by a lock, exactly like :class:`storage.JSONLRunLogger`.
    """

    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        session_id: str,
        *,
        agent_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.workspace = str(Path(workspace).resolve())
        self._cwd_branch = _git_branch(self.workspace)
        proj = project_dir(root, workspace)
        if agent_id is None:
            self.path = proj / f"{session_id}.jsonl"
        else:
            # Sub-agent transcripts live beside the parent session as sidechains.
            self.path = proj / session_id / "subagents" / f"agent-{agent_id}.jsonl"
        self._lock = threading.Lock()
        self._warned = False
        self.last_error: str | None = None
        self._round_checksums: set[str] | None = None
        self._round_index_size: int | None = None
        self._round_index_path = self.path.with_suffix(self.path.suffix + ".round-index.json")
        self._process_lock_path = self.path.with_suffix(self.path.suffix + ".write.lock")
        self._activity_path = self.path.with_suffix(
            self.path.suffix + f".active.{os.getpid()}"
        )

    def _session_record(self) -> dict[str, object]:
        return {
            "type": _SESSION,
            "v": SCHEMA_VERSION,
            "session_id": self.session_id,
            "cwd": self.workspace,
            "project_id": sanitize_project(self.workspace),
            "ts": time.time(),
        }

    def _ensure_session_header_locked(self, file) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        file.write(json.dumps(self._session_record(), ensure_ascii=False) + "\n")

    def _touch_activity_locked(self) -> None:
        self._activity_path.write_text(
            json.dumps({"pid": os.getpid(), "ts": time.time()}), encoding="utf-8"
        )

    def close(self) -> None:
        try:
            self._activity_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def append_message(self, message: Message) -> bool:
        message_data = message.to_dict()
        if message.metadata.get("sensitive"):
            message_data["content"] = "<redacted-sensitive-tool-output>"
        message_data = sanitize_log_payload(message_data)
        record = {
            "type": "message",
            "v": SCHEMA_VERSION,
            **message_data,
            "session_id": self.session_id,
            "cwd": self.workspace,
            "git_branch": self._cwd_branch,
            "ts": time.time(),
        }
        return await asyncio.to_thread(self._write_sync, record)

    async def append_meta(self, kind: str, payload: dict) -> bool:
        record = {
            "type": kind,
            "v": SCHEMA_VERSION,
            "session_id": self.session_id,
            "ts": time.time(),
            **payload,
        }
        return await asyncio.to_thread(self._write_sync, record)

    async def append_tool_round(
        self,
        assistant: Message,
        tool_results: list[Message],
        execution_manifest: dict[str, object],
    ) -> bool:
        """Persist an assistant tool request and every result as one checksummed line."""

        record = self._tool_round_record(assistant, tool_results, execution_manifest)
        return await asyncio.to_thread(self._write_tool_round_once_sync, record)

    async def append_compaction_snapshot(
        self,
        messages: list[Message],
        *,
        source_head: str | None,
    ) -> bool:
        """Persist the complete ordered resumable conversation as one record.

        Frozen contract (schema v4): this is the only authoritative compaction
        boundary record. The snapshot carries the full post-fold chain (parents
        re-linked onto the new summary root) plus a sha256 checksum over the
        canonical ``messages`` + ``source_head`` body; on load, a valid snapshot
        replaces everything before it (last boundary wins).
        """

        serialized: list[dict[str, object]] = []
        parent: str | None = None
        for message in messages:
            value = message.to_dict()
            value["parent_uuid"] = parent
            value["parent_id"] = parent
            if message.metadata.get("sensitive"):
                value["content"] = "<redacted-sensitive-tool-output>"
            clean = sanitize_log_payload(value)
            serialized.append(clean)
            parent = message.uuid
        body = {"messages": serialized, "source_head": source_head}
        canonical = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        record = {
            "type": _COMPACTION_SNAPSHOT,
            "v": SCHEMA_VERSION,
            **body,
            "checksum": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "compact_boundary": True,
            "session_id": self.session_id,
            "cwd": self.workspace,
            "git_branch": self._cwd_branch,
            "ts": time.time(),
        }
        return await asyncio.to_thread(self._write_sync, record)

    def _tool_round_record(
        self,
        assistant: Message,
        tool_results: list[Message],
        execution_manifest: dict[str, object],
    ) -> dict[str, object]:
        messages: list[dict[str, object]] = []
        for message in [assistant, *tool_results]:
            value = message.to_dict()
            if message.metadata.get("sensitive"):
                value["content"] = "<redacted-sensitive-tool-output>"
            messages.append(sanitize_log_payload(value))
        body = {
            "messages": messages,
            "execution_manifest": sanitize_log_payload(execution_manifest),
        }
        canonical = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        return {
            "type": "tool_round",
            "v": SCHEMA_VERSION,
            **body,
            "checksum": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "session_id": self.session_id,
            "cwd": self.workspace,
            "git_branch": self._cwd_branch,
            "ts": time.time(),
        }

    def recover_tool_round(self, payload: dict[str, object]) -> bool:
        """Recovery callback used for committed rounds missing transcript history."""

        if payload.get("session_id") not in {None, self.session_id}:
            return False
        expected_path = str(payload.get("transcript_path") or "")
        if expected_path and Path(expected_path).resolve() != self.path.resolve():
            return False
        assistant_raw = payload.get("assistant")
        results_raw = payload.get("tool_results")
        manifest = payload.get("execution_manifest")
        if (
            not isinstance(assistant_raw, dict)
            or not isinstance(results_raw, list)
            or not isinstance(manifest, dict)
        ):
            return False
        try:
            assistant = Message.from_dict(assistant_raw)
            tool_results = [
                Message.from_dict(item) for item in results_raw if isinstance(item, dict)
            ]
            if len(tool_results) != len(results_raw):
                return False
            record = self._tool_round_record(assistant, tool_results, manifest)
            return self._write_tool_round_once_sync(record)
        except (KeyError, TypeError, ValueError):
            return False

    def _write_tool_round_once_sync(self, record: dict[str, object]) -> bool:
        """Idempotently append a checksummed round under the transcript lock."""

        checksum = str(record.get("checksum") or "")
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock, FileLock(self._process_lock_path):
                checksums = self._load_round_index_locked()
                if checksum and checksum in checksums:
                    return True
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", errors="replace") as file:
                    self._ensure_session_header_locked(file)
                    file.write(line)
                    file.flush()
                    try:
                        os.fsync(file.fileno())
                    except OSError:
                        pass
                self._touch_activity_locked()
                if checksum:
                    checksums.add(checksum)
                    try:
                        self._write_round_index_locked(checksums)
                    except OSError:
                        pass
                self.last_error = None
            return True
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if not self._warned:
                print(
                    f"[transcript] write failed ({exc}); resume disabled this run",
                    file=sys.stderr,
                )
                self._warned = True
            return False

    def _load_round_index_locked(self) -> set[str]:
        if self._round_checksums is not None:
            try:
                current_size = self.path.stat().st_size if self.path.exists() else 0
            except OSError:
                current_size = -1
            if current_size == self._round_index_size:
                return self._round_checksums
            self._round_checksums = None
            self._round_index_size = None
        try:
            raw = json.loads(self._round_index_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw, dict)
                and raw.get("v") == 1
                and raw.get("transcript_size") == (self.path.stat().st_size if self.path.exists() else 0)
                and isinstance(raw.get("checksums"), list)
            ):
                self._round_checksums = {
                    item for item in raw["checksums"] if isinstance(item, str)
                }
                self._round_index_size = int(raw["transcript_size"])
                return self._round_checksums
        except (OSError, ValueError, TypeError):
            pass
        checksums: set[str] = set()
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as file:
                for line in file:
                    if '"type": "tool_round"' not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    checksum = entry.get("checksum") if isinstance(entry, dict) else None
                    if isinstance(checksum, str):
                        checksums.add(checksum)
        except OSError:
            pass
        self._round_checksums = checksums
        try:
            self._round_index_size = self.path.stat().st_size if self.path.exists() else 0
        except OSError:
            self._round_index_size = None
        return checksums

    def _write_round_index_locked(self, checksums: set[str]) -> None:
        transcript_size = self.path.stat().st_size
        payload = {
            "v": 1,
            "transcript_size": transcript_size,
            "checksums": sorted(checksums),
        }
        temporary = self._round_index_path.with_suffix(self._round_index_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._round_index_path)
        self._round_index_size = transcript_size

    def _refresh_round_index_size_locked(self, previous_size: int) -> None:
        """Advance a valid sidecar after a non-round append without rescanning history."""

        try:
            if (
                self._round_checksums is not None
                and self._round_index_size == previous_size
            ):
                self._write_round_index_locked(self._round_checksums)
                return
            raw = json.loads(self._round_index_path.read_text(encoding="utf-8"))
            checksums = raw.get("checksums") if isinstance(raw, dict) else None
            if (
                raw.get("v") != 1
                or raw.get("transcript_size") != previous_size
                or not isinstance(checksums, list)
            ):
                return
            self._round_checksums = {item for item in checksums if isinstance(item, str)}
            self._write_round_index_locked(self._round_checksums)
        except (OSError, ValueError, TypeError, AttributeError):
            return

    async def append_relink(self, uuid: str, parent_uuid: str | None) -> bool:
        """Record that ``uuid``'s parent should be re-pointed to ``parent_uuid`` on load.

        LEGACY (schema v4, frozen): ``compaction_snapshot`` is the only authoritative
        boundary record; no production code calls this any more. The load path still
        applies relinks last-wins for old transcripts; removal is scheduled at schema v5.
        """
        return await self.append_meta(_RELINK, {"uuid": uuid, "parent_uuid": parent_uuid})

    def _write_sync(self, record: dict) -> bool:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock, FileLock(self._process_lock_path):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                previous_size = self.path.stat().st_size if self.path.exists() else 0
                with self.path.open("a", encoding="utf-8", errors="replace") as file:
                    self._ensure_session_header_locked(file)
                    file.write(line)
                    file.flush()
                    try:
                        os.fsync(file.fileno())
                    except OSError:
                        pass
                self._touch_activity_locked()
                self._refresh_round_index_size_locked(previous_size)
                self.last_error = None
            return True
        except OSError as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            # Persistence is best-effort: a transcript write must never crash an
            # otherwise healthy run. Warn once, then stay quiet.
            if not self._warned:
                print(f"[transcript] write failed ({exc}); resume disabled this run", file=sys.stderr)
                self._warned = True
            return False


# --------------------------------------------------------------------------- read side


@dataclass(frozen=True, slots=True)
class TranscriptDiagnostic:
    code: str
    line: int | None = None
    entry_type: str | None = None
    detail: str = ""


@dataclass(slots=True)
class LoadedTranscript:
    path: Path
    session_id: str
    messages: dict[str, Message]          # uuid -> Message, in file order
    order: list[str] = field(default_factory=list)   # uuids in append order
    title: str | None = None
    tag: str | None = None
    git_branch: str | None = None
    workspace: Path | None = None
    diagnostics: list[TranscriptDiagnostic] = field(default_factory=list)

    @property
    def first_prompt(self) -> str:
        for uid in self.order:
            msg = self.messages[uid]
            if msg.role == "user" and msg.content.strip():
                flat = " ".join(msg.content.split())
                return flat[:200]
        return ""

    @property
    def message_count(self) -> int:
        return len(self.order)

    def latest_leaf(self) -> str | None:
        """The most recent leaf uuid — a message that is no one's parent.

        Resume continues from here. Walking the file backward finds the newest leaf even
        if a fork wrote sibling branches into the same file.
        """
        if not self.order:
            return None
        parents = {m.parent_uuid for m in self.messages.values() if m.parent_uuid}
        for uid in reversed(self.order):
            if uid not in parents and self._chain_is_valid(uid):
                return uid
        return None

    def _chain_is_valid(self, leaf: str) -> bool:
        seen: set[str] = set()
        uid: str | None = leaf
        while uid is not None:
            if uid in seen or uid not in self.messages:
                return False
            seen.add(uid)
            uid = self.messages[uid].parent_uuid
        return True


class _Accumulator:
    """Folds transcript entry lines into the maps a ``LoadedTranscript`` needs.

    Shared by the whole-file and the boundary-truncated read paths so both interpret
    entries identically. ``relinks`` are applied last (last-wins) to re-point parents.
    """

    __slots__ = (
        "session_id",
        "messages",
        "order",
        "relinks",
        "title",
        "tag",
        "branch",
        "workspace",
        "diagnostics",
        "_workspace_conflict",
        "_owner_session_id",
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: dict[str, Message] = {}
        self.order: list[str] = []
        self.relinks: dict[str, str | None] = {}
        self.title: str | None = None
        self.tag: str | None = None
        self.branch: str | None = None
        self.workspace: Path | None = None
        self.diagnostics: list[TranscriptDiagnostic] = []
        self._workspace_conflict = False
        self._owner_session_id: str | None = None

    def _diagnose(
        self, code: str, line: int | None, entry_type: object = None, detail: str = ""
    ) -> None:
        self.diagnostics.append(
            TranscriptDiagnostic(
                code=code,
                line=line,
                entry_type=entry_type if isinstance(entry_type, str) else None,
                detail=detail[:160],
            )
        )

    def feed(self, line: str, line_number: int | None = None) -> None:
        line = line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            self._diagnose("invalid_json", line_number)
            return
        if not isinstance(entry, dict):
            self._diagnose("record_not_object", line_number)
            return
        etype = entry.get("type")
        if not isinstance(etype, str):
            self._diagnose("invalid_type", line_number)
            return
        version = entry.get("v")
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool) or version < 1
            or version > SCHEMA_VERSION
        ):
            self._diagnose("unsupported_version", line_number, etype)
            return
        if etype == _SESSION:
            self._accept_envelope(entry, line_number, etype)
        elif etype == "message":
            message = self._validated_message(entry, line_number, etype)
            if message is not None and self._accept_envelope(
                entry, line_number, etype
            ):
                self._feed_message(message, entry, line_number)
        elif etype == "tool_round":
            messages = entry.get("messages")
            manifest = entry.get("execution_manifest")
            checksum = entry.get("checksum")
            if not isinstance(messages, list) or not isinstance(manifest, dict):
                self._diagnose("invalid_tool_round", line_number, etype)
                return
            body = {"messages": messages, "execution_manifest": manifest}
            canonical = json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
            )
            if checksum != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
                self._diagnose("checksum_mismatch", line_number, etype)
                return
            validated = [
                self._validated_message(raw, line_number, etype)
                for raw in messages
            ]
            ids = [item.uuid for item in validated if item is not None]
            if (
                any(item is None for item in validated)
                or len(ids) != len(set(ids))
                or any(item_id in self.messages for item_id in ids)
            ):
                self._diagnose("invalid_tool_round_message", line_number, etype)
                return
            if not self._accept_envelope(entry, line_number, etype):
                return
            for message in validated:
                assert message is not None
                self._feed_message(message, entry, line_number)
        elif etype == _COMPACTION_SNAPSHOT:
            messages = entry.get("messages")
            source_head = entry.get("source_head")
            checksum = entry.get("checksum")
            if not isinstance(messages, list):
                self._diagnose("invalid_snapshot", line_number, etype)
                return
            snapshot_body: dict[str, object] = {
                "messages": messages,
                "source_head": source_head,
            }
            canonical = json.dumps(
                snapshot_body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if checksum != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
                self._diagnose("checksum_mismatch", line_number, etype)
                return
            validated = [
                self._validated_message(raw, line_number, etype)
                for raw in messages
            ]
            ids = [item.uuid for item in validated if item is not None]
            if any(item is None for item in validated) or len(ids) != len(set(ids)):
                self._diagnose("invalid_snapshot_message", line_number, etype)
                return
            if not self._accept_envelope(entry, line_number, etype):
                return
            self.messages.clear()
            self.order.clear()
            self.relinks.clear()
            for message in validated:
                assert message is not None
                self._feed_message(message, entry, line_number)
        elif etype == _RELINK:
            msg_uuid = entry.get("uuid")
            parent_uuid = entry.get("parent_uuid")
            if isinstance(msg_uuid, str) and (
                parent_uuid is None or isinstance(parent_uuid, str)
            ):
                if self._accept_envelope(entry, line_number, etype):
                    self.relinks[msg_uuid] = parent_uuid
            else:
                self._diagnose("invalid_relink", line_number, etype)
        elif etype == _TITLE:
            title = entry.get("title")
            if isinstance(title, str):
                if self._accept_envelope(entry, line_number, etype):
                    self.title = title
            else:
                self._diagnose("invalid_title", line_number, etype)
        elif etype == _TAG:
            tag = entry.get("tag")
            if isinstance(tag, str):
                if self._accept_envelope(entry, line_number, etype):
                    self.tag = tag
            else:
                self._diagnose("invalid_tag", line_number, etype)
        # Unknown entry types are ignored, keeping the format forward-compatible.

    def _validated_message(
        self, value: object, line: int | None, entry_type: str
    ) -> Message | None:
        if not isinstance(value, dict):
            self._diagnose("message_not_object", line, entry_type)
            return None
        role = value.get("role")
        content = value.get("content", "")
        metadata = value.get("metadata", {})
        identity = value.get("message_id") or value.get("uuid")
        parent = value.get("parent_id", value.get("parent_uuid"))
        message_version = value.get("version", 1)
        if role not in {"system", "user", "assistant", "tool"}:
            self._diagnose("invalid_message_role", line, entry_type)
            return None
        if not isinstance(content, str) or not isinstance(metadata, dict):
            self._diagnose("invalid_message_fields", line, entry_type)
            return None
        for optional in ("name", "origin_id", "round_id"):
            if value.get(optional) is not None and not isinstance(value.get(optional), str):
                self._diagnose("invalid_message_fields", line, entry_type)
                return None
        if identity is not None and (not isinstance(identity, str) or not identity):
            self._diagnose("invalid_message_identity", line, entry_type)
            return None
        if parent is not None and not isinstance(parent, str):
            self._diagnose("invalid_message_parent", line, entry_type)
            return None
        if not isinstance(message_version, int) or isinstance(message_version, bool) or message_version < 1:
            self._diagnose("invalid_message_version", line, entry_type)
            return None
        try:
            return Message.from_dict(value)
        except (KeyError, TypeError, ValueError):
            self._diagnose("invalid_message", line, entry_type)
            return None

    def _feed_message(self, msg: Message, envelope: dict, line: int | None) -> None:
        if msg.uuid in self.messages:
            self._diagnose("duplicate_uuid", line, envelope.get("type"))
            return
        self.order.append(msg.uuid)
        self.messages[msg.uuid] = msg

    def _accept_envelope(
        self, envelope: dict, line: int | None, entry_type: str
    ) -> bool:
        raw_session = envelope.get("session_id")
        if entry_type == _SESSION and (
            not isinstance(raw_session, str) or not raw_session
        ):
            self._diagnose("invalid_session", line, entry_type)
            return False
        if raw_session is not None:
            if not isinstance(raw_session, str) or not raw_session:
                self._diagnose("invalid_envelope_session", line, entry_type)
                return False
            if self._owner_session_id is None:
                self._owner_session_id = raw_session
                self.session_id = raw_session
            elif raw_session != self._owner_session_id:
                self._diagnose(
                    "session_conflict", line, entry_type, str(raw_session)
                )
                return False
        timestamp = envelope.get("ts")
        if timestamp is not None and (
            isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
        ):
            self._diagnose("invalid_timestamp", line, entry_type)
            return False
        branch = envelope.get("git_branch")
        if branch is not None and not isinstance(branch, str):
            self._diagnose("invalid_git_branch", line, entry_type)
            return False
        if not self._capture_workspace(envelope, line):
            return False
        if branch:
            self.branch = branch
        return True

    def _capture_workspace(self, envelope: dict, line: int | None = None) -> bool:
        raw = envelope.get("cwd")
        if raw is None:
            return True
        if self._workspace_conflict:
            return False
        if not isinstance(raw, str) or not raw:
            self._diagnose("invalid_workspace", line, envelope.get("type"))
            return False
        try:
            candidate = Path(raw).resolve()
        except (OSError, ValueError):
            self._diagnose("invalid_workspace", line, envelope.get("type"))
            return False
        if self.workspace is None:
            self.workspace = candidate
        elif os.path.normcase(str(self.workspace)) != os.path.normcase(str(candidate)):
            # A transcript must never change project identity mid-file.
            self.workspace = None
            self._workspace_conflict = True
            self._diagnose("workspace_conflict", line, envelope.get("type"))
            return False
        return True

    def finish(self, path: Path) -> "LoadedTranscript":
        # Apply relinks: a compaction boundary re-points the kept tail's head at the
        # summary so the parent-walk stops at the boundary (pre-boundary turns drop out).
        for msg_uuid, parent in self.relinks.items():
            msg = self.messages.get(msg_uuid)
            if msg is not None:
                msg.parent_uuid = parent
        for msg_uuid, msg in self.messages.items():
            if msg.parent_uuid is not None and msg.parent_uuid not in self.messages:
                self._diagnose("orphan_parent", None, "message", msg_uuid)
        cycle_members: set[str] = set()
        for start in self.order:
            positions: dict[str, int] = {}
            walk: list[str] = []
            uid: str | None = start
            while uid is not None and uid in self.messages:
                if uid in positions:
                    cycle_members.update(walk[positions[uid] :])
                    break
                positions[uid] = len(walk)
                walk.append(uid)
                uid = self.messages[uid].parent_uuid
        for msg_uuid in sorted(cycle_members):
            self._diagnose("parent_cycle", None, "message", msg_uuid)
        return LoadedTranscript(
            path=path,
            session_id=self.session_id,
            messages=self.messages,
            order=self.order,
            title=self.title,
            tag=self.tag,
            git_branch=self.branch,
            workspace=self.workspace,
            diagnostics=self.diagnostics,
        )


def load_transcript(path: str | Path, *, skip_precompact: bool = True) -> LoadedTranscript:
    """Parse a session file into messages + metadata. Malformed lines are skipped.

    For files larger than ``SKIP_PRECOMPACT_THRESHOLD`` (and ``skip_precompact``), only
    the bytes from the last compaction boundary onward are parsed — the resume chain lives
    entirely after that boundary — plus a cheap scan of the pre-boundary region to rescue
    session metadata (title/tag) that would otherwise be skipped. Smaller files (and the
    no-boundary case) are read whole.
    """
    path = Path(path)
    acc = _Accumulator(session_id=path.stem)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    boundary_offset = 0
    if skip_precompact and size > SKIP_PRECOMPACT_THRESHOLD:
        boundary_offset = _last_boundary_offset(path)

    if boundary_offset > 0:
        for line in _scan_pre_boundary_metadata(path, boundary_offset):
            acc.feed(line)
        with path.open("rb") as file:
            file.seek(boundary_offset)
            for line_number, raw in enumerate(file, start=1):
                acc.feed(raw.decode("utf-8", "ignore"), line_number)
    else:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                acc.feed(line, line_number)

    return acc.finish(path)


def _last_boundary_offset(path: Path) -> int:
    """Byte offset of the start of the last compact-boundary line, or 0 if none.

    Scans forward in binary, summing line byte-lengths, so the returned offset is exactly
    where a ``file.seek`` lands to read the boundary line and everything after it.
    """
    offset = 0
    last = 0
    found = False
    try:
        with path.open("rb") as file:
            for raw in file:
                is_valid_snapshot = False
                if _SNAPSHOT_MARKER in raw:
                    try:
                        entry = json.loads(raw)
                        if not isinstance(entry, dict):
                            raise ValueError("snapshot must be a JSON object")
                        body = {
                            "messages": entry.get("messages"),
                            "source_head": entry.get("source_head"),
                        }
                        canonical = json.dumps(
                            body,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        is_valid_snapshot = entry.get("checksum") == hashlib.sha256(
                            canonical.encode("utf-8")
                        ).hexdigest()
                    except (UnicodeDecodeError, ValueError):
                        is_valid_snapshot = False
                if is_valid_snapshot or (
                    _SNAPSHOT_MARKER not in raw and _BOUNDARY_MARKER in raw
                ):
                    last = offset
                    found = True
                offset += len(raw)
    except OSError:
        return 0
    return last if found else 0


def _scan_pre_boundary_metadata(path: Path, end_offset: int) -> list[str]:
    """Rescue session-metadata lines (title/tag) from ``[0, end_offset)``.

    Truncating at the boundary would otherwise drop these, since they may have been
    written before the last fold. Cheap substring filter — only candidate lines are kept
    for the accumulator to JSON-parse.
    """
    markers = (b'"type": "%s"' % _TITLE.encode(), b'"type": "%s"' % _TAG.encode())
    out: list[str] = []
    consumed = 0
    try:
        with path.open("rb") as file:
            for raw in file:
                if consumed >= end_offset:
                    break
                if any(m in raw for m in markers):
                    out.append(raw.decode("utf-8", "ignore"))
                consumed += len(raw)
    except OSError:
        return out
    return out


def build_chain(loaded: LoadedTranscript, leaf: str | None = None) -> list[Message]:
    """Reconstruct the linear conversation ending at ``leaf`` (newest leaf if None).

    Follows ``parent_uuid`` backward and reverses to chronological order; the walk stops
    at the root (``parent_uuid is None``). ``seen`` guards against a malformed file with a
    parent cycle.
    """
    if leaf is None:
        leaf = loaded.latest_leaf()
    chain: list[Message] = []
    seen: set[str] = set()
    uid: str | None = leaf
    while uid is not None and uid in loaded.messages and uid not in seen:
        seen.add(uid)
        msg = loaded.messages[uid]
        chain.append(msg)
        uid = msg.parent_uuid
    chain.reverse()
    return chain


# --------------------------------------------------------------------------- listing


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    path: Path
    modified: float
    first_prompt: str
    message_count: int
    title: str | None = None
    tag: str | None = None
    git_branch: str | None = None
    workspace: Path | None = None


def read_lite(path: str | Path) -> SessionInfo | None:
    """Cheap metadata read for listing — never JSON-parses the whole file.

    One binary pass: the *original* first user prompt (near the head, so it survives
    boundary truncation), a message-line count, and the newest title/tag. Avoids the full
    ``load_transcript`` (and its tree reconstruction) when all we need is a list row.
    Returns ``None`` for an empty/unreadable file.
    """
    path = Path(path)
    first_prompt = ""
    count = 0
    title: str | None = None
    tag: str | None = None
    branch: str | None = None
    workspace: Path | None = None
    session_id = path.stem
    msg_marker = b'"type": "message"'
    title_marker = b'"type": "%s"' % _TITLE.encode()
    tag_marker = b'"type": "%s"' % _TAG.encode()
    try:
        with path.open("rb") as file:
            for raw in file:
                if b'"type": "session"' in raw:
                    try:
                        header = json.loads(raw)
                        if isinstance(header.get("cwd"), str):
                            workspace = Path(header["cwd"]).resolve()
                        session_id = header.get("session_id", session_id)
                    except (json.JSONDecodeError, OSError):
                        pass
                if msg_marker in raw:
                    count += 1
                    if not first_prompt:
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        session_id = entry.get("session_id", session_id)
                        branch = entry.get("git_branch") or branch
                        if workspace is None and isinstance(entry.get("cwd"), str):
                            workspace = Path(entry["cwd"]).resolve()
                        if entry.get("role") == "user" and str(entry.get("content", "")).strip():
                            first_prompt = " ".join(str(entry["content"]).split())[:200]
                elif title_marker in raw:
                    try:
                        title = json.loads(raw).get("title", title)
                    except json.JSONDecodeError:
                        pass
                elif tag_marker in raw:
                    try:
                        tag = json.loads(raw).get("tag", tag)
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return None
    if count == 0:
        return None
    return SessionInfo(
        session_id=session_id,
        path=path,
        modified=path.stat().st_mtime,
        first_prompt=first_prompt,
        message_count=count,
        title=title,
        tag=tag,
        git_branch=branch,
        workspace=workspace,
    )


def list_sessions(proj_dir: str | Path) -> list[SessionInfo]:
    """List sessions in one project dir, newest first. Sidechains are excluded.

    Sub-agent transcripts live under ``{session_id}/subagents/`` (a subdirectory), so a
    top-level ``*.jsonl`` glob naturally skips them.
    """
    proj = Path(proj_dir)
    if not proj.is_dir():
        return []
    infos: list[SessionInfo] = []
    for file in proj.glob("*.jsonl"):
        info = read_lite(file)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda i: i.modified, reverse=True)
    return infos


def latest_session(proj_dir: str | Path) -> SessionInfo | None:
    """Most recently modified session in the project dir (powers ``--continue``)."""
    sessions = list_sessions(proj_dir)
    return sessions[0] if sessions else None


def session_label(info: SessionInfo) -> str:
    """Human-readable one-liner for a session: custom title, else first prompt, else placeholder.

    The single source of truth for how a session is named in the UI (``polaris sessions``,
    the ``/resume`` list, and the slash-command completer) so a raw UUID is never shown.
    """
    return info.title or info.first_prompt or "(empty)"


@dataclass(frozen=True, slots=True)
class SessionLocation:
    path: Path
    workspace: Path
    # Non-fatal findings about how the session was located (e.g. it was only found
    # under a legacy, pre-hash project directory name). Empty on the normal path.
    diagnostics: tuple[TranscriptDiagnostic, ...] = ()


class AmbiguousSessionError(RuntimeError):
    pass


def locate_session(root: str | Path, cwd: str | Path, session_id: str) -> SessionLocation | None:
    """Locate a session and validate the workspace identity persisted inside it."""

    current = Path(cwd).resolve()
    base = Path(root).expanduser()
    legacy_dir = base / _legacy_sanitize_project(current)
    current_dirs = {
        project_dir(base, current),
        legacy_dir,
    }
    candidates = [folder / f"{session_id}.jsonl" for folder in current_dirs]
    if base.is_dir():
        candidates.extend(
            folder / f"{session_id}.jsonl"
            for folder in base.iterdir()
            if folder.is_dir()
        )
    matches: dict[Path, SessionLocation] = {}
    for candidate in candidates:
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            continue
        if not candidate.is_file() or resolved_candidate in matches:
            continue
        loaded = load_transcript(candidate, skip_precompact=False)
        workspace = loaded.workspace
        if workspace is None:
            if candidate.parent not in current_dirs:
                continue
            workspace = current
        matches[resolved_candidate] = SessionLocation(candidate, workspace)
    if len(matches) > 1:
        paths = ", ".join(str(item.path) for item in matches.values())
        raise AmbiguousSessionError(f"session id {session_id!r} is ambiguous: {paths}")
    location = next(iter(matches.values()), None)
    if location is None:
        return None
    # A hit under the legacy (pre-hash) project dir still resolves, but it is worth
    # surfacing: the legacy naming collides across distinct cwds (P1-N3), so the
    # operator should migrate it with ``migrate_legacy_project_dir``. Lookup and
    # rejection semantics are unchanged — this only annotates the result.
    if legacy_dir != project_dir(base, current) and os.path.normcase(
        str(location.path.resolve())
    ) == os.path.normcase(str((legacy_dir / f"{session_id}.jsonl").resolve())):
        logging.getLogger(__name__).warning(
            "session %s resolved via the legacy project directory %s; migrate it with "
            "agent_core.transcript.migrate_legacy_project_dir to the hashed project name",
            session_id,
            legacy_dir,
        )
        location = SessionLocation(
            location.path,
            location.workspace,
            (
                TranscriptDiagnostic(
                    code="legacy_project_dir",
                    detail=(
                        f"session found only in the legacy project directory {legacy_dir}; "
                        "migrate it with migrate_legacy_project_dir"
                    ),
                ),
            ),
        )
    return location


def find_session(root: str | Path, cwd: str | Path, session_id: str) -> Path | None:
    """Compatibility wrapper returning only the verified transcript path."""

    location = locate_session(root, cwd, session_id)
    return location.path if location is not None else None


def migrate_legacy_project_dir(root: str | Path, cwd: str | Path) -> Path | None:
    """Move a legacy-named project directory to the modern hashed name.

    Explicit, caller-driven migration for session stores written before project
    directories gained their collision-proof sha256 suffix (P1-N3): the
    ``_legacy_sanitize_project(cwd)`` directory under ``root`` is renamed to the
    ``sanitize_project(cwd)`` directory in a single filesystem rename. Returns the
    new directory path, or ``None`` when no legacy directory exists (nothing to
    migrate). Raises ``FileExistsError`` when the modern directory already exists —
    migration never overwrites or merges. Never invoked automatically on any
    startup or load path.
    """

    base = Path(root).expanduser()
    current = Path(cwd).resolve()
    source = base / _legacy_sanitize_project(current)
    target = project_dir(base, current)
    if source == target or not source.is_dir():
        return None
    if target.exists():
        raise FileExistsError(
            f"refusing to migrate {source} over the existing project directory {target}"
        )
    source.rename(target)
    return target


def fork_chain(loaded: LoadedTranscript, leaf: str | None = None) -> tuple[str, list[Message]]:
    """Produce a new session id and a fresh copy of the chain for ``--fork-session``.

    The copied messages keep their tree shape (new uuids, re-linked parents) so the fork
    is an independent branch; the source file is never touched. Returns the new session
    id and the cloned chain (oldest-first) ready to seed a new ``TranscriptStore``.
    """
    chain = build_chain(loaded, leaf)
    remap: dict[str, str] = {}
    cloned: list[Message] = []
    for msg in chain:
        new_uuid = uuid.uuid4().hex
        remap[msg.uuid] = new_uuid
        parent = remap.get(msg.parent_uuid) if msg.parent_uuid else None
        cloned.append(
            Message(
                role=msg.role,
                content=msg.content,
                name=msg.name,
                metadata=dict(msg.metadata),
                uuid=new_uuid,
                parent_uuid=parent,
            )
        )
    return new_session_id(), cloned
