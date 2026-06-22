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
The transcript is the *faithful* record of the conversation: compaction is an in-memory
optimization the running loop applies to what it sends the model, and never touches what
is written here — so a resume always reconstructs the true history and the live loop
re-compacts it as needed. Forking clones a chain under a fresh ``session_id`` with new,
re-linked uuids, leaving the source file untouched.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .models import Message

# Entry "type" values that are NOT conversation messages.
_TITLE = "custom-title"
_TAG = "tag"


def sanitize_project(cwd: str | Path) -> str:
    """Turn an absolute cwd into a flat, filesystem-safe directory name.

    Each non-alphanumeric character maps to ``-`` individually (matching the reference's
    scheme), so ``E:\\ZNGZ\\Code_copy`` becomes ``E--ZNGZ-Code-copy`` — distinct cwds
    never collide, and the same cwd always resolves to the same project dir.
    """
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
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = out.stdout.strip()
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

    async def append_message(self, message: Message) -> None:
        record = {
            "type": "message",
            **message.to_dict(),
            "session_id": self.session_id,
            "cwd": self.workspace,
            "git_branch": self._cwd_branch,
            "ts": time.time(),
        }
        await asyncio.to_thread(self._write_sync, record)

    async def append_meta(self, kind: str, payload: dict) -> None:
        record = {"type": kind, "session_id": self.session_id, "ts": time.time(), **payload}
        await asyncio.to_thread(self._write_sync, record)

    def _write_sync(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(line)
        except OSError as exc:
            # Persistence is best-effort: a transcript write must never crash an
            # otherwise healthy run. Warn once, then stay quiet.
            if not self._warned:
                print(f"[transcript] write failed ({exc}); resume disabled this run", file=sys.stderr)
                self._warned = True


# --------------------------------------------------------------------------- read side


@dataclass(slots=True)
class LoadedTranscript:
    path: Path
    session_id: str
    messages: dict[str, Message]          # uuid -> Message, in file order
    order: list[str] = field(default_factory=list)   # uuids in append order
    title: str | None = None
    tag: str | None = None
    git_branch: str | None = None

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
            if uid not in parents:
                return uid
        return self.order[-1]


def load_transcript(path: str | Path) -> LoadedTranscript:
    """Parse a session file into messages + metadata. Malformed lines are skipped."""
    path = Path(path)
    messages: dict[str, Message] = {}
    order: list[str] = []
    title: str | None = None
    tag: str | None = None
    branch: str | None = None
    session_id = path.stem

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = entry.get("type")
            if etype == "message":
                msg = Message.from_dict(entry)
                session_id = entry.get("session_id", session_id)
                branch = entry.get("git_branch") or branch
                if msg.uuid not in messages:
                    order.append(msg.uuid)
                messages[msg.uuid] = msg
            elif etype == _TITLE:
                title = entry.get("title", title)
            elif etype == _TAG:
                tag = entry.get("tag", tag)
            # Unknown entry types are ignored, keeping the format forward-compatible.

    return LoadedTranscript(
        path=path,
        session_id=session_id,
        messages=messages,
        order=order,
        title=title,
        tag=tag,
        git_branch=branch,
    )


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
        try:
            loaded = load_transcript(file)
        except OSError:
            continue
        if loaded.message_count == 0:
            continue
        infos.append(
            SessionInfo(
                session_id=loaded.session_id,
                path=file,
                modified=file.stat().st_mtime,
                first_prompt=loaded.first_prompt,
                message_count=loaded.message_count,
                title=loaded.title,
                tag=loaded.tag,
                git_branch=loaded.git_branch,
            )
        )
    infos.sort(key=lambda i: i.modified, reverse=True)
    return infos


def latest_session(proj_dir: str | Path) -> SessionInfo | None:
    """Most recently modified session in the project dir (powers ``--continue``)."""
    sessions = list_sessions(proj_dir)
    return sessions[0] if sessions else None


def find_session(root: str | Path, cwd: str | Path, session_id: str) -> Path | None:
    """Locate a session file by id: current project first, then across all projects.

    The cross-project scan is what lets ``--resume <id>`` work from a different cwd.
    """
    here = project_dir(root, cwd) / f"{session_id}.jsonl"
    if here.is_file():
        return here
    base = Path(root).expanduser()
    if not base.is_dir():
        return None
    for proj in base.iterdir():
        if not proj.is_dir():
            continue
        candidate = proj / f"{session_id}.jsonl"
        if candidate.is_file():
            return candidate
    return None


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
