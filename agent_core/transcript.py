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
_RELINK = "relink"

# Above this file size, the resume load skips everything before the last compaction
# boundary (reading only post-boundary bytes + a cheap pre-boundary metadata rescue),
# mirroring the reference's SKIP_PRECOMPACT_THRESHOLD. Small files are read whole.
SKIP_PRECOMPACT_THRESHOLD = 5 * 1024 * 1024

# Byte signature of a compact-boundary message line (we control the serialization, so the
# ``json.dumps`` default ``": "`` separator is stable). Used to locate the last boundary
# without JSON-parsing every line.
_BOUNDARY_MARKER = b'"compact_boundary": true'


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

    async def append_relink(self, uuid: str, parent_uuid: str | None) -> None:
        """Record that ``uuid``'s parent should be re-pointed to ``parent_uuid`` on load.

        Written at a compaction boundary to re-attach the kept tail's first message to the
        summary (the append-only file can't mutate the original line). ``load_transcript``
        applies these last-wins after parsing all messages.
        """
        await self.append_meta(_RELINK, {"uuid": uuid, "parent_uuid": parent_uuid})

    def _write_sync(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", errors="replace") as file:
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


class _Accumulator:
    """Folds transcript entry lines into the maps a ``LoadedTranscript`` needs.

    Shared by the whole-file and the boundary-truncated read paths so both interpret
    entries identically. ``relinks`` are applied last (last-wins) to re-point parents.
    """

    __slots__ = ("session_id", "messages", "order", "relinks", "title", "tag", "branch")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: dict[str, Message] = {}
        self.order: list[str] = []
        self.relinks: dict[str, str | None] = {}
        self.title: str | None = None
        self.tag: str | None = None
        self.branch: str | None = None

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return
        etype = entry.get("type")
        if etype == "message":
            msg = Message.from_dict(entry)
            self.session_id = entry.get("session_id", self.session_id)
            self.branch = entry.get("git_branch") or self.branch
            if msg.uuid not in self.messages:
                self.order.append(msg.uuid)
            self.messages[msg.uuid] = msg
        elif etype == _RELINK:
            self.relinks[entry["uuid"]] = entry.get("parent_uuid")
        elif etype == _TITLE:
            self.title = entry.get("title", self.title)
        elif etype == _TAG:
            self.tag = entry.get("tag", self.tag)
        # Unknown entry types are ignored, keeping the format forward-compatible.

    def finish(self, path: Path) -> "LoadedTranscript":
        # Apply relinks: a compaction boundary re-points the kept tail's head at the
        # summary so the parent-walk stops at the boundary (pre-boundary turns drop out).
        for uuid, parent in self.relinks.items():
            msg = self.messages.get(uuid)
            if msg is not None:
                msg.parent_uuid = parent
        return LoadedTranscript(
            path=path,
            session_id=self.session_id,
            messages=self.messages,
            order=self.order,
            title=self.title,
            tag=self.tag,
            git_branch=self.branch,
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
            for raw in file:
                acc.feed(raw.decode("utf-8", "ignore"))
    else:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                acc.feed(line)

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
                if _BOUNDARY_MARKER in raw:
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
    session_id = path.stem
    msg_marker = b'"type": "message"'
    title_marker = b'"type": "%s"' % _TITLE.encode()
    tag_marker = b'"type": "%s"' % _TAG.encode()
    try:
        with path.open("rb") as file:
            for raw in file:
                if msg_marker in raw:
                    count += 1
                    if not first_prompt:
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        session_id = entry.get("session_id", session_id)
                        branch = entry.get("git_branch") or branch
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
