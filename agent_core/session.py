"""Per-run session state shared across tools within a single ``ReActAgent.run()``.

Most tools need nothing beyond a workspace path (see ``WorkspacePathMixin``). A few
need *more*: the task-planning tool needs a mutable to-do list that persists across
tool turns, and the sub-agent (``dispatch_agent``) tool needs a way to spawn a fresh
``ReActAgent``. Threading those through the existing "just a workspace" seam would be
awkward, so they hang off a single ``SessionContext`` the agent owns and binds into
any session-aware tool.

This module deliberately imports nothing from ``agent_core.react`` or
``agent_core.tools`` — the sub-agent factory is injected by ``ReActAgent`` as a plain
callable, so there is no import cycle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

# Cap on the recently-read file snapshots kept on a session (Phase 3E). Only the most
# recent handful are ever re-injected after compaction, so a small bound is plenty and
# keeps a many-file run from growing the dict unboundedly.
_MAX_READ_FILE_STATE = 20

# Allowed to-do states, mirroring Claude Code's TodoWrite.
VALID_TODO_STATUS = frozenset({"pending", "in_progress", "completed"})
_TODO_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


@dataclass(slots=True)
class PlanState:
    active: bool = False
    previous_mode: str | None = None
    artifact_path: Path | None = None

    def enter(self, previous_mode: str, artifact_path: Path) -> None:
        if not self.active:
            self.previous_mode = previous_mode
        self.active = True
        self.artifact_path = artifact_path

    def clear(self) -> None:
        self.active = False
        self.previous_mode = None
        self.artifact_path = None


class PlanArtifactStore:
    """Agent-owned plan storage; callers never choose an arbitrary output path."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            try:
                base = Path.home()
            except RuntimeError:  # embedded/restricted environments may have no HOME
                base = Path.cwd()
            self.root = base / ".polaris" / "plans"
        else:
            self.root = Path(root).expanduser()

    def path_for(self, session_id: str, agent_id: str) -> Path:
        safe_session = _safe_plan_component(session_id or "session")
        safe_agent = _safe_plan_component(agent_id or "leader")
        return self.root / f"{safe_session}-{safe_agent}.md"

    def write(self, path: Path, content: str) -> None:
        encoded = content.encode("utf-8")
        if len(encoded) > 256 * 1024:
            raise ValueError("plan artifact exceeds the 256 KiB limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(encoded)
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(path)

    def read(self, path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None


def _safe_plan_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return cleaned[:80] or "unknown"


@dataclass(slots=True)
class Todo:
    content: str
    status: str = "pending"


class TodoStore:
    """A mutable, per-run to-do list the model overwrites wholesale via ``update_todos``."""

    def __init__(self) -> None:
        self._items: list[Todo] = []

    def replace(self, raw: object) -> list[Todo]:
        """Replace the whole list from a sequence of ``{content, status}`` mappings.

        Empty-content entries are dropped and unknown statuses fall back to
        ``pending`` so a sloppy model call can't corrupt the store.
        """
        items: list[Todo] = []
        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                content = str(entry.get("content", "")).strip()
                if not content:
                    continue
                status = str(entry.get("status", "pending"))
                if status not in VALID_TODO_STATUS:
                    status = "pending"
                items.append(Todo(content, status))
        self._items = items
        return list(self._items)

    def items(self) -> list[Todo]:
        return list(self._items)

    def render(self) -> str:
        if not self._items:
            return "(no todos)"
        return "\n".join(f"{_TODO_MARK.get(t.status, '[ ]')} {t.content}" for t in self._items)


@dataclass(slots=True)
class SessionContext:
    """State shared across tools for the lifetime of one ``run()``.

    ``subagent_factory`` is injected by ``ReActAgent`` (a closure over the agent) so
    ``dispatch_agent`` can spawn a child without importing the agent class. ``ui_notify``
    lets a tool surface a change (e.g. updated todos) to the live UI without holding a
    UI reference. ``depth``/``max_depth`` bound recursive sub-agent spawning.
    """

    workspace: Path = field(default_factory=lambda: Path.cwd().resolve())
    # Identifier of the resumable session transcript this run writes to. Threaded through
    # to sub-agents so their sidechain transcripts nest under the parent session dir.
    session_id: str = ""
    agent_id: str = "leader"
    parent_agent_id: str | None = None
    todos: TodoStore = field(default_factory=TodoStore)
    # Async factories: children are awaited on the shared event loop so several
    # children's API calls overlap (bounded by the shared provider gate). The trailing
    # ``str | None`` is an optional per-spawn model override (None → inherit the parent's
    # model); each spawn call chooses independently, so one leader can fan out a mix of
    # Haiku/Sonnet/Opus children.
    subagent_factory: Callable[[str, str, str | None], Awaitable[str]] | None = None
    teammate_factory: Callable[[str, str, str, str | None, str, str | None], Awaitable[str]] | None = None
    team_store: Any | None = None
    agent_name: str = "leader"
    team_id: str | None = None
    # Run id of the agent that spawned this one, for reconstructing concurrent
    # fan-out from the per-run JSONL logs. ``None`` for the top-level agent.
    parent_run_id: str | None = None
    ui_notify: Callable[[list[Todo]], None] | None = None
    # Per-run skill registry (loaded at agent startup), read by the ``skill`` tool to
    # resolve a model-invoked skill by name. ``None``/empty when skills are disabled or
    # none were found; ``Any`` to avoid importing the skills package into this seam.
    skills: Any | None = None
    # Run-log location for the active run, so a programmatic skill (e.g. ``debug``) can
    # read this run's JSONL events. ``run_id`` is the JSONLRunLogger's id.
    run_dir: str = "runs"
    run_id: str = ""
    plan_state: PlanState = field(default_factory=PlanState)
    plan_store: PlanArtifactStore = field(default_factory=PlanArtifactStore)
    permission_mode_setter: Callable[..., object] | None = None
    depth: int = 0
    max_depth: int = 1
    # Recently-read file snapshots, keyed by workspace-resolved path string and
    # recorded by the react loop (NOT the read tool — see Phase 3E). Insertion-ordered
    # newest-last so post-compaction re-injection can take the most-recent few. Capped
    # so a long run that reads many files can't grow this unbounded.
    read_file_state: dict[str, str] = field(default_factory=dict)

    def notify_todos(self) -> None:
        if self.ui_notify is not None:
            self.ui_notify(self.todos.items())

    def record_read(self, path: str, content: str) -> None:
        """Record the latest snapshot of a read file, newest-last by recency.

        Re-reading a file moves it to the end (most recent). The dict is capped at
        ``_MAX_READ_FILE_STATE`` entries; the oldest entries are evicted first.
        """
        if path in self.read_file_state:
            # Drop then re-insert so the key lands at the end (most recent).
            del self.read_file_state[path]
        self.read_file_state[path] = content
        while len(self.read_file_state) > _MAX_READ_FILE_STATE:
            oldest = next(iter(self.read_file_state))
            del self.read_file_state[oldest]


class SessionAwareMixin:
    """Mixin for tools that need the per-run ``SessionContext`` instead of just a workspace.

    Built with an optional session (a placeholder when none is given so the tool is
    still constructible during discovery); ``ReActAgent`` then calls ``bind_session``
    to repoint it at the live session — this covers the CLI path where the registry is
    built before the agent exists.
    """

    needs_session = True

    def __init__(self, session: SessionContext | None = None) -> None:
        self.session: SessionContext = session or SessionContext()

    def bind_session(self, session: SessionContext) -> None:
        self.session = session
