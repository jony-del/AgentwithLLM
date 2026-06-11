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
from pathlib import Path
from typing import Any

# Allowed to-do states, mirroring Claude Code's TodoWrite.
VALID_TODO_STATUS = frozenset({"pending", "in_progress", "completed"})
_TODO_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


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
    todos: TodoStore = field(default_factory=TodoStore)
    # Async factories: children are awaited on the shared event loop so several
    # children's API calls overlap (bounded by the shared provider gate).
    subagent_factory: Callable[[str, str], Awaitable[str]] | None = None
    teammate_factory: Callable[[str, str, str, str | None, str], Awaitable[str]] | None = None
    team_store: Any | None = None
    agent_name: str = "leader"
    team_id: str | None = None
    # Run id of the agent that spawned this one, for reconstructing concurrent
    # fan-out from the per-run JSONL logs. ``None`` for the top-level agent.
    parent_run_id: str | None = None
    ui_notify: Callable[[list[Todo]], None] | None = None
    depth: int = 0
    max_depth: int = 1

    def notify_todos(self) -> None:
        if self.ui_notify is not None:
            self.ui_notify(self.todos.items())


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
