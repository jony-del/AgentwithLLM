"""Self-registering catalog of built-in tools.

Each built-in tool class is tagged with ``@builtin_tool``, which appends it to a
module-level list at import time. ``discover()`` imports every module in this
package so all those decorators have fired, and ``default_tools()`` then builds
the instances. The payoff: adding a new tool is just **dropping a decorated class
into a file in this package** — `react.py` and this module never need editing.

Discovery runs lazily (at ``default_tools()`` time, not import time), so there is
no import-time cycle: tool modules import this module for the decorator, but this
module only imports them later, on demand.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.tools.base import Tool, WorkspacePathMixin

# Populated by the @builtin_tool decorator as tool modules are imported.
_BUILTIN: list[type[Tool]] = []
_discovered = False


def builtin_tool(cls: type[Tool]) -> type[Tool]:
    """Class decorator that registers a ``Tool`` subclass as a default built-in."""
    _BUILTIN.append(cls)
    return cls


def discover() -> None:
    """Import every submodule of this package so all ``@builtin_tool`` decorators run.

    Idempotent and import-once: the guard flips before the loop so a module that
    happens to re-enter discovery during its own import can't recurse.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    package = __name__.rsplit(".", 1)[0]  # "agent_core.tools"
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        importlib.import_module(f"{package}.{info.name}")


def builtin_tool_classes() -> list[type[Tool]]:
    """All registered built-in tool classes (triggers discovery on first call)."""
    discover()
    return list(_BUILTIN)


def default_tools(
    workspace: str | Path | None = None,
    session: SessionContext | None = None,
) -> list[Tool]:
    """Instantiate the default tool set.

    Session-aware tools (``SessionAwareMixin`` subclasses) receive ``session`` (or a
    placeholder if none is given — ``ReActAgent`` rebinds them later). Workspace-scoped
    tools (``WorkspacePathMixin`` subclasses) receive ``workspace`` when one is given;
    workspace-agnostic tools (e.g. ``echo``) are built with no args.
    """
    tools: list[Tool] = []
    for cls in builtin_tool_classes():
        if issubclass(cls, SessionAwareMixin):
            tools.append(cls(session) if session is not None else cls())  # type: ignore[call-arg]
        elif workspace is not None and issubclass(cls, WorkspacePathMixin):
            tools.append(cls(workspace))  # type: ignore[call-arg]
        else:
            tools.append(cls())
    return tools
