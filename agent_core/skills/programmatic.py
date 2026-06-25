"""Programmatic (Python-defined) skills, mirroring the reference's ``getPromptForCommand``.

A markdown skill renders a static body; a *programmatic* skill computes its prompt at
invocation time from the user args plus a read-only :class:`SkillPromptContext` (so it
can read the transcript, recent run log, workspace, etc.). These register themselves with
``@programmatic_skill`` exactly the way built-in tools use ``@builtin_tool`` — discovery
imports the modules so the decorators fire, then ``builtin_programmatic_skills()`` builds
the :class:`Skill` instances. Adding one is just dropping a decorated factory into a
module in this package; no wiring elsewhere.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.skills.models import Skill

# A factory returns a fully-built Skill (with its ``prompt_fn`` set). Factories — not
# Skill instances — are registered so a skill is rebuilt fresh each time the registry is
# constructed (no shared mutable state across agents).
SkillFactory = Callable[[], Skill]

_PROGRAMMATIC: list[SkillFactory] = []
_discovered = False


@dataclass(slots=True)
class SkillPromptContext:
    """Read-only context handed to a programmatic skill's ``prompt_fn``.

    ``session`` is the live :class:`~agent_core.session.SessionContext`; ``transcript``
    is the resumable transcript store (or ``None``); ``run_dir``/``run_id`` locate the
    current run's JSONL event log. Typed ``Any`` to avoid import cycles into the agent.
    """

    workspace: Path
    session: Any = None
    transcript: Any = None
    run_dir: str = "runs"
    run_id: str = ""

    @classmethod
    def from_session(cls, session: Any, *, transcript: Any = None) -> "SkillPromptContext":
        """Build a context from a live ``SessionContext`` (uniform across call paths)."""
        return cls(
            workspace=getattr(session, "workspace", Path.cwd()),
            session=session,
            transcript=transcript,
            run_dir=getattr(session, "run_dir", "runs"),
            run_id=getattr(session, "run_id", ""),
        )


def programmatic_skill(factory: SkillFactory) -> SkillFactory:
    """Register a ``() -> Skill`` factory as a built-in programmatic skill."""
    _PROGRAMMATIC.append(factory)
    return factory


def discover() -> None:
    """Import every submodule of this package so all ``@programmatic_skill`` decorators run.

    Idempotent and import-once (the guard flips before the loop so a module re-entering
    discovery during its own import can't recurse), mirroring ``tools.catalog.discover``.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    package = __name__.rsplit(".", 1)[0]  # "agent_core.skills"
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        # Only the builtin-skill modules define factories; importing the rest is harmless
        # but we scope to the known module to avoid importing unrelated siblings early.
        if info.name == "builtin_programmatic":
            importlib.import_module(f"{package}.{info.name}")


def builtin_programmatic_skills() -> list[Skill]:
    """Build all registered programmatic skills (triggers discovery on first call)."""
    discover()
    return [factory() for factory in _PROGRAMMATIC]
