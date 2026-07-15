"""Core data shapes for the skill subsystem.

A :class:`Skill` is a named, reusable prompt (a Markdown body) plus frontmatter
metadata that controls how it is surfaced and executed. Skills come from disk
(``SKILL.md`` files in user/project/bundled directories) and are looked up by the
chat ``/command`` dispatcher and by the model-facing ``skill`` tool.

This module is pure-stdlib so ``import agent_core`` never pulls a heavy dep here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.skills.programmatic import SkillPromptContext

# A programmatic skill computes its prompt at call time from the args + a read-only
# context, mirroring the reference's ``getPromptForCommand``. ``Any`` avoids importing
# the context type at runtime (it lives in ``programmatic.py``, which imports this module).
PromptFn = Callable[[str, "SkillPromptContext"], Awaitable[str]]


class SkillContext(str, Enum):
    """How an invoked skill runs.

    ``INLINE`` injects the rendered skill prompt into the *current* context (a chat
    turn, or a tool observation the model then acts on). ``FORK`` runs it in an
    isolated sub-agent with its own clean context, returning only the final answer.
    """

    INLINE = "inline"
    FORK = "fork"


@dataclass(slots=True)
class Skill:
    """A loaded skill: frontmatter metadata + the Markdown prompt ``body``.

    ``user_invocable`` controls whether a human can trigger it via ``/name`` in chat;
    ``disable_model_invocation`` hides it from the model-facing ``skill`` tool. The two
    are independent so a skill can be human-only, model-only, both, or neither.
    """

    name: str
    description: str
    body: str
    when_to_use: str = ""
    argument_hint: str = ""
    allowed_tools: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    model: str | None = None
    aliases: tuple[str, ...] = ()
    user_invocable: bool = True
    disable_model_invocation: bool = False
    context: SkillContext = SkillContext.INLINE
    source_path: Path | None = field(default=None, compare=False)
    # When set, the prompt is computed by this async callable at invocation time instead
    # of rendering ``body`` (a "programmatic" skill). ``body`` then holds a static
    # fallback/description. Excluded from equality so two skills compare by their data.
    prompt_fn: "PromptFn | None" = field(default=None, compare=False)

    @property
    def model_invocable(self) -> bool:
        """True when the model may call this skill through the ``skill`` tool."""
        return not self.disable_model_invocation

    @property
    def is_programmatic(self) -> bool:
        return self.prompt_fn is not None
