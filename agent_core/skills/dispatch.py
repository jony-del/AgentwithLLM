"""Slash-command parsing and skill-prompt rendering.

Used by the chat loop to turn a typed ``/name args`` line into a skill invocation,
and by both the chat loop and the ``skill`` tool to render a skill's body with its
arguments substituted. Pure functions, no I/O.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from agent_core.skills.models import Skill

logger = logging.getLogger(__name__)

# A command name is a leading word of letters/digits/_/-/: (the ``:`` allows
# plugin-style ``ns:name``). Crucially it has no ``/`` or ``.``, so ``/path/to/file``
# and ``/foo.txt`` are NOT treated as commands.
_COMMAND_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:-]*$")

# Placeholder the body may use to position the user's argument string explicitly.
_ARGS_PLACEHOLDER = "$ARGUMENTS"

# Built-in tools whose risk is READ. A fork skill whose declared ``allowed_tools`` are
# all read-only runs its child with the read-only preset; otherwise (or when nothing is
# declared) it gets the full preset so the skill can actually do work.
_READ_ONLY_TOOLS = frozenset(
    {"list_dir", "search_text", "git_diff", "echo", "read_text_file", "glob", "web_fetch", "web_search"}
)


def fork_preset(allowed_tools: "tuple[str, ...] | list[str]") -> str:
    """Pick the sub-agent capability preset (``read_only``/``full``) for a fork skill."""
    if allowed_tools and all(tool in _READ_ONLY_TOOLS for tool in allowed_tools):
        return "read_only"
    return "full"


@dataclass(slots=True)
class ParsedCommand:
    name: str
    args: str


def parse_slash_command(text: str) -> ParsedCommand | None:
    """Parse ``/name rest...`` into its name and argument string.

    Returns ``None`` when ``text`` doesn't start with ``/`` or has no name. The name
    is everything up to the first whitespace; the rest (trimmed) is the arguments.
    """
    if not text.startswith("/"):
        return None
    rest = text[1:]
    parts = rest.split(None, 1)
    if not parts:
        return None
    name = parts[0]
    args = parts[1].strip() if len(parts) > 1 else ""
    return ParsedCommand(name=name, args=args)


def looks_like_command(name: str) -> bool:
    """True when ``name`` looks like a command token rather than a path/filename."""
    return bool(_COMMAND_NAME.match(name))


def render_skill_prompt(skill: Skill, args: str) -> str:
    """Render a (markdown) skill's body with ``args`` applied.

    If the body contains ``$ARGUMENTS`` it is substituted in place; otherwise the
    argument string (when non-empty) is appended under the body. The result is the
    prompt fed to the agent (inline) or sub-agent (fork).
    """
    body = skill.body
    if _ARGS_PLACEHOLDER in body:
        return body.replace(_ARGS_PLACEHOLDER, args)
    if args:
        return f"{body}\n\n{args}"
    return body


async def build_skill_prompt(skill: Skill, args: str, ctx) -> str:
    """Compute a skill's prompt: run its ``prompt_fn`` if programmatic, else render the body.

    ``ctx`` is a ``SkillPromptContext`` (only used by programmatic skills). A failing
    ``prompt_fn`` degrades to an explanatory prompt rather than raising, so a broken skill
    never sinks the run.
    """
    if skill.prompt_fn is None:
        return render_skill_prompt(skill, args)
    try:
        return await skill.prompt_fn(args, ctx)
    except Exception as exc:  # noqa: BLE001 - a skill's prompt builder must not crash the run
        logger.warning(
            "skill %s prompt build failed, degrading to raw render: %s: %s",
            skill.name, type(exc).__name__, exc,
        )
        return (
            f"The skill {skill.name!r} failed to build its prompt "
            f"({type(exc).__name__}: {exc}). {render_skill_prompt(skill, args)}"
        )
