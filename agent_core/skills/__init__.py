"""Skill / slash-command subsystem.

A *skill* is a named, reusable prompt (Markdown body + frontmatter) loaded from disk.
Skills can be triggered two ways: a human types ``/name args`` in chat, or the model
calls the ``skill`` tool. Both resolve against a :class:`SkillRegistry` the agent
builds at startup from bundled, user (``~/.polaris/skills``), and project
(``./.polaris/skills``) directories.

Everything here is pure-stdlib so importing ``agent_core`` stays dependency-free.
"""

from __future__ import annotations

from agent_core.skills.config import SkillsConfig
from agent_core.skills.dispatch import (
    ParsedCommand,
    build_skill_prompt,
    fork_preset,
    looks_like_command,
    parse_slash_command,
    render_skill_prompt,
)
from agent_core.skills.frontmatter import parse_frontmatter
from agent_core.skills.loader import (
    BUNDLED_DIR,
    discover_skill_dirs,
    load_skill_file,
    load_skills,
)
from agent_core.skills.models import Skill, SkillContext
from agent_core.skills.programmatic import (
    SkillPromptContext,
    builtin_programmatic_skills,
    programmatic_skill,
)
from agent_core.skills.registry import SkillRegistry

__all__ = [
    "BUNDLED_DIR",
    "ParsedCommand",
    "Skill",
    "SkillContext",
    "SkillPromptContext",
    "SkillRegistry",
    "SkillsConfig",
    "build_skill_prompt",
    "builtin_programmatic_skills",
    "discover_skill_dirs",
    "fork_preset",
    "load_skill_file",
    "load_skills",
    "looks_like_command",
    "parse_frontmatter",
    "parse_slash_command",
    "programmatic_skill",
    "render_skill_prompt",
]
