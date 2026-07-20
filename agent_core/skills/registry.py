"""In-memory lookup of loaded skills by name and alias.

Mirrors the role of ``tools/registry.py`` for the skill subsystem: the agent builds
one of these at startup from the loaded skill files, binds it onto the session, and
both the chat ``/command`` dispatcher and the model-facing ``skill`` tool read it.
"""

from __future__ import annotations

import builtins

from agent_core.skills.models import Skill


class SkillRegistry:
    """Name/alias -> :class:`Skill`. Later additions win on a name collision."""

    def __init__(self, skills: "list[Skill] | tuple[Skill, ...]" = ()) -> None:
        self._by_name: dict[str, Skill] = {}
        self._aliases: dict[str, str] = {}
        for skill in skills:
            self.add(skill)

    def add(self, skill: Skill) -> None:
        self._by_name[skill.name] = skill
        for alias in skill.aliases:
            key = alias.strip().lower()
            if key:
                self._aliases[key] = skill.name

    def get(self, name: str) -> Skill | None:
        """Look up by exact name, then case-insensitive name, then alias."""
        if name in self._by_name:
            return self._by_name[name]
        lowered = name.strip().lower()
        for skill_name, skill in self._by_name.items():
            if skill_name.lower() == lowered:
                return skill
        target = self._aliases.get(lowered)
        return self._by_name.get(target) if target else None

    def list(self) -> builtins.list[Skill]:
        return builtins.list(self._by_name.values())

    def user_invocable(self) -> builtins.list[Skill]:
        return [skill for skill in self._by_name.values() if skill.user_invocable]

    def model_invocable(self) -> builtins.list[Skill]:
        return [skill for skill in self._by_name.values() if skill.model_invocable]

    def __len__(self) -> int:
        return len(self._by_name)

    def __bool__(self) -> bool:
        return bool(self._by_name)
