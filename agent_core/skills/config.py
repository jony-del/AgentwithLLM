"""Configuration for the skill subsystem (``[skills]`` toml table).

Kept in the skills package (not ``config.py``) so ``ReActConfig`` can carry it as a
nested dataclass without a circular import, mirroring how ``MemoryConfig`` lives in
the memory package. ``config.resolve_skills_config`` builds one from toml + env.
"""

from __future__ import annotations

from dataclasses import dataclass


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a toml scalar/list into a tuple of non-empty strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


@dataclass(slots=True)
class SkillsConfig:
    """Skill discovery + loading knobs.

    ``enabled`` is an *enabled capability* (default on): when true, skills are
    discovered and loaded eagerly at agent startup. ``skills_dirs`` adds extra
    highest-precedence directories; ``disabled`` suppresses skills by name. The
    user/project directory defaults follow the project's ``.polaris`` convention.
    """

    enabled: bool = True
    user_dir: str = "~/.polaris/skills"
    project_dir: str = ".polaris/skills"
    skills_dirs: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict | None) -> "SkillsConfig":
        config = cls()
        if not data:
            return config
        if "enabled" in data:
            value = data["enabled"]
            config.enabled = (
                value.strip().lower() in {"1", "true", "yes", "on"}
                if isinstance(value, str)
                else bool(value)
            )
        if data.get("user_dir"):
            config.user_dir = str(data["user_dir"])
        if data.get("project_dir"):
            config.project_dir = str(data["project_dir"])
        config.skills_dirs = _as_str_tuple(data.get("skills_dirs"))
        config.disabled = _as_str_tuple(data.get("disabled"))
        return config
