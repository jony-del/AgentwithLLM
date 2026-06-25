"""Disk discovery and loading of skill files.

Bundled skills (shipped in ``agent_core/skills/bundled``) and user/project skills go
through the **same** loader — a bundled skill is just a Markdown file we ship. A skill
lives either as ``<dir>/<name>/SKILL.md`` (directory form, name defaults to the folder)
or as a loose ``<dir>/<name>.md`` (file form, name defaults to the stem).

Precedence is low -> high: bundled -> user -> project -> extra ``skills_dirs``. A later
directory overrides an earlier one on a name collision, and the same file reached via
two paths (symlinks) is loaded once. Every file is best-effort: a malformed skill is
skipped, never fatal.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.skills.config import SkillsConfig
from agent_core.skills.frontmatter import parse_frontmatter
from agent_core.skills.models import Skill, SkillContext

BUNDLED_DIR = Path(__file__).resolve().parent / "bundled"


def _coerce_context(value: object) -> SkillContext:
    return SkillContext.FORK if str(value or "").strip().lower() == "fork" else SkillContext.INLINE


def _coerce_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        # Allow a comma-separated scalar as well as a real list.
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def load_skill_file(path: Path) -> Skill | None:
    """Parse one skill file into a :class:`Skill`, or ``None`` if unusable.

    The default name is the folder name for ``SKILL.md`` and the stem otherwise; a
    ``name:`` in frontmatter overrides it. A skill with neither a name nor any body is
    dropped.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    meta, body = parse_frontmatter(text)

    default_name = path.parent.name if path.name.lower() == "skill.md" else path.stem
    name = str(meta.get("name") or default_name).strip()
    body = body.strip()
    if not name or not body:
        return None

    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        body=body,
        when_to_use=str(meta.get("when_to_use") or "").strip(),
        argument_hint=str(meta.get("argument_hint") or "").strip(),
        allowed_tools=_coerce_tuple(meta.get("allowed_tools")),
        model=(str(meta.get("model")).strip() or None) if meta.get("model") else None,
        aliases=_coerce_tuple(meta.get("aliases")),
        user_invocable=meta.get("user_invocable", True) is not False,
        disable_model_invocation=meta.get("disable_model_invocation", False) is True,
        context=_coerce_context(meta.get("context")),
        source_path=path,
    )


def _iter_skill_files(directory: Path):
    """Yield candidate skill files under ``directory`` (directory form, then loose)."""
    if not directory.is_dir():
        return
    for sub in sorted(directory.iterdir()):
        if sub.is_dir():
            candidate = sub / "SKILL.md"
            if candidate.is_file():
                yield candidate
    for loose in sorted(directory.glob("*.md")):
        if loose.is_file():
            yield loose


def discover_skill_dirs(workspace: Path, config: SkillsConfig) -> list[Path]:
    """Ordered (low -> high precedence) skill directories to load from.

    Bundled first, then the user dir, the project dir (resolved against ``workspace``),
    and finally any explicit ``skills_dirs`` (highest). Non-existent dirs are kept in
    the list and simply yield nothing when iterated.
    """
    dirs: list[Path] = [BUNDLED_DIR]
    if config.user_dir:
        dirs.append(Path(config.user_dir).expanduser())
    if config.project_dir:
        project = Path(config.project_dir).expanduser()
        dirs.append(project if project.is_absolute() else (workspace / project))
    for extra in config.skills_dirs:
        dirs.append(Path(extra).expanduser())
    return dirs


def load_skills(dirs: list[Path], disabled: "tuple[str, ...] | list[str]" = ()) -> list[Skill]:
    """Load skills from ``dirs`` (low -> high precedence), deduped by real path.

    A later directory's skill of the same name replaces an earlier one. The same file
    reached twice (e.g. a symlinked dir) loads once. Names in ``disabled`` are dropped
    after loading so a user can suppress a bundled skill by name.
    """
    by_name: dict[str, Skill] = {}
    seen_paths: set[Path] = set()
    for directory in dirs:
        for path in _iter_skill_files(directory):
            try:
                real = path.resolve()
            except OSError:
                real = path
            if real in seen_paths:
                continue
            seen_paths.add(real)
            skill = load_skill_file(path)
            if skill is not None:
                by_name[skill.name] = skill
    blocked = {name.strip().lower() for name in disabled}
    return [skill for skill in by_name.values() if skill.name.lower() not in blocked]
