"""One-time project context discovered at run start (CLAUDE.md project instructions).

Pure standard library: ``import agent_core`` must not pull in heavy deps, and this
module is on that path. All disk IO runs in ``_xxx_sync`` helpers offloaded via
``asyncio.to_thread`` so the public entry point can stay ``async`` without blocking
the event loop (CLAUDE.md async-only invariant).

The module is deliberately env- and config-free: whether to inject at all is decided
one layer up (``ReActConfig.project_instructions``, resolved from toml + env by
``config.resolve_context_config``). Here we only discover, read, and join — and we
never raise: any failure degrades to ``None`` so a missing/unreadable file can never
sink a run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

# Mirrors Claude Code's MEMORY_INSTRUCTION_PROMPT so the model treats the block the
# same way it does in the reference runtime.
CLAUDE_MD_PREAMBLE = (
    "Codebase and user instructions are shown below. Be sure to adhere to these "
    "instructions. IMPORTANT: These instructions OVERRIDE any default behavior and "
    "you MUST follow them exactly as written."
)
DEFAULT_MAX_CHARS = 32000

# Marker appended when the joined text is truncated to ``max_chars``.
_TRUNCATION_SUFFIX = "\n...(truncated)"


async def build_project_instructions(
    workspace: Path,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_user_home: bool = True,
) -> str | None:
    """Discover and join CLAUDE.md files into one injectable block.

    Walks ``workspace`` up to the filesystem root collecting a ``CLAUDE.md`` per
    directory (root → workspace order, so the closest file wins), optionally
    prepending the user-global ``~/.claude/CLAUDE.md``. Returns ``None`` when there is
    nothing to inject (no files, all empty, or all unreadable). Never raises.
    """
    try:
        paths = await asyncio.to_thread(_discover_claude_md_sync, workspace, include_user_home)
        if not paths:
            return None
        return await asyncio.to_thread(_read_and_join_sync, paths, max_chars)
    except Exception:  # noqa: BLE001 - context injection must never fail a run
        return None


def _discover_claude_md_sync(workspace: Path, include_user_home: bool) -> list[Path]:
    """Return deduped CLAUDE.md paths in priority order (lowest → highest).

    Order: ``~/.claude/CLAUDE.md`` (if present and requested) first, then each
    directory from the filesystem root down to ``workspace``. Files closer to the
    workspace come later so the model weights them higher. Dedup is by resolved path.
    """
    candidates: list[Path] = []

    if include_user_home:
        user_file = Path.home() / ".claude" / "CLAUDE.md"
        candidates.append(user_file)

    workspace = workspace.resolve()
    # workspace itself + every parent, then reverse to get root → workspace.
    dirs = [workspace, *workspace.parents]
    for directory in reversed(dirs):
        candidates.append(directory / "CLAUDE.md")

    discovered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            key = candidate.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        discovered.append(candidate)
    return discovered


def _read_and_join_sync(paths: list[Path], max_chars: int) -> str | None:
    """Read each file, label it with its source path, and join under the preamble.

    A single unreadable file (permission/encoding/disappeared) is skipped; the rest
    still load. The joined text is truncated as a whole to ``max_chars``. Returns
    ``None`` when nothing readable/non-empty remains.
    """
    sections: list[str] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not content:
            continue
        sections.append(
            f"Contents of {path} (project instructions, checked into the codebase):\n\n{content}"
        )

    if not sections:
        return None

    text = f"{CLAUDE_MD_PREAMBLE}\n\n" + "\n\n".join(sections)
    if len(text) > max_chars:
        keep = max(0, max_chars - len(_TRUNCATION_SUFFIX))
        text = text[:keep] + _TRUNCATION_SUFFIX
    return text
