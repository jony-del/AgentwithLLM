"""Built-in programmatic skills — ported from the reference's dynamic bundled skills.

Each is a ``() -> Skill`` factory registered with ``@programmatic_skill``; its prompt is
computed by an async ``prompt_fn(args, ctx)`` at invocation time. All three are inline
and human-only (``disable_model_invocation``), matching the reference. They degrade
gracefully when their context (run log, workspace) is missing — a prompt is always
returned, never an exception.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_core.skills.models import Skill, SkillContext
from agent_core.skills.programmatic import SkillPromptContext, programmatic_skill

# --- lorem-ipsum -------------------------------------------------------------

_LOREM_WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure"
).split()

_LOREM_MAX_TOKENS = 100_000
_LOREM_DEFAULT_TOKENS = 500


def _generate_lorem(target_tokens: int) -> str:
    """Deterministically emit roughly ``target_tokens`` words of filler (1 word ≈ 1 token)."""
    count = max(1, min(target_tokens, _LOREM_MAX_TOKENS))
    words = [_LOREM_WORDS[i % len(_LOREM_WORDS)] for i in range(count)]
    # Break into ~12-word sentences for readability.
    sentences = []
    for start in range(0, count, 12):
        chunk = words[start:start + 12]
        sentences.append(" ".join(chunk).capitalize() + ".")
    return " ".join(sentences)


async def _lorem_prompt(args: str, ctx: SkillPromptContext) -> str:
    target = _LOREM_DEFAULT_TOKENS
    token = args.strip().split()[0] if args.strip() else ""
    if token.isdigit():
        target = int(token)
    text = _generate_lorem(target)
    return f"(filler text for context/compaction testing — ~{target} tokens)\n\n{text}"


@programmatic_skill
def _lorem_ipsum_skill() -> Skill:
    return Skill(
        name="lorem-ipsum",
        description="Generate filler text of a given token count for long-context / compaction testing.",
        body="Generate filler text for context testing.",
        argument_hint="[token_count]",
        when_to_use="When you need to pad the context to test compaction or long-context behaviour.",
        context=SkillContext.INLINE,
        disable_model_invocation=True,
        prompt_fn=_lorem_prompt,
    )


# --- debug -------------------------------------------------------------------

_DEBUG_TAIL_LINES = 40


async def _debug_prompt(args: str, ctx: SkillPromptContext) -> str:
    path = Path(ctx.run_dir) / f"{ctx.run_id}.jsonl"
    suffix = f"\n\nIssue described by the user:\n{args}" if args.strip() else ""

    def _read_tail() -> str | None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        return "\n".join(lines[-_DEBUG_TAIL_LINES:])

    tail = await asyncio.to_thread(_read_tail) if ctx.run_id else None
    if not tail:
        return (
            "Help debug the current session. The run-event log could not be read "
            f"(looked for {path}). Describe what you observe and inspect the workspace as "
            f"needed to diagnose the problem.{suffix}"
        )
    return (
        "Diagnose this session using its recent run-event log below. Identify any errors, "
        "failed tools, or unexpected behaviour and explain the likely cause and a fix.\n\n"
        f"<run_log path=\"{path}\">\n{tail}\n</run_log>{suffix}"
    )


@programmatic_skill
def _debug_skill() -> Skill:
    return Skill(
        name="debug",
        description="Diagnose the current session by reading its recent run-event log.",
        body="Diagnose the current session.",
        argument_hint="[issue description]",
        when_to_use="When the session is misbehaving and you want it diagnosed from its run log.",
        context=SkillContext.INLINE,
        disable_model_invocation=True,
        prompt_fn=_debug_prompt,
    )


# --- skillify ----------------------------------------------------------------


async def _skillify_prompt(args: str, ctx: SkillPromptContext) -> str:
    skills_dir = ctx.workspace / ".polaris" / "skills"

    def _existing() -> list[str]:
        try:
            return sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
        except OSError:
            return []

    existing = await asyncio.to_thread(_existing)
    existing_line = (
        f"Existing project skills: {', '.join(existing)}.\n" if existing else ""
    )
    target = args.strip() or "the main repeatable process from this conversation"
    return (
        "Capture a reusable skill from THIS session. Review what was accomplished in this "
        f"conversation and distill {target} into a reusable procedure.\n\n"
        f"Write it as a new skill file at `{skills_dir / '<skill-name>' / 'SKILL.md'}` with "
        "YAML frontmatter (name, description, when-to-use, and context: inline or fork) "
        "followed by a Markdown body of clear, ordered instructions. Use `$ARGUMENTS` "
        "where the caller should pass details. Keep it general enough to reuse, not a "
        "transcript of this one run.\n"
        f"{existing_line}"
        "Confirm the skill name with the user if it's ambiguous before writing the file."
    )


@programmatic_skill
def _skillify_skill() -> Skill:
    return Skill(
        name="skillify",
        description="Capture this session's repeatable process into a reusable SKILL.md.",
        body="Capture this session into a reusable skill.",
        argument_hint="[description of the process to capture]",
        when_to_use="When the user wants to turn what was just done into a reusable skill.",
        context=SkillContext.INLINE,
        disable_model_invocation=True,
        prompt_fn=_skillify_prompt,
    )
