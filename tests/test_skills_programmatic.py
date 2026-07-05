"""Tests for the programmatic (Python-defined) skill seam and the built-in dynamic skills."""

from __future__ import annotations

from pathlib import Path

from agent_core.skills import (
    Skill,
    SkillContext,
    SkillPromptContext,
    build_skill_prompt,
    builtin_programmatic_skills,
    programmatic_skill,
    render_skill_prompt,
)
from agent_core.skills.programmatic import _PROGRAMMATIC, discover


# --- the seam ----------------------------------------------------------------


async def test_build_skill_prompt_runs_prompt_fn() -> None:
    async def fn(args: str, ctx: SkillPromptContext) -> str:
        return f"computed::{args}::{ctx.run_id}"

    skill = Skill(name="dyn", description="d", body="static fallback", prompt_fn=fn)
    ctx = SkillPromptContext(workspace=Path.cwd(), run_id="abc")
    assert await build_skill_prompt(skill, "X", ctx) == "computed::X::abc"


async def test_build_skill_prompt_falls_back_to_body_for_markdown() -> None:
    skill = Skill(name="md", description="d", body="Body $ARGUMENTS")
    out = await build_skill_prompt(skill, "Y", SkillPromptContext(workspace=Path.cwd()))
    assert out == render_skill_prompt(skill, "Y") == "Body Y"


async def test_prompt_fn_failure_degrades_not_raises(caplog) -> None:
    async def boom(args: str, ctx: SkillPromptContext) -> str:
        raise RuntimeError("kaboom")

    skill = Skill(name="bad", description="d", body="fallback body", prompt_fn=boom)
    with caplog.at_level("WARNING", logger="agent_core.skills.dispatch"):
        out = await build_skill_prompt(skill, "", SkillPromptContext(workspace=Path.cwd()))
    assert "failed to build its prompt" in out
    assert "fallback body" in out  # still includes the static fallback
    # The degradation must be observable, not just inlined into the model-facing text.
    assert any(
        "prompt build failed" in record.getMessage() and "kaboom" in record.getMessage()
        for record in caplog.records
    )


def test_programmatic_skill_decorator_registers() -> None:
    before = len(_PROGRAMMATIC)

    @programmatic_skill
    def _factory() -> Skill:  # pragma: no cover - only the registration matters here
        return Skill(name="temp-test-skill", description="d", body="b")

    assert len(_PROGRAMMATIC) == before + 1
    _PROGRAMMATIC.pop()  # keep global state clean for other tests


def test_is_programmatic_flag() -> None:
    assert Skill(name="m", description="d", body="b").is_programmatic is False
    assert Skill(name="p", description="d", body="b", prompt_fn=lambda a, c: None).is_programmatic is True


# --- the three built-in dynamic skills ---------------------------------------


def test_builtin_programmatic_skills_present() -> None:
    discover()
    names = {s.name for s in builtin_programmatic_skills()}
    assert {"lorem-ipsum", "debug", "skillify"} <= names
    # All three are inline and human-only (not model-invocable), matching the reference.
    for skill in builtin_programmatic_skills():
        if skill.name in {"lorem-ipsum", "debug", "skillify"}:
            assert skill.context is SkillContext.INLINE
            assert skill.disable_model_invocation is True


def _skill(name: str) -> Skill:
    return next(s for s in builtin_programmatic_skills() if s.name == name)


async def test_lorem_ipsum_is_deterministic_and_sized() -> None:
    ctx = SkillPromptContext(workspace=Path.cwd())
    first = await build_skill_prompt(_skill("lorem-ipsum"), "25", ctx)
    second = await build_skill_prompt(_skill("lorem-ipsum"), "25", ctx)
    assert first == second  # deterministic
    assert "~25 tokens" in first


async def test_debug_degrades_without_run_log(tmp_path) -> None:
    ctx = SkillPromptContext(workspace=Path.cwd(), run_dir=str(tmp_path), run_id="missing")
    out = await build_skill_prompt(_skill("debug"), "it hangs", ctx)
    assert "could not be read" in out and "it hangs" in out


async def test_debug_reads_run_log(tmp_path) -> None:
    (tmp_path / "run1.jsonl").write_text('{"event": "boom"}\n', encoding="utf-8")
    ctx = SkillPromptContext(workspace=Path.cwd(), run_dir=str(tmp_path), run_id="run1")
    out = await build_skill_prompt(_skill("debug"), "", ctx)
    assert "boom" in out and "run_log" in out


async def test_skillify_reads_workspace_skills_dir(tmp_path) -> None:
    (tmp_path / ".polaris" / "skills" / "existing").mkdir(parents=True)
    ctx = SkillPromptContext(workspace=tmp_path)
    out = await build_skill_prompt(_skill("skillify"), "the deploy flow", ctx)
    assert "SKILL.md" in out and "deploy flow" in out
    assert "existing" in out  # lists the existing skill it discovered
