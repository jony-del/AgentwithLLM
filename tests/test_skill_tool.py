"""Tests for the model-facing ``skill`` tool and its wiring into ``ReActAgent``."""

from __future__ import annotations

from agent_core.models import ToolCall, ToolRisk
from agent_core.permission_types import PermissionBehavior
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.session import SessionContext
from agent_core.skills import Skill, SkillContext, SkillRegistry, SkillsConfig
from agent_core.tools.skill import SkillTool


def _session(skills: list[Skill], **kwargs) -> SessionContext:
    return SessionContext(skills=SkillRegistry(skills), **kwargs)


# --- unit: SkillTool ---------------------------------------------------------


def test_skill_tool_is_write_risk() -> None:
    assert SkillTool().risk is ToolRisk.WRITE


async def test_inline_invocation_returns_rendered_body() -> None:
    skill = Skill(name="note", description="d", body="Remember $ARGUMENTS", context=SkillContext.INLINE)
    tool = SkillTool(_session([skill]))
    result = await tool.run({"command": "note", "arguments": "the milk"})
    assert result.ok
    assert result.content == "Remember the milk"
    assert result.metadata["context"] == "inline"


async def test_pure_inline_skill_is_permission_safe() -> None:
    skill = Skill(name="note", description="d", body="Remember it", context=SkillContext.INLINE)
    tool = SkillTool(_session([skill]))

    result = await PermissionPolicy(PermissionMode.DEFAULT).evaluate(
        tool, ToolCall("skill", {"command": "note"})
    )

    assert result.behavior is PermissionBehavior.ALLOW


async def test_skill_with_extra_capability_requires_confirmation_even_in_bypass() -> None:
    skill = Skill(
        name="deploy",
        description="d",
        body="Deploy it",
        capabilities=("network",),
        allowed_tools=("bash",),
    )
    tool = SkillTool(_session([skill]))

    result = await PermissionPolicy(PermissionMode.BYPASS).evaluate(
        tool, ToolCall("skill", {"command": "deploy"})
    )

    assert result.behavior is PermissionBehavior.ASK
    assert result.bypass_immune


async def test_fork_invocation_calls_subagent_factory_with_preset_and_model() -> None:
    calls: list = []

    async def factory(task: str, preset: str, model: str | None = None) -> str:
        calls.append((task, preset, model))
        return "child answer"

    skill = Skill(
        name="audit",
        description="d",
        body="Audit it.",
        context=SkillContext.FORK,
        allowed_tools=("read_text_file", "search_text"),  # -> read_only preset
        model="claude-haiku-4-5-20251001",
    )
    tool = SkillTool(_session([skill], subagent_factory=factory))
    result = await tool.run({"command": "audit"})
    assert result.ok
    assert result.content == "child answer"
    assert result.metadata == {"skill": "audit", "context": "fork", "preset": "read_only"}
    assert calls == [("Audit it.", "read_only", "claude-haiku-4-5-20251001")]


async def test_fork_without_factory_is_unavailable() -> None:
    skill = Skill(name="f", description="d", body="b", context=SkillContext.FORK)
    result = await SkillTool(_session([skill])).run({"command": "f"})
    assert not result.ok
    assert result.metadata["error_type"] == "Unavailable"


async def test_unknown_skill_reports_available() -> None:
    skill = Skill(name="known", description="d", body="b")
    result = await SkillTool(_session([skill])).run({"command": "nope"})
    assert not result.ok
    assert result.metadata["error_type"] == "NotFound"
    assert "known" in result.content


async def test_empty_command_rejected() -> None:
    result = await SkillTool(_session([])).run({"command": "  "})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


async def test_model_invocation_disabled_skill_is_not_found() -> None:
    skill = Skill(name="secret", description="d", body="b", disable_model_invocation=True)
    result = await SkillTool(_session([skill])).run({"command": "secret"})
    assert not result.ok
    assert result.metadata["error_type"] == "NotFound"


def test_schema_lists_model_invocable_skills_only() -> None:
    visible = Skill(name="visible", description="Shows up", body="b")
    hidden = Skill(name="hidden", description="d", body="b", disable_model_invocation=True)
    schema = SkillTool(_session([visible, hidden])).schema_for_llm()
    assert schema["input_schema"]["properties"]["command"]["enum"] == ["visible"]
    assert "visible" in schema["description"]
    assert "hidden" not in schema["description"]


def test_schema_without_skills_has_no_enum() -> None:
    schema = SkillTool(_session([])).schema_for_llm()
    assert "enum" not in schema["input_schema"]["properties"]["command"]


def test_schema_does_not_mutate_input_schema() -> None:
    tool = SkillTool(_session([Skill(name="x", description="d", body="b")]))
    tool.schema_for_llm()
    assert "enum" not in tool.input_schema["properties"]["command"]


# --- wiring into ReActAgent --------------------------------------------------


def test_agent_loads_bundled_skills_and_registers_tool() -> None:
    agent = ReActAgent(provider=FakeProvider(), config=ReActConfig())
    assert "skill" in {tool.name for tool in agent.registry.list()}
    assert agent.session.skills is agent.skills
    assert {"commit", "review"} <= {skill.name for skill in agent.skills.list()}


def test_agent_hides_skill_tool_when_no_model_invocable_skills() -> None:
    # Disable every model-invocable bundled skill so none remain for the model to call.
    probe = ReActAgent(provider=FakeProvider(), config=ReActConfig())
    model_names = tuple(skill.name for skill in probe.skills.model_invocable())
    config = ReActConfig(skills=SkillsConfig(disabled=model_names))
    agent = ReActAgent(provider=FakeProvider(), config=config)
    assert agent.skills.model_invocable() == []
    assert "skill" not in {tool.name for tool in agent.registry.list()}


def test_agent_disabled_skills_yields_empty_registry() -> None:
    config = ReActConfig(skills=SkillsConfig(enabled=False))
    agent = ReActAgent(provider=FakeProvider(), config=config)
    assert len(agent.skills) == 0
    assert "skill" not in {tool.name for tool in agent.registry.list()}


def test_subagent_registry_excludes_skill_tool() -> None:
    agent = ReActAgent(provider=FakeProvider())
    child = agent._make_subagent_child("full")
    assert not isinstance(child, str)
    assert "skill" not in {tool.name for tool in child.registry.list()}
