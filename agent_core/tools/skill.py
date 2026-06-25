"""The model-facing ``skill`` tool.

Lets the model invoke a loaded skill by name, mirroring how a human types ``/name`` in
chat. An ``inline`` skill returns its rendered prompt as the tool result (the model then
follows those instructions); a ``fork`` skill runs in an isolated sub-agent — reusing the
session's depth-limited ``subagent_factory`` — and returns only the final answer.

The tool reads the per-run :class:`SkillRegistry` off ``self.session.skills`` (bound by
``ReActAgent``), so it never imports the agent or the loader directly. Its advertised
schema lists the *model-invocable* skills, so the model knows what it can call.
"""

from __future__ import annotations

from typing import Any

from agent_core.models import ToolRisk, ToolResult
from agent_core.session import SessionAwareMixin
from agent_core.skills.dispatch import build_skill_prompt, fork_preset
from agent_core.skills.models import SkillContext
from agent_core.skills.programmatic import SkillPromptContext
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class SkillTool(SessionAwareMixin, Tool):
    name = "skill"
    description = (
        "Invoke a named skill — a reusable, pre-written procedure for a common task. "
        "Pass the skill's name as 'command' and any free-form details as 'arguments'. "
        "Prefer a matching skill over improvising when one applies."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The skill name to invoke."},
            "arguments": {
                "type": "string",
                "description": "Free-form arguments/context passed to the skill (may be empty).",
            },
        },
        "required": ["command"],
    }
    risk = ToolRisk.WRITE

    def _available(self) -> list:
        registry = getattr(self.session, "skills", None)
        return registry.model_invocable() if registry else []

    def schema_for_llm(self) -> dict[str, Any]:
        """Advertise the model-invocable skills in the description + ``command`` enum."""
        skills = self._available()
        description = self.description
        if skills:
            lines = []
            for skill in skills:
                hint = f" (when: {skill.when_to_use})" if skill.when_to_use else ""
                lines.append(f"- {skill.name}: {skill.description}{hint}")
            description = description + "\n\nAvailable skills:\n" + "\n".join(lines)

        schema = dict(self.input_schema)
        properties = {key: dict(value) for key, value in schema["properties"].items()}
        if skills:
            properties["command"]["enum"] = [skill.name for skill in skills]
        schema["properties"] = properties
        return {"name": self.name, "description": description, "input_schema": schema}

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        # A fork skill may write anywhere in the workspace; serialize it like dispatch_agent.
        registry = getattr(self.session, "skills", None)
        skill = registry.get(str(arguments.get("command", ""))) if registry else None
        if skill is not None and skill.context is SkillContext.FORK:
            workspace = str(self.session.workspace.resolve())
            return ConcurrencySpec((ResourceLock("fs", workspace, "write", subtree=True),))
        return ConcurrencySpec(exclusive=True)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("command", "")).strip()
        args = str(arguments.get("arguments", "")).strip()
        if not name:
            return ToolResult(self.name, "command must not be empty", ok=False, metadata={"error_type": "BadArgs"})

        registry = getattr(self.session, "skills", None)
        if not registry:
            return ToolResult(
                self.name, "No skills are available in this context.", ok=False, metadata={"error_type": "Unavailable"}
            )
        skill = registry.get(name)
        if skill is None or not skill.model_invocable:
            available = ", ".join(s.name for s in registry.model_invocable()) or "(none)"
            return ToolResult(
                self.name,
                f"Unknown skill {name!r}. Available: {available}",
                ok=False,
                metadata={"error_type": "NotFound"},
            )

        prompt = await build_skill_prompt(skill, args, SkillPromptContext.from_session(self.session))
        if skill.context is SkillContext.FORK:
            factory = self.session.subagent_factory
            if factory is None:
                return ToolResult(
                    self.name,
                    "Fork skills need sub-agents, which are unavailable here.",
                    ok=False,
                    metadata={"error_type": "Unavailable"},
                )
            preset = fork_preset(skill.allowed_tools)
            try:
                answer = await factory(prompt, preset, skill.model)
            except Exception as exc:  # noqa: BLE001 - a skill failure must not crash the parent run
                return ToolResult(self.name, f"Skill error: {type(exc).__name__}: {exc}", ok=False)
            return ToolResult(
                self.name, answer, metadata={"skill": skill.name, "context": "fork", "preset": preset}
            )

        # Inline: hand the rendered instructions back so the model executes them in-context.
        return ToolResult(self.name, prompt, metadata={"skill": skill.name, "context": "inline"})
