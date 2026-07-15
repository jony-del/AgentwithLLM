from pathlib import Path

from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, ToolCall
from agent_core.permission_types import PermissionMode
from agent_core.react import ReActAgent, ReActConfig
from agent_core.session import PlanArtifactStore


class _FinalProvider:
    async def complete(self, messages, tools, config, stream=None, should_cancel=None):
        return LLMResult("done", stop_reason="end")


async def test_plan_only_writes_dedicated_artifact_and_exit_restores_mode(tmp_path: Path) -> None:
    agent = ReActAgent(
        _FinalProvider(),
        ReActConfig(
            run_dir=str(tmp_path / "runs"),
            session_dir="",
            memory=MemoryConfig(enabled=False),
            project_instructions=False,
            git_context=False,
        ),
    )
    agent.session.plan_store = PlanArtifactStore(tmp_path / "plans")
    agent.set_permission_mode(PermissionMode.PLAN)

    normal = await agent.executor.execute_many(
        [ToolCall("write_text_file", {"path": "normal.txt", "content": "x"})]
    )
    assert not normal[0].ok
    assert "Dry-run" not in normal[0].content
    assert not (agent.session.workspace / "normal.txt").exists()

    written = await agent.executor.execute_many(
        [ToolCall("write_plan", {"content": "# Plan\nImplement safely."})]
    )
    assert written[0].ok
    artifact = agent.session.plan_state.artifact_path
    assert artifact is not None and artifact.read_text(encoding="utf-8").startswith("# Plan")

    agent.permissions.prompter = lambda name, risk, arguments: "once"
    agent.permissions.interactive = True
    exited = await agent.executor.execute_many([ToolCall("exit_plan", {})])

    assert exited[0].ok
    assert agent.config.permission is PermissionMode.DEFAULT
    assert not agent.session.plan_state.active


def test_explicit_mode_switch_away_from_plan_honors_target(tmp_path: Path) -> None:
    agent = ReActAgent(_FinalProvider(), ReActConfig(run_dir=str(tmp_path), session_dir=""))
    agent.session.plan_store = PlanArtifactStore(tmp_path / "plans")
    agent.set_permission_mode(PermissionMode.ACCEPTEDITS)
    agent.set_permission_mode(PermissionMode.PLAN)
    assert agent.session.plan_state.previous_mode == PermissionMode.ACCEPTEDITS.value

    agent.set_permission_mode(PermissionMode.DEFAULT, source="slash")

    assert agent.config.permission is PermissionMode.DEFAULT
    assert not agent.session.plan_state.active


class _TwoTurnPlanProvider:
    def __init__(self) -> None:
        self.system_messages: list[str] = []
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None):
        self.calls += 1
        self.system_messages.append(next(message.content for message in messages if message.role == "system"))
        if self.calls == 1:
            return LLMResult(
                "tracking plan",
                tool_calls=[ToolCall("update_todos", {"todos": [{"content": "inspect", "status": "pending"}]})],
                stop_reason="tool_use",
            )
        return LLMResult("plan ready", stop_reason="end")


async def test_every_model_call_receives_plan_mode_system_context(tmp_path: Path) -> None:
    provider = _TwoTurnPlanProvider()
    agent = ReActAgent(
        provider,
        ReActConfig(
            run_dir=str(tmp_path),
            session_dir="",
            permission="plan",
            memory=MemoryConfig(enabled=False),
            project_instructions=False,
            git_context=False,
        ),
    )

    await agent.run("prepare a plan")

    assert len(provider.system_messages) >= 2
    assert all("<permission-mode-context>" in message for message in provider.system_messages)
    assert all("write_plan" in message and "exit_plan" in message for message in provider.system_messages)
