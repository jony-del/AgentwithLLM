from agent_core.agents.multi import MultiAgentCoordinator
from agent_core.models import ToolRisk
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent
from agent_core.session import SessionContext
from agent_core.tools.subagent import DispatchAgentTool


# --- DispatchAgentTool (unit, with a stub factory) ---------------------------


def test_dispatch_passes_task_and_preset_to_factory() -> None:
    calls: list = []

    def factory(task: str, preset: str) -> str:
        calls.append((task, preset))
        return f"done: {task}"

    tool = DispatchAgentTool(SessionContext(subagent_factory=factory))
    result = tool.run({"task": "research X", "tool_preset": "full"})
    assert result.ok
    assert result.content == "done: research X"
    assert result.metadata["preset"] == "full"
    assert calls == [("research X", "full")]


def test_dispatch_defaults_to_read_only() -> None:
    seen: list = []
    tool = DispatchAgentTool(SessionContext(subagent_factory=lambda t, p: seen.append(p) or "ok"))
    tool.run({"task": "do a thing"})
    assert seen == ["read_only"]


def test_dispatch_rejects_empty_task() -> None:
    tool = DispatchAgentTool(SessionContext(subagent_factory=lambda t, p: "x"))
    result = tool.run({"task": "   "})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


def test_dispatch_unavailable_without_factory() -> None:
    result = DispatchAgentTool(SessionContext()).run({"task": "anything"})
    assert not result.ok
    assert result.metadata["error_type"] == "Unavailable"


def test_dispatch_captures_child_error() -> None:
    def boom(task: str, preset: str) -> str:
        raise RuntimeError("kaboom")

    result = DispatchAgentTool(SessionContext(subagent_factory=boom)).run({"task": "t"})
    assert not result.ok
    assert "kaboom" in result.content


def test_dispatch_is_write_risk() -> None:
    assert DispatchAgentTool().risk is ToolRisk.WRITE


# --- end-to-end through a real (fake-provider) agent -------------------------


def test_agent_spawns_subagent_with_deterministic_answer() -> None:
    agent = ReActAgent(provider=FakeProvider())
    # FakeProvider answers a plain task with "Final answer: <task>".
    answer = agent._spawn_subagent("explore the repo", "read_only")
    assert answer == "Final answer: explore the repo"


def test_subagent_registry_excludes_dispatch_and_dangerous_read_only() -> None:
    agent = ReActAgent(provider=FakeProvider())
    # Re-derive what the read_only child would receive by replicating the filter the
    # factory uses, then assert the guarantees hold.
    from agent_core.tools.catalog import default_tools

    names_read_only = {
        t.name
        for t in default_tools(workspace=agent.session.workspace)
        if t.name != "dispatch_agent" and t.risk is ToolRisk.READ
    }
    assert "dispatch_agent" not in names_read_only  # no recursion
    assert "run_command" not in names_read_only  # no arbitrary exec
    assert "glob" in names_read_only and "search_text" in names_read_only


def test_dispatch_depth_ceiling_refuses() -> None:
    agent = ReActAgent(provider=FakeProvider())
    agent.session.depth = agent.session.max_depth  # already at the limit
    answer = agent._spawn_subagent("go deeper", "read_only")
    assert "max sub-agent depth" in answer


# --- MultiAgentCoordinator ---------------------------------------------------


def test_coordinator_runs_all_and_isolates_failure() -> None:
    class Ok:
        name = "ok"

        def run(self, task: str) -> str:
            return f"ok:{task}"

    class Bad:
        name = "bad"

        def run(self, task: str) -> str:
            raise ValueError("nope")

    results = MultiAgentCoordinator([Ok(), Bad()]).run_all("go")
    assert results["ok"] == "ok:go"
    assert "nope" in results["bad"]


def test_coordinator_empty() -> None:
    assert MultiAgentCoordinator([]).run_all("x") == {}
