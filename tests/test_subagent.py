import asyncio
import threading
import time

from agent_core.agents.multi import MultiAgentCoordinator
from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.session import SessionContext
from agent_core.tools.base import ConcurrencySpec, Tool, WorkspacePathMixin
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.subagent import DispatchAgentTool


# --- DispatchAgentTool (unit, with a stub factory) ---------------------------


async def test_dispatch_passes_task_and_preset_to_factory() -> None:
    calls: list = []

    async def factory(task: str, preset: str) -> str:
        calls.append((task, preset))
        return f"done: {task}"

    tool = DispatchAgentTool(SessionContext(subagent_factory=factory))
    result = await tool.run({"task": "research X", "tool_preset": "full"})
    assert result.ok
    assert result.content == "done: research X"
    assert result.metadata["preset"] == "full"
    assert calls == [("research X", "full")]


async def test_dispatch_defaults_to_read_only() -> None:
    seen: list = []

    async def factory(task: str, preset: str) -> str:
        seen.append(preset)
        return "ok"

    tool = DispatchAgentTool(SessionContext(subagent_factory=factory))
    await tool.run({"task": "do a thing"})
    assert seen == ["read_only"]


async def test_dispatch_rejects_empty_task() -> None:
    async def factory(task: str, preset: str) -> str:
        return "x"

    tool = DispatchAgentTool(SessionContext(subagent_factory=factory))
    result = await tool.run({"task": "   "})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


async def test_dispatch_unavailable_without_factory() -> None:
    result = await DispatchAgentTool(SessionContext()).run({"task": "anything"})
    assert not result.ok
    assert result.metadata["error_type"] == "Unavailable"


async def test_dispatch_captures_child_error() -> None:
    async def boom(task: str, preset: str) -> str:
        raise RuntimeError("kaboom")

    result = await DispatchAgentTool(SessionContext(subagent_factory=boom)).run({"task": "t"})
    assert not result.ok
    assert "kaboom" in result.content


def test_dispatch_is_write_risk() -> None:
    assert DispatchAgentTool().risk is ToolRisk.WRITE


class WorkspaceWriteSleepTool(WorkspacePathMixin, Tool):
    name = "workspace_write_sleep"
    description = "Sleep while declaring a workspace file write."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.WRITE

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict) -> ToolResult:
        time.sleep(float(arguments.get("delay", 0.15)))
        return ToolResult(self.name, "wrote")


async def test_dispatch_read_only_calls_can_run_concurrently(tmp_path) -> None:
    async def factory(task: str, preset: str) -> str:
        await asyncio.sleep(0.15)
        return f"{preset}:{task}"

    registry = ToolRegistry()
    registry.register(DispatchAgentTool(SessionContext(workspace=tmp_path, subagent_factory=factory)))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    results = await executor.execute_many(
        [
            ToolCall("dispatch_agent", {"task": "a"}),
            ToolCall("dispatch_agent", {"task": "b"}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.28
    assert [result.content for result in results] == ["read_only:a", "read_only:b"]


async def test_dispatch_full_conflicts_with_workspace_write(tmp_path) -> None:
    async def factory(task: str, preset: str) -> str:
        await asyncio.sleep(0.15)
        return f"{preset}:{task}"

    registry = ToolRegistry()
    registry.register(DispatchAgentTool(SessionContext(workspace=tmp_path, subagent_factory=factory)))
    registry.register(WorkspaceWriteSleepTool(tmp_path))
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    start = time.perf_counter()
    await executor.execute_many(
        [
            ToolCall("dispatch_agent", {"task": "a", "tool_preset": "full"}),
            ToolCall("workspace_write_sleep", {"path": "nested/file.txt", "delay": 0.15}),
        ]
    )
    elapsed = time.perf_counter() - start

    assert elapsed >= 0.28


async def test_parallel_subagents_overlap_shared_provider_access(tmp_path) -> None:
    """Two children dispatched in one turn now overlap their API calls.

    The shared provider gate bounds concurrency (default cap 8) rather than
    serializing on a mutex, so ``max_active`` exceeds 1 (real parallelism) while
    staying within the cap.
    """

    class TrackingProvider:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
            return await asyncio.to_thread(self._complete_sync, messages, tools, config, stream)

        def _complete_sync(self, messages, tools, config, stream=None) -> LLMResult:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                last = messages[-1]
                if last.role == "user" and last.content == "parent task":
                    return LLMResult(
                        "dispatching",
                        tool_calls=[
                            ToolCall("dispatch_agent", {"task": "child a"}),
                            ToolCall("dispatch_agent", {"task": "child b"}),
                        ],
                        stop_reason="tool_use",
                    )
                if last.role == "tool":
                    return LLMResult("parent done", stop_reason="end")
                if last.role == "user" and last.content.startswith("child "):
                    time.sleep(0.05)
                    return LLMResult(f"answer: {last.content}", stop_reason="end")
                return LLMResult("done", stop_reason="end")
            finally:
                with self._lock:
                    self.active -= 1

    provider = TrackingProvider()
    agent = ReActAgent(
        provider=provider,
        config=ReActConfig(
            run_dir=str(tmp_path),
            permission="auto",
            memory=MemoryConfig(enabled=False),
            max_tool_workers=2,
        ),
    )

    result = await agent.run("parent task")

    assert result.answer == "parent done"
    assert 1 < provider.max_active <= agent.config.max_api_concurrency


# --- end-to-end through a real (fake-provider) agent -------------------------


async def test_agent_spawns_subagent_with_deterministic_answer(tmp_path) -> None:
    agent = ReActAgent(provider=FakeProvider(), config=ReActConfig(run_dir=str(tmp_path)))
    # FakeProvider answers a plain task with "Final answer: <task>".
    answer = await agent._spawn_subagent("explore the repo", "read_only")
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


async def test_dispatch_depth_ceiling_refuses() -> None:
    agent = ReActAgent(provider=FakeProvider())
    agent.session.depth = agent.session.max_depth  # already at the limit
    answer = await agent._spawn_subagent("go deeper", "read_only")
    assert "max sub-agent depth" in answer


# --- MultiAgentCoordinator ---------------------------------------------------


async def test_coordinator_runs_all_and_isolates_failure() -> None:
    class Ok:
        name = "ok"

        async def run(self, task: str) -> str:
            return f"ok:{task}"

    class Bad:
        name = "bad"

        async def run(self, task: str) -> str:
            raise ValueError("nope")

    results = await MultiAgentCoordinator([Ok(), Bad()]).run_all("go")
    assert results["ok"] == "ok:go"
    assert "nope" in results["bad"]


async def test_coordinator_overlaps_agents_and_keeps_declared_order() -> None:
    class Slow:
        name = "slow"

        async def run(self, task: str) -> str:
            await asyncio.sleep(0.05)
            return f"slow:{task}"

    class Fast:
        name = "fast"

        async def run(self, task: str) -> str:
            return f"fast:{task}"

    start = time.perf_counter()
    results = await MultiAgentCoordinator([Slow(), Fast()]).run_all("go")
    elapsed = time.perf_counter() - start
    # Answers keep the agents' declared order regardless of completion order.
    assert list(results) == ["slow", "fast"]
    assert results == {"slow": "slow:go", "fast": "fast:go"}
    assert elapsed < 0.5  # ran concurrently, not back to back


async def test_coordinator_empty() -> None:
    assert await MultiAgentCoordinator([]).run_all("x") == {}
