from __future__ import annotations

import asyncio
import time
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from agent_core.agents.team import TeamStore
from agent_core.compression import CompressionConfig, CompressionPipeline
from agent_core.hooks import HookPipeline, MaxOutputPostHook, OutputLimitConfig
from agent_core.memory import MemoryConfig, MemoryExtractor, MemoryRetriever, MemoryStore
from agent_core.models import LLMContextTooLongError, Message, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import LLMProvider, gated_provider
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.storage import JSONLRunLogger
from agent_core.tools.catalog import default_tools
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.team import TeamInboxReadTool, TeamMessageSendTool
from agent_core.ui import AgentUI, NullUI


@dataclass(slots=True)
class ReActConfig:
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.2
    max_tokens: int = 2048
    # No fixed step cap by default: like Claude Code, the loop runs until the model
    # stops requesting tools. Set an int only if you want a hard ceiling on tool turns.
    max_steps: int | None = None
    # Wall-clock safety net so a runaway/stuck loop can't hang forever; tune as needed.
    max_wall_seconds: float = 300.0
    # Extended-thinking token budget for the Claude provider. None disables thinking
    # (default); a positive int enables it and is passed through _provider_config().
    thinking_budget: int | None = 4096
    # Stream tokens to a live UI as they arrive. Only takes effect when the UI is
    # live (ConsoleUI); NullUI never streams. CLI exposes this via --no-stream.
    stream: bool = True
    # Tools returned in the same model turn may run concurrently when their declared
    # resources do not conflict.
    parallel_tools: bool = True
    max_tool_workers: int = 4
    # Cap on simultaneous in-flight LLM API calls across the whole multi-agent
    # fan-out (leader + concurrent children), enforced by the shared provider gate.
    max_api_concurrency: int = 8
    # Sustained API request ceiling per minute across that same fan-out; 0 = unlimited.
    api_rate_limit_per_min: int = 0
    permission: PermissionMode | str = PermissionMode.DEFAULT
    run_dir: str = "runs"
    system_prompt: str = (
        "You are a ReAct agent. Reason briefly, call tools when useful, "
        "and return a final answer when the task is complete. "
        "For non-trivial, multi-step tasks, call update_todos first to lay out a plan, "
        "then keep it current — mark one item in_progress at a time and complete it before "
        "moving on. For self-contained sub-investigations, consider dispatch_agent to run "
        "them in a fresh context. For work that needs a team of cooperating agents, use "
        "the team tools explicitly: team_create, task_create, teammate_spawn, task_update, "
        "and team_status. Multiple tool calls in the same turn may run concurrently when "
        "their resources are independent; if an action needs the output from a previous "
        "tool call, wait until the next turn to request it."
    )
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputLimitConfig = field(default_factory=OutputLimitConfig)


@dataclass(slots=True)
class AgentRunResult:
    answer: str
    messages: list[Message]
    steps: int
    run_id: str


class ReActAgent:
    def __init__(
        self,
        provider: LLMProvider,
        config: ReActConfig | None = None,
        tools: ToolRegistry | None = None,
        hooks: HookPipeline | None = None,
        logger: JSONLRunLogger | None = None,
        memory_store: MemoryStore | None = None,
        retriever: MemoryRetriever | None = None,
        extractor: MemoryExtractor | None = None,
        team_store: TeamStore | None = None,
        ui: AgentUI | None = None,
    ) -> None:
        self.config = config or ReActConfig()
        # Wrap the provider in a shared, bounded concurrency gate (idempotent): the
        # top-level agent creates it from config, and children spawned with
        # ``provider=self.provider`` reuse the same gate, so the whole fan-out shares
        # one budget. Config must be set first so the knobs below resolve.
        self.provider = gated_provider(
            provider,
            max_concurrency=self.config.max_api_concurrency,
            rate_limit=self.config.api_rate_limit_per_min,
        )
        self.registry = tools or self.default_registry()
        self.logger = logger or JSONLRunLogger(self.config.run_dir)
        self.compression = CompressionPipeline(self.config.compression)
        self.ui = ui or NullUI()
        self.team_store = team_store or TeamStore(Path(self.config.run_dir) / "teams")
        # Per-run shared state for session-aware tools (planning, sub-agents). The
        # registry may have been built before this agent existed (the CLI path), so we
        # rebind every session-aware tool to *this* session below.
        self.session = SessionContext(
            workspace=Path.cwd().resolve(),
            subagent_factory=self._spawn_subagent,
            teammate_factory=self._spawn_teammate,
            team_store=self.team_store,
            ui_notify=self.ui.on_todos,
        )
        for tool in self.registry.list():
            if isinstance(tool, SessionAwareMixin):
                tool.bind_session(self.session)
        # Only wire an interactive prompter when the UI can actually ask the user;
        # otherwise an "ask" decision collapses to a denial (non-interactive behavior).
        permissions = PermissionPolicy(
            self.config.permission,
            prompter=self.ui.confirm_tool if self.ui.is_live else None,
        )
        self.executor = ToolExecutor(
            self.registry,
            permissions,
            hooks or HookPipeline(
                post_hooks=[
                    MaxOutputPostHook.from_config(
                        self.config.output, spill_dir=str(Path(self.config.run_dir) / "outputs")
                    )
                ]
            ),
            self.logger,
            self.ui,
            parallel_tools=self.config.parallel_tools,
            max_workers=self.config.max_tool_workers,
        )
        self.memory_store, self.retriever, self.extractor = self._build_memory(
            memory_store, retriever, extractor
        )

    def _build_memory(
        self,
        memory_store: MemoryStore | None,
        retriever: MemoryRetriever | None,
        extractor: MemoryExtractor | None,
    ) -> tuple[MemoryStore | None, MemoryRetriever | None, MemoryExtractor | None]:
        """Wire up cross-conversation memory, but only when it's enabled.

        Injected components win (tests/customisation); otherwise the missing pieces
        are built from ``config.memory``. When memory is disabled this returns all
        ``None`` and the run loop behaves exactly as it did before memory existed.
        """
        if not self.config.memory.enabled:
            return None, None, None
        store = memory_store or MemoryStore(Path(self.config.memory.dir) / "memory.jsonl")
        retriever = retriever or MemoryRetriever(store, self.config.memory)
        extractor = extractor or MemoryExtractor(
            self.provider, store, self.config.memory, self._provider_config()
        )
        return store, retriever, extractor

    @staticmethod
    def default_registry() -> ToolRegistry:
        # The tool set lives in the tools package (self-registered via @builtin_tool
        # and auto-discovered) — adding a tool there needs no change here.
        registry = ToolRegistry()
        for tool in default_tools():
            registry.register(tool)
        return registry

    async def run(
        self,
        task: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentRunResult:
        """Drive the ReAct loop to completion and return the final answer.

        The single (async) entry point: synchronous callers wrap the coroutine in
        one top-level ``asyncio.run(agent.run(task))``; async callers just await it.
        """
        messages = [
            Message("system", self.config.system_prompt),
            Message("user", task),
        ]
        await self.logger.write("user", {"content": task, **self._trace_fields()})
        await self._recall(task, messages)

        cancelled = should_cancel or (lambda: False)
        deadline = time.monotonic() + self.config.max_wall_seconds
        step = 0
        while True:
            # The natural exit below — the model returning no tool calls — is the primary
            # stop, so a task can take as many tool turns as it needs. These guards are only
            # a safety net so a runaway or stuck loop can't spin forever: a cooperative
            # cancel signal (e.g. the user pressing Esc), an optional hard step ceiling,
            # and a wall-clock deadline that bounds the whole run.
            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
            if self.config.max_steps is not None and step >= self.config.max_steps:
                return await self._stopped(messages, step, "max_steps", "reaching max_steps")
            if time.monotonic() > deadline:
                return await self._stopped(messages, step, "deadline", "reaching the wall-clock deadline")
            step += 1

            messages, events = self.compression.maybe_auto_compact(messages)
            for event in events:
                await self.logger.write("compression", asdict(event))

            # Stream tokens to the UI only when it is live and streaming is enabled.
            sink = self.ui if (self.ui.is_live and self.config.stream) else None
            self.ui.on_turn_start()
            try:
                result = await self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                    should_cancel=cancelled,
                )
            except LLMContextTooLongError:
                messages, events = self.compression.reactive_compact(messages)
                for event in events:
                    await self.logger.write("compression", {**asdict(event), "reactive": True})
                self.ui.on_turn_start()
                result = await self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                    should_cancel=cancelled,
                )

            await self.logger.write(
                "llm",
                {
                    "content": result.content,
                    "tool_calls": [asdict(tool_call) for tool_call in result.tool_calls],
                    "stop_reason": result.stop_reason,
                },
            )
            if result.thinking:
                self.ui.on_thinking(result.thinking)

            tool_call_payloads = [asdict(tool_call) for tool_call in result.tool_calls]
            assistant_metadata: dict[str, object] = {}
            if tool_call_payloads:
                assistant_metadata["tool_calls"] = tool_call_payloads
            # Preserve the raw thinking blocks so the provider can replay them on the
            # next turn (required by the API when thinking and tool use span turns).
            if result.thinking_blocks:
                assistant_metadata["thinking_blocks"] = result.thinking_blocks
            messages.append(Message("assistant", result.content, metadata=assistant_metadata))

            # Natural termination: the model stopped requesting tools, so this is the answer.
            if not result.tool_calls:
                self.ui.on_final(result.content)
                await self.logger.write("final", {"answer": result.content})
                await self._extract_memories(messages)
                return AgentRunResult(result.content, messages, step, self.logger.run_id)

            # Intermediate turn: show the reasoning that precedes the tool calls.
            self.ui.on_reasoning(result.content)

            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
            tool_results = await self.executor.execute_many(result.tool_calls, should_cancel=cancelled)
            for tool_call, tool_result in zip(result.tool_calls, tool_results, strict=True):
                observation = f"{tool_result.name}: {tool_result.content}"
                messages.append(
                    Message(
                        "tool",
                        observation,
                        name=tool_result.name,
                        metadata={**tool_result.metadata, "ok": tool_result.ok, "tool_call_id": tool_call.id},
                    )
                )

    async def _stopped(self, messages: list[Message], step: int, reason: str, human: str) -> AgentRunResult:
        answer = f"Stopped after {human} without a final answer."
        self.ui.on_stopped(reason, human)
        await self.logger.write("final", {"answer": answer, "stopped": reason})
        return AgentRunResult(answer, messages, step, self.logger.run_id)

    async def _recall(self, task: str, messages: list[Message]) -> None:
        """Inject relevant past memories as a pinned system block before the task."""
        if self.retriever is None:
            return
        recalled = await self.retriever.recall(task)
        if not recalled:
            return
        block = self.retriever.format_block(recalled)
        # Right after the main system prompt, before the user task, tagged so
        # extraction skips it and context_collapse keeps it pinned.
        messages.insert(1, Message("system", block, metadata={"memory": "recall"}))
        await self.logger.write("memory_recall", {"count": len(recalled), "ids": [r.id for r in recalled]})

    async def _extract_memories(self, messages: list[Message]) -> None:
        """Extraction at natural termination — goes through ``complete`` (and thus
        the shared ``GatedProvider``) without blocking the event loop. Best-effort; never
        raises: a failed extraction must not sink an otherwise completed run."""
        if self.extractor is None or not self.config.memory.auto_extract:
            return
        try:
            stored = await self.extractor.extract(messages, source_run_id=self.logger.run_id)
        except Exception as exc:  # noqa: BLE001 - extraction must not fail a finished run
            await self.logger.write("memory_extract", {"error": f"{type(exc).__name__}: {exc}"})
            return
        if stored:
            await self.logger.write("memory_extract", {"count": len(stored), "ids": [r.id for r in stored]})

    def _provider_config(self) -> dict[str, object]:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking_budget": self.config.thinking_budget,
            "stream": self.config.stream,
        }

    def _trace_fields(self) -> dict[str, object]:
        """Tracing metadata stamped on a run's opening log event.

        Lets concurrent fan-out be reconstructed from ``runs/*.jsonl``: a child's log
        carries the ``parent_run_id`` that spawned it plus its agent/team identity.
        """
        return {
            "agent_name": self.session.agent_name,
            "team_id": self.session.team_id,
            "parent_run_id": self.session.parent_run_id,
        }

    def _make_subagent_child(self, preset: str) -> "ReActAgent | str":
        """Build the ``dispatch_agent`` child (or a refusal string at the depth ceiling).

        The child reuses this agent's gated provider and scalar config but gets a
        narrowed tool set (``read_only`` = READ tools; ``full`` = READ+WRITE) and —
        crucially — **never** the ``dispatch_agent`` tool itself, so sub-agents can't
        recurse. A depth ceiling is a second guard. The child runs silently (``NullUI``)
        and writes its own run log, tagged with this agent's run id as parent.
        """
        if self.session.depth >= self.session.max_depth:
            return "[dispatch_agent] max sub-agent depth reached; refusing to spawn deeper."
        sub_registry = ToolRegistry()
        excluded = {
            "dispatch_agent",
            "team_create",
            "task_create",
            "teammate_spawn",
            "task_update",
            "team_status",
            "team_inbox_read",
            "team_message_send",
        }
        for tool in default_tools(workspace=self.session.workspace):
            if getattr(tool, "name", "") in excluded:
                continue  # prevent recursive fan-out and team orchestration from ordinary sub-agents
            if preset == "read_only" and tool.risk is not ToolRisk.READ:
                continue
            if preset != "full" and tool.risk is ToolRisk.DANGEROUS:
                continue  # never hand a child arbitrary command execution implicitly
            sub_registry.register(tool)
        # Disable memory in the child so a sub-task doesn't recall/extract on its own.
        child_config = replace(self.config, memory=replace(self.config.memory, enabled=False))
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=self.team_store,
            ui=NullUI(),
        )
        child.session.depth = self.session.depth + 1
        child.session.max_depth = self.session.max_depth
        child.session.parent_run_id = self.logger.run_id
        return child

    async def _spawn_subagent(self, task: str, preset: str = "read_only") -> str:
        """``dispatch_agent`` factory — awaits the child on the shared event loop."""
        child = self._make_subagent_child(preset)
        if isinstance(child, str):
            return child
        return (await child.run(task)).answer

    async def _make_teammate_child(
        self,
        team_id: str,
        name: str,
        role: str,
        task_id: str | None,
        preset: str,
    ) -> "tuple[ReActAgent, str] | str":
        """Build a teammate child and its prompt (or a refusal string at the ceiling)."""
        if self.session.depth >= self.session.max_depth:
            return "[teammate_spawn] max sub-agent depth reached; refusing to spawn deeper."

        store = self.team_store
        await store.add_member(team_id, name, role)
        team = await store.get_team(team_id)
        assigned_tasks = [
            task
            for task in await store.list_tasks(team_id)
            if task.get("owner") == name and task.get("status") != "completed"
        ]
        focus_task = await store.get_task(team_id, task_id) if task_id else None

        sub_registry = ToolRegistry()
        excluded = {"dispatch_agent", "team_create", "task_create", "teammate_spawn", "team_status"}
        for tool in default_tools(workspace=self.session.workspace):
            tool_name = getattr(tool, "name", "")
            if tool_name in excluded:
                continue
            if tool_name == "task_update":
                sub_registry.register(tool)
                continue
            if preset == "read_only" and tool.risk is not ToolRisk.READ:
                continue
            if preset != "full" and tool.risk is ToolRisk.DANGEROUS:
                continue
            sub_registry.register(tool)
        sub_registry.register(TeamInboxReadTool())
        sub_registry.register(TeamMessageSendTool())

        child_config = replace(
            self.config,
            permission="auto",
            memory=replace(self.config.memory, enabled=False),
        )
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=store,
            ui=NullUI(),
        )
        child.session.depth = self.session.depth + 1
        child.session.max_depth = self.session.max_depth
        child.session.agent_name = name
        child.session.team_id = team_id
        child.session.parent_run_id = self.logger.run_id
        prompt = self._teammate_prompt(team, name, role, focus_task, assigned_tasks)
        return child, prompt

    async def _spawn_teammate(
        self,
        team_id: str,
        name: str,
        role: str,
        task_id: str | None = None,
        preset: str = "read_only",
    ) -> str:
        """Teammate factory — awaits the teammate turn on the shared event loop."""
        built = await self._make_teammate_child(team_id, name, role, task_id, preset)
        if isinstance(built, str):
            return built
        child, prompt = built
        return (await child.run(prompt)).answer

    @staticmethod
    def _teammate_prompt(
        team: dict[str, object],
        name: str,
        role: str,
        focus_task: dict[str, object] | None,
        assigned_tasks: list[dict[str, object]],
    ) -> str:
        task_block = focus_task if focus_task is not None else assigned_tasks
        return (
            f"You are teammate '{name}' in team '{team['id']}'.\n"
            f"Role: {role}\n"
            f"Team goal: {team['goal']}\n"
            f"Leader: {team['leader']}\n"
            "Use team_inbox_read to read your inbox. Work only on tasks assigned to you "
            "or tasks you explicitly claim with task_update. When you finish or become "
            "blocked, call task_update with the new status and then call team_message_send "
            "to notify the leader.\n"
            f"Current task context:\n{json.dumps(task_block, ensure_ascii=False, indent=2, default=str)}"
        )
