from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent_core.compression import CompressionConfig, CompressionPipeline
from agent_core.hooks import HookPipeline, MaxOutputPostHook, OutputLimitConfig
from agent_core.memory import MemoryConfig, MemoryExtractor, MemoryRetriever, MemoryStore
from agent_core.models import LLMContextTooLongError, Message
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import LLMProvider
from agent_core.storage import JSONLRunLogger
from agent_core.tools.builtin import (
    EditFileTool,
    GitDiffTool,
    ListDirTool,
    RunCommandTool,
    RunTestsTool,
    SearchTextTool,
)
from agent_core.tools.demo import EchoTool, ReadTextFileTool, WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.ui import AgentUI, NullUI


@dataclass(slots=True)
class ReActConfig:
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.2
    max_tokens: int = 1024
    # No fixed step cap by default: like Claude Code, the loop runs until the model
    # stops requesting tools. Set an int only if you want a hard ceiling on tool turns.
    max_steps: int | None = None
    # Wall-clock safety net so a runaway/stuck loop can't hang forever; tune as needed.
    max_wall_seconds: float = 300.0
    # Extended-thinking token budget for the Claude provider. None disables thinking
    # (default); a positive int enables it and is passed through _provider_config().
    thinking_budget: int | None = None
    # Stream tokens to a live UI as they arrive. Only takes effect when the UI is
    # live (ConsoleUI); NullUI never streams. CLI exposes this via --no-stream.
    stream: bool = True
    permission: PermissionMode | str = PermissionMode.DEFAULT
    run_dir: str = "runs"
    system_prompt: str = (
        "You are a ReAct agent. Reason briefly, call tools when useful, "
        "and return a final answer when the task is complete."
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
        ui: AgentUI | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ReActConfig()
        self.registry = tools or self.default_registry()
        self.logger = logger or JSONLRunLogger(self.config.run_dir)
        self.compression = CompressionPipeline(self.config.compression)
        self.ui = ui or NullUI()
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
        registry = ToolRegistry()
        # Safe (READ) tools first, then mutating (WRITE) and command/test (DANGEROUS)
        # tools — the permission layer keys off each tool's `risk`.
        for tool in (
            EchoTool(),
            ReadTextFileTool(),
            ListDirTool(),
            SearchTextTool(),
            GitDiffTool(),
            WriteTextFileTool(),
            EditFileTool(),
            RunCommandTool(),
            RunTestsTool(),
        ):
            registry.register(tool)
        return registry

    def run(
        self,
        task: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AgentRunResult:
        messages = [
            Message("system", self.config.system_prompt),
            Message("user", task),
        ]
        self.logger.write("user", {"content": task})
        self._recall(task, messages)

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
                return self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
            if self.config.max_steps is not None and step >= self.config.max_steps:
                return self._stopped(messages, step, "max_steps", "reaching max_steps")
            if time.monotonic() > deadline:
                return self._stopped(messages, step, "deadline", "reaching the wall-clock deadline")
            step += 1

            messages, events = self.compression.maybe_auto_compact(messages)
            for event in events:
                self.logger.write("compression", asdict(event))

            # Stream tokens to the UI only when it is live and streaming is enabled.
            sink = self.ui if (self.ui.is_live and self.config.stream) else None
            self.ui.on_turn_start()
            try:
                result = self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink
                )
            except LLMContextTooLongError:
                messages, events = self.compression.reactive_compact(messages)
                for event in events:
                    self.logger.write("compression", {**asdict(event), "reactive": True})
                self.ui.on_turn_start()
                result = self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink
                )

            self.logger.write(
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
                self.logger.write("final", {"answer": result.content})
                self._extract_memories(messages)
                return AgentRunResult(result.content, messages, step, self.logger.run_id)

            # Intermediate turn: show the reasoning that precedes the tool calls.
            self.ui.on_reasoning(result.content)

            for tool_call in result.tool_calls:
                if cancelled():
                    return self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
                tool_result = self.executor.execute(tool_call)
                observation = f"{tool_result.name}: {tool_result.content}"
                messages.append(
                    Message(
                        "tool",
                        observation,
                        name=tool_result.name,
                        metadata={**tool_result.metadata, "ok": tool_result.ok, "tool_call_id": tool_call.id},
                    )
                )

    def _stopped(self, messages: list[Message], step: int, reason: str, human: str) -> AgentRunResult:
        answer = f"Stopped after {human} without a final answer."
        self.ui.on_stopped(reason, human)
        self.logger.write("final", {"answer": answer, "stopped": reason})
        return AgentRunResult(answer, messages, step, self.logger.run_id)

    def _recall(self, task: str, messages: list[Message]) -> None:
        """Inject relevant past memories as a pinned system block before the task."""
        if self.retriever is None:
            return
        recalled = self.retriever.recall(task)
        if not recalled:
            return
        block = self.retriever.format_block(recalled)
        # Right after the main system prompt, before the user task, tagged so
        # extraction skips it and context_collapse keeps it pinned.
        messages.insert(1, Message("system", block, metadata={"memory": "recall"}))
        self.logger.write("memory_recall", {"count": len(recalled), "ids": [r.id for r in recalled]})

    def _extract_memories(self, messages: list[Message]) -> None:
        """After a completed run, distil durable memories. Best-effort; never raises."""
        if self.extractor is None or not self.config.memory.auto_extract:
            return
        try:
            stored = self.extractor.extract(messages, source_run_id=self.logger.run_id)
        except Exception as exc:  # noqa: BLE001 - extraction must not fail a finished run
            self.logger.write("memory_extract", {"error": f"{type(exc).__name__}: {exc}"})
            return
        if stored:
            self.logger.write("memory_extract", {"count": len(stored), "ids": [r.id for r in stored]})

    def _provider_config(self) -> dict[str, object]:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking_budget": self.config.thinking_budget,
            "stream": self.config.stream,
        }
