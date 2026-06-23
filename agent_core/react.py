from __future__ import annotations

import asyncio
import time
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from agent_core.agents.team import TeamStore
from agent_core.compression import (
    CompressionConfig,
    CompressionEvent,
    CompressionPipeline,
    is_summary_message,
    parse_prompt_too_long_gap,
    shrink_oversize_messages,
    truncate_head_for_ptl_retry,
)
from agent_core.compression_summary import build_summarizer
from agent_core.context import (
    append_system_context,
    build_git_status,
    build_project_instructions,
    current_date_line,
    prepend_user_context,
)
from agent_core.hooks import HookPipeline, MaxOutputPostHook, OutputLimitConfig
from agent_core.memory import MemoryConfig, MemoryExtractor, MemoryRetriever, MemoryStore
from agent_core.models import LLMContextTooLongError, Message, ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import LLMProvider, gated_provider
from agent_core.session import SessionAwareMixin, SessionContext
from agent_core.storage import JSONLRunLogger
from agent_core.tokens import is_supported_model
from agent_core.tool_use_summary import (
    ToolUseSummaryConfig,
    ToolUseSummarizer,
    build_tool_use_summarizer,
)
from agent_core.transcript import TranscriptStore, new_session_id
from agent_core.tools.catalog import default_tools
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.team import TeamInboxReadTool, TeamMessageSendTool
from agent_core.ui import AgentUI, NullUI

# Injected once as a system message when the run crosses its soft deadline, so the
# model can land a useful final answer before the hard wall-clock stop discards work.
WRAPUP_TEXT = (
    "You are almost out of time for this task. Stop calling tools now and reply with "
    "your best final answer based on what you have so far, noting anything left undone."
)

# Bound on the reactive 413 recovery loop: after summarizing once, we peel the oldest
# whole API rounds and retry ``complete`` at most this many times. This is the guard that
# prevents a 413 → compact → 413 → … infinite loop — once the retries are exhausted (or
# nothing is left to drop) the overflow error propagates instead of spinning forever.
MAX_PTL_RETRIES = 5


def _unsupported_model_refusal(tool: str, model: str) -> str:
    """Actionable refusal when a spawn names a model from no known family.

    An unrecognised id would silently fall back to the conservative 200k window in
    ``tokens`` and be sent verbatim to the provider (a likely API 404), so spawns name
    a known family or omit ``model`` to inherit the parent's.
    """
    return (
        f"[{tool}] unsupported model {model!r}; refusing to spawn. Name a known family "
        "(e.g. claude-haiku-4-5-*, claude-sonnet-4-6, claude-opus-4-8, claude-fable-5) "
        "or omit model to inherit the parent's."
    )


@dataclass(slots=True)
class ReActConfig:
    model: str = "claude-opus-4-8"
    # NOTE: temperature applies only to legacy models (Haiku 4.5, Sonnet, Opus <= 4.6).
    # Opus 4.7+/Fable/Mythos reject sampling params, so the Claude provider drops it for
    # them (see providers/claude.py _is_adaptive_thinking_model). Left for debug runs
    # that override --model to a legacy id.
    temperature: float = 0.2
    # Answer-token cap. 16k is non-truncating headroom for Opus 4.8 (you're billed for
    # tokens actually produced, not the cap) and stays under the SDK's ~16k non-streaming
    # timeout guard while leaving room for streaming runs.
    max_tokens: int = 16000
    # No fixed step cap by default: like Claude Code, the loop runs until the model
    # stops requesting tools. Set an int only if you want a hard ceiling on tool turns.
    max_steps: int | None = None
    # Wall-clock safety net so a runaway/stuck loop can't hang forever. Configurable
    # via the [limits] toml table, AGENT_MAX_WALL_SECONDS, or --max-wall-seconds.
    # None disables the wall cap entirely (cooperative Esc-cancel still applies);
    # the whole sub-agent fan-out shares one budget (see run()'s deadline param).
    max_wall_seconds: float | None = 1800.0
    # Fraction of the run's wall budget after which a one-time "wrap up now" nudge is
    # injected, so the model can return a useful partial answer before the hard stop.
    # 1.0 (or any value >= 1) disables the nudge; ignored when there is no wall cap.
    soft_deadline_fraction: float = 0.9
    # Extended-thinking token budget for the Claude provider. None disables thinking
    # (default); a positive int enables it and is passed through _provider_config().
    thinking_budget: int | None = 4096
    # output_config.effort depth/cost control for effort-capable models (Opus 4.5+,
    # Sonnet 4.6, Fable/Mythos). "high" is the safe agentic default ("xhigh" is Opus
    # 4.8-best but errors on Sonnet 4.6/Opus<=4.6). The provider drops it for models
    # that don't support the level (see providers/claude.py _effort_for_model). None omits it.
    effort: str | None = "high"
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
    # Root for resumable session transcripts (distinct from the ``run_dir`` event log).
    # Mirrors the reference's ``~/.claude/projects`` so sessions are scoped per project
    # (a sanitized cwd subdir) and listable/resumable across projects. ``~`` is expanded.
    # Empty string disables transcript persistence entirely.
    session_dir: str = "~/.polaris/projects"
    # When True (reference behavior), a context-collapse fold writes a compact boundary +
    # summary into the transcript, so a resume loads the *compacted* state (only messages
    # after the last boundary) instead of replaying the full pre-fold history. When False,
    # the transcript stays a faithful full record and a resume reloads everything, letting
    # the live loop re-compact (cheaper/simpler; the original decoupled behavior).
    persist_compaction_boundary: bool = True
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
    # Discover project instructions (CLAUDE.md) at run start and inject them as the
    # ``claudeMd`` entry of the pinned ``<system-reminder>`` userContext user message.
    # Off when False (or AGENT_DISABLE_CLAUDE_MD is truthy, folded in by
    # config.resolve_context_config). The joined block is truncated to claudemd_max_chars.
    project_instructions: bool = True
    claudemd_max_chars: int = 32000
    # Collect a one-time git snapshot (branch/main/user/status/log) at run start and
    # append it as the ``gitStatus`` entry of the base system block (systemContext). Off
    # when False (or AGENT_DISABLE_GIT_CONTEXT is truthy, folded in by
    # config.resolve_context_config).
    git_context: bool = True
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputLimitConfig = field(default_factory=OutputLimitConfig)
    tool_use_summary: ToolUseSummaryConfig = field(default_factory=ToolUseSummaryConfig)


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
        session_id: str | None = None,
        transcript: "TranscriptStore | None" = None,
    ) -> None:
        self.config = config or ReActConfig()
        # Effective monotonic deadline of the in-flight run(), shared with children
        # spawned by the sub-agent/teammate factories so the whole fan-out is bounded
        # by one budget. Set at the top of run(); None when no run is active or uncapped.
        self._active_deadline: float | None = None
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
        # Resumable session transcript (distinct from the event logger above). An injected
        # store wins; otherwise build one from config unless ``session_dir`` is disabled
        # (empty). ``parent_uuid`` chaining is tracked across appends by ``_emit``.
        self.session_id = session_id or new_session_id()
        if transcript is not None:
            self.transcript: TranscriptStore | None = transcript
        elif self.config.session_dir:
            workspace = Path.cwd().resolve()
            self.transcript = TranscriptStore(self.config.session_dir, workspace, self.session_id)
        else:
            self.transcript = None
        # uuid of the last message appended to the transcript, so each new message links
        # back to its predecessor. Reset at the top of every run().
        self._last_message_uuid: str | None = None
        self.compression = CompressionPipeline(self.config.compression)
        # Track A summarizer (or None → deterministic Track B). Built from the gated
        # provider so summary calls share the fan-out's API budget; None for
        # FakeProvider / no key / disabled, keeping offline runs byte-stable.
        self._summarizer = build_summarizer(
            self.provider, self._provider_config(), self.config.compression
        )
        # Async tool-use progress label (UI-only, ephemeral). None when disabled or for
        # FakeProvider (offline byte-stable). The live-UI / leader-only gates are applied
        # at fire time (``_fire_tool_use_summary``) since ``self.ui`` is set just below.
        self._tool_use_summarizer: ToolUseSummarizer | None = build_tool_use_summarizer(
            self.provider, self._provider_config(), self.config.tool_use_summary
        )
        # The in-flight (or just-finished) label task; fired after a tool batch, awaited and
        # emitted on the next turn so the Haiku call overlaps the main model call. Reset per run.
        # ``_pending_tool_use_names`` carries that batch's tool names for the UI emit (the task
        # itself only returns the label string).
        self._pending_tool_use_summary: "asyncio.Task[str | None] | None" = None
        self._pending_tool_use_names: list[str] = []
        self.ui = ui or NullUI()
        self.team_store = team_store or TeamStore(Path(self.config.run_dir) / "teams")
        # Per-run shared state for session-aware tools (planning, sub-agents). The
        # registry may have been built before this agent existed (the CLI path), so we
        # rebind every session-aware tool to *this* session below.
        self.session = SessionContext(
            workspace=Path.cwd().resolve(),
            session_id=self.session_id,
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
        # Running prompt-token figure from the last response's usage (Phase 2B). The
        # auto-compact gate thresholds against this (parity with the reference) instead
        # of a char ratio; 0 until the first response with usage arrives.
        self._last_usage_tokens: int = 0

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
        deadline: float | None = None,
        history: list[Message] | None = None,
    ) -> AgentRunResult:
        """Drive the ReAct loop to completion and return the final answer.

        The single (async) entry point: synchronous callers wrap the coroutine in
        one top-level ``asyncio.run(agent.run(task))``; async callers just await it.

        ``deadline`` is a ``time.monotonic()`` timestamp shared by an enclosing run:
        sub-agents/teammates inherit the parent's deadline so the whole fan-out is
        bounded by one wall-clock budget instead of each child getting a fresh one.
        When ``None``, the deadline is derived from ``config.max_wall_seconds`` (and
        is itself ``None`` when that is unset, disabling the wall cap).

        ``history`` seeds the loop with a prior conversation — the mechanism behind both
        ``chat`` cross-turn memory and ``--resume``/``--continue``. The system prompt and
        project context are rebuilt fresh each call (so any system messages carried in
        ``history`` are dropped), and the history is assumed already persisted, so it is
        only re-linked into the message chain, not re-written to the transcript.
        """
        self._last_message_uuid = None
        self._pending_tool_use_summary = None
        self._run_start_time = time.monotonic()
        # A live UI's interactive permission prompt runs on a worker thread; give it
        # the running loop so it can bridge the prompt back onto the main thread.
        if self.ui.is_live:
            self.ui.bind_event_loop(asyncio.get_running_loop())
        user_message = Message("user", task)
        messages: list[Message] = [Message("system", self.config.system_prompt), user_message]
        await self.logger.write("user", {"content": task, **self._trace_fields()})
        # ``_recall``/``_inject_project_context`` position the recall and pinned
        # userContext blocks relative to the trailing user task, so the task must already
        # be in place. Final front order: system(+gitStatus) → (recall) → userContext → task.
        await self._recall(task, messages)
        await self._inject_project_context(messages)
        # Splice prior conversation in just before the new task (after the pinned context),
        # so the order is [system, (recall), userContext, ...history..., task]. System and
        # pinned context are rebuilt fresh above, so any carried in ``history`` are dropped
        # to avoid stacking a new copy on every turn.
        if history:
            insert_at = messages.index(user_message)
            for past in history:
                if past.role == "system" or past.metadata.get("pinned") == "user_context":
                    continue
                messages.insert(insert_at, past)
                insert_at += 1
                self._last_message_uuid = past.uuid
        # Chain + persist the new task (parent = last history message, or None).
        user_message.parent_uuid = self._last_message_uuid
        if self.transcript is not None:
            await self.transcript.append_message(user_message)
        self._last_message_uuid = user_message.uuid

        cancelled = should_cancel or (lambda: False)
        start = time.monotonic()
        if deadline is None and self.config.max_wall_seconds is not None:
            deadline = start + self.config.max_wall_seconds
        # Stash for the sub-agent/teammate factories so children share this budget.
        self._active_deadline = deadline
        # Soft deadline: a fraction of *this run's* window (not max_wall_seconds, which
        # may differ from the inherited budget), after which we nudge the model once.
        soft_threshold: float | None = None
        if deadline is not None and self.config.soft_deadline_fraction < 1.0:
            soft_threshold = start + (deadline - start) * self.config.soft_deadline_fraction
        wrapup_sent = False
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
            if deadline is not None:
                now = time.monotonic()
                if now > deadline:
                    return await self._stopped(messages, step, "deadline", "reaching the wall-clock deadline")
                # One-time soft nudge so the model can wrap up before the hard stop.
                if soft_threshold is not None and not wrapup_sent and now >= soft_threshold:
                    messages.append(Message("system", WRAPUP_TEXT, metadata={"deadline_wrapup": True}))
                    await self.logger.write("deadline_wrapup", {"step": step})
                    wrapup_sent = True
            step += 1

            # Build the post-compact file re-injection attachments once per turn; they
            # are appended to the conversation tail ONLY if a real fold happens (the
            # pipeline forwards them to build_post_compact_messages inside the collapse
            # stage). Empty when nothing has been read yet.
            attachments = self._build_read_attachments()
            before_compaction = list(messages)
            messages, events = await self.compression.auto_compact(
                messages,
                model=self.config.model,
                token_estimator=self._estimate_tokens,
                summarizer=self._summarizer,
                on_stage=self._compaction_reporter(reactive=False),
                attachments=attachments,
            )
            for event in events:
                await self.logger.write("compression", asdict(event))
            await self._commit_compaction_boundary(before_compaction, messages)

            # Stream tokens to the UI only when it is live and streaming is enabled.
            sink = self.ui if (self.ui.is_live and self.config.stream) else None
            self.ui.on_turn_start()
            try:
                result = await self.provider.complete(
                    messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                    should_cancel=cancelled,
                )
            except asyncio.CancelledError:
                if cancelled():
                    return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")
                raise
            except LLMContextTooLongError as exc:
                # Reactive recovery, bounded so a 413 can never loop forever: summarize
                # aggressively once, then retry ``complete`` up to MAX_PTL_RETRIES times,
                # peeling the oldest whole API rounds before each retry. If a retry still
                # 413s and nothing is left to drop (< 2 rounds), or the retries are
                # exhausted, the overflow propagates.
                gap = parse_prompt_too_long_gap(str(exc))
                before_compaction = list(messages)
                messages, events = await self.compression.reactive_compact(
                    messages,
                    model=self.config.model,
                    token_estimator=self._estimate_tokens,
                    summarizer=self._summarizer,
                    on_stage=self._compaction_reporter(reactive=True),
                    attachments=self._build_read_attachments(),
                )
                for event in events:
                    await self.logger.write("compression", {**asdict(event), "reactive": True})
                await self._commit_compaction_boundary(before_compaction, messages)
                result = None
                for _ in range(MAX_PTL_RETRIES):
                    self.ui.on_turn_start()
                    try:
                        result = await self.provider.complete(
                            messages, self.registry.schemas_for_llm(), self._provider_config(), stream=sink,
                            should_cancel=cancelled,
                        )
                        break
                    except asyncio.CancelledError:
                        if cancelled():
                            return await self._stopped(
                                messages, step, "interrupted", "being interrupted by the user (Esc)"
                            )
                        raise
                    except LLMContextTooLongError as exc_retry:
                        gap = parse_prompt_too_long_gap(str(exc_retry)) or gap
                        truncated = truncate_head_for_ptl_retry(
                            messages, token_gap=gap, token_estimator=self._estimate_tokens
                        )
                        if truncated is not None:
                            messages = truncated
                            await self.logger.write(
                                "compression",
                                {"stage": "ptl_head_truncate", "reactive": True, "kept": len(messages)},
                            )
                            continue
                        # No whole round is safe to drop (< 2 rounds) — a single oversized
                        # round/message is the whole overflow. Last resort: head/tail-truncate
                        # the largest non-preserved messages so the prompt finally fits. We
                        # must shed at least the known gap (else a fraction of the estimate).
                        need = gap or max(1, self._estimate_tokens(messages) // 5)
                        shrunk = shrink_oversize_messages(
                            messages, tokens_to_drop=need, token_estimator=self._estimate_tokens
                        )
                        if shrunk is None:
                            # Even the largest messages are already at their floor — nothing
                            # left to shrink. Surface the overflow rather than spin.
                            raise
                        messages = shrunk
                        await self.logger.write(
                            "compression",
                            {"stage": "ptl_shrink", "reactive": True, "kept": len(messages)},
                        )
                if result is None:
                    # Exhausted MAX_PTL_RETRIES without a successful completion.
                    raise exc

            # Track the running prompt token count from the response usage, when the
            # provider reports it, so the next turn's auto-compact gate thresholds against
            # real usage (parity with the reference) rather than only a char estimate.
            if result.usage is not None:
                self._last_usage_tokens = result.usage.context_tokens

            # Re-poll after the turn completes: an Esc pressed *during* the model
            # call (including a single-turn final answer that requests no tools, and
            # the non-streaming path where deltas can't be polled) is honored here at
            # the next safe point instead of being silently swallowed.
            if cancelled():
                return await self._stopped(messages, step, "interrupted", "being interrupted by the user (Esc)")

            # Emit the previous tool batch's progress label here — the model call just
            # finished, so the background Haiku task fired last turn has had that whole
            # call to resolve (near-zero added latency). This runs before both the final-
            # answer return and the next tool execution, so neither path drops a label.
            await self._flush_pending_tool_use_summary()

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
            await self._emit(messages, Message("assistant", result.content, metadata=assistant_metadata))

            # Natural termination: the model stopped requesting tools, so this is the answer.
            if not result.tool_calls:
                self.ui.on_final(result.content)
                self._emit_recap(messages, step, "completed")
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
                await self._emit(
                    messages,
                    Message(
                        "tool",
                        observation,
                        name=tool_result.name,
                        metadata={**tool_result.metadata, "ok": tool_result.ok, "tool_call_id": tool_call.id},
                    ),
                )
                # Record read-file state HERE (not in the read tool — its mixin shape
                # conflicts with SessionAwareMixin) so it can be re-injected after a
                # post-compaction fold. Defensive: odd/missing args just skip.
                self._record_read_result(tool_call, tool_result)

            # Fire (fire-and-forget) the tool-use progress label for this batch. It runs in
            # the background during next turn's model call and is awaited/emitted there.
            self._fire_tool_use_summary(
                list(zip(result.tool_calls, tool_results, strict=True)), result.content
            )

    def _emit_recap(self, messages: list[Message], step: int, reason: str) -> None:
        duration = time.monotonic() - getattr(self, "_run_start_time", time.monotonic())
        tool_counts = {}
        for msg in messages:
            if msg.role == "tool" and msg.name:
                tool_counts[msg.name] = tool_counts.get(msg.name, 0) + 1
        stats = {
            "duration": duration,
            "steps": step,
            "reason": reason,
            "tool_counts": tool_counts
        }
        self.ui.on_run_completed(stats)

    async def _emit(self, messages: list[Message], message: Message) -> None:
        """Append a real conversation turn: link it into the chain and persist it.

        The transcript is the faithful, append-only record of the conversation; ``uuid``/
        ``parent_uuid`` chain each turn to its predecessor. Compaction is a separate,
        in-memory-only optimization — its summary messages never come through here, so the
        transcript always reflects the true history and a resume reconstructs it intact
        (the live loop re-compacts as needed). Persistence is best-effort and never raises.
        """
        message.parent_uuid = self._last_message_uuid
        messages.append(message)
        if self.transcript is not None:
            await self.transcript.append_message(message)
        self._last_message_uuid = message.uuid

    async def _commit_compaction_boundary(
        self, before: list[Message], after: list[Message]
    ) -> None:
        """Persist a compaction boundary when a context-collapse fold actually happened.

        Mirrors the reference: the new summary becomes a transcript root
        (``parent_uuid=None`` + ``compact_boundary`` tag), the kept tail's first message is
        relinked onto it, and post-compact file attachments are chained on — so a
        ``--resume`` loads only the *compacted* state (turns after the last boundary), not
        the full pre-fold history. The fold is detected by diffing message uuids, so
        snip/microcompact (which truncate content in place, same uuids) are correctly
        ignored. No-op when persistence or the boundary feature is off; best-effort and
        decoupled from the loop's correctness, exactly like ``_emit``.
        """
        if self.transcript is None or not self.config.persist_compaction_boundary:
            return
        before_uuids = {m.uuid for m in before}
        new_msgs = [m for m in after if m.uuid not in before_uuids]
        if not new_msgs:
            return  # snip/microcompact only, or nothing changed — no boundary to write.
        summary = next((m for m in new_msgs if is_summary_message(m)), None)
        if summary is None:
            # A drop with no summary (e.g. emergency PTL head-truncation): don't write a
            # boundary — let a resume reload the full history and re-compact.
            return

        # The summary becomes a new root; record the real predecessor for forensics.
        if self._last_message_uuid is not None:
            summary.metadata["logical_parent_uuid"] = self._last_message_uuid
        summary.metadata["compact_boundary"] = True
        summary.parent_uuid = None
        await self.transcript.append_message(summary)

        # Everything after the summary in the folded list is either kept tail (already on
        # disk, uuid in ``before``) or new attachments. Preserved front matter sits before
        # the summary, so it is excluded here.
        tail = after[after.index(summary) + 1 :]
        recent = [m for m in tail if m.uuid in before_uuids]
        attachments = [m for m in tail if m.uuid not in before_uuids]

        if recent:
            # Re-point the kept tail's head onto the summary via an append-only relink
            # (the original line can't be mutated); keep the in-memory chain in sync.
            recent[0].parent_uuid = summary.uuid
            await self.transcript.append_relink(recent[0].uuid, summary.uuid)
            running = recent[-1].uuid
        else:
            running = summary.uuid

        for attachment in attachments:
            attachment.parent_uuid = running
            await self.transcript.append_message(attachment)
            running = attachment.uuid

        self._last_message_uuid = running

    async def _stopped(self, messages: list[Message], step: int, reason: str, human: str) -> AgentRunResult:
        """Shared exit path for run interruption (cancel / max_steps / deadline)."""
        self._emit_recap(messages, step, reason)
        await self._cancel_pending_tool_use_summary()
        answer = f"Stopped after {human} without a final answer."
        self.ui.on_stopped(reason, human)
        await self.logger.write("final", {"answer": answer, "stopped": reason})
        return AgentRunResult(answer, messages, step, self.logger.run_id)

    def _fire_tool_use_summary(
        self, batch: list[tuple[ToolCall, ToolResult]], last_assistant_text: str
    ) -> None:
        """Kick off (fire-and-forget) the async progress label for a finished tool batch.

        No-op unless a summarizer exists (feature on + real provider), the UI is live (no
        one to show a label to otherwise — don't waste an API call), and this is the leader
        (or sub-agent labels are explicitly enabled). The created task is awaited and emitted
        next turn by ``_flush_pending_tool_use_summary``.
        """
        if self._tool_use_summarizer is None or not self.ui.is_live or not batch:
            return
        if self.session.depth != 0 and not self.config.tool_use_summary.include_subagents:
            return
        self._pending_tool_use_names = [call.name for call, _ in batch]
        self._pending_tool_use_summary = asyncio.create_task(
            self._tool_use_summarizer(batch, last_assistant_text)
        )

    async def _flush_pending_tool_use_summary(self) -> None:
        """Await the pending label task and emit it to the UI + event log (never the API).

        Best-effort: any failure (including the task having degraded to ``None``) just drops
        the label — it must never sink a run. The label is observability only; it is written
        to ``runs/*.jsonl`` but never to the transcript or the API ``messages``.
        """
        task = self._pending_tool_use_summary
        if task is None:
            return
        names = self._pending_tool_use_names
        self._pending_tool_use_summary = None
        self._pending_tool_use_names = []
        try:
            label = await task
        except Exception:  # noqa: BLE001 - a missing label is non-fatal.
            return
        if not label:
            return
        self.ui.on_tool_use_summary(label, names)
        await self.logger.write("tool_use_summary", {"label": label, "tools": names})

    async def _cancel_pending_tool_use_summary(self) -> None:
        """Cancel and reap the in-flight label task without emitting it (used on abort)."""
        task = self._pending_tool_use_summary
        if task is None:
            return
        self._pending_tool_use_summary = None
        self._pending_tool_use_names = []
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - reap quietly (incl. CancelledError).
            pass

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

    async def _inject_project_context(self, messages: list[Message]) -> None:
        """Assemble run-start context the reference Open-ClaudeCode way.

        Runs right after ``_recall``, which already put any recall block at index 1, so
        messages are ``[system, (recall), user]`` here. We build two seams:

        - ``system_context`` ``{"gitStatus": <git block>}`` — appended to the *base*
          system message (``messages[0]``) as ``key: value`` lines via
          ``append_system_context``. Git thus rides inside the single system block
          (always preserved by compaction), not as a standalone system message.
        - ``user_context`` ``{"claudeMd": <CLAUDE.md>, "currentDate": <today>}`` —
          rendered as ONE pinned ``<system-reminder>`` user message via
          ``prepend_user_context`` and inserted immediately before the user task.

        Final order becomes ``system(+gitStatus) → (memory recall system) →
        userContext <system-reminder> user (pinned) → user task``.

        Each source is independently best-effort: a failure in one (or its absence)
        degrades to no injection for that part and never sinks the run. Log event
        names/sizes are unchanged (``git_status`` / ``project_instructions``).

        NOTE for Phase 2B: the userContext message is a *pinned user* message.
        Compaction must preserve pinned messages regardless of role (``_context_collapse``
        will enforce this); the ``pinned`` tag is the seam for that.
        """
        system_context: dict[str, str] = {}
        if self.config.git_context:
            try:
                git_block = await build_git_status(self.session.workspace)
            except Exception as exc:  # noqa: BLE001 - build_git_status shouldn't raise; defensive.
                await self.logger.write("git_status", {"error": f"{type(exc).__name__}: {exc}"})
                git_block = None
            if git_block:
                system_context["gitStatus"] = git_block
                await self.logger.write("git_status", {"chars": len(git_block)})

        if system_context:
            base = messages[0]
            messages[0] = Message(
                "system",
                append_system_context(base.content, system_context),
                metadata=base.metadata,
            )

        user_context: dict[str, str] = {}
        if self.config.project_instructions:
            try:
                text = await build_project_instructions(
                    self.session.workspace, max_chars=self.config.claudemd_max_chars
                )
            except Exception as exc:  # noqa: BLE001 - injection must not fail a run
                await self.logger.write("project_instructions", {"error": f"{type(exc).__name__}: {exc}"})
                text = None
            if text:
                user_context["claudeMd"] = text
                await self.logger.write("project_instructions", {"chars": len(text)})

        # currentDate always rides in userContext (cheap, stdlib-only).
        user_context["currentDate"] = current_date_line()

        meta = prepend_user_context(user_context)
        if meta is not None:
            # Insert immediately before the user task message (the last message at this
            # point). Memory recall, if any, stays its own pinned system message ahead
            # of this one.
            messages.insert(len(messages) - 1, meta)

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

    def _compaction_reporter(self, reactive: bool) -> Callable[[int, int, CompressionEvent], None]:
        """Build the per-stage callback that drives the UI's compaction progress bar.

        Only fires when compaction actually runs (the threshold gate lives in
        ``auto_compact``). Accumulates the overall before/after size and the
        non-empty stage details, emitting start → progress* → end. ``NullUI`` makes
        all three hooks no-ops, so non-interactive runs stay silent."""
        state: dict[str, object] = {"started": False, "before": 0, "after": 0, "details": []}

        def on_stage(done: int, total: int, event: CompressionEvent) -> None:
            if not state["started"]:
                self.ui.on_compaction_start(reactive)
                state["started"] = True
                state["before"] = event.before_chars
            state["after"] = event.after_chars
            if event.detail:
                state["details"].append(event.detail)  # type: ignore[union-attr]
            self.ui.on_compaction_progress(done / total, event.stage)
            if done == total:
                self.ui.on_compaction_end(
                    state["before"], state["after"], ", ".join(state["details"]), reactive  # type: ignore[arg-type]
                )

        return on_stage

    def _record_read_result(self, tool_call: ToolCall, tool_result: ToolResult) -> None:
        """Record a successful ``read_text_file`` result into the session read-state.

        The key is the workspace-resolved path string (stable across relative spellings);
        the value is the file content snapshot. Re-injection (after a fold) reads this back.
        Fully defensive — any odd/missing argument or resolve failure just skips, never
        raises, so a malformed call can't break the loop. (Deferred-tool delta
        re-announcements are intentionally NOT done here: this framework re-sends every
        tool schema each turn via ``registry.schemas_for_llm()``, so tools are never lost
        after compaction.)
        """
        if tool_call.name != "read_text_file" or not tool_result.ok:
            return
        try:
            raw_path = tool_call.arguments.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                return
            # Skip CLAUDE.md — it is already injected (and pinned) as the userContext
            # system-reminder, so re-attaching it after a fold is pure duplication that
            # wastes the re-injection budget. Mirrors the reference's CLAUDE.md exclusion.
            if Path(raw_path).name == "CLAUDE.md":
                return
            key = str((self.session.workspace / raw_path).resolve())
            self.session.record_read(key, tool_result.content)
        except Exception:  # noqa: BLE001 - read-state recording is best-effort, never fatal
            return

    def _build_read_attachments(self) -> list[Message]:
        """Build the post-compact file re-injection message from session read-state.

        Takes the most-recently-read files (newest first) up to
        ``post_compact_max_files``, each truncated to ``post_compact_max_chars_per_file``,
        within a total ``post_compact_total_budget_chars`` budget. Returns ``[]`` when
        nothing has been read. Emits ONE combined ``user`` message framed as untrusted
        situational context (avoids role-alternation worries and is cheaper than many
        messages), tagged ``metadata={"post_compact_attachment": True}`` so it is foldable
        conversation in the tail — NOT pinned (pinning would break the preserved-front
        invariant in ``_context_collapse``).
        """
        state = self.session.read_file_state
        if not state:
            return []
        config = self.config.compression
        max_files = config.post_compact_max_files
        # Budgets are token-based (char/4, matching the auto-compact gate); convert to a
        # char ceiling for the actual truncation, which operates on the raw string.
        per_file = config.post_compact_max_tokens_per_file * 4
        total_budget = config.post_compact_total_budget_tokens * 4
        if max_files <= 0 or per_file <= 0 or total_budget <= 0:
            return []

        sections: list[str] = []
        spent = 0
        # Newest-last dict → reverse for newest-first.
        for key, content in reversed(list(state.items())):
            if len(sections) >= max_files:
                break
            rel = self._relativize(key)
            body = content[:per_file]
            if len(content) > per_file:
                body = f"{body}\n[truncated {len(content) - per_file} chars]"
            remaining = total_budget - spent
            if remaining <= 0:
                break
            if len(body) > remaining:
                body = f"{body[:remaining]}\n[truncated to fit budget]"
            sections.append(f"## {rel}\n{body}")
            spent += len(body)
        if not sections:
            return []
        joined = "\n\n".join(sections)
        text = (
            "<system-reminder>\n"
            "Files you read earlier, re-attached after the conversation was compacted. "
            "This is a snapshot — the file may have changed since; re-read it if you need "
            "the current contents.\n"
            f"{joined}\n"
            "</system-reminder>"
        )
        return [Message("user", text, metadata={"post_compact_attachment": True})]

    def _relativize(self, key: str) -> str:
        """Workspace-relative path for the attachment heading; fall back to the raw key."""
        try:
            return str(Path(key).relative_to(self.session.workspace))
        except Exception:  # noqa: BLE001 - outside the workspace or unrelativizable
            return key

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate the prompt token footprint for the auto-compact gate.

        Uses the larger of the last response's reported context tokens and a cheap
        char/4 estimate of the current history — so the gate reflects real usage once a
        response arrives, but still rises with the not-yet-sent delta and works offline
        (FakeProvider reports usage too, but char/4 keeps the gate honest mid-turn).
        """
        return max(self._last_usage_tokens, sum(len(m.content) for m in messages) // 4)

    def _provider_config(self) -> dict[str, object]:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking_budget": self.config.thinking_budget,
            "effort": self.config.effort,
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

    def _make_subagent_child(self, preset: str, model: str | None = None) -> "ReActAgent | str":
        """Build the ``dispatch_agent`` child (or a refusal string at the depth ceiling).

        The child reuses this agent's gated provider and scalar config but gets a
        narrowed tool set (``read_only`` = READ tools; ``full`` = READ+WRITE) and —
        crucially — **never** the ``dispatch_agent`` tool itself, so sub-agents can't
        recurse. A depth ceiling is a second guard. The child runs silently (``NullUI``)
        and writes its own run log, tagged with this agent's run id as parent.

        An optional ``model`` overrides the child's model independently of this agent's
        (None → inherit). Compaction and the provider request shape adapt per-model
        automatically (each child builds its own summarizer / compaction threshold from
        its own config, and the shared provider picks the body shape per call), so a
        single leader can fan out a mix of Haiku/Sonnet/Opus children.
        """
        if self.session.depth >= self.session.max_depth:
            return "[dispatch_agent] max sub-agent depth reached; refusing to spawn deeper."
        if model and not is_supported_model(model):
            return _unsupported_model_refusal("dispatch_agent", model)
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
        if model:
            child_config = replace(child_config, model=model)
        agent_id = new_session_id()
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=self.team_store,
            ui=NullUI(),
            session_id=self.session_id,
            transcript=self._child_transcript(agent_id),
        )
        child.session.depth = self.session.depth + 1
        child.session.max_depth = self.session.max_depth
        child.session.parent_run_id = self.logger.run_id
        return child

    def _child_transcript(self, agent_id: str) -> "TranscriptStore | None":
        """A sidechain transcript for a spawned child, nested under this session.

        Children write to ``{session_id}/subagents/agent-{id}.jsonl`` so their turns are
        preserved but never surface as standalone resumable sessions (``list_sessions``
        only globs top-level ``*.jsonl``). ``None`` when persistence is disabled.
        """
        if self.transcript is None:
            return None
        return TranscriptStore(
            self.config.session_dir, self.session.workspace, self.session_id, agent_id=agent_id
        )

    async def _spawn_subagent(
        self, task: str, preset: str = "read_only", model: str | None = None
    ) -> str:
        """``dispatch_agent`` factory — awaits the child on the shared event loop."""
        child = self._make_subagent_child(preset, model)
        if isinstance(child, str):
            return child
        return (await child.run(task, deadline=self._active_deadline)).answer

    async def _make_teammate_child(
        self,
        team_id: str,
        name: str,
        role: str,
        task_id: str | None,
        preset: str,
        model: str | None = None,
    ) -> "tuple[ReActAgent, str] | str":
        """Build a teammate child and its prompt (or a refusal string at the ceiling).

        An optional ``model`` overrides the teammate's model independently (None →
        inherit), so one team can mix Haiku/Sonnet/Opus teammates; compaction and the
        provider shape adapt per-model automatically.
        """
        if self.session.depth >= self.session.max_depth:
            return "[teammate_spawn] max sub-agent depth reached; refusing to spawn deeper."
        if model and not is_supported_model(model):
            return _unsupported_model_refusal("teammate_spawn", model)

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
        if model:
            child_config = replace(child_config, model=model)
        child = ReActAgent(
            provider=self.provider,
            config=child_config,
            tools=sub_registry,
            team_store=store,
            ui=NullUI(),
            session_id=self.session_id,
            transcript=self._child_transcript(new_session_id()),
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
        model: str | None = None,
    ) -> str:
        """Teammate factory — awaits the teammate turn on the shared event loop."""
        built = await self._make_teammate_child(team_id, name, role, task_id, preset, model)
        if isinstance(built, str):
            return built
        child, prompt = built
        return (await child.run(prompt, deadline=self._active_deadline)).answer

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
