from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import PurePath

from agent_core.hooks import HookContext, HookEvent, HookOutcome, HookPipeline
from agent_core.models import Message, ToolCall, ToolResult
from agent_core.permission_audit import (
    build_permission_audit_event,
    sanitize_log_payload,
    summarize_arguments,
    summarize_tool_result,
)
from agent_core.permission_classifier import (
    AutoPermissionClassifier,
    AutoPermissionVerdict,
)
from agent_core.permissions import PermissionDecision, PermissionPolicy
from agent_core.permission_safety import is_secret_path
from agent_core.permission_types import (
    DecisionSource,
    PermissionBehavior,
    PermissionResult,
    PermissionUpdate,
)
from agent_core.storage import JSONLRunLogger
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.registry import ToolRegistry
from agent_core.ui import AgentUI, NullUI


class _PreparedCall:
    def __init__(
        self,
        index: int,
        tool_call: ToolCall,
        tool: Tool,
        spec: ConcurrencySpec,
        reason: str,
    ) -> None:
        self.index = index
        self.tool_call = tool_call
        self.tool = tool
        self.spec = spec
        self.reason = reason


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionPolicy,
        hooks: HookPipeline | None = None,
        logger: JSONLRunLogger | None = None,
        ui: AgentUI | None = None,
        permission_classifier: AutoPermissionClassifier | None = None,
        *,
        parallel_tools: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks or HookPipeline()
        self.logger = logger
        self.ui = ui or NullUI()
        self.permission_classifier = permission_classifier
        self.parallel_tools = parallel_tools
        self.max_workers = max(1, int(max_workers))

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        should_cancel: Callable[[], bool] | None = None,
        messages: list[Message] | None = None,
    ) -> list[ToolResult]:
        """Execute a complete turn through the incremental resource scheduler.

        Calls are prepared (hooks, permissions), partitioned into resource-conflict
        free waves, and each wave runs via ``asyncio.gather``: async-native tools
        (dispatch / teammate) run directly on the loop so children's API calls
        overlap, while ordinary blocking tools are offloaded to worker threads —
        bounded by ``max_workers`` so the thread ceiling holds.
        """
        batch = self.begin_batch(messages=messages, should_cancel=should_cancel)
        return await batch.finish(tool_calls, messages=messages)

    def begin_batch(
        self,
        *,
        messages: list[Message] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> "StreamingToolBatch":
        """Create one turn-scoped incremental execution batch."""
        return StreamingToolBatch(self, messages=messages, should_cancel=should_cancel)

    async def _run_tool(self, prepared: _PreparedCall, sync_semaphore: asyncio.Semaphore) -> ToolResult:
        if type(prepared.tool).run is not Tool.run:
            # Async-native tool (spawns child agents): run on the loop so concurrent
            # children share one event loop and the provider gate bounds API calls.
            result = await self._await_tool(prepared)
        else:
            # Ordinary blocking tool: the default Tool.run offloads _invoke to a
            # worker thread; the semaphore keeps the previous thread ceiling.
            async with sync_semaphore:
                result = await self._await_tool(prepared)
        return await self._post_and_finish(prepared, result)

    async def _await_tool(self, prepared: _PreparedCall) -> ToolResult:
        try:
            return await prepared.tool.run(prepared.tool_call.arguments)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure as a failed result
            return ToolResult(
                prepared.tool.name,
                f"Tool error: {exc}",
                ok=False,
                metadata={"error_type": type(exc).__name__},
            )

    @staticmethod
    def _cancelled_result(name: str) -> ToolResult:
        return ToolResult(name, "Tool skipped: cancelled", ok=False, metadata={"error_type": "Cancelled"})

    async def _execute_sequential(
        self,
        tool_calls: list[ToolCall],
        should_cancel: Callable[[], bool] | None,
        messages: list[Message] | None,
    ) -> list[ToolResult]:
        """No concurrency requested: await each call one at a time, in order."""
        results: list[ToolResult] = []
        for index, tool_call in enumerate(tool_calls):
            if should_cancel is not None and should_cancel():
                results.append(await self._finish(tool_call, self._cancelled_result(tool_call.name), "cancelled"))
                continue
            prepared = await self._prepare(index, tool_call, messages, should_cancel)
            if isinstance(prepared, ToolResult):
                results.append(prepared)
                continue
            if should_cancel is not None and should_cancel():
                results.append(
                    await self._finish(prepared.tool_call, self._cancelled_result(prepared.tool.name), "cancelled")
                )
                continue
            result = await self._await_tool(prepared)
            results.append(await self._post_and_finish(prepared, result))
        return results

    async def _prepare(
        self,
        index: int,
        tool_call: ToolCall,
        messages: list[Message] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> _PreparedCall | ToolResult:
        rewritten_call, pre_results = self.hooks.run_pre(tool_call)
        if self.logger:
            await self.logger.write(
                "tool_pre",
                {
                    "tool_call": {
                        "name": rewritten_call.name,
                        "id": rewritten_call.id,
                        "arguments_summary": summarize_arguments(
                            rewritten_call.name, rewritten_call.arguments
                        ),
                    },
                    "pre_results": sanitize_log_payload([asdict(result) for result in pre_results]),
                },
            )
        if any(not result.allowed for result in pre_results):
            result = ToolResult(rewritten_call.name, "Tool rejected by pre hook", ok=False)
            return await self._finish(rewritten_call, result, None)

        try:
            tool = self.registry.get(rewritten_call.name)
        except KeyError:
            result = ToolResult(
                rewritten_call.name,
                f"Unknown tool: {rewritten_call.name}",
                ok=False,
                metadata={"error_type": "UnknownTool"},
            )
            return await self._finish(rewritten_call, result, "unknown tool")

        self.ui.on_tool_call(
            tool.name, tool.risk.value, rewritten_call.arguments, label=self._render_args(tool, rewritten_call)
        )
        self.permissions.last_permission_updates = ()
        permission_context = self.permissions.build_context(tool, rewritten_call.arguments)
        permission_result = await self.permissions.evaluate(
            tool, rewritten_call, context=permission_context
        )
        originating_rule = permission_result.matched_rule
        if permission_result.updated_arguments is not None:
            rewritten_call = replace(rewritten_call, arguments=dict(permission_result.updated_arguments))
            permission_context = self.permissions.build_context(tool, rewritten_call.arguments)
        decision = self.permissions.as_legacy_decision(permission_result)
        classifier_verdict: AutoPermissionVerdict | None = None
        if decision.classify:
            pending_auto_ask = permission_result
            if self.permission_classifier is None:
                classifier_verdict = AutoPermissionVerdict(
                    False,
                    "auto mode classifier is unavailable",
                    unavailable=True,
                    failure_kind="unavailable",
                )
            else:
                try:
                    evaluate = getattr(self.permission_classifier, "evaluate", None)
                    if callable(evaluate):
                        classifier_verdict = await evaluate(
                            tool, rewritten_call, messages or [], should_cancel
                        )
                    else:
                        classifier_verdict = await self.permission_classifier.classify(
                            tool,
                            rewritten_call,
                            messages or [],
                            should_cancel,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # fail closed: evaluator failures are hard denials
                    classifier_verdict = AutoPermissionVerdict(
                        False,
                        f"auto mode evaluator failed: {type(exc).__name__}",
                        unavailable=True,
                        failure_kind="exception",
                    )
            if classifier_verdict.unavailable and self.permissions.interactive and not self.permissions.is_subagent:
                metadata = dict(pending_auto_ask.metadata or {})
                metadata.pop("automated_evaluation", None)
                metadata.update(
                    {
                        "original_behavior": "ask",
                        "auto_fallback": "interactive_prompt",
                        "failure_kind": classifier_verdict.failure_kind or "unavailable",
                        "classifier": asdict(classifier_verdict),
                    }
                )
                permission_result = PermissionResult.ask(
                    f"{pending_auto_ask.reason}; auto evaluator unavailable, manual review required",
                    decision_source=pending_auto_ask.decision_source,
                    updated_arguments=pending_auto_ask.updated_arguments,
                    metadata=metadata,
                    matched_rule=pending_auto_ask.matched_rule,
                    classifier_approvable=False,
                    bypass_immune=pending_auto_ask.bypass_immune,
                    suggestions=pending_auto_ask.suggestions,
                )
                decision = self.permissions.as_legacy_decision(permission_result)
            elif classifier_verdict.allowed:
                decision = PermissionDecision(
                    True,
                    reason="auto classifier allowed: " + classifier_verdict.reason,
                )
                permission_result = PermissionResult.allow(
                    decision.reason,
                    decision_source=DecisionSource.CLASSIFIER,
                    metadata={"classifier": asdict(classifier_verdict), "original_behavior": "ask"},
                )
            else:
                prefix = "auto classifier unavailable: " if classifier_verdict.unavailable else "auto classifier blocked: "
                decision = PermissionDecision(False, reason=prefix + classifier_verdict.reason)
                permission_result = PermissionResult.deny(
                    decision.reason,
                    decision_source=DecisionSource.CLASSIFIER,
                    metadata={
                        "original_behavior": "ask",
                        "classifier": asdict(classifier_verdict),
                        "auto_fallback": "headless_deny" if classifier_verdict.unavailable else "explicit_block",
                        "failure_kind": classifier_verdict.failure_kind,
                    },
                )
        # PermissionRequest (R1 programmatic approval): consulted only for ASK decisions
        # — interactive asks (ask_user) and their headless collapse (ask_collapsed) —
        # never for hard denies, so a hook cannot launder a deny rule. A hook allow
        # resolves the ask; a deny refuses it; no opinion falls through to the normal
        # path (interactive prompt / collapsed denial).
        hook_verdict: dict[str, object] | None = None
        if (decision.ask_user or decision.ask_collapsed) and self.hooks.permission_request_hooks:
            outcome = await self._run_permission_request(tool, rewritten_call, decision.reason)
            if outcome is not None and outcome.decision in {"allow", "deny"}:
                hook_verdict = {"decision": outcome.decision, "reason": outcome.reason}
                allowed = outcome.decision == "allow"
                decision = PermissionDecision(
                    allowed,
                    reason=(
                        f"PermissionRequest hook {'allowed' if allowed else 'denied'}"
                        + (f": {outcome.reason}" if outcome.reason else "")
                    ),
                )
                permission_result = (
                    PermissionResult.allow(
                        decision.reason,
                        decision_source=DecisionSource.HOOK,
                        matched_rule=originating_rule,
                    )
                    if allowed
                    else PermissionResult.deny(
                        decision.reason,
                        decision_source=DecisionSource.HOOK,
                        matched_rule=originating_rule,
                    )
                )
        # The confirm step may block on an interactive prompt (input()); run it on a
        # worker thread so a question to the user doesn't freeze other in-flight work.
        was_pending_ask = permission_result.behavior is PermissionBehavior.ASK
        pending_ask_metadata = permission_result.metadata
        decision = await asyncio.to_thread(self.permissions.confirm, decision, tool, rewritten_call)
        if tool.name == "exit_plan" and isinstance(pending_ask_metadata, dict):
            pending_ask_metadata = dict(pending_ask_metadata)
            requested = rewritten_call.arguments.get("requested_permissions", [])
            pending_ask_metadata["requested_permission_count"] = (
                len(requested) if isinstance(requested, list) else 0
            )
        if was_pending_ask:
            if decision.allowed:
                permission_result = PermissionResult.allow(
                    decision.reason,
                    decision_source=DecisionSource.USER,
                    metadata=pending_ask_metadata,
                    matched_rule=originating_rule,
                )
            elif not decision.ask_user:
                source = DecisionSource.USER if self.permissions.interactive else DecisionSource.MODE
                permission_result = PermissionResult.deny(
                    decision.reason,
                    decision_source=source,
                    metadata=pending_ask_metadata,
                    matched_rule=originating_rule,
                )
        if self.logger:
            classifier_payload = asdict(classifier_verdict) if classifier_verdict is not None else None
            permission_updates: list[dict[str, str]] = []
            permission_update: PermissionUpdate
            for permission_update in self.permissions.last_permission_updates:
                permission_updates.append(
                    {
                        "behavior": permission_update.behavior.value,
                        "rule": permission_update.rule,
                        "destination": permission_update.destination.value,
                    }
                )
            payload: dict[str, object] = build_permission_audit_event(
                tool.name,
                rewritten_call.arguments,
                permission_context,
                permission_result,
                classifier_payload,
                permission_updates,
            )
            payload["decision"] = asdict(decision)  # compatibility for existing replay readers
            if hook_verdict is not None:
                payload["permission_request_hook"] = hook_verdict
            if classifier_verdict is not None:
                payload["auto_classifier"] = classifier_payload
            await self.logger.write("permission", payload)
        if not decision.allowed:
            result = ToolResult(tool.name, f"Tool denied: {decision.reason}", ok=False)
            return await self._finish(rewritten_call, result, decision.reason)
        try:
            spec = tool.concurrency_spec(rewritten_call.arguments)
        except Exception as exc:
            result = ToolResult(tool.name, f"Tool error: {exc}", ok=False, metadata={"error_type": type(exc).__name__})
            return await self._finish(rewritten_call, result, decision.reason)
        return _PreparedCall(index, rewritten_call, tool, spec, decision.reason)

    async def _post_and_finish(self, prepared: _PreparedCall, result: ToolResult) -> ToolResult:
        result = self.hooks.run_post(prepared.tool_call, result)
        return await self._finish(prepared.tool_call, result, prepared.reason, tool=prepared.tool)

    async def _finish(
        self, tool_call: ToolCall, result: ToolResult, reason: str | None, tool: Tool | None = None
    ) -> ToolResult:
        """Log, surface the observation to the UI, and return one exit for every path."""
        if tool_call.name == "read_text_file" and is_secret_path(
            str(tool_call.arguments.get("path", ""))
        ):
            result.metadata["sensitive"] = True
        await self._log_result(tool_call, result, reason)
        if not result.ok:
            # Every failed result (denied, unknown tool, tool error) funnels through
            # here — the one seam for the observational PostToolUseFailure event.
            await self._fire_tool_failure(tool_call, result)
        diff = self._render_result(tool, tool_call, result) if tool is not None else None
        self.ui.on_tool_result(result, diff=diff)
        return result

    async def _run_permission_request(
        self, tool: Tool, tool_call: ToolCall, ask_reason: str
    ) -> HookOutcome | None:
        """Run the control-path PermissionRequest fold over a bounded projection.

        A crash in the runner itself yields NO opinion — the gated action does not
        silently proceed; it falls back to the normal ask path (interactive prompt,
        or the already-collapsed headless denial). External command/http adapters
        additionally carry their own ``fail_mode`` (default closed on this event).
        """
        arguments = {key: str(value)[:200] for key, value in tool_call.arguments.items()}
        ctx = HookContext(
            event=HookEvent.PERMISSION_REQUEST,
            messages=[],
            detail={
                "tool": tool.name,
                "risk": tool.risk.value,
                "ask_reason": ask_reason,
                "arguments": arguments,
            },
        )
        try:
            return await self.hooks.run_permission_request(ctx)
        except Exception as exc:  # noqa: BLE001 - crash → no opinion, never a silent allow
            if self.logger:
                await self.logger.write(
                    "hook",
                    {"event": "PermissionRequest", "error": f"{type(exc).__name__}: {exc}"},
                )
            return None

    async def _fire_tool_failure(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Fire PostToolUseFailure (C5): awaited, fail-open, logged only when subscribed
        (the failed ``tool_result`` record itself is already in the JSONL)."""
        if not self.hooks.tool_failure_hooks:
            return
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE_FAILURE,
            messages=[],
            detail={
                "tool": tool_call.name,
                "error_type": result.metadata.get("error_type"),
                "content": (result.content or "")[:300],
            },
        )
        error: str | None = None
        try:
            await self.hooks.run_tool_failure(ctx)
        except Exception as exc:  # noqa: BLE001 - observational; must never sink a run
            error = f"{type(exc).__name__}: {exc}"
        if self.logger:
            payload: dict[str, object] = {"event": "PostToolUseFailure", "tool": tool_call.name}
            if error:
                payload["error"] = error
            await self.logger.write("hook", payload)

    @staticmethod
    def _render_args(tool: Tool, tool_call: ToolCall) -> str | None:
        """A tool's optional compact argument label; never let display crash a run."""
        try:
            return tool.render_args(tool_call.arguments)
        except Exception:
            return None

    @staticmethod
    def _render_result(tool: Tool, tool_call: ToolCall, result: ToolResult) -> str | None:
        """A tool's optional unified-diff for the result branch; failures are swallowed."""
        if not result.ok:
            return None
        try:
            return tool.render_result(tool_call.arguments, result)
        except Exception:
            return None

    async def _log_result(self, tool_call: ToolCall, result: ToolResult, reason: str | None) -> None:
        if self.logger:
            await self.logger.write(
                "tool_result",
                {
                    "tool": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "arguments_summary": summarize_arguments(tool_call.name, tool_call.arguments),
                    "result": summarize_tool_result(result.content, result.metadata, result.ok),
                    "reason": reason,
                },
            )

    def _waves(self, calls: list[_PreparedCall]) -> list[list[_PreparedCall]]:
        waves: list[list[_PreparedCall]] = []
        current: list[_PreparedCall] = []
        for call in calls:
            if call.spec.exclusive:
                if current:
                    waves.append(current)
                    current = []
                waves.append([call])
                continue
            if any(self._conflicts(call.spec, existing.spec) for existing in current):
                waves.append(current)
                current = [call]
            else:
                current.append(call)
        if current:
            waves.append(current)
        return waves

    def _conflicts(self, left: ConcurrencySpec, right: ConcurrencySpec) -> bool:
        if left.exclusive or right.exclusive:
            return True
        for left_lock in left.locks:
            for right_lock in right.locks:
                if self._locks_conflict(left_lock, right_lock):
                    return True
        return False

    def _locks_conflict(self, left: ResourceLock, right: ResourceLock) -> bool:
        if left.namespace != right.namespace:
            return False
        if left.mode == "read" and right.mode == "read":
            return False
        return self._resource_keys_overlap(left, right)

    def _resource_keys_overlap(self, left: ResourceLock, right: ResourceLock) -> bool:
        left_key = self._normalize_key(left.key)
        right_key = self._normalize_key(right.key)
        if left_key == right_key:
            return True
        if left.subtree and self._is_child_key(right_key, left_key):
            return True
        if right.subtree and self._is_child_key(left_key, right_key):
            return True
        return False

    @staticmethod
    def _normalize_key(key: str) -> str:
        return os.path.normcase(os.path.normpath(str(key)))

    @staticmethod
    def _is_child_key(candidate: str, parent: str) -> bool:
        try:
            PurePath(candidate).relative_to(PurePath(parent))
        except ValueError:
            return False
        return True


@dataclass(slots=True)
class _TrackedCall:
    index: int
    original_call: ToolCall
    messages: list[Message]
    streamed: bool
    state: str = "preparing"
    prepared: _PreparedCall | None = None
    result: ToolResult | None = None
    error: Exception | None = None
    preparation_task: asyncio.Task[None] | None = None
    execution_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_requested: bool = False
    ready_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None


class StreamingToolBatch:
    """Incremental FIFO/resource scheduler for one assistant turn.

    Preparation is serialized because permission state and interactive prompts are
    ordered control paths. Prepared calls run as soon as they do not conflict with
    active earlier calls. Results remain buffered until the core loop appends the
    authoritative assistant message and its ordered tool observations.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        messages: list[Message] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self.executor = executor
        self.base_messages = list(messages or [])
        self.should_cancel = should_cancel
        self.created_at = time.monotonic()
        self.model_finished_at: float | None = None
        self.finished_at: float | None = None
        self._calls: list[_TrackedCall] = []
        self._by_id: dict[str, _TrackedCall] = {}
        self._prepare_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._sync_semaphore = asyncio.Semaphore(executor.max_workers)
        self._aborted = False
        self._closed = False
        self._duplicate_events = 0
        self._mismatched_events = 0
        self._orphaned_events = 0
        self.metrics: dict[str, object] = {}

    @staticmethod
    def _same_call(left: ToolCall, right: ToolCall) -> bool:
        return left.id == right.id and left.name == right.name and left.arguments == right.arguments

    def submit_streamed(self, tool_call: ToolCall, *, assistant_content: str = "") -> bool:
        """Queue a provider-finalized call without blocking the provider read loop."""
        if self._closed or self._aborted or not tool_call.id:
            return False
        existing = self._by_id.get(tool_call.id)
        if existing is not None:
            if self._same_call(existing.original_call, tool_call):
                self._duplicate_events += 1
            else:
                self._mismatched_events += 1
            return False
        streamed_calls = [tracked.original_call for tracked in self._calls if tracked.streamed]
        provisional = Message(
            "assistant",
            assistant_content,
            metadata={"tool_calls": [asdict(call) for call in [*streamed_calls, tool_call]]},
        )
        self._submit(tool_call, messages=[*self.base_messages, provisional], streamed=True)
        return True

    def _submit(self, tool_call: ToolCall, *, messages: list[Message], streamed: bool) -> _TrackedCall:
        tracked = _TrackedCall(len(self._calls), tool_call, list(messages), streamed)
        self._calls.append(tracked)
        if tool_call.id:
            self._by_id[tool_call.id] = tracked
        tracked.preparation_task = asyncio.create_task(self._prepare(tracked))
        return tracked

    async def _prepare(self, tracked: _TrackedCall) -> None:
        try:
            await self._prepare_call(tracked)
        except asyncio.CancelledError:
            await self._complete_cancelled(tracked)
        except Exception as exc:  # preserve execute_many's fail-loud control-path behavior
            await self._complete_error(tracked, exc)

    async def _prepare_call(self, tracked: _TrackedCall) -> None:
        async with self._prepare_lock:
            if self._should_stop(tracked):
                await self._complete_cancelled(tracked)
                return
            prepared = await self.executor._prepare(
                tracked.index,
                tracked.original_call,
                tracked.messages,
                self.should_cancel,
            )
        if isinstance(prepared, ToolResult):
            await self._complete(tracked, prepared)
            return
        tracked.ready_at = time.monotonic()
        async with self._state_lock:
            if self._should_stop(tracked):
                tracked.state = "cancelling"
                tracked.execution_task = asyncio.create_task(self._complete_cancelled(tracked))
                return
            tracked.prepared = prepared
            tracked.state = "queued"
            self._start_ready_locked()

    def _should_stop(self, tracked: _TrackedCall) -> bool:
        return (
            self._aborted
            or tracked.cancel_requested
            or (self.should_cancel is not None and self.should_cancel())
        )

    def _start_ready_locked(self) -> None:
        if self._aborted:
            return
        active = [tracked for tracked in self._calls if tracked.state == "running"]
        for tracked in self._calls:
            if tracked.state in {"completed", "running", "cancelling"}:
                continue
            # An earlier call that is not prepared yet is an ordering barrier.
            if tracked.state == "preparing":
                break
            if tracked.state != "queued" or tracked.prepared is None:
                break
            if tracked.cancel_requested:
                tracked.state = "cancelling"
                tracked.execution_task = asyncio.create_task(self._complete_cancelled(tracked))
                continue
            if not self.executor.parallel_tools and active:
                break
            if any(
                active_call.prepared is not None
                and self.executor._conflicts(tracked.prepared.spec, active_call.prepared.spec)
                for active_call in active
            ):
                break
            tracked.state = "running"
            tracked.started_at = time.monotonic()
            tracked.execution_task = asyncio.create_task(self._execute(tracked))
            active.append(tracked)

    async def _execute(self, tracked: _TrackedCall) -> None:
        prepared = tracked.prepared
        if prepared is None:
            await self._complete_cancelled(tracked)
            return
        try:
            result = await self.executor._run_tool(prepared, self._sync_semaphore)
        except asyncio.CancelledError:
            result = await self.executor._finish(
                prepared.tool_call,
                self.executor._cancelled_result(prepared.tool.name),
                "cancelled",
                tool=prepared.tool,
            )
        await self._complete(tracked, result)

    async def _complete_cancelled(self, tracked: _TrackedCall) -> None:
        if tracked.done.is_set():
            return
        call = tracked.prepared.tool_call if tracked.prepared is not None else tracked.original_call
        result = await self.executor._finish(
            call, self.executor._cancelled_result(call.name), "cancelled"
        )
        await self._complete(tracked, result)

    async def _complete(self, tracked: _TrackedCall, result: ToolResult) -> None:
        async with self._state_lock:
            if tracked.done.is_set():
                return
            tracked.result = result
            tracked.finished_at = time.monotonic()
            tracked.state = "completed"
            tracked.done.set()
            self._start_ready_locked()

    async def _complete_error(self, tracked: _TrackedCall, error: Exception) -> None:
        async with self._state_lock:
            if tracked.done.is_set():
                return
            tracked.error = error
            tracked.finished_at = time.monotonic()
            tracked.state = "completed"
            tracked.done.set()
            self._start_ready_locked()

    async def _cancel_not_started(self, tracked: _TrackedCall) -> None:
        async with self._state_lock:
            if tracked.state not in {"queued", "preparing"}:
                return
            tracked.cancel_requested = True
            if tracked.state == "queued":
                tracked.state = "cancelling"
                tracked.execution_task = asyncio.create_task(self._complete_cancelled(tracked))

    async def abort(self, reason: str) -> None:
        """Stop admitting calls and drain every task so none escapes the turn."""
        if self._closed:
            return
        self._aborted = True
        for tracked in self._calls:
            await self._cancel_not_started(tracked)
            prepared = tracked.prepared
            task = tracked.execution_task
            # Native async tools receive cooperative cancellation (shell tools stop
            # their supervised process tree). Default ``Tool.run`` calls are backed by
            # ``to_thread`` and cannot be killed safely, so those are drained instead.
            if (
                tracked.state == "running"
                and prepared is not None
                and type(prepared.tool).run is not Tool.run
                and task is not None
            ):
                task.cancel()
        await self._wait_all()
        self.finished_at = time.monotonic()
        self._closed = True
        await self._write_metrics(abort_reason=reason)

    async def finish(
        self,
        final_calls: list[ToolCall],
        *,
        messages: list[Message] | None = None,
    ) -> list[ToolResult]:
        """Reconcile the authoritative response and return results in its order."""
        if self._closed:
            raise RuntimeError("streaming tool batch is already closed")
        self.model_finished_at = time.monotonic()
        bindings: list[tuple[ToolCall, _TrackedCall | None]] = []
        final_ids = {call.id for call in final_calls if call.id}

        for call in final_calls:
            tracked = self._by_id.get(call.id) if call.id else None
            if tracked is not None:
                if self._same_call(tracked.original_call, call):
                    bindings.append((call, tracked))
                else:
                    self._mismatched_events += 1
                    await self._cancel_not_started(tracked)
                    bindings.append((call, None))
                continue
            tracked = self._submit(call, messages=list(messages or self.base_messages), streamed=False)
            bindings.append((call, tracked))

        for tracked in self._calls:
            if tracked.streamed and tracked.original_call.id not in final_ids:
                self._orphaned_events += 1
                await self._cancel_not_started(tracked)

        await self._wait_all()
        results: list[ToolResult] = []
        for call, tracked in bindings:
            if tracked is None:
                mismatch = ToolResult(
                    call.name,
                    "Tool skipped: streamed tool call differed from the final response",
                    ok=False,
                    metadata={"error_type": "StreamingToolProtocolMismatch"},
                )
                results.append(await self.executor._finish(call, mismatch, "protocol mismatch"))
                continue
            if tracked.error is not None:
                raise tracked.error
            if tracked.result is None:
                raise RuntimeError(f"missing tool result at index {tracked.index}")
            results.append(tracked.result)

        self.finished_at = time.monotonic()
        self._closed = True
        await self._write_metrics()
        return results

    async def _wait_all(self) -> None:
        while True:
            snapshot = list(self._calls)
            if snapshot:
                await asyncio.gather(*(tracked.done.wait() for tracked in snapshot))
            if len(snapshot) == len(self._calls):
                return

    async def _write_metrics(self, abort_reason: str | None = None) -> None:
        if not any(tracked.streamed for tracked in self._calls):
            return
        model_end = self.model_finished_at or time.monotonic()
        end = self.finished_at or time.monotonic()
        overlap = 0.0
        first_start: float | None = None
        early_started = 0
        for tracked in self._calls:
            if not tracked.streamed or tracked.started_at is None:
                continue
            first_start = (
                tracked.started_at if first_start is None else min(first_start, tracked.started_at)
            )
            # Windows' monotonic clock can quantize both boundaries to the same
            # millisecond even though submission happened before ``finish`` entered.
            early_started += int(tracked.started_at <= model_end)
            tracked_end = tracked.finished_at or end
            overlap += max(0.0, min(tracked_end, model_end) - tracked.started_at)
        self.metrics = {
            "streamed_calls": sum(tracked.streamed for tracked in self._calls),
            "early_started": early_started,
            "post_stream_submitted": sum(not tracked.streamed for tracked in self._calls),
            "first_tool_start_ms": (
                round((first_start - self.created_at) * 1000, 3) if first_start is not None else None
            ),
            "overlap_ms": round(overlap * 1000, 3),
            "post_stream_wait_ms": round(max(0.0, end - model_end) * 1000, 3),
            "duplicate_events": self._duplicate_events,
            "mismatched_events": self._mismatched_events,
            "orphaned_events": self._orphaned_events,
        }
        if abort_reason is not None:
            self.metrics["abort_reason"] = abort_reason
        if self.executor.logger is not None:
            await self.executor.logger.write("streaming_tool_batch", dict(self.metrics))
