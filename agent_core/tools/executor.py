from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import asdict, replace
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
from agent_core.permission_types import DecisionSource, PermissionBehavior, PermissionResult
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
        """Execute a turn's tool calls, used by ``ReActAgent.run``.

        Calls are prepared (hooks, permissions), partitioned into resource-conflict
        free waves, and each wave runs via ``asyncio.gather``: async-native tools
        (dispatch / teammate) run directly on the loop so children's API calls
        overlap, while ordinary blocking tools are offloaded to worker threads —
        bounded by ``max_workers`` so the thread ceiling holds.
        """
        if not tool_calls:
            return []
        if not self.parallel_tools:
            return await self._execute_sequential(tool_calls, should_cancel, messages)

        sync_semaphore = asyncio.Semaphore(self.max_workers)
        results: list[ToolResult | None] = [None] * len(tool_calls)
        runnable: list[_PreparedCall] = []
        for index, tool_call in enumerate(tool_calls):
            if should_cancel is not None and should_cancel():
                results[index] = await self._finish(tool_call, self._cancelled_result(tool_call.name), "cancelled")
                continue
            prepared = await self._prepare(index, tool_call, messages, should_cancel)
            if isinstance(prepared, ToolResult):
                results[index] = prepared
            else:
                runnable.append(prepared)

        for wave in self._waves(runnable):
            if should_cancel is not None and should_cancel():
                for prepared in wave:
                    results[prepared.index] = await self._finish(
                        prepared.tool_call, self._cancelled_result(prepared.tool.name), "cancelled"
                    )
                continue
            wave_results = await asyncio.gather(
                *(self._run_tool(prepared, sync_semaphore) for prepared in wave)
            )
            for prepared, result in zip(wave, wave_results, strict=True):
                results[prepared.index] = result

        completed: list[ToolResult] = []
        for index, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"missing tool result at index {index}")
            completed.append(result)
        return completed

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
            if self.permission_classifier is None:
                classifier_verdict = AutoPermissionVerdict(
                    False,
                    "auto mode classifier is unavailable",
                    unavailable=True,
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
                    classifier_verdict = AutoPermissionVerdict(
                        False,
                        "auto mode classifier was cancelled",
                        unavailable=True,
                    )
                except Exception as exc:  # fail closed: evaluator failures are hard denials
                    classifier_verdict = AutoPermissionVerdict(
                        False,
                        f"auto mode evaluator failed: {type(exc).__name__}",
                        unavailable=True,
                    )
            decision = PermissionDecision(
                classifier_verdict.allowed,
                reason=(
                    "auto classifier allowed: "
                    if classifier_verdict.allowed
                    else "auto classifier denied: "
                )
                + classifier_verdict.reason,
            )
            if classifier_verdict.allowed:
                permission_result = PermissionResult.allow(
                    decision.reason,
                    decision_source=DecisionSource.CLASSIFIER,
                    metadata={"classifier": asdict(classifier_verdict)},
                )
            else:
                permission_result = PermissionResult.deny(
                    decision.reason,
                    decision_source=DecisionSource.CLASSIFIER,
                    metadata={"classifier": asdict(classifier_verdict)},
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
        decision = await asyncio.to_thread(self.permissions.confirm, decision, tool, rewritten_call)
        if was_pending_ask:
            if decision.allowed:
                permission_result = PermissionResult.allow(
                    decision.reason,
                    decision_source=DecisionSource.USER,
                    matched_rule=originating_rule,
                )
            elif not decision.ask_user:
                source = DecisionSource.USER if self.permissions.interactive else DecisionSource.MODE
                permission_result = PermissionResult.deny(
                    decision.reason,
                    decision_source=source,
                    matched_rule=originating_rule,
                )
        if self.logger:
            classifier_payload = asdict(classifier_verdict) if classifier_verdict is not None else None
            payload: dict[str, object] = build_permission_audit_event(
                tool.name,
                rewritten_call.arguments,
                permission_context,
                permission_result,
                classifier_payload,
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
