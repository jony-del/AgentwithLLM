from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePath

from jsonschema import Draft202012Validator

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
from agent_core.providers.base import StreamedToolCall
from agent_core.storage import JSONLRunLogger
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionScope,
    ExecutionSafety,
    ResourceLock,
    Tool,
    ToolExecutionContext,
    ToolExecutionPolicy,
    WorkspacePathMixin,
)
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import (
    JournalStorage,
    JournalWriteError,
    RecoveryState,
    TurnExecutionJournal,
    WorkspaceTransaction,
    WorkspaceRecoveryRequired,
)
from agent_core.ui import AgentUI, NullUI


class _PreparedCall:
    def __init__(
        self,
        index: int,
        tool_call: ToolCall,
        tool: Tool,
        policy: ToolExecutionPolicy,
        reason: str,
        *,
        provisional: bool = False,
        authorization_deferred: bool = False,
    ) -> None:
        self.index = index
        self.tool_call = tool_call
        self.tool = tool
        self.policy = policy
        self.spec = policy.concurrency
        self.reason = reason
        self.provisional = provisional
        self.authorization_deferred = authorization_deferred
        self.start_observed = not provisional


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
        journal_dir: str | Path | None = None,
        journal_storage: JournalStorage | None = None,
    ) -> None:
        self.registry = registry
        if self.registry.workspace is None:
            scoped = next(
                (item for item in self.registry.list() if isinstance(item, WorkspacePathMixin)),
                None,
            )
            if scoped is not None:
                self.registry.rebind_workspace(str(scoped._workspace))
        self.permissions = permissions
        self.hooks = hooks or HookPipeline()
        self.logger = logger
        self.ui = ui or NullUI()
        self.permission_classifier = permission_classifier
        self.parallel_tools = parallel_tools
        self.max_workers = max(1, int(max_workers))
        workspace = Path(self.registry.workspace or Path.cwd()).resolve()
        if journal_storage is not None:
            self.journal_storage = journal_storage
        elif journal_dir is not None:
            self.journal_storage = JournalStorage.local(
                journal_dir,
                workspace=workspace,
                run_id=logger.run_id if logger is not None else "standalone",
            )
        else:
            self.journal_storage = JournalStorage.local(
                tempfile.mkdtemp(prefix="polaris-turn-journals-"),
                workspace=workspace,
                run_id=logger.run_id if logger is not None else "standalone",
            )
        self.journal_dir = self.journal_storage.run_root
        self._resource_leases: list[tuple[ConcurrencySpec, asyncio.Event, str]] = []

    def rebind_journal_storage(self, storage: JournalStorage) -> None:
        """Switch recovery ownership between turns (for an explicit session resume)."""

        self.journal_storage = storage
        self.journal_dir = storage.run_root

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        should_cancel: Callable[[], bool] | None = None,
        messages: list[Message] | None = None,
        execution_scope: ExecutionScope | None = None,
    ) -> list[ToolResult]:
        """Execute a complete turn through the incremental resource scheduler.

        Calls are prepared (hooks, permissions), partitioned into resource-conflict
        free waves, and each wave runs via ``asyncio.gather``: async-native tools
        (dispatch / teammate) run directly on the loop so children's API calls
        overlap, while ordinary blocking tools are offloaded to worker threads —
        bounded by ``max_workers`` so the thread ceiling holds.
        """
        batch = self.begin_batch(
            messages=messages,
            should_cancel=should_cancel,
            execution_scope=execution_scope,
        )
        return await batch.finish(tool_calls, messages=messages)

    def begin_batch(
        self,
        *,
        messages: list[Message] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        execution_scope: ExecutionScope | None = None,
    ) -> "StreamingToolBatch":
        """Create one turn-scoped incremental execution batch."""
        return StreamingToolBatch(
            self,
            messages=messages,
            should_cancel=should_cancel,
            execution_scope=execution_scope,
        )

    async def _run_tool(
        self,
        prepared: _PreparedCall,
        sync_semaphore: asyncio.Semaphore,
        execution_context: ToolExecutionContext,
    ) -> ToolResult:
        if type(prepared.tool).run is not Tool.run:
            # Async-native tool (spawns child agents): run on the loop so concurrent
            # children share one event loop and the provider gate bounds API calls.
            result = await self._await_tool(prepared, execution_context)
        else:
            # Ordinary blocking tool: the default Tool.run offloads _invoke to a
            # worker thread; the semaphore keeps the previous thread ceiling.
            async with sync_semaphore:
                result = await self._await_tool(prepared, execution_context)
        if prepared.policy.safety is ExecutionSafety.TRANSACTIONAL and result.ok:
            result = replace(
                result,
                metadata={**result.metadata, "execution_status": "staged"},
            )
        # Observation is deliberately separated from execution.  A provisional
        # success is not logged, shown, or sent through PostToolUse until the model
        # response has passed its authoritative termination and reconciliation gate.
        return result

    async def _await_tool(
        self,
        prepared: _PreparedCall,
        execution_context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        try:
            context = execution_context or ToolExecutionContext(uuid.uuid4().hex)
            return await prepared.tool.run_with_context(prepared.tool_call.arguments, context)
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

    @staticmethod
    def _result_dependency_reference(value: object) -> bool:
        if isinstance(value, dict):
            if any(str(key) in {"$tool_result", "$tool_call", "$tool_call_id"} for key in value):
                return True
            return any(ToolExecutor._result_dependency_reference(item) for item in value.values())
        if isinstance(value, list):
            return any(ToolExecutor._result_dependency_reference(item) for item in value)
        if isinstance(value, str):
            lowered = value.casefold()
            return (
                lowered.startswith("tool_result:")
                or "${tool_result" in lowered
                or "{{tool_result" in lowered
            )
        return False

    @staticmethod
    def _validation_failure(tool: Tool, arguments: dict[str, object]) -> ToolResult | None:
        if "_raw_arguments" in arguments:
            return ToolResult(
                tool.name,
                "Tool skipped: provider returned invalid JSON arguments",
                ok=False,
                metadata={"error_type": "InvalidToolArgumentsJSON"},
            )
        if ToolExecutor._result_dependency_reference(arguments):
            return ToolResult(
                tool.name,
                "Tool skipped: same-turn tool results require the next model inference",
                ok=False,
                metadata={"error_type": "ResultDependencyRequiresNextTurn"},
            )
        try:
            errors = sorted(
                Draft202012Validator(tool.input_schema).iter_errors(arguments),
                key=lambda item: list(item.absolute_path),
            )
        except Exception as exc:
            return ToolResult(
                tool.name,
                f"Tool skipped: invalid input schema: {exc}",
                ok=False,
                metadata={"error_type": "InvalidToolSchema"},
            )
        if not errors:
            return None
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        return ToolResult(
            tool.name,
            f"Tool arguments failed schema validation at {location}: {error.message}",
            ok=False,
            metadata={"error_type": "SchemaValidationError", "path": location},
        )

    def _active_lease_conflict(self, spec: ConcurrencySpec) -> str | None:
        active: list[tuple[ConcurrencySpec, asyncio.Event, str]] = []
        conflict: str | None = None
        for leased_spec, done, task_id in self._resource_leases:
            if done.is_set():
                continue
            active.append((leased_spec, done, task_id))
            if self._conflicts(spec, leased_spec):
                conflict = task_id
        self._resource_leases = active
        return conflict

    def _register_resource_lease(
        self, spec: ConcurrencySpec, done: asyncio.Event, task_id: str
    ) -> None:
        self._resource_leases.append((spec, done, task_id))

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
            prepared = await self._prepare(index, tool_call, messages, should_cancel, None)
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
        execution_scope: ExecutionScope | None,
        *,
        provisional: bool = False,
    ) -> _PreparedCall | ToolResult:
        try:
            original_tool = self.registry.get(tool_call.name)
        except KeyError:
            original_tool = None
        if original_tool is not None:
            invalid = self._validation_failure(original_tool, tool_call.arguments)
            if invalid is not None:
                if provisional:
                    return invalid
                return await self._finish(tool_call, invalid, "invalid arguments", tool=original_tool)

        if provisional and (self.hooks.pre_hooks or self.hooks.external_pre_tool_hooks):
            if original_tool is None:
                return ToolResult(
                    tool_call.name,
                    f"Unknown tool: {tool_call.name}",
                    ok=False,
                    metadata={"error_type": "UnknownTool"},
                )
            return _PreparedCall(
                index,
                tool_call,
                original_tool,
                ToolExecutionPolicy(),
                "authorization deferred until authoritative response",
                provisional=True,
                authorization_deferred=True,
            )

        rewritten_call, pre_results = self.hooks.run_pre(tool_call)
        external_pre = HookOutcome()
        if self.hooks.external_pre_tool_hooks:
            external_pre = await self.hooks.run_external_pre_tool(
                HookContext(
                    event=HookEvent.PRE_TOOL_USE,
                    messages=list(messages or []),
                    trigger=rewritten_call.name,
                    detail={
                        "tool_name": rewritten_call.name,
                        "tool_input": dict(rewritten_call.arguments),
                        "tool_use_id": rewritten_call.id,
                    },
                    execution_scope=execution_scope,
                )
            )
            updated_input = external_pre.metadata.get("updated_input")
            if isinstance(updated_input, dict):
                rewritten_call = replace(rewritten_call, arguments=dict(updated_input))
        if self.logger and not provisional:
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
                    "external_pre": sanitize_log_payload(asdict(external_pre)),
                },
            )
        if any(not result.allowed for result in pre_results) or external_pre.block:
            reason = external_pre.reason or "Tool rejected by pre hook"
            result = ToolResult(rewritten_call.name, reason, ok=False)
            if provisional:
                return result
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
            if provisional:
                return result
            return await self._finish(rewritten_call, result, "unknown tool")

        invalid = self._validation_failure(tool, rewritten_call.arguments)
        if invalid is not None:
            if provisional:
                return invalid
            return await self._finish(rewritten_call, invalid, "invalid rewritten arguments", tool=tool)

        if not provisional:
            self.ui.on_tool_call(
                tool.name,
                tool.risk.value,
                rewritten_call.arguments,
                label=self._render_args(tool, rewritten_call),
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
            invalid = self._validation_failure(tool, rewritten_call.arguments)
            if invalid is not None:
                if provisional:
                    return invalid
                return await self._finish(
                    rewritten_call,
                    invalid,
                    "invalid permission-rewritten arguments",
                    tool=tool,
                )
        decision = self.permissions.as_legacy_decision(permission_result)
        classifier_verdict: AutoPermissionVerdict | None = None
        if provisional and not decision.allowed:
            if decision.classify or decision.ask_user or decision.ask_collapsed:
                return _PreparedCall(
                    index,
                    rewritten_call,
                    tool,
                    ToolExecutionPolicy(),
                    "interactive or model authorization deferred",
                    provisional=True,
                    authorization_deferred=True,
                )
            return ToolResult(
                tool.name,
                f"Tool denied: {decision.reason}",
                ok=False,
                metadata={"error_type": "PermissionDenied"},
            )
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
            outcome = await self._run_permission_request(
                tool, rewritten_call, decision.reason, execution_scope=execution_scope
            )
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
        if not provisional:
            decision = await asyncio.to_thread(
                self.permissions.confirm, decision, tool, rewritten_call
            )
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
        if self.logger and not provisional:
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
            if provisional:
                return result
            return await self._finish(rewritten_call, result, decision.reason)
        try:
            # Recompute only after every hook/permission rewrite.  A policy failure
            # degrades to final-only/exclusive rather than guessing resources.
            policy = self.registry.execution_policy(tool, rewritten_call.arguments)
        except Exception as exc:
            policy = ToolExecutionPolicy()
            if self.logger and not provisional:
                await self.logger.write(
                    "tool_policy_degraded",
                    {"tool": tool.name, "error": f"{type(exc).__name__}: {exc}"},
                )
        if (
            policy.safety is ExecutionSafety.TRANSACTIONAL
            and policy.transaction_backend != "workspace"
        ):
            policy = ToolExecutionPolicy()
        if policy.safety is ExecutionSafety.TRANSACTIONAL and not any(
            lock.namespace == "fs" and lock.mode == "write"
            for lock in policy.concurrency.locks
        ):
            policy = ToolExecutionPolicy(
                safety=ExecutionSafety.FINAL_ONLY,
                concurrency=policy.concurrency,
                idempotency_key=policy.idempotency_key,
                safely_cancellable=policy.safely_cancellable,
                execution_timeout=policy.execution_timeout,
            )
        leased_task = self._active_lease_conflict(policy.concurrency)
        if leased_task is not None:
            result = ToolResult(
                tool.name,
                f"Tool skipped: dependency task {leased_task} is still running",
                ok=False,
                metadata={
                    "error_type": "DependencyStillRunning",
                    "task_id": leased_task,
                },
            )
            if provisional:
                return result
            return await self._finish(rewritten_call, result, "resource lease active", tool=tool)
        return _PreparedCall(
            index,
            rewritten_call,
            tool,
            policy,
            decision.reason,
            provisional=provisional,
        )

    async def _observe_provisional_start(self, prepared: _PreparedCall) -> None:
        """Publish a provisionally authorized call only after terminal proof."""

        if prepared.start_observed:
            return
        self.ui.on_tool_call(
            prepared.tool.name,
            prepared.tool.risk.value,
            prepared.tool_call.arguments,
            label=self._render_args(prepared.tool, prepared.tool_call),
        )
        if self.logger:
            await self.logger.write(
                "tool_pre",
                {
                    "tool_call": {
                        "name": prepared.tool_call.name,
                        "id": prepared.tool_call.id,
                        "arguments_summary": summarize_arguments(
                            prepared.tool_call.name, prepared.tool_call.arguments
                        ),
                    },
                    "pre_results": [],
                    "external_pre": {},
                    "deferred_observation": True,
                },
            )
            await self.logger.write(
                "permission",
                {
                    "tool": prepared.tool.name,
                    "decision": {"allowed": True, "reason": prepared.reason},
                    "deferred_observation": True,
                },
            )
        prepared.start_observed = True

    async def _post_and_finish(self, prepared: _PreparedCall, result: ToolResult) -> ToolResult:
        if result.ok and self.hooks.external_post_tool_hooks:
            outcome = await self.hooks.run_external_post_tool(
                HookContext(
                    event=HookEvent.POST_TOOL_USE,
                    messages=[],
                    trigger=prepared.tool_call.name,
                    detail={
                        "tool_name": prepared.tool_call.name,
                        "tool_input": dict(prepared.tool_call.arguments),
                        "tool_use_id": prepared.tool_call.id,
                        "tool_response": result.content[:50_000],
                    },
                )
            )
            updated_output = outcome.metadata.get("updated_output")
            if updated_output is not None:
                result = replace(result, content=str(updated_output))
            if outcome.additional_context:
                result = replace(
                    result,
                    content=result.content + "\n\n[PostToolUse hook]\n" + outcome.additional_context,
                )
        result = self.hooks.run_post(prepared.tool_call, result)
        return await self._finish(prepared.tool_call, result, prepared.reason, tool=prepared.tool)

    async def _finish(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        reason: str | None,
        tool: Tool | None = None,
        *,
        fire_failure_hook: bool = True,
    ) -> ToolResult:
        """Log, surface the observation to the UI, and return one exit for every path."""
        if tool_call.name == "read_text_file" and is_secret_path(
            str(tool_call.arguments.get("path", ""))
        ):
            result.metadata["sensitive"] = True
        await self._log_result(tool_call, result, reason)
        if not result.ok and fire_failure_hook:
            # Every failed result (denied, unknown tool, tool error) funnels through
            # here — the one seam for the observational PostToolUseFailure event.
            await self._fire_tool_failure(tool_call, result)
        diff = self._render_result(tool, tool_call, result) if tool is not None else None
        self.ui.on_tool_result(result, diff=diff)
        return result

    async def _run_permission_request(
        self,
        tool: Tool,
        tool_call: ToolCall,
        ask_reason: str,
        *,
        execution_scope: ExecutionScope | None = None,
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
            execution_scope=execution_scope,
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
    ordinal: int
    original_call: ToolCall
    messages: list[Message]
    streamed: bool
    explicit_ordinal: bool
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
    dependencies: set[int] = field(default_factory=set)
    hard_dependencies: set[int] = field(default_factory=set)
    safety_hint: ExecutionSafety = ExecutionSafety.FINAL_ONLY
    provider_item_id: str | None = None
    observed: bool = False
    underlying_task: asyncio.Task[ToolResult] | None = None
    cleanup_pending: bool = False
    arguments_completed_at: float = field(default_factory=time.monotonic)
    recovery_message_uuid: str = field(default_factory=lambda: uuid.uuid4().hex)


class StreamingToolBatch:
    """Incremental ordinal/DAG scheduler for one authoritative assistant turn."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        messages: list[Message] | None,
        should_cancel: Callable[[], bool] | None,
        execution_scope: ExecutionScope | None = None,
    ) -> None:
        self.executor = executor
        self.base_messages = list(messages or [])
        self.should_cancel = should_cancel
        self.execution_scope = execution_scope
        self.turn_id = uuid.uuid4().hex
        self.journal = TurnExecutionJournal(executor.journal_storage, self.turn_id)
        self.created_at = time.monotonic()
        self.model_finished_at: float | None = None
        self.finished_at: float | None = None
        self._calls: list[_TrackedCall] = []
        self._by_id: dict[str, _TrackedCall] = {}
        self._by_ordinal: dict[int, _TrackedCall] = {}
        self._prepare_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()
        self._sync_semaphore = asyncio.Semaphore(executor.max_workers)
        self._transaction: WorkspaceTransaction | None = None
        self._committed_paths: list[str] = []
        self._commit_finished_at: float | None = None
        self._aborted = False
        self._closed = False
        self._turn_validated = False
        self._allow_final_only = False
        self._precommit_final: set[int] = set()
        self._protocol_invalid = False
        self._journal_error: str | None = None
        self._duplicate_events = 0
        self._mismatched_events = 0
        self._orphaned_events = 0
        self._history_context: dict[str, object] | None = None
        self.metrics: dict[str, object] = {}

    def _create_task(self, awaitable, *, name: str | None = None):
        if self.execution_scope is not None:
            return self.execution_scope.create_task(awaitable, name=name)
        return asyncio.create_task(awaitable, name=name)

    @staticmethod
    def _same_call(left: ToolCall, right: ToolCall) -> bool:
        return left.id == right.id and left.name == right.name and left.arguments == right.arguments

    def submit_streamed(
        self,
        tool_call: ToolCall,
        *,
        assistant_content: str = "",
        ordinal: int | None = None,
    ) -> bool:
        """Queue a provider-finalized call without blocking the provider read loop."""
        if self._closed or self._aborted or not tool_call.id:
            return False
        existing = self._by_id.get(tool_call.id)
        if existing is not None:
            if self._same_call(existing.original_call, tool_call):
                self._duplicate_events += 1
            else:
                self._mismatched_events += 1
                self._protocol_invalid = True
            return False
        explicit_ordinal = ordinal is not None
        if ordinal is None:
            ordinal = max(self._by_ordinal, default=-1) + 1
        if ordinal < 0 or ordinal in self._by_ordinal:
            self._mismatched_events += 1
            self._protocol_invalid = True
            return False
        streamed_calls = [tracked.original_call for tracked in self._calls if tracked.streamed]
        provisional = Message(
            "assistant",
            assistant_content,
            metadata={"tool_calls": [asdict(call) for call in [*streamed_calls, tool_call]]},
        )
        self._submit(
            tool_call,
            messages=[*self.base_messages, provisional],
            streamed=True,
            ordinal=ordinal,
            explicit_ordinal=explicit_ordinal,
        )
        return True

    def submit_streamed_event(
        self,
        event: StreamedToolCall,
        *,
        assistant_content: str = "",
    ) -> bool:
        """Preferred admission API with stable provider item identity."""

        if event.tool_call.id != event.call_id:
            self._mismatched_events += 1
            self._protocol_invalid = True
            return False
        accepted = self.submit_streamed(
            event.tool_call,
            assistant_content=assistant_content,
            ordinal=event.ordinal,
        )
        if accepted:
            tracked = self._by_id.get(event.call_id)
            if tracked is not None:
                tracked.provider_item_id = event.provider_item_id
        return accepted

    def _submit(
        self,
        tool_call: ToolCall,
        *,
        messages: list[Message],
        streamed: bool,
        ordinal: int,
        explicit_ordinal: bool = True,
    ) -> _TrackedCall:
        safety_hint = ExecutionSafety.FINAL_ONLY
        try:
            safety_hint = self.executor.registry.get(tool_call.name).execution_safety
        except KeyError:
            pass
        tracked = _TrackedCall(
            len(self._calls), ordinal, tool_call, list(messages), streamed,
            explicit_ordinal, safety_hint=safety_hint,
        )
        self._calls.append(tracked)
        self._by_ordinal[ordinal] = tracked
        if tool_call.id:
            self._by_id[tool_call.id] = tracked
        try:
            arguments_digest = __import__("hashlib").sha256(
                repr(sorted(tool_call.arguments.items())).encode("utf-8", errors="replace")
            ).hexdigest()
            self.journal.record_telemetry(
                RecoveryState.DISCOVERED,
                ordinal=ordinal,
                tool_call_id=tool_call.id,
                tool=tool_call.name,
                arguments_digest=arguments_digest,
                streamed=streamed,
            )
        except (JournalWriteError, TypeError) as exc:
            self._journal_error = str(exc)
        tracked.preparation_task = self._create_task(self._prepare(tracked))
        return tracked

    async def _prepare(self, tracked: _TrackedCall) -> None:
        try:
            await self._prepare_call(tracked)
        except asyncio.CancelledError:
            await self._complete_cancelled(tracked)
        except Exception as exc:  # preserve execute_many's fail-loud control-path behavior
            await self._complete_error(tracked, exc)

    async def _prepare_call(self, tracked: _TrackedCall) -> None:
        if not self.executor.parallel_tools and tracked.ordinal > 0:
            previous = self._by_ordinal.get(tracked.ordinal - 1)
            if previous is not None:
                await previous.done.wait()
        if tracked.streamed and not self._turn_validated:
            if self._should_stop(tracked):
                await self._complete_cancelled(tracked)
                return
            prepared = await self.executor._prepare(
                tracked.index,
                tracked.original_call,
                tracked.messages,
                self.should_cancel,
                self.execution_scope,
                provisional=True,
            )
        else:
            # Interactive confirmation and mutable permission updates remain
            # serialized, while the side-effect-free streaming preflight above
            # can run concurrently.
            async with self._prepare_lock:
                if self._should_stop(tracked):
                    await self._complete_cancelled(tracked)
                    return
                prepared = await self.executor._prepare(
                    tracked.index,
                    tracked.original_call,
                    tracked.messages,
                    self.should_cancel,
                    self.execution_scope,
                )
        if isinstance(prepared, ToolResult):
            tracked.observed = not tracked.streamed or self._turn_validated
            await self._complete(tracked, prepared)
            return
        tracked.ready_at = time.monotonic()
        async with self._state_lock:
            if self._should_stop(tracked):
                tracked.state = "cancelling"
                tracked.execution_task = self._create_task(self._complete_cancelled(tracked))
                return
            tracked.prepared = prepared
            tracked.safety_hint = prepared.policy.safety
            tracked.state = "queued"
            try:
                await asyncio.to_thread(
                    self.journal.record,
                    RecoveryState.ADMITTED,
                    ordinal=tracked.ordinal,
                    tool=prepared.tool.name,
                    safety=prepared.policy.safety.value,
                )
            except JournalWriteError as exc:
                self._journal_error = str(exc)
            self._rebuild_dependencies_locked()
            self._start_ready_locked()

    def _should_stop(self, tracked: _TrackedCall) -> bool:
        return (
            self._aborted
            or tracked.cancel_requested
            or (self.should_cancel is not None and self.should_cancel())
            or (self.execution_scope is not None and self.execution_scope.cancelled())
            or (
                self.execution_scope is not None
                and self.execution_scope.deadline is not None
                and time.monotonic() >= self.execution_scope.deadline
            )
        )

    def _rebuild_dependencies_locked(self) -> None:
        prepared = sorted(
            (item for item in self._calls if item.prepared is not None),
            key=lambda item: item.ordinal,
        )
        for position, tracked in enumerate(prepared):
            tracked.dependencies.clear()
            tracked.hard_dependencies.clear()
            assert tracked.prepared is not None
            for earlier in prepared[:position]:
                assert earlier.prepared is not None
                if not self.executor._conflicts(earlier.prepared.spec, tracked.prepared.spec):
                    continue
                tracked.dependencies.add(earlier.index)
                if self._requires_success(earlier.prepared.spec, tracked.prepared.spec):
                    tracked.hard_dependencies.add(earlier.index)

    def _requires_success(self, earlier: ConcurrencySpec, later: ConcurrencySpec) -> bool:
        for left in earlier.locks:
            for right in later.locks:
                if self.executor._locks_conflict(left, right) and right.requires_success:
                    return True
        return False

    def _all_lower_ordinals_known_and_prepared(self, tracked: _TrackedCall) -> bool:
        for ordinal in range(tracked.ordinal):
            earlier = self._by_ordinal.get(ordinal)
            if earlier is None or earlier.state == "preparing":
                return False
        return True

    def _safety(self, tracked: _TrackedCall) -> ExecutionSafety:
        return tracked.prepared.policy.safety if tracked.prepared is not None else tracked.safety_hint

    def _can_enter_phase(self, tracked: _TrackedCall) -> bool:
        safety = self._safety(tracked)
        if safety is not ExecutionSafety.FINAL_ONLY:
            # A provider without a stable ordinal is never a speculative admission
            # source. It falls back to authoritative-turn execution.
            return tracked.explicit_ordinal or self._turn_validated
        if not self._turn_validated:
            return False
        if self._allow_final_only:
            return True
        if not self.executor.parallel_tools:
            return not any(
                item.ordinal < tracked.ordinal
                and self._safety(item) is ExecutionSafety.TRANSACTIONAL
                for item in self._calls
            )
        return tracked.index in self._precommit_final

    def _dependency_failure(self, tracked: _TrackedCall) -> tuple[str, str] | None:
        for index in tracked.dependencies:
            dependency = self._calls[index]
            if not dependency.done.is_set():
                continue
            if dependency.result is not None and dependency.result.metadata.get("resource_lease_running"):
                return "DependencyStillRunning", (
                    f"dependency task {dependency.result.metadata.get('task_id')} is still running"
                )
            if (
                index in tracked.hard_dependencies
                and (dependency.result is None or not dependency.result.ok)
            ):
                return "DependencyFailed", (
                    f"required dependency at ordinal {dependency.ordinal} failed"
                )
        return None

    def _start_ready_locked(self) -> None:
        if self._aborted:
            return
        active = [tracked for tracked in self._calls if tracked.state == "running"]
        if self._journal_error is not None:
            return
        for tracked in sorted(self._calls, key=lambda item: item.ordinal):
            if tracked.state in {"completed", "running", "cancelling"}:
                continue
            if tracked.state != "queued" or tracked.prepared is None:
                continue
            if not self._all_lower_ordinals_known_and_prepared(tracked):
                continue
            if not self._can_enter_phase(tracked):
                continue
            if tracked.cancel_requested:
                tracked.state = "cancelling"
                tracked.execution_task = self._create_task(self._complete_cancelled(tracked))
                continue
            unresolved = [
                self._calls[index]
                for index in tracked.dependencies
                if not self._calls[index].done.is_set()
            ]
            if unresolved:
                continue
            dependency_failure = self._dependency_failure(tracked)
            if dependency_failure is not None:
                tracked.state = "running"
                tracked.execution_task = self._create_task(
                    self._complete_dependency_failure(tracked, *dependency_failure)
                )
                active.append(tracked)
                continue
            if len(active) >= self.executor.max_workers:
                continue
            if not self.executor.parallel_tools and active:
                continue
            tracked.state = "running"
            tracked.started_at = time.monotonic()
            tracked.execution_task = self._create_task(self._execute(tracked))
            active.append(tracked)

    async def _complete_dependency_failure(
        self, tracked: _TrackedCall, error_type: str, detail: str
    ) -> None:
        prepared = tracked.prepared
        call = prepared.tool_call if prepared is not None else tracked.original_call
        result = ToolResult(
            call.name,
            f"Tool skipped: {detail}",
            ok=False,
            metadata={"error_type": error_type},
        )
        await self._complete(tracked, result)

    async def _ensure_transaction(self, prepared: _PreparedCall) -> WorkspaceTransaction:
        async with self._transaction_lock:
            if self._transaction is None:
                workspace = self.executor.registry.workspace
                if workspace is None:
                    raise RuntimeError("workspace transaction requires a bound workspace")
                self._transaction = await asyncio.to_thread(
                    WorkspaceTransaction,
                    workspace,
                    self.turn_id,
                    journal=self.journal,
                )
            await asyncio.to_thread(
                self._transaction.declare_resources,
                prepared.spec.locks,
            )
            return self._transaction

    async def _execute(self, tracked: _TrackedCall) -> None:
        prepared = tracked.prepared
        if prepared is None:
            await self._complete_cancelled(tracked)
            return
        external_started = False
        try:
            transaction = (
                self._transaction
                if (
                    prepared.policy.safety is ExecutionSafety.SPECULATIVE_SAFE
                    and any(
                        self._safety(self._calls[index]) is ExecutionSafety.TRANSACTIONAL
                        for index in tracked.dependencies
                    )
                )
                else None
            )
            if prepared.policy.safety is ExecutionSafety.TRANSACTIONAL:
                transaction = await self._ensure_transaction(prepared)
            elif transaction is not None:
                async with self._transaction_lock:
                    await asyncio.to_thread(
                        transaction.declare_resources,
                        prepared.spec.locks,
                    )
            stable_key: str | None
            if prepared.policy.safety is ExecutionSafety.FINAL_ONLY:
                stable_key = prepared.policy.idempotency_key or hashlib.sha256(
                    (
                        f"{self.journal.storage.project_id}:{self.journal.storage.session_id}:"
                        f"{self.turn_id}:{tracked.ordinal}:{prepared.tool_call.id or ''}:"
                        f"{prepared.tool.name}"
                    ).encode("utf-8")
                ).hexdigest()
                await asyncio.to_thread(
                    self.journal.record,
                    RecoveryState.EXTERNAL_INTENT,
                    ordinal=tracked.ordinal,
                    tool=prepared.tool.name,
                    idempotency_key=stable_key,
                )
                external_started = True
            else:
                stable_key = prepared.policy.idempotency_key
            context = ToolExecutionContext(
                self.turn_id,
                workspace_view=transaction,
                provisional=prepared.policy.safety is not ExecutionSafety.FINAL_ONLY,
                execution_scope=self.execution_scope,
                idempotency_key=stable_key,
            )
            run_task = self._create_task(
                self.executor._run_tool(prepared, self._sync_semaphore, context)
            )
            tracked.underlying_task = run_task
            if context.provisional and not prepared.policy.safely_cancellable:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(run_task),
                        timeout=prepared.policy.execution_timeout,
                    )
                except TimeoutError:
                    cleanup_pending = type(prepared.tool).run is Tool.run
                    if not run_task.done():
                        run_task.cancel()
                        await asyncio.gather(run_task, return_exceptions=True)
                    tracked.cleanup_pending = cleanup_pending
                    result = ToolResult(
                        prepared.tool.name,
                        "Tool execution exceeded its provisional timeout; cleanup is supervised",
                        ok=False,
                        metadata={
                            "error_type": "IndeterminateToolExecution",
                            "execution_status": (
                                "cleanup_pending" if cleanup_pending else "cleanup_complete"
                            ),
                        },
                    )
                    await asyncio.to_thread(
                        self.journal.record,
                        RecoveryState.CLEANUP_REQUIRED if cleanup_pending else RecoveryState.CLEANUP_COMPLETE,
                        ordinal=tracked.ordinal,
                        tool=prepared.tool.name,
                        indeterminate=True,
                    )
            else:
                if self.execution_scope is None:
                    result = await run_task
                else:
                    result = await self.execution_scope.run_awaitable(run_task)
            if transaction is not None:
                result = replace(
                    result,
                    content=result.content.replace(str(transaction.overlay), str(transaction.workspace)),
                    metadata={
                        **result.metadata,
                        "overlay_observation": (
                            prepared.policy.safety is ExecutionSafety.SPECULATIVE_SAFE
                        ),
                    },
                )
                await asyncio.to_thread(
                    self.journal.record,
                    RecoveryState.STAGED,
                    ordinal=tracked.ordinal,
                    tool=prepared.tool.name,
                    ok=result.ok,
                )
            elif prepared.policy.safety is ExecutionSafety.FINAL_ONLY:
                recovery_payload = self._build_recovery_payload(tracked, result)
                await asyncio.to_thread(
                    self.journal.record,
                    RecoveryState.EXTERNAL_OUTCOME_COMMITTED,
                    ordinal=tracked.ordinal,
                    tool=prepared.tool.name,
                    ok=result.ok,
                    idempotency_key=stable_key,
                    tool_result=asdict(result),
                    history_payload=recovery_payload,
                )
        except JournalWriteError as exc:
            error_type = (
                "IndeterminateExternalEffect" if external_started else "JournalWriteFailed"
            )
            detail = (
                "external operation outcome is unknown because the journal failed"
                if external_started
                else f"execution journal unavailable: {exc}"
            )
            result = ToolResult(
                prepared.tool.name,
                f"Tool failed: {detail}",
                ok=False,
                metadata={"error_type": error_type},
            )
        except asyncio.CancelledError:
            result = self.executor._cancelled_result(prepared.tool.name)
        except Exception as exc:
            result = ToolResult(
                prepared.tool.name,
                (
                    "Tool outcome is indeterminate after external execution: "
                    if external_started
                    else "Tool error: "
                ) + str(exc),
                ok=False,
                metadata={
                    "error_type": (
                        "IndeterminateExternalEffect"
                        if external_started
                        else type(exc).__name__
                    )
                },
            )
        await self._complete(tracked, result)

    async def _complete_cancelled(self, tracked: _TrackedCall) -> None:
        if tracked.done.is_set():
            return
        call = tracked.prepared.tool_call if tracked.prepared is not None else tracked.original_call
        result = self.executor._cancelled_result(call.name)
        await self._complete(tracked, result)

    async def _complete(self, tracked: _TrackedCall, result: ToolResult) -> None:
        async with self._state_lock:
            if tracked.done.is_set():
                return
            tracked.result = result
            tracked.finished_at = time.monotonic()
            tracked.state = "completed"
            prepared = tracked.prepared
            task_id = str(result.metadata.get("task_id") or "")
            if prepared is not None and task_id and result.metadata.get("state") == "running":
                supervisor = getattr(getattr(prepared.tool, "session", None), "process_supervisor", None)
                try:
                    process_task = supervisor.get(task_id) if supervisor is not None else None
                except KeyError:
                    process_task = None
                done = getattr(process_task, "done", None)
                if isinstance(done, asyncio.Event) and not done.is_set():
                    result.metadata["resource_lease_running"] = True
                    self.executor._register_resource_lease(prepared.spec, done, task_id)
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
                tracked.execution_task = self._create_task(self._complete_cancelled(tracked))

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
                and prepared.policy.safely_cancellable
                and task is not None
            ):
                task.cancel()
        await self._wait_all()
        await self._rollback(reason)
        self.finished_at = time.monotonic()
        self._closed = True
        await self._write_metrics(abort_reason=reason)
        self.journal.close()

    async def finish(
        self,
        final_calls: list[ToolCall],
        *,
        messages: list[Message] | None = None,
        termination_proven: bool = True,
        termination_event: str | None = None,
    ) -> list[ToolResult]:
        """Reconcile the authoritative response and return results in its order."""
        if self._closed:
            raise RuntimeError("streaming tool batch is already closed")
        self.model_finished_at = time.monotonic()
        self.metrics["termination_proven"] = termination_proven
        self.metrics["termination_event"] = termination_event
        if not termination_proven:
            unproven_bindings = [
                (call, self._by_id.get(call.id) if call.id else None)
                for call in final_calls
            ]
            await self._invalidate_turn("provider termination was not proven")
            if not final_calls:
                self.finished_at = time.monotonic()
                self._closed = True
                await self._write_metrics(abort_reason="termination_unproven")
                await asyncio.to_thread(self.journal.close)
                return []
            return await self._protocol_results(
                final_calls,
                unproven_bindings,
                error_type="ProviderTerminationUnproven",
            )
        if not final_calls and not self._calls:
            self.finished_at = self.model_finished_at
            self._closed = True
            self.journal._release_for_later_recovery()
            return []
        bindings: list[tuple[ToolCall, _TrackedCall | None]] = []
        final_ids = [call.id for call in final_calls if call.id]
        if len(final_ids) != len(set(final_ids)):
            self._protocol_invalid = True

        for ordinal, call in enumerate(final_calls):
            tracked = self._by_id.get(call.id) if call.id else None
            if tracked is not None:
                ordinal_matches = not tracked.explicit_ordinal or tracked.ordinal == ordinal
                if self._same_call(tracked.original_call, call) and ordinal_matches:
                    bindings.append((call, tracked))
                else:
                    self._mismatched_events += 1
                    self._protocol_invalid = True
                    await self._cancel_not_started(tracked)
                    bindings.append((call, None))
                continue
            if ordinal in self._by_ordinal:
                self._protocol_invalid = True
                bindings.append((call, None))
                continue
            tracked = self._submit(
                call,
                messages=list(messages or self.base_messages),
                streamed=False,
                ordinal=ordinal,
            )
            bindings.append((call, tracked))

        for tracked in self._calls:
            if tracked.streamed and tracked.original_call.id not in set(final_ids):
                self._orphaned_events += 1
                self._protocol_invalid = True
                await self._cancel_not_started(tracked)

        if self._protocol_invalid:
            await self._invalidate_turn("authoritative response mismatch")
            return await self._protocol_results(final_calls, bindings)

        try:
            await asyncio.to_thread(
                self.journal.record, RecoveryState.TURN_VALIDATED, calls=len(final_calls)
            )
        except JournalWriteError as exc:
            self._journal_error = str(exc)
            await self._invalidate_turn("journal validation failure")
            return await self._protocol_results(final_calls, bindings, error_type="JournalWriteFailed")

        async with self._state_lock:
            self._turn_validated = True
            self._start_ready_locked()

        await self._wait_prepared()
        if self._journal_error is not None:
            await self._invalidate_turn("journal admission failure")
            return await self._protocol_results(
                final_calls, bindings, error_type="JournalWriteFailed"
            )

        await self._authorize_provisional(messages)
        if self._journal_error is not None:
            await self._invalidate_turn("journal authorization failure")
            return await self._protocol_results(
                final_calls, bindings, error_type="JournalWriteFailed"
            )

        try:
            await self._record_recovery_ready(final_calls, bindings)
        except JournalWriteError as exc:
            self._journal_error = str(exc)
            await self._invalidate_turn("recovery payload journal failure")
            return await self._protocol_results(
                final_calls, bindings, error_type="JournalWriteFailed"
            )

        async with self._state_lock:
            self._rebuild_dependencies_locked()
            transactional = [
                item for item in self._calls
                if self._safety(item) is ExecutionSafety.TRANSACTIONAL
            ]
            if transactional:
                ancestors = set()
                pending = [index for item in transactional for index in item.dependencies]
                while pending:
                    index = pending.pop()
                    if index in ancestors:
                        continue
                    ancestors.add(index)
                    pending.extend(self._calls[index].dependencies)
                self._precommit_final = {
                    index for index in ancestors
                    if self._safety(self._calls[index]) is ExecutionSafety.FINAL_ONLY
                }
            else:
                self._allow_final_only = True
            self._start_ready_locked()

        if transactional:
            await self._wait_precommit()
            # Refresh the recovery record with real staged outputs before the
            # workspace commit gate.  The earlier checkpoint protects final-only
            # external intents; this one prevents a post-commit recovery transcript
            # from containing placeholder ``RecoveryPending`` observations.
            try:
                await self._record_recovery_ready(final_calls, bindings)
            except JournalWriteError as exc:
                self._journal_error = str(exc)
                await self._invalidate_turn("precommit history journal failure")
                return await self._protocol_results(
                    final_calls, bindings, error_type="JournalWriteFailed"
                )
            transaction_failed = any(
                item.result is None or not item.result.ok for item in transactional
            )
            if transaction_failed:
                await self._rollback("transactional tool failure")
                await self._mark_rolled_back(self._rollback_affected(transactional))
                await self._skip_remaining_final("transaction rolled back")
            else:
                try:
                    if self._transaction is not None:
                        commit_task = self._create_task(asyncio.to_thread(self._transaction.commit))
                        try:
                            self._committed_paths = await asyncio.shield(commit_task)
                        except asyncio.CancelledError:
                            self._committed_paths = await commit_task
                            raise
                        self._commit_finished_at = time.monotonic()
                    for item in transactional:
                        if item.result is not None:
                            item.result.metadata["execution_status"] = "finalized"
                            item.result.metadata["committed"] = True
                    for item in self._calls:
                        if item.result is not None and item.result.metadata.get("overlay_observation"):
                            item.result.metadata["execution_status"] = "finalized"
                            item.result.metadata["overlay_observation"] = False
                    # Best-effort post-commit refresh narrows the already-safe window
                    # to the transcript append.  A durable precommit payload and the
                    # committed state are sufficient for startup recovery if this
                    # refresh itself is interrupted.
                    try:
                        await self._record_recovery_ready(final_calls, bindings)
                    except JournalWriteError as exc:
                        self._journal_error = str(exc)
                    async with self._state_lock:
                        self._allow_final_only = True
                        self._start_ready_locked()
                except Exception as exc:
                    await self._rollback(f"commit failed: {exc}")
                    await self._mark_rolled_back(
                        self._rollback_affected(transactional),
                        detail=str(exc),
                        error_type=(
                            "IndeterminateWorkspaceCommit"
                            if isinstance(exc, WorkspaceRecoveryRequired)
                            else "RolledBack"
                        ),
                    )
                    await self._skip_remaining_final("workspace commit failed")

        await self._wait_all()
        # Final-only tools may have completed after the commit checkpoint. Persist
        # their actual outcomes before PostToolUse/UI observation side effects.
        try:
            await self._record_recovery_ready(final_calls, bindings)
        except JournalWriteError as exc:
            self._journal_error = str(exc)
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
            if not tracked.observed:
                prepared = tracked.prepared
                if prepared is not None:
                    tracked.result = await self.executor._post_and_finish(
                        prepared, tracked.result
                    )
                else:
                    tracked.result = await self.executor._finish(
                        tracked.original_call,
                        tracked.result,
                        "streamed preparation failed",
                    )
                tracked.observed = True
            results.append(tracked.result)

        self.finished_at = time.monotonic()
        self._closed = True
        await self._write_metrics()
        await self._close_unmanaged_journal()
        return results

    async def _authorize_provisional(self, messages: list[Message] | None) -> None:
        """Finish deferred control paths and publish early-call admission once."""

        for tracked in sorted(self._calls, key=lambda item: item.ordinal):
            prepared = tracked.prepared
            if prepared is None or not prepared.provisional:
                continue
            if prepared.authorization_deferred:
                authorized = await self.executor._prepare(
                    tracked.index,
                    tracked.original_call,
                    list(messages or tracked.messages),
                    self.should_cancel,
                    self.execution_scope,
                )
                if isinstance(authorized, ToolResult):
                    tracked.observed = True
                    tracked.prepared = None
                    await self._complete(tracked, authorized)
                    continue
                tracked.prepared = authorized
                tracked.safety_hint = authorized.policy.safety
                prepared = authorized
            else:
                await self.executor._observe_provisional_start(prepared)
                prepared.provisional = False
            try:
                await asyncio.to_thread(
                    self.journal.record,
                    RecoveryState.AUTHORIZED,
                    ordinal=tracked.ordinal,
                    tool=prepared.tool.name,
                    safety=prepared.policy.safety.value,
                )
            except JournalWriteError as exc:
                self._journal_error = str(exc)

    async def _record_recovery_ready(
        self,
        final_calls: list[ToolCall],
        bindings: list[tuple[ToolCall, _TrackedCall | None]],
    ) -> None:
        """Durably store a redacted, truthful recovery round before any commit."""

        if self._history_context is None:
            return
        payload = self._build_recovery_payload()
        if payload is None:
            return
        await asyncio.to_thread(
            self.journal.record,
            RecoveryState.HISTORY_READY,
            history_payload=payload,
            precommit=True,
        )

    def _build_recovery_payload(
        self,
        current: _TrackedCall | None = None,
        current_result: ToolResult | None = None,
    ) -> dict[str, object] | None:
        """Build one ordered, durable history projection from known outcomes."""

        if self._history_context is None:
            return None
        assistant = self._history_context.get("assistant")
        if not isinstance(assistant, dict):
            return None
        parent_uuid = str(assistant.get("uuid") or "") or None
        round_id = str(assistant.get("round_id") or assistant.get("uuid") or "") or None
        tool_messages: list[dict[str, object]] = []
        complete = True
        for tracked in sorted(self._calls, key=lambda item: item.ordinal):
            call = tracked.original_call
            result = current_result if tracked is current else tracked.result
            if result is None:
                complete = False
                result = ToolResult(
                    call.name,
                    "Tool was not completed before crash recovery",
                    ok=False,
                    metadata={
                        "error_type": "RecoveryNotExecuted",
                        "execution_status": "not_executed",
                    },
                )
            message = Message(
                "tool",
                f"{result.name}: {result.content}",
                name=result.name,
                metadata={
                    **result.metadata,
                    "ok": result.ok,
                    "tool_call_id": call.id,
                    "ordinal": tracked.ordinal,
                },
                uuid=tracked.recovery_message_uuid,
                parent_uuid=parent_uuid,
                round_id=round_id,
            )
            parent_uuid = message.uuid
            tool_messages.append(message.to_dict())
        manifest = self.execution_manifest()
        if current is not None and current_result is not None:
            calls = manifest.get("calls")
            if isinstance(calls, list):
                for item in calls:
                    if isinstance(item, dict) and item.get("ordinal") == current.ordinal:
                        item["ok"] = current_result.ok
                        item["status"] = current_result.metadata.get(
                            "execution_status", "finalized"
                        )
        return {
            **self._history_context,
            "tool_results": tool_messages,
            "execution_manifest": manifest,
            "complete": complete,
        }

    async def _wait_prepared(self) -> None:
        while True:
            tasks = [item.preparation_task for item in self._calls if item.preparation_task]
            if tasks:
                await asyncio.gather(*tasks)
            if all(item.state != "preparing" for item in self._calls):
                return

    async def _wait_precommit(self) -> None:
        while True:
            pending = [
                item for item in self._calls
                if (
                    self._safety(item) is not ExecutionSafety.FINAL_ONLY
                    or item.index in self._precommit_final
                )
                and not item.done.is_set()
            ]
            if not pending:
                return
            await asyncio.gather(*(item.done.wait() for item in pending))

    async def _rollback(self, reason: str) -> None:
        if self._transaction is not None:
            try:
                await asyncio.to_thread(self._transaction.rollback, reason)
            except JournalWriteError:
                pass
        else:
            try:
                await asyncio.to_thread(self.journal.record, RecoveryState.ROLLED_BACK, reason=reason)
            except JournalWriteError:
                pass

    async def _mark_rolled_back(
        self,
        transactional: list[_TrackedCall],
        *,
        detail: str = "",
        error_type: str = "RolledBack",
    ) -> None:
        for item in transactional:
            if item.result is not None and item.result.ok:
                item.result = replace(
                    item.result,
                    content="Tool result was rolled back" + (f": {detail}" if detail else ""),
                    ok=False,
                    metadata={
                        **item.result.metadata,
                        "error_type": error_type,
                        "execution_status": (
                            "recovery_required"
                            if error_type == "IndeterminateWorkspaceCommit"
                            else "rolled_back"
                        ),
                    },
                )

    def _rollback_affected(self, transactional: list[_TrackedCall]) -> list[_TrackedCall]:
        affected = list(transactional)
        affected_indexes = {item.index for item in affected}
        affected.extend(
            item
            for item in self._calls
            if item.index not in affected_indexes
            and item.result is not None
            and bool(item.result.metadata.get("overlay_observation"))
        )
        return affected

    async def _skip_remaining_final(self, detail: str) -> None:
        for item in self._calls:
            if self._safety(item) is not ExecutionSafety.FINAL_ONLY or item.done.is_set():
                continue
            prepared = item.prepared
            call = prepared.tool_call if prepared is not None else item.original_call
            result = ToolResult(
                call.name,
                f"Tool skipped: {detail}",
                ok=False,
                metadata={"error_type": "TransactionRolledBack"},
            )
            await self._complete(item, result)

    async def _invalidate_turn(self, reason: str) -> None:
        self._aborted = True
        for tracked in self._calls:
            await self._cancel_not_started(tracked)
            prepared = tracked.prepared
            task = tracked.execution_task
            if (
                tracked.state == "running"
                and prepared is not None
                and type(prepared.tool).run is not Tool.run
                and prepared.policy.safely_cancellable
                and task is not None
            ):
                task.cancel()
        await self._wait_all()
        await self._rollback(reason)
        await self._mark_rolled_back(
            self._rollback_affected(
                [item for item in self._calls if self._safety(item) is ExecutionSafety.TRANSACTIONAL]
            )
        )

    async def _protocol_results(
        self,
        final_calls: list[ToolCall],
        bindings: list[tuple[ToolCall, _TrackedCall | None]],
        *,
        error_type: str = "StreamingToolProtocolMismatch",
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call, _tracked in bindings:
            result = ToolResult(
                call.name,
                "Tool skipped: streamed tool round did not match the authoritative response",
                ok=False,
                metadata={"error_type": error_type},
            )
            results.append(
                await self.executor._finish(
                    call,
                    result,
                    "protocol mismatch",
                    fire_failure_hook=error_type != "ProviderTerminationUnproven",
                )
            )
        self.finished_at = time.monotonic()
        self._closed = True
        await self._write_metrics(abort_reason="protocol_mismatch")
        await self._close_unmanaged_journal()
        return results

    async def _close_unmanaged_journal(self) -> None:
        """Finish standalone batches that have no transcript persistence owner."""

        if self._history_context is None:
            await asyncio.to_thread(self.journal.close)

    async def _wait_all(self) -> None:
        while True:
            snapshot = list(self._calls)
            if snapshot:
                await asyncio.gather(*(tracked.done.wait() for tracked in snapshot))
            if len(snapshot) == len(self._calls):
                return

    def execution_manifest(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "committed_paths": list(self._committed_paths),
            "calls": [
                {
                    "ordinal": item.ordinal,
                    "tool_call_id": item.original_call.id,
                    "tool": item.original_call.name,
                    "safety": self._safety(item).value,
                    "status": item.result.metadata.get("execution_status", "finalized")
                    if item.result is not None else "missing",
                    "ok": item.result.ok if item.result is not None else False,
                }
                for item in sorted(self._calls, key=lambda value: value.ordinal)
            ],
        }

    def recovery_message_uuid(self, ordinal: int) -> str | None:
        tracked = self._by_ordinal.get(ordinal)
        return tracked.recovery_message_uuid if tracked is not None else None

    def prime_history_payload(self, payload: dict[str, object]) -> None:
        """Supply transcript identity before :meth:`finish` reaches commit."""

        self._history_context = dict(payload)

    def record_history_payload(self, payload: dict[str, object]) -> None:
        self.journal.record(RecoveryState.HISTORY_READY, history_payload=payload)

    async def record_history_payload_async(self, payload: dict[str, object]) -> None:
        await asyncio.to_thread(
            self.journal.record, RecoveryState.HISTORY_READY, history_payload=payload
        )

    def mark_history_persisted(self) -> None:
        self.journal.record(RecoveryState.HISTORY_PERSISTED)
        self.journal.close()

    async def mark_history_persisted_async(self) -> None:
        await asyncio.to_thread(self.journal.record, RecoveryState.HISTORY_PERSISTED)
        await asyncio.to_thread(self.journal.close)

    async def _write_metrics(self, abort_reason: str | None = None) -> None:
        if not any(tracked.streamed for tracked in self._calls):
            return
        model_end = self.model_finished_at or time.monotonic()
        end = self.finished_at or time.monotonic()
        overlap_windows: list[float] = []
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
            overlap_windows.append(
                max(0.0, min(tracked_end, model_end) - tracked.started_at)
            )
        self.metrics = {
            "streamed_calls": sum(tracked.streamed for tracked in self._calls),
            "early_started": early_started,
            "post_stream_submitted": sum(not tracked.streamed for tracked in self._calls),
            "first_tool_start_ms": (
                round((first_start - self.created_at) * 1000, 3) if first_start is not None else None
            ),
            # This is one critical-path estimate, never the sum of concurrent work.
            "critical_path_saved_ms": round(max(overlap_windows, default=0.0) * 1000, 3),
            "model_termination_ms": round((model_end - self.created_at) * 1000, 3),
            "commit_ms": (
                round((self._commit_finished_at - self.created_at) * 1000, 3)
                if self._commit_finished_at is not None
                else None
            ),
            "calls": [
                {
                    "ordinal": tracked.ordinal,
                    "arguments_complete_ms": round(
                        (tracked.arguments_completed_at - self.created_at) * 1000, 3
                    ),
                    "tool_start_ms": (
                        round((tracked.started_at - self.created_at) * 1000, 3)
                        if tracked.started_at is not None
                        else None
                    ),
                }
                for tracked in sorted(self._calls, key=lambda item: item.ordinal)
            ],
            "post_stream_wait_ms": round(max(0.0, end - model_end) * 1000, 3),
            "duplicate_events": self._duplicate_events,
            "mismatched_events": self._mismatched_events,
            "orphaned_events": self._orphaned_events,
        }
        if abort_reason is not None:
            self.metrics["abort_reason"] = abort_reason
        if self.executor.logger is not None:
            await self.executor.logger.write("streaming_tool_batch", dict(self.metrics))
