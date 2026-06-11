from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import PurePath

from agent_core.hooks import HookPipeline
from agent_core.models import ToolCall, ToolResult
from agent_core.permissions import PermissionPolicy
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
        *,
        parallel_tools: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks or HookPipeline()
        self.logger = logger
        self.ui = ui or NullUI()
        self.parallel_tools = parallel_tools
        self.max_workers = max(1, int(max_workers))

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        should_cancel: Callable[[], bool] | None = None,
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
            return await self._execute_sequential(tool_calls, should_cancel)

        sync_semaphore = asyncio.Semaphore(self.max_workers)
        results: list[ToolResult | None] = [None] * len(tool_calls)
        runnable: list[_PreparedCall] = []
        for index, tool_call in enumerate(tool_calls):
            if should_cancel is not None and should_cancel():
                results[index] = await self._finish(tool_call, self._cancelled_result(tool_call.name), "cancelled")
                continue
            prepared = await self._prepare(index, tool_call)
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
    ) -> list[ToolResult]:
        """No concurrency requested: await each call one at a time, in order."""
        results: list[ToolResult] = []
        for index, tool_call in enumerate(tool_calls):
            if should_cancel is not None and should_cancel():
                results.append(await self._finish(tool_call, self._cancelled_result(tool_call.name), "cancelled"))
                continue
            prepared = await self._prepare(index, tool_call)
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

    async def _prepare(self, index: int, tool_call: ToolCall) -> _PreparedCall | ToolResult:
        rewritten_call, pre_results = self.hooks.run_pre(tool_call)
        if self.logger:
            await self.logger.write(
                "tool_pre",
                {
                    "tool_call": asdict(rewritten_call),
                    "pre_results": [asdict(result) for result in pre_results],
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

        self.ui.on_tool_call(tool.name, tool.risk.value, rewritten_call.arguments)
        decision = self.permissions.decide(tool)
        # The confirm step may block on an interactive prompt (input()); run it on a
        # worker thread so a question to the user doesn't freeze other in-flight work.
        decision = await asyncio.to_thread(self.permissions.confirm, decision, tool, rewritten_call)
        if self.logger:
            await self.logger.write("permission", {"tool": tool.name, "decision": asdict(decision)})
        if not decision.allowed:
            result = ToolResult(tool.name, f"Tool denied: {decision.reason}", ok=False)
            return await self._finish(rewritten_call, result, decision.reason)
        if decision.dry_run:
            result = ToolResult(tool.name, f"Dry-run: would execute {tool.name} with {rewritten_call.arguments}")
            return await self._finish(rewritten_call, result, decision.reason)

        try:
            spec = tool.concurrency_spec(rewritten_call.arguments)
        except Exception as exc:
            result = ToolResult(tool.name, f"Tool error: {exc}", ok=False, metadata={"error_type": type(exc).__name__})
            return await self._finish(rewritten_call, result, decision.reason)
        return _PreparedCall(index, rewritten_call, tool, spec, decision.reason)

    async def _post_and_finish(self, prepared: _PreparedCall, result: ToolResult) -> ToolResult:
        result = self.hooks.run_post(prepared.tool_call, result)
        return await self._finish(prepared.tool_call, result, prepared.reason)

    async def _finish(self, tool_call: ToolCall, result: ToolResult, reason: str | None) -> ToolResult:
        """Log, surface the observation to the UI, and return one exit for every path."""
        await self._log_result(tool_call, result, reason)
        self.ui.on_tool_result(result)
        return result

    async def _log_result(self, tool_call: ToolCall, result: ToolResult, reason: str | None) -> None:
        if self.logger:
            await self.logger.write(
                "tool_result",
                {"tool_call": asdict(tool_call), "result": asdict(result), "reason": reason},
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
