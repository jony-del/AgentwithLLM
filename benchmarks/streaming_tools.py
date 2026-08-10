"""End-to-end synthetic benchmark for authoritative streaming tool execution.

The clock starts when model sampling starts and stops when ordered tool results are
ready.  Provider rows exercise capability negotiation; protocol-event correctness is
covered by the provider fixture tests.

Run from the repository root:
    python benchmarks/streaming_tools.py --runs 9
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agent_core.hooks import HookOutcome, HookPipeline
from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.permission_types import PermissionResult
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import ToolStreamBoundary
from agent_core.tools.base import (
    ConcurrencySpec,
    ExecutionSafety,
    Tool,
    WorkspacePathMixin,
)
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry

CALL_READY_SECONDS = 0.02
MODEL_TAIL_SECONDS = 0.08
TOOL_SECONDS = 0.075


class _DelayReadTool(Tool):
    name = "benchmark_read"
    description = "Synthetic async read delay."
    input_schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    risk = ToolRisk.READ
    execution_safety = ExecutionSafety.SPECULATIVE_SAFE
    safely_cancellable = True

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    async def run(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(TOOL_SECONDS)
        return ToolResult(self.name, str(arguments.get("key") or "done"))


class _DelayWriteTool(WorkspacePathMixin, Tool):
    name = "benchmark_write"
    description = "Synthetic path-scoped transactional write delay."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    risk = ToolRisk.WRITE
    accept_edits_safe = True
    execution_safety = ExecutionSafety.TRANSACTIONAL
    transaction_backend = "workspace"

    async def check_permissions(self, arguments: dict, context) -> PermissionResult:
        return PermissionResult.allow("benchmark fixture")

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec((self.workspace_lock(arguments["path"], "write"),))

    def _invoke(self, arguments: dict) -> ToolResult:
        time.sleep(TOOL_SECONDS)
        target = self.resolve_workspace_path(arguments["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(arguments["content"]), encoding="utf-8")
        return ToolResult(self.name, "staged")


class _AllowingPreHook:
    async def on_pre_tool(self, _context) -> HookOutcome:
        return HookOutcome()


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    boundary: ToolStreamBoundary = "explicit"
    calls: int = 1
    transactional: bool = False
    external_pre_hook: bool = False


SCENARIOS = (
    _Scenario("claude/explicit-read"),
    _Scenario("openai-responses/explicit-read"),
    _Scenario("openai-compatible/terminal-only", boundary="terminal_only"),
    _Scenario("fake/explicit-read"),
    _Scenario("scheduler/transactional-write", transactional=True),
    _Scenario("scheduler/multi-read", calls=3),
    _Scenario("scheduler/deferred-pre-hook", external_pre_hook=True),
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _executor(workspace: Path, scenario: _Scenario) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(_DelayWriteTool(workspace) if scenario.transactional else _DelayReadTool())
    registry.rebind_workspace(str(workspace))
    hooks = HookPipeline(
        external_pre_tool_hooks=[_AllowingPreHook()]
        if scenario.external_pre_hook
        else []
    )
    return ToolExecutor(
        registry,
        PermissionPolicy(PermissionMode.ACCEPTEDITS),
        hooks=hooks,
        journal_dir=workspace / ".journals",
    )


def _calls(scenario: _Scenario) -> list[ToolCall]:
    if scenario.transactional:
        return [
            ToolCall(
                "benchmark_write",
                {"path": "result.txt", "content": "new"},
                id="call_0",
            )
        ]
    return [
        ToolCall("benchmark_read", {"key": str(index)}, id=f"call_{index}")
        for index in range(scenario.calls)
    ]


async def _trial(scenario: _Scenario, *, early_enabled: bool) -> float:
    with tempfile.TemporaryDirectory(prefix="polaris-stream-bench-") as raw_workspace:
        workspace = Path(raw_workspace)
        executor = _executor(workspace, scenario)
        calls = _calls(scenario)
        batch = executor.begin_batch()
        started = time.perf_counter()
        await asyncio.sleep(CALL_READY_SECONDS)
        if early_enabled and scenario.boundary == "explicit":
            for ordinal, call in enumerate(calls):
                batch.submit_streamed(call, ordinal=ordinal)
        await asyncio.sleep(MODEL_TAIL_SECONDS)
        results = await batch.finish(
            calls, termination_proven=True, termination_event="benchmark.complete"
        )
        if not all(result.ok for result in results):
            raise RuntimeError([result.content for result in results])
        return time.perf_counter() - started


async def main(runs: int) -> None:
    print("scenario                                  median off/on    p95 off/on      median saved")
    print("-" * 96)
    for scenario in SCENARIOS:
        disabled = [await _trial(scenario, early_enabled=False) for _ in range(runs)]
        enabled = [await _trial(scenario, early_enabled=True) for _ in range(runs)]
        off_median = statistics.median(disabled)
        on_median = statistics.median(enabled)
        off_p95 = _percentile(disabled, 0.95)
        on_p95 = _percentile(enabled, 0.95)
        saved = max(0.0, off_median - on_median)
        percentage = saved / off_median * 100 if off_median else 0.0
        print(
            f"{scenario.name:<40} "
            f"{off_median * 1000:6.1f}/{on_median * 1000:6.1f} ms  "
            f"{off_p95 * 1000:6.1f}/{on_p95 * 1000:6.1f} ms  "
            f"{saved * 1000:6.1f} ms ({percentage:4.1f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=9, help="samples per mode and scenario")
    arguments = parser.parse_args()
    asyncio.run(main(max(1, arguments.runs)))
