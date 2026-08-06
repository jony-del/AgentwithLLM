"""Measure model/tool overlap with a deterministic synthetic turn.

Run from the repository root:
    python benchmarks/streaming_tools.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.tools.base import ConcurrencySpec, Tool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry

CALL_READY_SECONDS = 0.05
MODEL_TAIL_SECONDS = 0.15
TOOL_SECONDS = 0.15


class _DelayTool(Tool):
    name = "benchmark_delay"
    description = "Synthetic async delay."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    async def run(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(TOOL_SECONDS)
        return ToolResult(self.name, "done")


def _executor() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(_DelayTool())
    return ToolExecutor(registry, PermissionPolicy(PermissionMode.DEFAULT))


async def _baseline() -> float:
    executor = _executor()
    call = ToolCall("benchmark_delay", {}, id="benchmark_call")
    started = time.perf_counter()
    await asyncio.sleep(CALL_READY_SECONDS + MODEL_TAIL_SECONDS)
    results = await executor.execute_many([call])
    if not results[0].ok:
        raise RuntimeError(results[0].content)
    return time.perf_counter() - started


async def _streaming() -> float:
    executor = _executor()
    call = ToolCall("benchmark_delay", {}, id="benchmark_call")
    batch = executor.begin_batch()
    started = time.perf_counter()
    await asyncio.sleep(CALL_READY_SECONDS)
    batch.submit_streamed(call)
    await asyncio.sleep(MODEL_TAIL_SECONDS)
    results = await batch.finish([call])
    if not results[0].ok:
        raise RuntimeError(results[0].content)
    return time.perf_counter() - started


async def main() -> None:
    baseline = [await _baseline() for _ in range(5)]
    streaming = [await _streaming() for _ in range(5)]
    old = statistics.median(baseline)
    new = statistics.median(streaming)
    saved = old - new
    print(f"baseline median: {old * 1000:.1f} ms")
    print(f"streaming median: {new * 1000:.1f} ms")
    print(f"saved: {saved * 1000:.1f} ms ({saved / old * 100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
