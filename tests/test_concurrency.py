"""Tests for concurrent LLM API calls and the shared provider gate."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from agent_core.agents.multi import MultiAgentCoordinator
from agent_core.models import LLMResult, ToolCall, ToolResult, ToolRisk
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.providers.base import GatedProvider, ProviderGate, _TokenBucket, gated_provider
from agent_core.storage import JSONLRunLogger
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry


class _AsyncTracker:
    """Provider whose ``complete`` records peak concurrency."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self._lock = asyncio.Lock()

    async def complete(self, messages, tools, config, stream=None) -> LLMResult:
        async with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            return LLMResult("ok", stop_reason="end")
        finally:
            async with self._lock:
                self.active -= 1


# --- gate: bounded overlap ---------------------------------------------------


def test_gate_caps_concurrency() -> None:
    inner = _AsyncTracker()
    provider = gated_provider(inner, max_concurrency=2)

    async def drive() -> None:
        await asyncio.gather(*(provider.complete([], [], {}) for _ in range(8)))

    asyncio.run(drive())
    assert inner.calls == 8
    assert inner.max_active == 2  # overlapped, but never beyond the cap


def test_gate_allows_overlap_under_default_cap() -> None:
    inner = _AsyncTracker()
    provider = gated_provider(inner)  # default cap 8

    async def drive() -> None:
        await asyncio.gather(*(provider.complete([], [], {}) for _ in range(4)))

    asyncio.run(drive())
    assert inner.max_active == 4  # all four ran at once


def test_gated_provider_is_idempotent() -> None:
    inner = _AsyncTracker()
    once = gated_provider(inner, max_concurrency=3)
    twice = gated_provider(once, max_concurrency=99)  # must reuse the same gate
    assert twice is once
    assert isinstance(once, GatedProvider)
    assert once.gate.max_concurrency == 3


# --- cancel-aware gate -------------------------------------------------------


def test_gate_refuses_to_start_after_cancel() -> None:
    inner = _AsyncTracker()
    provider = gated_provider(inner, max_concurrency=2)

    async def drive() -> None:
        with pytest.raises(asyncio.CancelledError):
            await provider.complete([], [], {}, should_cancel=lambda: True)

    asyncio.run(drive())
    assert inner.calls == 0  # the inner provider was never invoked


# --- rate limiter ------------------------------------------------------------


def test_token_bucket_spaces_requests_beyond_burst() -> None:
    # 600/min -> 10/sec, ~10-token burst. 15 acquisitions => 5 paced at 0.1s each.
    bucket = _TokenBucket(rate_per_min=600)

    async def drive() -> float:
        start = time.perf_counter()
        for _ in range(15):
            await bucket.acquire()
        return time.perf_counter() - start

    elapsed = asyncio.run(drive())
    assert elapsed >= 0.4  # the sustained rate held the extra requests back


def test_token_bucket_unlimited_when_zero() -> None:
    bucket = _TokenBucket(rate_per_min=0)

    async def drive() -> float:
        start = time.perf_counter()
        for _ in range(1000):
            await bucket.acquire()
        return time.perf_counter() - start

    assert asyncio.run(drive()) < 0.2


# --- partial failure isolation ----------------------------------------------


async def test_coordinator_run_all_isolates_failure() -> None:
    class Ok:
        name = "ok"

        async def run(self, task: str) -> str:
            await asyncio.sleep(0.01)
            return f"ok:{task}"

    class Bad:
        name = "bad"

        async def run(self, task: str) -> str:
            raise ValueError("nope")

    results = await MultiAgentCoordinator([Ok(), Bad()]).run_all("go")
    assert results["ok"] == "ok:go"
    assert "nope" in results["bad"]


# --- executor: async path keeps the sync-tool thread ceiling -----------------


class _ConcurrentReadTool(Tool):
    """A READ tool that records how many copies run at once (distinct keys => one wave)."""

    name = "concurrent_read"
    description = "sleep while recording concurrency"
    input_schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    risk = ToolRisk.READ
    _lock = threading.Lock()
    active = 0
    max_active = 0

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(arguments["key"]), "read"),))

    def _invoke(self, arguments: dict) -> ToolResult:
        with _ConcurrentReadTool._lock:
            _ConcurrentReadTool.active += 1
            _ConcurrentReadTool.max_active = max(_ConcurrentReadTool.max_active, _ConcurrentReadTool.active)
        try:
            time.sleep(0.05)
            return ToolResult(self.name, "done")
        finally:
            with _ConcurrentReadTool._lock:
                _ConcurrentReadTool.active -= 1


async def test_execute_many_respects_max_tool_workers() -> None:
    _ConcurrentReadTool.active = 0
    _ConcurrentReadTool.max_active = 0
    registry = ToolRegistry()
    registry.register(_ConcurrentReadTool())
    executor = ToolExecutor(registry, PermissionPolicy(PermissionMode.AUTO), max_workers=2)

    calls = [ToolCall("concurrent_read", {"key": f"k{i}"}) for i in range(6)]
    results = await executor.execute_many(calls)

    assert len(results) == 6
    assert all(r.ok for r in results)
    assert _ConcurrentReadTool.max_active <= 2  # the thread ceiling held


# --- logger: concurrent writes stay well-formed ------------------------------


async def test_logger_concurrent_writes_are_atomic(tmp_path) -> None:
    logger = JSONLRunLogger(run_dir=str(tmp_path), run_id="concurrent")

    async def writer(index: int) -> None:
        for n in range(50):
            await logger.write("event", {"i": index, "n": n})

    # Each write lands on a worker thread; the internal threading.Lock keeps the
    # overlapping appends atomic.
    await asyncio.gather(*(writer(i) for i in range(8)))

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 * 50
    for line in lines:  # every line is intact, well-formed JSON (no interleaving)
        json.loads(line)
