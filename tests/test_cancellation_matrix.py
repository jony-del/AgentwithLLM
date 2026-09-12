"""Consolidated cancellation & lifecycle matrix (SECOND_ROUND_INDEPENDENT_AUDIT §9.2).

Every scenario asserts, as applicable: (1) a response-time upper bound, (2) zero
unfinished asyncio tasks after the unwind, (3) zero residual supervised processes,
(4) the provider gate slot released, and (5) SessionEnd fired at least once — with
single-teardown semantics when the host fires it repeatedly.

Row → coverage map (rows already pinned elsewhere are referenced, not duplicated):

- provider connect/read hang → ``test_provider_read_cancellation_*`` below (a hung
  connect and a hung read are the same code path: the provider is parked inside
  ``provider_attempt`` when the token fires).
- 429 retry backoff (run level) → ``test_run_cancelled_during_429_backoff_*`` below;
  the provider-unit mechanics are pinned in tests/test_concurrency.py
  (``test_scope_aware_provider_releases_gate_during_backoff``,
  ``test_scope_aware_backoff_sleep_wakes_on_cancellation``), and the token-wake sleep
  itself in tests/test_contracts.py
  (``test_scope_sleep_wakes_early_on_token_cancellation``).
- MCP hang → tests/test_mcp.py (``test_tool_run_is_interrupted_by_scope_cancellation``
  for the scope-driven interrupt, ``test_close_with_hung_call_completes_bounded`` for
  close-during-hang, ``test_call_tool_timeout_cancels_future_and_marks_session_suspect``
  for the timeout path).
- HTTP / command hooks → tests/test_hook_adapters.py
  (``test_http_adapter_scope_cancellation_abandons_request``) and
  tests/test_p2_followup.py (``test_command_hook_timeout_kills_and_reaps_process_tree``
  / ``test_command_hook_cancellation_kills_and_reaps_process_tree`` — those carry the
  PID-level zero-residual-process assertions for spawned hook processes).
- ordinary sync (thread-offloaded) tool →
  ``test_run_cancelled_mid_sync_tool_unwinds_at_safe_point`` below.
- subagent/teammate → tests/test_limits.py
  (``test_scope_cancellation_propagates_to_subagent`` plus its residual-checking
  sibling ``test_subagent_cancellation_releases_gate_and_leaves_no_tasks``).
- transaction commit/rollback → the workspace commit is cancellation-shielded and
  re-awaited in ``StreamingToolBatch.finish``; crash-during-commit recovery is pinned
  by tests/test_streaming_tool_safety.py
  (``test_recovery_restores_partially_applied_commit``) and tests/test_crash_matrix.py.
- nested + repeated cancellation → the scope-level idempotency tests below, plus the
  ``session_end_fired`` single-teardown guards in tests/test_subagent.py.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from agent_core.execution import CancellationToken, ExecutionScope
from agent_core.hooks import HookContext, HookPipeline
from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, LLMTransientError, ToolCall, ToolResult, ToolRisk
from agent_core.providers.base import GatedProvider, ProviderGate, provider_attempt
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.base import ConcurrencySpec, Tool
from agent_core.tools.registry import ToolRegistry

# Generous hard ceiling for every cancelled run in this file: the real wake path is
# the scope's ~50ms cancellation poll, so finishing anywhere near this bound means
# the cancellation never landed at all.
RUN_BOUND = 5.0


class _SessionEndSpy:
    """Counts SessionEnd firings (audit row: SessionEnd fires at least once)."""

    def __init__(self) -> None:
        self.calls = 0

    async def on_session_end(self, ctx: HookContext) -> None:
        self.calls += 1
        # Host-driven session end outside a run still hands hooks a real scope.
        assert ctx.execution_scope is not None


def _config(tmp_path: Path, **overrides) -> ReActConfig:
    base = dict(
        run_dir=str(tmp_path),
        session_dir="",
        permission="auto",
        memory=MemoryConfig(enabled=False),
        project_instructions=False,
        git_context=False,
        max_api_concurrency=1,  # a single slot makes gate release observable
    )
    base.update(overrides)
    return ReActConfig(**base)


def _gate(agent: ReActAgent) -> ProviderGate:
    provider = agent.provider
    assert isinstance(provider, GatedProvider)
    return provider.gate


def _gate_semaphore(gate: ProviderGate) -> asyncio.Semaphore:
    semaphore = gate._semaphore
    assert semaphore is not None, "the run never reached the provider gate"
    return semaphore


async def _assert_gate_reacquirable(gate: ProviderGate) -> None:
    # The cancelled run scope would reject the probe, so acquire scope-less.
    async with gate.attempt(None):
        pass


def _unfinished_tasks(before: set[asyncio.Task]) -> list[asyncio.Task]:
    return [task for task in asyncio.all_tasks() - before if not task.done()]


async def _reap_run_task(run_task: asyncio.Task) -> None:
    """Failure-path hygiene: never leave a spawned run pending at test teardown."""
    if not run_task.done():
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


# --- provider connect/read hang ----------------------------------------------


class _HungReadProvider:
    """Scope-aware provider parked mid-request, like a hung connect or read.

    Holds the gate slot while blocked; the only exit is the run scope's cancellation
    token waking the bounded ``scope.sleep`` (the unified path first-party providers
    use for their in-flight request waits).
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.calls = 0
        self.cancel_observed = False

    async def complete(self, messages, tools, config, stream=None, should_cancel=None, scope=None) -> LLMResult:
        assert scope is not None
        self.calls += 1
        async with provider_attempt(scope):
            self.entered.set()
            try:
                await scope.sleep(60)
            except asyncio.CancelledError:
                self.cancel_observed = True
                raise
        return LLMResult("unreachable", stop_reason="end")


async def test_provider_read_cancellation_releases_gate_and_fires_session_end(tmp_path: Path) -> None:
    provider = _HungReadProvider()
    spy = _SessionEndSpy()
    agent = ReActAgent(provider, _config(tmp_path), hooks=HookPipeline(session_end_hooks=[spy]))
    gate = _gate(agent)
    token = CancellationToken()
    outer = ExecutionScope.for_workspace(tmp_path, cancellation=token)
    before = asyncio.all_tasks()
    started = time.monotonic()
    run_task = asyncio.create_task(agent.run("hang on a read", execution_scope=outer))
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=RUN_BOUND)
        # While the provider is parked, the fan-out's single gate slot is held.
        semaphore = _gate_semaphore(gate)
        assert semaphore.locked()
        token.cancel("user interrupt")
        result = await asyncio.wait_for(run_task, timeout=RUN_BOUND)
        # The 60s park is abandoned promptly once the token fires.
        assert time.monotonic() - started < 2.0
        assert "interrupt" in result.answer.lower()
        assert provider.cancel_observed
        # The gate slot was released on the way out and can be re-acquired.
        assert not semaphore.locked()
        await asyncio.wait_for(_assert_gate_reacquirable(gate), timeout=1)
        # No unfinished asyncio task leaked out of the run.
        assert _unfinished_tasks(before) == []
        # This scenario spawns no processes; the supervisor must be empty regardless.
        assert agent.session.process_supervisor.running() == []
    finally:
        await _reap_run_task(run_task)
        await outer.close()
    # SessionEnd fires at least once — and the single-teardown guard keeps it at one.
    await agent.fire_session_end("test_exit")
    await agent.fire_session_end("test_exit")
    assert spy.calls == 1


# --- 429 retry backoff --------------------------------------------------------


class _Retrying429Provider:
    """Scope-aware provider that 429s once, then parks in a long retry backoff.

    Mirrors the first-party retry loops (see ``ClaudeProvider``): the gate is
    acquired per physical attempt via ``provider_attempt(scope)`` and released
    while the provider backs off through ``scope.sleep``.
    """

    def __init__(self) -> None:
        self.in_backoff = asyncio.Event()
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None, scope=None) -> LLMResult:
        assert scope is not None
        try:
            async with provider_attempt(scope):
                self.calls += 1
                raise LLMTransientError("429 too many requests")
        except LLMTransientError:
            self.in_backoff.set()
            await scope.sleep(60.0)  # retry backoff — woken early by cancellation
        # The retry attempt; a cancelled scope rejects it before it can run.
        async with provider_attempt(scope):
            self.calls += 1
            return LLMResult("retried", stop_reason="end")


async def test_run_cancelled_during_429_backoff_stops_bounded_and_frees_gate(tmp_path: Path) -> None:
    provider = _Retrying429Provider()
    spy = _SessionEndSpy()
    agent = ReActAgent(provider, _config(tmp_path), hooks=HookPipeline(session_end_hooks=[spy]))
    gate = _gate(agent)
    token = CancellationToken()
    outer = ExecutionScope.for_workspace(tmp_path, cancellation=token)
    before = asyncio.all_tasks()
    started = time.monotonic()
    run_task = asyncio.create_task(agent.run("hit a 429", execution_scope=outer))
    try:
        await asyncio.wait_for(provider.in_backoff.wait(), timeout=RUN_BOUND)
        # The failed attempt already released the slot: it stays free during backoff.
        semaphore = _gate_semaphore(gate)
        assert not semaphore.locked()
        token.cancel("user interrupt")
        result = await asyncio.wait_for(run_task, timeout=RUN_BOUND)
        assert time.monotonic() - started < 2.0  # nowhere near the 60s backoff
        assert "interrupt" in result.answer.lower()
        assert provider.calls == 1  # the retry never happened
        assert not semaphore.locked()
        await asyncio.wait_for(_assert_gate_reacquirable(gate), timeout=1)
        assert _unfinished_tasks(before) == []
        assert agent.session.process_supervisor.running() == []
    finally:
        await _reap_run_task(run_task)
        await outer.close()
    await agent.fire_session_end("test_exit")
    assert spy.calls == 1


# --- ordinary sync (thread-offloaded) tool ------------------------------------


class _BlockingSyncTool(Tool):
    """A slow ordinary tool: its ``_invoke`` blocks a worker thread on an event."""

    name = "blocking_sync"
    description = "block until the test releases it"
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.READ

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = False

    def concurrency_spec(self, arguments: dict) -> ConcurrencySpec:
        return ConcurrencySpec()

    def _invoke(self, arguments: dict) -> ToolResult:
        self.started.set()
        # Safety net so a broken test can never hang the suite; the test releases
        # the event right after asserting the run is still parked in the join.
        if self.release.wait(timeout=30):
            self.completed = True
            return ToolResult(self.name, "slow result")
        return ToolResult(self.name, "abandoned by test", ok=False)


class _SlowToolProvider:
    """Requests the blocking sync tool once, then would answer "done"."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None, scope=None) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "calling the slow tool",
                tool_calls=[ToolCall("blocking_sync", {}, id="toolu_slow")],
                stop_reason="tool_use",
            )
        return LLMResult("done", stop_reason="end")


async def test_run_cancelled_mid_sync_tool_unwinds_at_safe_point(tmp_path: Path) -> None:
    """A worker-thread tool cannot be killed, so cancellation unwinds the run at the
    next safe point: promptly after the tool returns, the loop stops instead of
    feeding the tool result into another model turn."""
    tool = _BlockingSyncTool()
    registry = ToolRegistry()
    registry.register(tool)
    provider = _SlowToolProvider()
    spy = _SessionEndSpy()
    agent = ReActAgent(
        provider,
        _config(tmp_path),
        tools=registry,
        hooks=HookPipeline(session_end_hooks=[spy]),
    )
    gate = _gate(agent)
    token = CancellationToken()
    outer = ExecutionScope.for_workspace(tmp_path, cancellation=token)
    before = asyncio.all_tasks()
    run_task = asyncio.create_task(agent.run("run the slow tool", execution_scope=outer))
    try:
        # The tool body runs on a worker thread; wait until it is genuinely inside.
        tool_started = await asyncio.wait_for(
            asyncio.to_thread(tool.started.wait, RUN_BOUND), timeout=RUN_BOUND + 5
        )
        assert tool_started, "sync tool never started"
        token.cancel("user interrupt")
        # The running thread is not killable: the run stays parked in the tool join
        # instead of being torn out from under the tool (documented limitation; the
        # supervised-timeout variant is pinned in test_streaming_tool_safety.py).
        await asyncio.sleep(0.1)
        assert not run_task.done()
        tool.release.set()
        released_at = time.monotonic()
        result = await asyncio.wait_for(run_task, timeout=RUN_BOUND)
        # Once the thread returns, the loop converts to an interrupted result promptly.
        assert time.monotonic() - released_at < 2.0
        assert tool.completed  # the thread ran to completion — nothing was killed
        assert "interrupt" in result.answer.lower()
        # The tool result was discarded: no second model turn ever consumed it.
        assert provider.calls == 1
        assert "slow result" not in result.answer
        await asyncio.wait_for(_assert_gate_reacquirable(gate), timeout=1)
        assert _unfinished_tasks(before) == []
        assert agent.session.process_supervisor.running() == []
    finally:
        tool.release.set()  # idempotent; covers failures before the explicit release
        await _reap_run_task(run_task)
        await outer.close()
    await agent.fire_session_end("test_exit")
    assert spy.calls == 1


# --- nested + repeated cancellation idempotency --------------------------------


def test_token_double_cancel_stays_cancelled_without_error() -> None:
    token = CancellationToken()
    token.cancel("first")
    token.cancel("second")  # repeated cancellation is a no-op, not an error
    assert token.cancelled()
    # The cancellation surface is stable: every poll raises the same CancelledError.
    for _ in range(2):
        with pytest.raises(asyncio.CancelledError):
            token.raise_if_cancelled()
    assert token.cancelled()


async def test_cancelling_parent_scope_reaches_existing_children(tmp_path: Path) -> None:
    parent = ExecutionScope.for_workspace(tmp_path)
    child = parent.child()
    grandchild = child.child(deadline=time.monotonic() + 60)

    parent.cancellation.cancel("stop the tree")

    # One shared token: every nesting level observes the cancellation.
    for scope in (parent, child, grandchild):
        assert scope.cancelled()
        with pytest.raises(asyncio.CancelledError):
            scope.raise_if_cancelled()
    # And no new work can be scheduled anywhere in the cancelled tree.
    parked = asyncio.sleep(60)
    with pytest.raises(asyncio.CancelledError):
        child.create_task(parked)
    parked.close()  # rejected before scheduling — close the unawaited coroutine
    await parent.close()


async def test_scope_close_twice_reaps_once_and_stays_quiet(tmp_path: Path) -> None:
    scope = ExecutionScope.for_workspace(tmp_path)
    child = scope.child()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def worker() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    task = child.create_task(worker())  # children share the parent's task registry
    await asyncio.wait_for(started.wait(), timeout=1)
    await scope.close()
    assert scope.cancelled()
    assert task.done()  # the first close reaped the child's task...
    assert finished.is_set()  # ...running its teardown to completion
    await scope.close()  # repeated closes are quiet no-ops
    await scope.close()
    assert task.done()


async def test_repeated_cancellation_fires_session_end_once(tmp_path: Path) -> None:
    """Cancelling the run token twice mid-call surfaces a single interrupted result,
    and SessionEnd — the run tree's single teardown — fires exactly once even when
    the host calls it twice (the agent-level guard lives in ``fire_session_end``;
    the spawn-level one is pinned in tests/test_subagent.py)."""
    provider = _HungReadProvider()
    spy = _SessionEndSpy()
    agent = ReActAgent(provider, _config(tmp_path), hooks=HookPipeline(session_end_hooks=[spy]))
    token = CancellationToken()
    outer = ExecutionScope.for_workspace(tmp_path, cancellation=token)
    before = asyncio.all_tasks()
    run_task = asyncio.create_task(agent.run("hang", execution_scope=outer))
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=RUN_BOUND)
        token.cancel("first")
        token.cancel("second")  # repeated cancellation adds no new surface
        result = await asyncio.wait_for(run_task, timeout=RUN_BOUND)
        assert "interrupt" in result.answer.lower()
        assert provider.calls == 1
        assert _unfinished_tasks(before) == []
    finally:
        await _reap_run_task(run_task)
        await outer.close()
    await agent.fire_session_end("test_exit")
    await agent.fire_session_end("test_exit")
    assert spy.calls == 1
