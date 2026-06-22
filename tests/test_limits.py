from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent_core.cli import build_agent
from agent_core.config import resolve_limits_config
from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, ToolCall
from agent_core.providers.fake import FakeProvider
from agent_core.react import AgentRunResult, ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger


class LoopingToolProvider:
    """Always asks for the echo tool, so only the safety guards can stop the loop."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        return LLMResult(
            "Calling echo",
            tool_calls=[ToolCall("echo", {"text": "x"}, id=f"t{self.calls}")],
            stop_reason="tool_use",
        )


class LoopThenDoneProvider:
    """Loops on the echo tool for ``tool_turns`` turns, then returns a final answer."""

    def __init__(self, tool_turns: int = 1) -> None:
        self.calls = 0
        self.tool_turns = tool_turns

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls += 1
        if self.calls <= self.tool_turns:
            return LLMResult(
                "Calling echo",
                tool_calls=[ToolCall("echo", {"text": "x"}, id=f"t{self.calls}")],
                stop_reason="tool_use",
            )
        return LLMResult("done", stop_reason="end")


def _config(tmp_path: Path, **overrides) -> ReActConfig:
    base = dict(run_dir=str(tmp_path), permission="auto", memory=MemoryConfig(enabled=False))
    base.update(overrides)
    return ReActConfig(**base)


# --- run() guards -----------------------------------------------------------


async def test_run_stops_at_deadline(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(LoopingToolProvider(), _config(tmp_path), logger=logger)
    # A deadline already in the past trips the wall-clock guard on the first check.
    result = await agent.run("loop forever", deadline=time.monotonic() - 1)
    assert "wall-clock" in result.answer.lower()


async def test_run_stops_at_max_steps(tmp_path: Path) -> None:
    provider = LoopingToolProvider()
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(provider, _config(tmp_path, max_steps=2), logger=logger)
    result = await agent.run("loop forever")
    assert "max_steps" in result.answer
    assert provider.calls == 2
    assert result.steps == 2


async def test_disabled_wall_cap_leaves_no_deadline(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), _config(tmp_path, max_wall_seconds=None), logger=logger)
    result = await agent.run("hello")
    assert "Final answer" in result.answer
    assert agent._active_deadline is None


async def test_soft_deadline_nudge_injected_once(tmp_path: Path) -> None:
    # fraction 0.0 makes the soft threshold equal to the start, so the nudge fires on
    # the first turn while the (future) hard deadline never trips.
    provider = LoopThenDoneProvider(tool_turns=2)
    logger = JSONLRunLogger(tmp_path)
    config = _config(tmp_path, max_wall_seconds=1000.0, soft_deadline_fraction=0.0)
    agent = ReActAgent(provider, config, logger=logger)
    result = await agent.run("loop")
    nudges = [m for m in result.messages if m.metadata.get("deadline_wrapup")]
    assert len(nudges) == 1
    assert "out of time" in nudges[0].content.lower()


# --- shared sub-agent budget ------------------------------------------------


async def test_subagent_inherits_parent_deadline(tmp_path: Path) -> None:
    agent = ReActAgent(FakeProvider(), _config(tmp_path))
    sentinel = time.monotonic() + 500
    agent._active_deadline = sentinel
    recorded: dict[str, object] = {}

    class StubChild:
        async def run(self, task, should_cancel=None, deadline=None) -> AgentRunResult:
            recorded["task"] = task
            recorded["deadline"] = deadline
            return AgentRunResult("child done", [], 0, "run-x")

    agent._make_subagent_child = lambda preset, model=None: StubChild()  # type: ignore[assignment]
    out = await agent._spawn_subagent("subtask", "read_only")
    assert out == "child done"
    assert recorded["deadline"] == sentinel


# --- config precedence ------------------------------------------------------


def test_resolve_limits_defaults(tmp_path: Path) -> None:
    values = resolve_limits_config(tmp_path / "absent.toml")
    assert values == {"max_wall_seconds": 1800.0, "max_steps": None, "soft_deadline_fraction": 0.9}


def test_resolve_limits_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text("[limits]\nmax_wall_seconds = 600\nmax_steps = 20\nsoft_deadline_fraction = 0.8\n")
    values = resolve_limits_config(path)
    assert values["max_wall_seconds"] == 600.0
    assert values["max_steps"] == 20
    assert values["soft_deadline_fraction"] == 0.8


def test_resolve_limits_zero_disables(tmp_path: Path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text("[limits]\nmax_wall_seconds = 0\nmax_steps = 0\n")
    values = resolve_limits_config(path)
    assert values["max_wall_seconds"] is None
    assert values["max_steps"] is None


def test_resolve_limits_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "agent.toml"
    path.write_text("[limits]\nmax_wall_seconds = 600\nmax_steps = 20\n")
    monkeypatch.setenv("AGENT_MAX_WALL_SECONDS", "120")
    monkeypatch.setenv("AGENT_MAX_STEPS", "9")
    values = resolve_limits_config(path)
    assert values["max_wall_seconds"] == 120.0
    assert values["max_steps"] == 9


def _cli_args(**overrides) -> argparse.Namespace:
    base = dict(
        model=None,
        permission=None,
        provider="fake",
        memory=None,
        quiet=True,
        no_stream=False,
        thinking_budget=None,
        max_api_concurrency=None,
        max_wall_seconds=None,
        max_steps=None,
        no_session_persistence=True,
        session_dir=None,
        resume=None,
        continue_=False,
        fork_session=False,
        session_id=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_agent_cli_limits_win(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # no agent.toml here -> defaults, then CLI override
    built = build_agent(_cli_args(max_wall_seconds=42.0, max_steps=7))
    agent, mcp = built.agent, built.mcp
    try:
        assert agent.config.max_wall_seconds == 42.0
        assert agent.config.max_steps == 7
    finally:
        if mcp is not None:
            mcp.close()


def test_build_agent_cli_zero_disables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    built = build_agent(_cli_args(max_wall_seconds=0, max_steps=0))
    agent, mcp = built.agent, built.mcp
    try:
        assert agent.config.max_wall_seconds is None
        assert agent.config.max_steps is None
    finally:
        if mcp is not None:
            mcp.close()
