"""Tests for the async tool-use progress label (UI-only, ephemeral).

Two layers:
- the seam (``build_tool_use_summarizer`` / ``render_tool_batch`` / ``clean_label``):
  no-tools streamed bounded call, single timeout, silent degrade, offline byte-stability;
- the ReAct loop wiring: fire-after-batch / flush-next-turn, live-UI + leader gating, and
  the invariant that the label reaches the UI + event log but NEVER the API messages or the
  resumable transcript.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from agent_core.models import LLMResult, Message, ToolCall, ToolResult
from agent_core.providers.base import LLMProvider, gated_provider
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger
from agent_core.tool_use_summary import (
    ToolUseSummaryConfig,
    build_tool_use_summarizer,
    clean_label,
    render_tool_batch,
)
from agent_core.ui import AgentUI


# --- stub providers ----------------------------------------------------------


class _RecordingProvider(LLMProvider):
    """Non-fake provider that records ``complete`` calls and returns a canned label."""

    def __init__(self, content: str = "Echoed hello", sleep: float = 0.0) -> None:
        self.content = content
        self.sleep = sleep
        self.calls: list[tuple[list[Message], list[dict[str, Any]], dict[str, Any]]] = []
        self.streams: list[Any] = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        self.calls.append((messages, tools, config))
        self.streams.append(stream)
        if self.sleep:
            await asyncio.sleep(self.sleep)
        return LLMResult(content=self.content, stop_reason="end_turn")


def _batch() -> list[tuple[ToolCall, ToolResult]]:
    return [(ToolCall("echo", {"text": "hi"}, id="t1"), ToolResult(name="echo", content="hi", ok=True))]


# --- seam: build_tool_use_summarizer gating ----------------------------------


def test_summarizer_is_none_when_disabled() -> None:
    assert build_tool_use_summarizer(_RecordingProvider(), {}, ToolUseSummaryConfig(enabled=False)) is None


def test_summarizer_is_none_for_fake_provider() -> None:
    cfg = ToolUseSummaryConfig(enabled=True)
    assert build_tool_use_summarizer(FakeProvider(), {}, cfg) is None
    # Detected even through the shared concurrency gate.
    assert build_tool_use_summarizer(gated_provider(FakeProvider()), {}, cfg) is None


# --- seam: the call shape ----------------------------------------------------


async def test_summarizer_issues_no_tools_streamed_bounded_call() -> None:
    provider = _RecordingProvider(content="Echoed hello")
    summarizer = build_tool_use_summarizer(
        provider,
        {"model": "claude-opus-4-8", "max_tokens": 4096, "stream": False, "thinking_budget": 4096},
        ToolUseSummaryConfig(enabled=True, model="claude-haiku-4-5-20251001", max_tokens=64),
    )
    assert summarizer is not None

    label = await summarizer(_batch(), "I will echo hi")

    assert label == "Echoed hello"
    (messages, tools, config), = provider.calls
    assert tools == []  # no tool use while writing a label
    assert config["model"] == "claude-haiku-4-5-20251001"  # forced to the cheap model
    assert config["max_tokens"] == 64  # tiny budget — labels are short
    assert config["stream"] is True
    assert config["thinking_budget"] is None
    assert provider.streams[0] is not None  # a (no-op) sink is passed
    # The transcript is framed as untrusted data, bounded in delimiters.
    assert "<tool_batch>" in messages[1].content


async def test_summarizer_empty_batch_makes_no_call() -> None:
    provider = _RecordingProvider()
    summarizer = build_tool_use_summarizer(provider, {}, ToolUseSummaryConfig(enabled=True))
    assert summarizer is not None
    assert await summarizer([], "context") is None
    assert provider.calls == []


# --- seam: silent degrade ----------------------------------------------------


async def test_summarizer_times_out_to_none() -> None:
    provider = _RecordingProvider(sleep=0.5)
    summarizer = build_tool_use_summarizer(
        provider, {}, ToolUseSummaryConfig(enabled=True, timeout_seconds=0.05)
    )
    assert summarizer is not None
    assert await summarizer(_batch(), "ctx") is None  # timed out → no label, no raise


async def test_summarizer_swallows_provider_error() -> None:
    class _Boom(LLMProvider):
        async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
            raise RuntimeError("provider down")

    summarizer = build_tool_use_summarizer(_Boom(), {}, ToolUseSummaryConfig(enabled=True))
    assert summarizer is not None
    assert await summarizer(_batch(), "ctx") is None


async def test_summarizer_empty_reply_is_none() -> None:
    provider = _RecordingProvider(content="   \n  ")
    summarizer = build_tool_use_summarizer(provider, {}, ToolUseSummaryConfig(enabled=True))
    assert summarizer is not None
    assert await summarizer(_batch(), "ctx") is None


# --- pure helpers ------------------------------------------------------------


def test_render_tool_batch_truncates_per_tool() -> None:
    big = "x" * 5000
    batch = [(ToolCall("read", {"path": big}, id="t1"), ToolResult(name="read", content=big, ok=True))]
    rendered = render_tool_batch(batch, "doing things", max_chars_per_tool=300)
    assert "<tool_batch>" in rendered and "</tool_batch>" in rendered
    assert "…" in rendered  # truncation marker present
    assert rendered.count("x" * 301) == 0  # no field exceeds the per-tool cap


def test_clean_label_takes_first_line_and_strips() -> None:
    assert clean_label('"Fixed the bug."\nextra noise') == "Fixed the bug"
    assert clean_label("") is None
    assert clean_label("   \n\n  ") is None
    assert clean_label("x" * 500).endswith("…")  # hard-capped


# --- loop wiring -------------------------------------------------------------


class _SpyUI(AgentUI):
    """Live UI that records progress labels (and auto-allows tools)."""

    is_live = True

    def __init__(self) -> None:
        self.labels: list[tuple[str, list[str]]] = []

    def on_tool_use_summary(self, label: str, tool_names: list[str]) -> None:
        self.labels.append((label, tool_names))

    def confirm_tool(self, tool_name, risk, arguments):
        return "always"


class _LabelingToolProvider(LLMProvider):
    """Drives a 2-turn echo loop and also answers the label sub-call.

    The label call is distinguished by an empty tool list + the label system prompt; the
    main loop calls ``echo`` once then finishes.
    """

    def __init__(self) -> None:
        self.main_calls = 0
        self.label_calls = 0

    async def complete(self, messages, tools, config, stream=None, should_cancel=None) -> LLMResult:
        if tools == [] and messages and "progress label" in messages[0].content:
            self.label_calls += 1
            return LLMResult("Echoed hi", stop_reason="end_turn")
        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResult(
                "Calling echo",
                tool_calls=[ToolCall("echo", {"text": "hi"}, id="t1")],
                stop_reason="tool_use",
            )
        return LLMResult("done", stop_reason="end")


def _agent(provider, ui, tmp_path: Path, **summary_kwargs) -> ReActAgent:
    config = ReActConfig(
        run_dir=str(tmp_path),
        permission="auto",
        session_dir="",  # no transcript persistence unless a test opts in
        tool_use_summary=ToolUseSummaryConfig(enabled=True, **summary_kwargs),
    )
    return ReActAgent(provider, config, logger=JSONLRunLogger(tmp_path), ui=ui)


async def test_label_emitted_to_ui_and_log_not_to_api(tmp_path: Path) -> None:
    provider = _LabelingToolProvider()
    ui = _SpyUI()
    agent = _agent(provider, ui, tmp_path)

    result = await agent.run("please echo hi")

    # Emitted once to the live UI with the batch's tool names.
    assert ui.labels == [("Echoed hi", ["echo"])]
    assert provider.label_calls == 1
    # NEVER injected into the API conversation.
    assert all("Echoed hi" not in m.content for m in result.messages if m.role != "assistant")
    assert all(m.metadata.get("tool_use_summary") is None for m in result.messages)
    # Written to the runs event log (observability), as a distinct event.
    events = [json.loads(line) for line in agent.logger.path.read_text("utf-8").splitlines()]
    summary_events = [e for e in events if e.get("event") == "tool_use_summary"]
    assert len(summary_events) == 1
    assert summary_events[0]["label"] == "Echoed hi"
    assert summary_events[0]["tools"] == ["echo"]


async def test_label_not_persisted_to_transcript(tmp_path: Path) -> None:
    provider = _LabelingToolProvider()
    ui = _SpyUI()
    config = ReActConfig(
        run_dir=str(tmp_path),
        permission="auto",
        session_dir=str(tmp_path / "sessions"),  # transcript ON
        tool_use_summary=ToolUseSummaryConfig(enabled=True),
    )
    agent = ReActAgent(provider, config, logger=JSONLRunLogger(tmp_path), ui=ui)

    await agent.run("please echo hi")

    assert ui.labels  # label was emitted to the UI
    # The label text appears nowhere in the persisted transcript files.
    blobs = [p.read_text("utf-8") for p in (tmp_path / "sessions").rglob("*.jsonl")]
    assert blobs  # a transcript was written
    assert all("Echoed hi" not in blob for blob in blobs)


async def test_no_label_when_ui_not_live(tmp_path: Path) -> None:
    # NullUI is not live → no one to show a label to → don't even call the model.
    provider = _LabelingToolProvider()
    agent = _agent(provider, AgentUI(), tmp_path)  # base AgentUI: is_live=False

    await agent.run("please echo hi")

    assert provider.label_calls == 0


async def test_no_label_for_subagent_depth(tmp_path: Path) -> None:
    provider = _RecordingProvider()
    ui = _SpyUI()
    agent = _agent(provider, ui, tmp_path)  # include_subagents defaults False
    agent.session.depth = 1  # simulate a sub-agent

    agent._fire_tool_use_summary(_batch(), "ctx")
    assert agent._pending_tool_use_summary is None  # gated out

    agent.session.depth = 0
    agent._fire_tool_use_summary(_batch(), "ctx")
    assert agent._pending_tool_use_summary is not None
    await agent._flush_pending_tool_use_summary()  # reap the task


async def test_cancel_pending_label_clears_without_emit(tmp_path: Path) -> None:
    provider = _RecordingProvider(sleep=1.0)
    ui = _SpyUI()
    agent = _agent(provider, ui, tmp_path)

    agent._fire_tool_use_summary(_batch(), "ctx")
    assert agent._pending_tool_use_summary is not None
    await agent._cancel_pending_tool_use_summary()

    assert agent._pending_tool_use_summary is None
    assert agent._pending_tool_use_names == []
    assert ui.labels == []  # cancelled → never emitted


async def test_fake_provider_run_fires_no_label(tmp_path: Path) -> None:
    # Offline byte-stability: FakeProvider → summarizer is None → no task, no label call.
    ui = _SpyUI()
    agent = _agent(FakeProvider(), ui, tmp_path)
    assert agent._tool_use_summarizer is None

    await agent.run("please use tool: echo")

    assert ui.labels == []
