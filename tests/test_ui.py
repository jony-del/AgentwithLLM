from pathlib import Path
from typing import Any

from agent_core.memory import MemoryConfig
from agent_core.models import Message, ToolResult
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import JSONLRunLogger
from agent_core.ui import ConsoleUI, NullUI


def _config(tmp_path: Path) -> ReActConfig:
    # Memory off: keeps the run deterministic (no post-run extraction call) and
    # avoids writing to the real ./memory directory during tests.
    return ReActConfig(run_dir=str(tmp_path), memory=MemoryConfig(enabled=False))


class RecordingUI(NullUI):
    """Captures the (event, payload) sequence the agent emits, in order."""

    is_live = True  # so the agent wires an interactive prompter (always-allow here)

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_thinking(self, text: str) -> None:
        self.events.append(("thinking", text))

    def on_reasoning(self, text: str) -> None:
        self.events.append(("reasoning", text))

    def on_tool_call(self, tool_name: str, risk: str, arguments: dict) -> None:
        self.events.append(("tool_call", tool_name))

    def on_tool_result(self, result: ToolResult) -> None:
        self.events.append(("tool_result", result.name))

    def on_final(self, answer: str) -> None:
        self.events.append(("final", answer))

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict) -> str:
        return "always"


def test_default_ui_is_null_and_silent(tmp_path: Path) -> None:
    agent = ReActAgent(FakeProvider(), _config(tmp_path))
    assert isinstance(agent.ui, NullUI)
    # A NullUI run still works end-to-end and emits nothing observable.
    result = agent.run("hello")
    assert "Final answer" in result.answer


def test_ui_emits_events_in_order(tmp_path: Path) -> None:
    ui = RecordingUI()
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), _config(tmp_path), logger=logger, ui=ui)

    agent.run("please use tool: echo")

    kinds = [kind for kind, _ in ui.events]
    assert kinds == ["reasoning", "tool_call", "tool_result", "final"]
    assert ui.events[1] == ("tool_call", "echo")
    assert ui.events[-1][0] == "final"


def test_always_allow_via_ui_grants_write_permission(tmp_path: Path) -> None:
    # The RecordingUI answers "always"; default permission mode would otherwise ask
    # for the write tool. The run completes (tool not denied), proving the prompter
    # path is exercised and grants permission.
    ui = RecordingUI()
    agent = ReActAgent(FakeProvider(), _config(tmp_path), ui=ui)
    result = agent.run("please use tool: write_text_file")
    assert "observation" in result.answer.lower()


# --- streaming ---------------------------------------------------------------


class _RecordingStream:
    def __init__(self) -> None:
        self.text: list[str] = []

    def on_text_delta(self, text: str) -> None:
        self.text.append(text)

    def on_thinking_delta(self, text: str) -> None:
        pass

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        pass


def test_fake_provider_streams_chunks_matching_content() -> None:
    sink = _RecordingStream()
    provider = FakeProvider()
    result = provider.complete([Message("user", "say hi please")], [], {}, stream=sink)
    # The streamed chunks reassemble exactly into the returned content.
    assert "".join(sink.text) == result.content
    assert len(sink.text) > 1  # actually chunked, not one blob


def test_console_ui_streams_then_finalizes_without_duplicate(capsys) -> None:
    ui = ConsoleUI(color=False)
    ui.on_turn_start()
    ui.on_text_delta("he")
    ui.on_text_delta("llo")
    ui.on_final("hello")  # finalizer: must NOT reprint the already-streamed text
    out = capsys.readouterr().out
    assert out.count("hello") == 1


def test_console_ui_finalizer_prints_full_when_not_streamed(capsys) -> None:
    ui = ConsoleUI(color=False)
    ui.on_turn_start()
    ui.on_final("the answer")  # no deltas this turn -> print the whole thing
    out = capsys.readouterr().out
    assert "the answer" in out
