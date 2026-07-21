from pathlib import Path
from typing import Any

from agent_core.compression import CompressionConfig
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

    def on_tool_call(self, tool_name: str, risk: str, arguments: dict, label: str | None = None) -> None:
        self.events.append(("tool_call", tool_name))

    def on_tool_result(self, result: ToolResult, diff: str | None = None) -> None:
        self.events.append(("tool_result", result.name))

    def on_final(self, answer: str) -> None:
        self.events.append(("final", answer))

    def on_compaction_start(self, reactive: bool) -> None:
        self.events.append(("compaction_start", reactive))

    def on_compaction_end(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        self.events.append(("compaction_end", reactive))

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict) -> str:
        return "always"


async def test_default_ui_is_null_and_silent(tmp_path: Path) -> None:
    agent = ReActAgent(FakeProvider(), _config(tmp_path))
    assert isinstance(agent.ui, NullUI)
    # A NullUI run still works end-to-end and emits nothing observable.
    result = await agent.run("hello")
    assert "Final answer" in result.answer


async def test_ui_emits_events_in_order(tmp_path: Path) -> None:
    ui = RecordingUI()
    logger = JSONLRunLogger(tmp_path)
    agent = ReActAgent(FakeProvider(), _config(tmp_path), logger=logger, ui=ui)

    await agent.run("please use tool: echo")

    kinds = [kind for kind, _ in ui.events]
    assert kinds == ["reasoning", "tool_call", "tool_result", "final"]
    assert ui.events[1] == ("tool_call", "echo")
    assert ui.events[-1][0] == "final"


async def test_compaction_emits_start_and_end(tmp_path: Path) -> None:
    # A tiny token window + tiny per-message char budget forces auto_compact to fire on
    # the first loop step (the long system prompt is microcompacted), so the UI must see
    # a start/end pair (the bar's bracketing events).
    ui = RecordingUI()
    config = ReActConfig(
        run_dir=str(tmp_path),
        memory=MemoryConfig(enabled=False),
        compression=CompressionConfig(
            context_window_tokens=200,
            autocompact_buffer_tokens=10,
            reserved_output_tokens_for_summary=10,
            max_message_chars=20,
            max_context_chars=40,
        ),
    )
    agent = ReActAgent(FakeProvider(), config, ui=ui)

    await agent.run("please use tool: echo")

    kinds = [kind for kind, _ in ui.events]
    assert "compaction_start" in kinds
    assert "compaction_end" in kinds


async def test_always_allow_via_ui_grants_write_permission(tmp_path: Path) -> None:
    # The RecordingUI answers "always"; default permission mode would otherwise ask
    # for the write tool. The run completes (tool not denied), proving the prompter
    # path is exercised and grants permission.
    ui = RecordingUI()
    agent = ReActAgent(FakeProvider(), _config(tmp_path), ui=ui)
    result = await agent.run("please use tool: write_text_file")
    assert "observation" in result.answer.lower()


async def test_console_questions_stop_batch_on_discussion(monkeypatch) -> None:
    from agent_core.terminal import question_picker

    seen: list[str] = []

    async def fake_picker(question):
        seen.append(question.id)
        if question.id == "first":
            return question_picker.QuestionResponse(
                "first", "answer", answer="A", selected_options=("A",)
            )
        return question_picker.QuestionResponse(
            "second",
            "discussion",
            answer="Compare these first",
            discussion="Compare these first",
        )

    monkeypatch.setattr(question_picker, "run_question_picker", fake_picker)
    questions = [
        {
            "id": question_id,
            "question": f"Question {question_id}?",
            "options": [
                {"label": "A", "description": "First"},
                {"label": "B", "description": "Second"},
            ],
        }
        for question_id in ("first", "second", "third")
    ]

    answers = await ConsoleUI(color=False).ask_questions(questions)

    assert seen == ["first", "second"]
    assert [answer["kind"] for answer in answers] == ["answer", "discussion"]


# --- permission prompt: see tests/test_terminal_permission.py ----------------
# confirm_tool now bridges to TerminalRenderer.ask_permission_async (prompt_toolkit),
# so the interactive prompt is covered there with a headless pipe input.


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


async def test_fake_provider_streams_chunks_matching_content() -> None:
    sink = _RecordingStream()
    provider = FakeProvider()
    from agent_core.providers.base import ProviderConfig

    result = await provider.complete([Message("user", "say hi please")], [], ProviderConfig(), stream=sink)
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


# --- compaction progress bar -------------------------------------------------


def test_console_ui_compaction_bar_and_summary(capsys) -> None:
    ui = ConsoleUI(color=False)
    ui.on_compaction_start(reactive=False)
    ui.on_compaction_progress(1 / 3, "snip")
    ui.on_compaction_progress(2 / 3, "microcompact")
    ui.on_compaction_progress(1.0, "context_collapse")
    ui.on_compaction_end(24000, 9000, "collapsed 12 msgs", reactive=False)
    out = capsys.readouterr().out
    assert "100%" in out  # the bar reached full
    assert "█" in out  # rendered with block glyphs
    assert "compacted 24.0k→9.0k chars" in out
    assert "collapsed 12 msgs" in out
    # The summary text never includes the compressed content itself.
    assert "answer" not in out


def test_console_ui_reactive_compaction_is_louder(capsys) -> None:
    ui = ConsoleUI(color=False)
    ui.on_compaction_start(reactive=True)
    ui.on_compaction_progress(1.0, "context_collapse")
    ui.on_compaction_end(31000, 7000, "collapsed 20 msgs", reactive=True)
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "overflowed" in out


def test_console_ui_compaction_summary_drops_empty_detail(capsys) -> None:
    ui = ConsoleUI(color=False)
    ui.on_compaction_start(reactive=False)
    ui.on_compaction_end(500, 500, "", reactive=False)
    out = capsys.readouterr().out
    assert "compacted 500→500 chars" in out
    assert " · " not in out  # no dangling separator when there's no detail
