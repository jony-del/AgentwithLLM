"""Rendering tests for the compact terminal renderer.

The renderer is driven with ``color=False`` and captured via ``capsys``; under a
non-TTY pytest stdout Rich emits plain text, so assertions match on content (and
the absence of box-drawing borders / ANSI escapes).
"""
from agent_core.terminal.app import TerminalRenderer


# --- compact tool stream -----------------------------------------------------


def test_tool_call_is_a_compact_bullet_line(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_call("echo", "write", {"text": "hi"})
    out = capsys.readouterr().out
    assert "● echo(" in out          # bullet header, not a boxed panel
    assert "[write]" in out
    assert "╭" not in out and "╰" not in out  # no Panel border around a tool call


def test_tool_result_is_an_indented_branch(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_result(True, "Wrote a.txt")
    out = capsys.readouterr().out
    assert "⎿" in out
    assert "Wrote a.txt" in out


def test_tool_result_error_is_marked(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_result(False, "No such file")
    out = capsys.readouterr().out
    assert "⎿" in out
    assert "✗" in out
    assert "No such file" in out


# --- markup safety -----------------------------------------------------------


def test_answer_with_brackets_is_not_swallowed_as_markup(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.reset_stream_state()
    r.print_final("use [bold]Text[/] and arr[0] here")
    out = capsys.readouterr().out
    assert "[bold]Text[/]" in out
    assert "arr[0]" in out


def test_tool_result_with_brackets_is_literal(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_result(True, "matched [pattern] in items[3]")
    out = capsys.readouterr().out
    assert "[pattern]" in out
    assert "items[3]" in out


# --- diff path ---------------------------------------------------------------


def test_diff_is_rendered_under_the_branch(capsys) -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old line\n+new line\n"
    r = TerminalRenderer(color=False)
    r.print_tool_result(True, "Replaced 1 occurrence(s)", diff=diff)
    out = capsys.readouterr().out
    assert "@@" in out
    assert "old line" in out and "new line" in out


# --- folding -----------------------------------------------------------------


def test_consecutive_reads_searches_fold_to_one_line(capsys) -> None:
    r = TerminalRenderer(color=False)  # verbose=False by default
    r.print_tool_call("view_file", "read", {"path": "a"})
    r.print_tool_result(True, "...")
    r.print_tool_call("view_file", "read", {"path": "b"})
    r.print_tool_result(True, "...")
    r.print_tool_call("search_text", "read", {"q": "TODO"})
    r.print_tool_result(True, "...")
    r.close_block()
    out = capsys.readouterr().out
    assert "Read 2 files" in out
    assert "Searched 1 patterns" in out
    # The individual foldable calls are not printed as their own headers.
    assert "view_file(" not in out
    assert "search_text(" not in out


def test_verbose_disables_folding(capsys) -> None:
    r = TerminalRenderer(color=False, verbose=True)
    r.print_tool_call("view_file", "read", {"path": "a"})
    out = capsys.readouterr().out
    assert "● view_file(" in out


def test_fold_state_resets_across_turns(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_call("view_file", "read", {"path": "a"})
    r.reset_stream_state()  # new turn flushes + clears the fold
    capsys.readouterr()  # drop first-turn output
    r.print_tool_call("view_file", "read", {"path": "b"})
    r.close_block()
    out = capsys.readouterr().out
    assert "Read 1 files" in out  # count restarted, not "Read 2 files"


# --- recap -------------------------------------------------------------------


def test_run_recap_is_a_flat_line(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_run_completed({"duration": 138, "steps": 5, "reason": "completed", "tool_counts": {"echo": 3}})
    out = capsys.readouterr().out
    assert "Done in 2m 18s" in out
    assert "5 steps" in out
    assert "3 echo" in out
    assert "╭" not in out  # flat line, not a "Run Recap" box


def test_run_recap_stopped_is_louder(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_run_completed({"duration": 5, "steps": 2, "reason": "max_steps", "tool_counts": {}})
    out = capsys.readouterr().out
    assert "Stopped (max_steps)" in out


# --- token usage gauge -------------------------------------------------------


def test_token_usage_splits_chat_and_baseline(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_token_usage(
        {
            "context_tokens": 12300,
            "conversation_tokens": 300,
            "window": 200000,
            "input_tokens": 12300,
            "output_tokens": 231,
        }
    )
    out = capsys.readouterr().out
    assert "ctx 12.3k/200.0k (6.2%)" in out
    assert "chat 300 / base 12.0k" in out  # base = context - conversation
    assert "12.3k in / 231 out" in out


def test_token_usage_fresh_session_reads_zero_chat(capsys) -> None:
    # A fresh / cleared session: almost all of the prompt is fixed baseline.
    r = TerminalRenderer(color=False)
    r.print_token_usage(
        {"context_tokens": 12100, "conversation_tokens": 0, "window": 200000, "input_tokens": 12100, "output_tokens": 44}
    )
    out = capsys.readouterr().out
    assert "chat 0 / base 12.1k" in out


def test_token_usage_clamps_conversation_to_context(capsys) -> None:
    # A conversation estimate above the real total must never render a negative base.
    r = TerminalRenderer(color=False)
    r.print_token_usage(
        {"context_tokens": 500, "conversation_tokens": 9000, "window": 200000, "input_tokens": 500, "output_tokens": 10}
    )
    out = capsys.readouterr().out
    assert "chat 500 / base 0" in out


def test_token_usage_without_split_uses_legacy_line(capsys) -> None:
    # No conversation_tokens key → the original single-figure line, no chat/base segment.
    r = TerminalRenderer(color=False)
    r.print_token_usage(
        {"context_tokens": 12300, "window": 200000, "input_tokens": 12300, "output_tokens": 231}
    )
    out = capsys.readouterr().out
    assert "ctx 12.3k/200.0k (6.2%)" in out
    assert "chat" not in out and "base" not in out


# --- color off ---------------------------------------------------------------


def test_color_false_emits_no_ansi(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.print_tool_call("echo", "write", {"text": "hi"})
    r.print_tool_result(True, "done")
    out = capsys.readouterr().out
    assert "\x1b[" not in out  # no SGR / control escapes when color is disabled


# --- streaming finalizer (no duplicate) --------------------------------------


def test_streamed_answer_is_not_reprinted_by_finalizer(capsys) -> None:
    r = TerminalRenderer(color=False)
    r.reset_stream_state()
    r.write_text_delta("hel")
    r.write_text_delta("lo")
    r.print_final("hello")  # finalizer on a streamed turn must not reprint
    out = capsys.readouterr().out
    assert out.count("hello") == 1
