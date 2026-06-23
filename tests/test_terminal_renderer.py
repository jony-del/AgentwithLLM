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
