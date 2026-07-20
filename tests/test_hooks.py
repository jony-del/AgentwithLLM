from pathlib import Path

from agent_core.config import resolve_output_config
from agent_core.hooks import MaxOutputPostHook, OutputLimitConfig
from agent_core.models import ToolCall, ToolResult


def _hook(tmp_path: Path, **kwargs) -> MaxOutputPostHook:
    return MaxOutputPostHook(spill_dir=tmp_path / "outputs", **kwargs)


def _result(content: str) -> ToolResult:
    return ToolResult(name="bash", content=content)


def test_small_output_is_unchanged(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=100)
    result = _result("just a few\nlines here")
    out = hook.after_tool(ToolCall(name="bash"), result)
    assert out is result  # same object, untouched
    assert not (tmp_path / "outputs").exists()  # nothing spilled


def test_read_text_file_is_exempt_from_spill(tmp_path: Path) -> None:
    # read_text_file is the designated pager; spilling its output would point a
    # pointer back at itself. The same oversized output from another tool DOES spill.
    hook = _hook(tmp_path, max_lines=100)
    content = "\n".join(f"line {i}" for i in range(5000))

    read = hook.after_tool(ToolCall(name="read_text_file"), _result(content))
    assert read.content == content              # returned verbatim
    assert not read.metadata.get("spilled")
    assert not (tmp_path / "outputs").exists()  # nothing written for the exempt tool

    other = hook.after_tool(ToolCall(name="bash"), _result(content))
    assert other.metadata.get("spilled") is True  # non-exempt tool still spills


def test_default_thresholds_are_coding_agent_sized() -> None:
    # Guard the raised defaults so an ordinary file (a few hundred lines) is never
    # truncated back to a tiny preview again.
    cfg = OutputLimitConfig()
    assert cfg.max_lines >= 2000
    assert cfg.max_chars >= 50000


def test_line_truncation_keeps_head_tail_and_counts_omitted(tmp_path: Path) -> None:
    # Legacy head/tail mode (pointer=False) — byte-for-byte regression guard.
    hook = _hook(tmp_path, max_lines=100, head_lines=20, tail_lines=20, pointer=False)
    content = "\n".join(f"line {i}" for i in range(500))
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert "[... omitted 460 lines ...]" in out.content  # 500 - 20 - 20
    assert "line 0" in out.content and "line 19" in out.content      # head kept
    assert "line 499" in out.content and "line 480" in out.content   # tail kept
    assert "line 250" not in out.content                              # middle dropped
    assert out.metadata["original_lines"] == 500
    assert out.metadata["post_hook"] == "max_output"


def test_full_output_is_spilled_verbatim(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=10)
    content = "\n".join(f"row {i}" for i in range(200))
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    path = Path(out.metadata["full_output_path"])
    assert path.exists()
    assert path.read_text(encoding="utf-8") == content   # complete, untouched
    assert str(path) in out.content                       # pointer appended


def test_char_fallback_for_single_huge_line(tmp_path: Path) -> None:
    # One enormous line never trips the line budget, so the char budget must.
    # Legacy head/tail mode (pointer=False) — byte-for-byte regression guard.
    hook = _hook(tmp_path, max_lines=100, max_chars=1000, pointer=False)
    content = "x" * 5000
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert "[... omitted 4000 characters ...]" in out.content
    assert len(out.content) < len(content)
    assert out.metadata["original_chars"] == 5000


def test_spill_disabled_still_truncates(tmp_path: Path) -> None:
    # Legacy head/tail mode (pointer=False) — byte-for-byte regression guard.
    hook = _hook(tmp_path, max_lines=10, spill=False, pointer=False)
    content = "\n".join(str(i) for i in range(100))
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert "omitted" in out.content
    assert out.metadata["full_output_path"] is None
    assert not (tmp_path / "outputs").exists()


# --- pointer mode (default): structured preview + retrievable on-disk pointer --------


def test_pointer_replaces_oversized_output_with_preview_and_ref(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=100, preview_chars=500)  # pointer=True by default
    content = "\n".join(f"line {i}" for i in range(2000))  # well over the line budget
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    # Structured, machine-recognizable pointer body.
    assert out.content.startswith("<tool_output_ref>")
    assert out.content.rstrip().endswith("</tool_output_ref>")
    # Head preview only — first lines kept, far tail dropped from live context.
    assert "line 0" in out.content
    assert "line 1999" not in out.content
    assert "read_text_file(" in out.content  # explicit retrieval instruction
    # Preview body is bounded by preview_chars (plus the small framing text).
    assert len(out.content) < len(content)

    # Machine-readable metadata.
    ref = out.metadata["tool_result_ref"]
    assert out.metadata["spilled"] is True
    assert out.metadata["preview_chars"] == 500
    assert ref is not None and ref == out.metadata["full_output_path"]
    assert str(ref) in out.content

    # The spill file holds the complete, untouched output.
    assert Path(ref).read_text(encoding="utf-8") == content


def test_pointer_is_idempotent_on_already_spilled_result(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=10, preview_chars=200)
    content = "\n".join(f"row {i}" for i in range(500))
    once = hook.after_tool(ToolCall(name="bash"), _result(content))
    assert once.metadata["spilled"] is True

    # Re-running on an already-spilled result returns it untouched (frozen decision).
    twice = hook.after_tool(ToolCall(name="bash"), once)
    assert twice is once
    # No second spill file was created.
    assert len(list((tmp_path / "outputs").iterdir())) == 1


def test_pointer_spill_disabled_degrades_to_preview_only(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=10, preview_chars=200, spill=False)
    content = "\n".join(f"row {i}" for i in range(500))
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert out.content.startswith("<tool_output_ref>")
    assert "spill disabled" in out.content
    assert "read_text_file(" not in out.content  # no path to point at
    assert out.metadata["tool_result_ref"] is None
    assert out.metadata["full_output_path"] is None
    assert not (tmp_path / "outputs").exists()


def test_pointer_small_output_is_unchanged(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=100)
    result = _result("a\nb\nc")
    out = hook.after_tool(ToolCall(name="bash"), result)
    assert out is result
    assert "spilled" not in out.metadata


# --- config wiring -----------------------------------------------------------


def test_output_config_from_dict_coerces_and_ignores_unknown() -> None:
    config = OutputLimitConfig.from_dict(
        {"max_lines": "40", "spill": "false", "head_lines": "5",
         "preview_chars": "500", "pointer": "false", "nope": 1}
    )
    assert config.max_lines == 40
    assert config.spill is False
    assert config.head_lines == 5
    assert config.preview_chars == 500
    assert config.pointer is False
    assert config.tail_lines == OutputLimitConfig().tail_lines  # default kept


def test_from_config_builds_matching_hook(tmp_path: Path) -> None:
    config = OutputLimitConfig(max_lines=30, max_chars=500, head_lines=3, tail_lines=4, spill=False)
    hook = MaxOutputPostHook.from_config(config, spill_dir=tmp_path)
    assert (hook.max_lines, hook.max_chars, hook.head_lines, hook.tail_lines, hook.spill) == (
        30, 500, 3, 4, False,
    )


def test_resolve_output_config_reads_table(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text("[output]\nmax_lines = 25\nspill = false\n", encoding="utf-8")
    config = resolve_output_config(toml)
    assert config.max_lines == 25
    assert config.spill is False


def test_resolve_output_config_without_table_is_defaults(tmp_path: Path) -> None:
    toml = tmp_path / "agent.toml"
    toml.write_text('model = "x"\n', encoding="utf-8")
    assert resolve_output_config(toml) == OutputLimitConfig()
