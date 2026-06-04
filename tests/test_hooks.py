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


def test_line_truncation_keeps_head_tail_and_counts_omitted(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=100, head_lines=20, tail_lines=20)
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
    hook = _hook(tmp_path, max_lines=100, max_chars=1000)
    content = "x" * 5000
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert "[... omitted 4000 characters ...]" in out.content
    assert len(out.content) < len(content)
    assert out.metadata["original_chars"] == 5000


def test_spill_disabled_still_truncates(tmp_path: Path) -> None:
    hook = _hook(tmp_path, max_lines=10, spill=False)
    content = "\n".join(str(i) for i in range(100))
    out = hook.after_tool(ToolCall(name="bash"), _result(content))

    assert "omitted" in out.content
    assert out.metadata["full_output_path"] is None
    assert not (tmp_path / "outputs").exists()


# --- config wiring -----------------------------------------------------------


def test_output_config_from_dict_coerces_and_ignores_unknown() -> None:
    config = OutputLimitConfig.from_dict(
        {"max_lines": "40", "spill": "false", "head_lines": "5", "nope": 1}
    )
    assert config.max_lines == 40
    assert config.spill is False
    assert config.head_lines == 5
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
