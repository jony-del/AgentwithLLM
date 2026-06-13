from pathlib import Path

import pytest

from agent_core.config import resolve_context_config
from agent_core.context import build_project_instructions


async def test_no_claude_md_returns_none(tmp_path: Path) -> None:
    result = await build_project_instructions(tmp_path, include_user_home=False)
    assert result is None


async def test_discovers_workspace_claude_md(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Use tabs, not spaces.", encoding="utf-8")

    result = await build_project_instructions(tmp_path, include_user_home=False)

    assert result is not None
    assert "Use tabs, not spaces." in result
    assert str(tmp_path / "CLAUDE.md") in result
    assert "OVERRIDE any default behavior" in result  # preamble present


async def test_multi_level_order_root_to_workspace(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("ROOT RULES", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "CLAUDE.md").write_text("SUB RULES", encoding="utf-8")

    result = await build_project_instructions(sub, include_user_home=False)

    assert result is not None
    assert "ROOT RULES" in result and "SUB RULES" in result
    # Root file is lower priority, so it appears before the workspace file.
    assert result.index("ROOT RULES") < result.index("SUB RULES")


async def test_truncates_oversized(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("x" * 5000, encoding="utf-8")

    result = await build_project_instructions(tmp_path, include_user_home=False, max_chars=500)

    assert result is not None
    assert len(result) <= 500
    assert result.endswith("...(truncated)")


async def test_unreadable_file_skipped(tmp_path: Path) -> None:
    # A directory named CLAUDE.md is not a regular file: discovery skips it, no raise.
    (tmp_path / "CLAUDE.md").mkdir()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "CLAUDE.md").write_text("REAL RULES", encoding="utf-8")

    result = await build_project_instructions(tmp_path / "sub", include_user_home=False)

    assert result is not None
    assert "REAL RULES" in result


async def test_user_home_included_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "CLAUDE.md").write_text("GLOBAL RULES", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    workspace = tmp_path / "proj"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text("PROJECT RULES", encoding="utf-8")

    result = await build_project_instructions(workspace, include_user_home=True)

    assert result is not None
    assert "GLOBAL RULES" in result and "PROJECT RULES" in result
    # User-global memory is lowest priority and comes first.
    assert result.index("GLOBAL RULES") < result.index("PROJECT RULES")


async def test_empty_file_returns_none(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("   \n", encoding="utf-8")
    result = await build_project_instructions(tmp_path, include_user_home=False)
    assert result is None


def test_disabled_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DISABLE_CLAUDE_MD", "1")
    values = resolve_context_config(tmp_path / "no-such.toml")
    assert values["project_instructions"] is False


def test_enabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_DISABLE_CLAUDE_MD", raising=False)
    values = resolve_context_config(tmp_path / "no-such.toml")
    assert values["project_instructions"] is True
    assert values["claudemd_max_chars"] == 32000
