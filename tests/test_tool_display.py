"""Tests for the ToolDisplayProvider seam: write/edit tools expose a compact
arg label and a unified-diff string the UI can render; read tools decline.
"""
from pathlib import Path

from agent_core.tools.builtin import EditFileTool, WriteTextFileTool
from agent_core.tools.editing import ApplyPatchTool, GlobTool, MultiEditTool


async def test_edit_file_renders_label_and_diff(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    args = {"path": "f.txt", "old_string": "beta", "new_string": "gamma"}
    result = await tool.run(args)

    assert tool.render_args(args) == "f.txt"
    diff = tool.render_result(args, result)
    assert diff is not None
    assert "@@" in diff
    assert "-beta" in diff and "+gamma" in diff


async def test_write_text_file_diff_shows_created_content(tmp_path: Path) -> None:
    tool = WriteTextFileTool(tmp_path)
    args = {"path": "new.txt", "content": "hello\nworld\n"}
    result = await tool.run(args)

    assert tool.render_args(args) == "new.txt"
    diff = tool.render_result(args, result)
    assert diff is not None
    assert "+hello" in diff and "+world" in diff


async def test_multi_edit_builds_diff_across_edits(tmp_path: Path) -> None:
    (tmp_path / "m.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = MultiEditTool(tmp_path)
    args = {
        "path": "m.txt",
        "edits": [
            {"old_string": "one", "new_string": "1"},
            {"old_string": "three", "new_string": "3"},
        ],
    }
    result = await tool.run(args)

    assert tool.render_args(args) == "m.txt, 2 edits"
    diff = tool.render_result(args, result)
    assert diff is not None
    assert "-one" in diff and "+1" in diff
    assert "-three" in diff and "+3" in diff


async def test_apply_patch_returns_patch_verbatim(tmp_path: Path) -> None:
    (tmp_path / "p.txt").write_text("x\ny\n", encoding="utf-8")
    patch = "--- a/p.txt\n+++ b/p.txt\n@@ -1,2 +1,2 @@\n x\n-y\n+z\n"
    tool = ApplyPatchTool(tmp_path)
    args = {"patch": patch}
    result = await tool.run(args)

    assert result.ok
    assert tool.render_args(args) == "p.txt"
    assert tool.render_result(args, result) == patch


async def test_read_tool_declines_display(tmp_path: Path) -> None:
    tool = GlobTool(tmp_path)
    args = {"pattern": "**/*.py"}
    result = await tool.run(args)
    # Defaults from ToolDisplayProvider: no custom label, no diff.
    assert tool.render_args(args) is None
    assert tool.render_result(args, result) is None


async def test_failed_edit_has_empty_diff(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("alpha\n", encoding="utf-8")
    tool = EditFileTool(tmp_path)
    args = {"path": "f.txt", "old_string": "nope", "new_string": "x"}
    result = await tool.run(args)
    assert not result.ok
    # No "diff" key was stashed on the failure path → render_result declines.
    assert tool.render_result(args, result) is None
