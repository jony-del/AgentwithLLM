from pathlib import Path

import pytest

from agent_core.models import ToolRisk
from agent_core.tools.editing import ApplyPatchTool, GlobTool, MultiEditTool
from agent_core.unified_diff import PatchError, PatchOperation, parse_unified_diff


# --- glob --------------------------------------------------------------------


async def test_glob_matches_recursively(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("y", encoding="utf-8")
    (tmp_path / "c.md").write_text("z", encoding="utf-8")
    out = (await GlobTool(tmp_path).run({"pattern": "**/*.py"})).content
    assert "src/a.py" in out
    assert "b.py" in out
    assert "c.md" not in out


async def test_glob_skips_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.py").write_text("x", encoding="utf-8")
    (tmp_path / "keep.py").write_text("y", encoding="utf-8")
    out = (await GlobTool(tmp_path).run({"pattern": "**/*.py"})).content
    assert "keep.py" in out
    assert ".git" not in out


async def test_glob_no_match(tmp_path: Path) -> None:
    assert (await GlobTool(tmp_path).run({"pattern": "**/*.zzz"})).content == "No files matched."


def test_glob_is_read_risk() -> None:
    assert GlobTool().risk is ToolRisk.READ


# --- multi_edit --------------------------------------------------------------


async def test_multi_edit_applies_in_order(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("alpha beta gamma", encoding="utf-8")
    result = await MultiEditTool(tmp_path).run(
        {
            "path": "f.txt",
            "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "gamma", "new_string": "GAMMA"},
            ],
        }
    )
    assert result.ok
    assert f.read_text(encoding="utf-8") == "ALPHA beta GAMMA"


async def test_multi_edit_sees_previous_edit(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("one", encoding="utf-8")
    result = await MultiEditTool(tmp_path).run(
        {
            "path": "f.txt",
            "edits": [
                {"old_string": "one", "new_string": "two"},
                {"old_string": "two", "new_string": "three"},
            ],
        }
    )
    assert result.ok
    assert f.read_text(encoding="utf-8") == "three"


async def test_multi_edit_is_atomic_on_failure(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_text("keep me", encoding="utf-8")
    result = await MultiEditTool(tmp_path).run(
        {
            "path": "f.txt",
            "edits": [
                {"old_string": "keep", "new_string": "KEEP"},
                {"old_string": "absent", "new_string": "x"},  # fails
            ],
        }
    )
    assert not result.ok
    assert result.metadata["failed_edit"] == 1
    assert f.read_text(encoding="utf-8") == "keep me"  # unchanged — nothing written


async def test_multi_edit_requires_nonempty_list(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    result = await MultiEditTool(tmp_path).run({"path": "f.txt", "edits": []})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


# --- apply_patch -------------------------------------------------------------


async def test_apply_patch_modifies_file(tmp_path: Path) -> None:
    f = tmp_path / "greet.py"
    f.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
    patch = (
        "--- a/greet.py\n"
        "+++ b/greet.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def greet():\n"
        "-    return 'hi'\n"
        "+    return 'hello'\n"
    )
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})
    assert result.ok, result.content
    assert f.read_text(encoding="utf-8") == "def greet():\n    return 'hello'\n"


async def test_apply_patch_creates_new_file(tmp_path: Path) -> None:
    patch = (
        "--- /dev/null\n"
        "+++ b/new.txt\n"
        "@@ -0,0 +1,2 @@\n"
        "+line one\n"
        "+line two\n"
    )
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})
    assert result.ok, result.content
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "line one\nline two\n"


async def test_apply_patch_rejects_when_context_missing(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("original content\n", encoding="utf-8")
    patch = (
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-totally different line\n"
        "+replacement\n"
    )
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})
    assert not result.ok
    assert result.metadata["error_type"] == "HunkFailed"
    assert f.read_text(encoding="utf-8") == "original content\n"  # untouched


async def test_apply_patch_is_atomic_across_files(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("aaa\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("bbb\n", encoding="utf-8")
    patch = (
        "--- a/good.txt\n"
        "+++ b/good.txt\n"
        "@@ -1 +1 @@\n"
        "-aaa\n"
        "+AAA\n"
        "--- a/bad.txt\n"
        "+++ b/bad.txt\n"
        "@@ -1 +1 @@\n"
        "-missing\n"
        "+XXX\n"
    )
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})
    assert not result.ok
    # The first file must NOT have been written because the second hunk failed.
    assert good.read_text(encoding="utf-8") == "aaa\n"
    assert bad.read_text(encoding="utf-8") == "bbb\n"


def test_apply_patch_is_write_risk() -> None:
    assert ApplyPatchTool().risk is ToolRisk.WRITE


async def test_apply_patch_deletes_file(tmp_path: Path) -> None:
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete\n", encoding="utf-8")
    patch = "--- a/obsolete.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-obsolete\n"

    parsed = parse_unified_diff(patch)
    assert parsed.files[0].operation is PatchOperation.DELETE
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})

    assert result.ok, result.content
    assert not target.exists()


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/old.txt\n+++ b/new.txt\n@@ -1 +1 @@\n-old\n+new\n",
        "--- /dev/null\n+++ b/../escape.txt\n@@ -0,0 +1 @@\n+x\n",
        "--- /dev/null\n+++ C:/absolute.txt\n@@ -0,0 +1 @@\n+x\n",
        (
            "--- /dev/null\n+++ b/dup.txt\n@@ -0,0 +1 @@\n+one\n"
            "--- /dev/null\n+++ b/dup.txt\n@@ -0,0 +1 @@\n+two\n"
        ),
    ],
)
def test_canonical_patch_parser_rejects_ambiguous_targets(patch: str) -> None:
    with pytest.raises(PatchError):
        parse_unified_diff(patch)


# --- byte-exact line endings (P2-13) -------------------------------------------
# Windows ``Path.write_text`` defaults to ``newline=None`` and would translate the
# model's LF-only content to CRLF; every editing tool must write bytes verbatim.


async def test_write_text_file_keeps_lf_endings(tmp_path: Path) -> None:
    from agent_core.tools.builtin import WriteTextFileTool

    target = tmp_path / "lf.txt"
    result = await WriteTextFileTool(tmp_path).run(
        {"path": "lf.txt", "content": "one\ntwo\nthree\n"}
    )
    assert result.ok, result.content
    assert target.read_bytes() == b"one\ntwo\nthree\n"


async def test_edit_file_keeps_lf_endings(tmp_path: Path) -> None:
    from agent_core.tools.builtin import EditFileTool

    target = tmp_path / "lf.txt"
    target.write_bytes(b"one\ntwo\n")
    result = await EditFileTool(tmp_path).run(
        {"path": "lf.txt", "old_string": "two", "new_string": "2\n2b"}
    )
    assert result.ok, result.content
    assert target.read_bytes() == b"one\n2\n2b\n"


async def test_multi_edit_keeps_lf_endings(tmp_path: Path) -> None:
    target = tmp_path / "lf.txt"
    target.write_bytes(b"alpha\nbeta\n")
    result = await MultiEditTool(tmp_path).run(
        {"path": "lf.txt", "edits": [{"old_string": "alpha", "new_string": "a1\na2"}]}
    )
    assert result.ok, result.content
    assert target.read_bytes() == b"a1\na2\nbeta\n"


async def test_apply_patch_keeps_lf_endings(tmp_path: Path) -> None:
    target = tmp_path / "lf.txt"
    target.write_bytes(b"line\n")
    patch = (
        "--- a/lf.txt\n"
        "+++ b/lf.txt\n"
        "@@ -1,1 +1,2 @@\n"
        " line\n"
        "+line two\n"
    )
    result = await ApplyPatchTool(tmp_path).run({"patch": patch})
    assert result.ok, result.content
    assert target.read_bytes() == b"line\nline two\n"
