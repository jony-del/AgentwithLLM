"""P1-2 (audit): the permission layer and the edit tool share one canonical parser.

``permission_safety.extract_path_targets()`` (authorization) and
``tools/editing._parse_unified_diff()`` (execution) both consume
``agent_core.unified_diff.parse_unified_diff``, so a patch can no longer be
authorized under one interpretation and executed under another. The property:
for randomly generated mixed diffs (create/update/delete, including
``+++ /dev/null`` delete form), EVERY delete/update target the canonical parser
produces appears in the permission layer's extracted target set; and deletes of
``.git/config``, ``.env``, ``.git/hooks/*`` are always caught (ask/deny).
Unparseable non-blank patches must fail closed (deny), never silently pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core.permission_safety import (
    extract_path_targets,
    inspect_paths,
    is_protected_path,
    is_secret_path,
)
from agent_core.permission_types import (
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
)
from agent_core.tools.editing import _parse_unified_diff
from agent_core.unified_diff import PatchError, PatchOperation, parse_unified_diff

_content = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-", max_size=20
)

_safe_target = st.lists(
    st.from_regex(r"[a-z][a-z0-9_-]{0,7}", fullmatch=True), min_size=1, max_size=3
).map("/".join)
_sensitive_target = st.sampled_from(
    [
        ".git/config",
        ".env",
        ".env.local",
        ".git/hooks/pre-commit",
        ".git/hooks/x",
        ".ssh/id_rsa",
        ".aws/credentials",
    ]
)
_patch_target = st.one_of(_safe_target, _sensitive_target)


def _entry(draw: st.DrawFn, path: str, operation: str) -> str:
    if operation == "create":
        adds = draw(st.lists(_content, min_size=1, max_size=3))
        lines = ["--- /dev/null", f"+++ b/{path}", f"@@ -0,0 +1,{len(adds)} @@"]
        lines += ["+" + content for content in adds]
    elif operation == "delete":
        olds = draw(st.lists(_content, min_size=1, max_size=3))
        lines = [f"--- a/{path}", "+++ /dev/null", f"@@ -1,{len(olds)} +0,0 @@"]
        lines += ["-" + content for content in olds]
    else:
        context = draw(st.lists(_content, max_size=2))
        olds = draw(st.lists(_content, min_size=1, max_size=2))
        adds = draw(st.lists(_content, min_size=1, max_size=2))
        lines = [f"--- a/{path}", f"+++ b/{path}", "@@ -1 +1 @@"]
        lines += [" " + content for content in context]
        lines += ["-" + content for content in olds]
        lines += ["+" + content for content in adds]
        lines += [" " + content for content in context[:1]]
    return "\n".join(lines)


@st.composite
def _patch(draw: st.DrawFn) -> str:
    count = draw(st.integers(1, 4))
    paths = draw(
        st.lists(_patch_target, min_size=count, max_size=count, unique_by=str.casefold)
    )
    entries = []
    for path in paths:
        operation = draw(st.sampled_from(["create", "update", "delete"]))
        entries.append(_entry(draw, path, operation))
    return "\n".join(entries) + "\n"


def _context(workspace: Path) -> PermissionContext:
    return PermissionContext(
        mode=PermissionMode.ACCEPTEDITS, workspace=workspace, interactive=False
    )


@given(patch=_patch())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_delete_and_update_targets_reach_permission_layer(
    patch: str, tmp_path: Path
) -> None:
    parsed = parse_unified_diff(patch)
    extracted = extract_path_targets("apply_patch", {"patch": patch})
    raws = {target.raw for target in extracted}

    # The permission layer sees exactly the canonical parser's targets...
    assert raws == set(parsed.targets)
    assert all(target.operation == "write" for target in extracted)
    # ...and in particular every delete/update target is among them.
    for file in parsed.files:
        if file.operation in {PatchOperation.DELETE, PatchOperation.UPDATE}:
            assert file.target in raws

    # The execution-layer compatibility shim agrees on the same targets.
    shim_targets = {target for target, _hunks, _is_new in _parse_unified_diff(patch)}
    assert shim_targets == set(parsed.targets)

    # Any secret/protected target must surface as ask/deny, never pass silently.
    sensitive = [
        file
        for file in parsed.files
        if is_protected_path(file.target) or is_secret_path(file.target)
    ]
    if sensitive:
        result = inspect_paths("apply_patch", {"patch": patch}, _context(tmp_path))
        assert result is not None
        assert result.behavior in {PermissionBehavior.ASK, PermissionBehavior.DENY}


@given(text=st.text(alphabet="-+@ ab/._\\\t\ndevullgit", max_size=300))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_unparseable_patches_fail_closed(text: str, tmp_path: Path) -> None:
    try:
        parsed = parse_unified_diff(text)
    except PatchError:
        parsed = None
    targets = extract_path_targets("apply_patch", {"patch": text})
    if parsed is None:
        assert targets == []
        if text.strip():
            result = inspect_paths("apply_patch", {"patch": text}, _context(tmp_path))
            assert result is not None
            assert result.behavior is PermissionBehavior.DENY
    else:
        assert {target.raw for target in targets} == set(parsed.targets)


# --- deterministic pins for the exact audit findings --------------------------------


@pytest.mark.parametrize(
    "target,expected",
    [
        (".git/config", PermissionBehavior.ASK),
        (".env", PermissionBehavior.ASK),
        (".env.local", PermissionBehavior.ASK),
        (".git/hooks/x", PermissionBehavior.DENY),
        (".git/hooks/pre-commit", PermissionBehavior.DENY),
    ],
)
def test_sensitive_delete_targets_are_caught(
    tmp_path: Path, target: str, expected: PermissionBehavior
) -> None:
    patch = f"--- a/{target}\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    parsed = parse_unified_diff(patch)
    assert parsed.files[0].operation is PatchOperation.DELETE
    extracted = extract_path_targets("apply_patch", {"patch": patch})
    assert [item.raw for item in extracted] == [target]
    result = inspect_paths("apply_patch", {"patch": patch}, _context(tmp_path))
    assert result is not None
    assert result.behavior is expected


def test_mixed_patch_with_sensitive_delete_is_caught(tmp_path: Path) -> None:
    patch = (
        "--- a/safe.txt\n+++ b/safe.txt\n@@ -1 +1 @@\n-old\n+new\n"
        "--- a/.git/config\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    )
    extracted = extract_path_targets("apply_patch", {"patch": patch})
    assert {item.raw for item in extracted} == {"safe.txt", ".git/config"}
    result = inspect_paths("apply_patch", {"patch": patch}, _context(tmp_path))
    assert result is not None
    assert result.behavior is PermissionBehavior.ASK
