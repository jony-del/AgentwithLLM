"""P0-1 (audit): journal recovery path-containment validation.

The recovery path rejects journal-controlled paths through three layers:
``TurnExecutionJournal._safe_relative`` (the ``changed`` list), ``_contained``
(canonical prefix check), and ``checked_path`` (symlink/reparse-aware lexical walk).
These properties fuzz all three: strategically generated hostile paths (absolute,
``..`` escapes, drive letters, UNC, device paths/names, backslashes, control
characters, trailing dots/spaces) must always be rejected, while canonical relative
paths must always be accepted unchanged.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core.recovery_paths import checked_path
from agent_core.tools.transaction import (
    _WINDOWS_RESERVED_NAME,
    TurnExecutionJournal,
    _contained,
)

_safe_part = st.from_regex(r"[a-z][a-z0-9_-]{0,7}(\.[a-z0-9]{1,4})?", fullmatch=True).filter(
    lambda part: _WINDOWS_RESERVED_NAME.fullmatch(part) is None
)
_safe_relative = st.lists(_safe_part, min_size=1, max_size=4).map("/".join)

_malicious = st.one_of(
    # POSIX absolute.
    _safe_relative.map(lambda p: "/" + p),
    # Windows drive absolute and drive-relative.
    st.tuples(st.sampled_from("CDEZcdez"), _safe_relative).map(lambda t: f"{t[0]}:/{t[1]}"),
    st.tuples(st.sampled_from("CDEZcdez"), _safe_relative).map(lambda t: f"{t[0]}:\\{t[1]}"),
    st.tuples(st.sampled_from("CDc"), _safe_relative).map(lambda t: f"{t[0]}:{t[1]}"),
    # UNC shares and Windows device namespaces.
    _safe_relative.map(lambda p: f"//server/share/{p}"),
    _safe_relative.map(lambda p: f"\\\\server\\share\\{p}"),
    _safe_relative.map(lambda p: f"//?/C:/{p}"),
    _safe_relative.map(lambda p: f"//./{p}"),
    # Traversal, leading/middle/trailing.
    _safe_relative.map(lambda p: "../" + p),
    st.tuples(_safe_relative, _safe_relative).map(lambda t: f"{t[0]}/../{t[1]}"),
    _safe_relative.map(lambda p: p + "/.."),
    st.just(".."),
    # Reserved DOS device names, bare or nested.
    st.sampled_from(
        ["NUL", "nul", "CON", "con", "AUX", "PRN", "CLOCK$", "COM1", "com3", "LPT9",
         "lpt1.txt", "con.txt", "NUL.log"]
    ),
    st.tuples(_safe_part, st.sampled_from(["CON", "nul.txt", "aux.log", "COM2"])).map(
        lambda t: f"{t[0]}/{t[1]}"
    ),
    # Backslash separators.
    st.lists(_safe_part, min_size=2, max_size=4).map("\\".join),
    # Trailing dot / space (Windows silently strips these).
    _safe_relative.map(lambda p: p + "."),
    _safe_relative.map(lambda p: p + " "),
    # NUL byte and control characters.
    _safe_relative.map(lambda p: p + "\x00"),
    _safe_relative.map(lambda p: "\x01" + p),
    _safe_relative.map(lambda p: p + "\x1f"),
    # Empty and current-directory components.
    _safe_relative.map(lambda p: "a//" + p),
    _safe_relative.map(lambda p: "./" + p),
    st.tuples(_safe_relative, _safe_relative).map(lambda t: f"{t[0]}/./{t[1]}"),
    st.just(""),
)


@given(path=_malicious)
def test_safe_relative_rejects_malicious_paths(path: str) -> None:
    assert TurnExecutionJournal._safe_relative(path) is None


def test_safe_relative_rejects_bare_current_directory() -> None:
    assert TurnExecutionJournal._safe_relative(".") is None


def test_single_component_is_a_valid_relative_path() -> None:
    assert TurnExecutionJournal._safe_relative("a") == "a"


def test_workspace_root_itself_is_never_a_contained_target(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert not _contained(root, root)
    assert not _contained(root / ".", root)


@given(path=_safe_relative)
def test_safe_relative_accepts_canonical_paths(path: str) -> None:
    assert TurnExecutionJournal._safe_relative(path) == path


@given(rel=_safe_relative)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_contained_accepts_workspace_children(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "root"
    assert _contained(root / rel, root)
    assert _contained(root / "deep" / rel, root)


@given(rel=_safe_relative)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_contained_rejects_escapes(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "root"
    assert not _contained(tmp_path / "elsewhere" / rel, root)
    assert not _contained(root / ".." / "elsewhere" / rel, root)
    assert not _contained(Path(root.anchor) / rel, root)
    assert not _contained(f"D:/{rel}", root)


@given(rel=_safe_relative)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checked_path_accepts_real_children(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    checked = checked_path(root / rel, root)
    assert checked == Path(os.path.abspath(root / rel))


@given(rel=_safe_relative)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checked_path_rejects_paths_outside_root(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    with pytest.raises(OSError):
        checked_path(tmp_path / "outside" / rel, root)
    with pytest.raises(OSError):
        checked_path(root / ".." / "sibling" / rel, root)


_REDIRECT_COUNTER = itertools.count()


@given(rel=_safe_relative)
@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checked_path_rejects_symlink_or_junction_component(
    tmp_path: Path, directory_redirect, rel: str
) -> None:
    case = next(_REDIRECT_COUNTER)
    root = tmp_path / f"root-{case}"
    root.mkdir()
    victim = tmp_path / f"victim-{case}"
    victim.mkdir()
    (victim / rel.split("/")[-1]).write_text("keep", encoding="utf-8")
    directory_redirect(root / "link", victim)
    with pytest.raises(OSError):
        checked_path(root / "link" / rel, root)
    # The redirect target's contents are never touched by the check itself.
    assert (victim / rel.split("/")[-1]).read_text(encoding="utf-8") == "keep"
