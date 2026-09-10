"""Export model changes as an official SWE-bench prediction patch.

This module deliberately keeps its own lenient header scanner instead of reusing
``agent_core.unified_diff.parse_unified_diff``: benchmark patches are git-generated
output that may contain renames or binary diffs, which the canonical parser rejects
by design. Nothing here authorizes or applies patches — it only exports and sanity
checks text for submission, so the strict shared parser stays the single authority
on the apply_patch security path.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any


@dataclass(frozen=True, slots=True)
class PatchExport:
    patch: str
    changed_files: tuple[str, ...]
    valid: bool
    error: str | None = None


def _git(workspace: Path, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        check=check,
        timeout=120,
    )


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _tracked_diff(workspace: Path) -> str:
    # ``prepare_repository`` records an immutable baseline ref.  Falling back
    # to HEAD keeps this helper useful for ordinary git workspaces and fixtures.
    baseline = "HEAD"
    ref = _git(workspace, ["rev-parse", "--verify", "refs/polaris/swebench-baseline"])
    if ref.returncode == 0 and ref.stdout.strip():
        baseline = _decode(ref.stdout).strip()
    result = _git(workspace, ["diff", "--binary", "--full-index", "--no-ext-diff", baseline, "--"])
    if result.returncode not in (0, 1):
        raise RuntimeError(_decode(result.stderr).strip() or "git diff failed")
    return _decode(result.stdout)


def _untracked_files(workspace: Path) -> list[str]:
    result = _git(workspace, ["ls-files", "--others", "--exclude-standard", "-z"])
    if result.returncode != 0:
        raise RuntimeError(_decode(result.stderr).strip() or "git ls-files failed")
    return [item for item in _decode(result.stdout).split("\0") if item]


def _untracked_diff(workspace: Path, relative: str) -> str:
    # git diff --no-index uses exit code 1 for a real difference.  /dev/null is
    # understood by Git on Windows as well as POSIX when passed as an argument.
    result = _git(
        workspace,
        ["diff", "--no-index", "--binary", "--full-index", "--no-ext-diff", "--", "/dev/null", relative],
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(_decode(result.stderr).strip() or f"cannot diff untracked file {relative}")
    return _decode(result.stdout)


def export_patch(workspace: str | Path, *, require_git: bool = True) -> PatchExport:
    """Return a deterministic diff containing tracked and untracked changes.

    The workspace is expected to have a synthetic baseline commit.  No gold patch
    or test patch is read by this function.
    """
    root = Path(workspace).resolve()
    try:
        tracked = _tracked_diff(root)
        untracked = [_untracked_diff(root, item) for item in _untracked_files(root)]
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        if require_git:
            return PatchExport("", (), False, f"patch export failed: {exc}")
        return PatchExport("", (), False, str(exc))
    chunks = [chunk for chunk in [tracked, *untracked] if chunk]
    patch = "".join(chunks)
    changed = tuple(_changed_paths(patch))
    return PatchExport(patch, changed, True)


def _changed_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].split("\t", 1)[0]
            if path != "/dev/null" and path not in paths:
                paths.append(path)
        elif line.startswith("--- a/") and "+++ " not in line:
            path = line[6:].split("\t", 1)[0]
            if path != "/dev/null" and path not in paths:
                paths.append(path)
    return paths


def validate_patch_text(patch: str) -> tuple[bool, str | None]:
    """Validate path safety and basic unified-diff structure before submission."""
    if not patch:
        return True, None
    saw_file = False
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")):
            saw_file = True
            raw = line[4:].split("\t", 1)[0].strip()
            if raw == "/dev/null":
                continue
            candidate = raw[2:] if raw[:2] in {"a/", "b/"} else raw
            path = Path(candidate)
            windows_path = PureWindowsPath(candidate)
            if (
                path.is_absolute()
                or windows_path.is_absolute()
                or windows_path.drive
                or candidate.startswith(("../", "..\\"))
                or ".." in path.parts
                or ".." in windows_path.parts
            ):
                return False, f"patch path escapes workspace: {candidate}"
            if "\x00" in candidate:
                return False, "patch contains a NUL path"
    if not saw_file:
        return False, "patch contains no file headers"
    return True, None


def save_patch(path: str | Path, patch: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patch, encoding="utf-8", errors="replace")
    return target


def prediction_record(instance_id: str, patch: str, *, model_name_or_path: str) -> dict[str, Any]:
    """Build the exact record shape consumed by the official Harness."""
    return {
        "instance_id": str(instance_id),
        "model_name_or_path": str(model_name_or_path),
        "model_patch": patch,
    }


def append_prediction(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def upsert_prediction(path: str | Path, record: dict[str, Any]) -> None:
    """Write one canonical prediction per instance while supporting resume/retry.

    The official Harness collapses JSONL rows into an ``instance_id`` mapping,
    but duplicate rows make artifacts ambiguous and can leave an old empty patch
    ahead of a successful retry.  This helper preserves other valid rows and
    atomically replaces the target after removing the record being updated.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    instance_id = str(record.get("instance_id", ""))
    rows: list[dict[str, Any]] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and str(value.get("instance_id", "")) != instance_id:
            rows.append(value)
    rows.append(dict(record))
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows),
        encoding="utf-8",
    )
    temporary.replace(target)
