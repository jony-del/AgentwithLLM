"""Canonical parser for permission checks and execution of unified diffs.

The parser validates the complete patch before exposing any target paths.  Both the
central permission pipeline and the edit tool consume this representation, so a patch
cannot be authorized under one interpretation and executed under another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath


class PatchError(ValueError):
    """A patch is malformed or contains an unsafe/ambiguous target."""


class PatchOperation(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class PatchHunk:
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatchFile:
    old_path: str | None
    new_path: str | None
    operation: PatchOperation
    hunks: tuple[PatchHunk, ...]

    @property
    def target(self) -> str:
        target = self.new_path if self.operation is not PatchOperation.DELETE else self.old_path
        assert target is not None
        return target


@dataclass(frozen=True, slots=True)
class PatchSet:
    files: tuple[PatchFile, ...]

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(file.target for file in self.files)


def parse_unified_diff(patch_text: str) -> PatchSet:
    """Parse and validate a multi-file unified diff.

    Renames are deliberately rejected: supporting them requires first-class source and
    destination authorization plus collision semantics.  Callers can express the same
    change as an explicit delete and create.
    """

    if "\x00" in patch_text:
        raise PatchError("patch contains a NUL byte")
    lines = patch_text.splitlines()
    files: list[PatchFile] = []
    seen: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise PatchError("old-file header is not followed by a new-file header")

        old_path = _normalize_path(lines[index][4:])
        new_path = _normalize_path(lines[index + 1][4:])
        if old_path is None and new_path is None:
            raise PatchError("both patch paths are /dev/null")
        if old_path is None:
            operation = PatchOperation.CREATE
        elif new_path is None:
            operation = PatchOperation.DELETE
        else:
            if old_path != new_path:
                raise PatchError("renames are not supported; use delete plus create")
            operation = PatchOperation.UPDATE

        index += 2
        hunks: list[PatchHunk] = []
        while index < len(lines) and lines[index].startswith("@@"):
            index += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            while index < len(lines):
                body = lines[index]
                if body.startswith(("@@", "--- ", "diff --git ")):
                    break
                if body.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not body or body[0] not in {" ", "+", "-"}:
                    raise PatchError(f"invalid hunk line at line {index + 1}")
                marker, content = body[0], body[1:]
                if marker in {" ", "-"}:
                    old_lines.append(content)
                if marker in {" ", "+"}:
                    new_lines.append(content)
                index += 1
            hunks.append(PatchHunk(tuple(old_lines), tuple(new_lines)))

        if not hunks:
            raise PatchError("file entry contains no hunks")
        patch_file = PatchFile(old_path, new_path, operation, tuple(hunks))
        identity = patch_file.target.casefold()
        if identity in seen:
            raise PatchError(f"duplicate patch target: {patch_file.target}")
        seen.add(identity)
        files.append(patch_file)

    if not files:
        raise PatchError("patch contained no file hunks")
    return PatchSet(tuple(files))


def _normalize_path(raw: str) -> str | None:
    path = raw.strip().split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    path = path.replace("\\", "/")
    if not path:
        raise PatchError("patch path is empty")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise PatchError(f"absolute patch path is not allowed: {path}")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise PatchError(f"patch path must stay inside the workspace: {path}")
    return posix.as_posix()
