from __future__ import annotations

import hashlib
import os
from pathlib import Path

from agent_core.codeintel.budget import QueryBudget
from agent_core.codeintel.models import FileVersion, StaleEvidence
from agent_core.permission_safety import is_secret_path

IGNORED_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
                         ".ruff_cache", "node_modules", ".venv", "venv", "env", "dist", "build",
                         ".idea", ".vscode", "runs", "memory", ".polaris"})


def worktree_id(root: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()[:24]


def contained(root: Path, raw: str, *, secrets: bool = False) -> Path:
    candidate = root / raw
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("code path escapes workspace")
    # Reject redirected ancestors, including Windows junctions.
    for component in [candidate, *candidate.parents]:
        if component == root or not component.is_relative_to(root):
            break
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError("redirected code paths are not indexed")
    if not secrets and is_secret_path(resolved):
        raise ValueError("secret paths are excluded from code indexing")
    return resolved


def read_snapshot(root: Path, raw: str, budget: QueryBudget, *, revision: int = 0,
                  allow_secret: bool = False) -> tuple[bytes, FileVersion]:
    path = contained(root, raw, secrets=allow_secret)
    from agent_core.tools.base import current_execution_context
    context = current_execution_context()
    view = context.workspace_view if context else None
    relative = path.relative_to(root).as_posix()
    if view is not None and root == view.execution_root() and hasattr(view, "resolve_read_path"):
        path = view.resolve_read_path(relative)
        contained(view.workspace if path.is_relative_to(view.workspace) else root, str(path), secrets=allow_secret)
    before = path.stat()
    if before.st_size > budget.config.max_file_bytes:
        raise BudgetError("file_size_limit")
    budget.consume(files=1)
    data = bytearray()
    with path.open("rb") as handle:
        while True:
            budget.check()
            chunk = handle.read(min(65536, budget.config.max_file_bytes - len(data) + 1))
            if not chunk:
                break
            budget.consume(bytes=len(chunk))
            data.extend(chunk)
            if len(data) > budget.config.max_file_bytes:
                raise BudgetError("file_size_limit")
    after = path.stat()
    if (before.st_mtime_ns, before.st_size, before.st_ino) != (after.st_mtime_ns, after.st_size, after.st_ino):
        raise StaleEvidence(f"file changed during read: {raw}")
    digest = hashlib.sha256(data).hexdigest()
    if view is not None and root == view.execution_root() and hasattr(view, "observe_query_read"):
        view.observe_query_read(relative, digest)
    version = FileVersion(worktree_id(root), relative, digest, revision)
    return bytes(data), version


class BudgetError(ValueError):
    pass


def verify_expected(root: Path, path: Path, expected: object, *, required: bool = False) -> None:
    if expected is None:
        if required and path.exists():
            raise StaleEvidence("a verified read is required before editing")
        return
    if not isinstance(expected, dict):
        raise StaleEvidence("expected_version must be a version object")
    relative = path.relative_to(root).as_posix()
    if expected.get("path", relative) != relative:
        raise StaleEvidence("version belongs to another file")
    if expected.get("sha256") is None:
        if expected.get("exists") is False and not path.exists():
            return
        raise StaleEvidence("expected_version has no content hash")
    digest = hashlib.sha256()
    if not path.is_file():
        raise StaleEvidence(f"file disappeared: {relative}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        raise StaleEvidence(f"file changed since retrieval: {relative}; read it again before editing")


def edit_precondition(root: Path, path: Path, expected: object = None) -> None:
    from agent_core.tools.base import current_execution_context

    context = current_execution_context()
    logical_root = context.logical_workspace if context and context.logical_workspace else root
    relative = path.relative_to(root).as_posix()
    if expected is None and context and context.read_versions is not None:
        expected = context.read_versions.get(str((logical_root / relative).resolve()))
    if isinstance(expected, dict) and expected.get("worktree_id") not in {None, worktree_id(logical_root)}:
        raise StaleEvidence("version belongs to another worktree")
    verify_expected(root, path, expected, required=bool(context and context.strict_versions))
