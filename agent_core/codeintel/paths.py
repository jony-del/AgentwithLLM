"""Component-aware glob matching without filesystem traversal."""
from __future__ import annotations

from functools import lru_cache
import fnmatch
import os

from agent_core.codeintel.snapshots import IGNORED_DIRS


def excluded_path(path: str, roots: tuple[str, ...], *, directory: bool = False) -> bool:
    """One inventory policy for catalog, cold scans, overlays and change events.

    Generated/runtime roots are relative to the workspace. In particular a source
    directory named ``pkg/memory`` or ``src/build`` is not a runtime directory.
    Explicit reads remain governed by the separate read/secret/redirect policy.
    """
    normalized = os.path.normcase(path.replace("\\", "/")).replace("\\", "/")
    parts = normalized.split("/")
    if any(part in IGNORED_DIRS for part in (parts if directory else parts[:-1])):
        return True
    return any(normalized == root or normalized.startswith(root + "/")
               for raw in roots for root in [os.path.normcase(raw).replace("\\", "/")])


def glob_match(path: str, pattern: str) -> bool:
    parts = tuple(path.replace("\\", "/").split("/"))
    patterns = tuple(pattern.replace("\\", "/").split("/"))

    @lru_cache(maxsize=4096)
    def match(i: int, j: int) -> bool:
        if j == len(patterns):
            return i == len(parts)
        if patterns[j] == "**":
            return match(i, j + 1) or (i < len(parts) and match(i + 1, j))
        return i < len(parts) and fnmatch.fnmatchcase(parts[i], patterns[j]) and match(i + 1, j + 1)

    return match(0, 0)
