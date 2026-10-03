"""Component-aware glob matching without filesystem traversal."""
from __future__ import annotations

from functools import lru_cache
import fnmatch


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
