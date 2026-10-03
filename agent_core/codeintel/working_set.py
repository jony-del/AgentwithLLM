from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import Any

from agent_core.codeintel.models import CodeHit


class TaskWorkingSet:
    """Bounded evidence references, independent of history and long-term memory."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple[str, int | None, int | None], dict[str, Any]] = OrderedDict()

    def add(self, hit: CodeHit, *, reason: str = "retrieval") -> None:
        key = (hit.path, hit.start_line, hit.end_line)
        self.entries.pop(key, None)
        self.entries[key] = {"path": hit.path, "start_line": hit.start_line, "end_line": hit.end_line,
                             "version": asdict(hit.version), "module": hit.module, "reason": reason}
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def invalidate(self, paths: tuple[str, ...]) -> None:
        changed = set(paths)
        for key in list(self.entries):
            if key[0] in changed:
                self.entries.pop(key)

    def modules(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(e["module"]) for e in reversed(list(self.entries.values()))))

    def render_references(self, max_bytes: int = 4000) -> str:
        lines = ["Code evidence references (historical; revalidate with code_context before editing):"]
        used = len(lines[0].encode())
        for entry in reversed(list(self.entries.values())):
            line = f"{entry['path']}:{entry['start_line']}-{entry['end_line']} sha256={entry['version']['sha256']}"
            size = len(line.encode()) + 1
            if used + size > max_bytes:
                break
            lines.append(line)
            used += size
        return "\n".join(lines) if len(lines) > 1 else ""
