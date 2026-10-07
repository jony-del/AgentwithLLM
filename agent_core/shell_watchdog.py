"""One-shot interactive-input diagnosis; silence alone is never a failure."""

from __future__ import annotations

from dataclasses import dataclass
import re


_PROMPTS = tuple(re.compile(pattern, re.I) for pattern in (
    r"\(y/n\)", r"\[y/n\]", r"\(yes/no\)",
    r"\b(?:Do you|Would you|Shall I|Are you sure|Ready to)\b.*\?\s*$",
    r"Press (?:any key|Enter)", r"Continue\?", r"Overwrite\?",
))


def looks_like_prompt(tail: str) -> bool:
    lines = tail.rstrip().splitlines()
    return bool(lines and any(pattern.search(lines[-1]) for pattern in _PROMPTS))


@dataclass(slots=True)
class ShellStallWatchdog:
    threshold: float = 45.0
    tail_bytes: int = 1024
    notified: bool = False

    def check(self, *, now: float, last_output_at: float, preview: bytes) -> str | None:
        if self.notified or now - last_output_at < self.threshold:
            return None
        tail = preview[-self.tail_bytes:].decode("utf-8", errors="replace")
        if not looks_like_prompt(tail):
            return None
        self.notified = True
        return tail
