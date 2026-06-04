from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_core.models import ToolCall, ToolResult


@dataclass(slots=True)
class OutputLimitConfig:
    """Limits for :class:`MaxOutputPostHook`, configurable via the ``[output]`` toml table."""

    max_lines: int = 100
    max_chars: int = 8000
    head_lines: int = 20
    tail_lines: int = 20
    spill: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputLimitConfig":
        from agent_core.config import overlay_dataclass

        return overlay_dataclass(cls(), data)


@dataclass(slots=True)
class HookResult:
    allowed: bool = True
    reason: str | None = None
    tool_call: ToolCall | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class PreToolHook(Protocol):
    def before_tool(self, tool_call: ToolCall) -> HookResult:
        ...


class PostToolHook(Protocol):
    def after_tool(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        ...


class HookPipeline:
    def __init__(
        self,
        pre_hooks: list[PreToolHook] | None = None,
        post_hooks: list[PostToolHook] | None = None,
    ) -> None:
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []

    def run_pre(self, tool_call: ToolCall) -> tuple[ToolCall, list[HookResult]]:
        current = tool_call
        results: list[HookResult] = []
        for hook in self.pre_hooks:
            result = hook.before_tool(current)
            results.append(result)
            if not result.allowed:
                return current, results
            if result.tool_call:
                current = result.tool_call
        return current, results

    def run_post(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        current = result
        for hook in self.post_hooks:
            current = hook.after_tool(tool_call, current)
        return current


class MaxOutputPostHook:
    """Truncate oversized tool output, spilling the full text to disk.

    A single tool call (e.g. a noisy shell command) can emit thousands of lines.
    Feeding all of that back to the model wastes context and money, so when the
    output exceeds a **line** budget (or, as a fallback for pathological single-line
    output, a **char** budget) this keeps only the head and tail — with a notice of
    how much was dropped — and writes the complete, untouched output to a file. The
    file path is appended to the truncated result so the full text stays retrievable
    (and is recorded in the run log via the result's metadata).
    """

    def __init__(
        self,
        max_lines: int = 100,
        max_chars: int = 8000,
        head_lines: int | None = None,
        tail_lines: int | None = None,
        spill_dir: str | Path = "runs/outputs",
        spill: bool = True,
    ) -> None:
        self.max_lines = max_lines
        self.max_chars = max_chars
        # Keep head+tail comfortably below the threshold so crossing it is a real
        # reduction, not a near-no-op at the boundary.
        self.head_lines = head_lines if head_lines is not None else max(1, max_lines // 5)
        self.tail_lines = tail_lines if tail_lines is not None else max(1, max_lines // 5)
        self.spill_dir = Path(spill_dir)
        self.spill = spill

    @classmethod
    def from_config(cls, config: OutputLimitConfig, spill_dir: str | Path) -> "MaxOutputPostHook":
        return cls(
            max_lines=config.max_lines,
            max_chars=config.max_chars,
            head_lines=config.head_lines,
            tail_lines=config.tail_lines,
            spill_dir=spill_dir,
            spill=config.spill,
        )

    def after_tool(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        content = result.content
        lines = content.splitlines()
        over_lines = len(lines) > self.max_lines
        over_chars = len(content) > self.max_chars
        if not (over_lines or over_chars):
            return result

        spill_path = self._spill(tool_call, content) if self.spill else None

        body = self._truncate_lines(lines) if over_lines else content
        body = self._truncate_chars(body)
        if spill_path is not None:
            body = (
                f"{body}\n[full output saved to {spill_path} — "
                f"{len(lines)} lines, {len(content)} chars]"
            )

        return ToolResult(
            name=result.name,
            ok=result.ok,
            content=body,
            metadata={
                **result.metadata,
                "post_hook": "max_output",
                "original_lines": len(lines),
                "original_chars": len(content),
                "full_output_path": str(spill_path) if spill_path is not None else None,
            },
        )

    def _truncate_lines(self, lines: list[str]) -> str:
        omitted = len(lines) - self.head_lines - self.tail_lines
        if omitted <= 0:
            return "\n".join(lines)
        head = lines[: self.head_lines]
        tail = lines[len(lines) - self.tail_lines :]
        return "\n".join([*head, f"[... omitted {omitted} lines ...]", *tail])

    def _truncate_chars(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        half = self.max_chars // 2
        omitted = len(text) - 2 * half
        return f"{text[:half]}\n[... omitted {omitted} characters ...]\n{text[-half:]}"

    def _spill(self, tool_call: ToolCall, content: str) -> Path:
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        raw_name = getattr(tool_call, "name", "") or "tool"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_name)[:40]
        path = self.spill_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}-{uuid.uuid4().hex[:8]}.txt"
        path.write_text(content, encoding="utf-8")
        return path

