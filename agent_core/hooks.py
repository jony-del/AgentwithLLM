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

    # Coding-agent sized: ordinary source/doc files (up to ~2000 lines, like Claude
    # Code's Read) return in full; only genuinely huge output spills to disk. Set
    # too low and the model loses the file content it just read.
    max_lines: int = 2000
    max_chars: int = 50000
    head_lines: int = 20
    tail_lines: int = 20
    spill: bool = True
    # When ``pointer`` is on (default), an oversized result is replaced in live context by
    # a structured preview pointer (head ``preview_chars`` + an on-disk path the model can
    # page back with ``read_text_file``) instead of the legacy head+tail truncation.
    # Set ``pointer=False`` to keep the byte-for-byte legacy behavior.
    preview_chars: int = 4000
    pointer: bool = True

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
    output, a **char** budget) the full, untouched text is written to a file and the
    live-context result is shrunk.

    Two shrink modes:

    - ``pointer=True`` (default): replace the result with a structured ``<tool_output_ref>``
      preview — the head ``preview_chars`` of the output plus the on-disk path and an
      explicit instruction to page the rest back with ``read_text_file``. This is the
      Open-ClaudeCode ``toolResultStorage`` shape: a small, machine-recognizable pointer
      (metadata ``tool_result_ref``/``spilled``) instead of a wall of truncated text.
    - ``pointer=False``: the legacy behavior — keep head+tail, note how much was dropped,
      and append the spill path. Kept as a byte-for-byte regression escape hatch.

    The decision is frozen at execution time: this hook runs once per tool result (in the
    executor's ``run_post``) *before* the result becomes a ``Message``, and the compaction
    pipeline never re-runs it — so a spilled result is not rewritten on later turns and the
    prompt-cache prefix stays stable. The ``spilled`` metadata guard makes that idempotence
    explicit against accidental double-invocation.
    """

    #: Tools whose output is never spilled/pointer-ised. ``read_text_file`` is the
    #: designated pager, so truncating its result into a pointer that points back at
    #: itself would be circular — exempt it so paging always returns real content.
    DEFAULT_EXEMPT_TOOLS = frozenset({"read_text_file"})

    def __init__(
        self,
        max_lines: int = 2000,
        max_chars: int = 50000,
        head_lines: int | None = None,
        tail_lines: int | None = None,
        spill_dir: str | Path = "runs/outputs",
        spill: bool = True,
        preview_chars: int = 4000,
        pointer: bool = True,
        exempt_tools: frozenset[str] | None = None,
    ) -> None:
        self.max_lines = max_lines
        self.max_chars = max_chars
        # Keep head+tail comfortably below the threshold so crossing it is a real
        # reduction, not a near-no-op at the boundary.
        self.head_lines = head_lines if head_lines is not None else max(1, max_lines // 5)
        self.tail_lines = tail_lines if tail_lines is not None else max(1, max_lines // 5)
        self.spill_dir = Path(spill_dir)
        self.spill = spill
        self.preview_chars = preview_chars
        self.pointer = pointer
        self.exempt_tools = self.DEFAULT_EXEMPT_TOOLS if exempt_tools is None else exempt_tools

    @classmethod
    def from_config(cls, config: OutputLimitConfig, spill_dir: str | Path) -> "MaxOutputPostHook":
        return cls(
            max_lines=config.max_lines,
            max_chars=config.max_chars,
            head_lines=config.head_lines,
            tail_lines=config.tail_lines,
            spill_dir=spill_dir,
            spill=config.spill,
            preview_chars=config.preview_chars,
            pointer=config.pointer,
        )

    def after_tool(self, tool_call: ToolCall, result: ToolResult) -> ToolResult:
        # Freeze guard: an already-spilled result is never reprocessed (idempotent),
        # so re-invocation can't churn a frozen message and break the cache prefix.
        if result.metadata.get("spilled"):
            return result
        # The dedicated pager self-limits and supports offset/limit; never spill it,
        # or paging a large file would just hand back another pointer (circular).
        if tool_call.name in self.exempt_tools:
            return result

        content = result.content
        lines = content.splitlines()
        over_lines = len(lines) > self.max_lines
        over_chars = len(content) > self.max_chars
        if not (over_lines or over_chars):
            return result

        spill_path = self._spill(tool_call, content) if self.spill else None

        metadata: dict[str, object] = {
            **result.metadata,
            "post_hook": "max_output",
            "original_lines": len(lines),
            "original_chars": len(content),
            "full_output_path": str(spill_path) if spill_path is not None else None,
        }

        if self.pointer:
            body = self._build_pointer(spill_path, content, len(lines))
            metadata["spilled"] = True
            metadata["preview_chars"] = self.preview_chars
            metadata["tool_result_ref"] = str(spill_path) if spill_path is not None else None
        else:
            body = self._truncate_lines(lines) if over_lines else content
            body = self._truncate_chars(body)
            if spill_path is not None:
                body = (
                    f"{body}\n[full output saved to {spill_path} — "
                    f"{len(lines)} lines, {len(content)} chars]"
                )

        return ToolResult(name=result.name, ok=result.ok, content=body, metadata=metadata)

    def _build_pointer(self, spill_path: Path | None, content: str, original_lines: int) -> str:
        """Render the structured preview pointer that replaces an oversized result.

        The preview is the **head** of the output (``preview_chars``); the tail is not
        kept inline — the model pages it back from ``spill_path`` via ``read_text_file``
        when it needs it. Framed in ``<tool_output_ref>`` delimiters so preview text that
        looks like instructions can't hijack the turn.
        """
        original_chars = len(content)
        preview = content[: self.preview_chars]
        omitted = original_chars - len(preview)
        out = ["<tool_output_ref>"]
        if spill_path is not None:
            out.append(
                f"[truncated tool output — {original_lines} lines / {original_chars} chars; "
                f"full output at {spill_path}]"
            )
        else:
            out.append(
                f"[truncated tool output — {original_lines} lines / {original_chars} chars; "
                "full output not saved (spill disabled)]"
            )
        out.append(f"Preview (first ~{self.preview_chars} chars):")
        out.append(preview)
        if omitted > 0:
            out.append(f"…[omitted {omitted} chars]")
        if spill_path is not None:
            out.append(
                "To read the full output (including the tail), use "
                f'read_text_file("{spill_path}") with offset/limit to page through it.'
            )
        out.append("</tool_output_ref>")
        return "\n".join(out)

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

