"""UI adapter that records Agent telemetry for benchmark reporting."""

from __future__ import annotations

from typing import Any

from agent_core.models import ToolResult
from agent_core.ui import AgentUI, NullUI, PermissionChoice


class BenchmarkRecordingUI(AgentUI):
    """Delegate display events while retaining a machine-readable run summary."""

    def __init__(self, inner: AgentUI | None = None) -> None:
        self.inner = inner or NullUI()
        self.is_live = self.inner.is_live
        self.stats: dict[str, Any] = {}
        self.final_answer = ""
        self.tool_events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None

    def on_turn_start(self) -> None:
        self.inner.on_turn_start()

    def on_text_delta(self, text: str) -> None:
        self.inner.on_text_delta(text)

    def on_thinking_delta(self, text: str) -> None:
        self.inner.on_thinking_delta(text)

    def on_tool_args_delta(self, tool_name: str, partial_json: str) -> None:
        self.inner.on_tool_args_delta(tool_name, partial_json)

    def on_thinking(self, text: str) -> None:
        self.inner.on_thinking(text)

    def on_reasoning(self, text: str) -> None:
        self.inner.on_reasoning(text)

    def on_tool_call(self, tool_name: str, risk: str, arguments: dict[str, Any], label: str | None = None) -> None:
        self.tool_events.append({"tool": tool_name, "risk": risk})
        self.inner.on_tool_call(tool_name, risk, arguments, label)

    def on_tool_result(self, result: ToolResult, diff: str | None = None) -> None:
        self.inner.on_tool_result(result, diff)

    def on_final(self, answer: str) -> None:
        self.final_answer = answer
        self.inner.on_final(answer)

    def on_todos(self, todos: list[Any]) -> None:
        self.inner.on_todos(todos)

    def on_tool_use_summary(self, label: str, tool_names: list[str]) -> None:
        self.inner.on_tool_use_summary(label, tool_names)

    def on_stopped(self, reason: str, human: str) -> None:
        self.stop_reason = reason
        self.inner.on_stopped(reason, human)

    def on_token_usage(self, usage: dict[str, Any]) -> None:
        self.inner.on_token_usage(usage)

    def on_run_completed(self, stats: dict[str, Any]) -> None:
        self.stats = dict(stats)
        self.inner.on_run_completed(stats)

    def on_compaction_start(self, reactive: bool) -> None:
        self.inner.on_compaction_start(reactive)

    def on_compaction_progress(self, fraction: float, stage: str) -> None:
        self.inner.on_compaction_progress(fraction, stage)

    def on_compaction_end(self, before_chars: int, after_chars: int, detail: str, reactive: bool) -> None:
        self.inner.on_compaction_end(before_chars, after_chars, detail, reactive)

    def bind_event_loop(self, loop: Any) -> None:
        self.inner.bind_event_loop(loop)

    def confirm_tool(self, tool_name: str, risk: str, arguments: dict[str, Any]) -> PermissionChoice:
        return self.inner.confirm_tool(tool_name, risk, arguments)

    def request_permission(self, request: Any) -> Any:
        return self.inner.request_permission(request)

    def confirm_action(self, message: str) -> bool:
        return self.inner.confirm_action(message)

    async def pick_model(self, current_model: str, current_effort: str | None, spec: Any) -> Any:
        return await self.inner.pick_model(current_model, current_effort, spec)

    async def pick_permission_mode(self, current_mode: str, forbidden_modes: tuple[str, ...] = ()) -> str | None:
        return await self.inner.pick_permission_mode(current_mode, forbidden_modes)

    async def ask_questions(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.inner.ask_questions(questions)
