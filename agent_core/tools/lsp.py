from __future__ import annotations

import json

from agent_core.lsp import LSPManager
from agent_core.models import ToolRisk, ToolResult
from agent_core.sandbox import SandboxAwareMixin
from agent_core.session import SessionAwareMixin
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool
from agent_core.tools.catalog import builtin_tool


@builtin_tool
class LSPTool(SessionAwareMixin, SandboxAwareMixin, Tool):
    name = "lsp"
    description = "Query definitions, references, symbols, hover, call hierarchy, implementations, or diagnostics."
    deferred = True
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["definition", "references", "hover", "document_symbols", "workspace_symbols", "implementation", "prepare_call_hierarchy", "incoming_calls", "outgoing_calls", "diagnostics"],
            },
            "path": {"type": "string"},
            "line": {"type": "integer", "minimum": 0},
            "character": {"type": "integer", "minimum": 0},
            "query": {"type": "string"},
        },
        "required": ["operation"],
    }
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        path = str(arguments.get("path", self.session.workspace))
        return ConcurrencySpec((ResourceLock("lsp", path, "read"),))

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        config = self.session.tool_suite.lsp if self.session.tool_suite is not None else None
        if config is None or not config.servers:
            return ToolResult(self.name, "No LSP servers are configured.", ok=False)
        if not self.sandbox.is_enabled():
            return ToolResult(
                self.name,
                "LSP startup refused because no enforcing sandbox backend is active.",
                ok=False,
            )
        manager = self.session.lsp_manager
        if manager is None:
            manager = LSPManager(
                config, self.session.workspace,
                event_sink=self.session.audit_event,
                sandbox=self.sandbox,
            )
            self.session.lsp_manager = manager
        try:
            response = await manager.request(
                str(arguments.get("operation", "")),
                path=str(arguments["path"]) if arguments.get("path") else None,
                line=int(str(arguments.get("line", 0))),
                character=int(str(arguments.get("character", 0))),
                query=str(arguments.get("query", "")),
            )
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            return ToolResult(self.name, f"LSP request failed: {exc}", ok=False)
        content = json.dumps(response, ensure_ascii=False, indent=2, default=str)
        if len(content) > 100_000:
            content = content[:100_000] + "\n[LSP response truncated]"
        return ToolResult(self.name, content)
