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
    description = ("Query definitions, references, symbols, hover, call hierarchy, implementations, or diagnostics. "
                   "line/character are zero-based UTF-16 positions. With code intelligence enabled, definition, "
                   "references and implementation return versioned location hints; validate edit anchors with code_validate.")
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
            "mode": {"type": "string", "enum": ["raw", "structured"],
                     "description": "document_symbols only: structured integrates versioned candidate evidence; raw keeps the legacy payload."},
        },
        "required": ["operation"],
    }
    risk = ToolRisk.READ

    def concurrency_spec(self, arguments: dict[str, object]) -> ConcurrencySpec:
        return ConcurrencySpec((ResourceLock("fs", str(self.session.workspace.resolve()), "read", subtree=True),))

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
            from agent_core.codeintel.lsp_evidence import OPERATIONS, query_document_symbols, query_evidence
            from agent_core.codeintel.runtime import codeintel_enabled
            operation = str(arguments.get("operation"))
            structured_symbols = operation == "document_symbols" and arguments.get("mode") == "structured"
            if codeintel_enabled(self.session) and (operation in OPERATIONS or structured_symbols):
                if not arguments.get("path"):
                    raise ValueError("path is required for LSP evidence")
                if structured_symbols:
                    page = await query_document_symbols(self.session, manager, path=str(arguments["path"]))
                else:
                    page = await query_evidence(self.session, manager, operation,
                        path=str(arguments["path"]), line=int(str(arguments.get("line", 0))),
                        character=int(str(arguments.get("character", 0))))
                body = page.to_dict()
                return ToolResult(self.name, json.dumps(body, ensure_ascii=False), metadata={
                    "codeintel": body, "code_recovery": {"tool": self.name, "arguments": dict(arguments)}})
            response = await manager.request(
                operation,
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
