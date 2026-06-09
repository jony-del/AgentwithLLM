from __future__ import annotations

from typing import Any

from agent_core.mcp.config import MCPServerConfig
from agent_core.models import ToolResult, ToolRisk
from agent_core.tools.adapters import ToolAdapter
from agent_core.tools.base import ConcurrencySpec, ResourceLock, Tool

_RISK = {
    "read": ToolRisk.READ,
    "write": ToolRisk.WRITE,
    "dangerous": ToolRisk.DANGEROUS,
}


def _flatten_content(content: Any) -> str:
    """Render an MCP ``CallToolResult.content`` list to plain text for the agent.

    Text blocks contribute their ``.text``; non-text blocks (images, embedded
    resources) are noted by type so the observation stays readable.
    """
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[{getattr(block, 'type', 'content')} block]")
    return "\n".join(parts)


class MCPTool(Tool):
    """Wrap a single MCP server tool as a synchronous :class:`Tool`.

    The tool is registered as ``"<server>__<tool>"`` to avoid cross-server name
    collisions, and inherits the server's configured ``risk`` (default DANGEROUS) so the
    permission layer gates it correctly.
    """

    def __init__(self, manager: Any, server: MCPServerConfig, descriptor: Any) -> None:
        self.name = f"{server.name}__{descriptor.name}"
        self.description = (getattr(descriptor, "description", None) or "").strip()
        self.input_schema = getattr(descriptor, "inputSchema", None) or {
            "type": "object",
            "properties": {},
        }
        self.risk = _RISK.get((server.risk or "dangerous").lower(), ToolRisk.DANGEROUS)
        self._manager = manager
        self._server = server.name
        self._remote = descriptor.name

    def concurrency_spec(self, arguments: dict[str, Any]) -> ConcurrencySpec:
        if self.risk is ToolRisk.READ:
            # MCP sessions are long-lived async streams behind a sync wrapper; keep
            # calls to the same server serialized unless the client proves otherwise.
            return ConcurrencySpec((ResourceLock("mcp", self._server, "write"),))
        return ConcurrencySpec(exclusive=True)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self._manager.call_tool(self._server, self._remote, arguments)
        except Exception as exc:  # noqa: BLE001 - surface transport/timeout errors as a failed result
            return ToolResult(self.name, f"MCP tool error: {exc}", ok=False)
        text = _flatten_content(getattr(result, "content", None))
        # The result is an error if the server flagged it; tolerate both spellings.
        is_error = getattr(result, "isError", getattr(result, "is_error", False))
        return ToolResult(self.name, text, ok=not is_error, metadata={"mcp_server": self._server})


class MCPAdapter(ToolAdapter):
    """Expose every tool discovered by an :class:`MCPClientManager` as :class:`Tool`s."""

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    def list_tools(self) -> list[Tool]:
        return [MCPTool(self._manager, server, descriptor) for server, descriptor in self._manager.tools()]
