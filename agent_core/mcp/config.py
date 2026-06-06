from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MCPServerConfig:
    """One MCP server, configured under ``[mcp.servers.<name>]`` in agent.toml.

    ``transport`` selects the connection mode: ``"stdio"`` launches ``command`` +
    ``args`` as a subprocess; ``"streamable-http"`` connects to ``url`` (the 2025
    Streamable HTTP transport). ``risk`` is the per-server override for the risk every
    tool from this server is registered with — defaulting to the safe ``"dangerous"``.
    """

    name: str = ""
    transport: str = "stdio"  # "stdio" | "streamable-http"
    # stdio transport
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    # streamable-http transport
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # permission risk this server's tools are registered with: read | write | dangerous
    risk: str = "dangerous"
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any] | None) -> "MCPServerConfig":
        from agent_core.config import overlay_dataclass

        return overlay_dataclass(cls(name=name), data)


@dataclass(slots=True)
class MCPConfig:
    servers: list[MCPServerConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MCPConfig":
        """Build from the ``[mcp]`` toml table.

        Servers live in the ``[mcp.servers.<name>]`` sub-tables, so ``data["servers"]``
        is a mapping of server-name -> body. A missing/oddly-typed table yields no
        servers (MCP simply stays off).
        """
        if not data:
            return cls()
        servers_table = data.get("servers")
        if not isinstance(servers_table, dict):
            return cls()
        servers = [
            MCPServerConfig.from_dict(name, body)
            for name, body in servers_table.items()
            if isinstance(body, dict)
        ]
        return cls(servers=servers)
