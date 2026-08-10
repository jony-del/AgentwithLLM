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
    headers_helper: str = ""
    oauth: dict[str, Any] = field(default_factory=dict)
    user_config: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    always_load: bool = False
    roots: list[str] = field(default_factory=list)
    notifications: bool = True
    # permission risk this server's tools are registered with: read | write | dangerous
    risk: str = "dangerous"
    enabled: bool = True
    # Set by host discovery/activation code, never trusted from remote annotations.
    discovered: bool = False
    trust_tier: str = "local_user_declared"
    network_policy: str = "default"  # default | public-only

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any] | None) -> "MCPServerConfig":
        from agent_core.config import overlay_dataclass

        raw = dict(data or {})
        aliases = {
            "type": "transport", "headersHelper": "headers_helper",
            "userConfig": "user_config", "alwaysLoad": "always_load",
        }
        for source, target in aliases.items():
            if source in raw and target not in raw:
                raw[target] = raw[source]
        if "timeout" in raw:
            try:
                timeout = float(raw["timeout"])
                # Claude's JSON shape stores milliseconds; the legacy Polaris TOML
                # shape (which uses ``transport``) keeps its historical seconds.
                claude_shape = any(key in data for key in aliases) if data else False
                raw["timeout"] = timeout / 1000 if claude_shape else timeout
            except (TypeError, ValueError):
                raw["timeout"] = 60.0
        return overlay_dataclass(cls(name=name), raw)


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
