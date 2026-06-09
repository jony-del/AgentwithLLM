from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


def coerce_to_type(declared_type: Any, value: Any) -> Any:
    """Coerce a raw (often string) value to a dataclass field's declared type.

    With ``from __future__ import annotations`` a field's ``type`` is its *name* as
    a string (e.g. ``"int"``), so we match on those. Anything we don't recognise is
    passed through untouched.
    """
    if declared_type in ("bool", bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if declared_type in ("int", int):
        return int(value)
    if declared_type in ("float", float):
        return float(value)
    return value


def overlay_dataclass(config: _T, data: dict[str, Any] | None) -> _T:
    """Mutate ``config`` in place from a mapping (e.g. a toml table), then return it.

    Unknown keys are ignored and absent fields keep their defaults, so a partial or
    forward-compatible table loads cleanly. Values are coerced to each field's type.
    """
    if not data:
        return config
    valid = {field.name: field.type for field in fields(config)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key in valid:
            setattr(config, key, coerce_to_type(valid[key], value))
    return config


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _unquote_env_value(value: str) -> str:
    value = _strip_inline_comment(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = _unquote_env_value(value.strip())


def load_agent_toml(path: str | Path = "agent.toml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}
    with config_path.open("rb") as file:
        return tomllib.load(file)


def resolve_config(
    cli_values: dict[str, Any],
    config_file: str | Path = "agent.toml",
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    load_dotenv(env_file)
    file_values = load_agent_toml(config_file)
    defaults = {
        "model": "claude-sonnet-4-6",
        "permission": "default",
        "provider": "claude",
    }
    env_values = {
        "model": os.getenv("AGENT_MODEL"),
        "permission": os.getenv("AGENT_PERMISSION"),
        "provider": os.getenv("AGENT_PROVIDER"),
    }
    # Only scalar settings are merged here; the ``[memory]`` table is resolved
    # separately (see resolve_memory_config) so its nested fields don't collide
    # with the flat keys above.
    scalar_file_values = {key: value for key, value in file_values.items() if not isinstance(value, dict)}
    merged = {**defaults, **scalar_file_values}
    merged.update({key: value for key, value in env_values.items() if value is not None})
    merged.update({key: value for key, value in cli_values.items() if value is not None})
    return merged


def resolve_memory_config(cli_enabled: bool | None, config_file: str | Path = "agent.toml") -> "MemoryConfig":
    """Resolve a :class:`MemoryConfig` from, lowest to highest priority:
    defaults → ``[memory]`` toml table → ``AGENT_MEMORY`` env → ``--memory`` CLI flag.

    Only ``enabled`` is overridable by env/CLI; the numeric tunables live in toml.
    """
    from agent_core.memory.config import MemoryConfig

    table = load_agent_toml(config_file).get("memory")
    config = MemoryConfig.from_dict(table if isinstance(table, dict) else None)

    env = os.getenv("AGENT_MEMORY")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}
    if cli_enabled is not None:
        config.enabled = cli_enabled
    return config


def resolve_output_config(config_file: str | Path = "agent.toml") -> "OutputLimitConfig":
    """Resolve the tool-output truncation limits from the ``[output]`` toml table."""
    from agent_core.hooks import OutputLimitConfig

    table = load_agent_toml(config_file).get("output")
    return OutputLimitConfig.from_dict(table if isinstance(table, dict) else None)


def resolve_concurrency_config(config_file: str | Path = "agent.toml") -> dict[str, Any]:
    """Resolve resource-level tool concurrency settings from ``[concurrency]``."""
    table = load_agent_toml(config_file).get("concurrency")
    values = {"parallel_tools": True, "max_tool_workers": 4}
    if not isinstance(table, dict):
        return values
    if "parallel_tools" in table:
        values["parallel_tools"] = coerce_to_type(bool, table["parallel_tools"])
    if "max_tool_workers" in table:
        values["max_tool_workers"] = max(1, coerce_to_type(int, table["max_tool_workers"]))
    return values


def resolve_mcp_config(config_file: str | Path = "agent.toml") -> "MCPConfig":
    """Resolve the MCP servers from the ``[mcp]`` toml table (``[mcp.servers.<name>]``).

    Returns an empty config (no servers) when the table is absent, so MCP stays off and
    the ``mcp`` SDK is never imported unless a server is actually configured.
    """
    from agent_core.mcp.config import MCPConfig

    table = load_agent_toml(config_file).get("mcp")
    return MCPConfig.from_dict(table if isinstance(table, dict) else None)
