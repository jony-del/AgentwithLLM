"""Shared host-environment expansion policy for plugin/MCP executable boundaries.

Prompt-bearing plugin content never expands host variables at all (see
``agent_core.plugins.env._expand_plugin_vars``).  Explicit ``env``/``headers`` maps are
the only places ``${VAR}`` references resolve, and both plugin loaders and the MCP
transport must apply the same sensitive-name judgement so a grant meaning cannot
drift between load time and connect time.
"""

from __future__ import annotations

import os
import re

_HOST_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# A variable name is sensitive when any ``_``-separated part matches; this catches
# OPENAI_API_KEY / AWS_SECRET_ACCESS_KEY / GITHUB_TOKEN while leaving PATH, PATHEXT,
# MONKEY-style names alone.
SENSITIVE_ENV_PARTS = frozenset(
    {
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "CREDENTIALS",
        "PRIVATE",
        "AUTH",
        "AUTHORIZATION",
        "SESSION",
        "COOKIE",
    }
)


def is_sensitive_env_name(name: str) -> bool:
    parts = re.split(r"[^A-Za-z0-9]+", name.upper())
    return any(part in SENSITIVE_ENV_PARTS for part in parts)


def referenced_host_variables(value: str) -> tuple[str, ...]:
    """Names of host variables a value would expand, in first-use order."""

    return tuple(dict.fromkeys(match.group(1) for match in _HOST_ENV_RE.finditer(value)))


def expand_host_env(
    value: str,
    *,
    allow_sensitive: bool = True,
    blocked_exc: type[Exception] = ValueError,
) -> str:
    """Resolve ``${VAR}``/``${VAR:-default}`` host references.

    With ``allow_sensitive=False`` a sensitive reference raises ``blocked_exc``
    naming the variable (never the value); non-sensitive references still resolve.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if not allow_sensitive and is_sensitive_env_name(name):
            raise blocked_exc(
                f"reference to sensitive host environment variable {name} requires "
                "an explicit grant"
            )
        return os.environ.get(name, default or "")

    return _HOST_ENV_RE.sub(replace, value)
