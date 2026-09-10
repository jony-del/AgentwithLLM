"""Environment expansion helpers for plugin configuration and secrets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from agent_core.config import user_settings_path
from agent_core.env_security import (
    expand_host_env,
    is_sensitive_env_name,
    referenced_host_variables,
)
from .models import PluginError
from .store import _KEYCHAIN_PREFIX

def _plugin_secret_targets(workspace: Path) -> tuple[str, ...]:
    targets: set[str] = set()
    try:
        import tomllib

        path = user_settings_path()
        loaded = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        configs = loaded.get("plugin_configs", {})
        if isinstance(configs, dict):
            for values in configs.values():
                if not isinstance(values, dict):
                    continue
                for value in values.values():
                    if isinstance(value, str) and value.startswith(_KEYCHAIN_PREFIX):
                        targets.add(
                            "Polaris/plugin-config/" + value.removeprefix(_KEYCHAIN_PREFIX)
                        )
    except (OSError, RuntimeError, UnicodeDecodeError, ValueError):
        pass
    return tuple(sorted(targets))


def _plugin_option_env(values: dict[str, Any]) -> dict[str, str]:
    return {
        f"CLAUDE_PLUGIN_OPTION_{key}": (
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        )
        for key, value in values.items()
    }


def _expand_executable_env(
    value: str,
    *,
    plugin_id: str | None = None,
    env_access_granted: bool = False,
    audit: Any = None,
) -> str:
    """Resolve host variables only for an explicit executable environment value.

    Plugin-sourced values (``plugin_id`` set) may reference sensitive host variables
    (API keys, tokens, …) only when the project granted that plugin ``env-access``;
    otherwise loading fails closed with the variable name (never its value).  Granted
    expansions are audited by name.
    """

    if plugin_id is None:
        return expand_host_env(value)
    names = referenced_host_variables(value)
    sensitive = [name for name in names if is_sensitive_env_name(name)]
    if sensitive and not env_access_granted:
        raise PluginError(
            f"plugin {plugin_id} references sensitive host variable "
            f"{sensitive[0]} without an env-access grant"
        )
    if sensitive and audit is not None:
        audit.write(
            "env_expansion",
            {"plugin_id": plugin_id, "variables": sorted(sensitive)},
        )
    return expand_host_env(value)


def _expand_plugin_vars(
    value: Any,
    root: Path,
    workspace: Path,
    plugin_data: Path,
    user_config: dict[str, Any],
    *,
    allow_user_config: bool = True,
) -> Any:
    if isinstance(value, str):
        result = (
            value.replace("${CLAUDE_PLUGIN_ROOT}", str(root))
            .replace("${CLAUDE_PLUGIN_DATA}", str(plugin_data))
            .replace("${CLAUDE_PROJECT_DIR}", str(workspace))
        )
        if not allow_user_config and "${user_config." in result:
            raise PluginError("monitor commands cannot reference user_config values")
        if allow_user_config:
            result = re.sub(
                r"\$\{user_config\.([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda match: str(user_config.get(match.group(1), "")),
                result,
            )
        # Arbitrary host environment variables are deliberately left literal here.
        # Prompt-bearing plugin content only receives framework paths and public
        # ``user_config``. Executable components resolve environment references solely
        # in their explicit ``env``/``headers`` maps at the process/transport boundary.
        return result
    if isinstance(value, list):
        return [
            _expand_plugin_vars(
                item, root, workspace, plugin_data, user_config,
                allow_user_config=allow_user_config,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _expand_plugin_vars(
                item, root, workspace, plugin_data, user_config,
                allow_user_config=allow_user_config,
            )
            for key, item in value.items()
        }
    return value
