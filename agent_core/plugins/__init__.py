"""Claude-compatible plugin installation, validation, and generation swaps.

No marketplace is preloaded. Installation only copies/records files; executable
components (hooks and MCP servers) are activated solely by an explicit enable followed
by ``/reload-plugins``, or by the policy-gated capability manager at a turn boundary.
"""

from agent_core import secret_store as secret_store
from agent_core.mcp import MCPClientManager as MCPClientManager
from agent_core.plugin_spec import MarketplaceSourceConfig as MarketplaceSourceConfig

from .models import (
    MarketplaceRecord as MarketplaceRecord,
    PluginBundle as PluginBundle,
    PluginError as PluginError,
    PluginGeneration as PluginGeneration,
    PluginRecord as PluginRecord,
    PluginStateStatus as PluginStateStatus,
    PreparedPluginArtifact as PreparedPluginArtifact,
    is_git_commit_pin as is_git_commit_pin,
    is_safe_plugin_name as is_safe_plugin_name,
    is_sha256_pin as is_sha256_pin,
)
from .store import (
    copy_marketplace_plugin_tree as copy_marketplace_plugin_tree,
    plugin_home as plugin_home,
    plugin_tree_digest as plugin_tree_digest,
    validate_plugin as validate_plugin,
)
from .sources import (
    _extract_zip_bytes as _extract_zip_bytes,
    _git_output as _git_output,
)
from .env import (
    _expand_executable_env as _expand_executable_env,
    _expand_plugin_vars as _expand_plugin_vars,
)
from .sandbox import (
    sandbox_runtime_environment as sandbox_runtime_environment,
    sandboxed_guest_invocation as sandboxed_guest_invocation,
)
from .components import _apply_skill_provenance as _apply_skill_provenance
from .manager import PluginManager as PluginManager
from .runtime import (
    _commit_generation as _commit_generation,
    _prepare_generation as _prepare_generation,
    activate_plugin as activate_plugin,
    reload_plugins as reload_plugins,
)

__all__ = [
    "MCPClientManager",
    "MarketplaceRecord",
    "MarketplaceSourceConfig",
    "PluginBundle",
    "PluginError",
    "PluginGeneration",
    "PluginManager",
    "PluginRecord",
    "PluginStateStatus",
    "PreparedPluginArtifact",
    "_apply_skill_provenance",
    "_commit_generation",
    "_expand_executable_env",
    "_expand_plugin_vars",
    "_extract_zip_bytes",
    "_git_output",
    "_prepare_generation",
    "activate_plugin",
    "copy_marketplace_plugin_tree",
    "is_git_commit_pin",
    "is_safe_plugin_name",
    "is_sha256_pin",
    "plugin_home",
    "plugin_tree_digest",
    "reload_plugins",
    "sandbox_runtime_environment",
    "sandboxed_guest_invocation",
    "secret_store",
    "validate_plugin",
]
