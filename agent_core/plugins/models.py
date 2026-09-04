"""Plugin state records, bundle/generation models, and name/pin validators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from agent_core.capability_security import TrustTier
from agent_core.mcp import MCPClientManager
from agent_core.skills import Skill, SkillRegistry

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


_REMOTE_SOURCE = re.compile(r"^(?:https?|ssh|git)://|^git@")


_SHA256_PIN = re.compile(r"^[0-9a-fA-F]{64}$")


_IMMUTABLE_GIT_COMMIT = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class PluginError(RuntimeError):
    pass


def is_safe_plugin_name(value: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(value))


def is_sha256_pin(value: str) -> bool:
    return bool(_SHA256_PIN.fullmatch(value))


def is_git_commit_pin(value: str) -> bool:
    """Return true only for a full immutable Git object id, never a branch/tag/ref."""

    return bool(_IMMUTABLE_GIT_COMMIT.fullmatch(value))


def _string_tuple(value: object) -> tuple[str, ...]:
    values = value if isinstance(value, list) else str(value or "").split(",")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _positive_int(value: object) -> int | None:
    text = str(value or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


@dataclass(slots=True)
class PluginRecord:
    plugin_id: str
    name: str
    marketplace: str
    path: str
    source: str
    version: str = ""
    installed_at: float = 0.0
    description: str = ""
    keywords: tuple[str, ...] = ()
    integrity: str = ""
    artifact_digest: str = ""
    commit: str = ""
    source_kind: str = ""
    source_ref: str = ""
    resolved_version: str = ""
    marketplace_snapshot: str = ""
    components: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    manifest: dict[str, Any] | None = None
    auto_installed: bool = False
    required_by: tuple[str, ...] = ()
    source_identity: str = ""
    trust_tier: str = TrustTier.COMMUNITY.value


@dataclass(slots=True)
class MarketplaceRecord:
    name: str
    source: dict[str, Any]
    snapshot_path: str
    snapshot_commit: str = ""
    catalog_digest: str = ""
    refreshed_at: float = 0.0
    source_id: str = ""
    canonical_source: str = ""
    trust_tier: str = TrustTier.COMMUNITY.value
    last_attempt: float = 0.0
    last_success: float = 0.0
    failure_count: int = 0
    next_retry: float = 0.0
    last_error: str = ""


@dataclass(frozen=True, slots=True)
class PluginStateStatus:
    schema_version: int
    legacy: bool
    reset_required: bool
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedPluginArtifact:
    name: str
    marketplace: str
    path: str
    digest: str
    commit: str
    marketplace_snapshot: str
    source_identity: str
    trust_tier: str


@dataclass(slots=True)
class PluginBundle:
    skills: list[Skill]
    hooks: list[tuple[str, Any]]
    mcp_manager: MCPClientManager | None
    mcp_tools: list[Any]
    agents: dict[str, str]
    components: dict[str, Any]
    plugin_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class PluginGeneration:
    bundle: PluginBundle
    skills: SkillRegistry
    hooks: Any
