"""Read-only deployment permission policy loaded from an administrator-owned file."""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent_core.permission_rules import RuleSet, parse_rule
from agent_core.permission_types import (
    ManagedPolicySnapshot,
    PermissionMode,
    PermissionRuleSource,
    parse_permission_mode,
)


class ManagedPolicyError(ValueError):
    """The configured managed policy exists but is invalid or unreadable."""


@dataclass(frozen=True, slots=True)
class ManagedPolicyDefinition:
    """A deployment policy. Central safety checks remain stronger than managed allow."""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    ask: tuple[str, ...] = ()
    forbidden_modes: frozenset[PermissionMode] = frozenset()
    require_sandbox_for_unattended: bool = False
    allow_managed_rules_only: bool = False
    disable_persistent_grants: bool = False
    digest: str = ""

    def rules(self) -> RuleSet:
        return RuleSet.from_lists(
            allow=list(self.allow),
            deny=list(self.deny),
            ask=list(self.ask),
            source=PermissionRuleSource.MANAGED,
        )

    def snapshot(self) -> ManagedPolicySnapshot:
        return ManagedPolicySnapshot(
            forbidden_modes=self.forbidden_modes,
            require_sandbox_for_unattended=self.require_sandbox_for_unattended,
            allow_managed_rules_only=self.allow_managed_rules_only,
            disable_persistent_grants=self.disable_persistent_grants,
            policy_digest=self.digest,
        )


class ManagedPolicyProvider(Protocol):
    def load(self) -> ManagedPolicyDefinition: ...


class NullManagedPolicyProvider:
    def load(self) -> ManagedPolicyDefinition:
        return ManagedPolicyDefinition()


class StaticManagedPolicyProvider:
    def __init__(self, definition: ManagedPolicyDefinition) -> None:
        self.definition = definition

    def load(self) -> ManagedPolicyDefinition:
        return self.definition


def default_managed_policy_path() -> Path:
    override = os.getenv("POLARIS_MANAGED_POLICY_PATH")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Windows":
        root = os.getenv("ProgramData", r"C:\ProgramData")
        return Path(root) / "Polaris" / "managed-policy.toml"
    if system == "Darwin":
        return Path("/Library/Application Support/Polaris/managed-policy.toml")
    return Path("/etc/polaris/managed-policy.toml")


class FileManagedPolicyProvider:
    """Loads ``[managed.permissions]`` and re-reads it on every call.

    A missing implicit system file means no enterprise restrictions. An explicit path
    (constructor or environment) must exist. Any existing but malformed file raises,
    allowing startup to fail and live reload to fail closed.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_managed_policy_path()
        self.explicit = path is not None or bool(os.getenv("POLARIS_MANAGED_POLICY_PATH"))

    def load(self) -> ManagedPolicyDefinition:
        if not self.path.exists():
            if self.explicit:
                raise ManagedPolicyError(f"managed policy file does not exist: {self.path}")
            return ManagedPolicyDefinition()
        try:
            data = self.path.read_bytes()
            import tomllib

            raw = tomllib.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ManagedPolicyError(f"invalid managed policy {self.path}: {exc}") from exc
        table: Any = raw.get("managed", {})
        table = table.get("permissions", {}) if isinstance(table, dict) else None
        if not isinstance(table, dict):
            raise ManagedPolicyError("[managed.permissions] must be a TOML table")

        def string_list(key: str) -> tuple[str, ...]:
            value = table.get(key, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ManagedPolicyError(f"managed.permissions.{key} must be an array of strings")
            items = tuple(item.strip() for item in value if item.strip())
            if key in {"allow", "deny", "ask"}:
                invalid = next((item for item in items if parse_rule(item) is None), None)
                if invalid is not None:
                    raise ManagedPolicyError(
                        f"managed.permissions.{key} contains invalid rule: {invalid!r}"
                    )
            return items

        modes: set[PermissionMode] = set()
        for raw_mode in string_list("forbidden_modes"):
            try:
                modes.add(parse_permission_mode(raw_mode))
            except ValueError as exc:
                raise ManagedPolicyError(f"unknown forbidden permission mode: {raw_mode}") from exc

        def boolean(key: str) -> bool:
            value = table.get(key, False)
            if not isinstance(value, bool):
                raise ManagedPolicyError(f"managed.permissions.{key} must be boolean")
            return value

        return ManagedPolicyDefinition(
            allow=string_list("allow"),
            deny=string_list("deny"),
            ask=string_list("ask"),
            forbidden_modes=frozenset(modes),
            require_sandbox_for_unattended=boolean("require_sandbox_for_unattended"),
            allow_managed_rules_only=boolean("allow_managed_rules_only"),
            disable_persistent_grants=boolean("disable_persistent_grants"),
            digest=hashlib.sha256(data).hexdigest(),
        )
