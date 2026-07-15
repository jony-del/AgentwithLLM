"""Read-only managed permission policy extension point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_core.permission_rules import RuleSet
from agent_core.permission_types import ManagedPolicySnapshot, PermissionMode, PermissionRuleSource


@dataclass(frozen=True, slots=True)
class ManagedPolicyDefinition:
    """A deployment policy that can tighten, but never widen, permissions."""

    deny: tuple[str, ...] = ()
    ask: tuple[str, ...] = ()
    forbidden_modes: frozenset[PermissionMode] = frozenset()
    require_sandbox_for_unattended: bool = False

    def rules(self) -> RuleSet:
        return RuleSet.from_lists(
            deny=list(self.deny),
            ask=list(self.ask),
            source=PermissionRuleSource.MANAGED,
        )

    def snapshot(self) -> ManagedPolicySnapshot:
        return ManagedPolicySnapshot(
            forbidden_modes=self.forbidden_modes,
            require_sandbox_for_unattended=self.require_sandbox_for_unattended,
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
