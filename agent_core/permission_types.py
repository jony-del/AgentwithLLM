"""Shared permission contracts used by the policy engine and individual tools.

This module intentionally contains data shapes only.  Keeping the contracts free of
runtime imports from the tool/executor layers avoids a permission-specific import
cycle as tools begin implementing asynchronous, argument-aware checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPTEDITS = "acceptedits"
    PLAN = "plan"
    AUTO = "auto"
    DONTASK = "dontask"
    BYPASS = "bypass"

    @classmethod
    def _missing_(cls, value: object) -> "PermissionMode | None":
        if isinstance(value, str):
            canonical = PERMISSION_MODE_ALIASES.get(value.strip().casefold())
            if canonical is not None:
                return cls(canonical)
        return None


PERMISSION_MODE_ALIASES: dict[str, str] = {
    "default": "default",
    "acceptedits": "acceptedits",
    "plan": "plan",
    "auto": "auto",
    "dontask": "dontask",
    "bypass": "bypass",
    # Reference-project / Claude Code spellings.
    "bypasspermissions": "bypass",
}


def parse_permission_mode(value: PermissionMode | str) -> PermissionMode:
    """Parse canonical modes and compatibility aliases, returning canonical values."""
    return value if isinstance(value, PermissionMode) else PermissionMode(value)


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
    PASSTHROUGH = "passthrough"


class DecisionSource(str, Enum):
    SCHEMA = "schema"
    MANAGED = "managed"
    RULE = "rule"
    CENTRAL_SAFETY = "central_safety"
    TOOL = "tool"
    SANDBOX = "sandbox"
    MODE = "mode"
    RISK_FALLBACK = "risk_fallback"
    CLASSIFIER = "classifier"
    HOOK = "hook"
    USER = "user"


class PermissionRuleSource(str, Enum):
    MANAGED = "managed"
    USER = "user"
    PROJECT = "project"
    LOCAL = "local"
    CLI = "cli"
    SESSION = "session"


class PermissionDestination(str, Enum):
    SESSION = "session"
    LOCAL = "local"
    PROJECT = "project"
    USER = "user"


@dataclass(frozen=True, slots=True)
class PermissionSuggestion:
    rule: str
    reason: str
    destination: PermissionDestination = PermissionDestination.SESSION


@dataclass(frozen=True, slots=True)
class PermissionUpdate:
    behavior: PermissionBehavior
    rule: str
    destination: PermissionDestination


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool_name: str
    risk: str
    arguments: Mapping[str, Any]
    reason: str
    suggestions: tuple[PermissionSuggestion, ...] = ()
    persistent_grants_disabled: bool = False
    session_grants_disabled: bool = False


@dataclass(frozen=True, slots=True)
class PermissionResponse:
    allow: bool
    updates: tuple[PermissionUpdate, ...] = ()
    reason: str = ""
    updated_arguments: Mapping[str, Any] | None = None


class ToolCallSource(str, Enum):
    MODEL = "model"
    API = "api"
    SLASH = "slash"
    HOOK = "hook"
    SUBAGENT = "subagent"
    TEAM = "team"
    SKILL = "skill"


@dataclass(frozen=True, slots=True)
class PermissionRule:
    source: PermissionRuleSource
    behavior: PermissionBehavior
    tool_name: str
    content: str | None = None
    raw: str = ""


@dataclass(frozen=True, slots=True)
class SandboxState:
    enabled: bool = False
    backend: str = "noop"
    will_sandbox: bool = False
    excluded: bool = False
    auto_allow_enabled: bool = False


@dataclass(frozen=True, slots=True)
class SessionAuthorizationView:
    tool_names: frozenset[str] = frozenset()
    command_fingerprints: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class WebDomainPolicySnapshot:
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    unattended: bool = False


@dataclass(frozen=True, slots=True)
class ManagedPolicySnapshot:
    forbidden_modes: frozenset[PermissionMode] = frozenset()
    require_sandbox_for_unattended: bool = False
    allow_managed_rules_only: bool = False
    disable_persistent_grants: bool = False
    policy_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanStateSnapshot:
    active: bool = False
    previous_mode: PermissionMode | None = None
    artifact_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PermissionContext:
    mode: PermissionMode
    workspace: Path
    interactive: bool
    sandbox: SandboxState = SandboxState()
    rules: Any = None
    session_authorizations: SessionAuthorizationView = SessionAuthorizationView()
    is_subagent: bool = False
    parent_mode: PermissionMode | None = None
    parent_agent_id: str | None = None
    tool_source: ToolCallSource = ToolCallSource.MODEL
    web_policy: WebDomainPolicySnapshot = WebDomainPolicySnapshot()
    managed_policy: ManagedPolicySnapshot = ManagedPolicySnapshot()
    plan_state: PlanStateSnapshot = PlanStateSnapshot()


@dataclass(frozen=True, slots=True)
class PermissionResult:
    behavior: PermissionBehavior
    reason: str
    decision_source: DecisionSource
    updated_arguments: dict[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    matched_rule: PermissionRule | None = None
    classifier_approvable: bool = False
    bypass_immune: bool = False
    suggestions: tuple[PermissionSuggestion, ...] = ()

    @classmethod
    def allow(
        cls,
        reason: str = "allowed by tool",
        *,
        decision_source: DecisionSource = DecisionSource.TOOL,
        updated_arguments: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        matched_rule: PermissionRule | None = None,
    ) -> "PermissionResult":
        return cls(
            PermissionBehavior.ALLOW,
            reason,
            decision_source,
            updated_arguments,
            metadata,
            matched_rule,
        )

    @classmethod
    def ask(
        cls,
        reason: str,
        *,
        decision_source: DecisionSource = DecisionSource.TOOL,
        updated_arguments: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        matched_rule: PermissionRule | None = None,
        classifier_approvable: bool = False,
        bypass_immune: bool = False,
        suggestions: tuple[PermissionSuggestion, ...] = (),
    ) -> "PermissionResult":
        return cls(
            PermissionBehavior.ASK,
            reason,
            decision_source,
            updated_arguments,
            metadata,
            matched_rule,
            classifier_approvable,
            bypass_immune,
            suggestions,
        )

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        decision_source: DecisionSource = DecisionSource.TOOL,
        updated_arguments: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        matched_rule: PermissionRule | None = None,
        bypass_immune: bool = True,
        suggestions: tuple[PermissionSuggestion, ...] = (),
    ) -> "PermissionResult":
        return cls(
            PermissionBehavior.DENY,
            reason,
            decision_source,
            updated_arguments,
            metadata,
            matched_rule,
            False,
            bypass_immune,
            suggestions,
        )

    @classmethod
    def passthrough(
        cls,
        reason: str = "tool has no specific permission decision",
        *,
        updated_arguments: dict[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PermissionResult":
        return cls(
            PermissionBehavior.PASSTHROUGH,
            reason,
            DecisionSource.TOOL,
            updated_arguments,
            metadata,
        )
