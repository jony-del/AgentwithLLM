from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from agent_core.managed_policy import FileManagedPolicyProvider, ManagedPolicyProvider
from agent_core.models import ToolRisk
from agent_core.permission_rules import (
    _SHELL_COMMAND_TOOLS,
    RuleSet,
    _normalize_subcommand,
    _split_subcommands,
)
from agent_core.permission_safety import (
    check_web_endpoint,
    inspect_paths,
)
from agent_core.permission_types import (
    DecisionSource,
    ManagedPolicySnapshot,
    PermissionBehavior,
    PermissionContext,
    PermissionDestination,
    PermissionMode,
    PermissionRequest,
    PermissionResponse,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
    PermissionSuggestion,
    PermissionUpdate,
    PlanStateSnapshot,
    SandboxState,
    SessionAuthorizationView,
    ToolCallSource,
    WebDomainPolicySnapshot,
)
from agent_core.permission_store import PermissionPersistenceError, persist_allow_rules

if TYPE_CHECKING:
    from agent_core.models import ToolCall
    from agent_core.sandbox import SandboxManager
    from agent_core.tools.base import Tool

Prompter = Callable[..., str | PermissionResponse]

PERMISSION_MODE_LABELS: dict[PermissionMode, str] = {
    PermissionMode.DEFAULT: "manual mode on",
    PermissionMode.ACCEPTEDITS: "accept edits on",
    PermissionMode.PLAN: "plan mode on",
    PermissionMode.AUTO: "auto mode on",
    PermissionMode.DONTASK: "don't ask mode on",
    PermissionMode.BYPASS: "bypass permissions on",
}

SHIFT_TAB_PERMISSION_MODES: tuple[PermissionMode, ...] = (
    PermissionMode.DEFAULT,
    PermissionMode.ACCEPTEDITS,
    PermissionMode.PLAN,
    PermissionMode.AUTO,
)


def permission_mode_label(mode: PermissionMode | str) -> str:
    return PERMISSION_MODE_LABELS[PermissionMode(mode)]


def next_shift_tab_permission_mode(mode: PermissionMode | str) -> PermissionMode:
    current = PermissionMode(mode)
    try:
        index = SHIFT_TAB_PERMISSION_MODES.index(current)
    except ValueError:
        return PermissionMode.DEFAULT
    return SHIFT_TAB_PERMISSION_MODES[(index + 1) % len(SHIFT_TAB_PERMISSION_MODES)]


@dataclass(slots=True)
class PermissionDecision:
    """Deprecated boolean adapter retained for UI and embedded-client compatibility."""

    allowed: bool
    ask_user: bool = False
    reason: str = ""
    ask_collapsed: bool = False
    classify: bool = False


class PermissionPolicy:
    """Single ordered permission engine combining central and per-tool decisions."""

    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.DEFAULT,
        prompter: Prompter | None = None,
        rules: RuleSet | None = None,
        sandbox: "SandboxManager | None" = None,
        *,
        workspace: str | Path | None = None,
        is_subagent: bool = False,
        parent_mode: PermissionMode | str | None = None,
        parent_agent_id: str | None = None,
        tool_source: ToolCallSource = ToolCallSource.MODEL,
        managed_policy: ManagedPolicySnapshot | None = None,
        managed_policy_provider: ManagedPolicyProvider | None = None,
        plan_state: Any | None = None,
        allow_unsandboxed_unattended: bool | None = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        self.prompter = prompter
        self.interactive = prompter is not None
        self.managed_policy_provider = managed_policy_provider or FileManagedPolicyProvider()
        managed_definition = self.managed_policy_provider.load()
        self._base_rules = rules or RuleSet()
        self._session_rules = RuleSet()
        self.rules = managed_definition.rules().merge(self._base_rules)
        self.sandbox = sandbox
        self.workspace = Path(workspace).resolve() if workspace is not None else None
        self.is_subagent = is_subagent
        self.parent_mode = PermissionMode(parent_mode) if parent_mode is not None else None
        self.parent_agent_id = parent_agent_id
        self.tool_source = tool_source
        self._managed_snapshot_override = managed_policy
        self.managed_policy = managed_policy or managed_definition.snapshot()
        self._managed_reload_error: str | None = None
        self.plan_state = plan_state
        if allow_unsandboxed_unattended is None:
            opt_out = os.getenv("AGENT_SANDBOX_ALLOW_UNATTENDED", "")
            allow_unsandboxed_unattended = opt_out.strip().casefold() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.allow_unsandboxed_unattended = allow_unsandboxed_unattended
        self._session_allow: set[str] = set()
        self._session_allow_commands: set[str] = set()
        self._validators: dict[tuple[str, int], Draft202012Validator] = {}
        self.last_permission_updates: tuple[PermissionUpdate, ...] = ()

    async def evaluate(
        self,
        tool: "Tool",
        tool_call: "ToolCall | None" = None,
        *,
        context: PermissionContext | None = None,
    ) -> PermissionResult:
        """Evaluate one call using the documented deterministic permission order."""
        arguments = dict(tool_call.arguments) if tool_call is not None else {}
        reload_error = self.refresh_managed_policy()
        if reload_error is not None:
            return PermissionResult.deny(
                f"managed policy reload failed: {reload_error}",
                decision_source=DecisionSource.MANAGED,
                metadata={"managed_policy_reload_error": reload_error[:200]},
            )
        ctx = (
            replace(context, mode=self.mode, rules=self.rules, managed_policy=self.managed_policy)
            if context is not None
            else self.build_context(tool, arguments)
        )

        preflight = await self._preflight(tool, arguments, ctx)
        if preflight is not None:
            return self._apply_final_mode(preflight, tool)

        try:
            tool_result = await tool.check_permissions(arguments, ctx)
        except Exception as exc:  # tool policy bugs may never fail open
            return PermissionResult.deny(
                f"tool permission check failed: {type(exc).__name__}",
                decision_source=DecisionSource.TOOL,
                metadata={"error": str(exc)[:200]},
            )

        if tool_result.updated_arguments is not None:
            updated = dict(tool_result.updated_arguments)
            updated_context = self.build_context(tool, updated)
            updated_preflight = await self._preflight(tool, updated, updated_context)
            if updated_preflight is not None:
                return self._apply_final_mode(updated_preflight, tool)
            arguments = updated
            ctx = updated_context

        if tool_result.behavior is PermissionBehavior.DENY:
            return tool_result
        if tool_result.behavior is PermissionBehavior.ASK:
            return self._apply_final_mode(tool_result, tool)

        if tool.requires_user_interaction:
            return self._apply_final_mode(
                PermissionResult.ask(
                    "tool requires an explicit interactive approval channel",
                    decision_source=DecisionSource.CENTRAL_SAFETY,
                    updated_arguments=tool_result.updated_arguments,
                    bypass_immune=True,
                ),
                tool,
            )

        if tool_result.behavior is PermissionBehavior.PASSTHROUGH:
            sandbox_allow = self._sandbox_auto_allows(tool.name, arguments)
            if sandbox_allow:
                return PermissionResult.allow(
                    "allowed because this exact command will run sandboxed",
                    decision_source=DecisionSource.SANDBOX,
                    updated_arguments=tool_result.updated_arguments,
                )
            if self.mode is PermissionMode.BYPASS:
                return PermissionResult.allow(
                    "bypass mode allows the remaining passthrough action",
                    decision_source=DecisionSource.MODE,
                    updated_arguments=tool_result.updated_arguments,
                )
            if self._session_allowed(tool.name, arguments):
                return PermissionResult.allow(
                    "allowed for this session",
                    decision_source=DecisionSource.RULE,
                    updated_arguments=tool_result.updated_arguments,
                    matched_rule=PermissionRule(
                        PermissionRuleSource.SESSION,
                        PermissionBehavior.ALLOW,
                        tool.name,
                        raw=tool.name,
                    ),
                )
            allow_rule = self._allow_match(tool.name, arguments)
            if allow_rule is not None:
                return PermissionResult.allow(
                    "allowed by rule",
                    decision_source=DecisionSource.RULE,
                    updated_arguments=tool_result.updated_arguments,
                    matched_rule=allow_rule,
                )
            return self._apply_final_mode(
                self._risk_fallback(tool, tool_result.updated_arguments), tool
            )

        # Session/explicit allow provenance is retained even when the tool independently
        # classified the operation as safe.  These checks occur after every deny/ask.
        if self._session_allowed(tool.name, arguments):
            return PermissionResult.allow(
                "allowed for this session",
                decision_source=DecisionSource.RULE,
                updated_arguments=tool_result.updated_arguments,
                matched_rule=PermissionRule(
                    PermissionRuleSource.SESSION,
                    PermissionBehavior.ALLOW,
                    tool.name,
                    raw=tool.name,
                ),
            )
        allow_rule = self._allow_match(tool.name, arguments)
        if allow_rule is not None:
            return PermissionResult.allow(
                "allowed by rule",
                decision_source=DecisionSource.RULE,
                updated_arguments=tool_result.updated_arguments,
                matched_rule=allow_rule,
            )

        # Tool ALLOW is honored only after every central/rule check above succeeded.
        return tool_result

    async def _preflight(
        self,
        tool: "Tool",
        arguments: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionResult | None:
        schema_result = self._validate_schema(tool, arguments)
        if schema_result is not None:
            return schema_result

        if context.mode in context.managed_policy.forbidden_modes:
            return PermissionResult.deny(
                f"permission mode {context.mode.value!r} is forbidden by managed policy",
                decision_source=DecisionSource.MANAGED,
            )
        managed_deny = self.rules.deny_match(
            tool.name, arguments, source=PermissionRuleSource.MANAGED
        )
        if managed_deny is not None:
            return PermissionResult.deny(
                "denied by managed policy",
                decision_source=DecisionSource.MANAGED,
                matched_rule=managed_deny,
            )
        deny_rule = self.rules.deny_match(tool.name, arguments)
        if deny_rule is not None:
            return PermissionResult.deny(
                "denied by rule",
                decision_source=DecisionSource.RULE,
                matched_rule=deny_rule,
            )

        unattended_modes = {
            PermissionMode.AUTO,
            PermissionMode.DONTASK,
            PermissionMode.BYPASS,
        }
        if context.mode in unattended_modes and not context.sandbox.enabled:
            if context.managed_policy.require_sandbox_for_unattended:
                return PermissionResult.deny(
                    "managed policy requires a Sandbox for unattended permission modes",
                    decision_source=DecisionSource.MANAGED,
                )
            if not self.allow_unsandboxed_unattended:
                return PermissionResult.deny(
                    "unattended permission modes require a Sandbox or an explicit opt-out",
                    decision_source=DecisionSource.CENTRAL_SAFETY,
                )

        if context.is_subagent and context.parent_mode is not None:
            allowed_child_modes = {
                PermissionMode.DEFAULT: {PermissionMode.DEFAULT},
                PermissionMode.ACCEPTEDITS: {PermissionMode.DEFAULT, PermissionMode.ACCEPTEDITS},
                PermissionMode.PLAN: {PermissionMode.PLAN},
                PermissionMode.AUTO: {
                    PermissionMode.DEFAULT,
                    PermissionMode.ACCEPTEDITS,
                    PermissionMode.AUTO,
                },
                PermissionMode.DONTASK: {PermissionMode.DEFAULT, PermissionMode.DONTASK},
                PermissionMode.BYPASS: {
                    PermissionMode.DEFAULT,
                    PermissionMode.ACCEPTEDITS,
                    PermissionMode.DONTASK,
                    PermissionMode.BYPASS,
                },
            }[context.parent_mode]
            if context.mode not in allowed_child_modes:
                return PermissionResult.deny(
                    "sub-agent permission mode exceeds its parent capability envelope",
                    decision_source=DecisionSource.CENTRAL_SAFETY,
                )

        # Plan is centrally constrained so a faulty/custom tool cannot self-declare a
        # normal mutation safe.  The dedicated plan-artifact capability is added later.
        plan_capabilities = {
            "write_plan",
            "exit_plan",
            "update_todos",
            "task_create",
            "task_update",
            "team_create",
            "teammate_spawn",
            "team_message_send",
            "dispatch_agent",
            "skill",
        }
        if (
            context.mode is PermissionMode.PLAN
            and tool.risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS}
            and tool.name not in plan_capabilities
        ):
            return PermissionResult.deny(
                "plan mode is read-only except for registered planning capabilities",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )

        path_result = inspect_paths(tool.name, arguments, context)
        if path_result is not None:
            return path_result

        if tool.name == "web_fetch":
            web_result = check_web_endpoint(str(arguments.get("url", "")), context)
            if web_result is not None:
                return web_result

        ask_rule = self.rules.ask_match(tool.name, arguments)
        if ask_rule is not None:
            return PermissionResult.ask(
                (
                    "confirmation required by managed policy"
                    if ask_rule.source is PermissionRuleSource.MANAGED
                    else "confirmation required by explicit ask rule"
                ),
                decision_source=(
                    DecisionSource.MANAGED
                    if ask_rule.source is PermissionRuleSource.MANAGED
                    else DecisionSource.RULE
                ),
                matched_rule=ask_rule,
                bypass_immune=True,
            )
        return None

    def build_context(self, tool: "Tool", arguments: dict[str, Any]) -> PermissionContext:
        workspace = self.workspace
        if workspace is None:
            workspace = Path(getattr(tool, "workspace", Path.cwd())).resolve()
        web_policy = getattr(tool, "web_policy", None)
        allowed_domains = tuple(getattr(web_policy, "allowed_domains", ()) or ())
        blocked_domains = tuple(getattr(web_policy, "blocked_domains", ()) or ())
        sandbox = self._sandbox_state(tool.name, arguments)
        return PermissionContext(
            mode=self.mode,
            workspace=workspace,
            interactive=self.interactive,
            sandbox=sandbox,
            rules=self.rules,
            session_authorizations=SessionAuthorizationView(
                frozenset(self._session_allow),
                frozenset(self._session_allow_commands),
            ),
            is_subagent=self.is_subagent,
            parent_mode=self.parent_mode,
            parent_agent_id=self.parent_agent_id,
            tool_source=self.tool_source,
            web_policy=WebDomainPolicySnapshot(
                allowed_domains,
                blocked_domains,
                bool(getattr(tool, "unattended", False)),
            ),
            managed_policy=self.managed_policy,
            plan_state=PlanStateSnapshot(
                active=bool(getattr(self.plan_state, "active", False)),
                previous_mode=(
                    PermissionMode(getattr(self.plan_state, "previous_mode"))
                    if getattr(self.plan_state, "previous_mode", None)
                    else None
                ),
                artifact_path=getattr(self.plan_state, "artifact_path", None),
            ),
        )

    def _validate_schema(self, tool: "Tool", arguments: dict[str, Any]) -> PermissionResult | None:
        schema = tool.input_schema
        key = (tool.name, id(schema))
        try:
            validator = self._validators.get(key)
            if validator is None:
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(schema)
                self._validators[key] = validator
            validator.validate(arguments)
        except (SchemaError, ValidationError) as exc:
            path = ".".join(str(part) for part in getattr(exc, "absolute_path", ()))
            location = f" at {path}" if path else ""
            return PermissionResult.deny(
                f"tool input schema validation failed{location}: {exc.message}",
                decision_source=DecisionSource.SCHEMA,
                metadata={"validation_path": path},
            )
        return None

    def _risk_fallback(
        self,
        tool: "Tool",
        updated_arguments: dict[str, Any] | None,
    ) -> PermissionResult:
        if tool.risk is ToolRisk.READ:
            return PermissionResult.allow(
                "legacy ToolRisk fallback allows read-only tools",
                decision_source=DecisionSource.RISK_FALLBACK,
                updated_arguments=updated_arguments,
            )
        if self.mode in {PermissionMode.PLAN, PermissionMode.DONTASK}:
            return PermissionResult.deny(
                f"{self.mode.value} mode denies an unresolved {tool.risk.value} fallback",
                decision_source=DecisionSource.RISK_FALLBACK,
                updated_arguments=updated_arguments,
            )
        return PermissionResult.ask(
            "unmigrated side-effecting tool requires confirmation",
            decision_source=DecisionSource.RISK_FALLBACK,
            updated_arguments=updated_arguments,
            classifier_approvable=self.mode is PermissionMode.AUTO,
        )

    def _apply_final_mode(self, result: PermissionResult, tool: "Tool") -> PermissionResult:
        if result.behavior is not PermissionBehavior.ASK:
            return result
        if self.mode is PermissionMode.DONTASK:
            return PermissionResult.deny(
                f"{result.reason}; dontask mode denies prompts",
                decision_source=DecisionSource.MODE,
                updated_arguments=result.updated_arguments,
                metadata=result.metadata,
                matched_rule=result.matched_rule,
            )
        if self.mode is PermissionMode.AUTO and result.classifier_approvable:
            if tool.requires_user_interaction:
                return result
            metadata = dict(result.metadata or {})
            metadata["automated_evaluation"] = True
            return PermissionResult.ask(
                result.reason,
                decision_source=result.decision_source,
                updated_arguments=result.updated_arguments,
                metadata=metadata,
                matched_rule=result.matched_rule,
                classifier_approvable=True,
                bypass_immune=result.bypass_immune,
                suggestions=result.suggestions,
            )
        return result

    def refresh_managed_policy(self) -> str | None:
        """Refresh the administrator policy at a tool boundary; errors fail closed."""
        if self._managed_snapshot_override is not None:
            return None
        try:
            definition = self.managed_policy_provider.load()
        except Exception as exc:  # policy I/O/parser failures may never retain stale grants
            self._managed_reload_error = f"{type(exc).__name__}: {exc}"
            return self._managed_reload_error
        self.managed_policy = definition.snapshot()
        self.rules = definition.rules().merge(self._base_rules).merge(self._session_rules)
        self._managed_reload_error = None
        return None

    def _allow_match(self, tool_name: str, arguments: dict[str, Any]) -> PermissionRule | None:
        source = PermissionRuleSource.MANAGED if self.managed_policy.allow_managed_rules_only else None
        return self.rules.allow_match(tool_name, arguments, source=source)

    def decide(self, tool: "Tool", tool_call: "ToolCall | None" = None) -> PermissionDecision:
        """Synchronous compatibility wrapper; async callers must use ``evaluate``."""
        if tool_call is None:
            return self._legacy_without_arguments(tool)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.as_legacy_decision(asyncio.run(self.evaluate(tool, tool_call)))
        raise RuntimeError("PermissionPolicy.decide() cannot run inside an event loop; await evaluate()")

    def _legacy_without_arguments(self, tool: "Tool") -> PermissionDecision:
        """Preserve the pre-contract risk-only API when callers provide no arguments.

        Production execution always supplies a ToolCall and therefore always uses the
        schema/argument-aware pipeline.  This adapter exists for older embedded clients
        that used ``decide(tool)`` as a coarse capability probe.
        """
        if self.rules.deny_matches(tool.name, {}):
            return PermissionDecision(False, reason="denied by rule")
        if self.mode is PermissionMode.PLAN and tool.risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS}:
            return PermissionDecision(False, reason="plan mode is read-only")
        if self._session_allowed(tool.name, {}):
            return PermissionDecision(True, reason="allowed for this session")
        if self.rules.ask_matches(tool.name, {}):
            return self.as_legacy_decision(
                PermissionResult.ask("confirmation required by explicit ask rule", bypass_immune=True)
            )
        if self.mode is PermissionMode.BYPASS:
            return PermissionDecision(True, reason="bypass mode allows")
        if self._allow_match(tool.name, {}) is not None:
            return PermissionDecision(True, reason="allowed by rule")
        if tool.risk is ToolRisk.READ:
            return PermissionDecision(True, reason=f"{self.mode.value} allows read tools")
        if self.mode is PermissionMode.ACCEPTEDITS and tool.accept_edits_safe:
            return PermissionDecision(True, reason="acceptedits allows workspace file edits")
        if self.mode is PermissionMode.AUTO:
            return PermissionDecision(False, classify=True, reason="auto mode requires classification")
        if self.mode is PermissionMode.DONTASK:
            return PermissionDecision(False, reason="dontask mode denies prompts")
        return self.as_legacy_decision(PermissionResult.ask("default requires confirmation"))

    def as_legacy_decision(self, result: PermissionResult) -> PermissionDecision:
        if result.behavior is PermissionBehavior.ALLOW:
            return PermissionDecision(True, reason=result.reason)
        if result.behavior is PermissionBehavior.DENY:
            return PermissionDecision(False, reason=result.reason)
        classify = bool((result.metadata or {}).get("automated_evaluation"))
        if classify:
            return PermissionDecision(False, reason=result.reason, classify=True)
        if self.interactive:
            return PermissionDecision(False, ask_user=True, reason=result.reason)
        return PermissionDecision(False, reason=f"{result.reason}; non-interactive", ask_collapsed=True)

    def _session_allowed(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.managed_policy.allow_managed_rules_only:
            return False
        command_arg = _SHELL_COMMAND_TOOLS.get(name)
        if command_arg is None:
            return name in self._session_allow
        normalized = [
            _normalize_subcommand(part)
            for part in _split_subcommands(str(arguments.get(command_arg, "")))
        ]
        normalized = [part for part in normalized if part]
        return bool(normalized) and all(part in self._session_allow_commands for part in normalized)

    def _remember_always(self, tool: "Tool", tool_call: "ToolCall") -> None:
        command_arg = _SHELL_COMMAND_TOOLS.get(tool.name)
        if command_arg is None:
            self._session_allow.add(tool.name)
            return
        for part in _split_subcommands(str(tool_call.arguments.get(command_arg, ""))):
            normalized = _normalize_subcommand(part)
            if normalized:
                self._session_allow_commands.add(normalized)

    def suggested_rules(self, tool: "Tool", tool_call: "ToolCall") -> tuple[PermissionSuggestion, ...]:
        """Return least-privilege exact grants suitable for a session or config file."""
        command_arg = _SHELL_COMMAND_TOOLS.get(tool.name)
        if command_arg is not None:
            suggestions = []
            for part in _split_subcommands(str(tool_call.arguments.get(command_arg, ""))):
                normalized = _normalize_subcommand(part)
                if normalized:
                    suggestions.append(
                        PermissionSuggestion(
                            f"{tool.name}({normalized})",
                            "exact normalized shell subcommand",
                        )
                    )
            return tuple(suggestions)
        for key in ("path", "file_path", "url", "target", "pattern"):
            value = tool_call.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return (
                    PermissionSuggestion(
                        f"{tool.name}({value.strip()})",
                        f"exact {key} for this tool",
                    ),
                )
        return (PermissionSuggestion(tool.name, "whole-tool grant; review scope carefully"),)

    def add_session_rule(self, rule: str) -> None:
        parsed = RuleSet.from_lists(allow=[rule], source=PermissionRuleSource.SESSION)
        if not parsed.allow:
            raise ValueError(f"invalid permission rule: {rule!r}")
        self._session_rules = self._session_rules.merge(parsed)
        # Preserve the current managed definition while immediately activating the rule.
        self.rules = self.rules.merge(parsed)

    def inherit_scoped_session_grants(
        self, parent: "PermissionPolicy", allowed_tools: frozenset[str]
    ) -> None:
        """Copy only parent grants that remain inside an already-narrowed child registry."""
        if self.managed_policy.allow_managed_rules_only:
            return
        self._session_allow.update(parent._session_allow & allowed_tools)
        if allowed_tools & _SHELL_COMMAND_TOOLS.keys():
            self._session_allow_commands.update(parent._session_allow_commands)
        inherited = RuleSet(
            allow=[rule for rule in parent._session_rules.allow if rule.tool_name in allowed_tools]
        )
        self._session_rules = self._session_rules.merge(inherited)
        self.rules = self.rules.merge(inherited)

    def _apply_updates(self, updates: tuple[PermissionUpdate, ...]) -> None:
        destinations = {update.destination for update in updates}
        if len({item for item in destinations if item is not PermissionDestination.SESSION}) > 1:
            raise ValueError("one permission response cannot persist to multiple files")
        for update in updates:
            if update.behavior is not PermissionBehavior.ALLOW:
                raise ValueError("interactive permission updates may only add allow rules")
            if self.managed_policy.allow_managed_rules_only:
                raise ValueError("managed policy ignores non-managed allow rules")
            parsed = RuleSet.from_lists(allow=[update.rule], source=PermissionRuleSource.SESSION)
            if not parsed.allow:
                raise ValueError(f"invalid permission rule: {update.rule!r}")
            if update.destination is not PermissionDestination.SESSION:
                if self.managed_policy.disable_persistent_grants:
                    raise ValueError("managed policy disables persistent permission grants")
        persistent = [item for item in updates if item.destination is not PermissionDestination.SESSION]
        if persistent:
            workspace = self.workspace or Path.cwd().resolve()
            destination = persistent[0].destination
            persist_allow_rules(tuple(item.rule for item in persistent), destination, workspace)
        for update in updates:
            self.add_session_rule(update.rule)
        self.last_permission_updates = updates

    def confirm(self, decision: PermissionDecision, tool: "Tool", tool_call: "ToolCall") -> PermissionDecision:
        if not decision.ask_user or self.prompter is None:
            return decision
        self.last_permission_updates = ()
        request = PermissionRequest(
            tool.name,
            tool.risk.value,
            dict(tool_call.arguments),
            decision.reason,
            self.suggested_rules(tool, tool_call),
            self.managed_policy.disable_persistent_grants
            or self.managed_policy.allow_managed_rules_only,
            self.managed_policy.allow_managed_rules_only,
        )
        try:
            import inspect

            inspect.signature(self.prompter).bind(request)
        except (TypeError, ValueError):
            # Legacy embedded-client adapter: (tool_name, risk, arguments) -> choice.
            choice = self.prompter(tool.name, tool.risk.value, tool_call.arguments)
        else:
            choice = self.prompter(request)
        if isinstance(choice, PermissionResponse):
            if not choice.allow:
                return PermissionDecision(False, reason=choice.reason or "user rejected")
            try:
                self._apply_updates(choice.updates)
            except (OSError, ValueError, PermissionPersistenceError) as exc:
                return PermissionDecision(False, reason=f"permission grant was not applied: {exc}")
            if choice.updated_arguments is not None:
                tool_call.arguments = dict(choice.updated_arguments)
            return PermissionDecision(True, reason=choice.reason or "user confirmed")
        if choice == "always":
            self._remember_always(tool, tool_call)
            return PermissionDecision(True, reason="user allowed for this session")
        if choice == "once":
            return PermissionDecision(True, reason="user confirmed")
        return PermissionDecision(False, reason="user rejected")

    def _sandbox_state(self, name: str, arguments: dict[str, Any]) -> SandboxState:
        if self.sandbox is None:
            return SandboxState()
        command_arg = _SHELL_COMMAND_TOOLS.get(name)
        command: str | None
        if command_arg is not None:
            command = str(arguments.get(command_arg, ""))
        elif name == "run_tests":
            command = None
        else:
            return SandboxState(
                enabled=bool(getattr(self.sandbox, "is_enabled", lambda: False)()),
                backend=str(getattr(self.sandbox, "backend_name", "unknown")),
            )
        is_enabled = getattr(self.sandbox, "is_enabled", None)
        will_sandbox = self.sandbox.should_sandbox(command)
        enabled = bool(is_enabled()) if callable(is_enabled) else bool(will_sandbox)
        return SandboxState(
            enabled=enabled,
            backend=str(getattr(self.sandbox, "backend_name", "unknown")),
            will_sandbox=will_sandbox,
            excluded=enabled and not will_sandbox,
            auto_allow_enabled=bool(self.sandbox.config.auto_allow_command_if_sandboxed),
        )

    def _sandbox_auto_allows(self, name: str, arguments: dict[str, Any]) -> bool:
        state = self._sandbox_state(name, arguments)
        return state.auto_allow_enabled and state.will_sandbox and not state.excluded
