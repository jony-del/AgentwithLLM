from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from agent_core.models import ToolRisk
from agent_core.permission_rules import (
    _SHELL_COMMAND_TOOLS,
    RuleSet,
    _normalize_subcommand,
    _split_subcommands,
)

if TYPE_CHECKING:
    from agent_core.models import ToolCall
    from agent_core.sandbox import SandboxManager
    from agent_core.tools.base import Tool

# Asks the user about a pending tool. Args: tool name, risk value, arguments.
# Returns "once" (run now), "always" (allow for the rest of the session), or "deny".
Prompter = Callable[[str, str, dict[str, Any]], str]

# File-write path arguments whose target is checked against the sensitive-path
# "safety net" (bypass-immune ask).
_SENSITIVE_PATH_KEYS = ("path", "file_path")
_SENSITIVE_DIRS = frozenset({".git", ".polaris", ".claude"})
_SENSITIVE_BASENAMES = frozenset({"agent.toml", "settings.json", "settings.local.json"})

# Secret-bearing paths: reading these is as dangerous as writing them (exfiltration),
# so the safety net below applies to EVERY risk class, READ included, and is
# bypass-immune. Directory names match path segments; basename patterns are fnmatch
# globs against the final component (case-insensitive).
_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".kube"})
_SECRET_BASENAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "credentials",
    "credentials.json",
)


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPTEDITS = "acceptedits"
    PLAN = "plan"
    AUTO = "auto"
    DONTASK = "dontask"
    # Allow everything except deny-rules, ask-rules, and the sensitive-path safety net.
    BYPASS = "bypass"


@dataclass(slots=True)
class PermissionDecision:
    allowed: bool
    dry_run: bool = False
    ask_user: bool = False
    reason: str = ""


class PermissionPolicy:
    """Argument-aware permission gate: fine-grained rules + mode + sandbox coupling.

    The decision pipeline (:meth:`decide`) mirrors Open-ClaudeCode's ordered evaluation:
    deny rules → sensitive-path safety net → ask rules → sandbox auto-allow → bypass →
    allow rules → the legacy per-mode ``ToolRisk`` matrix. The first four are
    *argument-aware* (they inspect the actual command/path/URL); the final matrix is the
    original coarse gate, preserved so a config with no rules behaves exactly as before.
    """

    def __init__(
        self,
        mode: PermissionMode | str = PermissionMode.DEFAULT,
        prompter: Prompter | None = None,
        rules: RuleSet | None = None,
        sandbox: "SandboxManager | None" = None,
    ) -> None:
        self.mode = PermissionMode(mode)
        # We can only ask the user when a prompter is wired (an interactive UI).
        # Without one, an "ask" collapses into a denial (matches old non-interactive).
        self.prompter = prompter
        self.interactive = prompter is not None
        # Fine-grained allow/deny/ask rules (empty by default → pure mode behavior).
        self.rules = rules or RuleSet()
        # The active OS sandbox manager, for the "auto-allow if sandboxed" coupling.
        self.sandbox = sandbox
        # Tools the user chose to "always allow" for the lifetime of this session.
        # Non-shell tools are remembered by tool name; shell-command tools are
        # remembered per normalized sub-command (so "always" for `git status` does
        # NOT silently allow `rm -rf` later — see _session_allowed/_remember_always).
        self._session_allow: set[str] = set()
        self._session_allow_commands: set[str] = set()

    def decide(self, tool: "Tool", tool_call: "ToolCall | None" = None) -> PermissionDecision:
        name = tool.name
        arguments = tool_call.arguments if tool_call is not None else {}

        # 1. Deny rules win outright (argument-aware; command decomposition inside).
        #    Nothing beats a deny — including a prior session "always".
        if self.rules.deny_matches(name, arguments):
            return PermissionDecision(False, reason="denied by rule")

        # 2. Secret safety net (bypass-immune, ALL risk classes): reading a
        #    secret-bearing path is an exfiltration channel, so even READ tools confirm.
        if _targets_secret_path(arguments):
            return self._maybe_ask("accessing a secret-bearing path requires confirmation")

        # 3. Sensitive-path safety net (bypass-immune): writing a protected path
        #    always confirms.
        if tool.risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS} and _targets_sensitive_path(arguments):
            return self._maybe_ask("writing a protected path requires confirmation")

        # 4. Session "always" grants — after deny and the safety nets, so neither
        #    can be washed out by an earlier broad confirmation.
        if self._session_allowed(name, arguments):
            return PermissionDecision(True, reason="allowed for this session")

        # 5. Ask rules → confirm, UNLESS the command will run sandboxed (next step).
        sandbox_allows = self._sandbox_auto_allows(name, arguments)
        if self.rules.ask_matches(name, arguments) and not sandbox_allows:
            return self._maybe_ask("confirmation required by rule")

        # 6. Sandbox coupling: a command that will actually be sandboxed skips the prompt
        #    (the OS sandbox is the real boundary).
        if sandbox_allows:
            return PermissionDecision(True, reason="allowed: command runs sandboxed")

        # 7. Bypass mode allows everything that survived deny/safety/ask above.
        if self.mode == PermissionMode.BYPASS:
            return PermissionDecision(True, reason="bypass mode allows")

        # 8. Explicit allow rules.
        if self.rules.allow_matches(name, arguments):
            return PermissionDecision(True, reason="allowed by rule")

        # 9. Fall back to the original per-mode ToolRisk matrix (unchanged behavior).
        return self._decide_by_mode(tool)

    # -- session "always" bookkeeping -------------------------------------------------

    def _session_allowed(self, name: str, arguments: dict[str, Any]) -> bool:
        """Whether a prior "always" covers this call.

        Shell-command tools are matched per normalized sub-command: every sub-command
        of the current line must have been individually always-allowed. Other tools
        keep the original tool-name grant.
        """
        command_arg = _SHELL_COMMAND_TOOLS.get(name)
        if command_arg is None:
            return name in self._session_allow
        subs = [
            _normalize_subcommand(sub)
            for sub in _split_subcommands(str(arguments.get(command_arg, "")))
        ]
        subs = [sub for sub in subs if sub]
        return bool(subs) and all(sub in self._session_allow_commands for sub in subs)

    def _remember_always(self, tool: "Tool", tool_call: "ToolCall") -> None:
        command_arg = _SHELL_COMMAND_TOOLS.get(tool.name)
        if command_arg is None:
            self._session_allow.add(tool.name)
            return
        for sub in _split_subcommands(str(tool_call.arguments.get(command_arg, ""))):
            normalized = _normalize_subcommand(sub)
            if normalized:
                self._session_allow_commands.add(normalized)

    def _decide_by_mode(self, tool: "Tool") -> PermissionDecision:
        risk = tool.risk
        if self.mode == PermissionMode.PLAN:
            if risk in {ToolRisk.WRITE, ToolRisk.DANGEROUS}:
                return PermissionDecision(True, dry_run=True, reason="plan mode dry-run")
            return PermissionDecision(True, reason="plan mode read allowed")
        if self.mode == PermissionMode.ACCEPTEDITS:
            if risk in {ToolRisk.READ, ToolRisk.WRITE}:
                return PermissionDecision(True, reason="acceptedits allows read/write")
            return self._maybe_ask("acceptedits requires confirmation for dangerous tools")
        if self.mode == PermissionMode.AUTO:
            if risk == ToolRisk.DANGEROUS:
                return PermissionDecision(False, reason="auto denies dangerous tools")
            return PermissionDecision(True, reason="auto allows read/write")
        if self.mode == PermissionMode.DONTASK:
            if risk == ToolRisk.DANGEROUS:
                return PermissionDecision(False, reason="dontask denies dangerous tools")
            return PermissionDecision(True, reason="dontask allows safe tools")
        if risk == ToolRisk.READ:
            return PermissionDecision(True, reason="default allows read tools")
        return self._maybe_ask("default requires confirmation")

    def _sandbox_auto_allows(self, name: str, arguments: dict[str, Any]) -> bool:
        """True when ``name`` is a shell-command tool whose command will be sandboxed."""
        if self.sandbox is None:
            return False
        command_arg = _SHELL_COMMAND_TOOLS.get(name)
        if command_arg is None:
            return False
        if not self.sandbox.config.auto_allow_command_if_sandboxed:
            return False
        command = str(arguments.get(command_arg, ""))
        return self.sandbox.should_sandbox(command)

    def confirm(self, decision: PermissionDecision, tool: "Tool", tool_call: "ToolCall") -> PermissionDecision:
        if not decision.ask_user or self.prompter is None:
            return decision
        choice = self.prompter(tool.name, tool.risk.value, tool_call.arguments)
        if choice == "always":
            self._remember_always(tool, tool_call)
            return PermissionDecision(True, reason="user allowed for this session")
        if choice == "once":
            return PermissionDecision(True, reason="user confirmed")
        return PermissionDecision(False, reason="user rejected")

    def _maybe_ask(self, reason: str) -> PermissionDecision:
        if self.interactive:
            return PermissionDecision(False, ask_user=True, reason=reason)
        return PermissionDecision(False, reason=f"{reason}; non-interactive")


def _targets_secret_path(arguments: dict[str, Any]) -> bool:
    """Whether a path argument targets secret-bearing material (.env, keys, creds).

    Applies to every risk class: reading a secret is an exfiltration channel, not
    just writing one. Same deliberate narrowness as ``_targets_sensitive_path`` —
    shell redirections are the deny-rule/sandbox layer's job.
    """
    for key in _SENSITIVE_PATH_KEYS:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            continue
        segments = value.replace("\\", "/").lower().split("/")
        if any(seg in _SECRET_DIRS for seg in segments):
            return True
        basename = segments[-1]
        if any(fnmatch.fnmatch(basename, pattern) for pattern in _SECRET_BASENAME_PATTERNS):
            return True
    return False


def _targets_sensitive_path(arguments: dict[str, Any]) -> bool:
    """Whether a file-write argument targets a protected path (config / .git / .polaris).

    Deliberately narrow (path-bearing tools only): a shell command that redirects into a
    protected path is not caught here — that surface is covered by deny rules and the OS
    sandbox instead.
    """
    for key in _SENSITIVE_PATH_KEYS:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            continue
        segments = value.replace("\\", "/").lower().split("/")
        if any(seg in _SENSITIVE_DIRS for seg in segments):
            return True
        if segments[-1] in _SENSITIVE_BASENAMES:
            return True
    return False
