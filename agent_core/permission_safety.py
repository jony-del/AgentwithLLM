"""Central, tool-independent security invariants for permission evaluation."""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_core.permission_rules import _match_domain
from agent_core.permission_types import (
    DecisionSource,
    PermissionContext,
    PermissionBehavior,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    PermissionRuleSource,
)

_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".kube"})
_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "credentials",
    "credentials.json",
)
_PROTECTED_DIRS = frozenset({".git", ".polaris", ".claude"})
_PROTECTED_FILES = frozenset({"agent.toml", "settings.json", "settings.local.json"})

_READ_PATH_TOOLS = {
    "list_dir": ("path", "."),
    "search_text": ("path", "."),
    "glob": ("path", "."),
    "read_text_file": ("path", None),
    "git_diff": ("path", "."),
}
_WRITE_PATH_TOOLS = {
    "write_text_file": "path",
    "edit_file": "path",
    "multi_edit": "path",
}


@dataclass(frozen=True, slots=True)
class PathTarget:
    raw: str
    operation: str


def is_secret_path(path: str | Path) -> bool:
    parts = [part.casefold() for part in Path(str(path).replace("\\", "/")).parts]
    if any(part in _SECRET_DIRS for part in parts):
        return True
    basename = parts[-1] if parts else ""
    return any(fnmatch.fnmatch(basename, pattern) for pattern in _SECRET_PATTERNS)


def is_protected_path(path: str | Path) -> bool:
    parts = [part.casefold() for part in Path(str(path).replace("\\", "/")).parts]
    if any(part in _PROTECTED_DIRS for part in parts):
        return True
    return bool(parts) and parts[-1] in _PROTECTED_FILES


def _is_persistence_path(path: str | Path) -> bool:
    parts = [part.casefold() for part in Path(str(path).replace("\\", "/")).parts]
    return ".git" in parts and "hooks" in parts


def extract_path_targets(tool_name: str, arguments: dict[str, Any]) -> list[PathTarget]:
    read_spec = _READ_PATH_TOOLS.get(tool_name)
    if read_spec is not None:
        key, default = read_spec
        value = arguments.get(key, default)
        return [PathTarget(str(value), "read")] if value is not None else []

    write_key = _WRITE_PATH_TOOLS.get(tool_name)
    if write_key is not None:
        value = arguments.get(write_key)
        return [PathTarget(str(value), "write")] if value is not None else []

    if tool_name == "apply_patch":
        patch = str(arguments.get("patch", ""))
        targets: list[PathTarget] = []
        for line in patch.splitlines():
            if not line.startswith("+++ "):
                continue
            target = line[4:].strip().split("\t", 1)[0]
            if target == "/dev/null":
                continue
            if target.startswith("b/"):
                target = target[2:]
            targets.append(PathTarget(target, "write"))
        return targets

    if tool_name == "run_tests" and arguments.get("target"):
        target = str(arguments["target"]).split("::", 1)[0]
        if target and not target.startswith("-"):
            return [PathTarget(target, "read")]
    return []


def _resolve_workspace_target(workspace: Path, raw: str) -> Path:
    if "\x00" in raw:
        raise ValueError("path contains a NUL byte")
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (workspace / candidate).resolve(strict=False)
    root = workspace.resolve(strict=False)
    try:
        common = Path(os.path.commonpath([os.path.normcase(str(root)), os.path.normcase(str(resolved))]))
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {raw}") from exc
    if os.path.normcase(str(common)) != os.path.normcase(str(root)):
        raise ValueError(f"path escapes workspace: {raw}")
    return resolved


def inspect_paths(
    tool_name: str,
    arguments: dict[str, Any],
    context: PermissionContext,
) -> PermissionResult | None:
    targets = extract_path_targets(tool_name, arguments)
    if tool_name == "apply_patch" and str(arguments.get("patch", "")).strip() and not targets:
        return PermissionResult.deny(
            "patch targets could not be determined safely",
            decision_source=DecisionSource.CENTRAL_SAFETY,
        )

    resolved_targets: list[str] = []
    for target in targets:
        raw_secret = is_secret_path(target.raw)
        try:
            resolved = _resolve_workspace_target(context.workspace, target.raw)
        except (OSError, ValueError) as exc:
            reason = str(exc)
            if raw_secret:
                reason = f"secret-bearing path is not a permitted workspace target: {target.raw}"
            return PermissionResult.deny(
                reason,
                decision_source=DecisionSource.CENTRAL_SAFETY,
                metadata={"path": target.raw, "operation": target.operation},
            )
        resolved_targets.append(str(resolved))
        # Check both spellings: the supplied path catches a security-looking name on a
        # non-existent target; the resolved path catches a symlink to one.
        secret = raw_secret or is_secret_path(resolved)
        protected = is_protected_path(target.raw) or is_protected_path(resolved)
        if target.operation == "read" and secret:
            return PermissionResult.ask(
                "accessing a secret-bearing path requires explicit approval",
                decision_source=DecisionSource.CENTRAL_SAFETY,
                metadata={"path": target.raw, "operation": "read"},
                bypass_immune=True,
            )
        if target.operation == "write" and (
            _is_persistence_path(target.raw) or _is_persistence_path(resolved)
        ):
            return PermissionResult.deny(
                "writing a persistence path is prohibited",
                decision_source=DecisionSource.CENTRAL_SAFETY,
                metadata={"path": target.raw, "operation": "write"},
            )
        if target.operation == "write" and (secret or protected):
            return PermissionResult.ask(
                "writing a protected path or secret-bearing file requires explicit approval",
                decision_source=DecisionSource.CENTRAL_SAFETY,
                metadata={"path": target.raw, "operation": "write"},
                bypass_immune=True,
            )
    return None


def ordinary_read_permission(
    tool_name: str,
    arguments: dict[str, Any],
    context: PermissionContext,
) -> PermissionResult:
    allow_rule = context.rules.allow_match(tool_name, arguments) if context.rules is not None else None
    if allow_rule is not None:
        return PermissionResult.allow(
            "read allowed by rule",
            decision_source=DecisionSource.RULE,
            matched_rule=allow_rule,
        )
    return PermissionResult.allow("ordinary workspace read", decision_source=DecisionSource.TOOL)


def ordinary_write_permission(
    tool_name: str,
    arguments: dict[str, Any],
    context: PermissionContext,
) -> PermissionResult:
    allow_rule = context.rules.allow_match(tool_name, arguments) if context.rules is not None else None
    if allow_rule is not None:
        return PermissionResult.allow(
            "edit allowed by rule",
            decision_source=DecisionSource.RULE,
            matched_rule=allow_rule,
        )
    if tool_name in context.session_authorizations.tool_names:
        return PermissionResult.allow(
            "edit allowed for this session",
            decision_source=DecisionSource.RULE,
            matched_rule=PermissionRule(
                PermissionRuleSource.SESSION,
                PermissionBehavior.ALLOW,
                tool_name,
                raw=tool_name,
            ),
        )
    if context.mode in {PermissionMode.ACCEPTEDITS, PermissionMode.AUTO}:
        return PermissionResult.allow(
            "mode allows an ordinary workspace file edit",
            decision_source=DecisionSource.TOOL,
        )
    if context.mode is PermissionMode.BYPASS:
        return PermissionResult.passthrough("ordinary edit may be resolved by bypass mode")
    if context.mode is PermissionMode.PLAN:
        return PermissionResult.deny(
            "plan mode permits only the dedicated plan artifact",
            decision_source=DecisionSource.CENTRAL_SAFETY,
        )
    return PermissionResult.ask(
        "ordinary workspace file edit requires confirmation",
        decision_source=DecisionSource.TOOL,
    )


def check_web_endpoint(url_or_host: str, context: PermissionContext) -> PermissionResult | None:
    parsed = urlparse(url_or_host)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return PermissionResult.deny(
            "only http/https network targets are permitted",
            decision_source=DecisionSource.CENTRAL_SAFETY,
        )
    host = (parsed.hostname or url_or_host).strip().casefold().rstrip(".")
    if not host:
        return PermissionResult.deny("network target has no host", decision_source=DecisionSource.CENTRAL_SAFETY)
    for domain in context.web_policy.blocked_domains:
        if _match_domain(host, domain):
            return PermissionResult.deny(
                f"domain {host!r} is blocked by policy",
                decision_source=DecisionSource.CENTRAL_SAFETY,
            )
    if context.web_policy.unattended and not any(
        _match_domain(host, domain) for domain in context.web_policy.allowed_domains
    ):
        return PermissionResult.deny(
            f"unattended mode requires an allowlisted domain: {host!r}",
            decision_source=DecisionSource.CENTRAL_SAFETY,
        )
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and _unsafe_ip(literal):
        return PermissionResult.deny(
            f"refusing internal/private network address {literal}",
            decision_source=DecisionSource.CENTRAL_SAFETY,
        )
    return None


def resolve_and_check_public_host(url_or_host: str) -> None:
    """Raise ValueError unless every DNS answer for an endpoint is public."""
    parsed = urlparse(url_or_host)
    host = parsed.hostname or url_or_host
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _unsafe_ip(ip):
            raise ValueError(f"refusing internal/private address {ip} for host {host!r}")


def _unsafe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
