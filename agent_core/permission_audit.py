"""Safe summaries and redaction for permission and tool audit records."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent_core.command_security import analyze_command
from agent_core.permission_types import PermissionContext, PermissionResult

PERMISSION_AUDIT_SCHEMA_VERSION = 1

_SENSITIVE_KEY = re.compile(
    r"(?i)(?:password|passwd|token|authorization|cookie|api[_-]?key|private[_-]?key|credential|secret)"
)
_PEM_BLOCK = re.compile(
    r"-----BEGIN [^-\r\n]+-----.*?-----END [^-\r\n]+-----",
    re.DOTALL,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b([A-Z0-9_.-]*(?:TOKEN|PASSWORD|SECRET|API[_-]?KEY|CREDENTIAL)[A-Z0-9_.-]*)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_AUTH_VALUE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")


def summarize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        lowered = key.casefold()
        if _SENSITIVE_KEY.search(lowered):
            summary[key] = "<redacted>"
        elif lowered in {"content", "old_string", "new_string", "patch"}:
            text = str(value)
            summary[key] = _digest_summary(text)
        elif lowered == "command":
            text = str(value)
            analysis = analyze_command(text)
            summary[key] = {
                "sha256": _sha256(text),
                "chars": len(text),
                "segments": len(analysis.segments),
                "category": analysis.category,
            }
        elif lowered == "url":
            summary[key] = _sanitize_url(str(value))
        else:
            summary[key] = _bounded_safe_value(value)
    summary["_tool"] = tool_name
    return summary


def build_permission_audit_event(
    tool_name: str,
    arguments: dict[str, Any],
    context: PermissionContext,
    result: PermissionResult,
    classifier_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matched_rule = None
    rule_source = None
    if result.matched_rule is not None:
        matched_rule = {
            "tool": result.matched_rule.tool_name,
            "content": result.matched_rule.content,
            "behavior": result.matched_rule.behavior.value,
        }
        rule_source = result.matched_rule.source.value
    return {
        "schema_version": PERMISSION_AUDIT_SCHEMA_VERSION,
        "tool": tool_name,
        "arguments_summary": summarize_arguments(tool_name, arguments),
        "mode": context.mode.value,
        "final_behavior": result.behavior.value,
        "reason": redact_secret_material(result.reason),
        "decision_source": result.decision_source.value,
        "matched_rule": matched_rule,
        "rule_source": rule_source,
        "sandboxed": context.sandbox.will_sandbox,
        "classifier_result": sanitize_log_payload(classifier_result),
        "parent_agent_id": context.parent_agent_id,
        "tool_source": context.tool_source.value,
    }


def summarize_tool_result(content: str, metadata: dict[str, Any], ok: bool) -> dict[str, Any]:
    return {
        "ok": ok,
        "content": _digest_summary(content),
        "metadata": sanitize_log_payload(metadata),
    }


def sanitize_log_payload(value: Any, key: str = "") -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _SENSITIVE_KEY.search(key):
            return "<redacted>"
        if key.casefold() == "url":
            return _sanitize_url(value)
        return redact_secret_material(value)
    if isinstance(value, dict):
        return {str(k): sanitize_log_payload(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_log_payload(item, key) for item in value]
    try:
        return sanitize_log_payload(asdict(value), key)
    except (TypeError, ValueError):
        return redact_secret_material(str(value))


def redact_secret_material(text: str) -> str:
    redacted = _PEM_BLOCK.sub("<redacted-pem>", text)
    redacted = _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    redacted = _AUTH_VALUE.sub(lambda match: f"{match.group(1)} <redacted>", redacted)
    return redacted


def _bounded_safe_value(value: Any) -> Any:
    if isinstance(value, str):
        safe = redact_secret_material(value)
        return safe if len(safe) <= 200 else {**_digest_summary(safe), "preview": safe[:80]}
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, list):
        return [_bounded_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _bounded_safe_value(item) for key, item in list(value.items())[:20]}
    return _bounded_safe_value(str(value))


def _digest_summary(text: str) -> dict[str, Any]:
    return {"chars": len(text), "sha256": _sha256(text)}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid-url>"
    host = parsed.hostname or ""
    netloc = host
    try:
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
