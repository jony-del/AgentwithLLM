"""Conservative semantic analysis for compound shell commands."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from agent_core.permission_rules import _normalize_subcommand, _split_subcommands
from agent_core.permission_types import PermissionBehavior

_HIJACK_ASSIGNMENT = re.compile(r"(?i)(?:^|\s)(?:PATH|LD_[A-Z0-9_]*|DYLD_[A-Z0-9_]*)=")
_POWERSHELL_HIJACK = re.compile(r"(?i)\$env:(?:PATH|LD_[A-Z0-9_]*|DYLD_[A-Z0-9_]*)\s*=")
_DYNAMIC_SHELL = re.compile(r"`|\$\(|<\(|>\(|\b(?:eval|Invoke-Expression)\b", re.IGNORECASE)
_NETWORK_TO_SHELL = re.compile(
    r"(?is)\b(?:curl|wget|Invoke-WebRequest|iwr)\b.*\|\s*(?:sh|bash|zsh|pwsh|powershell|python)\b"
)
_DESTRUCTIVE = (
    re.compile(r"(?i)\bgit\s+(?:reset\s+--hard|clean\s+-[^\s]*f)\b"),
    re.compile(r"(?i)\b(?:mkfs|diskpart|format)\b"),
    re.compile(r"(?i)\bdd\s+.*\bof="),
    re.compile(r"(?i)\bRemove-Item\b.*(?:-Recurse.*-Force|-Force.*-Recurse)"),
    re.compile(r"(?i)\brm\s+-[^\s]*r[^\s]*f[^\s]*\s+(?:/|~|[A-Za-z]:[\\/])"),
)
_PERSISTENCE = re.compile(
    r"(?i)(?:\.git[\\/]hooks|schtasks\s+/create|\bsc(?:\.exe)?\s+create|"
    r"\breg(?:\.exe)?\s+add\b.*\\Run\b|Set-ExecutionPolicy)"
)
_NETWORK = re.compile(
    r"(?i)\b(?:curl|wget|Invoke-WebRequest|iwr|Invoke-RestMethod|irm|ssh|scp|ftp|"
    r"pip\s+install|npm\s+(?:install|publish)|git\s+(?:push|fetch|pull|clone))\b"
)
_FILE_MUTATION = re.compile(
    r"(?i)\b(?:rm|mv|cp|mkdir|rmdir|touch|chmod|chown|sed\s+-i|tee|"
    r"Remove-Item|Move-Item|Copy-Item|New-Item|Set-Content|Add-Content)\b|(?:^|\s)(?:>>?|2>)"
)
_SECRET_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=:/\\])(?:\.env(?:\.[^\s\"']+)?|[^\s\"']+\.(?:pem|key|p12|pfx)|"
    r"id_(?:rsa|ed25519)[^\s\"']*|credentials(?:\.json)?|\.ssh|\.aws|\.gnupg|\.kube)"
)
_PROTECTED_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=:/\\])(?:\.git|\.polaris|\.claude|agent\.toml|"
    r"settings(?:\.local)?\.json)(?:$|[\s\"'/\\])"
)
_SHELL_WRAPPERS = frozenset({"bash", "sh", "zsh", "pwsh", "powershell", "cmd"})


@dataclass(frozen=True, slots=True)
class CommandAnalysis:
    behavior: PermissionBehavior
    reason: str
    segments: tuple[str, ...]
    category: str
    classifier_approvable: bool = False
    bypass_immune: bool = False


def analyze_command(command: str, *, _depth: int = 0) -> CommandAnalysis:
    if not command.strip():
        return CommandAnalysis(PermissionBehavior.DENY, "command must not be empty", (), "invalid")
    if _unbalanced_quotes(command):
        return CommandAnalysis(PermissionBehavior.DENY, "command contains unbalanced quotes", (), "invalid")
    if _depth > 3:
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "nested shell wrapper depth cannot be approved statically",
            (),
            "dynamic",
            bypass_immune=True,
        )

    segments = tuple(_split_subcommands(command))
    if not segments:
        return CommandAnalysis(PermissionBehavior.DENY, "command has no executable segment", (), "invalid")
    if _NETWORK_TO_SHELL.search(command):
        return CommandAnalysis(PermissionBehavior.DENY, "download-and-execute pipeline is prohibited", segments, "destructive")
    if _PERSISTENCE.search(command):
        return CommandAnalysis(PermissionBehavior.DENY, "persistence operation is prohibited", segments, "persistence")
    if any(pattern.search(command) for pattern in _DESTRUCTIVE):
        return CommandAnalysis(PermissionBehavior.DENY, "destructive command is prohibited", segments, "destructive")
    if _HIJACK_ASSIGNMENT.search(command) or _POWERSHELL_HIJACK.search(command):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "binary-hijack environment assignments require explicit approval",
            segments,
            "environment",
            bypass_immune=True,
        )
    if _DYNAMIC_SHELL.search(command):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "dynamic shell evaluation cannot be approved statically",
            segments,
            "dynamic",
            bypass_immune=True,
        )

    nested_analyses: list[CommandAnalysis] = []
    for segment in segments:
        wrapped = _wrapped_command(segment)
        if wrapped is None:
            continue
        if not wrapped:
            return CommandAnalysis(
                PermissionBehavior.ASK,
                "shell wrapper payload cannot be approved statically",
                segments,
                "dynamic",
                bypass_immune=True,
            )
        nested = analyze_command(wrapped, _depth=_depth + 1)
        if nested.behavior is PermissionBehavior.DENY:
            return CommandAnalysis(
                PermissionBehavior.DENY,
                nested.reason,
                segments,
                nested.category,
                bypass_immune=True,
            )
        nested_analyses.append(nested)

    if _SECRET_PATH.search(command):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "shell access to a secret path requires explicit approval",
            segments,
            "secret",
            bypass_immune=True,
        )
    if _PROTECTED_PATH.search(command) and _FILE_MUTATION.search(command):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "shell mutation of a protected path requires explicit approval",
            segments,
            "protected",
            bypass_immune=True,
        )

    wrapped_segments = {index for index, segment in enumerate(segments) if _wrapped_command(segment) is not None}
    categories = [
        _segment_category(segment) for index, segment in enumerate(segments) if index not in wrapped_segments
    ]
    categories.extend(analysis.category for analysis in nested_analyses)
    if all(category == "read" for category in categories):
        return CommandAnalysis(PermissionBehavior.ALLOW, "all command segments are read-only", segments, "read")
    if any(category == "network" for category in categories):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "network command requires explicit approval",
            segments,
            "network",
        )
    if any(category == "file_mutation" for category in categories):
        return CommandAnalysis(
            PermissionBehavior.ASK,
            "shell file mutation requires approval",
            segments,
            "file_mutation",
            classifier_approvable=True,
        )
    return CommandAnalysis(
        PermissionBehavior.ASK,
        "command semantics require automated or human review",
        segments,
        "development",
        classifier_approvable=True,
    )


def _wrapped_command(segment: str) -> str | None:
    """Return a static shell-wrapper payload, empty string when it cannot be isolated."""
    normalized = _normalize_subcommand(segment)
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError:
        return ""
    if not tokens:
        return None
    executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    executable = executable.removesuffix(".exe")
    if executable not in _SHELL_WRAPPERS:
        return None
    option_names = {"-c", "-lc", "-command", "/c"}
    for index, token in enumerate(tokens[1:], start=1):
        if token.casefold() in option_names:
            payload = tokens[index + 1 :]
            return " ".join(payload) if payload else ""
    return ""


def _segment_category(segment: str) -> str:
    normalized = _normalize_subcommand(segment)
    if _NETWORK.search(normalized):
        return "network"
    if _FILE_MUTATION.search(normalized):
        return "file_mutation"
    try:
        tokens = shlex.split(normalized, posix=True)
    except ValueError:
        return "unknown"
    if not tokens:
        return "unknown"
    executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
    args = [token.casefold() for token in tokens[1:]]
    if executable in {"pwd", "ls", "dir", "rg", "grep", "findstr", "where", "whoami"}:
        return "read"
    if executable in {"get-childitem", "get-location", "select-string"}:
        return "read"
    if executable == "git" and args:
        if args[0] in {"status", "diff", "log", "show", "rev-parse"}:
            return "read"
        if args[0] == "branch" and "--show-current" in args:
            return "read"
    if executable in {"python", "python.exe", "python3", "node", "pytest", "ruff", "mypy"}:
        return "development"
    return "unknown"


def _unbalanced_quotes(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
    return quote is not None
