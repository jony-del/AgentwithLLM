"""Fine-grained, argument-aware permission rules — the cross-platform *policy layer*.

This is the portable core adapted from Open-ClaudeCode's permission engine
(``shellRuleMatching.ts`` + ``bashPermissions.ts`` + the rule model in
``types/permissions.ts``). It has **no OS dependency** and does the real work of
"细粒度权限控制": deciding whether a *specific* tool call is allowed, denied, or
should be asked about, based on rules written as ``ToolName(content)`` strings.

The static per-tool ``ToolRisk`` gate in :mod:`agent_core.permissions` only sees the
tool *class*; these rules see the *arguments* (the shell command, the path, the URL),
so a user can say ``run_command(git *)`` allow but ``run_command(rm *)`` deny.

Design invariants (mirroring the reference + the project's security-provenance stance):

- **Fail safe.** A rule string we can't parse is *dropped*, never raised — a sloppy
  config degrades to fewer rules, never a crash (parity with ``_parse_external_hook``).
- **Deny is aggressive, allow is conservative.** For a compound shell command, *any*
  sub-command matching a deny rule denies the whole thing; an allow only fires when
  *every* sub-command is covered. Deny is matched against both the raw and the
  normalized sub-command so wrapper/env-var tricks can't dodge it.
- **Anti-evasion.** Leading env-var assignments and safe wrappers (``timeout``,
  ``nohup`` …) are stripped before *allow* matching so ``FOO=bar npm run x`` still
  matches ``npm:*`` — but binary-hijack vars (``PATH``, ``LD_*``, ``DYLD_*``) are
  **never** stripped, so they can't be used to smuggle a different binary past an
  allow rule.

Pure functions only; nothing here is async.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

# Tools whose primary argument is a free-form shell command line, so their rule
# content is matched with shell decomposition + anti-evasion rather than as a plain
# string. Keyed by the argument that holds the command.
_SHELL_COMMAND_TOOLS = {"run_command": "command"}

# Argument keys tried (in order) to extract the single string a non-shell rule matches
# against — a path for file tools, a URL for web tools, a target for the test runner.
_CANDIDATE_KEYS = ("path", "file_path", "url", "target", "pattern")

# Safe leading ``NAME=value`` assignments that may be stripped before allow-matching a
# shell command. Deliberately small; anything not listed is left in place (fail safe).
SAFE_ENV_VARS = frozenset(
    {"LANG", "LC_ALL", "TERM", "TZ", "HOME", "USER", "PWD", "SHELL", "PYTHONUTF8", "PYTHONIOENCODING"}
)

# Env vars that can hijack which binary runs (or how it loads code). NEVER stripped:
# leaving them in the sub-command means it won't match a naive allow rule, so it falls
# through to ask/deny. Mirrors the reference ``BINARY_HIJACK_VARS`` SECURITY note.
BINARY_HIJACK_VARS = re.compile(r"^(LD_|DYLD_|PATH$)")

# Command wrappers that are safe to strip *only* in their simplest, unflagged form
# (``nohup cmd``) or with a single well-formed leading argument we understand
# (``timeout 30 cmd``). Any unrecognized flag → we stop stripping (fail safe).
_SAFE_WRAPPERS = frozenset({"timeout", "nohup", "nice", "stdbuf", "ionice"})

# Top-level shell operators a compound command is split on. Matched outside quotes.
_SHELL_OPERATORS = ("&&", "||", "|", ";", "\n")

_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ParsedRule:
    """One parsed ``ToolName(content)`` rule.

    ``content`` is ``None`` for a whole-tool rule (``WebFetch`` denies the tool
    entirely); otherwise it's the raw inside-the-parens pattern.
    """

    tool_name: str
    content: str | None = None


def parse_rule(raw: str) -> ParsedRule | None:
    """Parse ``"ToolName(content)"`` / ``"ToolName"`` → :class:`ParsedRule`, or ``None``.

    Returns ``None`` (drop the rule) for anything that isn't a well-formed rule string,
    so a bad entry degrades instead of crashing rule-set construction.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    open_paren = text.find("(")
    if open_paren == -1:
        # Whole-tool rule, e.g. "WebFetch".
        return ParsedRule(text)
    if not text.endswith(")"):
        return None
    tool = text[:open_paren].strip()
    content = text[open_paren + 1 : -1].strip()
    if not tool:
        return None
    return ParsedRule(tool, content or None)


@dataclass(slots=True)
class RuleSet:
    """A parsed, argument-aware allow/deny/ask rule set.

    Rules from every source (defaults → toml → env → cli → session) are merged into
    these three flat lists; precedence between *behaviors* is enforced by the decision
    pipeline in :meth:`agent_core.permissions.PermissionPolicy.decide` (deny beats ask
    beats allow), not by source order.
    """

    allow: list[ParsedRule] = field(default_factory=list)
    deny: list[ParsedRule] = field(default_factory=list)
    ask: list[ParsedRule] = field(default_factory=list)

    @classmethod
    def from_lists(
        cls,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        ask: list[str] | None = None,
    ) -> "RuleSet":
        """Build from raw rule-string lists, silently dropping unparseable entries."""

        def _parse(items: list[str] | None) -> list[ParsedRule]:
            out: list[ParsedRule] = []
            for item in items or []:
                rule = parse_rule(item)
                if rule is not None:
                    out.append(rule)
            return out

        return cls(allow=_parse(allow), deny=_parse(deny), ask=_parse(ask))

    def merge(self, other: "RuleSet") -> "RuleSet":
        """Return a new set with ``other``'s rules appended (used to layer sources)."""
        return RuleSet(
            allow=self.allow + other.allow,
            deny=self.deny + other.deny,
            ask=self.ask + other.ask,
        )

    @property
    def is_empty(self) -> bool:
        return not (self.allow or self.deny or self.ask)

    # -- behavior queries used by the decision pipeline -------------------------------

    def deny_matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return self._matches(self.deny, tool_name, arguments, behavior=PermissionBehavior.DENY)

    def ask_matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return self._matches(self.ask, tool_name, arguments, behavior=PermissionBehavior.ASK)

    def allow_matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        return self._matches(self.allow, tool_name, arguments, behavior=PermissionBehavior.ALLOW)

    def _matches(
        self,
        rules: list[ParsedRule],
        tool_name: str,
        arguments: dict[str, Any],
        *,
        behavior: PermissionBehavior,
    ) -> bool:
        scoped = [r for r in rules if r.tool_name == tool_name]
        if not scoped:
            return False
        # A whole-tool rule (no content) governs every call to that tool.
        if any(r.content is None for r in scoped):
            return True
        patterns = [r.content for r in scoped if r.content is not None]

        command_arg = _SHELL_COMMAND_TOOLS.get(tool_name)
        if command_arg is not None:
            command = str(arguments.get(command_arg, ""))
            return _match_shell_command(command, patterns, behavior)

        candidate = _candidate_string(tool_name, arguments)
        if candidate is None:
            return False
        return any(_match_scalar(tool_name, candidate, p) for p in patterns)


# -- scalar (path / url / target) matching ------------------------------------------


def _candidate_string(tool_name: str, arguments: dict[str, Any]) -> str | None:
    for key in _CANDIDATE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _match_scalar(tool_name: str, candidate: str, pattern: str) -> bool:
    """Match a single string candidate against one rule pattern.

    Supports the ``domain:example.com`` form for web tools (host suffix match) and
    otherwise treats the pattern as a glob (``*``/``**``/``?``) over the candidate.
    """
    if pattern.startswith("domain:"):
        return _match_domain(candidate, pattern[len("domain:") :])
    return _glob_match(pattern, candidate)


def _match_domain(candidate: str, domain: str) -> bool:
    host = urlparse(candidate).hostname or candidate
    host = host.lower().rstrip(".")
    domain = domain.lower().rstrip(".")
    if not domain:
        return False
    return host == domain or host.endswith("." + domain)


def _glob_match(pattern: str, text: str) -> bool:
    """Glob match where ``**`` spans path separators and ``*`` does not.

    Falls back to a plain-substring check would be *too* loose, so we anchor the
    translated regex. Anything that fails to compile simply doesn't match (fail safe).
    """
    try:
        regex = _glob_to_regex(pattern)
    except re.error:
        return False
    return regex.match(text) is not None


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    out: list[str] = ["^"]
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")  # ** — spans separators
                i += 2
                # swallow an immediately following "/" so "**/x" matches "x"
                if i < n and pattern[i] == "/":
                    out.append("(?:.*/)?")
                    i += 1
                continue
            out.append("[^/\\\\]*")  # * — within a path segment
        elif char == "?":
            out.append("[^/\\\\]")
        else:
            out.append(re.escape(char))
        i += 1
    out.append("$")
    return re.compile("".join(out))


# -- shell command matching ----------------------------------------------------------


def _match_shell_command(command: str, patterns: list[str], behavior: PermissionBehavior) -> bool:
    """Decompose a (possibly compound) shell command and match it against patterns.

    - DENY: True if *any* sub-command matches *any* deny pattern (checked against both
      the raw and the normalized form, so wrappers/env vars can't dodge a deny).
    - ALLOW: True only if *every* sub-command matches *some* allow pattern.
    - ASK: True if *any* sub-command matches *any* ask pattern.
    """
    subcommands = _split_subcommands(command)
    if not subcommands:
        return False

    if behavior == PermissionBehavior.ALLOW:
        return all(
            any(_match_shell_rule(p, _normalize_subcommand(sub)) for p in patterns) for sub in subcommands
        )

    # DENY / ASK — any sub-command hitting any pattern is enough.
    for sub in subcommands:
        forms = {sub.strip(), _normalize_subcommand(sub)}
        if any(_match_shell_rule(p, form) for p in patterns for form in forms if form):
            return True
    return False


def _split_subcommands(command: str) -> list[str]:
    """Split a command line on top-level ``&& || | ; \\n`` outside quotes.

    A quote-aware character scan; not a full shell grammar, but enough that operators
    inside quoted strings ("echo a;b") don't over-split into fake sub-commands.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        char = command[i]
        if quote:
            buf.append(char)
            if char == quote:
                quote = None
            i += 1
            continue
        if char in ("'", '"'):
            quote = char
            buf.append(char)
            i += 1
            continue
        matched_op = None
        for op in _SHELL_OPERATORS:
            if command.startswith(op, i):
                matched_op = op
                break
        if matched_op is not None:
            parts.append("".join(buf))
            buf = []
            i += len(matched_op)
            continue
        buf.append(char)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _normalize_subcommand(sub: str) -> str:
    """Strip safe env-var prefixes and safe wrappers, then collapse whitespace.

    Stops stripping the moment it meets a binary-hijack env var or an unrecognized
    wrapper/flag, leaving the rest intact so it *won't* match a naive allow rule.
    """
    try:
        tokens = shlex.split(sub, posix=True)
    except ValueError:
        tokens = sub.split()
    tokens = _strip_leading_env(tokens)
    tokens = _strip_safe_wrappers(tokens)
    return " ".join(tokens).strip()


def _strip_leading_env(tokens: list[str]) -> list[str]:
    while tokens and _ENV_ASSIGN.match(tokens[0]):
        name = tokens[0].split("=", 1)[0]
        if BINARY_HIJACK_VARS.match(name) or name not in SAFE_ENV_VARS:
            break  # never strip hijack/unknown vars — fail safe
        tokens = tokens[1:]
    return tokens


def _strip_safe_wrappers(tokens: list[str]) -> list[str]:
    """Strip a leading safe wrapper (and its one understood argument), fixed-point.

    Conservative: ``timeout``/``nice``/``stdbuf``/``ionice`` are stripped only when
    their single leading argument is a plain value (a number, ``30s``, an ``NN``); any
    flag we don't understand halts stripping so an injected ``timeout -k$(id) 10`` is
    left intact and won't match an allow rule.
    """
    changed = True
    while changed and tokens:
        changed = False
        head = tokens[0]
        if head == "nohup":
            tokens = tokens[1:]
            changed = True
            continue
        if head == "env":
            rest = _strip_leading_env(tokens[1:])
            if rest != tokens[1:]:
                tokens = rest
                changed = True
            else:
                # bare "env cmd" — safe to drop; "env -i"/"env FOO=.." handled above/below
                if len(tokens) > 1 and not tokens[1].startswith("-"):
                    tokens = tokens[1:]
                    changed = True
            continue
        if head in _SAFE_WRAPPERS and len(tokens) >= 2:
            arg = tokens[1]
            if arg.startswith("-"):
                break  # unknown flag — stop (fail safe)
            if re.fullmatch(r"\d+[smhd]?", arg) or arg.isdigit():
                tokens = tokens[2:]
                changed = True
                continue
            # wrapper with no numeric arg (e.g. "nice cmd") — drop just the wrapper
            tokens = tokens[1:]
            changed = True
            continue
    return tokens


def _match_shell_rule(pattern: str, command: str) -> bool:
    """Match one shell command string against one rule pattern (exact/prefix/wildcard)."""
    kind, value = _parse_shell_pattern(pattern)
    if kind == "exact":
        return command == value
    if kind == "prefix":
        return command == value or command.startswith(value + " ")
    # wildcard
    try:
        return _shell_wildcard_regex(value).match(command) is not None
    except re.error:
        return False


def _parse_shell_pattern(pattern: str) -> tuple[str, str]:
    """Classify a rule pattern: ``npm:*`` → prefix, ``git * --force`` → wildcard, else exact.

    Mirrors the reference ``parsePermissionRule``: a trailing ``:*`` is the legacy
    prefix form; any other unescaped ``*`` is a wildcard; otherwise it's an exact match.
    """
    if pattern.endswith(":*"):
        return "prefix", pattern[:-2]
    if "*" in pattern:
        return "wildcard", pattern
    return "exact", pattern


def _shell_wildcard_regex(pattern: str) -> re.Pattern[str]:
    """``*`` → ``.*`` (DOTALL, so wildcards span heredocs/newlines).

    A trailing `` *`` also matches the bare command (``git *`` matches ``git``),
    matching the reference's optional-trailing-args behavior.
    """
    if pattern.endswith(" *"):
        body = re.escape(pattern[:-2])
        return re.compile("^" + body + "(?: .*)?$", re.DOTALL)
    escaped = re.escape(pattern).replace(r"\*", ".*")
    return re.compile("^" + escaped + "$", re.DOTALL)
