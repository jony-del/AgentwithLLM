"""TOFU (trust-on-first-use) policy for privilege-widening repo configuration (D2).

An in-repo ``agent.toml`` is repo-controlled input: cloning a repository must not be
able to grant itself allow rules, external hooks (arbitrary commands/URLs), sandbox
relaxations, MCP servers (arbitrary subprocesses launched at startup), autonomous
capability sources, or a web egress allowlist (unattended exfiltration targets). The rule:

- Repo config may always TIGHTEN policy — deny/ask rules and every non-widening table
  pass through untouched.
- The widening subset (see :func:`widening_subset`) requires user approval. On an
  interactive terminal the user is asked once; approval records the subset's
  fingerprint in the user-level trust store (``~/.polaris/trusted.json``, keyed by
  project path). Any later CHANGE to that subset re-prompts — the SSH host-key model.
- Unattended (no TTY), with no recorded trust: the widening subset is DROPPED with a
  warning — never silently honored. A previously recorded, unchanged fingerprint still
  counts as trust, so "run once interactively, then headless" works.

Pure stdlib; failures degrade in the strict direction (can't read/write the store →
treat as untrusted) and are always logged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default user-level trust store. Overridable via AGENT_TRUST_STORE (tests, odd homes).
DEFAULT_TRUST_STORE = "~/.polaris/trusted.json"


class TrustClassification(str, Enum):
    """How a repository-controlled setting affects host authority."""

    TIGHTENING = "tightening"
    NEUTRAL = "neutral"
    TRUSTED_ONLY = "trusted_only"


_MISSING = object()


@dataclass(frozen=True, slots=True)
class TrustRule:
    """One declarative entry in the repository-config trust matrix.

    ``capture_path`` allows a group such as ``capabilities`` to be fingerprinted and
    removed as one unit.  Extraction and filtering intentionally consume this same
    table, preventing their security decisions from drifting apart.
    """

    path: tuple[str, ...]
    label: str
    when: Callable[[Any], bool]
    capture_path: tuple[str, ...] | None = None
    classification: TrustClassification = TrustClassification.TRUSTED_ONLY


def _present(value: Any) -> bool:
    return value is not _MISSING and value not in (None, "", [], {}, ())


def _truthy(value: Any) -> bool:
    if value is _MISSING:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_false(value: Any) -> bool:
    return value is not _MISSING and not _truthy(value)


def _not_wsl2(value: Any) -> bool:
    return value is not _MISSING and str(value).casefold() != "wsl2"


def _privileged_permission(value: Any) -> bool:
    return str(value).strip().casefold() in {"acceptedits", "auto", "bypass", "bypasspermissions"}


# Values omitted here either do not affect authority or are explicitly tightening.
# Unknown keys inside the protected tables are handled fail-closed below.
TRUST_MATRIX: tuple[TrustRule, ...] = (
    TrustRule(("model",), "model", _present),
    TrustRule(("provider",), "provider", _present),
    TrustRule(("permission",), "permission", _privileged_permission),
    TrustRule(("permissions", "allow"), "permissions.allow", _present),
    TrustRule(("hooks", "external"), "hooks.external", _present),
    TrustRule(("hooks", "enabled"), "hooks.enabled", _is_false),
    TrustRule(("hooks", "prompt_validation", "enabled"), "hooks.prompt_validation.enabled", _is_false),
    TrustRule(
        ("hooks", "prompt_validation", "reject_control_chars"),
        "hooks.prompt_validation.reject_control_chars",
        _is_false,
    ),
    TrustRule(
        ("hooks", "prompt_validation", "neutralize_framing"),
        "hooks.prompt_validation.neutralize_framing",
        _is_false,
    ),
    TrustRule(("sandbox", "excluded_commands"), "sandbox.excluded_commands", _present),
    TrustRule(
        ("sandbox", "auto_allow_command_if_sandboxed"),
        "sandbox.auto_allow_command_if_sandboxed",
        _truthy,
    ),
    TrustRule(
        ("sandbox", "allow_unattended_unsandboxed"),
        "sandbox.allow_unattended_unsandboxed",
        _truthy,
    ),
    TrustRule(
        ("sandbox", "allow_unsandboxed_commands"),
        "sandbox.allow_unsandboxed_commands",
        _truthy,
    ),
    TrustRule(("sandbox", "fail_if_unavailable"), "sandbox.fail_if_unavailable", _is_false),
    TrustRule(
        ("sandbox", "network", "allowed_domains"),
        "sandbox.network.allowed_domains",
        _present,
    ),
    TrustRule(
        ("sandbox", "network", "allow_local_binding"),
        "sandbox.network.allow_local_binding",
        _truthy,
    ),
    TrustRule(("sandbox", "filesystem", "allow_read"), "sandbox.filesystem.allow_read", _present),
    TrustRule(("sandbox", "filesystem", "allow_write"), "sandbox.filesystem.allow_write", _present),
    TrustRule(("sandbox", "container", "runtime"), "sandbox.container.runtime", _present),
    TrustRule(("sandbox", "container", "image"), "sandbox.container.image", _present),
    TrustRule(("sandbox", "container", "oci_runtime"), "sandbox.container.oci_runtime", _present),
    TrustRule(("sandbox", "container", "auto_pull"), "sandbox.container.auto_pull", _truthy),
    TrustRule(
        ("sandbox", "container", "read_only_rootfs"),
        "sandbox.container.read_only_rootfs",
        _is_false,
    ),
    TrustRule(
        ("sandbox", "container", "drop_all_capabilities"),
        "sandbox.container.drop_all_capabilities",
        _is_false,
    ),
    TrustRule(
        ("sandbox", "container", "no_new_privileges"),
        "sandbox.container.no_new_privileges",
        _is_false,
    ),
    TrustRule(
        ("sandbox", "container", "windows_isolation"),
        "sandbox.container.windows_isolation",
        _not_wsl2,
    ),
    TrustRule(("sandbox", "vm", "provider"), "sandbox.vm.provider", _present),
    TrustRule(("sandbox", "vm", "base_image"), "sandbox.vm.base_image", _present),
    TrustRule(("sandbox", "vm", "vm_name"), "sandbox.vm.vm_name", _present),
    TrustRule(("sandbox", "vm", "snapshot_name"), "sandbox.vm.snapshot_name", _present),
    TrustRule(("sandbox", "vm", "guest_host"), "sandbox.vm.guest_host", _present),
    TrustRule(("sandbox", "vm", "reset_each_task"), "sandbox.vm.reset_each_task", _is_false),
    TrustRule(("mcp", "servers"), "mcp.servers", _present),
    TrustRule(
        ("capabilities",),
        "capabilities",
        _present,
        capture_path=("capabilities",),
    ),
    TrustRule(("tools", "execution_policies"), "tools.execution_policies", _present),
    TrustRule(("tools", "policies"), "tools.execution_policies", _present),
    TrustRule(("tools", "shell", "bash", "executable"), "tools.shell.bash.executable", _present),
    TrustRule(
        ("tools", "shell", "powershell", "executable"),
        "tools.shell.powershell.executable",
        _present,
    ),
    TrustRule(("tools", "lsp", "servers"), "tools.lsp.servers", _present),
    TrustRule(("tools", "lsp", "autodetect"), "tools.lsp.autodetect", _truthy),
    TrustRule(("tools", "scheduler", "database"), "tools.scheduler.database", _present),
    TrustRule(("tools", "worktree", "root"), "tools.worktree.root", _present),
    TrustRule(("skills", "user_dir"), "skills.user_dir", _present),
    TrustRule(("skills", "skills_dirs"), "skills.skills_dirs", _present),
    TrustRule(("plugins",), "plugins", _present, capture_path=("plugins",)),
    TrustRule(("web", "allowed_domains"), "web.allowed_domains", _present),
)


_PROTECTED_KEYS: dict[tuple[str, ...], frozenset[str]] = {
    ("permissions",): frozenset({"allow", "ask", "deny"}),
    ("hooks",): frozenset({"enabled", "external", "builtin", "prompt_validation"}),
    ("hooks", "prompt_validation"): frozenset(
        {"enabled", "max_chars", "reject_control_chars", "neutralize_framing"}
    ),
    ("sandbox",): frozenset(
        {
            "enabled", "backend", "fail_if_unavailable",
            "auto_allow_command_if_sandboxed", "allow_unsandboxed_commands",
            "allow_unattended_unsandboxed", "excluded_commands", "network",
            "filesystem", "container", "vm",
        }
    ),
    ("sandbox", "network"): frozenset({"allowed_domains", "allow_local_binding"}),
    ("sandbox", "filesystem"): frozenset({"allow_write", "deny_write", "deny_read", "allow_read"}),
    ("sandbox", "container"): frozenset(
        {
            "runtime", "image", "auto_pull", "oci_runtime", "read_only_rootfs",
            "drop_all_capabilities", "no_new_privileges", "memory", "cpus",
            "pids_limit", "windows_isolation",
        }
    ),
    ("sandbox", "vm"): frozenset(
        {"provider", "base_image", "vm_name", "snapshot_name", "guest_host", "reset_each_task"}
    ),
    ("mcp",): frozenset({"servers"}),
    ("tools",): frozenset({"execution_policies", "policies", "shell", "lsp", "notebook", "worktree", "scheduler"}),
    ("web",): frozenset({"allowed_domains", "blocked_domains"}),
}


def trust_store_path() -> Path:
    env = os.getenv("AGENT_TRUST_STORE")
    return Path(env).expanduser() if env else Path(DEFAULT_TRUST_STORE).expanduser()


def widening_subset(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract the privilege-widening subset of a raw agent.toml mapping.

    Only entries that actually widen are captured (an empty allow list or a
    default-valued flag is not a grant), so a repo with no widening config never
    prompts. The subset is also what gets fingerprinted — adding/changing any of
    these re-triggers TOFU.
    """
    subset: dict[str, Any] = {}
    for rule in TRUST_MATRIX:
        value = _get_path(raw, rule.path)
        if rule.classification is TrustClassification.TRUSTED_ONLY and rule.when(value):
            capture = rule.capture_path or rule.path
            subset[rule.label] = copy.deepcopy(_get_path(raw, capture))
    for path, value in _unknown_protected_entries(raw):
        subset[".".join(path)] = copy.deepcopy(value)
    return subset


def strip_widening(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``raw`` with every widening entry removed.

    Tightening content (deny/ask rules, builtin hook toggles, every other table)
    survives untouched.
    """
    out = copy.deepcopy(raw)
    removals: set[tuple[str, ...]] = set()
    for rule in TRUST_MATRIX:
        if rule.classification is TrustClassification.TRUSTED_ONLY and rule.when(
            _get_path(raw, rule.path)
        ):
            removals.add(rule.capture_path or rule.path)
    removals.update(path for path, _value in _unknown_protected_entries(raw))
    for path in sorted(removals, key=len, reverse=True):
        _delete_path(out, path)
    return out


def _get_path(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = raw
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _delete_path(raw: dict[str, Any], path: tuple[str, ...]) -> None:
    if not path:
        return
    parent = _get_path(raw, path[:-1])
    if isinstance(parent, dict):
        parent.pop(path[-1], None)


def _unknown_protected_entries(raw: dict[str, Any]) -> list[tuple[tuple[str, ...], Any]]:
    """Return unknown privilege-table keys so new syntax fails closed by default."""

    unknown: list[tuple[tuple[str, ...], Any]] = []
    for parent_path, known in _PROTECTED_KEYS.items():
        parent = _get_path(raw, parent_path)
        if not isinstance(parent, dict):
            continue
        for key, value in parent.items():
            if key not in known:
                unknown.append((parent_path + (str(key),), value))
    return unknown


def fingerprint(subset: dict[str, Any]) -> str:
    """Canonical sha256 of the widening subset (stable across key order)."""
    canonical = json.dumps(subset, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrustStore:
    """The user-level record of approved repo-config fingerprints.

    A flat JSON mapping ``{project_path: {"fingerprint": ..., "approved_at": ...}}``.
    Read/write failures degrade to "untrusted" (strict direction) with a log line.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or trust_store_path()

    def status(self, project: Path, fp: str) -> str:
        """``"trusted"`` (recorded and unchanged), ``"changed"``, or ``"unknown"``."""
        entry = self._load().get(str(project))
        if not isinstance(entry, dict):
            return "unknown"
        recorded = entry.get("fingerprint")
        if recorded == fp:
            return "trusted"
        return "changed" if recorded else "unknown"

    def record(self, project: Path, fp: str) -> None:
        data = self._load()
        data[str(project)] = {"fingerprint": fp, "approved_at": __import__("time").time()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("could not persist trust store %s: %s", self.path, exc)

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("could not read trust store %s (%s); treating as empty", self.path, exc)
            return {}


def _default_prompter(message: str) -> bool:
    """Interactive TOFU prompt; only usable on a real terminal."""
    try:
        print(message, file=sys.stderr)
        answer = input("Trust this repo configuration? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        return False
    return answer in {"y", "yes"}


def _render_prompt(project: Path, subset: dict[str, Any], changed: bool) -> str:
    head = (
        f"The repository config at {project} {'CHANGED its' if changed else 'requests'} "
        "privilege-widening settings (allow rules / external hooks / sandbox relaxations "
        "/ MCP servers / web egress allowlist). Repo config can tighten policy freely, but widening needs your "
        "approval (recorded per project; you will be re-asked if it changes):"
    )
    # Values can contain credentials (for example MCP headers).  The decision UI and
    # audit trail name affected settings without copying secrets into terminal logs.
    body = "\n".join(f"  - {key}" for key in sorted(subset))
    return f"{head}\n{body}"


def apply_repo_trust_policy(
    raw: dict[str, Any],
    *,
    project: Path,
    store: TrustStore | None = None,
    prompter: Callable[[str], bool] | None = None,
    interactive: bool | None = None,
) -> dict[str, Any]:
    """Enforce D2 on a repo-sourced raw config mapping; return the effective mapping.

    - no widening content → returned unchanged (no prompt, no store touch);
    - recorded, unchanged fingerprint → returned unchanged;
    - interactive → prompt; approval records the fingerprint and keeps the config,
      refusal (or a changed fingerprint the user declines) strips the widening subset;
    - unattended with no valid trust → widening subset stripped, with a warning.
    """
    subset = widening_subset(raw)
    if not subset:
        return raw

    store = store or TrustStore()
    fp = fingerprint(subset)
    status = store.status(project, fp)
    if status == "trusted":
        return raw

    if interactive is None:
        try:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):
            interactive = False

    if interactive:
        ask = prompter or _default_prompter
        if ask(_render_prompt(project, subset, changed=status == "changed")):
            store.record(project, fp)
            return raw
        logger.warning(
            "repo config widening DECLINED for %s; dropping: %s",
            project, ", ".join(sorted(subset)),
        )
        return strip_widening(raw)

    logger.warning(
        "unattended run: dropping untrusted privilege-widening repo config for %s "
        "(%s%s). Run once interactively to approve it (TOFU).",
        project,
        ", ".join(sorted(subset)),
        "; NOTE: previously trusted config has CHANGED" if status == "changed" else "",
    )
    return strip_widening(raw)
