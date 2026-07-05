"""TOFU (trust-on-first-use) policy for privilege-widening repo configuration (D2).

An in-repo ``agent.toml`` is repo-controlled input: cloning a repository must not be
able to grant itself allow rules, external hooks (arbitrary commands/URLs), sandbox
relaxations, MCP servers (arbitrary subprocesses launched at startup), or a web
egress allowlist (unattended exfiltration targets). The rule:

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
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default user-level trust store. Overridable via AGENT_TRUST_STORE (tests, odd homes).
DEFAULT_TRUST_STORE = "~/.polaris/trusted.json"


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

    permissions = raw.get("permissions")
    if isinstance(permissions, dict) and permissions.get("allow"):
        subset["permissions.allow"] = permissions["allow"]

    hooks = raw.get("hooks")
    if isinstance(hooks, dict) and hooks.get("external"):
        subset["hooks.external"] = hooks["external"]

    sandbox = raw.get("sandbox")
    if isinstance(sandbox, dict):
        if sandbox.get("excluded_commands"):
            subset["sandbox.excluded_commands"] = sandbox["excluded_commands"]
        for flag in ("auto_allow_command_if_sandboxed", "allow_unattended_unsandboxed"):
            if _truthy(sandbox.get(flag)):
                subset[f"sandbox.{flag}"] = True

    mcp = raw.get("mcp")
    if isinstance(mcp, dict) and mcp.get("servers"):
        subset["mcp.servers"] = mcp["servers"]

    # [web].allowed_domains widens egress in unattended modes (D10). blocked_domains
    # only tightens and passes through freely.
    web = raw.get("web")
    if isinstance(web, dict) and web.get("allowed_domains"):
        subset["web.allowed_domains"] = web["allowed_domains"]

    return subset


def strip_widening(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``raw`` with every widening entry removed.

    Tightening content (deny/ask rules, builtin hook toggles, every other table)
    survives untouched.
    """
    out = copy.deepcopy(raw)
    permissions = out.get("permissions")
    if isinstance(permissions, dict):
        permissions.pop("allow", None)
    hooks = out.get("hooks")
    if isinstance(hooks, dict):
        hooks.pop("external", None)
    sandbox = out.get("sandbox")
    if isinstance(sandbox, dict):
        sandbox.pop("excluded_commands", None)
        sandbox.pop("auto_allow_command_if_sandboxed", None)
        sandbox.pop("allow_unattended_unsandboxed", None)
    mcp = out.get("mcp")
    if isinstance(mcp, dict):
        mcp.pop("servers", None)
    web = out.get("web")
    if isinstance(web, dict):
        web.pop("allowed_domains", None)
    return out


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


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
    body = json.dumps(subset, indent=2, ensure_ascii=False, default=str)
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
