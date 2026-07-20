from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # annotation-only imports; runtime imports stay deferred per-resolver
    from agent_core.compression import CompressionConfig
    from agent_core.hooks import ExternalHookSpec, HooksConfig, OutputLimitConfig
    from agent_core.mcp.config import MCPConfig
    from agent_core.memory.config import MemoryConfig
    from agent_core.permission_rules import RuleSet
    from agent_core.sandbox import SandboxConfig
    from agent_core.skills import SkillsConfig
    from agent_core.tool_use_summary import ToolUseSummaryConfig
    from agent_core.tools.web import WebPolicyConfig
    from agent_core.tool_config import ToolSuiteConfig

_T = TypeVar("_T")


def coerce_to_type(declared_type: Any, value: Any) -> Any:
    """Coerce a raw (often string) value to a dataclass field's declared type.

    With ``from __future__ import annotations`` a field's ``type`` is its *name* as
    a string (e.g. ``"int"``), so we match on those. Anything we don't recognise is
    passed through untouched.
    """
    if declared_type in ("bool", bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if declared_type in ("int", int):
        return int(value)
    if declared_type in ("float", float):
        return float(value)
    return value


def overlay_dataclass(config: _T, data: dict[str, Any] | None) -> _T:
    """Mutate ``config`` in place from a mapping (e.g. a toml table), then return it.

    Unknown keys are ignored and absent fields keep their defaults, so a partial or
    forward-compatible table loads cleanly. Values are coerced to each field's type.
    """
    if not data:
        return config
    valid = {field.name: field.type for field in fields(config)}  # type: ignore[arg-type]
    for key, value in data.items():
        if key in valid:
            setattr(config, key, coerce_to_type(valid[key], value))
    return config


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def _unquote_env_value(value: str) -> str:
    value = _strip_inline_comment(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key or (not override and key in os.environ):
            continue
        os.environ[key] = _unquote_env_value(value.strip())


# The conventional in-repo config filename. ONLY this default relative path is treated
# as repo-controlled input (D2 trust policy below); an explicitly passed path was chosen
# by the caller (user/test) and is honored as user-level config.
_REPO_DEFAULT_CONFIG = "agent.toml"

# Per-process cache of the trust-filtered repo config, keyed by (project, mtime, store),
# so the TOFU prompt fires at most once per process even though every resolver re-reads
# the file.
_TRUST_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}


def user_settings_path() -> Path:
    override = os.getenv("POLARIS_SETTINGS_PATH")
    return Path(override).expanduser() if override else Path.home() / ".polaris" / "settings.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_agent_toml(path: str | Path = "agent.toml") -> dict[str, Any]:
    config_path = Path(path)
    try:
        user = _read_toml(user_settings_path())
    except RuntimeError:
        user = {}
    raw = _read_toml(config_path)
    if str(path) == _REPO_DEFAULT_CONFIG and raw:
        raw = _apply_repo_trust(config_path, raw)
    return _deep_merge(user, raw)


def _apply_repo_trust(config_path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """D2: filter privilege-widening keys of the in-repo agent.toml through TOFU.

    Deny/ask rules and every non-widening table pass through; allow rules, external
    hooks, sandbox relaxations, and MCP servers require recorded (or freshly granted)
    user trust — see :mod:`agent_core.trust`. Failures degrade in the strict
    direction: if the policy itself errors, the widening keys are stripped.
    """
    import copy

    from agent_core import trust

    project = config_path.resolve().parent
    try:
        mtime = config_path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    key = (str(project), mtime, str(trust.trust_store_path()))
    cached = _TRUST_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    try:
        effective = trust.apply_repo_trust_policy(raw, project=project)
    except Exception as exc:  # noqa: BLE001 - policy failure must not widen privileges
        import logging

        logging.getLogger(__name__).warning(
            "repo-config trust policy failed (%s: %s); stripping widening keys",
            type(exc).__name__, exc,
        )
        effective = trust.strip_widening(raw)
    _TRUST_CACHE[key] = effective
    return copy.deepcopy(effective)


def resolve_config(
    cli_values: dict[str, Any],
    config_file: str | Path = "agent.toml",
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    load_dotenv(env_file)
    file_values = load_agent_toml(config_file)
    defaults = {
        "model": "claude-opus-4-8",
        "permission": "default",
        "provider": "claude",
        "effort": "high",
    }
    env_values = {
        "model": os.getenv("AGENT_MODEL"),
        "permission": os.getenv("AGENT_PERMISSION"),
        "provider": os.getenv("AGENT_PROVIDER"),
        "effort": os.getenv("AGENT_EFFORT"),
    }
    # Only scalar settings are merged here; the ``[memory]`` table is resolved
    # separately (see resolve_memory_config) so its nested fields don't collide
    # with the flat keys above.
    scalar_file_values = {key: value for key, value in file_values.items() if not isinstance(value, dict)}
    merged = {**defaults, **scalar_file_values}
    merged.update({key: value for key, value in env_values.items() if value is not None})
    merged.update({key: value for key, value in cli_values.items() if value is not None})
    from agent_core.permission_types import parse_permission_mode

    merged["permission"] = parse_permission_mode(str(merged["permission"])).value
    return merged


def resolve_memory_config(cli_enabled: bool | None, config_file: str | Path = "agent.toml") -> "MemoryConfig":
    """Resolve a :class:`MemoryConfig` from, lowest to highest priority:
    defaults → ``[memory]`` toml table → ``AGENT_MEMORY`` env → ``--memory`` CLI flag.

    Only ``enabled`` is overridable by env/CLI; the numeric tunables live in toml.
    """
    from agent_core.memory.config import MemoryConfig

    table = load_agent_toml(config_file).get("memory")
    config = MemoryConfig.from_dict(table if isinstance(table, dict) else None)

    env = os.getenv("AGENT_MEMORY")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}
    if cli_enabled is not None:
        config.enabled = cli_enabled
    return config


def resolve_web_config(config_file: str | Path = "agent.toml") -> "WebPolicyConfig":
    """Resolve the ``[web]`` outbound domain policy (decision D10).

    ``blocked_domains`` tightens and always applies; ``allowed_domains`` from an
    in-repo file is a privilege-widening key filtered through the TOFU trust policy
    (``load_agent_toml`` applies it before this function sees the table).
    """
    from agent_core.tools.web import WebPolicyConfig

    table = load_agent_toml(config_file).get("web")
    return WebPolicyConfig.from_dict(table if isinstance(table, dict) else None)


def resolve_tool_suite_config(config_file: str | Path = "agent.toml") -> "ToolSuiteConfig":
    """Resolve the nested ``[tools]`` lifecycle/capability configuration."""
    from agent_core.tool_config import ToolSuiteConfig

    table = load_agent_toml(config_file).get("tools")
    return ToolSuiteConfig.from_dict(table if isinstance(table, dict) else None)


def resolve_output_config(config_file: str | Path = "agent.toml") -> "OutputLimitConfig":
    """Resolve the tool-output truncation limits from the ``[output]`` toml table."""
    from agent_core.hooks import OutputLimitConfig

    table = load_agent_toml(config_file).get("output")
    return OutputLimitConfig.from_dict(table if isinstance(table, dict) else None)


def resolve_tool_use_summary_config(
    config_file: str | Path = "agent.toml",
) -> "ToolUseSummaryConfig":
    """Resolve the tool-use progress-label settings from the ``[tool_use_summary]`` toml table."""
    from agent_core.tool_use_summary import ToolUseSummaryConfig

    table = load_agent_toml(config_file).get("tool_use_summary")
    return ToolUseSummaryConfig.from_dict(table if isinstance(table, dict) else None)


def resolve_concurrency_config(config_file: str | Path = "agent.toml") -> dict[str, Any]:
    """Resolve concurrency settings from ``[concurrency]``, then env, then CLI.

    Covers both resource-level tool scheduling (``parallel_tools`` /
    ``max_tool_workers``) and the shared LLM API-call gate (``max_api_concurrency`` /
    ``api_rate_limit_per_min``). Precedence: defaults → ``agent.toml`` → env
    (``AGENT_MAX_API_CONCURRENCY`` / ``AGENT_API_RATE_LIMIT``). CLI flags, when given,
    are layered on by the caller.
    """
    table = load_agent_toml(config_file).get("concurrency")
    values = {
        "parallel_tools": True,
        "max_tool_workers": 4,
        "max_api_concurrency": 8,
        "api_rate_limit_per_min": 0,
    }
    if isinstance(table, dict):
        if "parallel_tools" in table:
            values["parallel_tools"] = coerce_to_type(bool, table["parallel_tools"])
        if "max_tool_workers" in table:
            values["max_tool_workers"] = max(1, coerce_to_type(int, table["max_tool_workers"]))
        if "max_api_concurrency" in table:
            values["max_api_concurrency"] = max(1, coerce_to_type(int, table["max_api_concurrency"]))
        if "api_rate_limit_per_min" in table:
            values["api_rate_limit_per_min"] = max(0, coerce_to_type(int, table["api_rate_limit_per_min"]))

    env_concurrency = os.getenv("AGENT_MAX_API_CONCURRENCY")
    if env_concurrency is not None:
        values["max_api_concurrency"] = max(1, coerce_to_type(int, env_concurrency))
    env_rate = os.getenv("AGENT_API_RATE_LIMIT")
    if env_rate is not None:
        values["api_rate_limit_per_min"] = max(0, coerce_to_type(int, env_rate))
    return values


def resolve_limits_config(config_file: str | Path = "agent.toml") -> dict[str, Any]:
    """Resolve the run-level safety limits from ``[limits]``, then env.

    Covers the wall-clock deadline and optional hard step cap that bound a whole
    ``ReActAgent.run()`` (the fan-out shares one wall budget). Precedence:
    defaults → ``agent.toml`` → env (``AGENT_MAX_WALL_SECONDS`` / ``AGENT_MAX_STEPS``).
    CLI flags, when given, are layered on by the caller.

    Convention: ``0`` (or absent) means *disabled* for ``max_wall_seconds`` and
    ``max_steps``, surfaced as ``None`` so an uncapped run relies on cooperative
    cancel alone. ``soft_deadline_fraction`` keeps the ``ReActConfig`` default
    unless overridden.
    """

    def _none_if_zero(value: Any) -> Any:
        return None if value is None or value <= 0 else value

    table = load_agent_toml(config_file).get("limits")
    values: dict[str, Any] = {
        "max_wall_seconds": 1800.0,
        "max_steps": None,
        "soft_deadline_fraction": 0.9,
    }
    if isinstance(table, dict):
        if "max_wall_seconds" in table:
            values["max_wall_seconds"] = _none_if_zero(coerce_to_type(float, table["max_wall_seconds"]))
        if "max_steps" in table:
            values["max_steps"] = _none_if_zero(coerce_to_type(int, table["max_steps"]))
        if "soft_deadline_fraction" in table:
            values["soft_deadline_fraction"] = coerce_to_type(float, table["soft_deadline_fraction"])

    env_wall = os.getenv("AGENT_MAX_WALL_SECONDS")
    if env_wall is not None:
        values["max_wall_seconds"] = _none_if_zero(coerce_to_type(float, env_wall))
    env_steps = os.getenv("AGENT_MAX_STEPS")
    if env_steps is not None:
        values["max_steps"] = _none_if_zero(coerce_to_type(int, env_steps))
    return values


def resolve_session_dir(config_file: str | Path = "agent.toml") -> str:
    """Resolve the resumable-transcript root: defaults → ``[session] dir`` → env.

    Precedence ends here; a ``--session-dir`` CLI flag is layered on by the caller.
    ``~`` is left unexpanded (the transcript layer expands it). An empty string
    disables session persistence — useful for tests and ``--no-session-persistence``.
    """
    value = "~/.polaris/projects"
    table = load_agent_toml(config_file).get("session")
    if isinstance(table, dict) and "dir" in table:
        value = str(table["dir"])
    env = os.getenv("AGENT_SESSION_DIR")
    if env is not None:
        value = env
    return value


def resolve_persist_compaction_boundary(config_file: str | Path = "agent.toml") -> bool:
    """Whether a compaction fold writes a compact boundary into the transcript.

    Precedence: default ``True`` → ``[session] persist_compaction_boundary`` →
    ``AGENT_PERSIST_COMPACT_BOUNDARY`` env. ``False`` keeps the transcript a faithful
    full record (resume reloads everything; the loop re-compacts).
    """
    value = True
    table = load_agent_toml(config_file).get("session")
    if isinstance(table, dict) and "persist_compaction_boundary" in table:
        value = coerce_to_type(bool, table["persist_compaction_boundary"])
    env = os.getenv("AGENT_PERSIST_COMPACT_BOUNDARY")
    if env is not None:
        value = env.strip().lower() in {"1", "true", "yes", "on"}
    return value


def resolve_context_config(config_file: str | Path = "agent.toml") -> dict[str, Any]:
    """Resolve one-time project-context settings from ``[context]``, then env.

    Covers CLAUDE.md project-instruction injection and the git-status snapshot.
    Precedence: defaults → ``agent.toml`` → env. The toggles have a single source of
    truth here (the ``context`` module itself stays env-free): ``AGENT_DISABLE_CLAUDE_MD``
    forces ``project_instructions`` off (mirroring the reference runtime's
    ``CLAUDE_CODE_DISABLE_CLAUDE_MDS``) and ``AGENT_DISABLE_GIT_CONTEXT`` forces
    ``git_context`` off, when truthy.
    """
    table = load_agent_toml(config_file).get("context")
    values: dict[str, Any] = {
        "project_instructions": True,
        "git_context": True,
        "claudemd_max_chars": 32000,
    }
    if isinstance(table, dict):
        if "project_instructions" in table:
            values["project_instructions"] = coerce_to_type(bool, table["project_instructions"])
        if "git_context" in table:
            values["git_context"] = coerce_to_type(bool, table["git_context"])
        if "claudemd_max_chars" in table:
            values["claudemd_max_chars"] = max(0, coerce_to_type(int, table["claudemd_max_chars"]))

    disable = os.getenv("AGENT_DISABLE_CLAUDE_MD")
    if disable is not None and coerce_to_type(bool, disable):
        values["project_instructions"] = False
    disable_git = os.getenv("AGENT_DISABLE_GIT_CONTEXT")
    if disable_git is not None and coerce_to_type(bool, disable_git):
        values["git_context"] = False
    return values


def resolve_compression_config(config_file: str | Path = "agent.toml") -> "CompressionConfig":
    """Resolve context-compaction settings from the ``[compression]`` toml table, env.

    Covers the deterministic shrink thresholds and the Track A (LLM summary) knobs.
    Precedence: defaults → ``[compression]`` → env. ``AGENT_DISABLE_LLM_SUMMARY``
    (truthy) forces ``use_llm_summary`` off so a run is pinned to deterministic
    Track B regardless of provider. ``AGENT_AUTOCOMPACT_PCT_OVERRIDE`` (percent of the
    effective window, 0 < p <= 100) lowers the token-based auto-compact threshold for
    testing. Unknown keys in the table are ignored.
    """
    from agent_core.compression import CompressionConfig

    table = load_agent_toml(config_file).get("compression")
    config = overlay_dataclass(CompressionConfig(), table if isinstance(table, dict) else None)

    disable = os.getenv("AGENT_DISABLE_LLM_SUMMARY")
    if disable is not None and coerce_to_type(bool, disable):
        config.use_llm_summary = False

    pct = os.getenv("AGENT_AUTOCOMPACT_PCT_OVERRIDE")
    if pct is not None:
        try:
            parsed = float(pct)
        except ValueError:
            parsed = 0.0
        if 0 < parsed <= 100:
            config.autocompact_pct_override = parsed
    return config


def resolve_skills_config(config_file: str | Path = "agent.toml") -> "SkillsConfig":
    """Resolve the skill subsystem settings from the ``[skills]`` toml table, then env.

    Precedence: defaults → ``[skills]`` → ``AGENT_SKILLS`` env (only toggles ``enabled``).
    Skills are an *enabled capability* (default on), discovered and loaded eagerly at
    agent startup. ``AGENT_SKILLS`` truthy/falsey lets a run force them on/off without
    editing toml. Unknown keys in the table are ignored.
    """
    from agent_core.skills import SkillsConfig

    table = load_agent_toml(config_file).get("skills")
    config = SkillsConfig.from_dict(table if isinstance(table, dict) else None)

    env = os.getenv("AGENT_SKILLS")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}
    return config


_HOOK_TYPES = frozenset({"command", "http", "prompt", "agent"})


def resolve_hooks_config(config_file: str | Path = "agent.toml") -> "HooksConfig":
    """Resolve the lifecycle-hook subsystem from the ``[hooks]`` toml table, then env.

    Shape::

        [hooks]
        enabled = true
        [hooks.builtin]            # toggles for built-in programmatic hooks
        stop_completion = true
        ...
        [hooks.prompt_validation]  # UserPromptSubmit input firewall (on by default)
        enabled = true
        ...
        [[hooks.external]]         # config-driven external hooks (flat array)
        event = "Stop"
        type = "command"
        command = "..."

    Precedence: defaults → ``[hooks]`` → ``AGENT_HOOKS`` env (only toggles ``enabled``).
    External specs are validated leniently: an entry with an unknown ``event``/``type``
    or a missing required field for its type is dropped (not raised), so a sloppy config
    degrades to fewer hooks instead of crashing construction. Unknown keys are ignored.
    """
    from agent_core.hooks import HookEvent, HooksConfig

    table = load_agent_toml(config_file).get("hooks")
    config = HooksConfig()
    if isinstance(table, dict):
        if "enabled" in table:
            config.enabled = coerce_to_type(bool, table["enabled"])
        builtin = table.get("builtin")
        if isinstance(builtin, dict):
            config.builtin = config.builtin.from_dict(builtin)
        prompt_validation = table.get("prompt_validation")
        if isinstance(prompt_validation, dict):
            config.prompt_validation = config.prompt_validation.from_dict(prompt_validation)
        external = table.get("external")
        if isinstance(external, list):
            valid_events = {event.value for event in HookEvent}
            for entry in external:
                spec = _parse_external_hook(entry, valid_events)
                if spec is not None:
                    config.external.append(spec)

    env = os.getenv("AGENT_HOOKS")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}
    return config


def _parse_external_hook(entry: object, valid_events: set[str]) -> "ExternalHookSpec | None":
    """Validate one ``[[hooks.external]]`` entry into an ``ExternalHookSpec`` or ``None``.

    Drops the entry (returns ``None``) when it is not a table, names an unknown event or
    type, or omits the field its type needs (``command``/``url``/``prompt``). This is the
    "degrade, don't crash" guard for untrusted-ish project config.
    """
    from agent_core.hooks import ExternalHookSpec

    if not isinstance(entry, dict):
        return None
    event = str(entry.get("event", "")).strip()
    hook_type = str(entry.get("type", "")).strip().lower()
    if event not in valid_events or hook_type not in _HOOK_TYPES:
        return None
    required = {"command": "command", "http": "url", "prompt": "prompt", "agent": "prompt"}[hook_type]
    if not entry.get(required):
        return None
    headers = entry.get("headers")
    timeout_raw = entry.get("timeout", 30.0)
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        timeout = 30.0
    # fail_mode is a SECURITY option, so its parse failure degrades in the strict
    # direction: anything that isn't exactly "open" becomes "closed" (a typo on a
    # gate hook must not silently fail open). Absent → "open" for the observational
    # events, but "closed" for the control-path PermissionRequest event (a crashed
    # approval gate must refuse, not wave things through — new gates default closed).
    fail_mode_raw = entry.get("fail_mode")
    if fail_mode_raw is None:
        fail_mode = "closed" if event == "PermissionRequest" else "open"
    else:
        fail_mode = "open" if str(fail_mode_raw).strip().lower() == "open" else "closed"
    return ExternalHookSpec(
        event=event,
        type=hook_type,
        matcher=(str(entry["matcher"]) if entry.get("matcher") is not None else None),
        command=(str(entry["command"]) if entry.get("command") is not None else None),
        url=(str(entry["url"]) if entry.get("url") is not None else None),
        prompt=(str(entry["prompt"]) if entry.get("prompt") is not None else None),
        model=(str(entry["model"]) if entry.get("model") is not None else None),
        headers=(dict(headers) if isinstance(headers, dict) else None),
        timeout=max(0.1, timeout),
        fail_mode=fail_mode,
    )


def resolve_sandbox_config(config_file: str | Path = "agent.toml") -> "SandboxConfig":
    """Resolve the OS sandbox settings from the ``[sandbox]`` toml table, then env.

    Precedence: defaults → ``[sandbox]`` (incl. ``[sandbox.network]`` /
    ``[sandbox.filesystem]`` / ``[sandbox.container]`` / ``[sandbox.vm]``) →
    ``AGENT_SANDBOX`` (toggles ``enabled``) / ``AGENT_SANDBOX_FAIL_IF_UNAVAILABLE`` /
    ``AGENT_SANDBOX_BACKEND`` (isolation tier) env. ``--sandbox/--no-sandbox`` and
    ``--sandbox-backend`` CLI flags are layered on by the caller. Disabled by default;
    unknown keys are ignored.
    """
    from agent_core.sandbox import SandboxConfig
    from agent_core.sandbox.config import _normalize_backend

    table = load_agent_toml(config_file).get("sandbox")
    config = SandboxConfig.from_dict(table if isinstance(table, dict) else None)

    env = os.getenv("AGENT_SANDBOX")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}
    env_fail = os.getenv("AGENT_SANDBOX_FAIL_IF_UNAVAILABLE")
    if env_fail is not None:
        config.fail_if_unavailable = env_fail.strip().lower() in {"1", "true", "yes", "on"}
    env_backend = os.getenv("AGENT_SANDBOX_BACKEND")
    if env_backend is not None:
        config.backend = _normalize_backend(env_backend)
    return config


def resolve_permission_rules(config_file: str | Path = "agent.toml") -> "RuleSet":
    """Resolve fine-grained allow/deny/ask rules from the ``[permissions]`` toml table.

    Shape::

        [permissions]
        allow = ["bash(git *)", "read_text_file(/src/**)"]
        deny  = ["bash(rm *)", "web_fetch(domain:evil.example)"]
        ask   = ["bash", "powershell"]

    Unparseable rule strings are dropped (not raised), so a sloppy table degrades to
    fewer rules. CLI ``--allow/--deny/--ask`` rules are layered on by the caller via
    :meth:`RuleSet.merge`. The permission *mode* stays the top-level ``permission`` key.
    """
    from agent_core.permission_rules import RuleSet
    from agent_core.permission_types import PermissionRuleSource

    def _load(path: str | Path, source: PermissionRuleSource) -> RuleSet:
        table = load_agent_toml(path).get("permissions")
        if not isinstance(table, dict):
            return RuleSet()

        def _rules(key: str) -> list[str]:
            value = table.get(key)
            rules = [str(item) for item in value] if isinstance(value, list) else []
            legacy = [item for item in rules if "run_command" in item]
            if legacy:
                raise ValueError(
                    f"legacy run_command rules are unsupported: {legacy!r}. Split by dialect, "
                    "for example bash(git *) and powershell(Get-ChildItem *)."
                )
            return rules

        return RuleSet.from_lists(
            _rules("allow"), _rules("deny"), _rules("ask"), source=source
        )

    # An explicit config path is user-selected and therefore carries user provenance.
    if str(config_file) != _REPO_DEFAULT_CONFIG:
        return _load(config_file, PermissionRuleSource.USER)

    layered = RuleSet()
    try:
        user_file = Path.home() / ".polaris" / "agent.toml"
    except RuntimeError:
        user_file = Path("__no_user_permission_file__")
    layered = layered.merge(_load(user_file, PermissionRuleSource.USER))
    layered = layered.merge(_load(config_file, PermissionRuleSource.PROJECT))
    layered = layered.merge(_load("agent.local.toml", PermissionRuleSource.LOCAL))
    return layered


def resolve_mcp_config(config_file: str | Path = "agent.toml") -> "MCPConfig":
    """Resolve the MCP servers from the ``[mcp]`` toml table (``[mcp.servers.<name>]``).

    Returns an empty config (no servers) when the table is absent, so MCP stays off and
    the ``mcp`` SDK is never imported unless a server is actually configured.
    """
    from agent_core.mcp.config import MCPConfig

    table = load_agent_toml(config_file).get("mcp")
    return MCPConfig.from_dict(table if isinstance(table, dict) else None)
