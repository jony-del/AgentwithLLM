"""Configuration contracts for the industrial tool lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any

from agent_core.tools.base import ExecutionSafety, LockMode


@dataclass(frozen=True, slots=True)
class ResourcePolicyConfig:
    namespace: str
    mode: LockMode
    key: str | None = None
    argument: str | None = None
    subtree: bool = False
    requires_success: bool = False


@dataclass(frozen=True, slots=True)
class ToolPolicyConfig:
    safety: ExecutionSafety
    exclusive: bool = True
    transaction_backend: str | None = None
    resources: tuple[ResourcePolicyConfig, ...] = ()
    idempotency_argument: str | None = None
    safely_cancellable: bool = False
    execution_timeout: float = 30.0


def _parse_tool_policies(raw: object) -> dict[str, ToolPolicyConfig]:
    if not isinstance(raw, dict):
        return {}
    policies: dict[str, ToolPolicyConfig] = {}
    for qualified_name, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            safety = ExecutionSafety(str(value.get("safety", "final_only")).casefold())
        except ValueError:
            continue
        resources: list[ResourcePolicyConfig] = []
        raw_resources = value.get("resources", [])
        if isinstance(raw_resources, list):
            for resource in raw_resources:
                if not isinstance(resource, dict):
                    continue
                namespace = str(resource.get("namespace", "")).strip()
                mode = str(resource.get("mode", "")).casefold()
                key = resource.get("key")
                argument = resource.get("argument")
                if not namespace or mode not in {"read", "write"}:
                    continue
                if (key is None) == (argument is None):
                    continue
                resources.append(
                    ResourcePolicyConfig(
                        namespace=namespace,
                        mode=mode,  # type: ignore[arg-type]
                        key=str(key) if key is not None else None,
                        argument=str(argument) if argument is not None else None,
                        subtree=bool(resource.get("subtree", False)),
                        requires_success=bool(resource.get("requires_success", False)),
                    )
                )
        policies[str(qualified_name)] = ToolPolicyConfig(
            safety=safety,
            exclusive=bool(value.get("exclusive", True)),
            transaction_backend=(
                str(value["transaction_backend"]) if value.get("transaction_backend") else None
            ),
            resources=tuple(resources),
            idempotency_argument=(
                str(value["idempotency_argument"]) if value.get("idempotency_argument") else None
            ),
            safely_cancellable=bool(value.get("safely_cancellable", False)),
            execution_timeout=max(0.1, float(value.get("execution_timeout", 30.0))),
        )
    return policies


@dataclass(slots=True)
class BashConfig:
    executable: str | None = None
    enabled: bool = True


@dataclass(slots=True)
class PowerShellConfig:
    executable: str | None = None
    enabled: bool = True


@dataclass(slots=True)
class ShellToolConfig:
    enabled: bool = True
    timeout: int = 30
    max_timeout: int = 600
    auto_background_seconds: float = 15.0
    max_tasks: int = 16
    preview_bytes: int = 256 * 1024
    log_bytes: int = 32 * 1024 * 1024
    shutdown_grace_seconds: float = 3.0
    bash: BashConfig = field(default_factory=BashConfig)
    powershell: PowerShellConfig = field(default_factory=PowerShellConfig)


@dataclass(slots=True)
class LSPServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    extensions: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    initialization_options: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    workspace_folder: str = ""
    transport: str = "stdio"
    startup_timeout: float = 15.0
    shutdown_timeout: float = 2.0
    restart_on_crash: bool = True
    max_restarts: int = 3
    plugin_root: str = ""
    diagnostics: bool = True
    timeout: float = 15.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LSPServerConfig":
        extensions = raw.get(
            "extensionToLanguage",
            raw.get("extensions", raw.get("extension_language", {})),
        )
        initialization = raw.get("initializationOptions", raw.get("initialization_options", {}))
        startup_ms = raw.get("startupTimeout")
        shutdown_ms = raw.get("shutdownTimeout")
        return cls(
            name=str(raw.get("name", "")).strip(),
            command=str(raw.get("command", "")).strip(),
            args=tuple(str(item) for item in raw.get("args", ())),
            extensions={str(key).lower(): str(value) for key, value in dict(extensions or {}).items()},
            env={str(key): str(value) for key, value in dict(raw.get("env", {})).items()},
            initialization_options=dict(initialization or {}),
            settings=dict(raw.get("settings", {}) or {}),
            workspace_folder=str(raw.get("workspaceFolder") or ""),
            transport=str(raw.get("transport") or "stdio").casefold(),
            startup_timeout=max(0.1, float(startup_ms) / 1000 if startup_ms is not None else 15.0),
            shutdown_timeout=max(0.1, float(shutdown_ms) / 1000 if shutdown_ms is not None else 2.0),
            restart_on_crash=raw.get("restartOnCrash", True) is not False,
            max_restarts=max(0, int(raw.get("maxRestarts", 3))),
            plugin_root=str(raw.get("plugin_root") or ""),
            diagnostics=raw.get("diagnostics", True) is not False,
            timeout=max(0.1, float(raw.get("timeout", 15.0))),
        )


@dataclass(slots=True)
class LSPToolConfig:
    autodetect: bool = False
    max_restarts: int = 3
    servers: list[LSPServerConfig] = field(default_factory=list)


@dataclass(slots=True)
class NotebookToolConfig:
    max_bytes: int = 16 * 1024 * 1024
    max_output_chars: int = 8_000


@dataclass(slots=True)
class WorktreeToolConfig:
    root: str = ".polaris/worktrees"
    stale_days: int = 30


@dataclass(slots=True)
class SchedulerToolConfig:
    enabled: bool = True
    max_jobs: int = 50
    max_prompt_chars: int = 16_000
    database: str = "~/.polaris/scheduler.sqlite3"
    max_delivery_attempts: int = 3
    retry_base_seconds: float = 30
    retry_max_seconds: float = 600
    delivery_lease_seconds: float = 1800

    def database_path(self) -> Path:
        return Path(self.database).expanduser()


@dataclass(slots=True)
class ToolSuiteConfig:
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)
    lsp: LSPToolConfig = field(default_factory=LSPToolConfig)
    notebook: NotebookToolConfig = field(default_factory=NotebookToolConfig)
    worktree: WorktreeToolConfig = field(default_factory=WorktreeToolConfig)
    scheduler: SchedulerToolConfig = field(default_factory=SchedulerToolConfig)
    execution_policies: dict[str, ToolPolicyConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ToolSuiteConfig":
        raw = raw or {}
        shell_raw = dict(raw.get("shell", {}) or {})
        bash_raw = dict(shell_raw.pop("bash", {}) or {})
        ps_raw = dict(shell_raw.pop("powershell", {}) or {})
        shell = ShellToolConfig()
        for key, value in shell_raw.items():
            if hasattr(shell, key):
                setattr(shell, key, value)
        shell.timeout = max(1, int(shell.timeout))
        shell.max_timeout = max(shell.timeout, int(shell.max_timeout))
        shell.auto_background_seconds = max(0.0, float(shell.auto_background_seconds))
        shell.max_tasks = max(1, int(shell.max_tasks))
        shell.preview_bytes = max(1024, int(shell.preview_bytes))
        shell.log_bytes = max(shell.preview_bytes, int(shell.log_bytes))
        shell.shutdown_grace_seconds = max(0.1, float(shell.shutdown_grace_seconds))
        shell.bash = BashConfig(
            executable=str(bash_raw["executable"]) if bash_raw.get("executable") else None,
            enabled=bool(bash_raw.get("enabled", True)),
        )
        shell.powershell = PowerShellConfig(
            executable=str(ps_raw["executable"]) if ps_raw.get("executable") else None,
            enabled=bool(ps_raw.get("enabled", True)),
        )

        lsp_raw = dict(raw.get("lsp", {}) or {})
        servers = [
            LSPServerConfig.from_dict(item)
            for item in lsp_raw.get("servers", [])
            if isinstance(item, dict) and item.get("name") and item.get("command")
        ]
        autodetect = bool(lsp_raw.get("autodetect", False))
        if autodetect:
            known = (
                ("pyright", "pyright-langserver", ("--stdio",), {".py": "python"}),
                ("typescript", "typescript-language-server", ("--stdio",),
                 {".ts": "typescript", ".tsx": "typescriptreact", ".js": "javascript"}),
                ("rust-analyzer", "rust-analyzer", (), {".rs": "rust"}),
                ("gopls", "gopls", (), {".go": "go"}),
                ("clangd", "clangd", (), {".c": "c", ".cc": "cpp", ".cpp": "cpp"}),
            )
            configured_names = {server.name for server in servers}
            configured_commands = {server.command for server in servers}
            for name, command, args, extensions in known:
                executable = shutil.which(command)
                if executable and name not in configured_names and command not in configured_commands:
                    servers.append(LSPServerConfig(name, executable, args, extensions))
        lsp = LSPToolConfig(
            autodetect=autodetect,
            max_restarts=max(0, min(3, int(lsp_raw.get("max_restarts", 3)))),
            servers=servers,
        )
        notebook = NotebookToolConfig(**{
            key: value for key, value in dict(raw.get("notebook", {}) or {}).items()
            if key in {"max_bytes", "max_output_chars"}
        })
        worktree = WorktreeToolConfig(**{
            key: value for key, value in dict(raw.get("worktree", {}) or {}).items()
            if key in {"root", "stale_days"}
        })
        scheduler = SchedulerToolConfig(**{
            key: value for key, value in dict(raw.get("scheduler", {}) or {}).items()
            if key in {
                "enabled", "max_jobs", "max_prompt_chars", "database",
                "max_delivery_attempts", "retry_base_seconds", "retry_max_seconds",
                "delivery_lease_seconds",
            }
        })
        scheduler.max_delivery_attempts = max(1, int(scheduler.max_delivery_attempts))
        scheduler.retry_base_seconds = max(0.0, float(scheduler.retry_base_seconds))
        scheduler.retry_max_seconds = max(
            scheduler.retry_base_seconds, float(scheduler.retry_max_seconds)
        )
        scheduler.delivery_lease_seconds = max(1.0, float(scheduler.delivery_lease_seconds))
        policies = _parse_tool_policies(
            raw.get("execution_policies", raw.get("policies", {}))
        )
        return cls(
            shell=shell,
            lsp=lsp,
            notebook=notebook,
            worktree=worktree,
            scheduler=scheduler,
            execution_policies=policies,
        )
