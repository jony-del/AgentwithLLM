"""Configuration contracts for the industrial tool lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any


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
    timeout: float = 15.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LSPServerConfig":
        extensions = raw.get("extensions", raw.get("extension_language", {}))
        return cls(
            name=str(raw.get("name", "")).strip(),
            command=str(raw.get("command", "")).strip(),
            args=tuple(str(item) for item in raw.get("args", ())),
            extensions={str(key).lower(): str(value) for key, value in dict(extensions or {}).items()},
            env={str(key): str(value) for key, value in dict(raw.get("env", {})).items()},
            initialization_options=dict(raw.get("initialization_options", {}) or {}),
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

    def database_path(self) -> Path:
        return Path(self.database).expanduser()


@dataclass(slots=True)
class ToolSuiteConfig:
    shell: ShellToolConfig = field(default_factory=ShellToolConfig)
    lsp: LSPToolConfig = field(default_factory=LSPToolConfig)
    notebook: NotebookToolConfig = field(default_factory=NotebookToolConfig)
    worktree: WorktreeToolConfig = field(default_factory=WorktreeToolConfig)
    scheduler: SchedulerToolConfig = field(default_factory=SchedulerToolConfig)

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
            if key in {"enabled", "max_jobs", "max_prompt_chars", "database"}
        })
        return cls(shell=shell, lsp=lsp, notebook=notebook, worktree=worktree, scheduler=scheduler)
