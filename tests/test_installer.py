from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from installer.install import (
    EXIT_UNSUPPORTED,
    Check,
    InstallError,
    Installer,
    Options,
    RestartRequired,
    Runner,
    StateStore,
    UnsupportedHost,
    checksum_for,
    decode_process_output,
    format_windows_exit_code,
    merge_path_values,
    node_archive_name,
    normalize_machine,
    parse_node_major,
    verify_sha256,
)
from agent_core.sandbox.config import SandboxContainerConfig


class FakeRunner(Runner):
    def __init__(self, present=(), *, outcomes=None, versions=None, dry_run=False) -> None:
        super().__init__(dry_run=dry_run)
        self.present = set(present)
        self.outcomes = outcomes or {}
        self.versions = versions or {}
        self.calls: list[list[str]] = []

    def which(self, command: str) -> str | None:
        return f"/bin/{command}" if command in self.present else None

    def run(self, argv, **kwargs):
        rendered = [str(item) for item in argv]
        self.calls.append(rendered)
        key = tuple(rendered)
        outcome = self.outcomes.get(key, 0)
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, tuple):
            returncode, stdout, stderr = outcome
        else:
            returncode, stdout, stderr = outcome, "", ""
        command = Path(rendered[0]).name
        if rendered[-1:] == ["--version"]:
            stdout = self.versions.get(command, f"{command} 1.0")
        if rendered[-4:] == ["machine", "list", "--format", "json"]:
            stdout = "[]"
        return subprocess.CompletedProcess(rendered, returncode, stdout, stderr)


def _options(tmp_path, **overrides) -> Options:
    values = {"source": tmp_path, "skip_sandbox": False}
    values.update(overrides)
    return Options(**values)


def _winget_rg_command(*, force: bool = False) -> list[str]:
    command = [
        "winget",
        "install",
        "--id",
        "BurntSushi.ripgrep.MSVC",
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    if force:
        command.append("--force")
    return command


def _winget_rg_list_command() -> list[str]:
    return [
        "winget",
        "list",
        "--id",
        "BurntSushi.ripgrep.MSVC",
        "--exact",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]


def test_node_platform_mapping_and_version_parsing() -> None:
    assert normalize_machine("AMD64") == "x86_64"
    assert normalize_machine("aarch64") == "arm64"
    assert parse_node_major("v24.4.1") == 24
    assert node_archive_name("v24.4.1", "windows", "x86_64").endswith("win-x64.zip")
    assert node_archive_name("v24.4.1", "darwin", "arm64").endswith("darwin-arm64.tar.gz")
    assert node_archive_name("v24.4.1", "linux", "x86_64").endswith("linux-x64.tar.xz")


def test_release_manifest_matches_public_bootstraps_and_sandbox_default() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "installer/manifest.json").read_text(encoding="utf-8"))
    assert manifest["uv_version"] in (root / "install.ps1").read_text(encoding="utf-8")
    assert manifest["uv_version"] in (root / "install.sh").read_text(encoding="utf-8")
    assert manifest["sandbox_image"] == SandboxContainerConfig().image


def test_checksum_verification(tmp_path) -> None:
    archive = tmp_path / "asset.zip"
    archive.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert checksum_for("asset.zip", f"{digest}  *asset.zip\n") == digest
    verify_sha256(archive, digest)
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        verify_sha256(archive, "0" * 64)


def test_state_store_records_only_install_metadata(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = StateStore(path)
    state.mark("host:git", component="git", source="apt")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == 2
    assert loaded["components"]["git"]["installed_by_polaris"] is True
    assert StateStore(path).completed("host:git")
    assert StateStore(path).component_owned("git")


def test_schema_one_ownership_is_migrated_as_unverified(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "completed": {"project:tool": 1},
                "components": {
                    "polaris": {
                        "installed_by_polaris": True,
                        "source": "legacy",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    state = StateStore(path)

    assert state.component("polaris")["legacy_unverified"] is True
    assert not state.component_owned("polaris")


def test_check_mode_is_non_mutating_when_runtime_is_ready(tmp_path) -> None:
    runner = FakeRunner(
        {"git", "rg", "node", "npm", "npx", "podman", "polaris"},
        versions={"node": "v24.4.1"},
    )
    installer = Installer(
        _options(tmp_path, check=True),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )
    checks = installer.install()
    assert all(isinstance(item, Check) for item in checks)
    assert all(item.ok for item in checks)
    assert not (tmp_path / "state.json").exists()
    assert not any("install" in call for call in runner.calls)


def test_uv_tool_install_records_exact_uninstall_receipt(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    uv = tmp_path / ("uv.exe" if os.name == "nt" else "uv")
    uv.write_text("uv", encoding="utf-8")
    tool_root = tmp_path / "uv-tools"
    bin_dir = tmp_path / "uv-bin"
    executable = bin_dir / ("polaris.exe" if os.name == "nt" else "polaris")

    class ProjectRunner(FakeRunner):
        def which(self, command: str) -> str | None:
            if command == "uv":
                return str(uv)
            return None

        def run(self, argv, **kwargs):
            rendered = [str(item) for item in argv]
            self.calls.append(rendered)
            if rendered[1:] == ["tool", "dir"]:
                return subprocess.CompletedProcess(rendered, 0, str(tool_root), "")
            if rendered[1:] == ["tool", "dir", "--bin"]:
                return subprocess.CompletedProcess(rendered, 0, str(bin_dir), "")
            if rendered[1:3] == ["tool", "install"]:
                (tool_root / "agent-with-llm").mkdir(parents=True)
                bin_dir.mkdir()
                executable.write_text("launcher", encoding="utf-8")
            return subprocess.CompletedProcess(rendered, 0, "", "")

    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path, skip_sandbox=True),
        runner=ProjectRunner(),
        state=state,
        system="windows" if os.name == "nt" else "linux",
        machine="x86_64",
    )

    installer._install_project()

    receipt = state.component("polaris")
    assert receipt["installed_by_polaris"] is True
    assert receipt["install_kind"] == "uv-tool"
    assert receipt["package"] == "agent-with-llm"
    assert Path(receipt["executable"]) == executable.resolve()
    assert Path(receipt["environment"]) == (tool_root / "agent-with-llm").resolve()
    assert Path(receipt["bootstrap_python"]) == Path(sys.executable).resolve()


def test_reused_polaris_is_recorded_as_external_not_owned(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    runner = FakeRunner({"polaris", "uv"})
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path, skip_sandbox=True),
        runner=runner,
        state=state,
        system="windows" if os.name == "nt" else "linux",
        machine="x86_64",
    )

    installer._install_project()

    receipt = state.component("polaris")
    assert receipt["install_kind"] == "external"
    assert receipt["installed_by_polaris"] is False
    assert not state.component_owned("polaris")


def test_runner_dry_run_executes_probes_but_skips_mutations(monkeypatch) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ready", "")

    monkeypatch.setattr("installer.install.subprocess.run", fake_run)
    runner = Runner(dry_run=True)
    probe = runner.run(["tool", "info"], capture=True)
    mutation = runner.run(["tool", "install"], mutates=True)
    assert probe.stdout == "ready"
    assert mutation.stdout == ""
    assert calls == [["tool", "info"]]


def test_runner_decodes_wsl_utf16_output_without_locale_errors(monkeypatch) -> None:
    message = (
        "未安装适用于 Linux 的 Windows 子系统。"
        "可通过运行 ‘wsl.exe --install’ 进行安装。"
    )

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 50, b"", message.encode("utf-16le"))

    monkeypatch.setattr("installer.install.subprocess.run", fake_run)

    proc = Runner().run(["wsl", "--status"], capture=True)

    assert proc.returncode == 50
    assert proc.stderr == message
    assert decode_process_output(b"\x90broken")


def test_winget_nonzero_is_accepted_when_path_refresh_finds_command(
    tmp_path, monkeypatch, capsys
) -> None:
    initial = _winget_rg_command()
    runner = FakeRunner(
        {"winget"},
        outcomes={tuple(initial): (2316632107, "", "没有可用的升级。")},
    )
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )
    monkeypatch.setattr(
        "installer.install.refresh_windows_path", lambda: runner.present.add("rg")
    )

    installer._install_packages(["rg"])

    assert _winget_rg_command(force=True) not in runner.calls
    assert "0x8A15002B" in capsys.readouterr().out


def test_winget_does_nothing_when_host_commands_are_already_available(tmp_path) -> None:
    runner = FakeRunner({"winget", "git", "rg"})
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )

    installer._install_host_commands()

    assert runner.calls == []
    assert not (tmp_path / "state.json").exists()


def test_winget_repairs_stale_package_record_with_one_forced_install(
    tmp_path, monkeypatch
) -> None:
    initial = _winget_rg_command()
    forced = _winget_rg_command(force=True)
    runner = FakeRunner(
        {"winget"},
        outcomes={tuple(initial): 2316632107, tuple(forced): 0},
    )
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )
    refreshes = 0

    def refresh_path() -> None:
        nonlocal refreshes
        refreshes += 1
        if refreshes == 2:
            runner.present.add("rg")

    monkeypatch.setattr("installer.install.refresh_windows_path", refresh_path)

    installer._install_packages(["rg"])

    assert runner.calls == [_winget_rg_list_command(), initial, forced]


def test_winget_preexisting_package_repaired_for_path_is_not_claimed(
    tmp_path, monkeypatch
) -> None:
    initial = _winget_rg_command()
    runner = FakeRunner(
        {"winget", "git"},
        outcomes={tuple(_winget_rg_list_command()): 0, tuple(initial): 2316632107},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    monkeypatch.setattr(
        "installer.install.refresh_windows_path", lambda: runner.present.add("rg")
    )

    installer._install_host_commands()

    assert not state.component_owned("rg")
    assert state.completed("host:rg")


def test_winget_failed_repair_is_actionable_and_does_not_mark_state(
    tmp_path, monkeypatch
) -> None:
    initial = _winget_rg_command()
    forced = _winget_rg_command(force=True)
    runner = FakeRunner(
        {"winget", "git"},
        outcomes={tuple(initial): 2316632107, tuple(forced): 5},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    monkeypatch.setattr("installer.install.refresh_windows_path", lambda: None)

    with pytest.raises(InstallError, match="winget uninstall --id BurntSushi.ripgrep.MSVC"):
        installer._install_host_commands()

    assert not state.component_owned("rg")
    assert not (tmp_path / "state.json").exists()


def test_path_merge_preserves_session_entries_and_deduplicates() -> None:
    separator = os.pathsep
    merged = merge_path_values(
        separator.join([r"C:\session", r"C:\shared"]),
        separator.join([r"C:\system", r"C:\shared"]),
        r"C:\user",
    )
    assert merged.split(separator) == [
        r"C:\session",
        r"C:\shared",
        r"C:\system",
        r"C:\user",
    ]


def test_windows_exit_code_format_accepts_signed_and_unsigned_values() -> None:
    assert "0x8A15002B" in format_windows_exit_code(2316632107)
    assert "0x8A15002B" in format_windows_exit_code(-1978335189)


def test_runtime_detection_skips_broken_podman(tmp_path) -> None:
    outcomes = {("podman", "info"): 1, ("docker", "info"): 0}
    runner = FakeRunner({"podman", "docker"}, outcomes=outcomes)
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="linux",
        machine="x86_64",
    )
    assert installer._select_usable_runtime() == "docker"


def test_windows_wsl_enable_requires_resumable_restart(tmp_path) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={("wsl", "--status"): 50, ("wsl", "--version"): 1},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    with pytest.raises(RestartRequired):
        installer._ensure_windows_wsl()
    assert state.completed("windows:wsl-enabled")
    elevated = [call for call in runner.calls if call[0] == "powershell.exe"]
    assert len(elevated) == 1
    assert "--install" in elevated[0][-1]
    assert "--update" not in elevated[0][-1]


def test_windows_wsl_updates_only_when_an_installation_is_present(tmp_path) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={("wsl", "--status"): 1, ("wsl", "--version"): 0},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )

    with pytest.raises(RestartRequired):
        installer._ensure_windows_wsl()

    assert state.completed("windows:wsl-updated")
    elevated = [call for call in runner.calls if call[0] == "powershell.exe"]
    assert len(elevated) == 1
    assert "--update" in elevated[0][-1]
    assert "--install" not in elevated[0][-1]


def test_windows_elevated_runner_captures_utf16_streams_and_cleans_temp_files(
    tmp_path, monkeypatch
) -> None:
    stdout_message = "WSL 功能已启用。"
    stderr_message = "需要重新启动 Windows。"
    capture_dirs: list[Path] = []
    real_temporary_directory = tempfile.TemporaryDirectory

    def tracked_temporary_directory(*args, **kwargs):
        kwargs["dir"] = tmp_path
        temporary = real_temporary_directory(*args, **kwargs)
        capture_dirs.append(Path(temporary.name))
        return temporary

    class CaptureRunner(FakeRunner):
        def run(self, argv, **kwargs):
            rendered = [str(item) for item in argv]
            self.calls.append(rendered)
            capture_dir = capture_dirs[-1]
            (capture_dir / "stdout.bin").write_bytes(stdout_message.encode("utf-16le"))
            (capture_dir / "stderr.bin").write_bytes(stderr_message.encode("utf-16le"))
            return subprocess.CompletedProcess(rendered, 3010, b"", b"")

    monkeypatch.setattr(
        "installer.install.tempfile.TemporaryDirectory", tracked_temporary_directory
    )
    runner = CaptureRunner()
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )

    result = installer._run_windows_elevated(
        "wsl.exe", ["--install", "--no-distribution"]
    )

    assert result.args == ["wsl.exe", "--install", "--no-distribution"]
    assert result.returncode == 3010
    assert result.stdout == stdout_message
    assert result.stderr == stderr_message
    assert capture_dirs and all(not path.exists() for path in capture_dirs)
    assert "-Verb RunAs" in runner.calls[0][-1]
    assert "-EncodedCommand" in runner.calls[0][-1]
    assert ".InnerException" in runner.calls[0][-1]
    assert runner.calls[0][0] == "powershell.exe"


def test_windows_elevated_runner_maps_uac_hresult_to_1223(tmp_path) -> None:
    class CancelledRunner(FakeRunner):
        def run(self, argv, **kwargs):
            rendered = [str(item) for item in argv]
            return subprocess.CompletedProcess(
                rendered,
                -2147023673,
                "",
                "The operation was canceled by the user.",
            )

    installer = Installer(
        _options(tmp_path),
        runner=CancelledRunner(),
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )

    result = installer._run_windows_elevated(
        "wsl.exe", ["--install", "--no-distribution"]
    )

    assert result.returncode == 1223
    assert "canceled" in result.stderr


def test_windows_wsl_uac_cancel_does_not_try_inbox(tmp_path, monkeypatch) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={("wsl", "--status"): 50, ("wsl", "--version"): 1},
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    elevated_calls: list[list[str]] = []

    def cancel_uac(executable: str, arguments: list[str]):
        elevated_calls.append(arguments)
        return subprocess.CompletedProcess(
            [executable, *arguments], 1223, "", "用户取消了操作。"
        )

    monkeypatch.setattr(installer, "_run_windows_elevated", cancel_uac)

    with pytest.raises(InstallError, match="1223") as raised:
        installer._ensure_windows_wsl()

    assert elevated_calls == [["--install", "--no-distribution"]]
    assert "--inbox" not in str(raised.value)
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("returncode", [0, 1641, 3010])
def test_windows_wsl_accepted_install_codes_require_resumable_restart(
    tmp_path, monkeypatch, returncode
) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={
            ("wsl", "--status"): [50, 50],
            ("wsl", "--version"): 1,
        },
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    monkeypatch.setattr(
        installer,
        "_run_windows_elevated",
        lambda executable, arguments: subprocess.CompletedProcess(
            [executable, *arguments], returncode, "", ""
        ),
    )

    with pytest.raises(RestartRequired, match="restart Windows"):
        installer._ensure_windows_wsl()

    assert state.completed("windows:wsl-enabled")


@pytest.mark.parametrize("returncode", [0, 1])
def test_windows_wsl_install_is_accepted_when_reprobe_succeeds(
    tmp_path, monkeypatch, returncode
) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={
            ("wsl", "--status"): [50, 0],
            ("wsl", "--version"): 1,
        },
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    elevated_calls: list[list[str]] = []

    def install_with_warning(executable: str, arguments: list[str]):
        elevated_calls.append(arguments)
        return subprocess.CompletedProcess(
            [executable, *arguments], returncode, "", "A servicing warning occurred."
        )

    monkeypatch.setattr(installer, "_run_windows_elevated", install_with_warning)

    installer._ensure_windows_wsl()

    assert elevated_calls == [["--install", "--no-distribution"]]
    assert state.completed("windows:wsl-enabled")


def test_windows_wsl_falls_back_to_inbox_after_standard_failure(
    tmp_path, monkeypatch
) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={
            ("wsl", "--status"): [50, 50, 0],
            ("wsl", "--version"): 1,
        },
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    elevated_calls: list[list[str]] = []

    def elevated(executable: str, arguments: list[str]):
        elevated_calls.append(arguments)
        returncode = 1 if "--inbox" not in arguments else 0
        return subprocess.CompletedProcess(
            [executable, *arguments], returncode, "", "standard path failed"
        )

    monkeypatch.setattr(installer, "_run_windows_elevated", elevated)

    installer._ensure_windows_wsl()

    assert elevated_calls == [
        ["--install", "--no-distribution"],
        ["--install", "--no-distribution", "--inbox"],
    ]
    assert state.completed("windows:wsl-enabled")
    assert state.data["components"]["wsl2"]["source"] == "windows-inbox"


def test_windows_wsl_reports_both_failed_install_paths(tmp_path, monkeypatch) -> None:
    runner = FakeRunner(
        {"wsl"},
        outcomes={
            ("wsl", "--status"): [50, 50, 50],
            ("wsl", "--version"): 1,
        },
    )
    state = StateStore(tmp_path / "state.json")
    installer = Installer(
        _options(tmp_path),
        runner=runner,
        state=state,
        system="windows",
        machine="x86_64",
    )
    results = iter(
        (
            subprocess.CompletedProcess(
                [], 5, "标准输出诊断", "标准安装被 Windows 拒绝"
            ),
            subprocess.CompletedProcess(
                [], 87, "inbox 输出诊断", "inbox 参数错误"
            ),
        )
    )
    monkeypatch.setattr(
        installer, "_run_windows_elevated", lambda executable, arguments: next(results)
    )

    with pytest.raises(InstallError) as raised:
        installer._ensure_windows_wsl()

    message = str(raised.value)
    assert "wsl.exe --install --no-distribution" in message
    assert "wsl.exe --install --no-distribution --inbox" in message
    assert "5 (0x00000005)" in message
    assert "87 (0x00000057)" in message
    assert "标准安装被 Windows 拒绝" in message
    assert "inbox 参数错误" in message
    assert not (tmp_path / "state.json").exists()


def test_windows_wsl_dry_run_skips_uac_and_persistent_state(
    tmp_path, monkeypatch, capsys
) -> None:
    subprocess_calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        rendered = [str(item) for item in argv]
        subprocess_calls.append(rendered)
        returncode = 50 if rendered[-1] == "--status" else 1
        return subprocess.CompletedProcess(rendered, returncode, b"", b"")

    monkeypatch.setattr("installer.install.subprocess.run", fake_run)
    runner = Runner(dry_run=True)
    installer = Installer(
        _options(tmp_path, dry_run=True),
        runner=runner,
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="x86_64",
    )

    installer._ensure_windows_wsl()

    assert subprocess_calls == [["wsl", "--status"], ["wsl", "--version"]]
    assert not (tmp_path / "state.json").exists()
    output = capsys.readouterr().out
    assert "+ [administrator] wsl.exe --install --no-distribution" in output
    assert "--inbox" not in output
    assert "EncodedCommand" not in output


def test_windows_arm_is_an_explicit_unsupported_host(tmp_path) -> None:
    installer = Installer(
        _options(tmp_path),
        runner=FakeRunner(),
        state=StateStore(tmp_path / "state.json"),
        system="windows",
        machine="arm64",
    )
    with pytest.raises(UnsupportedHost):
        installer._validate_host()
    assert EXIT_UNSUPPORTED == 30
