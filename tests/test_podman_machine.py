"""Machine lifecycle tests never touch the host's Podman installation."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from agent_core.sandbox.backends import podman_machine as pm
from agent_core.sandbox.config import SandboxContainerConfig


class FakePodman:
    def __init__(self):
        self.calls = []
        self.now = 0.0
        self.ready = False
        self.stuck = False
        self.race = False
        self.start_code = 0
        self.machine = {
            "Name": "podman-machine-default", "VMType": "wsl",
            "Running": False, "Starting": False, "Port": 51780,
            "IdentityPath": "C:/Users/demo/.local/share/containers/machine",
        }
        self.machines = [self.machine]
        self.connection = {
            "Name": self.machine["Name"], "Default": True, "IsMachine": True,
            "URI": "ssh://user@127.0.0.1:51780/run/user/1000/podman/podman.sock",
            "Identity": self.machine["IdentityPath"],
        }
        self.connections = [self.connection]

    def run(self, argv, **kwargs):
        self.calls.append(argv[1:])
        assert kwargs["timeout"] <= 120.0 - self.now
        assert kwargs["stdin"] == subprocess.DEVNULL
        self.now += min(1.0, kwargs["timeout"])
        args = argv[1:]
        if args == ["info"]:
            return subprocess.CompletedProcess(argv, 0 if self.ready else 125, b"", b"socket refused")
        if args == ["machine", "list", "--format", "json"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.machines).encode(), b"")
        if args == ["system", "connection", "list", "--format", "json"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.connections).encode(), b"")
        if args == ["machine", "start", "podman-machine-default"]:
            if self.start_code == 0 or self.race:
                self.machine["Starting"] = True
                self.ready = not self.stuck
            return subprocess.CompletedProcess(argv, self.start_code, b"", b"WSL startup error")
        raise AssertionError(f"unexpected command: {argv}")

    def sleep(self, seconds):
        self.now += seconds
        if self.machine["Starting"] and not self.stuck:
            self.ready = True

    @property
    def starts(self):
        return [args for args in self.calls if args[:2] == ["machine", "start"]]


@pytest.fixture
def podman(monkeypatch):
    fake = FakePodman()
    monkeypatch.delenv("CONTAINER_HOST", raising=False)
    monkeypatch.delenv("CONTAINER_CONNECTION", raising=False)
    monkeypatch.setattr(pm.subprocess, "run", fake.run)
    monkeypatch.setattr(pm.time, "monotonic", lambda: fake.now)
    monkeypatch.setattr(pm.time, "sleep", fake.sleep)
    return fake


def ensure(*, auto_start=True, **kwargs):
    pm.ensure_podman_ready("podman", SandboxContainerConfig(auto_start_machine=auto_start, **kwargs))


def test_ready_runtime_needs_no_machine_commands(podman):
    podman.ready = True
    ensure()
    assert podman.calls == [["info"]]


def test_stopped_machine_starts_once_and_next_prepare_reuses_it(podman, capsys):
    ensure()
    ensure()
    assert podman.starts == [["machine", "start", "podman-machine-default"]]
    assert "Starting Podman Machine" in capsys.readouterr().err
    assert podman.calls[-1] == ["info"]


def test_disabled_autostart_reports_verified_stopped_state(podman):
    with pytest.raises(pm.PodmanUnavailable, match="is stopped.*podman machine start.*exit 125"):
        ensure(auto_start=False)
    assert not podman.starts


def test_already_starting_waits_without_second_start(podman):
    podman.machine["Starting"] = True
    ensure()
    assert not podman.starts


def test_running_but_unreachable_is_not_reported_as_stopped(podman):
    podman.machine["Running"] = True
    with pytest.raises(pm.PodmanUnavailable, match="is running.*socket refused"):
        ensure(auto_start=False)
    assert not podman.starts


def test_start_race_checks_state_instead_of_matching_stderr(podman):
    podman.start_code = 125
    podman.race = True
    ensure()
    assert len(podman.starts) == 1
    assert podman.calls.count(["machine", "list", "--format", "json"]) == 2


def test_start_failure_retains_exit_code_and_stderr(podman):
    podman.start_code = 125
    with pytest.raises(pm.PodmanUnavailable, match="failed to start.*exit 125: WSL startup error"):
        ensure()
    assert len(podman.starts) == 1


def test_start_and_poll_share_one_deadline(podman):
    podman.stuck = True
    with pytest.raises(pm.PodmanUnavailable, match="timed out.*socket refused"):
        ensure()
    assert podman.now == 120.0
    assert len(podman.starts) == 1


@pytest.mark.parametrize("change, message", [
    (lambda p: p.machines.clear(), "not initialized"),
    (lambda p: p.machine.update(VMType="hyperv"), "not a WSL2 machine"),
    (lambda p: p.machine.update(Starting="false"), "invalid running/starting state"),
    (lambda p: p.connection.update(Name="another-machine"), "does not match"),
    (lambda p: p.connection.update(URI="ssh://user@remote.example:51780/run/podman.sock"), "does not match"),
    (lambda p: p.connection.update(Identity="C:/other/key"), "does not match"),
    (lambda p: p.connection.update(IsMachine=False), "does not match"),
    (lambda p: p.connection.update(URI="ssh://user@127.0.0.1:invalid/"), "does not match"),
])
def test_unsafe_or_missing_target_never_starts(podman, change, message):
    change(podman)
    with pytest.raises(pm.PodmanUnavailable, match=message):
        ensure()
    assert not podman.starts


def test_connection_environment_override_is_respected(podman, monkeypatch):
    monkeypatch.setenv("CONTAINER_CONNECTION", "remote")
    with pytest.raises(pm.PodmanUnavailable, match="does not match"):
        ensure()
    assert not podman.starts


def test_host_environment_override_does_not_start_local_machine(podman, monkeypatch):
    monkeypatch.setenv("CONTAINER_HOST", "ssh://user@remote.example")
    with pytest.raises(pm.PodmanUnavailable, match="CONTAINER_HOST overrides"):
        ensure()
    assert not podman.starts


def test_selected_root_connection_can_match_machine(podman, monkeypatch):
    podman.connection.update(Name="podman-machine-default-root", Default=False)
    monkeypatch.setenv("CONTAINER_CONNECTION", "podman-machine-default-root")
    ensure()
    assert len(podman.starts) == 1


@pytest.mark.parametrize("name", ["", "--rootful", "name\nother", "a/b"])
def test_invalid_machine_name_is_rejected_before_discovery(podman, name):
    with pytest.raises(pm.PodmanUnavailable, match="machine_name is invalid"):
        ensure(podman_machine_name=name)
    assert podman.calls == [["info"]]


@pytest.mark.parametrize("error, message", [
    (subprocess.TimeoutExpired(["podman", "info"], 10, stderr=b"slow response"), "timed out.*slow response"),
    (OSError("permission denied"), "could not execute.*permission denied"),
])
def test_probe_exception_keeps_actual_diagnostic(podman, monkeypatch, error, message):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(pm.subprocess, "run", fail)
    with pytest.raises(pm.PodmanUnavailable, match=message):
        ensure()
    assert not podman.starts


def test_invalid_machine_json_does_not_trigger_initialization(podman, monkeypatch):
    original = podman.run
    def run(argv, **kwargs):
        if argv[1:3] == ["machine", "list"]:
            return subprocess.CompletedProcess(argv, 0, b"not-json", b"")
        return original(argv, **kwargs)
    monkeypatch.setattr(pm.subprocess, "run", run)
    with pytest.raises(pm.PodmanUnavailable, match="invalid JSON"):
        ensure()
    assert not podman.starts


def test_start_timeout_is_not_retried(podman, monkeypatch):
    original = podman.run
    def run(argv, **kwargs):
        if argv[1:3] == ["machine", "start"]:
            podman.calls.append(argv[1:])
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr=b"WSL did not respond")
        return original(argv, **kwargs)
    monkeypatch.setattr(pm.subprocess, "run", run)
    with pytest.raises(pm.PodmanUnavailable, match="machine start.*timed out.*WSL did not respond"):
        ensure()
    assert len(podman.starts) == 1


@pytest.fixture
def container_probes(podman, monkeypatch, tmp_path):
    from agent_core.sandbox.backends import container as cb
    from agent_core.sandbox.invocation import GuestRuntimeManifest, REQUIRED_GUEST_TOOLS

    monkeypatch.setattr(cb, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(cb, "_runtime_candidates", lambda requested: [requested])
    monkeypatch.setattr(cb, "_map_workspace_path", lambda path, isolation: str(path))
    monkeypatch.setattr(cb, "_run_probe", lambda argv: True)
    manifest = GuestRuntimeManifest.from_dict({
        "protocol_version": 1, "guest_os": "linux", "architecture": "amd64",
        "tools": {tool: f"/usr/bin/{tool}" for tool in REQUIRED_GUEST_TOOLS},
    })
    monkeypatch.setattr(cb, "_probe_manifest", lambda runtime, config: manifest)
    return cb


@pytest.mark.parametrize("probe, message", [
    ("_image_exists", "immutable image is not present"),
    ("_probe_mount", "workspace bind-mount probe failed"),
    ("_probe_security", "security canary failed"),
    ("_probe_network_denied", "network-deny canary failed"),
])
def test_machine_start_does_not_bypass_sandbox_probes(
    podman, container_probes, monkeypatch, tmp_path, probe, message,
):
    from agent_core.sandbox import SandboxConfig, SandboxManager, SandboxUnavailableError

    monkeypatch.setattr(container_probes, probe, lambda *args: False)
    config = SandboxConfig.from_dict({
        "enabled": True, "backend": "container",
        "container": {"runtime": "podman", "auto_start_machine": True},
    })
    manager = SandboxManager(config, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError, match=message):
        manager.prepare()
    assert len(podman.starts) == 1
    assert not manager.prepared and not manager.is_enabled()


def test_successful_start_prepares_manager_only_once(podman, container_probes, tmp_path):
    from agent_core.sandbox import SandboxConfig, SandboxManager

    config = SandboxConfig.from_dict({
        "enabled": True, "backend": "container",
        "container": {"runtime": "podman", "auto_start_machine": True},
    })
    manager = SandboxManager(config, workspace=tmp_path)
    manager.prepare()
    calls = list(podman.calls)
    manager.prepare()
    assert podman.calls == calls
    assert manager.prepared and manager.is_enabled()


@pytest.mark.parametrize("platform, runtime", [("linux", "podman"), ("darwin", "podman"), ("win32", "docker")])
def test_other_backends_do_not_start_podman(podman, container_probes, monkeypatch, tmp_path, platform, runtime):
    from agent_core.sandbox import SandboxConfig

    monkeypatch.setattr(container_probes, "sys", SimpleNamespace(platform=platform))
    config = SandboxConfig.from_dict({
        "enabled": True, "backend": "container",
        "container": {"runtime": runtime, "auto_start_machine": True},
    })
    backend = container_probes.ContainerBackend(config, workspace=tmp_path)
    backend.prepare()
    assert not podman.calls


def test_health_canary_never_starts_machine(podman, container_probes, monkeypatch, tmp_path):
    from agent_core import health
    import agent_core.process_supervisor as supervisor
    import agent_core.scheduler_service as scheduler

    monkeypatch.setattr(health, "_RUNTIME_DISTRIBUTIONS", ())
    monkeypatch.setattr(health, "_HOST_COMMANDS", ())
    monkeypatch.setattr(health, "_command_version", lambda executable: "version")
    monkeypatch.setattr(health, "_usable_container_runtime", lambda: "podman")
    monkeypatch.setattr(supervisor, "resolve_bash_executable", lambda _: "bash")
    monkeypatch.setattr(supervisor, "resolve_powershell_executable", lambda _: "pwsh")
    monkeypatch.setattr(scheduler, "default_receipt_path", lambda: tmp_path / "missing.json")
    checks = health.collect_dependency_checks()
    canary = next(check for check in checks if check.name == "sandbox-canary")
    assert canary.status == "error" and "is stopped" in canary.detail
    assert not podman.starts
