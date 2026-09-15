"""Local image selection must be immutable, offline and fail closed."""

import subprocess

import pytest

from agent_core.sandbox import SandboxConfig, SandboxInvocation, SandboxManager, SandboxUnavailableError
from agent_core.sandbox.backends import container
from agent_core.sandbox.config import SandboxContainerConfig, validate_container_image, validate_image_reference
from agent_core.tools.base import ExecutionScope

IMAGE = "sha256:" + "a" * 64


@pytest.mark.parametrize("runtime", ["podman", "podman.exe", "C:/Program Files/Podman/podman.exe"])
def test_full_local_id_requires_explicit_offline_podman(runtime):
    assert validate_container_image(SandboxContainerConfig(runtime=runtime, image=IMAGE)) == IMAGE


@pytest.mark.parametrize("runtime,auto_pull", [("auto", False), ("docker", False), ("nerdctl", False), ("podman", True)])
def test_local_id_rejects_other_runtimes_and_pulls(runtime, auto_pull):
    with pytest.raises(ValueError, match="explicit Podman and auto_pull=false"):
        validate_container_image(SandboxContainerConfig(runtime=runtime, image=IMAGE, auto_pull=auto_pull))


@pytest.mark.parametrize("image", ["latest", "polaris:dev", "a" * 64, "sha256:abc", "sha256:" + "g" * 64])
def test_tags_and_incomplete_ids_are_rejected(image):
    with pytest.raises(ValueError):
        validate_container_image(SandboxContainerConfig(runtime="podman", image=image))


def test_release_lock_validator_still_requires_registry_digest():
    with pytest.raises(ValueError):
        validate_image_reference(IMAGE)


@pytest.mark.parametrize("actual,code,expected", [(IMAGE, 0, True), ("a" * 64, 0, True), ("b" * 64, 0, False), ("a" * 12, 0, False), (IMAGE, 125, False)])
def test_local_inspect_compares_the_complete_id(monkeypatch, actual, code, expected):
    def capture(argv):
        assert argv == ["podman", "image", "inspect", "--format", "{{.Id}}", IMAGE]
        return subprocess.CompletedProcess(argv, code, (actual + "\n").encode(), b"")
    monkeypatch.setattr(container, "_run_capture", capture)
    assert container._image_exists("podman", IMAGE) is expected


def test_local_probes_and_tool_runs_never_pull():
    config = SandboxContainerConfig(runtime="podman", image=IMAGE)
    assert "--pull=never" in container._probe_run_prefix("podman", config)
    assert "--pull=never" in container._hardened_run_prefix("podman", config)
    assert "--interactive" in container._hardened_run_prefix("podman", config)


def test_missing_local_image_cannot_pull_or_execute_host(monkeypatch, tmp_path):
    monkeypatch.setattr(container.shutil, "which", lambda name: name if name == "podman" else None)
    monkeypatch.setattr(container, "ensure_podman_ready", lambda *_: None)
    calls = []
    def capture(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 125 if "inspect" in argv else 0, b"", b"missing")
    monkeypatch.setattr(container, "_run_capture", capture)
    config = SandboxConfig.from_dict({"enabled": True, "backend": "container", "container": {"runtime": "podman", "image": IMAGE}})
    manager = SandboxManager(config, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError, match="rebuild.*build_sandbox.py"):
        manager.prepare()
    invocation = SandboxInvocation.create(
        ["host-must-not-run"], guest_argv=["@bash", "-lc", "true"],
        required_guest_capabilities=("bash",), scope=ExecutionScope.for_workspace(tmp_path),
    )
    with pytest.raises(SandboxUnavailableError):
        manager.wrap_invocation(invocation)
    assert not manager.is_enabled()
    assert not any("pull" in argv or "run" in argv for argv in calls)


def test_invalid_autopull_is_rejected_before_runtime_access(monkeypatch, tmp_path):
    monkeypatch.setattr(container.shutil, "which", lambda name: name)
    def unexpected(*_):
        raise AssertionError("invalid local image policy must not contact Podman")
    monkeypatch.setattr(container, "ensure_podman_ready", unexpected)
    monkeypatch.setattr(container, "_run_capture", unexpected)
    config = SandboxConfig.from_dict({"enabled": True, "container": {"runtime": "podman", "image": IMAGE, "auto_pull": True}})
    with pytest.raises(SandboxUnavailableError, match="auto_pull=false"):
        SandboxManager(config, workspace=tmp_path).prepare()


@pytest.mark.parametrize("module,isolated", [("mcp_server_time", True), ("mcp_server_git", True), ("mcp_server_fetch", True), ("custom_server", False)])
def test_reference_servers_select_declared_isolated_python(monkeypatch, tmp_path, module, isolated):
    from agent_core.cli import _sandbox_mcp_config
    from agent_core.mcp import MCPConfig, MCPServerConfig
    from agent_core.plugins.sandbox import sandbox_runtime_environment

    monkeypatch.setenv("APPDATA", "C:/Users/test/AppData/Roaming")
    monkeypatch.setenv("CONTAINER_CONNECTION", "my-machine")
    monkeypatch.setenv("SECRET_API_KEY", "must-not-be-inherited")
    env = sandbox_runtime_environment()
    assert env["APPDATA"].endswith("Roaming") and env["CONTAINER_CONNECTION"] == "my-machine"
    assert "SECRET_API_KEY" not in env
    seen = []
    class Sandbox:
        config = SandboxConfig(enabled=True)
        capabilities = {"python", "mcp-python"}
        def wrap_invocation(self, invocation):
            seen.append(invocation)
            return ["podman", "run", "--interactive", "--network", "none", IMAGE], False
    server = MCPServerConfig(name="server", command="python", args=["-m", module])
    result = _sandbox_mcp_config(MCPConfig([server]), Sandbox(), tmp_path)
    expected = "mcp-python" if isolated else "python"
    assert seen[0].guest_argv[0] == f"@{expected}"
    assert seen[0].required_guest_capabilities == frozenset({expected})
    assert result.servers[0].env["APPDATA"] == env["APPDATA"]
    assert result.servers[0].command == "podman"
    assert server.command == "python" and not server.env
