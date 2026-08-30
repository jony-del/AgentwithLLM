"""Tests for the sandbox enforcement layer: pluggable tier backends, selection,
lifecycle, and fail-closed preparation.

Backends are organised by isolation tier (native/container/vm). Nothing here runs a real
sandbox — platform + runtime availability are mocked via ``sys.platform`` and
``shutil.which``, and lifecycle probes via ``subprocess.run``.
"""

import shutil
import sys

import pytest

from agent_core.sandbox import (
    GuestCapabilityUnavailable,
    SandboxConfig,
    SandboxManager,
    SandboxInvocation,
    SandboxPreparationState,
    SandboxTier,
    SandboxUnavailableError,
    get_shared_manager,
)
from agent_core.sandbox.config import SandboxContainerConfig
from agent_core.sandbox.invocation import GuestRuntimeManifest
from agent_core.sandbox.backends import (
    BubblewrapBackend,
    ContainerBackend,
    NativeBackend,
    SeatbeltBackend,
    VmBackend,
)
from agent_core.tools.base import ExecutionScope


_IMAGE = SandboxContainerConfig().image
_MANIFEST = GuestRuntimeManifest.from_dict(
    {
        "protocol_version": 1,
        "guest_os": "linux",
        "architecture": "amd64",
        "tools": {
            "bash": "/usr/bin/bash",
            "pwsh": "/usr/bin/pwsh",
            "python": "/usr/bin/python3",
            "node": "/usr/bin/node",
            "npm": "/usr/bin/npm",
            "npx": "/usr/bin/npx",
            "pyright-langserver": "/usr/bin/pyright-langserver",
        },
    }
)


def _invocation(*guest: str) -> SandboxInvocation:
    return SandboxInvocation.create(
        ["C:/host/tool.exe"],
        guest_argv=guest,
        required_guest_capabilities=(guest[0][1:],) if guest and guest[0].startswith("@") else (),
    )


def _only(*present):
    """A shutil.which replacement that reports only ``present`` binaries on PATH."""
    return lambda name: f"/usr/bin/{name}" if name in present else None


def _force_linux(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _force_macos(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _force_windows(monkeypatch, *present) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(shutil, "which", _only(*present))


def _native(**kwargs) -> SandboxConfig:
    return SandboxConfig.from_dict({"enabled": True, "backend": "native", **kwargs})


# -- config --------------------------------------------------------------------------


def test_config_from_dict_nested_tables() -> None:
    config = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "container",
            "excluded_commands": ["bazel:*"],
            "filesystem": {"deny_read": ["~/.ssh"]},
            "network": {"allowed_domains": ["api.example.com"]},
            "container": {"image": "alpine", "runtime": "docker", "memory": "512m"},
            "vm": {"provider": "hyperv", "reset_each_task": False},
        }
    )
    assert config.enabled is True
    assert config.backend == "container"
    assert config.excluded_commands == ["bazel:*"]
    assert config.filesystem.deny_read == ["~/.ssh"]
    assert config.network.allowed_domains == ["api.example.com"]
    assert config.container.image == "alpine"
    assert config.container.runtime == "docker"
    assert config.container.memory == "512m"
    assert config.vm.provider == "hyperv"
    assert config.vm.reset_each_task is False


def test_config_defaults() -> None:
    config = SandboxConfig()
    assert config.enabled is False
    assert config.backend == "auto"
    assert config.fail_if_unavailable is True
    assert config.allow_unsandboxed_commands is False
    assert "@sha256:" in config.container.image


def test_config_bad_backend_degrades_to_auto() -> None:
    assert SandboxConfig.from_dict({"backend": "nonsense"}).backend == "auto"
    assert SandboxConfig.from_dict({"backend": "VM"}).backend == "vm"  # case-normalised


def test_guest_manifest_and_invocation_fail_closed() -> None:
    with pytest.raises(GuestCapabilityUnavailable, match="protocol"):
        GuestRuntimeManifest.from_dict({"protocol_version": 99, "guest_os": "linux", "architecture": "amd64", "tools": {}})
    with pytest.raises(GuestCapabilityUnavailable, match="Windows executable"):
        GuestRuntimeManifest.from_dict({
            "protocol_version": 1, "guest_os": "linux", "architecture": "amd64",
            "tools": {"python": "C:/Python/python.exe"},
        })
    with pytest.raises(GuestCapabilityUnavailable, match="no Linux guest argv"):
        SandboxInvocation.create(["python.exe", "-V"]).guest_command(_MANIFEST)
    with pytest.raises(GuestCapabilityUnavailable, match="Windows-only"):
        SandboxInvocation.create(
            ["host"], guest_argv=["@python", "C:/repo/test.py"],
            required_guest_capabilities=("python",),
        ).guest_command(_MANIFEST)


def test_guest_invocation_resolves_only_manifest_aliases() -> None:
    invocation = SandboxInvocation.create(
        ["C:/Python/python.exe", "-m", "pytest"],
        guest_argv=["@python", "-m", "pytest"],
        required_guest_capabilities=("python",),
    )
    assert invocation.guest_command(_MANIFEST) == ["/usr/bin/python3", "-m", "pytest"]
    with pytest.raises(GuestCapabilityUnavailable, match="missing uvx"):
        SandboxInvocation.create(
            ["host"], guest_argv=["@uvx"], required_guest_capabilities=("uvx",)
        ).guest_command(_MANIFEST)


def test_container_rejects_mutable_image_before_any_command(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman")
    cfg = SandboxConfig.from_dict({
        "enabled": True, "backend": "container", "container": {"image": "alpine:latest"}
    })
    manager = SandboxManager(cfg, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError, match="@sha256"):
        manager.prepare()


# -- prepare idempotence + process-level sharing (§5.6) --------------------------------


def test_prepare_is_idempotent_and_teardown_rearms(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native())
    assert manager.preparation_state is SandboxPreparationState.SELECTED
    calls = {"n": 0}
    original = manager._backend.prepare

    def counting() -> None:
        calls["n"] += 1
        original()

    monkeypatch.setattr(manager._backend, "prepare", counting)
    manager.prepare()
    manager.prepare()
    manager.prepare()
    assert calls["n"] == 1  # heavyweight readying ran once

    manager.teardown()  # releases resources → the manager may be readied again
    manager.prepare()
    assert calls["n"] == 2


def test_get_shared_manager_reuses_one_instance_per_config(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    first = get_shared_manager(_native())
    second = get_shared_manager(_native())  # equal config + workspace → same manager
    assert first is second


def test_get_shared_manager_distinct_configs_get_distinct_managers(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    enabled = get_shared_manager(_native())
    disabled = get_shared_manager(SandboxConfig())
    assert enabled is not disabled


# -- native tier selection -----------------------------------------------------------


def test_native_selection_linux(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native())
    assert isinstance(manager._backend, NativeBackend)
    assert isinstance(manager._backend.strategy, BubblewrapBackend)
    assert manager.backend_tier is SandboxTier.NATIVE
    assert not manager.is_enabled()
    manager.prepare()
    assert manager.is_enabled()


def test_native_selection_macos(monkeypatch) -> None:
    _force_macos(monkeypatch, "sandbox-exec")
    manager = SandboxManager(_native())
    assert isinstance(manager._backend.strategy, SeatbeltBackend)


def test_native_on_windows_does_not_isolate(monkeypatch) -> None:
    _force_windows(monkeypatch)
    with pytest.raises(SandboxUnavailableError):
        SandboxManager(_native())


# -- auto selection prefers container -------------------------------------------------


def test_auto_prefers_container_when_runtime_present(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap", "podman")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert isinstance(manager._backend, ContainerBackend)
    assert manager.backend_tier is SandboxTier.CONTAINER


def test_auto_falls_back_to_native_without_runtime(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")  # no container runtime
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert isinstance(manager._backend, NativeBackend)


def test_auto_never_selects_vm(monkeypatch) -> None:
    # Everything present; auto must still not pick the heavyweight VM tier.
    _force_linux(monkeypatch, "bwrap", "podman", "kata-runtime")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "auto"}))
    assert manager.backend_tier is SandboxTier.CONTAINER


# -- downgrade chain -----------------------------------------------------------------


def test_explicit_vm_degrades_to_container(monkeypatch) -> None:
    # VM requested but no kata runtime; docker present → container wins.
    _force_linux(monkeypatch, "docker")
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "vm"}))
    assert isinstance(manager._backend, ContainerBackend)
    assert manager._backend.runtime == "docker"


def test_explicit_container_degrades_to_native(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")  # no runtime → native
    manager = SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "container"}))
    assert isinstance(manager._backend, NativeBackend)


def test_all_unavailable_fails_closed(monkeypatch) -> None:
    _force_linux(monkeypatch)  # nothing present
    with pytest.raises(SandboxUnavailableError, match="podman"):
        SandboxManager(SandboxConfig.from_dict({"enabled": True, "backend": "container"}))


# -- degradation / passthrough -------------------------------------------------------


def test_windows_enabled_without_runtime_fails_closed(monkeypatch) -> None:
    _force_windows(monkeypatch)
    with pytest.raises(SandboxUnavailableError):
        SandboxManager(SandboxConfig(enabled=True))


def test_disabled_manager_never_wraps(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap", "podman")
    manager = SandboxManager(SandboxConfig(enabled=False))
    assert not manager.is_enabled()
    assert manager.unavailable_reason() is None  # silent when the user didn't ask for it
    spec, _ = manager.wrap("ls", True, command="ls")
    assert spec == "ls"


# -- native wrapping (bwrap / seatbelt strategies) -----------------------------------


def test_bubblewrap_wrap_prefixes_command(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native(), workspace=tmp_path)
    manager.prepare()
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "bwrap"
    assert "--ro-bind" in spec
    assert "--unshare-net" in spec  # network default-deny
    assert str(tmp_path.resolve()) in spec
    assert spec[-3:] == ["/bin/sh", "-c", "echo hi"]


def test_bubblewrap_wrap_argv_spec(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "bwrap")
    manager = SandboxManager(_native(), workspace=tmp_path)
    manager.prepare()
    spec, shell = manager.wrap(["pytest", "-q"], False, command=None)
    assert shell is False
    assert spec[0] == "bwrap"
    assert spec[-2:] == ["pytest", "-q"]


def test_seatbelt_wrap_builds_profile(monkeypatch, tmp_path) -> None:
    _force_macos(monkeypatch, "sandbox-exec")
    config = _native()
    config.filesystem.deny_read = ["/etc/secret"]
    manager = SandboxManager(config, workspace=tmp_path)
    manager.prepare()
    spec, shell = manager.wrap("echo hi", True, command="echo hi")
    assert shell is False
    assert spec[0] == "sandbox-exec"
    assert spec[1] == "-p"
    profile = spec[2]
    assert "(deny file-write*)" in profile
    assert "secret" in profile
    assert "(deny network*)" in profile


# -- container wrapping --------------------------------------------------------------


def _container_manager(monkeypatch, tmp_path, *present, image_present=True, **container):
    _force_linux(monkeypatch, *present)
    # Image inspect / pull probes: succeed only when image_present.
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._run_probe",
        lambda argv: image_present or "pull" in argv,
    )
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._probe_manifest", lambda runtime, cfg: _MANIFEST
    )
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": _IMAGE, **container}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    manager.prepare()
    return manager


def test_container_wrap_hardening_flags(monkeypatch, tmp_path) -> None:
    manager = _container_manager(
        monkeypatch, tmp_path, "podman", memory="512m", cpus="1.5", pids_limit="128"
    )
    spec, shell = manager.wrap_invocation(_invocation("@bash", "-lc", "echo hi"), command="echo hi")
    assert shell is False
    assert spec[0] == "podman"
    assert spec[1] == "run"
    assert "--rm" in spec
    assert ["--network", "none"] == [spec[i] for i in (_pair(spec, "--network"))]
    assert "--read-only" in spec
    assert "--cap-drop" in spec and "ALL" in spec
    assert "no-new-privileges" in spec
    assert "512m" in spec and "1.5" in spec and "128" in spec
    assert _IMAGE in spec
    assert spec[-3:] == ["/usr/bin/bash", "-lc", "echo hi"]
    user = spec[spec.index("--user") + 1]
    assert user != "0:0"


def test_read_only_workspace_is_mounted_once(monkeypatch, tmp_path) -> None:
    manager = _container_manager(monkeypatch, tmp_path, "podman")
    invocation = SandboxInvocation.create(
        ["C:/host/tool.exe"],
        guest_argv=["@bash", "-lc", "true"],
        required_guest_capabilities=("bash",),
        scope=ExecutionScope.for_workspace(
            tmp_path, read_only_roots=(tmp_path,), workspace_writable=False
        ),
    )
    spec, _ = manager.wrap_invocation(invocation)
    mounts = [spec[index + 1] for index, item in enumerate(spec) if item == "-v"]
    assert mounts == [f"{tmp_path}:{tmp_path}:ro"]


def test_container_prefers_podman_over_docker(monkeypatch, tmp_path) -> None:
    manager = _container_manager(monkeypatch, tmp_path, "podman", "docker")
    assert manager._backend.runtime == "podman"


def test_container_prepare_falls_through_broken_runtime(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman", "docker")

    def fake_probe(argv):
        return argv[0] == "docker"

    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", fake_probe)
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._probe_manifest", lambda runtime, cfg: _MANIFEST
    )
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": _IMAGE}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    assert manager._backend.runtime == "podman"  # PATH preference before readiness
    manager.prepare()
    assert manager._backend.runtime == "docker"  # first actually usable runtime
    assert manager.is_enabled()


def test_container_network_opt_in_is_rejected(monkeypatch, tmp_path) -> None:
    manager = _container_manager(monkeypatch, tmp_path, "podman")
    manager.config.network.allowed_domains = ["api.example.com"]
    with pytest.raises(RuntimeError, match="network=deny"):
        manager.wrap_invocation(_invocation("@bash", "-lc", "curl x"), command="curl x")


def test_container_windows_path_mapping(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "docker")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: True)
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._probe_manifest", lambda runtime, cfg: _MANIFEST
    )
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": _IMAGE}}
    )
    manager = SandboxManager(cfg, workspace="E:/proj/app")
    manager.prepare()
    spec, _ = manager.wrap_invocation(_invocation("@bash", "-lc", "ls"), command="ls")
    # Both sides use the WSL-visible spelling accepted by the successful mount probe.
    mount = spec[spec.index("-v") + 1]
    assert mount == "/mnt/e/proj/app:/mnt/e/proj/app"
    assert spec[spec.index("-w") + 1] == "/mnt/e/proj/app"


def test_container_prepare_pulls_missing_image(monkeypatch, tmp_path) -> None:
    calls = []
    _force_linux(monkeypatch, "podman")

    def fake_probe(argv):
        calls.append(argv)
        # image inspect fails (missing); pull succeeds.
        if "image" in argv and "inspect" in argv:
            return False
        return True

    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", fake_probe)
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._probe_manifest", lambda runtime, cfg: _MANIFEST
    )
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": _IMAGE, "auto_pull": True}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    manager.prepare()
    assert any("pull" in c for c in calls)
    assert manager.is_enabled()  # still the container backend after a successful pull


def test_container_prepare_missing_image_no_autopull_fails_closed(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: False)
    cfg = SandboxConfig.from_dict(
        {"enabled": True, "backend": "container", "container": {"image": _IMAGE, "auto_pull": False}}
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError):
        manager.prepare()
    assert manager.preparation_state is SandboxPreparationState.UNAVAILABLE
    assert not manager.is_enabled()


def test_container_prepare_fail_if_unavailable_raises(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: False)
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "container",
            "fail_if_unavailable": True,
            "container": {"image": _IMAGE, "auto_pull": False},
        }
    )
    manager = SandboxManager(cfg, workspace=tmp_path)
    with pytest.raises(SandboxUnavailableError):
        manager.prepare()


# -- vm tier -------------------------------------------------------------------------


def test_vm_kata_selected_on_linux(monkeypatch, tmp_path) -> None:
    _force_linux(monkeypatch, "podman", "kata-runtime")
    monkeypatch.setattr("agent_core.sandbox.backends.container._run_probe", lambda argv: True)
    monkeypatch.setattr(
        "agent_core.sandbox.backends.container._probe_manifest", lambda runtime, cfg: _MANIFEST
    )
    cfg = SandboxConfig.from_dict({"enabled": True, "backend": "vm"})
    manager = SandboxManager(cfg, workspace=tmp_path)
    assert isinstance(manager._backend, VmBackend)
    assert manager._backend.strategy_name == "kata"
    manager.prepare()
    spec, shell = manager.wrap_invocation(_invocation("@bash", "-lc", "echo hi"), command="echo hi")
    assert shell is False
    # Kata reuses the container launcher with a --runtime override.
    assert spec[0] == "podman"
    assert "--runtime" in spec and "kata-runtime" in spec


def test_vm_hyperv_lifecycle_order(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "powershell", "ssh")
    calls = []
    monkeypatch.setattr(
        "agent_core.sandbox.backends.vm._run_vm_command",
        lambda argv: calls.append(argv) or True,
    )
    monkeypatch.setattr("agent_core.sandbox.backends.vm._remote_manifest", lambda prefix: _MANIFEST)
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "vm",
            "vm": {"provider": "hyperv", "guest_host": "sandbox-vm", "reset_each_task": True},
        }
    )
    manager = SandboxManager(cfg, workspace="E:/proj")
    assert manager._backend.strategy_name == "hyperv"
    manager.prepare()
    manager.reset()
    # wrap runs the command over SSH into the guest.
    spec, _ = manager.wrap_invocation(_invocation("@bash", "-lc", "echo hi"), command="echo hi")
    assert spec[0] == "ssh" and spec[1] == "sandbox-vm"
    manager.teardown()
    joined = " ".join(" ".join(c) for c in calls)
    assert "Checkpoint-VM" in joined  # prepare created a base snapshot
    assert "Restore-VMSnapshot" in joined  # reset rolled back


def test_vm_reset_noop_when_disabled(monkeypatch, tmp_path) -> None:
    _force_windows(monkeypatch, "powershell", "ssh")
    calls = []
    monkeypatch.setattr(
        "agent_core.sandbox.backends.vm._run_vm_command",
        lambda argv: calls.append(argv) or True,
    )
    cfg = SandboxConfig.from_dict(
        {
            "enabled": True,
            "backend": "vm",
            "vm": {"provider": "hyperv", "guest_host": "vm", "reset_each_task": False},
        }
    )
    manager = SandboxManager(cfg, workspace="E:/proj")
    manager.reset()
    assert not any("Restore-VMSnapshot" in " ".join(c) for c in calls)


# -- excluded commands ---------------------------------------------------------------


def test_excluded_command_is_rejected_when_sandbox_enabled(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    config = _native(excluded_commands=["bazel:*"])
    manager = SandboxManager(config)
    with pytest.raises(SandboxUnavailableError, match="excluded_commands"):
        manager.prepare()


# -- fail_if_unavailable (construction-time) -----------------------------------------


def test_fail_if_unavailable_raises_on_windows(monkeypatch) -> None:
    _force_windows(monkeypatch)
    config = SandboxConfig(enabled=True, fail_if_unavailable=True)
    with pytest.raises(SandboxUnavailableError):
        SandboxManager(config)


def test_fail_if_unavailable_ok_when_available(monkeypatch) -> None:
    _force_linux(monkeypatch, "bwrap")
    config = _native(fail_if_unavailable=True)
    manager = SandboxManager(config)
    manager.prepare()
    assert manager.is_enabled()


def test_fail_if_unavailable_ignored_when_disabled(monkeypatch) -> None:
    _force_windows(monkeypatch)
    SandboxManager(SandboxConfig(enabled=False, fail_if_unavailable=True))


def _pair(spec, flag):
    i = spec.index(flag)
    return (i, i + 1)
