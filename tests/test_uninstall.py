from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import agent_core.uninstall as uninstall
from agent_core.uninstall import (
    EXIT_OK,
    OwnershipError,
    UninstallPlan,
    Uninstaller,
    UsageError,
    apply_plan,
    run_uninstall,
)


def _executable_name() -> str:
    return "polaris.exe" if os.name == "nt" else "polaris"


def _make_uv_install(tmp_path: Path, *, owned: bool = True) -> dict[str, Path]:
    tool_root = tmp_path / "tools"
    environment = tool_root / "agent-with-llm"
    bin_dir = tmp_path / "bin"
    executable = bin_dir / _executable_name()
    uv = tmp_path / ("uv.exe" if os.name == "nt" else "uv")
    python = tmp_path / ("python.exe" if os.name == "nt" else "python")
    state_path = tmp_path / "state" / "install-state.json"
    data_path = tmp_path / ".polaris"
    runtime_root = tmp_path / "runtime"
    environment.mkdir(parents=True)
    bin_dir.mkdir()
    executable.write_text("launcher", encoding="utf-8")
    uv.write_text("uv", encoding="utf-8")
    python.write_text("python", encoding="utf-8")
    data_path.mkdir()
    (data_path / "agent.toml").write_text("provider = 'fake'\n", encoding="utf-8")
    state_path.parent.mkdir()
    state = {
        "schema": 2,
        "completed": {"project:tool": 1, "host:git": 1},
        "components": {
            "polaris": {
                "installed_by_polaris": owned,
                "install_kind": "uv-tool" if owned else "external",
                "package": "agent-with-llm",
                "executable": str(executable),
                "environment": str(environment),
                "source": str(tmp_path / "source"),
                "uv": str(uv),
                "tool_root": str(tool_root),
                "bin_dir": str(bin_dir),
                "bootstrap_python": str(python),
            },
            "git": {
                "installed_by_polaris": True,
                "source": "winget",
            },
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return {
        "tool_root": tool_root,
        "environment": environment,
        "bin_dir": bin_dir,
        "executable": executable,
        "uv": uv,
        "python": python,
        "state": state_path,
        "data": data_path,
        "runtime": runtime_root,
    }


def _uninstaller(paths: dict[str, Path], **kwargs) -> Uninstaller:
    return Uninstaller(
        state_path=paths["state"],
        data_path=paths["data"],
        runtime_root=paths["runtime"],
        current_executable=kwargs.get("current_executable", paths["executable"]),
        runner=kwargs.get("runner", uninstall._run_process),
    )


def _fake_uv_uninstall(paths: dict[str, Path]):
    def run(argv, **kwargs):
        assert [str(item) for item in argv[1:]] == [
            "tool",
            "uninstall",
            "agent-with-llm",
        ]
        shutil.rmtree(paths["environment"])
        paths["executable"].unlink()
        return subprocess.CompletedProcess(argv, 0, b"removed", b"")

    return run


def test_uv_tool_uninstall_removes_private_environment_but_preserves_host_and_data(
    tmp_path, monkeypatch
) -> None:
    paths = _make_uv_install(tmp_path)
    plan = _uninstaller(paths).build_plan()
    monkeypatch.setattr(uninstall, "_run_process", _fake_uv_uninstall(paths))

    apply_plan(plan)

    assert not paths["environment"].exists()
    assert not paths["executable"].exists()
    assert paths["data"].is_dir()
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert "polaris" not in state["components"]
    assert state["components"]["git"]["installed_by_polaris"] is True
    assert "project:tool" not in state["completed"]
    assert "host:git" in state["completed"]


def test_purge_data_removes_only_user_level_data_and_state(tmp_path, monkeypatch) -> None:
    paths = _make_uv_install(tmp_path)
    project = tmp_path / "project"
    (project / ".polaris").mkdir(parents=True)
    (project / "runs").mkdir()
    plan = _uninstaller(paths).build_plan(purge_data=True)
    monkeypatch.setattr(uninstall, "_run_process", _fake_uv_uninstall(paths))
    monkeypatch.setattr(uninstall, "default_data_path", lambda: paths["data"])

    apply_plan(plan)

    assert not paths["data"].exists()
    assert not paths["state"].exists()
    assert (project / ".polaris").is_dir()
    assert (project / "runs").is_dir()


def test_external_conda_or_pip_install_is_refused_with_exact_guidance(tmp_path) -> None:
    paths = _make_uv_install(tmp_path, owned=False)

    with pytest.raises(OwnershipError, match="pip.*uninstall.*agent-with-llm"):
        _uninstaller(paths).build_plan()

    assert paths["environment"].exists()
    assert paths["executable"].exists()


def test_external_windows_environment_guidance_uses_its_own_python(tmp_path) -> None:
    environment = tmp_path / "conda-env"
    scripts = environment / "Scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "polaris.exe"
    executable.write_text("launcher", encoding="utf-8")
    python = environment / "python.exe"
    python.write_text("python", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema": 2,
                "completed": {},
                "components": {
                    "polaris": {
                        "installed_by_polaris": False,
                        "install_kind": "external",
                        "package": "agent-with-llm",
                        "executable": str(executable),
                        "environment": str(environment),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OwnershipError) as raised:
        Uninstaller(
            state_path=state,
            data_path=tmp_path / "data",
            runtime_root=tmp_path / "runtime",
            current_executable=executable,
        ).build_plan()

    assert str(python.resolve()) in str(raised.value)


def test_receipt_for_a_different_active_command_is_refused(tmp_path) -> None:
    paths = _make_uv_install(tmp_path)
    foreign = tmp_path / "conda" / _executable_name()
    foreign.parent.mkdir()
    foreign.write_text("foreign", encoding="utf-8")

    with pytest.raises(OwnershipError, match="not proven"):
        _uninstaller(paths, current_executable=foreign).build_plan()


def test_legacy_uv_receipt_is_accepted_only_after_uv_show_paths_confirmation(
    tmp_path, monkeypatch
) -> None:
    paths = _make_uv_install(tmp_path)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["schema"] = 1
    state["components"]["polaris"] = {
        "installed_by_polaris": True,
        "source": str(tmp_path / "source"),
    }
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(
        uninstall.shutil,
        "which",
        lambda command: str(paths["uv"]) if command == "uv" else None,
    )

    def runner(argv, **kwargs):
        args = [str(item) for item in argv]
        if args[1:] == ["tool", "list", "--show-paths"]:
            output = (
                "agent-with-llm v0.1.0\n"
                f"- polaris ({paths['executable'].resolve()})\n"
            )
        elif args[1:] == ["tool", "dir"]:
            output = str(paths["tool_root"])
        elif args[1:] == ["tool", "dir", "--bin"]:
            output = str(paths["bin_dir"])
        elif args[1:] == ["python", "find", "3.12"]:
            output = str(paths["python"])
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, output, "")

    plan = _uninstaller(paths, runner=runner).build_plan()

    assert plan.kind == "uv-tool"
    assert plan.legacy is True


def test_legacy_receipt_does_not_claim_a_shadowing_conda_command(
    tmp_path, monkeypatch
) -> None:
    paths = _make_uv_install(tmp_path)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["schema"] = 1
    state["components"]["polaris"] = {
        "installed_by_polaris": True,
        "source": str(tmp_path / "source"),
    }
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(uninstall.shutil, "which", lambda command: str(paths["uv"]))
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "not found")

    with pytest.raises(OwnershipError, match="not proven"):
        _uninstaller(paths, runner=runner).build_plan()


def test_installer_owned_dev_venv_uses_marker_and_preserves_source(tmp_path) -> None:
    source = tmp_path / "checkout"
    environment = source / ".venv"
    executable = environment / ("Scripts/polaris.exe" if os.name == "nt" else "bin/polaris")
    executable.parent.mkdir(parents=True)
    executable.write_text("launcher", encoding="utf-8")
    (environment / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    (environment / ".polaris-install.json").write_text(
        json.dumps({"package": "agent-with-llm", "source": str(source)}),
        encoding="utf-8",
    )
    (source / "keep.txt").write_text("source", encoding="utf-8")
    python = tmp_path / ("python.exe" if os.name == "nt" else "python")
    python.write_text("python", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "completed": {"project:dev": 1},
                "components": {
                    "polaris": {
                        "installed_by_polaris": True,
                        "install_kind": "dev-venv",
                        "package": "agent-with-llm",
                        "source": str(source),
                        "environment": str(environment),
                        "executable": str(executable),
                        "bootstrap_python": str(python),
                        "uv": "",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manager = Uninstaller(
        state_path=state_path,
        data_path=tmp_path / "data",
        runtime_root=tmp_path / "runtime",
        current_executable=executable,
    )

    apply_plan(manager.build_plan())

    assert not environment.exists()
    assert (source / "keep.txt").read_text(encoding="utf-8") == "source"


def test_tampered_dev_receipt_outside_source_is_rejected(tmp_path) -> None:
    paths = _make_uv_install(tmp_path)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    receipt = state["components"]["polaris"]
    receipt["install_kind"] = "dev-venv"
    receipt["source"] = str(tmp_path / "source")
    receipt["environment"] = str(tmp_path / "unrelated")
    paths["state"].write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(OwnershipError, match="source/.venv"):
        _uninstaller(paths).build_plan()


def test_redirected_dev_environment_is_never_followed(
    tmp_path: Path,
    directory_redirect: Callable[[Path, Path], str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "valuable-environment"
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    link = source / ".venv"
    directory_redirect(link, target)
    executable = link / ("Scripts/polaris.exe" if os.name == "nt" else "bin/polaris")
    python = tmp_path / ("python.exe" if os.name == "nt" else "python")
    python.write_text("python", encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "schema": 2,
                "completed": {},
                "components": {
                    "polaris": {
                        "installed_by_polaris": True,
                        "install_kind": "dev-venv",
                        "package": "agent-with-llm",
                        "source": str(source),
                        "environment": str(link),
                        "executable": str(executable),
                        "bootstrap_python": str(python),
                        "uv": "",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OwnershipError, match=r"symlinked development|source/\.venv"):
        Uninstaller(
            state_path=state,
            data_path=tmp_path / "data",
            runtime_root=tmp_path / "runtime",
        ).build_plan()

    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_private_node_receipt_is_removed_but_host_receipts_remain(
    tmp_path, monkeypatch
) -> None:
    paths = _make_uv_install(tmp_path)
    node_path = paths["runtime"] / "node-v24-test"
    node_bin = node_path if os.name == "nt" else node_path / "bin"
    node_bin.mkdir(parents=True)
    (node_bin / ("node.exe" if os.name == "nt" else "node")).write_text(
        "node", encoding="utf-8"
    )
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["components"]["node"] = {
        "installed_by_polaris": True,
        "install_kind": "managed-runtime",
        "path": str(node_path),
        "bin": str(node_bin),
        "links": [],
    }
    state["completed"]["runtime:node"] = 1
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    plan = _uninstaller(paths).build_plan()
    monkeypatch.setattr(uninstall, "_run_process", _fake_uv_uninstall(paths))
    monkeypatch.setattr(uninstall, "_remove_windows_user_path", lambda path: None)

    apply_plan(plan)

    assert not node_path.exists()
    saved = json.loads(paths["state"].read_text(encoding="utf-8"))
    assert "node" not in saved["components"]
    assert "git" in saved["components"]


def test_dry_run_and_missing_noninteractive_confirmation_do_not_mutate(
    tmp_path, capsys
) -> None:
    paths = _make_uv_install(tmp_path)
    before = paths["state"].read_bytes()

    assert run_uninstall(dry_run=True, state_path=paths["state"]) == EXIT_OK
    assert "[dry-run]" in capsys.readouterr().out
    assert paths["state"].read_bytes() == before
    assert paths["environment"].exists()

    with pytest.raises(UsageError, match="requires --yes"):
        run_uninstall(non_interactive=True, state_path=paths["state"])
    assert paths["state"].read_bytes() == before


def test_stage_worker_uses_external_python_and_unique_restricted_files(
    tmp_path, monkeypatch
) -> None:
    paths = _make_uv_install(tmp_path)
    plan = _uninstaller(paths).build_plan()
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(argv, **kwargs):
        calls.append(([str(item) for item in argv], kwargs))
        return object()

    monkeypatch.setattr(uninstall.subprocess, "Popen", fake_popen)
    log = uninstall._stage_worker(plan)
    try:
        assert calls
        command, kwargs = calls[0]
        assert Path(command[0]).resolve() == paths["python"].resolve()
        assert "--parent-pid" in command
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert (log.parent / "plan.json").is_file()
        assert (log.parent / "uninstall-worker.py").is_file()
        if os.name != "nt":
            assert (log.parent.stat().st_mode & 0o777) == 0o700
    finally:
        shutil.rmtree(log.parent, ignore_errors=True)


def test_worker_revalidates_and_writes_a_completion_log(tmp_path, monkeypatch) -> None:
    worker = tmp_path / "worker.py"
    plan_path = tmp_path / "plan.json"
    log = tmp_path / "uninstall.log"
    worker.write_text("worker", encoding="utf-8")
    plan_path.write_text(json.dumps({"kind": "none"}), encoding="utf-8")
    applied: list[UninstallPlan] = []
    monkeypatch.setattr(uninstall, "__file__", str(worker))
    monkeypatch.setattr(uninstall, "_parent_running", lambda pid: False)
    monkeypatch.setattr(uninstall, "apply_plan", applied.append)

    # Supply every required dataclass field while keeping this a no-op plan.
    empty = _uninstaller(_make_uv_install(tmp_path / "install"))._empty_plan(
        purge_data=False
    )
    plan_path.write_text(json.dumps(uninstall.asdict(empty)), encoding="utf-8")
    assert uninstall._worker_main(plan_path, 123, log) == EXIT_OK

    assert applied and applied[0].kind == "none"
    assert "successfully" in log.read_text(encoding="utf-8")
    assert not plan_path.exists()
    assert not worker.exists()


def test_cli_parser_dispatches_uninstall_flags(monkeypatch) -> None:
    from agent_core.cli import main

    received = []
    monkeypatch.setattr(
        uninstall,
        "uninstall_from_cli",
        lambda args: received.append((args.dry_run, args.purge_data, args.yes)) or 0,
    )

    assert main(["uninstall", "--dry-run", "--purge-data", "--yes"]) == 0
    assert received == [(True, True, True)]


def test_public_bootstraps_expose_recovery_uninstall_without_installing_uv() -> None:
    root = Path(__file__).resolve().parents[1]
    powershell = (root / "install.ps1").read_text(encoding="utf-8")
    shell = (root / "install.sh").read_text(encoding="utf-8")

    assert "[switch]$Uninstall" in powershell
    assert "agent_core\\uninstall.py" in powershell
    assert "$Uninstall -or $Check -or $DryRun" in powershell
    assert "--uninstall" in shell
    assert "agent_core/uninstall.py" in shell
    assert "if ((UNINSTALL))" in shell
