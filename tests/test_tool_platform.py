from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_core.notebook import edit_notebook, format_notebook, notebook_fingerprint
from agent_core.process_supervisor import ProcessSupervisor, ShellUnavailableError, resolve_bash_executable
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.scheduler import CronError, CronExpression, SchedulerStore
from agent_core import scheduler_service
from agent_core.tool_config import LSPServerConfig, LSPToolConfig, ShellToolConfig, WorktreeToolConfig
from agent_core.tools.base import Tool
from agent_core.tools.registry import ToolRegistry
from agent_core.worktree import WorktreeManager
from agent_core.lsp import LSPManager


class _Deferred(Tool):
    name = "deferred_demo"
    description = "diagnostic demo capability"
    input_schema = {"type": "object", "properties": {}}


def test_deferred_registry_activates_search_match() -> None:
    registry = ToolRegistry()
    registry.register_deferred(_Deferred.name, _Deferred.description, _Deferred)
    assert registry.list() == []
    result = registry.search("diagnostic")
    assert result == [{"name": "deferred_demo", "description": _Deferred.description, "activated": True}]
    assert registry.get("deferred_demo").name == "deferred_demo"


def test_deferred_registry_hides_unavailable() -> None:
    registry = ToolRegistry()
    registry.register_deferred(
        _Deferred.name, _Deferred.description, _Deferred,
        available=lambda: (False, "missing dependency"),
    )
    assert registry.search("diagnostic") == []


def _notebook() -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "code", "execution_count": 1, "metadata": {"keep": True},
                "outputs": [{"output_type": "display_data", "data": {"image/png": "abc", "text/plain": ["ok"]}}],
                "source": ["print('old')\n"],
            }
        ],
        "metadata": {"custom": {"preserve": True}}, "nbformat": 4, "nbformat_minor": 5,
    }


def test_notebook_structured_read_and_atomic_edit(tmp_path: Path) -> None:
    path = tmp_path / "demo.ipynb"
    path.write_text(json.dumps(_notebook()), encoding="utf-8")
    rendered, metadata = format_notebook(path)
    assert "binary image omitted" in rendered and "legacy-" in rendered
    cell_id = rendered.split("--- cell ", 1)[1].split(" ", 1)[0]
    result = edit_notebook(
        path, expected=notebook_fingerprint(path), cell_id=cell_id,
        new_source="print('new')\n", cell_type=None, edit_mode="replace",
    )
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["metadata"]["custom"]["preserve"] is True
    assert updated["cells"][0]["metadata"]["keep"] is True
    assert updated["cells"][0]["source"] == ["print('new')\n"]
    assert result["cell_id"] == cell_id
    assert metadata["cell_count"] == 1


def test_notebook_stale_read_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "demo.ipynb"
    path.write_text(json.dumps(_notebook()), encoding="utf-8")
    fingerprint = notebook_fingerprint(path)
    path.write_text(json.dumps({**_notebook(), "metadata": {"external": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed since"):
        edit_notebook(
            path, expected=fingerprint, cell_id="missing", new_source="", cell_type=None, edit_mode="delete"
        )


def test_cron_parser_and_store_coalescing(tmp_path: Path) -> None:
    expression = CronExpression.parse("*/5 * * * *")
    assert expression.minutes == set(range(0, 60, 5))
    store = SchedulerStore(tmp_path / "cron.sqlite3")
    now = 1_700_000_000.0
    job = store.create(
        owner_session="s", owner_agent="a", schedule="* * * * *", timezone="UTC",
        prompt="check status", persistent=False, now=now,
    )
    store.heartbeat("s", "a", ttl=120, now=now + 61)
    deliveries = store.route_due(now=float(job["next_run"]) + 1)
    assert len(deliveries) == 1
    store.route_due(now=float(job["next_run"]) + 61)
    assert store.get(str(job["id"]))["coalesced"] == 1


def test_cron_rejects_invalid_expression() -> None:
    with pytest.raises(CronError):
        CronExpression.parse("* * *")


def test_scheduler_missed_recurring_catches_up_once_and_one_shot_requires_resolution(
    tmp_path: Path,
) -> None:
    store = SchedulerStore(tmp_path / "cron.sqlite3")
    now = 1_700_000_000.0
    recurring = store.create(
        owner_session="s", owner_agent="a", schedule="* * * * *", timezone="UTC",
        prompt="recurring", persistent=True, now=now,
    )
    one_shot = store.create(
        owner_session="s", owner_agent="a", schedule="* * * * *", timezone="UTC",
        prompt="once", persistent=True, one_shot=True, now=now,
    )
    due = max(float(recurring["next_run"]), float(one_shot["next_run"])) + 1
    assert store.route_due(now=due) == []
    catchups = store.heartbeat("s", "a", now=due + 1)
    assert [item["job_id"] for item in catchups] == [recurring["id"]]
    missed = store.missed_one_shots("s", "a")
    assert [item["id"] for item in missed] == [one_shot["id"]]
    store.resolve_missed_one_shot(str(one_shot["id"]), deliver=True, now=due + 2)
    pending = store.pending("s", "a")
    once_delivery = next(item for item in pending if item["job_id"] == one_shot["id"])
    store.complete_delivery(int(once_delivery["id"]))
    with pytest.raises(CronError):
        store.get(str(one_shot["id"]))


@pytest.mark.skipif(not subprocess.run, reason="subprocess unavailable")
async def test_supervisor_powershell_output_and_stop(tmp_path: Path) -> None:
    from agent_core.process_supervisor import resolve_powershell_executable

    try:
        resolve_powershell_executable()
    except ShellUnavailableError:
        pytest.skip("PowerShell is not installed")
    supervisor = ProcessSupervisor(ShellToolConfig(preview_bytes=4096, log_bytes=8192), tmp_path / "logs")
    task = await supervisor.start("powershell", "Write-Output 'hello'", tmp_path, timeout=10)
    await supervisor.wait(task.id, 10)
    output = await supervisor.output(task.id, block=False, timeout=0, tail_lines=None)
    assert output["state"] == "completed" and "hello" in output["output"]

    restarted = ProcessSupervisor(
        ShellToolConfig(preview_bytes=4096, log_bytes=8192), tmp_path / "logs"
    )
    historical = await restarted.output(task.id, block=False, timeout=0, tail_lines=None)
    assert historical["historical"] is True and "hello" in historical["output"]
    await restarted.shutdown()

    long_task = await supervisor.start("powershell", "Start-Sleep -Seconds 20", tmp_path, timeout=30)
    await supervisor.stop(long_task.id)
    assert long_task.state == "stopped"
    await supervisor.shutdown()


def test_windowsapps_bash_is_not_trusted(monkeypatch, tmp_path: Path) -> None:
    if __import__("os").name != "nt":
        pytest.skip("Windows-specific trusted Bash policy")
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "missing"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    with pytest.raises(ShellUnavailableError, match="Git for Windows"):
        resolve_bash_executable()


def test_scheduler_windows_service_receipt_upgrade_and_exact_uninstall(
    monkeypatch, tmp_path: Path,
) -> None:
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    receipt = tmp_path / "service.json"
    database = tmp_path / "scheduler.sqlite3"
    calls: list[list[str]] = []
    monkeypatch.setattr(scheduler_service.sys, "platform", "win32")
    monkeypatch.setattr(scheduler_service, "_run", lambda argv: calls.append(argv))

    installed = scheduler_service.install_user_service(
        executable=executable, database=database, receipt_path=receipt
    )
    assert installed["service_id"] == scheduler_service.SERVICE_ID
    create = next(call for call in calls if call[:2] == ["schtasks", "/Create"])
    task_command = create[create.index("/TR") + 1]
    assert len(task_command) <= 261
    assert "-EncodedCommand" not in task_command
    assert str(executable.resolve()) in task_command
    assert str(database.resolve()) in task_command
    assert any(call[:2] == ["schtasks", "/Run"] for call in calls)

    calls.clear()
    scheduler_service.install_user_service(
        executable=executable, database=database, receipt_path=receipt
    )
    assert calls[0][:2] == ["schtasks", "/Delete"]
    with pytest.raises(RuntimeError, match="does not match"):
        scheduler_service.uninstall_user_service(
            expected_executable=tmp_path / "other.exe", receipt_path=receipt
        )
    scheduler_service.uninstall_user_service(
        expected_executable=executable, receipt_path=receipt
    )
    assert not receipt.exists()


async def test_session_worktree_create_remove_and_dirty_protection(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    registry = ToolRegistry()
    registry.register(_Deferred())
    sandbox = SimpleNamespace(workspace=repo)
    permission_roots: list[Path] = []
    session = SimpleNamespace(
        workspace=repo, process_supervisor=None, lsp_manager=None, logger=None,
        session_id="session123456", registry=registry,
        permission_workspace_setter=permission_roots.append,
    )
    manager = WorktreeManager(session, registry, sandbox, WorktreeToolConfig())
    state = await manager.create_and_enter("isolated")
    assert session.workspace == state.path and state.path.is_dir()
    assert permission_roots[-1] == state.path
    kept = await manager.exit("keep", discard_changes=False)
    assert kept["action"] == "keep" and session.workspace == repo
    resumed = await manager.create_and_enter("isolated")
    assert resumed is state and session.workspace == state.path
    (state.path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        await manager.exit("remove", discard_changes=False)
    details = await manager.exit("remove", discard_changes=True)
    assert details["dirty"] is True and session.workspace == repo and not state.path.exists()


async def test_subagent_worktree_isolated_and_clean_result_is_not_merged(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    agent = ReActAgent(
        FakeProvider(), ReActConfig(run_dir=str(tmp_path / "runs"), session_dir=""),
        workspace=repo,
    )
    try:
        answer = await agent._spawn_subagent(
            "inspect without changes", "read_only", isolation="worktree"
        )
        assert '\"retained\": false' in answer
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "base\n"
    finally:
        await agent.fire_session_end("test")


async def test_fake_lsp_initialize_hover_diagnostics_and_cleanup(tmp_path: Path) -> None:
    server_script = tmp_path / "fake_lsp.py"
    server_script.write_text(
        """
import json, sys
def send(value):
    body=json.dumps(value,separators=(',',':')).encode()
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode()+body);sys.stdout.buffer.flush()
while True:
    headers={}
    while True:
        line=sys.stdin.buffer.readline()
        if not line: raise SystemExit(0)
        if line in (b'\\r\\n',b'\\n'): break
        k,_,v=line.decode().partition(':');headers[k.lower().strip()]=v.strip()
    msg=json.loads(sys.stdin.buffer.read(int(headers['content-length'])))
    method=msg.get('method')
    if method=='initialize': send({'jsonrpc':'2.0','id':msg['id'],'result':{'capabilities':{}}})
    elif method=='textDocument/didOpen':
        uri=msg['params']['textDocument']['uri']
        send({'jsonrpc':'2.0','method':'textDocument/publishDiagnostics','params':{'uri':uri,'diagnostics':[{'message':'demo'}]}})
    elif method=='textDocument/hover': send({'jsonrpc':'2.0','id':msg['id'],'result':{'contents':'hovered'}})
    elif method=='shutdown': send({'jsonrpc':'2.0','id':msg['id'],'result':None})
    elif 'id' in msg: send({'jsonrpc':'2.0','id':msg['id'],'result':[]})
    if method=='exit': raise SystemExit(0)
""".lstrip(),
        encoding="utf-8",
    )
    source = tmp_path / "demo.py"
    source.write_text("value = 1\n", encoding="utf-8")
    config = LSPToolConfig(
        servers=[LSPServerConfig("fake", sys.executable, (str(server_script),), {".py": "python"}, timeout=3)]
    )
    manager = LSPManager(config, tmp_path)
    hover = await manager.request("hover", path="demo.py", line=0, character=1)
    assert hover == {"contents": "hovered"}
    await asyncio.sleep(0.05)
    diagnostics = await manager.request("diagnostics", path="demo.py")
    assert diagnostics == [{"message": "demo"}]
    await manager.close()
