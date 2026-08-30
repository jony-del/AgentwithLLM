"""Real OCI sandbox acceptance suite, enabled only on dedicated CI runners."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from agent_core.cli import _sandbox_mcp_config
from agent_core.lsp import LSPManager
from agent_core.mcp import MCPAdapter, MCPClientManager, MCPConfig, MCPServerConfig
from agent_core.sandbox import (
    SandboxConfig,
    SandboxInvocation,
    SandboxManager,
    SandboxUnavailableError,
)
from agent_core.tool_config import LSPServerConfig, LSPToolConfig
from agent_core.tools.base import ExecutionScope
from agent_core.workflow_runtime import WorkflowRuntime


IMAGE = os.getenv("POLARIS_SANDBOX_E2E_IMAGE", "")
RUNTIME = os.getenv("POLARIS_SANDBOX_E2E_RUNTIME", "")
ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not IMAGE or not RUNTIME, reason="real OCI sandbox E2E is release-runner only"
)


@pytest.fixture(scope="module")
def sandbox() -> SandboxManager:
    config = SandboxConfig.from_dict({
        "enabled": True,
        "backend": "container",
        "container": {"runtime": RUNTIME, "image": IMAGE, "auto_pull": True},
    })
    manager = SandboxManager(config, workspace=ROOT)
    manager.prepare()
    assert manager.requested and manager.prepared and manager.is_enabled()
    assert manager.backend_name == "container"
    assert manager.image == IMAGE and re.search(r"@sha256:[0-9a-f]{64}$", manager.image)
    yield manager
    manager.teardown()


def _run(
    sandbox: SandboxManager,
    guest_argv: list[str],
    capabilities: tuple[str, ...],
    *,
    host_argv: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    invocation = SandboxInvocation.create(
        host_argv or ["C:\\host\\must-not-run.exe"],
        guest_argv=guest_argv,
        required_guest_capabilities=capabilities,
        scope=ExecutionScope.for_workspace(ROOT, network="deny"),
    )
    argv, shell = sandbox.wrap_invocation(invocation)
    assert not shell and isinstance(argv, list)
    rendered = "\n".join(argv)
    assert not re.search(r"[A-Za-z]:[\\/]", rendered)
    assert not re.search(r"\.(?:exe|cmd|bat)(?:$|\s)", rendered, re.IGNORECASE)
    return subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120
    )


def test_real_oci_shell_unicode_pytest_node_and_security(sandbox: SandboxManager) -> None:
    bash = _run(sandbox, ["@bash", "-lc", "printf '沙箱-ok\\n'"], ("bash",))
    assert bash.returncode == 0 and bash.stdout == "沙箱-ok\n"

    pwsh = _run(
        sandbox,
        ["@pwsh", "-NoLogo", "-NoProfile", "-Command", "[Console]::Write('电源-ok')"],
        ("pwsh",),
    )
    assert pwsh.returncode == 0 and pwsh.stdout == "电源-ok"

    node = _run(sandbox, ["@node", "-e", "process.stdout.write(process.version)"], ("node",))
    assert node.returncode == 0 and node.stdout.startswith("v24.")

    temporary = Path(tempfile.mkdtemp(prefix="sandbox e2e 中文 ", dir=ROOT))
    try:
        test_file = temporary / "test_guest.py"
        test_file.write_text("def test_guest():\n    assert True\n", encoding="utf-8")
        relative = test_file.relative_to(ROOT).as_posix()
        tests = _run(
            sandbox, ["@python", "-m", "pytest", "-q", relative], ("python",)
        )
        assert tests.returncode == 0, tests.stdout + tests.stderr
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    with tempfile.NamedTemporaryFile(prefix="polaris-host-canary-", delete=False) as handle:
        host_canary = Path(handle.name)
    try:
        guest_canary = sandbox.translate_path(host_canary)
        security = _run(
            sandbox,
            [
                "@python", "-c",
                "import os,pathlib; assert os.geteuid()!=0; "
                f"assert not pathlib.Path({guest_canary!r}).exists()",
            ],
            ("python",),
        )
        assert security.returncode == 0, security.stderr
    finally:
        host_canary.unlink(missing_ok=True)


async def test_real_pyright_uses_guest_uris(sandbox: SandboxManager) -> None:
    temporary = Path(tempfile.mkdtemp(prefix="lsp 中文 ", dir=ROOT))
    source = temporary / "sample.py"
    source.write_text("value: int = 1\nprint(value)\n", encoding="utf-8")
    config = LSPToolConfig(servers=[LSPServerConfig(
        name="pyright",
        command="pyright-langserver",
        args=("--stdio",),
        extensions={".py": "python"},
        workspace_folder=temporary.relative_to(ROOT).as_posix(),
        startup_timeout=60.0,
        timeout=60.0,
    )])
    manager = LSPManager(config, ROOT, sandbox=sandbox)
    try:
        result = await manager.request(
            "definition",
            path=source.relative_to(ROOT).as_posix(),
            line=1,
            character=7,
        )
        rendered = str(result)
        assert result and source.resolve().as_uri() in rendered
        assert sandbox.translate_path(source) not in rendered
    finally:
        await manager.close()
        shutil.rmtree(temporary, ignore_errors=True)


async def test_real_node_workflow_uses_guest_runtime(sandbox: SandboxManager) -> None:
    async def unused_agent(*_args) -> str:
        raise AssertionError("workflow should not create an agent")

    result = await WorkflowRuntime().run(
        "return args.message;",
        {"message": "工作流-ok"},
        unused_agent,
        sandbox=sandbox,
        workspace=ROOT,
        require_sandbox=True,
        timeout=60.0,
    )
    assert result == "工作流-ok"


def test_real_guest_mcp_server(sandbox: SandboxManager) -> None:
    config = MCPConfig([MCPServerConfig(
        name="time",
        command="python",
        args=["-m", "mcp_server_time", "--local-timezone=UTC"],
        timeout=60.0,
    )])
    prepared = _sandbox_mcp_config(config, sandbox, ROOT)
    manager = MCPClientManager(prepared, connect_timeout=60.0)
    try:
        manager.start()
        tools = MCPAdapter(manager).list_tools()
        assert any("time" in tool.name.casefold() for tool in tools)
    finally:
        manager.close()


@pytest.mark.skipif(
    os.getenv("POLARIS_SANDBOX_E2E_EXPECT_STOPPED") != "1",
    reason="stopped-runtime canary is orchestrated only by the Windows release job",
)
def test_stopped_runtime_fails_closed_without_host_execution() -> None:
    config = SandboxConfig.from_dict({
        "enabled": True,
        "backend": "container",
        "container": {"runtime": RUNTIME, "image": IMAGE, "auto_pull": False},
    })
    manager = SandboxManager(config, workspace=ROOT)
    canary = Path(tempfile.gettempdir()) / "polaris-host-fallback-must-not-run"
    canary.unlink(missing_ok=True)
    invocation = SandboxInvocation.create(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(canary)!r}).touch()"],
        guest_argv=["@bash", "-lc", "exit 0"],
        required_guest_capabilities=("bash",),
        scope=ExecutionScope.for_workspace(ROOT, network="deny"),
    )
    try:
        with pytest.raises(SandboxUnavailableError):
            manager.prepare()
        with pytest.raises(SandboxUnavailableError):
            manager.wrap_invocation(invocation)
        assert not canary.exists()
    finally:
        canary.unlink(missing_ok=True)
