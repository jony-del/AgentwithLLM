from __future__ import annotations

from pathlib import Path

from agent_core.managed_policy import (
    FileManagedPolicyProvider,
    ManagedPolicyDefinition,
    StaticManagedPolicyProvider,
)
from agent_core.models import ToolCall, ToolRisk
from agent_core.permission_classifier import AutoPermissionVerdict
from agent_core.permission_store import persist_allow_rule
from agent_core.permission_types import (
    DecisionSource,
    PermissionBehavior,
    PermissionContext,
    PermissionDestination,
    PermissionMode,
    PlanStateSnapshot,
    PermissionResponse,
    PermissionUpdate,
)
from agent_core.permissions import PermissionPolicy
from agent_core.session import PlanArtifactStore, PlanState, SessionContext
from agent_core.tools.base import Tool
from agent_core.tools.builtin import EchoTool, WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.plan_mode import ExitPlanTool
from agent_core.tools.registry import ToolRegistry


class GenericWriteTool(Tool):
    name = "generic_write"
    description = "test write"
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    risk = ToolRisk.WRITE

    def _invoke(self, arguments):
        from agent_core.models import ToolResult

        return ToolResult(self.name, "ok")


def test_reference_mode_aliases_are_canonicalized() -> None:
    assert PermissionMode("acceptEdits") is PermissionMode.ACCEPTEDITS
    assert PermissionMode("dontAsk") is PermissionMode.DONTASK
    assert PermissionMode("bypassPermissions") is PermissionMode.BYPASS
    assert PermissionMode("ACCEPTEDITS").value == "acceptedits"


async def test_managed_policy_file_loads_and_hot_reload_failure_denies(tmp_path: Path) -> None:
    path = tmp_path / "managed.toml"
    path.write_text(
        "[managed.permissions]\n"
        "deny = ['echo']\n"
        "forbidden_modes = ['bypassPermissions']\n"
        "disable_persistent_grants = true\n",
        encoding="utf-8",
    )
    policy = PermissionPolicy(
        workspace=tmp_path,
        managed_policy_provider=FileManagedPolicyProvider(path),
    )
    denied = await policy.evaluate(EchoTool(), ToolCall("echo", {"text": "x"}))
    assert denied.behavior is PermissionBehavior.DENY
    assert denied.decision_source is DecisionSource.MANAGED
    assert PermissionMode.BYPASS in policy.managed_policy.forbidden_modes
    assert policy.managed_policy.disable_persistent_grants

    path.write_text("[managed.permissions\n", encoding="utf-8")
    failed = await policy.evaluate(EchoTool(), ToolCall("echo", {"text": "x"}))
    assert failed.behavior is PermissionBehavior.DENY
    assert "reload failed" in failed.reason


def test_atomic_permission_persistence_is_idempotent_and_preserves_other_tables(tmp_path: Path) -> None:
    path = tmp_path / "agent.local.toml"
    path.write_text(
        "# keep me\n[permissions]\nallow = [\n    'echo',\n]\ndeny = ['run_command(rm *)']\n\n[web]\nblocked_domains = ['x.test']\n",
        encoding="utf-8",
    )
    for _ in range(2):
        assert (
            persist_allow_rule(
                "write_text_file(notes.txt)", PermissionDestination.LOCAL, tmp_path
            )
            == path
        )
    text = path.read_text(encoding="utf-8")
    assert text.count("write_text_file(notes.txt)") == 1
    assert "# keep me" in text
    assert "[web]" in text


def test_structured_prompt_can_persist_exact_local_grant(tmp_path: Path) -> None:
    def prompt(request):
        suggestion = request.suggestions[0]
        return PermissionResponse(
            True,
            (
                PermissionUpdate(
                    PermissionBehavior.ALLOW,
                    suggestion.rule,
                    PermissionDestination.LOCAL,
                ),
            ),
            "approved",
        )

    policy = PermissionPolicy(workspace=tmp_path, prompter=prompt)
    tool = WriteTextFileTool(tmp_path)
    call = ToolCall("write_text_file", {"path": "notes.txt", "content": "x"})
    decision = policy.confirm(policy.decide(tool, call), tool, call)
    assert decision.allowed
    assert "write_text_file(notes.txt)" in (tmp_path / "agent.local.toml").read_text(
        encoding="utf-8"
    )
    assert policy.decide(tool, call).allowed


def test_managed_only_rules_ignore_existing_session_and_project_allows(tmp_path: Path) -> None:
    policy = PermissionPolicy(
        prompter=lambda *args: "always",
        workspace=tmp_path,
        managed_policy_provider=StaticManagedPolicyProvider(
            ManagedPolicyDefinition(
                allow=("generic_write",),
                allow_managed_rules_only=True,
            )
        ),
    )
    tool = GenericWriteTool()
    call = ToolCall("generic_write", {})
    # Simulate a grant created before an administrator enabled managed-only rules.
    policy._session_allow.add("other_write")
    policy.add_session_rule("other_write")
    assert policy.decide(tool, call).allowed

    other = type(
        "OtherWrite",
        (GenericWriteTool,),
        {"name": "other_write"},
    )()
    blocked = policy.decide(other, ToolCall("other_write", {}))
    assert not blocked.allowed


async def test_auto_unavailable_falls_back_to_interactive_prompt(tmp_path: Path) -> None:
    class Unavailable:
        async def evaluate(self, *args, **kwargs):
            return AutoPermissionVerdict(
                False, "offline", unavailable=True, failure_kind="transport"
            )

    registry = ToolRegistry()
    registry.register(GenericWriteTool())
    prompted = []

    def prompt(*args):
        prompted.append(args)
        return "once"

    policy = PermissionPolicy(
        PermissionMode.AUTO,
        prompter=prompt,
        workspace=tmp_path,
        allow_unsandboxed_unattended=True,
    )
    result = await ToolExecutor(
        registry, policy, permission_classifier=Unavailable()
    ).execute_many([ToolCall("generic_write", {})])
    assert result[0].ok
    assert prompted


async def test_plan_exit_installs_only_valid_scoped_session_bundle(tmp_path: Path) -> None:
    store = PlanArtifactStore(tmp_path / "plans")
    path = store.path_for("session", "leader")
    store.write(path, "1. run tests")
    state = PlanState(True, PermissionMode.DEFAULT.value, path)
    grants: list[str] = []
    session = SessionContext(
        workspace=tmp_path,
        plan_state=state,
        plan_store=store,
        permission_mode_setter=lambda mode, **kwargs: mode,
        permission_grant_setter=grants.append,
        registered_tool_names=frozenset({"exit_plan", "run_command"}),
    )
    tool = ExitPlanTool(session)
    args = {
        "requested_permissions": [
            {"rule": "run_command(python -m pytest -q)", "reason": "verify implementation"}
        ]
    }
    context = PermissionContext(
        PermissionMode.PLAN,
        tmp_path,
        interactive=True,
        plan_state=PlanStateSnapshot(True, PermissionMode.DEFAULT, path),
    )
    checked = await tool.check_permissions(args, context)
    assert checked.behavior is PermissionBehavior.ASK
    invoked = tool._invoke(args)
    assert invoked.ok
    assert grants == ["run_command(python -m pytest -q)"]

    bad = {
        "requested_permissions": [
            {"rule": "run_command(rm -rf /)", "reason": "unsafe"}
        ]
    }
    state.active = True
    state.previous_mode = PermissionMode.DEFAULT.value
    state.artifact_path = path
    rejected = await tool.check_permissions(bad, context)
    assert rejected.behavior is PermissionBehavior.DENY
