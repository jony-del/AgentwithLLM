from pathlib import Path

import pytest

from agent_core.managed_policy import ManagedPolicyDefinition, StaticManagedPolicyProvider
from agent_core.models import ToolCall
from agent_core.permission_types import (
    DecisionSource,
    PermissionBehavior,
    PermissionMode,
    PermissionRuleSource,
)
from agent_core.permissions import PermissionPolicy
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.builtin import EchoTool, ReadTextFileTool


async def test_managed_deny_is_hard_and_preserves_provenance(tmp_path: Path) -> None:
    provider = StaticManagedPolicyProvider(
        ManagedPolicyDefinition(deny=("echo",), require_sandbox_for_unattended=False)
    )
    result = await PermissionPolicy(
        PermissionMode.BYPASS,
        workspace=tmp_path,
        managed_policy_provider=provider,
    ).evaluate(EchoTool(), ToolCall("echo", {"text": "hello"}))

    assert result.behavior is PermissionBehavior.DENY
    assert result.matched_rule is not None
    assert result.matched_rule.source is PermissionRuleSource.MANAGED
    assert "managed" in result.reason


def test_managed_policy_can_forbid_runtime_mode(tmp_path: Path) -> None:
    provider = StaticManagedPolicyProvider(
        ManagedPolicyDefinition(
            forbidden_modes=frozenset({PermissionMode.BYPASS}),
            require_sandbox_for_unattended=False,
        )
    )
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path), session_dir=""),
        managed_policy_provider=provider,
    )

    with pytest.raises(ValueError, match="forbidden by managed policy"):
        agent.set_permission_mode(PermissionMode.BYPASS)


def test_managed_policy_rejects_forbidden_initial_mode(tmp_path: Path) -> None:
    provider = StaticManagedPolicyProvider(
        ManagedPolicyDefinition(
            forbidden_modes=frozenset({PermissionMode.PLAN}),
            require_sandbox_for_unattended=False,
        )
    )

    with pytest.raises(ValueError, match="forbidden by managed policy"):
        ReActAgent(
            FakeProvider(),
            ReActConfig(run_dir=str(tmp_path), session_dir="", permission="plan"),
            managed_policy_provider=provider,
        )


async def test_managed_sandbox_requirement_overrides_explicit_opt_out(tmp_path: Path) -> None:
    provider = StaticManagedPolicyProvider(
        ManagedPolicyDefinition(require_sandbox_for_unattended=True)
    )
    result = await PermissionPolicy(
        PermissionMode.AUTO,
        workspace=tmp_path,
        managed_policy_provider=provider,
        allow_unsandboxed_unattended=True,
    ).evaluate(ReadTextFileTool(tmp_path), ToolCall("read_text_file", {"path": "README.md"}))

    assert result.behavior is PermissionBehavior.DENY
    assert result.decision_source is DecisionSource.MANAGED
