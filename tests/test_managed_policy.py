import json
from pathlib import Path

import pytest

from agent_core.managed_policy import ManagedPolicyDefinition, StaticManagedPolicyProvider
from agent_core.models import ToolCall
from agent_core.permission_types import (
    DecisionSource,
    ManagedPolicySnapshot,
    PermissionBehavior,
    PermissionMode,
    PermissionRuleSource,
)
from agent_core.permissions import PermissionPolicy
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.storage import read_events
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


def test_managed_policy_active_writes_startup_event(tmp_path: Path) -> None:
    provider = StaticManagedPolicyProvider(
        ManagedPolicyDefinition(
            deny=("echo",),
            forbidden_modes=frozenset({PermissionMode.BYPASS}),
            digest="0123456789abcdef",
        )
    )
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path), session_dir=""),
        managed_policy_provider=provider,
    )

    events = [e for e in read_events(agent.logger.path) if e["event"] == "managed_policy"]
    assert len(events) == 1
    event = events[0]
    assert event["state"] == "active"
    assert event["digest"] == "0123456789ab"
    assert event["forbidden_modes"] == ["bypass"]
    assert event["rules"] == {"allow": 0, "deny": 1, "ask": 0}
    # Rule text is the administrator's deployment detail and never enters the log.
    assert "echo" not in json.dumps(event)


def test_no_managed_policy_writes_no_startup_event(tmp_path: Path) -> None:
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(run_dir=str(tmp_path), session_dir=""),
        managed_policy_provider=StaticManagedPolicyProvider(ManagedPolicyDefinition()),
    )

    # The log file is created lazily on first write; no events at all also passes.
    events = (
        [e for e in read_events(agent.logger.path) if e["event"] == "managed_policy"]
        if agent.logger.path.exists()
        else []
    )
    assert events == []


class _ChangingProvider:
    def __init__(self) -> None:
        self.definition = ManagedPolicyDefinition(deny=("echo",), digest="a" * 64)

    def load(self) -> ManagedPolicyDefinition:
        return self.definition


def test_managed_policy_reload_notifies_listener_once_per_digest_change() -> None:
    provider = _ChangingProvider()
    seen: list[ManagedPolicySnapshot] = []
    policy = PermissionPolicy(
        PermissionMode.DEFAULT,
        managed_policy_provider=provider,
        managed_policy_listener=lambda snapshot, definition: seen.append(snapshot),
    )

    assert policy.refresh_managed_policy() is None
    assert seen == []  # same digest as the constructor load: no churn event

    provider.definition = ManagedPolicyDefinition(deny=("echo",), digest="b" * 64)
    assert policy.refresh_managed_policy() is None
    assert [snapshot.policy_digest for snapshot in seen] == ["b" * 64]

    assert policy.refresh_managed_policy() is None
    assert len(seen) == 1  # unchanged digest stays silent
