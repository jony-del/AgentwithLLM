from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_core.permission_types import (
    DecisionSource,
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
    PermissionResult,
)
from agent_core.tools.builtin import EchoTool


async def test_tool_default_permission_is_passthrough(tmp_path: Path) -> None:
    result = await EchoTool().check_permissions(
        {"text": "hello"},
        PermissionContext(PermissionMode.DEFAULT, tmp_path, interactive=False),
    )

    assert result.behavior is PermissionBehavior.PASSTHROUGH
    assert result.decision_source is DecisionSource.TOOL
    assert not result.classifier_approvable
    assert not result.bypass_immune


def test_permission_result_factories_are_fail_safe() -> None:
    ask = PermissionResult.ask("needs review")
    deny = PermissionResult.deny("blocked")

    assert ask.behavior is PermissionBehavior.ASK
    assert not ask.classifier_approvable
    assert not ask.bypass_immune
    assert deny.behavior is PermissionBehavior.DENY
    assert deny.bypass_immune


def test_permission_context_is_frozen(tmp_path: Path) -> None:
    context = PermissionContext(PermissionMode.DEFAULT, tmp_path, interactive=False)

    with pytest.raises(FrozenInstanceError):
        context.mode = PermissionMode.BYPASS  # type: ignore[misc]
