from agent_core.models import ToolCall
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.tools.demo import EchoTool, WriteTextFileTool


def test_plan_mode_dry_runs_write_tools() -> None:
    decision = PermissionPolicy(PermissionMode.PLAN).decide(WriteTextFileTool())
    assert decision.allowed
    assert decision.dry_run


def test_default_allows_read_tools() -> None:
    decision = PermissionPolicy(PermissionMode.DEFAULT).decide(EchoTool())
    assert decision.allowed
    assert not decision.dry_run


def test_default_noninteractive_denies_write_tools() -> None:
    # No prompter wired => non-interactive => an "ask" collapses into a denial.
    decision = PermissionPolicy(PermissionMode.DEFAULT).decide(WriteTextFileTool())
    assert not decision.allowed


def _prompter(answer: str):
    return lambda name, risk, args: answer


def test_confirm_once_allows_only_this_call() -> None:
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("once"))
    tool = WriteTextFileTool()
    call = ToolCall("write_text_file", {"path": "x", "content": "y"})

    decision = policy.confirm(policy.decide(tool), tool, call)
    assert decision.allowed
    # "once" does not persist: the next decide() still wants to ask.
    assert policy.decide(tool).ask_user


def test_confirm_always_grants_for_session() -> None:
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("always"))
    tool = WriteTextFileTool()
    call = ToolCall("write_text_file", {"path": "x", "content": "y"})

    first = policy.confirm(policy.decide(tool), tool, call)
    assert first.allowed
    # The tool is now session-allowed, so the next decide() grants without asking.
    second = policy.decide(tool)
    assert second.allowed
    assert not second.ask_user


def test_confirm_deny_rejects() -> None:
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("deny"))
    tool = WriteTextFileTool()
    call = ToolCall("write_text_file", {"path": "x", "content": "y"})

    decision = policy.confirm(policy.decide(tool), tool, call)
    assert not decision.allowed

