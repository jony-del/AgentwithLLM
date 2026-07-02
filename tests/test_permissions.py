from dataclasses import dataclass

from agent_core.models import ToolCall
from agent_core.permission_rules import RuleSet
from agent_core.permissions import PermissionMode, PermissionPolicy
from agent_core.tools.builtin import EchoTool, RunCommandTool, WriteTextFileTool


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


# -- argument-aware rule pipeline ----------------------------------------------------


def test_deny_rule_blocks_matching_command() -> None:
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    policy = PermissionPolicy(PermissionMode.AUTO, rules=rules)
    tool = RunCommandTool()
    denied = policy.decide(tool, ToolCall("run_command", {"command": "rm -rf /"}))
    assert not denied.allowed
    assert "denied by rule" in denied.reason
    # A non-matching command is unaffected by the deny rule (auto still denies DANGEROUS).
    other = policy.decide(tool, ToolCall("run_command", {"command": "ls"}))
    assert not other.allowed
    assert "auto denies" in other.reason


def test_allow_rule_permits_command_under_default() -> None:
    rules = RuleSet.from_lists(allow=["run_command(git *)"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules)
    tool = RunCommandTool()
    decision = policy.decide(tool, ToolCall("run_command", {"command": "git status"}))
    assert decision.allowed
    assert "allowed by rule" in decision.reason


def test_deny_beats_allow() -> None:
    rules = RuleSet.from_lists(allow=["run_command(git *)"], deny=["run_command(git push *)"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules)
    tool = RunCommandTool()
    assert not policy.decide(tool, ToolCall("run_command", {"command": "git push --force"})).allowed
    assert policy.decide(tool, ToolCall("run_command", {"command": "git status"})).allowed


def test_bypass_mode_allows_unmatched_but_deny_still_wins() -> None:
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    policy = PermissionPolicy(PermissionMode.BYPASS, rules=rules)
    tool = RunCommandTool()
    # bypass allows a dangerous command the plain matrix would block...
    assert policy.decide(tool, ToolCall("run_command", {"command": "curl x"})).allowed
    # ...but a deny rule is bypass-immune.
    assert not policy.decide(tool, ToolCall("run_command", {"command": "rm x"})).allowed


def test_sensitive_path_safety_net_is_bypass_immune() -> None:
    # Even under bypass, writing a protected path must confirm (non-interactive → deny).
    policy = PermissionPolicy(PermissionMode.BYPASS)
    tool = WriteTextFileTool()
    decision = policy.decide(tool, ToolCall("write_text_file", {"path": ".git/config", "content": "x"}))
    assert not decision.allowed
    assert "protected path" in decision.reason


@dataclass
class _StubSandboxConfig:
    auto_allow_command_if_sandboxed: bool = True


class _StubSandbox:
    """Minimal stand-in exposing what PermissionPolicy calls on a SandboxManager."""

    def __init__(self, sandboxes: bool) -> None:
        self.config = _StubSandboxConfig()
        self._sandboxes = sandboxes

    def should_sandbox(self, command):
        return self._sandboxes


def test_sandbox_coupling_auto_allows_sandboxed_command() -> None:
    # An ask rule would normally prompt, but a command that will be sandboxed skips it.
    rules = RuleSet.from_lists(ask=["run_command"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules, sandbox=_StubSandbox(True))
    tool = RunCommandTool()
    decision = policy.decide(tool, ToolCall("run_command", {"command": "ls"}))
    assert decision.allowed
    assert "sandboxed" in decision.reason


def test_sandbox_coupling_off_when_not_sandboxed() -> None:
    rules = RuleSet.from_lists(ask=["run_command"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules, sandbox=_StubSandbox(False))
    tool = RunCommandTool()
    decision = policy.decide(tool, ToolCall("run_command", {"command": "ls"}))
    # not sandboxed → the ask rule stands (non-interactive → deny)
    assert not decision.allowed


def test_no_rules_preserves_legacy_behavior() -> None:
    # Empty rule set + no sandbox → identical to the original per-mode matrix.
    policy = PermissionPolicy(PermissionMode.AUTO)
    assert policy.decide(EchoTool(), ToolCall("echo", {})).allowed
    assert not policy.decide(RunCommandTool(), ToolCall("run_command", {"command": "ls"})).allowed

