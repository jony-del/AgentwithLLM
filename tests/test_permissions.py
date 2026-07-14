from dataclasses import dataclass

from agent_core.models import ToolCall, ToolRisk
from agent_core.permission_rules import RuleSet
from agent_core.permissions import (
    PermissionMode,
    PermissionPolicy,
    next_shift_tab_permission_mode,
    permission_mode_label,
)
from agent_core.tools.base import Tool
from agent_core.tools.builtin import EchoTool, ReadTextFileTool, RunCommandTool, WriteTextFileTool


def test_plan_mode_strictly_denies_write_tools() -> None:
    decision = PermissionPolicy(PermissionMode.PLAN).decide(WriteTextFileTool())
    assert not decision.allowed
    assert not decision.dry_run
    assert "read-only" in decision.reason


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
    # A non-matching command is delegated to the automated classifier.
    other = policy.decide(tool, ToolCall("run_command", {"command": "ls"}))
    assert not other.allowed
    assert other.classify


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
    # The stub opts IN explicitly — the real SandboxConfig default is False (D4),
    # asserted separately below.
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


def test_sandbox_auto_allow_defaults_off() -> None:
    # D4: a sandboxed command still confirms unless the operator opted in explicitly.
    from agent_core.sandbox import SandboxConfig

    assert SandboxConfig().auto_allow_command_if_sandboxed is False

    rules = RuleSet.from_lists(ask=["run_command"])
    sandbox = _StubSandbox(True)
    sandbox.config.auto_allow_command_if_sandboxed = False
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules, sandbox=sandbox)
    decision = policy.decide(RunCommandTool(), ToolCall("run_command", {"command": "ls"}))
    assert not decision.allowed  # ask rule stands; non-interactive → deny


def test_sandbox_coupling_off_when_not_sandboxed() -> None:
    rules = RuleSet.from_lists(ask=["run_command"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, rules=rules, sandbox=_StubSandbox(False))
    tool = RunCommandTool()
    decision = policy.decide(tool, ToolCall("run_command", {"command": "ls"}))
    # not sandboxed → the ask rule stands (non-interactive → deny)
    assert not decision.allowed


def test_no_rules_uses_auto_fast_path_and_classifier_boundary() -> None:
    policy = PermissionPolicy(PermissionMode.AUTO)
    assert policy.decide(EchoTool(), ToolCall("echo", {})).allowed
    action = policy.decide(RunCommandTool(), ToolCall("run_command", {"command": "ls"}))
    assert not action.allowed and action.classify


class _GenericWriteTool(Tool):
    name = "generic_write"
    description = "A stateful action that is not a native workspace edit."
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.WRITE


def test_acceptedits_allows_native_edits_but_not_generic_write_actions() -> None:
    policy = PermissionPolicy(PermissionMode.ACCEPTEDITS, prompter=_prompter("deny"))
    assert policy.decide(WriteTextFileTool()).allowed
    generic = policy.decide(_GenericWriteTool())
    assert generic.ask_user
    assert not generic.allowed


def test_dontask_converts_all_prompts_to_hard_denials() -> None:
    policy = PermissionPolicy(PermissionMode.DONTASK, prompter=_prompter("once"))
    generic = policy.decide(_GenericWriteTool())
    assert not generic.allowed
    assert not generic.ask_user
    assert "dontask" in generic.reason


def test_plan_mode_cannot_be_bypassed_by_allow_rule() -> None:
    rules = RuleSet.from_lists(allow=["write_text_file"])
    decision = PermissionPolicy(PermissionMode.PLAN, rules=rules).decide(WriteTextFileTool())
    assert not decision.allowed
    assert "read-only" in decision.reason


def test_shift_tab_cycles_only_the_four_interactive_modes() -> None:
    mode = PermissionMode.DEFAULT
    seen = []
    for _ in range(4):
        mode = next_shift_tab_permission_mode(mode)
        seen.append(mode)
    assert seen == [
        PermissionMode.ACCEPTEDITS,
        PermissionMode.PLAN,
        PermissionMode.AUTO,
        PermissionMode.DEFAULT,
    ]
    assert next_shift_tab_permission_mode(PermissionMode.DONTASK) is PermissionMode.DEFAULT
    assert next_shift_tab_permission_mode(PermissionMode.BYPASS) is PermissionMode.DEFAULT
    assert permission_mode_label(PermissionMode.DEFAULT) == "manual mode on"


# -- secret-path safety net (reads included, bypass-immune) --------------------------


def test_reading_env_file_requires_confirmation() -> None:
    # READ tools are auto-allowed in general, but a secret-bearing path still confirms
    # (non-interactive → deny). Reading a secret is an exfiltration channel.
    policy = PermissionPolicy(PermissionMode.DEFAULT)
    tool = ReadTextFileTool()
    decision = policy.decide(tool, ToolCall("read_text_file", {"path": ".env"}))
    assert not decision.allowed
    assert "secret" in decision.reason
    # An ordinary read is unaffected.
    assert policy.decide(tool, ToolCall("read_text_file", {"path": "src/main.py"})).allowed


def test_secret_path_safety_net_is_bypass_immune() -> None:
    policy = PermissionPolicy(PermissionMode.BYPASS)
    tool = ReadTextFileTool()
    for path in (".env", "config/.env.production", "~/.ssh/known_hosts", "certs/server.pem", ".aws/credentials"):
        decision = policy.decide(tool, ToolCall("read_text_file", {"path": path}))
        assert not decision.allowed, path
        assert "secret" in decision.reason


def test_secret_path_applies_to_writes_too() -> None:
    policy = PermissionPolicy(PermissionMode.ACCEPTEDITS)  # would normally allow writes
    tool = WriteTextFileTool()
    decision = policy.decide(tool, ToolCall("write_text_file", {"path": ".env", "content": "K=v"}))
    assert not decision.allowed


def test_secret_deny_rule_still_beats_the_ask() -> None:
    # An explicit deny outranks the confirm-based safety net (hard no, even interactively).
    rules = RuleSet.from_lists(deny=["read_text_file(**/.env)"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("once"), rules=rules)
    tool = ReadTextFileTool()
    decision = policy.decide(tool, ToolCall("read_text_file", {"path": "app/.env"}))
    assert not decision.allowed
    assert "denied by rule" in decision.reason


# -- session "always" granularity ------------------------------------------------------


def test_always_for_shell_command_is_per_subcommand() -> None:
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("always"))
    tool = RunCommandTool()
    call = ToolCall("run_command", {"command": "git status"})

    first = policy.confirm(policy.decide(tool, call), tool, call)
    assert first.allowed

    # The same normalized command is now session-allowed...
    again = policy.decide(tool, ToolCall("run_command", {"command": "git status"}))
    assert again.allowed and not again.ask_user
    # ...but a DIFFERENT command still asks — "always" must not be a whole-tool grant.
    other = policy.decide(tool, ToolCall("run_command", {"command": "rm -rf build"}))
    assert not other.allowed
    assert other.ask_user


def test_always_compound_command_requires_every_subcommand_granted() -> None:
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("always"))
    tool = RunCommandTool()
    call = ToolCall("run_command", {"command": "git status"})
    policy.confirm(policy.decide(tool, call), tool, call)

    # A compound line is covered only when every sub-command was granted.
    partial = policy.decide(tool, ToolCall("run_command", {"command": "git status && rm x"}))
    assert not partial.allowed


def test_session_always_does_not_override_deny_or_safety_net() -> None:
    rules = RuleSet.from_lists(deny=["run_command(rm *)"])
    policy = PermissionPolicy(PermissionMode.DEFAULT, prompter=_prompter("always"), rules=rules)

    write_tool = WriteTextFileTool()
    write_call = ToolCall("write_text_file", {"path": "notes.txt", "content": "x"})
    policy.confirm(policy.decide(write_tool, write_call), write_tool, write_call)
    # write_text_file is session-allowed now, but a protected path still confirms.
    guarded = policy.decide(write_tool, ToolCall("write_text_file", {"path": ".git/config", "content": "x"}))
    assert not guarded.allowed
    assert guarded.ask_user

    run_tool = RunCommandTool()
    ls_call = ToolCall("run_command", {"command": "ls"})
    policy.confirm(policy.decide(run_tool, ls_call), run_tool, ls_call)
    # A deny rule is immune to any prior "always".
    denied = policy.decide(run_tool, ToolCall("run_command", {"command": "rm -rf /"}))
    assert not denied.allowed
    assert "denied by rule" in denied.reason

