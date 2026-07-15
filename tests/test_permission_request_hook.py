"""PermissionRequest hook (R1): the programmatic approval seam for "ask" decisions.

Control-path contract:
- consulted ONLY for asks (interactive ``ask_user`` and the headless ``ask_collapsed``),
  never for hard denies — a hook cannot launder a deny rule;
- folding is fail-closed: any deny wins over any allow;
- a crashing hook yields NO opinion — the gated action falls back to the normal ask
  path (interactive prompt / headless denial), never a silent allow;
- with no hooks registered, behavior is exactly the pre-hook behavior.
"""

from __future__ import annotations

from pathlib import Path

from agent_core.hooks import HookContext, HookEvent, HookOutcome, HookPipeline
from agent_core.models import ToolCall, ToolResult, ToolRisk
from agent_core.permission_rules import RuleSet
from agent_core.permissions import PermissionPolicy
from agent_core.tools.base import ConcurrencySpec, Tool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.builtin import WriteTextFileTool


class _DangerousTool(Tool):
    name = "dangerous_thing"
    description = "test tool that needs confirmation"
    input_schema = {"type": "object", "properties": {}}
    risk = ToolRisk.DANGEROUS

    def __init__(self) -> None:
        self.invocations = 0

    def concurrency_spec(self, arguments) -> ConcurrencySpec:
        return ConcurrencySpec()

    def _invoke(self, arguments) -> ToolResult:
        self.invocations += 1
        return ToolResult(self.name, "ran")


class _DecidingHook:
    def __init__(self, decision: str | None, reason: str | None = None, boom: bool = False):
        self.decision = decision
        self.reason = reason
        self.boom = boom
        self.contexts: list[HookContext] = []

    async def on_permission_request(self, ctx: HookContext) -> HookOutcome:
        self.contexts.append(ctx)
        if self.boom:
            raise RuntimeError("approver crashed")
        return HookOutcome(decision=self.decision, reason=self.reason)


def _executor(
    tool: Tool,
    hooks: list | None = None,
    *,
    prompter=None,
    rules: RuleSet | None = None,
) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(tool)
    policy = PermissionPolicy("default", prompter=prompter, rules=rules)
    pipeline = HookPipeline(permission_request_hooks=hooks or [])
    return ToolExecutor(registry, policy, pipeline)


async def test_hook_allow_resolves_headless_ask() -> None:
    tool = _DangerousTool()
    hook = _DecidingHook("allow", reason="audited ok")
    executor = _executor(tool, [hook])

    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert result.ok and result.content == "ran"
    assert tool.invocations == 1
    # The hook received the bounded ask projection.
    ctx = hook.contexts[0]
    assert ctx.event is HookEvent.PERMISSION_REQUEST
    assert ctx.detail["tool"] == "dangerous_thing" and ctx.detail["risk"] == "dangerous"


async def test_hook_deny_refuses_and_tool_never_runs() -> None:
    tool = _DangerousTool()
    executor = _executor(tool, [_DecidingHook("deny", reason="not on my watch")])

    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert not result.ok and "denied" in result.content
    assert "not on my watch" in result.content
    assert tool.invocations == 0


async def test_deny_wins_over_allow_fail_closed_fold() -> None:
    tool = _DangerousTool()
    executor = _executor(tool, [_DecidingHook("allow"), _DecidingHook("deny")])
    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert not result.ok
    assert tool.invocations == 0


async def test_crashing_hook_gives_no_opinion_never_silent_allow() -> None:
    tool = _DangerousTool()
    executor = _executor(tool, [_DecidingHook(None, boom=True)])
    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    # Headless collapse still applies: denied, tool never ran — fail-closed.
    assert not result.ok
    assert tool.invocations == 0


async def test_hard_deny_never_consults_the_hook() -> None:
    tool = _DangerousTool()
    hook = _DecidingHook("allow")  # would allow — but must never be asked
    executor = _executor(tool, [hook], rules=RuleSet.from_lists(deny=["dangerous_thing"]))

    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert not result.ok
    assert tool.invocations == 0
    assert hook.contexts == []  # a deny rule is not an ask


async def test_no_hooks_behavior_is_unchanged() -> None:
    tool = _DangerousTool()
    executor = _executor(tool, hooks=[])
    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert not result.ok and "non-interactive" in result.content
    assert tool.invocations == 0


async def test_interactive_hook_allow_bypasses_the_prompter() -> None:
    prompts: list[str] = []

    def prompter(name: str, risk: str, arguments: dict) -> str:
        prompts.append(name)
        return "deny"

    tool = _DangerousTool()
    executor = _executor(tool, [_DecidingHook("allow")], prompter=prompter)
    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert result.ok and tool.invocations == 1
    assert prompts == []  # the hook resolved the ask; the user was never prompted


async def test_interactive_no_opinion_falls_through_to_prompter() -> None:
    def prompter(name: str, risk: str, arguments: dict) -> str:
        return "once"

    tool = _DangerousTool()
    executor = _executor(tool, [_DecidingHook(None)], prompter=prompter)
    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])
    assert result.ok and tool.invocations == 1  # user confirmation still works


async def test_dontask_converts_ask_before_permission_request_hook() -> None:
    tool = _DangerousTool()
    hook = _DecidingHook("allow")
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy("dontask", prompter=lambda name, risk, arguments: "once"),
        HookPipeline(permission_request_hooks=[hook]),
    )

    (result,) = await executor.execute_many([ToolCall("dangerous_thing", {})])

    assert not result.ok
    assert tool.invocations == 0
    assert hook.contexts == []


async def test_safety_ask_can_use_explicit_permission_request_hook(tmp_path: Path) -> None:
    tool = WriteTextFileTool(tmp_path)
    hook = _DecidingHook("allow", reason="delegated human approval")
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy("default", workspace=tmp_path),
        HookPipeline(permission_request_hooks=[hook]),
    )

    (result,) = await executor.execute_many(
        [ToolCall("write_text_file", {"path": ".git/config", "content": "x"})]
    )

    assert result.ok
    assert (tmp_path / ".git" / "config").read_text(encoding="utf-8") == "x"
    assert len(hook.contexts) == 1


async def test_permission_request_hook_cannot_allow_central_hard_deny(tmp_path: Path) -> None:
    tool = WriteTextFileTool(tmp_path)
    hook = _DecidingHook("allow")
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(
        registry,
        PermissionPolicy("bypass", workspace=tmp_path),
        HookPipeline(permission_request_hooks=[hook]),
    )

    (result,) = await executor.execute_many(
        [ToolCall("write_text_file", {"path": ".git/hooks/pre-commit", "content": "x"})]
    )

    assert not result.ok
    assert hook.contexts == []
    assert not (tmp_path / ".git" / "hooks" / "pre-commit").exists()
