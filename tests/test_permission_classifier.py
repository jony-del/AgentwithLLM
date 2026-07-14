from __future__ import annotations

from agent_core.models import LLMResult, Message, ToolCall, TokenUsage
from agent_core.permission_classifier import ProviderAutoPermissionClassifier
from agent_core.providers.base import LLMProvider, ProviderConfig
from agent_core.tools.builtin import RunCommandTool


class _ClassifierProvider(LLMProvider):
    def __init__(self, result: LLMResult) -> None:
        self.result = result
        self.calls = []

    async def complete(self, messages, tools, config, stream=None, should_cancel=None):
        self.calls.append((messages, tools, config))
        return self.result


def _classifier(provider: LLMProvider) -> ProviderAutoPermissionClassifier:
    return ProviderAutoPermissionClassifier(
        provider,
        lambda: ProviderConfig(model="test-model", max_tokens=1024),
    )


async def test_provider_classifier_accepts_structured_tool_verdict() -> None:
    result = LLMResult(
        content="",
        tool_calls=[
            ToolCall("classify_permission", {"allow": True, "reason": "matches the request"})
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=2),
    )
    provider = _ClassifierProvider(result)
    verdict = await _classifier(provider).classify(
        RunCommandTool(),
        ToolCall("run_command", {"command": "git status"}),
        [Message("user", "Inspect the repository")],
    )

    assert verdict.allowed
    assert verdict.reason == "matches the request"
    assert verdict.model == "test-model"
    assert verdict.usage and verdict.usage["input_tokens"] == 10
    assert provider.calls[0][2].max_tokens == 256
    assert provider.calls[0][2].stream is False


async def test_provider_classifier_fails_closed_on_invalid_output() -> None:
    provider = _ClassifierProvider(LLMResult(content="not a decision"))
    verdict = await _classifier(provider).classify(
        RunCommandTool(), ToolCall("run_command", {"command": "rm -rf build"}), []
    )
    assert not verdict.allowed
    assert verdict.unavailable
    assert "invalid" in verdict.reason


async def test_classifier_projection_excludes_untrusted_non_user_context() -> None:
    provider = _ClassifierProvider(
        LLMResult(
            content='{"allow":false,"reason":"blocked"}',
        )
    )
    messages = [
        Message("system", "SYSTEM_SECRET"),
        Message("user", "PROJECT_INSTRUCTIONS", metadata={"pinned": "user_context"}),
        Message("user", "HOOK_INSTRUCTION", metadata={"hook": "user_prompt_context"}),
        Message("user", "real user request"),
        Message("assistant", "ASSISTANT_PROSE", metadata={"tool_calls": []}),
        Message("tool", "TOOL_OUTPUT_INJECTION"),
    ]
    await _classifier(provider).classify(
        RunCommandTool(), ToolCall("run_command", {"command": "git status"}), messages
    )

    sent = provider.calls[0][0][-1].content
    assert "real user request" in sent
    assert "SYSTEM_SECRET" not in sent
    assert "PROJECT_INSTRUCTIONS" not in sent
    assert "HOOK_INSTRUCTION" not in sent
    assert "ASSISTANT_PROSE" not in sent
    assert "TOOL_OUTPUT_INJECTION" not in sent


async def test_classifier_rejects_oversized_current_action_without_api_call() -> None:
    provider = _ClassifierProvider(
        LLMResult(
            content="",
            tool_calls=[ToolCall("classify_permission", {"allow": True, "reason": "x"})],
        )
    )
    verdict = await _classifier(provider).classify(
        RunCommandTool(),
        ToolCall("run_command", {"command": "x" * 20_000}),
        [Message("user", "run it")],
    )
    assert not verdict.allowed and verdict.unavailable
    assert provider.calls == []
