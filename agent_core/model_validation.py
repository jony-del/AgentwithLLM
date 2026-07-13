from __future__ import annotations

from agent_core import tokens

CLAUDE_PROVIDER = "claude"
OPENAI_PROVIDER = "openai"
OPENAI_COMPAT_PROVIDER = "openai-compat"
FAKE_PROVIDER = "fake"

PROVIDERS = (CLAUDE_PROVIDER, OPENAI_PROVIDER, OPENAI_COMPAT_PROVIDER, FAKE_PROVIDER)


def is_model_allowed(provider: str, model: str | None) -> bool:
    """Return whether ``model`` is acceptable for an explicitly chosen provider."""
    if provider == FAKE_PROVIDER:
        return True
    model_id = (model or "").strip()
    if not model_id:
        return False
    if provider == CLAUDE_PROVIDER:
        return tokens.is_supported_model(model_id)
    if provider in {OPENAI_PROVIDER, OPENAI_COMPAT_PROVIDER}:
        return True
    return False


def unsupported_model_message(tool: str, provider: str, model: str) -> str:
    if provider == CLAUDE_PROVIDER:
        return (
            f"[{tool}] unsupported model {model!r} for provider 'claude'; refusing to spawn. "
            "Name a known Claude family (e.g. claude-haiku-4-5-*, claude-sonnet-4-6, "
            "claude-opus-4-8, claude-fable-5) or omit model to inherit the parent's."
        )
    return (
        f"[{tool}] unsupported model {model!r} for provider {provider!r}; refusing to spawn. "
        "Use a non-empty model id or omit model to inherit the parent's."
    )
