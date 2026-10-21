"""Credential resolution: env-first, keyring second, clean degradation."""

from __future__ import annotations

import sys
import types

import pytest

from agent_core import credentials
from agent_core.providers import ClaudeProvider, OpenAICompatProvider, OpenAIResponsesProvider
from agent_core.providers.base import ProviderConfig

_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY")


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch):
    """No ambient keys, fresh keyring cache per test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    credentials.clear_cache()
    yield
    credentials.clear_cache()


def _install_fake_keyring(monkeypatch, store: dict | None = None) -> types.SimpleNamespace:
    """A working in-memory keyring stand-in (the real extra stays uninstalled)."""
    monkeypatch.delenv("POLARIS_DISABLE_KEYRING", raising=False)
    state = store if store is not None else {}
    calls: list[tuple[str, str]] = []

    fake = types.SimpleNamespace()

    def get_password(service: str, account: str) -> str | None:
        calls.append(("get", account))
        return state.get((service, account))

    def set_password(service: str, account: str, secret: str) -> None:
        calls.append(("set", account))
        state[(service, account)] = secret

    fake.get_password = get_password
    fake.set_password = set_password
    fake.calls = calls
    fake.store = state
    monkeypatch.setitem(sys.modules, "keyring", fake)
    # A newly installed backend must not be masked by negative lookups cached earlier.
    credentials.clear_cache()
    return fake


def test_env_var_wins_over_keyring(monkeypatch) -> None:
    fake = _install_fake_keyring(monkeypatch, {("polaris", "ANTHROPIC_API_KEY"): "kr-key"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    assert credentials.get_api_key("ANTHROPIC_API_KEY") == "env-key"
    assert fake.calls == []  # keyring never consulted when the env var is set


def test_keyring_used_when_env_absent(monkeypatch) -> None:
    _install_fake_keyring(monkeypatch, {("polaris", "ANTHROPIC_API_KEY"): "kr-key"})

    assert credentials.get_api_key("ANTHROPIC_API_KEY") == "kr-key"


def test_missing_keyring_package_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("POLARIS_DISABLE_KEYRING", raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)  # import raises ImportError

    assert credentials.get_api_key("ANTHROPIC_API_KEY") is None
    assert not credentials.keyring_available()


def test_keyring_backend_failure_degrades_to_none(monkeypatch) -> None:
    monkeypatch.delenv("POLARIS_DISABLE_KEYRING", raising=False)
    fake = types.SimpleNamespace()

    def boom(service: str, account: str) -> str:
        raise RuntimeError("no secret service on this bus")

    fake.get_password = boom
    monkeypatch.setitem(sys.modules, "keyring", fake)

    assert credentials.get_api_key("ANTHROPIC_API_KEY") is None


def test_keyring_lookup_is_cached_until_cleared(monkeypatch) -> None:
    fake = _install_fake_keyring(monkeypatch, {("polaris", "OPENAI_API_KEY"): "kr-key"})

    assert credentials.get_api_key("OPENAI_API_KEY") == "kr-key"
    assert credentials.get_api_key("OPENAI_API_KEY") == "kr-key"
    assert fake.calls == [("get", "OPENAI_API_KEY")]  # one backend hit, then cache

    credentials.clear_cache()
    assert credentials.get_api_key("OPENAI_API_KEY") == "kr-key"
    assert fake.calls == [("get", "OPENAI_API_KEY"), ("get", "OPENAI_API_KEY")]


def test_store_api_key_writes_keyring_and_primes_cache(monkeypatch) -> None:
    fake = _install_fake_keyring(monkeypatch)

    credentials.store_api_key("OPENAI_API_KEY", "new-key")

    assert fake.store[("polaris", "OPENAI_API_KEY")] == "new-key"
    fake.store.clear()  # prove the follow-up read came from the cache, not the store
    assert credentials.get_api_key("OPENAI_API_KEY") == "new-key"
    assert ("get", "OPENAI_API_KEY") not in fake.calls


def test_store_without_keyring_raises_actionable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)  # POLARIS_DISABLE_KEYRING still set

    with pytest.raises(RuntimeError, match="keyring"):
        credentials.store_api_key("ANTHROPIC_API_KEY", "k")


def test_disable_env_var_bypasses_keyring(monkeypatch) -> None:
    _install_fake_keyring(monkeypatch, {("polaris", "ANTHROPIC_API_KEY"): "kr-key"})
    monkeypatch.setenv("POLARIS_DISABLE_KEYRING", "1")

    assert credentials.get_api_key("ANTHROPIC_API_KEY") is None
    assert not credentials.keyring_available()
    with pytest.raises(RuntimeError, match="keyring"):
        credentials.store_api_key("ANTHROPIC_API_KEY", "k")


def test_provider_credential_mapping_and_legacy_fallback(monkeypatch) -> None:
    assert credentials.provider_credential("fake") is None
    assert credentials.provider_credential("claude") is None

    monkeypatch.setenv("OPENAI_API_KEY", "legacy")
    assert credentials.provider_credential("openai") == "legacy"
    # openai-compat falls back to the legacy var when its own is unset.
    assert credentials.provider_credential("openai-compat") == "legacy"

    _install_fake_keyring(monkeypatch, {("polaris", "OPENAI_COMPAT_API_KEY"): "kr-compat"})
    # The canonical var (keyring) outranks the legacy env var.
    assert credentials.provider_credential("openai-compat") == "kr-compat"


def test_providers_resolve_keys_through_helper(monkeypatch) -> None:
    _install_fake_keyring(
        monkeypatch,
        {
            ("polaris", "ANTHROPIC_API_KEY"): "kr-anthropic",
            ("polaris", "OPENAI_API_KEY"): "kr-openai",
            ("polaris", "OPENAI_COMPAT_API_KEY"): "kr-compat",
        },
    )

    assert ClaudeProvider().api_key == "kr-anthropic"
    assert OpenAIResponsesProvider().api_key == "kr-openai"
    assert OpenAICompatProvider().api_key == "kr-compat"


def test_provider_env_behavior_byte_identical(monkeypatch) -> None:
    # With the env var set, construction sees exactly what it saw before the helper.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    _install_fake_keyring(monkeypatch, {("polaris", "ANTHROPIC_API_KEY"): "kr-other"})
    assert ClaudeProvider().api_key == "env-key"
    # An explicit constructor argument still outranks everything.
    assert ClaudeProvider(api_key="explicit").api_key == "explicit"


def test_compat_provider_falls_back_to_legacy_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    assert OpenAICompatProvider().api_key == "legacy-key"


async def test_missing_key_errors_mention_env_var_and_setup() -> None:
    cases = [
        (ClaudeProvider(), "ANTHROPIC_API_KEY"),
        (OpenAIResponsesProvider(), "OPENAI_API_KEY"),
        (OpenAICompatProvider(), "OPENAI_COMPAT_API_KEY"),
    ]
    for provider, env_var in cases:
        provider.api_key = None
        with pytest.raises(RuntimeError) as excinfo:
            await provider.complete([], [], ProviderConfig(model="m"))
        assert env_var in str(excinfo.value)
        assert "polaris setup" in str(excinfo.value)
