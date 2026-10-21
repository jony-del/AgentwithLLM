"""Centralized provider API-key resolution: environment first, then the OS keyring.

Every real provider resolves its key through :func:`get_api_key` instead of reading
``os.environ`` directly. Precedence is deliberately narrow:

1. the process environment (including a project ``.env`` loaded at startup) — this
   keeps behavior byte-identical for anyone who already exports the variable;
2. the OS credential store, via the optional ``keyring`` package (the
   ``credentials`` extra), under service name ``polaris`` with the env var name as
   the account — this is what ``polaris setup`` writes to.

The environment half is never cached, so a process that mutates ``os.environ``
(tests, ``.env`` loading) always sees fresh values. The keyring half — an import
that may be missing and a backend call that may be slow or fail — is cached per
env-var name for the life of the process; :func:`store_api_key` primes the cache
on write and :func:`clear_cache` is the test seam.

Setting ``POLARIS_DISABLE_KEYRING`` to a truthy value bypasses keyring entirely
(both reads and writes), which keeps test suites hermetic on machines where a
real credential was previously stored.
"""

from __future__ import annotations

import os

# keyring service name; the account name is the env var (e.g. "ANTHROPIC_API_KEY").
KEYRING_SERVICE = "polaris"

# Canonical env var(s) each real provider reads its API key from, in resolution
# order. openai-compat accepts the legacy OPENAI_API_KEY as a documented fallback
# (with a deprecation warning when it is the one that resolves).
PROVIDER_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai-compat": ("OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"),
}

_TRUTHY = {"1", "true", "yes", "on"}

# Per-process cache of keyring lookups (positive and negative). Environment reads
# bypass it entirely so env changes are always picked up.
_cache: dict[str, str | None] = {}


def keyring_disabled() -> bool:
    return os.getenv("POLARIS_DISABLE_KEYRING", "").strip().lower() in _TRUTHY


def _keyring_module():
    """Import keyring lazily; ``None`` when the optional extra is absent or disabled."""
    if keyring_disabled():
        return None
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def keyring_available() -> bool:
    """Whether the OS credential store can be offered as a storage option."""
    return _keyring_module() is not None


def get_api_key(env_var_name: str) -> str | None:
    """Resolve an API key: process environment first, then the OS credential store.

    Returns ``None`` when neither source has it. An empty-string env var counts as
    unset. A missing keyring package, a disabled keyring, or a backend failure all
    degrade to ``None`` — never an exception.
    """
    value = os.getenv(env_var_name)
    if value:
        return value
    if env_var_name in _cache:
        return _cache[env_var_name]
    stored: str | None = None
    keyring = _keyring_module()
    if keyring is not None:
        try:
            stored = keyring.get_password(KEYRING_SERVICE, env_var_name)
        except Exception:  # noqa: BLE001 - a broken backend must not break the run
            stored = None
    _cache[env_var_name] = stored or None
    return _cache[env_var_name]


def store_api_key(env_var_name: str, api_key: str) -> None:
    """Persist an API key in the OS credential store and prime the local cache.

    Raises ``RuntimeError`` with an actionable message when keyring is not
    installed, is disabled, or the backend rejects the write; callers (the setup
    wizard) catch it and fall back to environment-variable instructions.
    """
    keyring = _keyring_module()
    if keyring is None:
        raise RuntimeError(
            "the 'keyring' package is not available; install the credentials extra "
            "(pip install 'agent-with-llm[credentials]') or set the env var instead"
        )
    try:
        keyring.set_password(KEYRING_SERVICE, env_var_name, api_key)
    except Exception as exc:  # noqa: BLE001 - surface the backend failure as guidance
        raise RuntimeError(f"the OS credential store rejected the key: {exc}") from exc
    _cache[env_var_name] = api_key


def provider_credential(provider: str) -> str | None:
    """The API key a provider would resolve right now, or ``None``.

    Unknown providers and ``fake`` have no credential and return ``None``.
    """
    for env_var_name in PROVIDER_KEY_ENV_VARS.get(provider, ()):
        value = get_api_key(env_var_name)
        if value:
            return value
    return None


def clear_cache() -> None:
    """Drop cached keyring lookups (test seam; env reads were never cached)."""
    _cache.clear()
