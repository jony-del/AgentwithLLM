"""First-run configuration wizard behind ``polaris setup``.

Interactive on a TTY: pick a provider, model, and (for openai-compat) base URL,
paste an API key with hidden input, store it in the OS credential store when the
optional ``keyring`` extra is available (otherwise print exact per-OS env-var
instructions), write the non-secret settings to ``~/.polaris/settings.toml``, and
optionally make one minimal API call to verify connectivity.

Non-interactive (piped stdin, ``--non-interactive``): never prompts. Given
``--provider`` it writes config from flags (``--api-key-env VAR`` declares the key
is already exported); without it, prints what to configure and exits non-zero.

The API key is never written to ``settings.toml`` or any other plaintext file —
only to the OS credential store (service ``polaris``, account = the env var name).
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from dataclasses import dataclass
from typing import Any

from agent_core import credentials
from agent_core.config import user_settings_path, write_user_settings


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env_var: str
    description: str
    key_url: str
    default_model: str | None = None
    needs_base_url: bool = False


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        "claude",
        "ANTHROPIC_API_KEY",
        "Anthropic Messages API",
        "https://console.anthropic.com/",
        default_model="claude-opus-4-8",
    ),
    ProviderSpec(
        "openai",
        "OPENAI_API_KEY",
        "OpenAI Responses API",
        "https://platform.openai.com/api-keys",
        default_model="gpt-4.1-mini",
    ),
    ProviderSpec(
        "openai-compat",
        "OPENAI_COMPAT_API_KEY",
        "OpenAI-compatible endpoint (vLLM, LM Studio, DeepSeek, ...)",
        "",
        needs_base_url=True,
    ),
)

_SPECS_BY_NAME = {spec.name: spec for spec in PROVIDER_SPECS}


def default_arguments(*, provider: str | None = None) -> argparse.Namespace:
    """A fresh argument set for launching the wizard inline (e.g. from `chat`)."""
    return argparse.Namespace(
        provider=provider,
        model=None,
        base_url=None,
        api_key_env=None,
        verify=False,
        non_interactive=False,
    )


def run_setup(args: argparse.Namespace) -> int:
    """Entry point for ``polaris setup`` (and the inline first-run offer)."""
    if getattr(args, "non_interactive", False) or not _interactive():
        return _run_non_interactive(args)
    try:
        return _run_interactive(args)
    except KeyboardInterrupt:
        print("\nSetup cancelled; nothing was written.")
        return 130
    except EOFError:
        print("\nSetup cancelled (input closed); nothing was written.")
        return 1


def verify_connection(provider: str, model: str | None, base_url: str | None) -> bool:
    """One minimal real API call with the configured provider; never raises."""
    if not credentials.provider_credential(provider):
        print(f"[verify] skipped: no credential resolves for provider {provider!r}.")
        return False
    from agent_core.cli import _make_provider
    from agent_core.models import Message
    from agent_core.providers import ProviderConfig

    instance = _make_provider({"provider": provider, "base_url": base_url})
    config = ProviderConfig(model=model or "", max_tokens=8, stream=False, timeout=30.0)
    try:
        asyncio.run(
            instance.complete([Message("user", "Reply with the single word: ok")], [], config)
        )
    except Exception as exc:  # noqa: BLE001 - a check reports; it must not fail the setup
        print(f"[verify] connection check FAILED: {type(exc).__name__}: {exc}")
        return False
    print("[verify] connection OK — the provider answered a test request.")
    return True


def _interactive() -> bool:
    """Both stdin and stdout are real terminals (prompts can be seen and answered)."""
    try:
        return bool(sys.stdin) and sys.stdin.isatty() and bool(sys.stdout) and sys.stdout.isatty()
    except (ValueError, OSError):
        return False


def _current_user_settings() -> dict[str, Any]:
    """Existing user settings (for defaults on a re-run); ``{}`` when unreadable."""
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}
    path = user_settings_path()
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, ValueError):
        return {}


def _env_instructions(spec: ProviderSpec) -> str:
    """Exact per-OS commands for persisting the API key as an environment variable."""
    env_var = spec.env_var
    lines = [f"Set {env_var} in your environment before running polaris:"]
    if os.name == "nt":
        lines.append(
            f'  persist for your user (PowerShell): '
            f'[Environment]::SetEnvironmentVariable("{env_var}", "<your-key>", "User")'
        )
        lines.append(f'  current shell only: $env:{env_var} = "<your-key>"')
        lines.append("  (open a NEW terminal after persisting)")
    else:
        lines.append(f'  persist: add to ~/.bashrc / ~/.zshrc:  export {env_var}="<your-key>"')
        lines.append(f'  current shell only: export {env_var}="<your-key>"')
    return "\n".join(lines)


def _run_interactive(args: argparse.Namespace) -> int:
    existing = _current_user_settings()
    print("Polaris setup — first-run configuration\n")
    print(f"Non-secret settings are written to {user_settings_path()};")
    print("the API key is never written to a file.\n")

    provider = _choose_provider(args, existing)
    spec = _SPECS_BY_NAME[provider]
    model = _choose_model(args, spec, existing)
    base_url = _choose_base_url(args, spec, existing)
    api_key = _prompt_api_key(spec, args)

    if api_key:
        _store_key(spec, api_key)
    elif credentials.provider_credential(provider):
        print(f"Keeping the existing credential for provider {provider!r}.")
    else:
        print(f"\nNo API key entered. {_env_instructions(spec)}")

    settings: dict[str, Any] = {"provider": provider}
    if model:
        settings["model"] = model
    if base_url:
        settings["base_url"] = base_url
    path = write_user_settings(settings)
    print(f"\nSaved provider={provider}" + (f", model={model}" if model else "") +
          (f", base_url={base_url}" if base_url else "") + f" to {path}")

    verify = bool(getattr(args, "verify", False))
    if not verify:
        answer = input("Run a quick connectivity check against the provider now? [Y/n] ").strip().lower()
        verify = answer in {"", "y", "yes"}
    if verify:
        verify_connection(provider, model, base_url)
    print('\nSetup complete. Start a session with `polaris` or `polaris run "<task>"`.')
    return 0


def _choose_provider(args: argparse.Namespace, existing: dict[str, Any]) -> str:
    """The --provider flag wins; otherwise a numbered menu with a sane default."""
    if getattr(args, "provider", None):
        return str(args.provider)
    current = str(existing.get("provider") or "")
    default = current if current in _SPECS_BY_NAME else "claude"
    print("Choose a provider:")
    for index, spec in enumerate(PROVIDER_SPECS, 1):
        marker = " (default)" if spec.name == default else ""
        print(f"  {index}) {spec.name:<14}{spec.description}{marker}")
    while True:
        raw = input(f"Selection [1-{len(PROVIDER_SPECS)} or name; default {default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(PROVIDER_SPECS):
            return PROVIDER_SPECS[int(raw) - 1].name
        if raw in _SPECS_BY_NAME:
            return raw
        print(f"Unknown selection {raw!r}; enter a number 1-{len(PROVIDER_SPECS)} or a provider name.")


def _choose_model(
    args: argparse.Namespace, spec: ProviderSpec, existing: dict[str, Any]
) -> str | None:
    if getattr(args, "model", None):
        return str(args.model)
    default = None
    if existing.get("provider") == spec.name and existing.get("model"):
        default = str(existing["model"])
    default = default or spec.default_model
    prompt = f"Model [{default}]: " if default else "Model id (e.g. deepseek-chat): "
    model = input(prompt).strip() or default
    if not model:
        print("No model set; pass --model or set AGENT_MODEL before running.")
    return model


def _choose_base_url(
    args: argparse.Namespace, spec: ProviderSpec, existing: dict[str, Any]
) -> str | None:
    if getattr(args, "base_url", None):
        return str(args.base_url).rstrip("/")
    if not spec.needs_base_url:
        return None
    default = os.getenv("OPENAI_COMPAT_BASE_URL") or (
        str(existing["base_url"])
        if existing.get("provider") == spec.name and existing.get("base_url")
        else None
    )
    prompt = (
        f"Base URL [{default}]: "
        if default
        else "Base URL (e.g. http://localhost:8000 or https://api.deepseek.com): "
    )
    base_url = (input(prompt).strip() or default or "").rstrip("/")
    if not base_url:
        print("No base URL set; the provider default (https://api.openai.com) will be used.")
        return None
    return base_url


def _prompt_api_key(spec: ProviderSpec, args: argparse.Namespace) -> str:
    """Hidden key entry; empty keeps an existing credential or defers to the env var."""
    api_key_env = getattr(args, "api_key_env", None)
    if api_key_env:
        if os.getenv(api_key_env):
            print(f"Using the API key already set in {api_key_env}.")
            return ""
        print(f"[warning] {api_key_env} is not set in this shell.")
    if spec.key_url:
        print(f"Get an API key at {spec.key_url}")
    hint = (
        "leave empty to keep the existing credential"
        if credentials.provider_credential(spec.name)
        else "leave empty to configure it later"
    )
    return getpass.getpass(f"{spec.env_var} (input hidden; {hint}): ").strip()


def _store_key(spec: ProviderSpec, api_key: str) -> None:
    """Offer the OS credential store; on decline/unavailability print env instructions."""
    if credentials.keyring_available():
        answer = input(
            f"Store {spec.env_var} in the OS credential store "
            f"(service {credentials.KEYRING_SERVICE!r})? [Y/n] "
        ).strip().lower()
        if answer in {"", "y", "yes"}:
            try:
                credentials.store_api_key(spec.env_var, api_key)
            except RuntimeError as exc:
                print(f"[warning] could not store the key: {exc}")
            else:
                print(f"Stored {spec.env_var} in the OS credential store.")
                return
    else:
        print("(Install the credentials extra — pip install 'agent-with-llm[credentials]' —")
        print(" to store keys in the OS credential store instead.)")
    print(_env_instructions(spec))


def _run_non_interactive(args: argparse.Namespace) -> int:
    provider = getattr(args, "provider", None)
    if not provider or provider not in _SPECS_BY_NAME:
        print("polaris setup is not prompting (non-interactive input detected).\n")
        print("Configure from flags instead, for example:")
        print("  polaris setup --provider claude --model claude-opus-4-8 --api-key-env ANTHROPIC_API_KEY")
        print(
            "  polaris setup --provider openai-compat --model <id> --base-url http://localhost:8000 "
            "--api-key-env OPENAI_COMPAT_API_KEY"
        )
        print("\nAPI keys come from the environment or the OS credential store;")
        print("they are never written to files. Run `polaris setup` in a terminal for the wizard.")
        return 2
    spec = _SPECS_BY_NAME[provider]
    model = getattr(args, "model", None) or spec.default_model
    base_url = (getattr(args, "base_url", None) or "").rstrip("/") or None

    settings: dict[str, Any] = {"provider": provider}
    if model:
        settings["model"] = model
    if base_url:
        settings["base_url"] = base_url
    path = write_user_settings(settings)
    print(f"Saved provider={provider}" + (f", model={model}" if model else "") +
          (f", base_url={base_url}" if base_url else "") + f" to {path}")
    if spec.needs_base_url and not base_url and not os.getenv("OPENAI_COMPAT_BASE_URL"):
        print("[warning] no base URL configured; the provider default (https://api.openai.com) will be used.")

    api_key_env = getattr(args, "api_key_env", None)
    if api_key_env:
        if credentials.get_api_key(api_key_env):
            print(f"{api_key_env} resolves — credential ready.")
        else:
            print(f"[warning] {api_key_env} does not resolve in this environment; set it before running polaris.")
    elif not credentials.provider_credential(provider):
        print(_env_instructions(spec))
    else:
        print(f"A credential for provider {provider!r} already resolves.")

    if getattr(args, "verify", False):
        verify_connection(provider, model, base_url)
    return 0
