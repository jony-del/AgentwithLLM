"""Setup wizard flow, settings.toml writing, and the first-run credential gate."""

from __future__ import annotations

import argparse
import builtins
import getpass
import os
import sys
import types

import pytest

from agent_core import cli, credentials, setup_wizard
from agent_core.config import resolve_config, write_user_settings
from agent_core.providers import ClaudeProvider

_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_COMPAT_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_COMPAT_BASE_URL",
    "AGENT_PROVIDER",
    "AGENT_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Hermetic credentials + a tmp user-settings path for every test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POLARIS_SETTINGS_PATH", str(tmp_path / "settings.toml"))
    credentials.clear_cache()
    yield
    credentials.clear_cache()


class _FakeTTY:
    """Wraps the captured stream so isatty() is True but output stays visible."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    def isatty(self) -> bool:
        return True

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)


def _make_interactive(monkeypatch, answers: list[str], keys: list[str] | None = None) -> None:
    """Force the wizard's TTY check on and script input()/getpass() answers."""
    monkeypatch.setattr(sys, "stdin", _FakeTTY(sys.stdin))
    monkeypatch.setattr(sys, "stdout", _FakeTTY(sys.stdout))
    prompts = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(prompts))
    key_prompts = iter(keys or [])
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(key_prompts))


def _install_fake_keyring(monkeypatch) -> types.SimpleNamespace:
    monkeypatch.delenv("POLARIS_DISABLE_KEYRING", raising=False)
    state: dict = {}

    fake = types.SimpleNamespace()
    fake.get_password = lambda service, account: state.get((service, account))

    def set_password(service: str, account: str, secret: str) -> None:
        state[(service, account)] = secret

    fake.set_password = set_password
    fake.store = state
    monkeypatch.setitem(sys.modules, "keyring", fake)
    credentials.clear_cache()
    return fake


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        provider=None,
        model=None,
        base_url=None,
        api_key_env=None,
        verify=False,
        non_interactive=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _settings_path(tmp_path):
    return tmp_path / "settings.toml"


# --- interactive flows -------------------------------------------------------


def test_interactive_claude_flow_stores_key_in_keyring(tmp_path, monkeypatch, capsys) -> None:
    fake = _install_fake_keyring(monkeypatch)
    # menu "" (default claude), model "" (default), store "" (default yes), verify "n"
    _make_interactive(monkeypatch, ["", "", "", "n"], keys=["sk-ant-test"])

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 0
    assert fake.store[("polaris", "ANTHROPIC_API_KEY")] == "sk-ant-test"
    text = _settings_path(tmp_path).read_text(encoding="utf-8")
    assert 'provider = "claude"' in text
    assert 'model = "claude-opus-4-8"' in text
    assert "sk-ant-test" not in text  # the key never touches the settings file
    out = capsys.readouterr().out
    assert "Stored ANTHROPIC_API_KEY in the OS credential store" in out
    assert "Setup complete" in out


def test_interactive_openai_compat_flow_without_keyring(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)  # extra not installed
    _make_interactive(
        monkeypatch,
        ["3", "local-7b", "http://localhost:8000/", "n"],  # menu, model, base URL, verify
        keys=["compat-key"],
    )

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 0
    text = _settings_path(tmp_path).read_text(encoding="utf-8")
    assert 'provider = "openai-compat"' in text
    assert 'model = "local-7b"' in text
    assert 'base_url = "http://localhost:8000"' in text  # trailing slash stripped
    assert "compat-key" not in text
    out = capsys.readouterr().out
    assert "OPENAI_COMPAT_API_KEY" in out
    if os.name == "nt":
        assert "SetEnvironmentVariable" in out
    else:
        assert 'export OPENAI_COMPAT_API_KEY' in out
    # The written settings drive the real config layering.
    values = resolve_config(
        {"model": None, "permission": None, "provider": None, "effort": None},
        config_file=tmp_path / "absent.toml",
        env_file=tmp_path / "absent.env",
    )
    assert values["provider"] == "openai-compat"
    assert values["model"] == "local-7b"
    assert values["base_url"] == "http://localhost:8000"


def test_interactive_empty_key_keeps_existing_env_credential(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-env-key")
    _make_interactive(monkeypatch, ["", "", "n"], keys=[""])  # no store prompt without a key

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 0
    assert "Keeping the existing credential" in capsys.readouterr().out
    assert 'provider = "claude"' in _settings_path(tmp_path).read_text(encoding="utf-8")


def test_interactive_keyring_store_failure_falls_back_to_env_instructions(
    tmp_path, monkeypatch, capsys
) -> None:
    fake = _install_fake_keyring(monkeypatch)

    def rejecting_store(service: str, account: str, secret: str) -> None:
        raise RuntimeError("backend locked")

    fake.set_password = rejecting_store
    _make_interactive(monkeypatch, ["", "", "", "n"], keys=["sk-ant-test"])

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "could not store the key" in out
    assert "ANTHROPIC_API_KEY" in out  # env-var instructions printed instead
    assert 'provider = "claude"' in _settings_path(tmp_path).read_text(encoding="utf-8")


def test_interactive_eof_cancels_without_writing(tmp_path, monkeypatch, capsys) -> None:
    _make_interactive(monkeypatch, [])

    def raising_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", raising_input)

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 1
    assert "cancelled" in capsys.readouterr().out
    assert not _settings_path(tmp_path).exists()


# --- non-interactive flows ---------------------------------------------------


def test_non_interactive_flags_write_config_without_prompting(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-exported")

    def forbidden(prompt: str = "") -> str:
        raise AssertionError("non-interactive setup must never prompt")

    monkeypatch.setattr(builtins, "input", forbidden)
    monkeypatch.setattr(getpass, "getpass", forbidden)

    exit_code = setup_wizard.run_setup(
        _args(provider="claude", model="claude-test-1", api_key_env="ANTHROPIC_API_KEY")
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY resolves" in out
    text = _settings_path(tmp_path).read_text(encoding="utf-8")
    assert 'provider = "claude"' in text
    assert 'model = "claude-test-1"' in text
    assert "already-exported" not in text


def test_non_interactive_without_provider_guides_and_fails(tmp_path, monkeypatch, capsys) -> None:
    # No --non-interactive flag: piped stdin (not a TTY) triggers the same path.
    class _Pipe:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _Pipe())

    exit_code = setup_wizard.run_setup(_args())

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "not prompting" in out
    assert "--provider" in out
    assert not _settings_path(tmp_path).exists()


def test_settings_roundtrip_preserves_unrelated_content(tmp_path, monkeypatch) -> None:
    settings = _settings_path(tmp_path)
    settings.write_text(
        '# keep me\npermission = "plan"\n\n[output]\nmax_line_chars = 1000\n',
        encoding="utf-8",
    )

    exit_code = setup_wizard.run_setup(_args(provider="openai", model="gpt-x", non_interactive=True))

    assert exit_code == 0
    text = settings.read_text(encoding="utf-8")
    assert "# keep me" in text
    assert 'permission = "plan"' in text
    assert "max_line_chars = 1000" in text
    assert 'provider = "openai"' in text
    assert 'model = "gpt-x"' in text


def test_write_user_settings_rejects_secrets_and_non_scalars(tmp_path) -> None:
    with pytest.raises(ValueError, match="secret-looking"):
        write_user_settings({"api_key": "sk-nope"})
    with pytest.raises(ValueError, match="scalar"):
        write_user_settings({"model": ["not", "a", "scalar"]})


# --- connectivity check ------------------------------------------------------


def _stub_complete(result=None, error: Exception | None = None):
    async def complete(self, messages, tools, config, stream=None, should_cancel=None, scope=None):
        if error is not None:
            raise error
        return result

    return complete


def test_verify_success_does_not_fail_setup(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ClaudeProvider, "complete", _stub_complete(result=object()))

    exit_code = setup_wizard.run_setup(_args(provider="claude", model="m", verify=True, non_interactive=True))

    assert exit_code == 0
    assert "[verify] connection OK" in capsys.readouterr().out


def test_verify_failure_is_reported_but_setup_still_succeeds(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(ClaudeProvider, "complete", _stub_complete(error=RuntimeError("boom")))

    exit_code = setup_wizard.run_setup(_args(provider="claude", model="m", verify=True, non_interactive=True))

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[verify] connection check FAILED" in out
    assert "boom" in out


def test_verify_skipped_without_credential(tmp_path, monkeypatch, capsys) -> None:
    exit_code = setup_wizard.run_setup(_args(provider="claude", model="m", verify=True, non_interactive=True))

    assert exit_code == 0
    assert "[verify] skipped" in capsys.readouterr().out


# --- base_url wiring in _make_provider ----------------------------------------


def test_settings_base_url_applies_only_when_env_absent(tmp_path, monkeypatch) -> None:
    provider = cli._make_provider({"provider": "openai-compat", "base_url": "http://settings.example"})
    assert provider.base_url == "http://settings.example"

    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://env.example")
    provider = cli._make_provider({"provider": "openai-compat", "base_url": "http://settings.example"})
    assert provider.base_url == "https://env.example"  # env keeps its precedence

    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL")
    provider = cli._make_provider({"provider": "claude", "base_url": "https://settings.example"})
    assert provider.base_url == "https://settings.example"
    provider = cli._make_provider({"provider": "claude"})
    assert provider.base_url == "https://api.anthropic.com"  # built-in default intact


# --- first-run credential gate in build_agent ---------------------------------


def _agent_args(**overrides) -> argparse.Namespace:
    base = dict(
        model=None,
        permission=None,
        provider="claude",
        memory=None,
        quiet=True,
        no_stream=False,
        thinking_budget=None,
        max_api_concurrency=None,
        max_wall_seconds=None,
        max_steps=None,
        no_session_persistence=True,
        session_dir=None,
        resume=None,
        continue_=False,
        fork_session=False,
        session_id=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_gate_non_tty_fails_actionable(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        cli.build_agent(_agent_args())

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "polaris setup" in message


def test_gate_tty_decline_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _make_interactive(monkeypatch, ["n"])

    with pytest.raises(RuntimeError, match="polaris setup"):
        cli.build_agent(_agent_args())


def test_gate_tty_accept_runs_wizard_inline_and_continues(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bash = tmp_path / "bash.exe"
    bash.write_bytes(b"")
    monkeypatch.setenv("POLARIS_BASH_PATH", str(bash))
    _make_interactive(monkeypatch, ["y"])
    calls = []

    def fake_run_setup(args) -> int:
        calls.append(args)
        os.environ["ANTHROPIC_API_KEY"] = "fresh-key"
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", fake_run_setup)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    built = cli.build_agent(_agent_args())

    assert len(calls) == 1
    assert credentials.provider_credential("claude") == "fresh-key"
    assert built.agent.config.provider == "claude"


def test_gate_fake_provider_never_prompts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    bash = tmp_path / "bash.exe"
    bash.write_bytes(b"")
    monkeypatch.setenv("POLARIS_BASH_PATH", str(bash))

    def forbidden(prompt: str = "") -> str:
        raise AssertionError("fake provider must never trigger the credential prompt")

    monkeypatch.setattr(builtins, "input", forbidden)

    built = cli.build_agent(_agent_args(provider="fake"))
    assert built.agent.config.provider == "fake"


def test_run_command_missing_credential_exits_1_with_guidance(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["run", "hi", "--provider", "claude"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "polaris setup" in err
    assert "ANTHROPIC_API_KEY" in err
