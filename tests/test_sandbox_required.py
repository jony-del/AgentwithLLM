"""D3 enforcement: unattended permission modes require a working sandbox.

``auto`` / ``dontask`` / ``bypass`` execute tools without per-call confirmation, so
constructing an agent in one of those modes with no sandbox must refuse (fail-closed),
offer an interactive confirmation when a live UI exists, and honor only the explicit
opt-outs (config flag / env var). The suite-wide opt-out lives in ``conftest.py``;
these tests remove it to exercise the real behavior.
"""

from pathlib import Path

import pytest

from agent_core.providers.fake import FakeProvider
from agent_core.permissions import PermissionMode
from agent_core.react import ReActAgent, ReActConfig
from agent_core.sandbox import SandboxConfig, SandboxRequiredError
from agent_core.storage import read_events
from agent_core.ui import NullUI


def _config(tmp_path: Path, **overrides) -> ReActConfig:
    base = dict(
        run_dir=str(tmp_path),
        project_instructions=False,
        git_context=False,
        session_dir="",
    )
    base.update(overrides)
    return ReActConfig(**base)


@pytest.fixture()
def no_optout(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SANDBOX_ALLOW_UNATTENDED", raising=False)


@pytest.mark.parametrize("mode", ["auto", "dontask", "bypass"])
def test_unattended_mode_without_sandbox_refuses(no_optout, tmp_path: Path, mode: str) -> None:
    with pytest.raises(SandboxRequiredError):
        ReActAgent(FakeProvider(), _config(tmp_path, permission=mode))


def test_attended_modes_are_unaffected(no_optout, tmp_path: Path) -> None:
    for mode in ("default", "acceptedits", "plan"):
        ReActAgent(FakeProvider(), _config(tmp_path, permission=mode))


def test_config_opt_out_allows_construction(no_optout, tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        permission="auto",
        sandbox=SandboxConfig(allow_unattended_unsandboxed=True),
    )
    ReActAgent(FakeProvider(), config)  # must not raise


def test_env_opt_out_allows_construction(no_optout, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_SANDBOX_ALLOW_UNATTENDED", "1")
    ReActAgent(FakeProvider(), _config(tmp_path, permission="auto"))  # must not raise


class _ConfirmingUI(NullUI):
    is_live = True

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def confirm_action(self, message: str) -> bool:
        self.asked.append(message)
        return self.answer


def test_interactive_confirmation_allows(no_optout, tmp_path: Path) -> None:
    ui = _ConfirmingUI(answer=True)
    ReActAgent(FakeProvider(), _config(tmp_path, permission="auto"), ui=ui)
    assert ui.asked and "without per-call confirmation" in ui.asked[0]


def test_interactive_decline_refuses(no_optout, tmp_path: Path) -> None:
    ui = _ConfirmingUI(answer=False)
    with pytest.raises(SandboxRequiredError):
        ReActAgent(FakeProvider(), _config(tmp_path, permission="auto"), ui=ui)


def test_runtime_switch_confirms_once_and_updates_live_policy(no_optout, tmp_path: Path) -> None:
    ui = _ConfirmingUI(answer=True)
    agent = ReActAgent(FakeProvider(), _config(tmp_path), ui=ui)

    agent.set_permission_mode("auto", source="test")
    assert agent.config.permission is PermissionMode.AUTO
    assert agent.permissions.mode is PermissionMode.AUTO
    assert len(ui.asked) == 1

    agent.set_permission_mode("dontask", source="test")
    assert agent.permissions.mode is PermissionMode.DONTASK
    assert len(ui.asked) == 1  # session acknowledgement is reused
    switches = [event for event in read_events(agent.logger.path) if event["event"] == "permission_mode"]
    assert [(event["from"], event["to"]) for event in switches] == [
        ("default", "auto"),
        ("auto", "dontask"),
    ]


def test_runtime_switch_decline_keeps_previous_mode(no_optout, tmp_path: Path) -> None:
    ui = _ConfirmingUI(answer=False)
    agent = ReActAgent(FakeProvider(), _config(tmp_path), ui=ui)

    with pytest.raises(SandboxRequiredError):
        agent.set_permission_mode("auto", source="test")
    assert agent.config.permission is PermissionMode.DEFAULT
    assert agent.permissions.mode is PermissionMode.DEFAULT


def test_runtime_switch_survives_audit_write_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    agent = ReActAgent(FakeProvider(), _config(tmp_path))

    def fail(_event, _payload):
        raise PermissionError("read-only log")

    monkeypatch.setattr(agent.logger, "write_nowait", fail)
    agent.set_permission_mode("acceptedits", source="test")

    assert agent.permissions.mode is PermissionMode.ACCEPTEDITS
    assert "could not write permission_mode audit event" in caplog.text
