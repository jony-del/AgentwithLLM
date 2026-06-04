import sys

from agent_core.cli import _force_utf8_output


class _RecordingStream:
    """Stand-in for a TextIO stream that records reconfigure() calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _PlainStream:
    """A stream without reconfigure(), like a pipe wrapper or a test capture."""


def test_force_utf8_output_reconfigures_streams(monkeypatch) -> None:
    out, err = _RecordingStream(), _RecordingStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    _force_utf8_output()

    expected = {"encoding": "utf-8", "errors": "replace"}
    assert out.calls == [expected]
    assert err.calls == [expected]


def test_force_utf8_output_ignores_streams_without_reconfigure(monkeypatch) -> None:
    # Streams lacking reconfigure() (captured/redirected) must not raise.
    monkeypatch.setattr(sys, "stdout", _PlainStream())
    monkeypatch.setattr(sys, "stderr", _PlainStream())

    _force_utf8_output()  # should be a no-op, not an AttributeError
