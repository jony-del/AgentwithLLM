import json
import sys

from agent_core.cli import _clean_surrogates, _force_utf8_output, _make_provider
from agent_core.providers import ClaudeProvider, FakeProvider, OpenAICompatProvider, OpenAIResponsesProvider
from agent_core.storage import JSONLRunLogger


def test_make_provider_routes_explicit_protocols() -> None:
    assert isinstance(_make_provider({"provider": "claude"}), ClaudeProvider)
    assert isinstance(_make_provider({"provider": "openai"}), OpenAIResponsesProvider)
    assert isinstance(_make_provider({"provider": "openai-compat"}), OpenAICompatProvider)
    assert isinstance(_make_provider({"provider": "fake"}), FakeProvider)


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


def test_clean_surrogates_strips_lone_surrogates() -> None:
    # A lone surrogate (born from non-TTY stdin's surrogateescape decode) must be
    # collapsed so the result re-encodes to UTF-8 without raising.
    cleaned = _clean_surrogates("hello\udcbfworld")
    assert all(not (0xDC80 <= ord(c) <= 0xDCFF) for c in cleaned)
    cleaned.encode("utf-8")  # would raise on a surviving surrogate
    assert _clean_surrogates("plain ascii") == "plain ascii"


def test_run_logger_write_survives_surrogates(tmp_path) -> None:
    # A surrogate-laden payload must never crash the JSONL append, and the file
    # must stay valid (each line parseable as JSON).
    logger = JSONLRunLogger(run_dir=tmp_path, run_id="surrogate")
    logger._write_sync({"event": "user", "content": "bad\udcbfbyte"})

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "user"
