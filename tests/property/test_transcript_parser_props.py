"""P2-N2 (audit): the transcript loader skips malformed lines instead of crashing.

``load_transcript()`` documents "Malformed lines are skipped"; ``_Accumulator.feed()``
is the single funnel for that contract. These properties fuzz it with legal JSON of
the wrong shape (``[]``, ``"str"``), truncated JSON, relink records missing ``uuid``,
wrong field types, and arbitrary text — none may raise, and every well-formed
``message`` record interleaved with the garbage must still be accepted.

Deep JSON and invalid UTF-8 regressions are also pinned below.
"""

from __future__ import annotations

import json
import hashlib
import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core.models import Message
from agent_core.transcript import (
    SKIP_PRECOMPACT_THRESHOLD, TranscriptStore, _Accumulator, build_chain, load_transcript,
)

# Bounded-depth JSON: the known RecursionError gap is exercised separately below.
_JSON_SCALARS = st.none() | st.booleans() | st.integers(-1000, 1000) | st.floats(
    allow_nan=False, allow_infinity=False, width=32
) | st.text(string.printable, max_size=20)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(string.ascii_lowercase, min_size=1, max_size=6), children, max_size=4),
    max_leaves=15,
)

_VALID_ROLES = ["system", "user", "assistant", "tool"]


def _message_line(uuid: str, role: str, content: str) -> str:
    record = {"type": "message", **Message(role, content).to_dict()}
    record["uuid"] = uuid
    record["message_id"] = uuid
    return json.dumps(record, ensure_ascii=False)


def _truncated_line() -> st.SearchStrategy[str]:
    """A valid message record cut strictly before its end: never valid JSON."""

    base = _message_line("truncated-victim", "user", "content that gets cut off")
    return st.integers(1, len(base) - 2).map(lambda cut: base[:cut])


def _never_an_accepted_message(line: str) -> bool:
    """Keep fuzzed lines from accidentally passing full message validation."""

    try:
        value = json.loads(line)
    except ValueError:
        return True
    return not (
        isinstance(value, dict)
        and value.get("type") == "message"
        and value.get("role") in _VALID_ROLES
    )


_malformed_line = st.one_of(
    # Legal JSON that is not an object at all.
    st.lists(_JSON_VALUES, max_size=3).map(lambda v: json.dumps(v, ensure_ascii=False)),
    st.text(string.printable, min_size=1, max_size=20).map(
        lambda s: json.dumps(s, ensure_ascii=False)
    ),
    st.integers().map(str),
    st.sampled_from(["null", "true", "false", "3.14", '"str"', "[]", "{}"]),
    # Truncated JSON bytes.
    _truncated_line(),
    # relink without a uuid (audit P2-N2 named this crash shape).
    st.just('{"type": "relink", "parent_uuid": null}'),
    st.just('{"type": "relink"}'),
    st.just('{"type": "relink", "uuid": 42, "parent_uuid": "x"}'),
    # Wrong field types on otherwise plausible records.
    st.just('{"type": "message", "role": 5, "content": "x"}'),
    st.just('{"type": "message", "role": "user", "content": ["not", "str"]}'),
    st.just('{"type": "message", "role": "user", "content": "x", "uuid": ""}'),
    st.just('{"type": "message", "role": "user", "content": "x", "parent_uuid": 7}'),
    st.just('{"type": "message", "role": "user", "content": "x", "version": true}'),
    st.just('{"type": 5}'),
    st.just('{"v": "four", "type": "message", "role": "user", "content": "x"}'),
    st.just('{"v": 999, "type": "message", "role": "user", "content": "x"}'),
    st.just('{"type": "tool_round", "messages": "not-a-list", "execution_manifest": {}}'),
    st.just('{"type": "session"}'),
    st.just('{"type": "custom-title", "title": [1]}'),
    st.just('{"type": "tag", "tag": {}}'),
    # Arbitrary small dicts (never a "type" key, so never an accepted record) and
    # plain text filtered to never spell a fully valid message record.
    st.dictionaries(
        st.text(string.ascii_lowercase, min_size=1, max_size=6).filter(lambda k: k != "type"),
        _JSON_VALUES,
        max_size=4,
    ).map(lambda d: json.dumps(d, ensure_ascii=False)),
    st.text(string.ascii_letters + string.digits + " \t{}[]\",:", max_size=60).filter(
        _never_an_accepted_message
    ),
)


@given(lines=st.lists(_malformed_line, max_size=30))
def test_feed_never_raises_on_malformed_lines(lines: list[str]) -> None:
    acc = _Accumulator("session-prop")
    for number, line in enumerate(lines, start=1):
        acc.feed(line, number)
    loaded = acc.finish(Path("property.jsonl"))
    assert all(isinstance(m, Message) for m in loaded.messages.values())


@given(data=st.data())
def test_valid_messages_survive_interleaved_corruption(data: st.DataObject) -> None:
    count = data.draw(st.integers(1, 8))
    expected = {}
    valid_lines = []
    for index in range(count):
        role = data.draw(st.sampled_from(_VALID_ROLES))
        content = data.draw(st.text(string.printable, max_size=40))
        uuid = f"prop-uuid-{index}"
        expected[uuid] = (role, content)
        valid_lines.append(_message_line(uuid, role, content))
    malformed = data.draw(st.lists(_malformed_line, max_size=15))
    # Interleave: garbage before, between, and after the valid records.
    lines = []
    for line in valid_lines:
        lines.append(line)
        if malformed:
            lines.append(malformed.pop(0))
    lines.extend(malformed)

    acc = _Accumulator("session-prop")
    for number, line in enumerate(lines, start=1):
        acc.feed(line, number)
    loaded = acc.finish(Path("property.jsonl"))

    assert set(loaded.messages) == set(expected)
    for uuid, (role, content) in expected.items():
        message = loaded.messages[uuid]
        assert message.role == role
        assert message.content == content


_FILE_COUNTER = 0


@given(data=st.data())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_load_transcript_end_to_end_skips_garbage(data: st.DataObject, tmp_path: Path) -> None:
    global _FILE_COUNTER
    count = data.draw(st.integers(1, 5))
    uuids = []
    lines = []
    for index in range(count):
        uuid = f"file-uuid-{index}"
        uuids.append(uuid)
        content = data.draw(st.text(string.printable, max_size=30))
        lines.append(_message_line(uuid, "user", content))
    garbage = data.draw(st.lists(_malformed_line, max_size=10))
    lines.extend(garbage)
    _FILE_COUNTER += 1
    path = tmp_path / f"session-{_FILE_COUNTER}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = load_transcript(path, skip_precompact=False)

    assert set(loaded.messages) == set(uuids)


# --- malformed-byte and depth regression cases ------------------------------------


def test_deeply_nested_json_line_is_skipped_not_raised() -> None:
    acc = _Accumulator("session-deep")
    acc.feed("[" * 20000 + "]" * 20000, 1)
    assert [item.code for item in acc.diagnostics] == ["invalid_json"]


def test_invalid_utf8_bytes_are_skipped_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "session-bytes.jsonl"
    path.write_bytes(
        b'{"type": "message", "role": "user", "content": "hi"}\n' + b"\xff\xfebinary\n"
    )
    loaded = load_transcript(path, skip_precompact=False)
    assert len(loaded.messages) == 1
    assert [item.code for item in loaded.diagnostics] == ["invalid_utf8"]


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("damage", ["utf8", "depth", "checksum", "message_shape"])
async def test_corrupt_new_snapshot_keeps_previous_chain(tmp_path: Path, large: bool, damage: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = TranscriptStore(tmp_path / "sessions", workspace, "corrupt-snapshot")
    try:
        # Force the optimized reader as well as the ordinary reader.
        if large:
            await store.append_message(Message("user", "x" * (SKIP_PRECOMPACT_THRESHOLD + 1)))
        first = Message("user", "surviving snapshot", uuid="first")
        await store.append_compaction_snapshot([first], source_head=None)
        if damage == "utf8":
            raw = b'{"type": "compaction_snapshot", "messages": [], "bad": "\xff"}\n'
        elif damage == "depth":
            raw = b'{"type": "compaction_snapshot", "messages": ' + b"[" * 20000 + b"]" * 20000 + b"}\n"
        else:
            body = {"messages": [{"role": [], "content": "wrong"}], "source_head": "first"}
            digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
            raw = (json.dumps({"type": "compaction_snapshot", **body,
                               "checksum": digest if damage == "message_shape" else "wrong"}) + "\n").encode()
        with store.path.open("ab") as handle:
            handle.write(raw)
            # Invalid bytes inside otherwise valid JSON must not be repaired into a message.
            handle.write(b'{"type":"message","role":"user","content":"bad\xff"}\n')
        await store.append_message(Message("assistant", "after corruption", uuid="last", parent_uuid="first"))
    finally:
        store.close()
    loaded = load_transcript(store.path)
    assert [m.uuid for m in build_chain(loaded)] == ["first", "last"]
    codes = [item.code for item in loaded.diagnostics]
    # Each damage mode must be diagnosed by name (message_shape also carries the
    # per-message detail alongside the aggregate invalid_snapshot_message), and
    # the trailing bad-byte line is always skipped with its own diagnostic.
    expected_code = {
        "utf8": "invalid_utf8",
        "depth": "invalid_json",
        "checksum": "checksum_mismatch",
        "message_shape": "invalid_snapshot_message",
    }[damage]
    assert expected_code in codes
    assert "invalid_utf8" in codes
    assert len(loaded.diagnostics) <= 3
    assert [m.to_dict() for m in build_chain(load_transcript(store.path, skip_precompact=False))] == [
        m.to_dict() for m in build_chain(loaded)
    ]
