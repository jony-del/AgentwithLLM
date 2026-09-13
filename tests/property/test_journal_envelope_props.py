"""P0-1 (audit): fuzzed recovery journals can never mutate anything outside the
trusted recovery scope.

Each example builds a real checksummed journal in a fresh ``JournalStorage.local``
scope, then applies a drawn mutation: truncation, byte flips, field retyping/drops
(unsigned), reordered/duplicated lines, random bytes/JSON — or a *re-signed* forgery
written through the real journal API whose ``overlay`` points at a victim directory
outside the workspace, whose ``workspace`` lies, or whose ``changed`` list carries
traversal/drive/UNC/device paths.

The safety property is universal: after ``TurnExecutionJournal.recover_all(...,
dry_run=False)`` every byte outside the journal root is identical, the victim is
intact, and invalid envelopes end in ``rejected``/``foreign`` — never a destructive
action.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal

_CASES = itertools.count()

_fields = ["v", "sequence", "state", "turn_id", "owner", "checksum", "previous_checksum", "ts"]
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-10, 10),
    st.text(max_size=10),
    st.lists(st.integers(-5, 5), max_size=3),
    st.just({}),
)
_json_value = st.recursive(
    _json_scalar,
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(min_size=1, max_size=5), children, max_size=3),
    max_leaves=8,
)

# Hostile ``changed`` entries (subset of the path-containment strategy in
# test_path_containment_props.py) plus plausible in-workspace names that are never
# actually created, and non-string junk.
_evil_relative = st.sampled_from(
    [
        "../victim/keep.txt",
        "../../victim",
        "/etc/passwd",
        "C:/victim/keep.txt",
        "c:/windows/system32/drivers/etc/hosts",
        "//server/share/keep.txt",
        "\\\\server\\share\\keep.txt",
        "//?/C:/victim",
        "NUL",
        "dir/CON.txt",
        "a/../../victim/keep.txt",
        "sub\\..\\..\\victim",
        "x.txt.",
        "x.txt ",
        "a//b.txt",
        "./x.txt",
        "..",
    ]
)
_changed_entry = st.one_of(
    _evil_relative,
    st.from_regex(r"f[0-9]{1,2}\.txt", fullmatch=True),
    st.just(5),
    st.just(None),
)


@st.composite
def _mutation(draw: st.DrawFn) -> tuple[str, object]:
    kind = draw(
        st.sampled_from(
            [
                "none",
                "truncate",
                "flip",
                "retype",
                "drop_field",
                "reorder",
                "duplicate",
                "random_bytes",
                "random_json",
                "evil_overlay",
                "evil_workspace",
                "evil_changed",
            ]
        )
    )
    if kind == "truncate":
        return kind, draw(st.floats(0.05, 0.95))
    if kind == "flip":
        return kind, (draw(st.integers(0, 10**9)), draw(st.integers(1, 255)))
    if kind == "retype":
        return kind, (draw(st.sampled_from(_fields + ["overlay"])), draw(_json_scalar))
    if kind == "drop_field":
        return kind, draw(st.sampled_from(_fields))
    if kind == "random_bytes":
        return kind, draw(st.binary(min_size=1, max_size=256))
    if kind == "random_json":
        return kind, draw(st.lists(_json_value, min_size=1, max_size=4))
    if kind == "evil_overlay":
        return kind, draw(
            st.sampled_from(
                ["victim", "victim_file", "workspace", "drive", "unc", "relative", "empty", "slash", "double_dot"]
            )
        )
    if kind == "evil_workspace":
        return kind, draw(st.sampled_from(["victim", "elsewhere", "drive", "empty"]))
    if kind == "evil_changed":
        changed = draw(st.lists(_changed_entry, min_size=1, max_size=3))
        existed_kind = draw(st.sampled_from(["matching", "empty", "wrong_type", "not_a_dict"]))
        if existed_kind == "matching":
            existed = {str(item): draw(st.booleans()) for item in changed}
        elif existed_kind == "empty":
            existed = {}
        elif existed_kind == "wrong_type":
            existed = {str(item): "yes" for item in changed}
        else:
            existed = "not-a-dict"
        return kind, (changed, existed)
    return kind, None


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _baseline_journal(storage: JournalStorage) -> TurnExecutionJournal:
    """A real, checksummed, unfinished journal: telemetry + transaction_opened."""

    journal = TurnExecutionJournal(storage)
    overlay = journal.create_overlay(journal.turn_id)
    journal.record("discovered", ordinal=0, tool_name="write_text_file")
    journal.record(
        "transaction_opened", overlay=str(overlay), workspace=str(storage.workspace)
    )
    journal._release_for_later_recovery()
    return journal


def _evil_overlay_value(choice: str, victim: Path, workspace: Path) -> str:
    return {
        "victim": str(victim),
        "victim_file": str(victim / "keep.txt"),
        "workspace": str(workspace),
        "drive": "C:/polaris-victim",
        "unc": "//server/share/polaris-victim",
        "relative": "overlay",
        "empty": "",
        "slash": "/",
        "double_dot": str(victim.parent / ".." / "victim"),
    }[choice]


def _evil_workspace_value(choice: str, victim: Path, workspace: Path) -> str:
    return {
        "victim": str(victim),
        "elsewhere": str(workspace.parent / "elsewhere"),
        "drive": "C:/elsewhere",
        "empty": "",
    }[choice]


def _write_mutated_journal(
    storage: JournalStorage, workspace: Path, victim: Path, mutation: tuple[str, object]
) -> bool:
    """Write the journal for ``mutation``; return True when bytes stayed valid."""

    kind, value = mutation
    if kind in {"evil_overlay", "evil_workspace", "evil_changed"}:
        # Forgery with a valid checksum chain, written through the real record API.
        journal = TurnExecutionJournal(storage)
        overlay = journal.create_overlay(journal.turn_id)
        overlay_value: str = str(overlay)
        workspace_value: str = str(storage.workspace)
        if kind == "evil_overlay":
            overlay_value = _evil_overlay_value(str(value), victim, workspace)
        elif kind == "evil_workspace":
            workspace_value = _evil_workspace_value(str(value), victim, workspace)
        journal.record("transaction_opened", overlay=overlay_value, workspace=workspace_value)
        if kind == "evil_changed":
            changed, existed = value  # type: ignore[misc]
            journal.record("commit_started", overlay=overlay_value, changed=changed, existed=existed)
        journal._release_for_later_recovery()
        return True

    journal = _baseline_journal(storage)
    raw = journal.path.read_bytes()
    if kind == "none":
        return True
    if kind == "truncate":
        cut = max(1, int(len(raw) * float(value)))
        journal.path.write_bytes(raw[:cut])
        return raw[:cut] == raw
    if kind == "flip":
        index, mask = value  # type: ignore[misc]
        mutated = bytearray(raw)
        mutated[index % len(mutated)] ^= mask
        journal.path.write_bytes(bytes(mutated))
        return False
    if kind == "random_bytes":
        journal.path.write_bytes(value)  # type: ignore[arg-type]
        return False
    if kind == "random_json":
        text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in value)  # type: ignore[union-attr]
        journal.path.write_text(text, encoding="utf-8")
        return False
    lines = raw.decode("utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    if kind == "reorder":
        records = list(reversed(records))
    elif kind == "duplicate":
        records = records + [records[-1]]
    elif kind == "retype":
        field, replacement = value  # type: ignore[misc]
        records[0][field] = replacement
    elif kind == "drop_field":
        records[0].pop(str(value), None)
    else:  # pragma: no cover - strategy exhaustiveness guard
        raise AssertionError(f"unknown mutation {kind}")
    mutated_text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    journal.path.write_text(mutated_text, encoding="utf-8")
    return mutated_text.encode("utf-8") == raw


_FORGED_KINDS = {"evil_overlay", "evil_workspace"}
_RAW_CORRUPTION_KINDS = {"flip", "drop_field", "reorder", "duplicate", "random_bytes", "random_json"}


@given(data=st.data())
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_recovery_never_mutates_outside_journal_root(tmp_path: Path, data: st.DataObject) -> None:
    case = tmp_path / f"case-{next(_CASES)}"
    workspace = case / "workspace"
    workspace.mkdir(parents=True)
    victim = case / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("safe", encoding="utf-8")
    storage = JournalStorage.local(case / "journals", workspace=workspace)

    mutation = data.draw(_mutation())
    stayed_valid = _write_mutated_journal(storage, workspace, victim, mutation)

    before = _snapshot(case)
    events: list[dict] = []
    outcomes = TurnExecutionJournal.recover_all(
        storage, dry_run=False, audit_writer=events.append
    )
    after = _snapshot(case)

    # The universal safety property: nothing outside the journal root changed.
    outside_before = {k: v for k, v in before.items() if not k.startswith("journals/")}
    outside_after = {k: v for k, v in after.items() if not k.startswith("journals/")}
    assert outside_after == outside_before
    assert (victim / "keep.txt").read_bytes() == b"safe"

    assert len(outcomes) == 1
    status = outcomes[0]["status"]
    assert status in {"rejected", "foreign", "rolled_back", "recovery_failed"}
    kind = mutation[0]
    if kind == "none":
        assert status == "rolled_back"
    elif kind in _FORGED_KINDS or (kind in _RAW_CORRUPTION_KINDS and not stayed_valid):
        assert status in {"rejected", "foreign"}
    elif kind == "retype" and not stayed_valid:
        assert status in {"rejected", "foreign"}
    elif kind in {"truncate", "evil_changed"} or (kind == "retype" and stayed_valid):
        # A line-boundary truncation or a harmless retype yields a valid journal whose
        # recovery only removes the in-scope overlay.
        assert status in {"rejected", "foreign", "rolled_back"}
    assert len(events) >= 1
