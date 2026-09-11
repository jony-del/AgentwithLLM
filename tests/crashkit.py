"""Shared crash-injection harness for the second-round audit §9.1 matrix.

A *crash point* is a durable boundary: the injected fault fires only after the
boundary record has been fsync'd to the journal (or after the transcript append
has durably completed), mirroring a process that dies between "state is durable"
and "the next statement runs". In-process tests raise :class:`SimulatedCrash`
(a ``BaseException``, so no production ``except Exception`` can swallow it);
the process-level child passes ``crash=os._exit`` instead. Both layers then run
the same recovery and pin the same truth via :func:`assert_no_replay_and_truthful`.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agent_core.memory.config import MemoryConfig
from agent_core.models import ToolResult, ToolRisk
from agent_core.providers.fake import FakeProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.base import Tool
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.transaction import JournalStorage, TurnExecutionJournal
from agent_core.transcript import TranscriptStore, build_chain, load_transcript, project_dir

SESSION_ID = "session-crash"
PROMPT = "tool:counting_external once"
SIDE_EFFECT_LOG = "side_effects.log"


class SimulatedCrash(BaseException):
    """Process-death signal: deliberately not an ``Exception`` subclass."""


class CrashPoint(str, Enum):
    """The five durable boundaries from the audit §9.1 crash-injection matrix."""

    EXTERNAL_INTENT_COMMITTED = "external_intent_committed"
    EXTERNAL_OUTCOME_COMMITTED = "external_outcome_committed"
    HISTORY_READY_COMMITTED = "history_ready_committed"
    TRANSCRIPT_APPENDED = "transcript_appended"
    HISTORY_PERSISTED = "history_persisted"


_JOURNAL_STATE = {
    CrashPoint.EXTERNAL_INTENT_COMMITTED: "external_intent",
    CrashPoint.EXTERNAL_OUTCOME_COMMITTED: "external_outcome_committed",
    CrashPoint.HISTORY_READY_COMMITTED: "history_ready",
    CrashPoint.HISTORY_PERSISTED: "history_persisted",
}

# Points raised on the agent's main coroutine propagate out of ``agent.run``;
# points raised inside an executor task kill that task and leave the run
# stalled on a completion that can never arrive (which the test then cancels —
# exactly how a real crash leaves the caller).
MAIN_PATH_POINTS = frozenset(
    {
        CrashPoint.HISTORY_READY_COMMITTED,
        CrashPoint.TRANSCRIPT_APPENDED,
        CrashPoint.HISTORY_PERSISTED,
    }
)


class CountingExternalTool(Tool):
    """FINAL_ONLY external tool whose every side effect is one durable log line."""

    name = "counting_external"
    description = "record one external side effect"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "additionalProperties": False,
    }
    risk = ToolRisk.WRITE

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.calls = 0

    def _invoke(self, arguments: dict) -> ToolResult:
        self.calls += 1
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"effect {self.calls}\n")
        return ToolResult(self.name, f"count={self.calls}")


class CrashInjector:
    """Arm one crash boundary; restore() unpatches. Usable with or without pytest."""

    def __init__(self, point: CrashPoint | str, crash: object = None) -> None:
        self.point = CrashPoint(point)
        self.triggered = threading.Event()
        self._crash = crash if callable(crash) else self._raise
        self._originals: list[tuple[type, str, object]] = []

    def _raise(self) -> None:
        raise SimulatedCrash(f"simulated process death at {self.point.value}")

    def install(self) -> CrashInjector:
        if self.point is CrashPoint.TRANSCRIPT_APPENDED:
            original_append = TranscriptStore.append_tool_round
            injector = self

            async def append_then_crash(store, *args, **kwargs):
                persisted = await original_append(store, *args, **kwargs)
                injector.triggered.set()
                injector._crash()
                return persisted

            TranscriptStore.append_tool_round = append_then_crash
            self._originals.append((TranscriptStore, "append_tool_round", original_append))
        else:
            armed_state = _JOURNAL_STATE[self.point]
            original_record = TurnExecutionJournal.record
            injector = self

            def record_then_crash(journal, state, **payload):
                # The durable write (fsync) completes first; the process "dies"
                # only afterwards, so the boundary record is always on disk.
                original_record(journal, state, **payload)
                name = getattr(state, "value", state)
                if name == armed_state and not (name == "history_ready" and payload.get("precommit")):
                    injector.triggered.set()
                    injector._crash()

            TurnExecutionJournal.record = record_then_crash
            self._originals.append((TurnExecutionJournal, "record", original_record))
        return self

    def restore(self) -> None:
        while self._originals:
            owner, attribute, original = self._originals.pop()
            setattr(owner, attribute, original)


@dataclass(frozen=True)
class CrashWorkspace:
    """Paths shared by the crashed run and the recovery pass."""

    root: Path
    workspace: Path
    session_id: str = SESSION_ID

    @property
    def session_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def transcript_path(self) -> Path:
        return project_dir(self.session_dir, self.workspace) / f"{self.session_id}.jsonl"


def build_crash_agent(env: CrashWorkspace) -> tuple[ReActAgent, CountingExternalTool, list]:
    """One-tool agent with batch capture; POLARIS_HOME must already be set."""

    tool = CountingExternalTool(env.workspace / SIDE_EFFECT_LOG)
    registry = ToolRegistry()
    registry.register(tool)
    agent = ReActAgent(
        FakeProvider(),
        ReActConfig(
            run_dir=str(env.root / "runs"),
            session_dir=str(env.session_dir),
            permission="bypass",
            memory=MemoryConfig(enabled=False),
            project_instructions=False,
            git_context=False,
        ),
        workspace=env.workspace,
        tools=registry,
        session_id=env.session_id,
    )
    batches: list = []
    original_begin = agent.executor.begin_batch

    def capturing_begin(*args, **kwargs):
        batch = original_begin(*args, **kwargs)
        batches.append(batch)
        return batch

    agent.executor.begin_batch = capturing_begin
    return agent, tool, batches


async def drive_to_crash(
    agent: ReActAgent,
    injector: CrashInjector,
    batches: list,
    *,
    prompt: str = PROMPT,
    timeout: float = 15.0,
) -> None:
    """Run one turn and unwind it the way the armed boundary dictates."""

    task = asyncio.create_task(agent.run(prompt))
    reached = await asyncio.to_thread(injector.triggered.wait, timeout)
    if not reached:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise AssertionError(f"crash point {injector.point.value} was never reached")
    try:
        if injector.point in MAIN_PATH_POINTS:
            try:
                await asyncio.wait_for(task, timeout)
            except SimulatedCrash:
                pass
            else:
                raise AssertionError(f"run completed despite the {injector.point.value} crash")
        else:
            # The crash killed the executor task, so the run stalls in finish();
            # structural cancellation unwinds it without touching the journal.
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError(f"run completed despite the {injector.point.value} crash")
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise AssertionError(f"run did not unwind after the {injector.point.value} crash")
    _retrieve_crashed_tasks(batches)


def _retrieve_crashed_tasks(batches: list) -> None:
    """Mark crashed executor tasks retrieved so asyncio never logs them later."""

    for batch in batches:
        for tracked in getattr(batch, "_calls", ()):
            for attribute in ("preparation_task", "execution_task", "underlying_task"):
                task = getattr(tracked, attribute, None)
                if task is None or not task.done() or task.cancelled():
                    continue
                error = task.exception()
                assert error is None or isinstance(error, SimulatedCrash), (
                    f"executor task failed with an unexpected error: {error!r}"
                )


@dataclass(frozen=True)
class CrashSnapshot:
    """Observable world at one instant: side effects, transcript, journal."""

    journal_path: Path
    effects: int
    effect_lines: list[str]
    transcript_bytes: bytes
    chain_roles: list[str]
    tool_contents: list[str]
    tool_ok: list[bool]
    journal_bytes: bytes
    journal_last_state: str


def snapshot(workspace: Path, transcript_path: Path, journal_path: Path) -> CrashSnapshot:
    log = workspace / SIDE_EFFECT_LOG
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    transcript_bytes = transcript_path.read_bytes() if transcript_path.exists() else b""
    chain = build_chain(load_transcript(transcript_path)) if transcript_path.exists() else []
    tool_messages = [message for message in chain if message.role == "tool"]
    records = TurnExecutionJournal.load(journal_path)
    assert records, f"journal {journal_path} failed checksum verification"
    return CrashSnapshot(
        journal_path=journal_path,
        effects=len(lines),
        effect_lines=lines,
        transcript_bytes=transcript_bytes,
        chain_roles=[message.role for message in chain],
        tool_contents=[message.content for message in tool_messages],
        tool_ok=[bool(message.metadata.get("ok")) for message in tool_messages],
        journal_bytes=journal_path.read_bytes(),
        journal_last_state=records[-1]["state"],
    )


def find_single_journal(storage: JournalStorage) -> Path:
    """Locate the crashed run's journal without trusting any index."""

    paths = sorted(
        path
        for run_root in storage.recovery_root.glob("r-*")
        for path in run_root.glob("*.jsonl")
    )
    assert len(paths) == 1, f"expected exactly one crashed journal, found {paths}"
    return paths[0]


def run_recovery(
    storage: JournalStorage,
    session_dir: Path,
    workspace: Path,
    session_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Path]:
    """Apply startup recovery twice (the second pass proves idempotence)."""

    transcript = TranscriptStore(session_dir, workspace, session_id)
    try:
        outcomes = TurnExecutionJournal.recover_all(
            storage, dry_run=False, history_writer=transcript.recover_tool_round
        )
        again = TurnExecutionJournal.recover_all(
            storage, dry_run=False, history_writer=transcript.recover_tool_round
        )
    finally:
        transcript.close()
    return outcomes, again, transcript.path


def assert_no_replay_and_truthful(
    point: CrashPoint | str,
    *,
    before: CrashSnapshot,
    outcomes: list[dict[str, str]],
    outcomes_again: list[dict[str, str]],
    after: CrashSnapshot,
) -> None:
    """The audit §9.1 contract: no replayed side effects, no false failure history."""

    point = CrashPoint(point)
    turn_id = before.journal_path.stem
    expected_boundary = {
        CrashPoint.EXTERNAL_INTENT_COMMITTED: "external_intent",
        CrashPoint.EXTERNAL_OUTCOME_COMMITTED: "external_outcome_committed",
        CrashPoint.HISTORY_READY_COMMITTED: "history_ready",
        CrashPoint.TRANSCRIPT_APPENDED: "history_ready",
        CrashPoint.HISTORY_PERSISTED: "history_persisted",
    }[point]
    assert before.journal_last_state == expected_boundary, (
        f"crash must leave {expected_boundary} as the last durable record"
    )

    if point is CrashPoint.EXTERNAL_INTENT_COMMITTED:
        # Intent is durable but the outcome is unknowable: the tool never ran,
        # and recovery must report indeterminacy without replaying or finalizing.
        assert before.effects == 0
        assert outcomes == [{"turn_id": turn_id, "status": "IndeterminateExternalEffect"}]
        assert after.effects == 0
        assert after.journal_bytes == before.journal_bytes
        assert outcomes_again == outcomes
        assert after.chain_roles == ["user"]
        return

    # Every later boundary: the external effect happened exactly once, on disk,
    # and recovery never re-executes it.
    assert before.effects == 1
    assert after.effects == 1
    assert after.effect_lines == ["effect 1"]

    if point is CrashPoint.HISTORY_PERSISTED:
        # The terminal record was already durable: recovery is a no-op.
        assert outcomes == []
        assert outcomes_again == []
        assert after.journal_bytes == before.journal_bytes
    else:
        assert outcomes == [{"turn_id": turn_id, "status": "history_persisted"}]
        assert outcomes_again == []
        assert after.journal_last_state == "journal_closed"

    # Truthful, complete history: exactly one round carrying the real result.
    assert after.chain_roles == ["user", "assistant", "tool"]
    assert after.tool_contents == ["counting_external: count=1"]
    assert after.tool_ok == [True]

    if point in (CrashPoint.EXTERNAL_OUTCOME_COMMITTED, CrashPoint.HISTORY_READY_COMMITTED):
        # journal-first ordering: the transcript still lacked the round at crash time.
        assert before.chain_roles == ["user"]
    if point is CrashPoint.TRANSCRIPT_APPENDED:
        # The round was already durable; replay is a checksum no-op.
        assert before.chain_roles == ["user", "assistant", "tool"]
        assert after.transcript_bytes == before.transcript_bytes
    if point is CrashPoint.HISTORY_PERSISTED:
        assert before.chain_roles == ["user", "assistant", "tool"]
