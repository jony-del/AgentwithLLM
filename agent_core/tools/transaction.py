"""Turn-scoped workspace transactions and durable execution journals.

The transaction presents built-in file tools with a private workspace snapshot.
Nothing reaches the real workspace until the authoritative model response has been
reconciled and the scheduler calls :meth:`WorkspaceTransaction.commit`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import re
import shlex
import shutil
import tempfile
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from agent_core.file_lock import FileLock
from agent_core.permission_audit import redact_secret_material
from agent_core.recovery_paths import checked_path, identity, private_path


_SNAPSHOT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "runs",
}
_PROCESS_TOKEN = f"{os.getpid()}:{time.time_ns()}:{uuid.uuid4().hex}"
_TURN_ID = re.compile(r"^[0-9a-f]{32}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_DEVICE_PATH = re.compile(r"^(?:[/\\]{2}[?.][/\\]|[/\\]{2}[^/\\]+[/\\])")
_WINDOWS_RESERVED_NAME = re.compile(
    r"(?i)^(?:con|prn|aux|nul|clock\$|com[1-9]|lpt[1-9])(?:\..*)?$"
)
_AUDIT_LOCK = threading.Lock()
_LIVE_JOURNALS: weakref.WeakValueDictionary[Path, TurnExecutionJournal] = weakref.WeakValueDictionary()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _contained(candidate: str | Path, parent: str | Path, *, allow_equal: bool = False) -> bool:
    canonical_candidate = _canonical(candidate)
    canonical_parent = _canonical(parent)
    if canonical_candidate == canonical_parent:
        return allow_equal
    try:
        canonical_candidate.relative_to(canonical_parent)
    except ValueError:
        return False
    return True


def _project_identity(workspace: str | Path) -> str:
    canonical = os.path.normcase(str(_canonical(workspace)))
    return hashlib.sha256(canonical.encode("utf-8", errors="strict")).hexdigest()


def _identity_component(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()
    return f"{kind[:1]}-{digest[:16]}"


def _secure_mkdir(path: Path) -> None:
    checked_path(path, Path(path.anchor))
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        checked_path(directory, Path(directory.anchor))


def _open_state(storage: JournalStorage, path: Path, mode: str) -> Any:
    """Create private regular files and verify the opened object before writing."""
    storage.validate(path)
    def opener(name: str, flags: int) -> int:
        return os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    handle = open(path, mode, encoding=None if "b" in mode else "utf-8", opener=opener)
    try:
        storage.validate(path)
        info = os.fstat(handle.fileno())
        if info.st_nlink != 1 or (info.st_dev, info.st_ino) != identity(path):
            raise JournalWriteError("recovery file changed while opening")
        return handle
    except BaseException:
        handle.close()
        raise


@dataclass(frozen=True, slots=True)
class JournalStorage:
    """Trusted recovery scope supplied by the runtime, never by journal contents."""

    workspace: Path
    project_id: str
    session_id: str
    run_id: str
    recovery_root: Path
    run_root: Path
    partitioned: bool
    anchor: Path
    private_root: Path | None = None
    _identities: dict[Path, tuple[int, int]] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def local(
        cls,
        root: str | Path,
        *,
        workspace: str | Path | None = None,
        session_id: str = "standalone",
        run_id: str = "standalone",
    ) -> "JournalStorage":
        """Create an explicitly trusted flat scope, primarily for embedding/tests."""

        raw_root = Path(root).expanduser().absolute()
        anchor = raw_root.parent.resolve()
        canonical_root = checked_path(anchor / raw_root.name, anchor)
        canonical_workspace = _canonical(workspace or canonical_root.parent)
        return cls(
            workspace=canonical_workspace,
            project_id=_project_identity(canonical_workspace),
            session_id=str(session_id),
            run_id=str(run_id),
            recovery_root=canonical_root,
            run_root=canonical_root,
            partitioned=False,
            anchor=anchor,
            _identities={anchor: found} if (found := identity(anchor)) is not None else {},
        )

    @classmethod
    def user_state(
        cls,
        workspace: str | Path,
        session_id: str,
        run_id: str,
    ) -> "JournalStorage":
        """Create a project/session/run partition outside the project workspace."""

        canonical_workspace = _canonical(workspace)
        configured_home = os.getenv("POLARIS_HOME")
        try:
            candidate = Path(configured_home).expanduser() if configured_home else Path.home() / ".polaris"
            if not candidate.is_absolute():
                raise OSError("POLARIS_HOME must be an absolute user state directory")
            anchor = candidate.parent.resolve()
            state_home = checked_path(anchor / candidate.name, anchor)
            state_root = state_home / "recovery-journals"
        except (OSError, RuntimeError, ValueError) as exc:
            raise JournalWriteError(f"cannot establish recovery state directory: {exc}") from exc
        if _contained(state_root, canonical_workspace, allow_equal=True):
            raise JournalWriteError(
                "no recovery state directory is canonically outside the project workspace"
            )
        project_id = _project_identity(canonical_workspace)
        project_root = state_root / f"p-{project_id[:16]}"
        session_root = project_root / _identity_component("session", str(session_id))
        run_root = session_root / _identity_component("run", str(run_id))
        storage = cls(
            workspace=canonical_workspace,
            project_id=project_id,
            session_id=str(session_id),
            run_id=str(run_id),
            recovery_root=session_root,
            run_root=run_root,
            partitioned=True,
            anchor=anchor,
            private_root=state_home,
        )
        storage.validate()
        return storage

    def validate(self, path: Path | None = None) -> None:
        """Recheck fixed roots, directory identities, and optional state-file targets."""
        try:
            seen: set[Path] = set()
            if self.partitioned and _contained(self.recovery_root, self.workspace, allow_equal=True):
                raise OSError("recovery state is inside the workspace")
            for root in dict.fromkeys((self.recovery_root, self.run_root)):
                checked_path(root, self.anchor)
                current = root
                while current not in seen:
                    seen.add(current)
                    found = identity(current)
                    previous = self._identities.get(current)
                    if previous is not None and found != previous:
                        raise OSError("recovery directory identity changed")
                    if found is not None:
                        self._identities[current] = found
                    if self.private_root is not None and (current == self.private_root or self.private_root in current.parents):
                        private_path(current)
                    if current == self.anchor:
                        break
                    current = current.parent
            if path is not None:
                checked_path(path, self.recovery_root)
                if self.private_root is not None:
                    current = path
                    while current not in seen:
                        private_path(current)
                        current = current.parent
        except (OSError, ValueError, RuntimeError) as exc:
            raise JournalWriteError(f"unsafe recovery state: {exc}") from exc

    def for_run(self, run_id: str, run_root: str | Path) -> "JournalStorage":
        return JournalStorage(
            workspace=self.workspace,
            project_id=self.project_id,
            session_id=self.session_id,
            run_id=run_id,
            recovery_root=self.recovery_root,
            run_root=Path(run_root),
            partitioned=self.partitioned,
            anchor=self.anchor,
            private_root=self.private_root,
            _identities=self._identities,
        )


def _redact_recovery_payload(value: Any) -> Any:
    """Preserve recoverability while removing marked and obvious secrets."""

    if isinstance(value, list):
        return [_redact_recovery_payload(item) for item in value]
    if not isinstance(value, dict):
        return redact_secret_material(value) if isinstance(value, str) else value
    metadata = value.get("metadata")
    sensitive = bool(value.get("sensitive")) or (
        isinstance(metadata, dict) and bool(metadata.get("sensitive"))
    )
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if sensitive and key == "content":
            sanitized[key] = "<redacted-sensitive-tool-output>"
        else:
            sanitized[str(key)] = _redact_recovery_payload(item)
    return sanitized


class _JournalOwnership:
    """Cross-process advisory lock whose lifetime is tied to an open handle."""

    def __init__(self, path: Path, storage: JournalStorage) -> None:
        self.path = path
        self.storage = storage
        self.handle: Any = None

    def acquire(self) -> bool:
        self.storage.validate(self.path)
        _secure_mkdir(self.path.parent)
        self.storage.validate(self.path)
        handle = _open_state(self.storage, self.path, "a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt as lock_module

                lock_module.locking(handle.fileno(), lock_module.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on POSIX CI
                posix_lock_module: Any = __import__("fcntl")

                getattr(posix_lock_module, "flock")(
                    handle.fileno(),
                    getattr(posix_lock_module, "LOCK_EX")
                    | getattr(posix_lock_module, "LOCK_NB"),
                )
        except (OSError, BlockingIOError):
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt as lock_module

                lock_module.locking(handle.fileno(), lock_module.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised on POSIX CI
                posix_lock_module: Any = __import__("fcntl")

                getattr(posix_lock_module, "flock")(
                    handle.fileno(), getattr(posix_lock_module, "LOCK_UN")
                )
        except OSError:
            pass
        finally:
            handle.close()
        # Keep the lock inode: unlinking after unlock can split concurrent owners
        # between an old open handle and a newly created sidecar.


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _path_kind(path: Path) -> str:
    """Return a stable filesystem kind without following a final symlink."""

    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "missing"


def _ignore_snapshot(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _SNAPSHOT_IGNORES}


class JournalWriteError(RuntimeError):
    """A required execution-journal transition could not be made durable."""


class RecoveryState(str, Enum):
    """Frozen contract (journal schema v4): every state a turn-journal record may carry.

    The write path (``record`` / ``record_telemetry``) validates against this enum, so
    journals can only ever contain states defined here. The read path maps
    ``LEGACY_STATE_ALIASES`` after checksum verification, so older records keep their
    on-disk bytes while all downstream logic sees canonical states only.
    """

    # Tool-call lifecycle (written by ToolExecutor).
    DISCOVERED = "discovered"  # telemetry only, never gating
    ADMITTED = "admitted"
    AUTHORIZED = "authorized"
    STAGED = "staged"
    CLEANUP_REQUIRED = "cleanup_required"
    CLEANUP_COMPLETE = "cleanup_complete"
    EXTERNAL_INTENT = "external_intent"
    EXTERNAL_OUTCOME_COMMITTED = "external_outcome_committed"
    TURN_VALIDATED = "turn_validated"
    HISTORY_READY = "history_ready"
    HISTORY_PERSISTED = "history_persisted"  # terminal
    # Workspace-transaction lifecycle (written by WorkspaceTransaction).
    TRANSACTION_OPENED = "transaction_opened"
    COMMIT_STARTED = "commit_started"
    COMMITTED = "committed"
    RECOVERY_REQUIRED = "recovery_required"
    ROLLED_BACK = "rolled_back"  # terminal
    # Recovery bookkeeping.
    RECOVERY_ACTIONS_APPLIED = "recovery_actions_applied"
    JOURNAL_CLOSED = "journal_closed"  # terminal
    # Legacy terminal states: readable from pre-v4 journals, never written now.
    INDETERMINATE_EXTERNAL_EFFECT = "indeterminate_external_effect"  # terminal
    EXTERNAL_EFFECT_HISTORY_MISSING = "external_effect_history_missing"  # terminal


# Terminal states as plain strings (frozenset[str]) so JSON-loaded records compare
# directly. A journal whose last record is in this set is finished and skipped by
# recovery scans.
TERMINAL_RECOVERY_STATES = frozenset(
    state.value
    for state in (
        RecoveryState.HISTORY_PERSISTED,
        RecoveryState.ROLLED_BACK,
        RecoveryState.JOURNAL_CLOSED,
        RecoveryState.INDETERMINATE_EXTERNAL_EFFECT,
        RecoveryState.EXTERNAL_EFFECT_HISTORY_MISSING,
    )
)

# Schema v3 names accepted on the read path only; writes must use RecoveryState.
LEGACY_STATE_ALIASES = {
    "external_outcome": RecoveryState.EXTERNAL_OUTCOME_COMMITTED.value,
}


class WorkspaceRecoveryRequired(RuntimeError):
    """Commit failed and immediate restoration could not prove consistency."""


@dataclass(slots=True)
class RecoveryReport:
    """Reviewable recovery operations, with no journal-controlled output paths."""

    project_id: str
    session_id: str
    outcomes: list[dict[str, str]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(item["status"] != "foreign" for item in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "blocked": self.blocked,
            "outcomes": self.outcomes,
            "plans": self.plans,
        }


class RecoveryRequiredError(RuntimeError):
    def __init__(self, report: RecoveryReport) -> None:
        self.report = report
        session = json.dumps(report.session_id, ensure_ascii=False)
        argument = ("'" + report.session_id.replace("'", "''") + "'") if os.name == "nt" else shlex.quote(report.session_id)
        super().__init__(
            f"Session {session} has unresolved recovery state; model and tool execution is paused. "
            f"Inspect with: polaris recovery --session-id {argument}. "
            "Use --apply only after reviewing the plan."
        )


class TurnExecutionJournal:
    """Append-only state machine for a single assistant tool round.

    Required transitions are fsync'd.  Callers must treat :class:`JournalWriteError`
    as an action gate: external effects and workspace commit are forbidden when the
    intent cannot first be persisted.
    """

    SCHEMA_VERSION = 4
    READABLE_SCHEMA_VERSIONS = frozenset({3, 4})
    MAX_BYTES = 16 * 1024 * 1024
    MAX_RECORDS = 10_000
    MAX_LINE_BYTES = 1024 * 1024
    TERMINAL_STATES = TERMINAL_RECOVERY_STATES

    def __init__(
        self,
        root: str | Path | JournalStorage,
        turn_id: str | None = None,
        *,
        _ownership: _JournalOwnership | None = None,
        _nonce: str | None = None,
        _sequence: int = 0,
        _previous_checksum: str = "",
        _schema_version: int | None = None,
    ) -> None:
        self.storage = root if isinstance(root, JournalStorage) else JournalStorage.local(root)
        self.root = self.storage.run_root
        self.turn_id = turn_id or uuid.uuid4().hex
        if not _TURN_ID.fullmatch(self.turn_id):
            raise JournalWriteError("turn journal id must be a 32-character lowercase hex value")
        self.path = self.root / f"{self.turn_id}.jsonl"
        self.storage.validate(self.path)
        self.nonce = _nonce or uuid.uuid4().hex
        if not _NONCE.fullmatch(self.nonce):
            raise JournalWriteError("turn journal nonce is invalid")
        self.overlay_root = self.root / "overlays"
        self._ownership = _ownership or _JournalOwnership(
            self.root / f"{self.turn_id}.lock", self.storage
        )
        if _ownership is None and not self._ownership.acquire():
            raise JournalWriteError(f"turn journal {self.turn_id!r} is already owned")
        self._sequence = _sequence
        self._previous_checksum = _previous_checksum
        self._schema_version = _schema_version or self.SCHEMA_VERSION
        self._requests: queue.Queue[
            tuple[str, dict[str, Any], threading.Event | None, list[BaseException]] | None
        ] = queue.Queue()
        self._closed = False
        self._file_identity: tuple[int, int] | None = None
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"turn-journal-{self.turn_id[:8]}",
            daemon=True,
        )
        self._writer.start()
        if _ownership is None:
            _LIVE_JOURNALS[self.path] = self

    @staticmethod
    def _validated_state(state: str) -> str:
        """Return the canonical state string, rejecting anything outside the contract."""

        try:
            return RecoveryState(state).value
        except ValueError:
            raise JournalWriteError(f"unknown recovery state {state!r}") from None

    def record(self, state: str, **payload: Any) -> None:
        """Enqueue a required transition and wait for its durable fsync ack."""

        if self._closed:
            raise JournalWriteError("turn journal is closed")
        state = self._validated_state(state)
        ack = threading.Event()
        errors: list[BaseException] = []
        self._requests.put((state, _redact_recovery_payload(payload), ack, errors))
        ack.wait()
        if errors:
            raise JournalWriteError(
                f"could not persist turn state {state!r}: {errors[0]}"
            ) from errors[0]

    def record_telemetry(self, state: str, **payload: Any) -> None:
        """Queue non-gating telemetry without blocking the SSE consumer."""

        if not self._closed:
            state = self._validated_state(state)
            self._requests.put((state, _redact_recovery_payload(payload), None, []))

    def _writer_loop(self) -> None:
        handle: Any = None
        while True:
            request = self._requests.get()
            if request is None:
                if handle is not None:
                    handle.close()
                return
            state, payload, ack, errors = request
            record: dict[str, Any] = {
                **payload,
                "v": self._schema_version,
                "turn_id": self.turn_id,
                "sequence": self._sequence,
                "state": state,
                "ts": time.time(),
                "pid": os.getpid(),
                "process_token": _PROCESS_TOKEN,
                "owner": {
                    "project_id": self.storage.project_id,
                    "session_id": self.storage.session_id,
                    "run_id": self.storage.run_id,
                    "workspace": str(self.storage.workspace),
                    "nonce": self.nonce,
                },
                "previous_checksum": self._previous_checksum,
            }
            canonical = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            record["checksum"] = checksum
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            try:
                if handle is None:
                    handle = _open_state(self.storage, self.path, "a")
                    opened = os.fstat(handle.fileno())
                    self._file_identity = (opened.st_dev, opened.st_ino)
                # Keep the verified file handle for this journal's lifetime. A
                # redirected pathname cannot redirect subsequent appends. Check
                # identity/link count before each durable transition as well.
                info = os.fstat(handle.fileno())
                if info.st_nlink != 1 or (info.st_dev, info.st_ino) != identity(self.path):
                    raise JournalWriteError("journal file was replaced or hard linked")
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._sequence += 1
                self._previous_checksum = checksum
            except (OSError, JournalWriteError) as exc:
                errors.append(exc)
            finally:
                if ack is not None:
                    ack.set()

    def close(self) -> None:
        if self._closed:
            return
        # A durable no-op makes every earlier telemetry request visible first.
        try:
            self.record(RecoveryState.JOURNAL_CLOSED)
        except JournalWriteError:
            pass
        self._closed = True
        self._requests.put(None)
        self._writer.join(timeout=2)
        self._ownership.release()

    def _release_for_later_recovery(self) -> None:
        """Release ownership without writing a terminal state."""

        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._writer.join(timeout=2)
        self._ownership.release()

    def create_overlay(self, turn_id: str) -> Path:
        """Create a turn overlay under this journal's controlled run root."""

        self.storage.validate(self.overlay_root)
        _secure_mkdir(self.overlay_root)
        container = Path(
            tempfile.mkdtemp(prefix=f"polaris-turn-{turn_id[:8]}-", dir=self.overlay_root)
        )
        if not _contained(container, self.overlay_root):  # pragma: no cover - mkdtemp invariant
            raise JournalWriteError("created overlay escaped its controlled root")
        return container

    def discard_overlay(self, overlay: str | Path) -> None:
        """Delete only a canonical child of this journal's controlled overlay root."""

        candidate = Path(overlay)
        self.storage.validate(candidate)
        if not _contained(candidate, self.overlay_root) or candidate.parent != self.overlay_root:
            raise JournalWriteError("refusing to delete an overlay outside its controlled root")
        if candidate.exists():
            self._validate_overlay_tree(candidate)
            self.storage.validate(candidate)
            shutil.rmtree(candidate)

    @staticmethod
    def _validate_overlay_tree(overlay: Path) -> None:
        # Reject directory redirects before rmtree, including Windows junctions.
        for directory, dirs, files in os.walk(overlay, followlinks=False):
            for name in (*dirs, *files):
                checked_path(Path(directory) / name, overlay)

    @classmethod
    def _load_verified(
        cls, path: str | Path
    ) -> tuple[list[dict[str, Any]], str | None]:
        records: list[dict[str, Any]] = []
        journal_path = Path(path)
        try:
            if journal_path.stat().st_size > cls.MAX_BYTES:
                return records, "journal_too_large"
            with journal_path.open("rb") as handle:
                raw = handle.read(cls.MAX_BYTES + 1)
            if len(raw) > cls.MAX_BYTES:
                return records, "journal_too_large"
        except OSError as exc:
            return records, f"journal_read_failed:{type(exc).__name__}"
        if not raw or not raw.endswith(b"\n"):
            return records, "journal_truncated"
        lines = raw.splitlines()
        if len(lines) > cls.MAX_RECORDS:
            return records, "too_many_records"
        previous_checksum = ""
        schema_version: int | None = None
        expected_owner: dict[str, Any] | None = None
        for expected_sequence, raw_line in enumerate(lines):
            if len(raw_line) > cls.MAX_LINE_BYTES:
                return [], "record_too_large"
            try:
                value = json.loads(raw_line.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                return [], "invalid_json"
            if not isinstance(value, dict):
                return [], "record_not_object"
            if type(value.get("v")) is not int or value["v"] not in cls.READABLE_SCHEMA_VERSIONS:
                return [], "unsupported_schema"
            if schema_version is None:
                schema_version = int(value["v"])
            elif value.get("v") != schema_version:
                return [], "schema_changed"
            if type(value.get("sequence")) is not int or value["sequence"] != expected_sequence:
                return [], "invalid_sequence"
            if not isinstance(value.get("state"), str) or not value["state"]:
                return [], "invalid_state"
            turn_id = value.get("turn_id")
            if not isinstance(turn_id, str) or not _TURN_ID.fullmatch(turn_id):
                return [], "invalid_turn_id"
            owner = value.get("owner")
            if not isinstance(owner, dict):
                return [], "invalid_owner"
            if set(owner) != {"project_id", "session_id", "run_id", "workspace", "nonce"}:
                return [], "invalid_owner_fields"
            if not all(isinstance(owner.get(key), str) for key in owner):
                return [], "invalid_owner_types"
            if not _NONCE.fullmatch(str(owner.get("nonce"))):
                return [], "invalid_nonce"
            if expected_owner is None:
                expected_owner = owner
            elif owner != expected_owner:
                return [], "owner_changed"
            if value.get("previous_checksum") != previous_checksum:
                return [], "broken_checksum_chain"
            checksum = value.get("checksum")
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                return [], "invalid_checksum"
            body = dict(value)
            body.pop("checksum", None)
            try:
                canonical = json.dumps(
                    body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
                )
                actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            except (ValueError, UnicodeError, RecursionError):
                return [], "invalid_record_encoding"
            if not hmac.compare_digest(checksum, actual):
                return [], "checksum_mismatch"
            previous_checksum = checksum
            # Checksum verified over the original bytes; only now normalize legacy
            # state names so every downstream reader sees canonical RecoveryState values.
            value["state"] = LEGACY_STATE_ALIASES.get(value["state"], value["state"])
            records.append(value)
        if any(item.get("turn_id") != records[0].get("turn_id") for item in records):
            return [], "turn_id_changed"
        return records, None

    @classmethod
    def load(cls, path: str | Path) -> list[dict[str, Any]]:
        """Load only a complete, checksummed journal chain."""

        records, error = cls._load_verified(path)
        return records if error is None else []

    @staticmethod
    def _safe_relative(raw: object) -> str | None:
        if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
            return None
        if _WINDOWS_DEVICE_PATH.match(raw):
            return None
        windows = PureWindowsPath(raw)
        posix = PurePosixPath(raw)
        if windows.is_absolute() or windows.drive or windows.root or posix.is_absolute():
            return None
        if any(part in {"", ".", ".."} for part in posix.parts):
            return None
        if any(_WINDOWS_RESERVED_NAME.fullmatch(part) for part in posix.parts):
            return None
        if any(part.endswith((".", " ")) or any(ord(char) < 32 for char in part) for part in posix.parts):
            return None
        normalized = posix.as_posix()
        if normalized != raw or ":" in normalized:
            return None
        return normalized

    @classmethod
    def _validate_owner(
        cls,
        storage: JournalStorage,
        path: Path,
        records: list[dict[str, Any]],
    ) -> tuple[JournalStorage | None, str | None]:
        owner = records[0]["owner"]
        if owner["project_id"] != storage.project_id:
            return None, "foreign_project"
        if owner["session_id"] != storage.session_id:
            return None, "foreign_session"
        if _canonical(owner["workspace"]) != storage.workspace:
            return None, "foreign_workspace"
        if _project_identity(owner["workspace"]) != owner["project_id"]:
            return None, "project_identity_mismatch"
        if path.stem != records[0]["turn_id"]:
            return None, "turn_filename_mismatch"
        owner_run = str(owner["run_id"])
        if storage.partitioned:
            expected_root = storage.recovery_root / _identity_component("run", owner_run)
            if _canonical(path.parent) != _canonical(expected_root):
                return None, "foreign_run_partition"
            journal_storage = storage.for_run(owner_run, expected_root)
        else:
            if owner_run != storage.run_id or _canonical(path.parent) != storage.run_root:
                return None, "foreign_run"
            journal_storage = storage
        return journal_storage, None

    @classmethod
    def _recovery_plan(
        cls,
        storage: JournalStorage,
        path: Path,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        storage.validate(path)
        completion = next((item for item in reversed(records) if item["state"] == RecoveryState.RECOVERY_ACTIONS_APPLIED), None)
        if completion is not None and (
            completion.get("recovery_version") != 1
            or completion.get("outcome") not in (RecoveryState.HISTORY_PERSISTED, RecoveryState.ROLLED_BACK)
        ):
            return None, "invalid_recovery_checkpoint"
        is_committed = completion is not None or any(item["state"] == RecoveryState.COMMITTED for item in records)
        overlay_values: list[str] = []
        for record in records:
            if "overlay" not in record:
                continue
            value = record["overlay"]
            if not isinstance(value, str) or not value:
                return None, "invalid_overlay_type"
            overlay_values.append(value)
        overlay: Path | None = None
        overlay_root = path.parent / "overlays"
        if overlay_values:
            overlay = Path(overlay_values[0])
            if not overlay.is_absolute():
                return None, "overlay_outside_controlled_root"
            if any(Path(value) != overlay for value in overlay_values[1:]):
                return None, "overlay_changed"
            if not _contained(overlay, overlay_root) or overlay.parent != overlay_root:
                return None, "overlay_outside_controlled_root"
            storage.validate(overlay)
            if not overlay.name.startswith("polaris-turn-"):
                return None, "invalid_overlay_name"
            if overlay.is_symlink() or (overlay.exists() and not overlay.is_dir()):
                return None, "unsafe_overlay_kind"
            cls._validate_overlay_tree(overlay)

        opened = [item for item in records if item.get("state") == RecoveryState.TRANSACTION_OPENED]
        for record in opened:
            workspace = record.get("workspace")
            if not isinstance(workspace, str) or _canonical(workspace) != storage.workspace:
                return None, "workspace_owner_mismatch"
        if opened and overlay is None:
            return None, "transaction_missing_overlay"

        commit = next(
            (item for item in reversed(records) if item.get("state") == RecoveryState.COMMIT_STARTED),
            None,
        )
        replacements: list[dict[str, Any]] = []
        if commit is not None:
            if not opened or overlay is None:
                return None, "commit_missing_transaction"
            changed = commit.get("changed")
            existed = commit.get("existed")
            if not isinstance(changed, list) or not isinstance(existed, dict):
                return None, "invalid_commit_fields"
            normalized: list[str] = []
            for raw_relative in changed:
                relative = cls._safe_relative(raw_relative)
                if relative is None or relative in normalized:
                    return None, "unsafe_changed_path"
                normalized.append(relative)
            if set(existed) != set(normalized) or not all(type(value) is bool for value in existed.values()):
                return None, "invalid_existence_map"
            recovery_root = overlay / "recovery"
            if not _contained(recovery_root, overlay):
                return None, "invalid_recovery_root"
            for relative in normalized:
                target = storage.workspace / PurePosixPath(relative)
                backup = recovery_root / PurePosixPath(relative)
                if not _contained(target, storage.workspace) or not _contained(backup, recovery_root):
                    return None, "canonical_containment_failed"
                checked_path(target, storage.workspace)
                storage.validate(backup)
                target_kind = _path_kind(target)
                if target_kind == "symlink":
                    return None, "workspace_symlink_escape"
                did_exist = existed[relative]
                if is_committed:
                    continue  # Committed workspace changes must never be rolled back.
                if target_kind not in {"missing", "file"}:
                    return None, "unsafe_recovery_target"
                if did_exist:
                    if backup.is_symlink() or not backup.is_file():
                        return None, "missing_or_unsafe_backup"
                    if not _contained(backup.resolve(strict=True), recovery_root):
                        return None, "backup_symlink_escape"
                elif target_kind not in {"missing", "file"}:
                    return None, "unsafe_recovery_target"
                replacements.append(
                    {"relative": relative, "target": target, "backup": backup, "existed": did_exist}
                )

        committed = next(
            (item for item in reversed(records) if item.get("state") == RecoveryState.COMMITTED),
            None,
        )
        if committed is not None:
            changed = committed.get("changed")
            if not isinstance(changed, list) or any(cls._safe_relative(item) is None for item in changed):
                return None, "invalid_committed_paths"
        return {
            "overlay": overlay, "replacements": replacements,
            "completed_status": completion["outcome"] if completion is not None else None,
        }, None

    @staticmethod
    def _audit(
        storage: JournalStorage,
        payload: dict[str, Any],
        audit_writer: Callable[[dict[str, Any]], None] | None,
    ) -> bool:
        body = {
            "schema_version": 1,
            "event": "turn_journal_recovery",
            "ts": time.time(),
            "project_id": storage.project_id,
            "session_id": storage.session_id,
            "current_run_id": storage.run_id,
            **payload,
        }
        canonical = json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        body["checksum"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        line = json.dumps(body, ensure_ascii=False, default=str) + "\n"
        try:
            with _AUDIT_LOCK:
                storage.validate()
                _secure_mkdir(storage.recovery_root)
                audit_path = storage.recovery_root / "recovery-audit.log"
                storage.validate(audit_path)
                with _open_state(storage, audit_path, "a") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
        except (OSError, JournalWriteError):
            return False
        if audit_writer is not None:
            try:
                audit_writer(dict(body))
            except Exception:
                logging.getLogger(__name__).warning("recovery run-audit callback failed", exc_info=True)
        return True

    @classmethod
    def _all_journal_paths(cls, storage: JournalStorage) -> list[Path]:
        storage.validate()
        base = storage.recovery_root
        if not base.exists() or not base.is_dir():
            return []
        if not storage.partitioned:
            return sorted(base.glob("*.jsonl"))
        paths: list[Path] = []
        for run_root in sorted(base.glob("r-*")):
            storage.validate(run_root)
            if not run_root.is_dir():
                raise JournalWriteError("recovery run partition is not a directory")
            paths.extend(sorted(run_root.glob("*.jsonl")))
        return paths

    @classmethod
    def _open_index_path(cls, storage: JournalStorage) -> Path:
        return storage.recovery_root / ".open-journals.json"


    @classmethod
    def _journal_paths(cls, storage: JournalStorage) -> list[Path]:
        """Scan shallow session partitions without repairing or trusting the index.

        An incomplete index must never hide recovery state from the execution gate.
        Its contents are deliberately not parsed on this security-sensitive path.
        """
        storage.validate()
        if not storage.recovery_root.exists():
            return []
        index = cls._open_index_path(storage)
        storage.validate(index)
        storage.validate(index.with_suffix(".lock"))
        storage.validate(storage.recovery_root / ".retention.lock")
        storage.validate(storage.recovery_root / ".retention-state.json")
        storage.validate(storage.recovery_root / "recovery-audit.log")
        paths: list[Path] = []
        for path in cls._all_journal_paths(storage):
            try:
                storage.validate(path)
            except JournalWriteError:
                paths.append(path)
                continue
            records, error = cls._load_verified(path)
            if error is not None or not records or records[-1]["state"] not in cls.TERMINAL_STATES:
                paths.append(path)
        return paths

    @classmethod
    def prune_terminal(
        cls,
        storage: JournalStorage,
        *,
        retention_days: int = 7,
        max_per_session: int = 500,
        scan_interval_seconds: int = 86_400,
        now: float | None = None,
    ) -> dict[str, int]:
        """Delete terminal journals under one cross-process retention lock."""

        lock_path = storage.recovery_root / ".retention.lock"
        storage.validate(lock_path)
        _secure_mkdir(storage.recovery_root)
        _open_state(storage, lock_path, "a+b").close()
        with FileLock(lock_path):
            return cls._prune_terminal_unlocked(
                storage,
                retention_days=retention_days,
                max_per_session=max_per_session,
                scan_interval_seconds=scan_interval_seconds,
                now=now,
            )

    @classmethod
    def _prune_terminal_unlocked(
        cls,
        storage: JournalStorage,
        *,
        retention_days: int = 7,
        max_per_session: int = 500,
        scan_interval_seconds: int = 86_400,
        now: float | None = None,
    ) -> dict[str, int]:
        """Delete only verified terminal journals, at most once per scan interval."""

        now = time.time() if now is None else now
        marker = storage.recovery_root / ".retention-state.json"
        storage.validate(marker)
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
            if now - float(raw.get("last_scan", 0)) < scan_interval_seconds:
                return {"deleted": 0, "bytes": 0, "errors": 0, "skipped": 1}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        terminal: list[tuple[Path, float, int]] = []
        for path in cls._all_journal_paths(storage):
            storage.validate(path)
            records, error = cls._load_verified(path)
            if error is not None or not records or records[-1].get("state") not in cls.TERMINAL_STATES:
                continue
            _owner, error = cls._validate_owner(storage, path, records)
            if error is not None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            terminal.append((path, stat.st_mtime, stat.st_size))
        terminal.sort(key=lambda item: item[1], reverse=True)
        cutoff = now - max(1, retention_days) * 86_400
        selected = [
            item for index, item in enumerate(terminal)
            if item[1] < cutoff or index >= max(1, max_per_session)
        ]
        report = {"deleted": 0, "bytes": 0, "errors": 0, "skipped": 0}
        for path, _modified, size in selected:
            ownership = _JournalOwnership(path.with_suffix(".lock"), storage)
            if not ownership.acquire():
                continue
            try:
                storage.validate(path)
                path.unlink()
                report["deleted"] += 1
                report["bytes"] += size
            except OSError:
                report["errors"] += 1
            finally:
                ownership.release()
        try:
            _secure_mkdir(storage.recovery_root)
            temporary = marker.with_suffix(marker.suffix + f".{uuid.uuid4().hex}.tmp")
            with _open_state(storage, temporary, "x") as output:
                output.write(json.dumps({"v": 1, "last_scan": now}))
            os.replace(temporary, marker)
        except OSError:
            report["errors"] += 1
        return report

    @classmethod
    def _reconcile_history_payload(
        cls,
        records: list[dict[str, Any]],
        history_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Overlay indeterminate external intents onto the latest durable history."""

        payload = json.loads(json.dumps(history_payload, ensure_ascii=False, default=str))
        outcomes = {
            int(item["ordinal"])
            for item in records
            if item.get("state") == RecoveryState.EXTERNAL_OUTCOME_COMMITTED
            and isinstance(item.get("ordinal"), int)
        }
        uncertain = {
            int(item["ordinal"])
            for item in records
            if item.get("state") == RecoveryState.EXTERNAL_INTENT
            and isinstance(item.get("ordinal"), int)
            and int(item["ordinal"]) not in outcomes
        }
        results = payload.get("tool_results")
        if not isinstance(results, list):
            return payload
        for index, raw in enumerate(results):
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("metadata")
            ordinal = metadata.get("ordinal", index) if isinstance(metadata, dict) else index
            if ordinal not in uncertain:
                continue
            updated = dict(metadata) if isinstance(metadata, dict) else {}
            updated.update(
                {
                    "ok": False,
                    "error_type": "IndeterminateExternalEffect",
                    "execution_status": "indeterminate",
                    "ordinal": ordinal,
                }
            )
            raw["content"] = (
                f"{raw.get('name') or 'tool'}: External operation may have completed; "
                "automatic replay was suppressed"
            )
            raw["metadata"] = updated
        payload["complete"] = True
        return payload

    @classmethod
    def _inspect_one(
        cls, storage: JournalStorage, path: Path, history_path: Path | None,
    ) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
        """No locks, writers, index repairs, or recovery mutations."""
        try:
            storage.validate(path)
            storage.validate(path.with_suffix(".lock"))
            if not _TURN_ID.fullmatch(path.stem):
                return {"turn_id": path.stem, "status": "rejected", "reason": "invalid_turn_filename"}, None
            records, error = cls._load_verified(path)
            if error is not None:
                return {"turn_id": path.stem, "status": "rejected", "reason": error}, None
            owner, error = cls._validate_owner(storage, path, records)
            if error is not None or owner is None:
                return {"turn_id": path.stem, "status": "foreign", "reason": error or "invalid_owner"}, None
            states = {item["state"] for item in records}
            if records[-1]["state"] in cls.TERMINAL_STATES:
                return None, None
            plan, error = cls._recovery_plan(owner, path, records)
            if error is not None or plan is None:
                return {"turn_id": path.stem, "status": "rejected", "reason": error or "invalid_plan"}, None
            known_outcomes = {
                item.get("ordinal") for item in records
                if item["state"] == RecoveryState.EXTERNAL_OUTCOME_COMMITTED
                and type(item.get("ordinal")) is int
            }
            if not plan["completed_status"] and any(
                item["state"] == RecoveryState.EXTERNAL_INTENT and (
                    type(item.get("ordinal")) is not int or item["ordinal"] not in known_outcomes
                ) for item in records
            ):
                return {"turn_id": path.stem, "status": "IndeterminateExternalEffect"}, None
            payload = next(
                (item.get("history_payload") for item in reversed(records) if item.get("history_payload")), None,
            )
            needs_history = not plan["completed_status"] and bool(states & {RecoveryState.COMMITTED.value, RecoveryState.EXTERNAL_OUTCOME_COMMITTED.value})
            if isinstance(payload, dict):
                payload = cls._reconcile_history_payload(records, payload)
            if needs_history and isinstance(payload, dict):
                if payload.get("session_id") not in {None, storage.session_id}:
                    return {"turn_id": path.stem, "status": "rejected", "reason": "history_session_mismatch"}, None
                expected = payload.get("transcript_path")
                if expected and (
                    not isinstance(expected, str) or history_path is None
                    or _canonical(expected) != history_path.absolute()
                ):
                    return {"turn_id": path.stem, "status": "rejected", "reason": "history_target_mismatch"}, None
            if needs_history and history_path is not None:
                cls._check_history_target(history_path)
                from agent_core.models import Message

                if not isinstance(payload, dict) or not isinstance(payload.get("assistant"), dict) or not isinstance(payload.get("tool_results"), list) or not isinstance(payload.get("execution_manifest"), dict):
                    return {"turn_id": path.stem, "status": "rejected", "reason": "invalid_history_payload"}, None
                Message.from_dict(payload["assistant"])
                for result in payload["tool_results"]:
                    if not isinstance(result, dict):
                        return {"turn_id": path.stem, "status": "rejected", "reason": "invalid_history_payload"}, None
                    Message.from_dict(result)
            actions = [
                {"action": "restore_file" if item["existed"] else "delete_file", "target": str(item["target"])}
                for item in reversed(plan["replacements"])
            ]
            if needs_history:
                actions.append({"action": "append_history", "target": str(history_path) if history_path else "runtime_history_writer"})
            if plan["overlay"] is not None and plan["overlay"].exists():
                actions.append({"action": "remove_overlay", "target": str(plan["overlay"])})
            action = "recover_history" if needs_history else "rollback_workspace"
            if plan["completed_status"]:
                action = "cleanup_overlay"
            snapshots: dict[Path, tuple[int, ...] | None] = {}
            targets = [path, storage.workspace]
            if plan["overlay"] is not None:
                targets.append(plan["overlay"])
            for item in plan["replacements"]:
                targets.extend((item["target"], item["backup"]))
            if needs_history and history_path is not None:
                targets.append(history_path)
            for target in targets:
                for component in (target, *target.parents):
                    snapshots[component] = cls._fingerprint(component)
            plan.update(
                storage=owner, records=records, states=states, payload=payload,
                needs_history=needs_history, action=action, actions=actions, snapshots=snapshots,
            )
            return {"turn_id": path.stem, "status": f"would_{action}"}, plan
        except (OSError, JournalWriteError, ValueError, TypeError, KeyError, RecursionError) as exc:
            return {"turn_id": path.stem, "status": "rejected", "reason": f"unsafe_recovery_state:{type(exc).__name__}"}, None

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, ...] | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if path.is_dir():
            return info.st_dev, info.st_ino
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    @staticmethod
    def _check_history_target(path: Path) -> None:
        for target in (
            path, path.with_suffix(path.suffix + ".round-index.json"),
            path.with_suffix(path.suffix + ".round-index.json.tmp"),
            path.with_suffix(path.suffix + ".write.lock"),
            path.with_suffix(path.suffix + f".active.{os.getpid()}"),
        ):
            checked_path(target.absolute(), Path(target.absolute().anchor))

    @classmethod
    def inspect_recovery(
        cls, storage: JournalStorage, *, history_path: Path | None = None,
        audit_writer: Callable[[dict[str, Any]], None] | None = None,
        _include_active: bool = True,
    ) -> RecoveryReport:
        """Preview all unfinished journals; only the independent audit may be appended."""
        report = RecoveryReport(storage.project_id, storage.session_id)
        try:
            paths = cls._journal_paths(storage)
        except (OSError, JournalWriteError, ValueError) as exc:
            paths = []
            report.outcomes.append({"turn_id": "", "status": "rejected", "reason": f"unsafe_recovery_state:{type(exc).__name__}"})
        for path in paths:
            if not _include_active:
                live = _LIVE_JOURNALS.get(path)
                if (
                    live is not None and not live._closed and live._ownership.handle is not None
                    and live.storage.project_id == storage.project_id
                    and live.storage.session_id == storage.session_id
                    and live._file_identity is not None and live._file_identity == identity(path)
                ):
                    # Parent and sibling tool rounds are live work, not crash
                    # recovery. Only real in-memory owners count; journal PIDs,
                    # tokens, and nonce claims never grant this exemption.
                    continue
            outcome, plan = cls._inspect_one(storage, path, history_path)
            if outcome is None:
                continue
            report.outcomes.append(outcome)
            if plan is not None:
                report.plans.append({
                    "turn_id": path.stem, "run_id": plan["storage"].run_id,
                    "action": plan["action"], "actions": plan["actions"],
                })
        for outcome in list(report.outcomes):
            detail = next((plan for plan in report.plans if plan["turn_id"] == outcome["turn_id"]), {})
            if not cls._audit(storage, {**detail, **outcome, "phase": "plan", "dry_run": True}, audit_writer):
                report.outcomes.append({"turn_id": outcome["turn_id"], "status": "audit_failed"})
                break
        return report

    @classmethod
    def require_recovered(cls, storage: JournalStorage, *, history_path: Path | None = None) -> RecoveryReport:
        report = cls.inspect_recovery(storage, history_path=history_path, _include_active=False)
        if report.blocked:
            raise RecoveryRequiredError(report)
        return report

    @classmethod
    def recover_all(
        cls,
        storage: JournalStorage,
        *,
        history_writer: Callable[[dict[str, Any]], bool] | None = None,
        dry_run: bool = True,
        audit_writer: Callable[[dict[str, Any]], None] | None = None,
        history_path: Path | None = None,
        authorization_source: str = "explicit_api",
    ) -> list[dict[str, str]]:
        """Preview by default. Explicit apply revalidates under exclusive ownership.

        Checksums detect corruption, not authenticity. The trusted runtime scope,
        path checks and the caller's explicit apply decision remain action gates.
        External tools are never retried here.
        """
        if history_path is None:
            writer_owner = getattr(history_writer, "__self__", None)
            candidate = getattr(writer_owner, "path", None)
            if isinstance(candidate, Path):
                history_path = candidate
        if dry_run:
            return cls.inspect_recovery(storage, history_path=history_path, audit_writer=audit_writer).outcomes
        try:
            paths = cls._journal_paths(storage)
        except (OSError, JournalWriteError, ValueError) as exc:
            failure = {"turn_id": "", "status": "rejected", "reason": f"unsafe_recovery_state:{type(exc).__name__}"}
            cls._audit(storage, {**failure, "phase": "validation", "dry_run": False}, audit_writer)
            return [failure]
        outcomes: list[dict[str, str]] = []
        for path in paths:
            outcome, plan = cls._inspect_one(storage, path, history_path)
            if outcome is None:
                continue
            if plan is None:
                cls._audit(storage, {**outcome, "phase": "validation", "dry_run": False}, audit_writer)
                outcomes.append(outcome)
                continue
            ownership = _JournalOwnership(path.with_suffix(".lock"), storage)
            journal: TurnExecutionJournal | None = None
            terminal_status: str | None = None
            try:
                if not ownership.acquire():
                    outcomes.append({"turn_id": path.stem, "status": "busy"})
                    cls._audit(storage, {**outcomes[-1], "phase": "ownership", "dry_run": False}, audit_writer)
                    continue
                # The preview is advisory. Re-read under the lock before authorizing any action.
                outcome, plan = cls._inspect_one(storage, path, history_path)
                if outcome is None:
                    continue
                if plan is None:
                    outcomes.append(outcome)
                    continue
                before = {
                    "turn_id": path.stem, "owner_run_id": plan["storage"].run_id,
                    "phase": "before", "status": "authorized", "action": plan["action"],
                    "actions": plan["actions"], "dry_run": False,
                    "authorization_source": authorization_source,
                }
                if not cls._audit(storage, before, audit_writer):
                    outcomes.append({"turn_id": path.stem, "status": "audit_failed"})
                    continue
                # Audit callbacks, concurrent processes, and file replacement cannot change the plan.
                storage.validate(path)
                for target, fingerprint in plan["snapshots"].items():
                    if cls._fingerprint(target) != fingerprint:
                        raise OSError("recovery target changed after validation")
                fresh_outcome, fresh = cls._inspect_one(storage, path, history_path)
                if fresh is None or fresh_outcome != outcome or fresh["records"] != plan["records"]:
                    raise OSError("recovery journal or paths changed after validation")
                if plan["needs_history"] and (history_writer is None or not isinstance(plan["payload"], dict)):
                    outcomes.append({"turn_id": path.stem, "status": "committed_history_pending"})
                    continue
                records = plan["records"]
                journal = cls(
                    plan["storage"], path.stem, _ownership=ownership,
                    _nonce=records[0]["owner"]["nonce"], _sequence=len(records),
                    _previous_checksum=records[-1]["checksum"],
                    _schema_version=records[0]["v"],
                )
                for item in reversed(plan["replacements"]):
                    target, backup = item["target"], item["backup"]
                    storage.validate(backup)
                    checked_path(target, storage.workspace)
                    if cls._fingerprint(target) != plan["snapshots"][target]:
                        raise OSError("workspace target changed during recovery")
                    if cls._fingerprint(backup) != plan["snapshots"][backup]:
                        raise OSError("recovery backup changed during recovery")
                    if item["existed"]:
                        # Preserve the durable backup if a later replacement fails.
                        target.parent.mkdir(parents=True, exist_ok=True)
                        checked_path(target, storage.workspace)
                        descriptor, temporary_name = tempfile.mkstemp(prefix=".polaris-recovery-", dir=target.parent)
                        temporary = Path(temporary_name)
                        try:
                            with os.fdopen(descriptor, "wb") as destination, backup.open("rb") as source:
                                shutil.copyfileobj(source, destination)
                                destination.flush()
                                os.fsync(destination.fileno())
                            shutil.copystat(backup, temporary, follow_symlinks=False)
                            checked_path(target, storage.workspace)
                            os.replace(temporary, target)
                        finally:
                            checked_path(temporary, storage.workspace)
                            temporary.unlink(missing_ok=True)
                    elif target.exists():
                        target.unlink()
                if plan["completed_status"]:
                    terminal_status = plan["completed_status"]
                elif plan["needs_history"]:
                    if history_path is not None:
                        cls._check_history_target(history_path)
                    if history_writer is None or not history_writer(plan["payload"]):
                        raise OSError("history writer did not persist the recovered round")
                    terminal_status = RecoveryState.HISTORY_PERSISTED.value
                elif RecoveryState.EXTERNAL_INTENT.value in plan["states"]:
                    # Preserve the evidence: external outcomes need manual reconciliation.
                    outcomes.append({"turn_id": path.stem, "status": "IndeterminateExternalEffect"})
                    continue
                else:
                    terminal_status = RecoveryState.ROLLED_BACK.value
                if not plan["completed_status"]:
                    # Retrying after cleanup/final-journal failure must not require
                    # backups already removed by that cleanup, nor undo later edits.
                    journal.record(RecoveryState.RECOVERY_ACTIONS_APPLIED, recovery_version=1, outcome=terminal_status)
                if plan["overlay"] is not None:
                    journal.discard_overlay(plan["overlay"])
                # Only terminalize after ALL recovery actions succeed.
                journal.record(terminal_status, recovery=True)
                outcomes.append({"turn_id": path.stem, "status": terminal_status})
                journal.close()
            except Exception as exc:
                outcomes.append({"turn_id": path.stem, "status": "recovery_failed", "reason": type(exc).__name__})
            finally:
                if journal is not None:
                    journal._release_for_later_recovery()
                ownership.release()
                if outcomes and outcomes[-1]["turn_id"] == path.stem:
                    if not cls._audit(
                        storage, {**outcomes[-1], "phase": "after", "dry_run": False}, audit_writer,
                    ) and outcomes[-1]["status"] != "audit_failed":
                        outcomes.append({"turn_id": path.stem, "status": "audit_failed"})
        return outcomes


@dataclass(slots=True)
class _Replacement:
    target: Path
    backup: Path | None
    existed: bool


class WorkspaceTransaction:
    """Path-scoped sparse overlay with conflict-checked per-file commit.

    Only resources declared by tool policies are materialized.  The overlay is
    therefore also an enforcement boundary: any changed path outside a declared
    filesystem write lock makes commit fail closed.
    """

    def __init__(
        self,
        workspace: str | Path,
        turn_id: str,
        *,
        journal: TurnExecutionJournal,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.turn_id = turn_id
        self.journal = journal
        if self.workspace != self.journal.storage.workspace:
            raise JournalWriteError("transaction workspace does not match journal ownership")
        self._container = self.journal.create_overlay(turn_id)
        self.overlay = self._container / "workspace"
        self._closed = False
        self._recovery_required = False
        self._baseline: dict[str, str | None] = {}
        self._baseline_kinds: dict[str, str] = {}
        self._subtree_members: dict[str, frozenset[str]] = {}
        self._write_scopes: list[tuple[str, bool]] = []
        self._declared_scopes: set[tuple[str, str, bool]] = set()
        self.overlay.mkdir(parents=True, exist_ok=True)
        self.journal.record(RecoveryState.TRANSACTION_OPENED, overlay=str(self._container), workspace=str(self.workspace))

    def execution_root(self) -> Path:
        return self.overlay

    def ensure_paths(self, paths: list[Path]) -> None:
        """Backward-compatible exact writable-path declaration."""

        for source in paths:
            self._declare_path(source, mode="write", subtree=False)

    def declare_resources(self, locks: object) -> None:
        for lock in locks if isinstance(locks, (list, tuple)) else ():
            if getattr(lock, "namespace", None) != "fs":
                continue
            self._declare_path(
                Path(str(getattr(lock, "key", ""))),
                mode=str(getattr(lock, "mode", "read")),
                subtree=bool(getattr(lock, "subtree", False)),
            )

    def _declare_path(self, source: Path, *, mode: str, subtree: bool) -> None:
        candidate = source if source.is_absolute() else self.workspace / source
        resolved = candidate.resolve(strict=False)
        try:
            relative_path = resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"transaction resource escapes workspace: {source}") from exc
        relative = relative_path.as_posix()
        scope = (relative, mode, subtree)
        if scope in self._declared_scopes:
            return
        self._declared_scopes.add(scope)
        if mode == "write":
            self._write_scopes.append((relative, subtree))
        self._materialize(resolved, relative_path, subtree=subtree)

    def _materialize(self, source: Path, relative: Path, *, subtree: bool) -> None:
        key = relative.as_posix()
        if source.is_symlink():
            raise RuntimeError(f"symbolic links are not valid transaction resources: {key}")
        if not source.exists():
            self._baseline.setdefault(key, None)
            self._baseline_kinds.setdefault(key, "missing")
            if subtree:
                self._subtree_members.setdefault(key, frozenset())
            return
        target = self.overlay / relative
        if source.is_dir():
            self._baseline_kinds.setdefault(key, "directory")
            target.mkdir(parents=True, exist_ok=True)
            if not subtree:
                return
            members: set[str] = set()
            for child in source.rglob("*"):
                child_relative = child.relative_to(self.workspace)
                if any(part in _SNAPSHOT_IGNORES for part in child_relative.parts):
                    continue
                if child.is_symlink():
                    raise RuntimeError(
                        f"symbolic links are not supported in transaction scopes: {child_relative.as_posix()}"
                    )
                child_key = child_relative.as_posix()
                members.add(child_key)
                self._baseline_kinds.setdefault(
                    child_key, "directory" if child.is_dir() else "file"
                )
                destination = self.overlay / child_relative
                if child.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(child, destination, follow_symlinks=False)
                self._baseline.setdefault(child_key, _digest(child))
            self._subtree_members.setdefault(key, frozenset(members))
            return
        self._baseline_kinds.setdefault(key, "file")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target, follow_symlinks=False)
        self._baseline.setdefault(key, _digest(source))

    @staticmethod
    def _contains(candidate: str, parent: str) -> bool:
        if candidate == parent:
            return True
        try:
            Path(candidate).relative_to(Path(parent))
        except ValueError:
            return False
        return True

    def _write_allowed(self, relative: str) -> bool:
        return any(
            relative == scope or (subtree and self._contains(relative, scope))
            for scope, subtree in self._write_scopes
        )

    def changed_paths(self) -> list[str]:
        changed: list[str] = []
        current: set[str] = set()
        for path in self.overlay.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(
                    f"transaction created or modified a symbolic link: {path.relative_to(self.overlay)}"
                )
            if path.is_file():
                relative = path.relative_to(self.overlay).as_posix()
                current.add(relative)
                if _digest(path) != self._baseline.get(relative):
                    changed.append(relative)
        # File tools currently do not delete files, but retaining deletion detection
        # makes the transaction correct for future audited editors.
        changed.extend(relative for relative in self._baseline if relative not in current)
        unique = sorted(set(changed))
        outside = [relative for relative in unique if not self._write_allowed(relative)]
        if outside:
            raise RuntimeError(
                "transaction changed paths outside declared write locks: "
                + ", ".join(outside[:10])
            )
        return unique

    def _assert_unchanged(self, relative: str) -> None:
        target = self.workspace / Path(relative)
        expected_kind = self._baseline_kinds.get(relative, "missing")
        actual_kind = _path_kind(target)
        if actual_kind == "symlink":
            raise RuntimeError(f"workspace target became a symbolic link: {relative}")
        if actual_kind != expected_kind:
            raise RuntimeError(f"workspace changed during tool turn: {relative}")
        if _digest(target) != self._baseline.get(relative):
            raise RuntimeError(f"workspace changed during tool turn: {relative}")

    def _assert_subtree_unchanged(self, relative: str) -> None:
        root = self.workspace / Path(relative)
        current: set[str] = set()
        if root.exists():
            for child in root.rglob("*"):
                child_relative = child.relative_to(self.workspace)
                if any(part in _SNAPSHOT_IGNORES for part in child_relative.parts):
                    continue
                if child.is_symlink():
                    raise RuntimeError(
                        "workspace subtree gained a symbolic link during tool turn: "
                        f"{child_relative.as_posix()}"
                    )
                current.add(child_relative.as_posix())
        if frozenset(current) != self._subtree_members[relative]:
            raise RuntimeError(f"workspace subtree changed during tool turn: {relative}")

    def commit(self) -> list[str]:
        if self._closed:
            raise RuntimeError("workspace transaction is already closed")
        changed = self.changed_paths()
        existed_map = {
            relative: (self.workspace / Path(relative)).exists() for relative in changed
        }
        self.journal.record(
            RecoveryState.COMMIT_STARTED,
            changed=changed,
            existed=existed_map,
            overlay=str(self._container),
        )
        # Validate the complete declared read/write baseline, not only files that
        # happened to change in the overlay.
        # Include overlay-created paths: they were absent at admission and must not
        # overwrite a file another process created while the model was streaming.
        for relative in set(self._baseline) | set(changed):
            self._assert_unchanged(relative)
        for relative in self._subtree_members:
            self._assert_subtree_unchanged(relative)

        recovery = self._container / "recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        applied: list[_Replacement] = []
        try:
            for relative in changed:
                source = self.overlay / Path(relative)
                target = self.workspace / Path(relative)
                existed = target.exists()
                backup: Path | None = None
                if existed:
                    backup = recovery / Path(relative)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup, follow_symlinks=False)
                applied.append(_Replacement(target, backup, existed))
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.parent / f".{target.name}.{self.turn_id}.tmp"
                    shutil.copy2(source, temporary, follow_symlinks=False)
                    os.replace(temporary, target)
                elif existed:
                    target.unlink()
            self.journal.record(RecoveryState.COMMITTED, changed=changed)
        except Exception as commit_error:
            restore_errors: list[str] = []
            for item in reversed(applied):
                try:
                    if item.existed and item.backup is not None:
                        os.replace(item.backup, item.target)
                    elif item.target.exists():
                        item.target.unlink()
                except OSError as exc:
                    # The durable recovery directory remains referenced by the journal.
                    restore_errors.append(f"{item.target}: {exc}")
            if restore_errors:
                self._recovery_required = True
                try:
                    self.journal.record(RecoveryState.RECOVERY_REQUIRED, errors=restore_errors)
                except JournalWriteError:
                    pass
                raise WorkspaceRecoveryRequired(
                    "workspace commit outcome requires journal recovery"
                ) from commit_error
            raise
        self._closed = True
        self.journal.discard_overlay(self._container)
        return changed

    def rollback(self, reason: str) -> None:
        if self._closed:
            return
        if self._recovery_required:
            # Keep overlay/recovery backups for startup recovery; deleting them would
            # turn an observable indeterminate commit into unrecoverable data loss.
            return
        self.journal.discard_overlay(self._container)
        self._closed = True
        self.journal.record(RecoveryState.ROLLED_BACK, reason=reason)
