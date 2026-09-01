"""Turn-scoped workspace transactions and durable execution journals.

The transaction presents built-in file tools with a private workspace snapshot.
Nothing reaches the real workspace until the authoritative model response has been
reconciled and the scheduler calls :meth:`WorkspaceTransaction.commit`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from agent_core.permission_audit import redact_secret_material


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
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":  # pragma: no branch - permissions are meaningful on POSIX only
        try:
            path.chmod(0o700)
        except OSError:
            pass


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

        canonical_root = _canonical(root)
        canonical_workspace = _canonical(workspace or canonical_root.parent)
        return cls(
            workspace=canonical_workspace,
            project_id=_project_identity(canonical_workspace),
            session_id=str(session_id),
            run_id=str(run_id),
            recovery_root=canonical_root,
            run_root=canonical_root,
            partitioned=False,
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
        candidates: list[Path] = []
        if configured_home:
            candidates.append(Path(configured_home).expanduser())
        else:
            try:
                candidates.append(Path.home() / ".polaris")
            except RuntimeError:
                pass
            candidates.append(Path(tempfile.gettempdir()) / "polaris-state")
        state_root: Path | None = None
        for candidate in candidates:
            recovery = _canonical(candidate / "recovery-journals")
            if not _contained(recovery, canonical_workspace, allow_equal=True):
                state_root = recovery
                break
        if state_root is None:
            raise JournalWriteError(
                "no recovery state directory is canonically outside the project workspace"
            )
        project_id = _project_identity(canonical_workspace)
        project_root = state_root / f"p-{project_id[:16]}"
        session_root = project_root / _identity_component("session", str(session_id))
        run_root = session_root / _identity_component("run", str(run_id))
        return cls(
            workspace=canonical_workspace,
            project_id=project_id,
            session_id=str(session_id),
            run_id=str(run_id),
            recovery_root=session_root,
            run_root=run_root,
            partitioned=True,
        )

    def for_run(self, run_id: str, run_root: str | Path) -> "JournalStorage":
        return JournalStorage(
            workspace=self.workspace,
            project_id=self.project_id,
            session_id=self.session_id,
            run_id=run_id,
            recovery_root=self.recovery_root,
            run_root=_canonical(run_root),
            partitioned=self.partitioned,
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

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def acquire(self) -> bool:
        _secure_mkdir(self.path.parent)
        handle = self.path.open("a+b")
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
        try:
            self.path.unlink()
        except OSError:
            pass


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


class WorkspaceRecoveryRequired(RuntimeError):
    """Commit failed and immediate restoration could not prove consistency."""


class TurnExecutionJournal:
    """Append-only state machine for a single assistant tool round.

    Required transitions are fsync'd.  Callers must treat :class:`JournalWriteError`
    as an action gate: external effects and workspace commit are forbidden when the
    intent cannot first be persisted.
    """

    SCHEMA_VERSION = 3
    MAX_BYTES = 16 * 1024 * 1024
    MAX_RECORDS = 10_000
    MAX_LINE_BYTES = 1024 * 1024

    def __init__(
        self,
        root: str | Path | JournalStorage,
        turn_id: str | None = None,
        *,
        _ownership: _JournalOwnership | None = None,
        _nonce: str | None = None,
        _sequence: int = 0,
        _previous_checksum: str = "",
    ) -> None:
        self.storage = root if isinstance(root, JournalStorage) else JournalStorage.local(root)
        self.root = self.storage.run_root
        self.turn_id = turn_id or uuid.uuid4().hex
        if not _TURN_ID.fullmatch(self.turn_id):
            raise JournalWriteError("turn journal id must be a 32-character lowercase hex value")
        self.path = self.root / f"{self.turn_id}.jsonl"
        self.nonce = _nonce or uuid.uuid4().hex
        if not _NONCE.fullmatch(self.nonce):
            raise JournalWriteError("turn journal nonce is invalid")
        self.overlay_root = self.root / "overlays"
        self._ownership = _ownership or _JournalOwnership(
            self.root / f"{self.turn_id}.lock"
        )
        if _ownership is None and not self._ownership.acquire():
            raise JournalWriteError(f"turn journal {self.turn_id!r} is already owned")
        self._sequence = _sequence
        self._previous_checksum = _previous_checksum
        self._requests: queue.Queue[
            tuple[str, dict[str, Any], threading.Event | None, list[BaseException]] | None
        ] = queue.Queue()
        self._closed = False
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"turn-journal-{self.turn_id[:8]}",
            daemon=True,
        )
        self._writer.start()

    def record(self, state: str, **payload: Any) -> None:
        """Enqueue a required transition and wait for its durable fsync ack."""

        if self._closed:
            raise JournalWriteError("turn journal is closed")
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
            self._requests.put((state, _redact_recovery_payload(payload), None, []))

    def _writer_loop(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            state, payload, ack, errors = request
            record: dict[str, Any] = {
                **payload,
                "v": self.SCHEMA_VERSION,
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
                _secure_mkdir(self.root)
                with self.path.open("a", encoding="utf-8", errors="strict") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._sequence += 1
                self._previous_checksum = checksum
            except OSError as exc:
                errors.append(exc)
            finally:
                if ack is not None:
                    ack.set()

    def close(self) -> None:
        if self._closed:
            return
        # A durable no-op makes every earlier telemetry request visible first.
        try:
            self.record("journal_closed")
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

        _secure_mkdir(self.overlay_root)
        container = Path(
            tempfile.mkdtemp(prefix=f"polaris-turn-{turn_id[:8]}-", dir=self.overlay_root)
        )
        if not _contained(container, self.overlay_root):  # pragma: no cover - mkdtemp invariant
            raise JournalWriteError("created overlay escaped its controlled root")
        return container

    def discard_overlay(self, overlay: str | Path) -> None:
        """Delete only a canonical child of this journal's controlled overlay root."""

        candidate = _canonical(overlay)
        if not _contained(candidate, self.overlay_root):
            raise JournalWriteError("refusing to delete an overlay outside its controlled root")
        if candidate.exists() or candidate.is_symlink():
            shutil.rmtree(candidate)

    @classmethod
    def _load_verified(
        cls, path: str | Path
    ) -> tuple[list[dict[str, Any]], str | None]:
        records: list[dict[str, Any]] = []
        journal_path = Path(path)
        try:
            if journal_path.stat().st_size > cls.MAX_BYTES:
                return records, "journal_too_large"
            raw = journal_path.read_bytes()
        except OSError as exc:
            return records, f"journal_read_failed:{type(exc).__name__}"
        if not raw or not raw.endswith(b"\n"):
            return records, "journal_truncated"
        lines = raw.splitlines()
        if len(lines) > cls.MAX_RECORDS:
            return records, "too_many_records"
        previous_checksum = ""
        expected_owner: dict[str, Any] | None = None
        for expected_sequence, raw_line in enumerate(lines):
            if len(raw_line) > cls.MAX_LINE_BYTES:
                return [], "record_too_large"
            try:
                value = json.loads(raw_line.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, ValueError):
                return [], "invalid_json"
            if not isinstance(value, dict):
                return [], "record_not_object"
            if value.get("v") != cls.SCHEMA_VERSION:
                return [], "unsupported_schema"
            if value.get("sequence") != expected_sequence:
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
            canonical = json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(checksum, actual):
                return [], "checksum_mismatch"
            previous_checksum = checksum
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
            overlay = _canonical(overlay_values[0])
            if any(_canonical(value) != overlay for value in overlay_values[1:]):
                return None, "overlay_changed"
            if not _contained(overlay, overlay_root) or overlay.parent != _canonical(overlay_root):
                return None, "overlay_outside_controlled_root"
            if not overlay.name.startswith("polaris-turn-"):
                return None, "invalid_overlay_name"
            if overlay.is_symlink() or (overlay.exists() and not overlay.is_dir()):
                return None, "unsafe_overlay_kind"

        opened = [item for item in records if item.get("state") == "transaction_opened"]
        for record in opened:
            workspace = record.get("workspace")
            if not isinstance(workspace, str) or _canonical(workspace) != storage.workspace:
                return None, "workspace_owner_mismatch"
        if opened and overlay is None:
            return None, "transaction_missing_overlay"

        commit = next(
            (item for item in reversed(records) if item.get("state") == "commit_started"),
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
                target_kind = _path_kind(target)
                if target_kind == "symlink":
                    return None, "workspace_symlink_escape"
                did_exist = existed[relative]
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
            (item for item in reversed(records) if item.get("state") == "committed"),
            None,
        )
        if committed is not None:
            changed = committed.get("changed")
            if not isinstance(changed, list) or any(cls._safe_relative(item) is None for item in changed):
                return None, "invalid_committed_paths"
        return {"overlay": overlay, "replacements": replacements}, None

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
                _secure_mkdir(storage.recovery_root)
                audit_path = storage.recovery_root / "recovery-audit.log"
                with audit_path.open("a", encoding="utf-8", errors="strict") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError:
            return False
        if audit_writer is not None:
            try:
                audit_writer(dict(body))
            except Exception:
                pass
        return True

    @classmethod
    def _journal_paths(cls, storage: JournalStorage) -> list[Path]:
        base = storage.recovery_root
        if not base.exists() or not base.is_dir():
            return []
        if not storage.partitioned:
            return sorted(base.glob("*.jsonl"))
        paths: list[Path] = []
        for run_root in sorted(base.glob("r-*")):
            if run_root.is_symlink() or not run_root.is_dir() or not _contained(run_root, base):
                cls._audit(
                    storage,
                    {"phase": "validation", "status": "rejected_run_directory", "dry_run": True},
                    None,
                )
                continue
            paths.extend(sorted(run_root.glob("*.jsonl")))
        return paths

    @classmethod
    def recover_all(
        cls,
        storage: JournalStorage,
        *,
        history_writer: Callable[[dict[str, Any]], bool] | None = None,
        dry_run: bool = False,
        audit_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, str]]:
        """Recover only verified, runtime-owned journals without retrying external work.

        Validation, ownership, checksums, and every canonical path are checked before
        the first mutation. ``dry_run`` emits the same independent audit plan without
        writing a journal, transcript, workspace path, or overlay.
        """

        outcomes: list[dict[str, str]] = []
        for path in cls._journal_paths(storage):
            if path.is_symlink() or not _contained(path, storage.recovery_root):
                reason = "journal_path_escape"
                cls._audit(
                    storage,
                    {"turn_id": path.stem, "phase": "validation", "status": "rejected", "reason": reason, "dry_run": dry_run},
                    audit_writer,
                )
                outcomes.append({"turn_id": path.stem, "status": "rejected", "reason": reason})
                continue
            if not _TURN_ID.fullmatch(path.stem):
                reason = "invalid_turn_filename"
                cls._audit(
                    storage,
                    {"turn_id": path.stem, "phase": "validation", "status": "rejected", "reason": reason, "dry_run": dry_run},
                    audit_writer,
                )
                outcomes.append({"turn_id": path.stem, "status": "rejected", "reason": reason})
                continue
            ownership = _JournalOwnership(path.with_suffix(".lock"))
            if not ownership.acquire():
                continue
            records, error = cls._load_verified(path)
            if error is not None:
                ownership.release()
                cls._audit(
                    storage,
                    {"turn_id": path.stem, "phase": "validation", "status": "rejected", "reason": error, "dry_run": dry_run},
                    audit_writer,
                )
                outcomes.append({"turn_id": path.stem, "status": "rejected", "reason": error})
                continue
            journal_storage, error = cls._validate_owner(storage, path, records)
            if error is not None or journal_storage is None:
                ownership.release()
                reason = error or "invalid_owner"
                cls._audit(
                    storage,
                    {"turn_id": path.stem, "phase": "ownership", "status": "foreign", "reason": reason, "dry_run": dry_run},
                    audit_writer,
                )
                outcomes.append({"turn_id": path.stem, "status": "foreign", "reason": reason})
                continue
            states = [str(item["state"]) for item in records]
            turn_id = path.stem
            if states[-1] in {
                "history_persisted",
                "rolled_back",
                "indeterminate_external_effect",
                "external_effect_history_missing",
                "journal_closed",
            }:
                ownership.release()
                continue
            plan, error = cls._recovery_plan(journal_storage, path, records)
            if error is not None or plan is None:
                ownership.release()
                reason = error or "invalid_recovery_plan"
                cls._audit(
                    storage,
                    {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "planning", "status": "rejected", "reason": reason, "dry_run": dry_run},
                    audit_writer,
                )
                outcomes.append({"turn_id": turn_id, "status": "rejected", "reason": reason})
                continue
            journal = cls(
                journal_storage,
                turn_id,
                _ownership=ownership,
                _nonce=str(records[0]["owner"]["nonce"]),
                _sequence=len(records),
                _previous_checksum=str(records[-1]["checksum"]),
            )
            history_payload = next(
                (item.get("history_payload") for item in reversed(records) if item.get("history_payload")),
                None,
            )
            history_recoverable = "committed" in states or "external_outcome" in states
            action = "rollback_workspace" if "committed" not in states else "recover_history"
            if history_recoverable and history_writer is not None and isinstance(history_payload, dict):
                action = "recover_history"
            if dry_run:
                journal._release_for_later_recovery()
                status = f"would_{action}"
                cls._audit(
                    storage,
                    {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "plan", "status": status, "actions": len(plan["replacements"]), "dry_run": True},
                    audit_writer,
                )
                outcomes.append({"turn_id": turn_id, "status": status})
                continue
            if not cls._audit(
                storage,
                {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "before", "status": "authorized", "action": action, "actions": len(plan["replacements"]), "dry_run": False},
                audit_writer,
            ):
                journal._release_for_later_recovery()
                outcomes.append({"turn_id": turn_id, "status": "audit_failed"})
                continue
            if (
                history_recoverable
                and history_writer is not None
                and isinstance(history_payload, dict)
            ):
                if history_writer(history_payload):
                    journal.record("history_persisted", recovery=True)
                    overlay = plan["overlay"]
                    if "committed" in states and overlay is not None:
                        journal.discard_overlay(overlay)
                    outcomes.append({"turn_id": turn_id, "status": "history_persisted"})
                    cls._audit(
                        storage,
                        {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "after", "status": "history_persisted", "dry_run": False},
                        audit_writer,
                    )
                    journal.close()
                    continue
            overlay = plan["overlay"]
            if "committed" not in states:
                recovery_failed = False
                for replacement in reversed(plan["replacements"]):
                    target = replacement["target"]
                    backup = replacement["backup"]
                    try:
                        if (
                            not _contained(target, journal_storage.workspace)
                            or target.is_symlink()
                            or not _contained(backup, Path(plan["overlay"]) / "recovery")
                            or (replacement["existed"] and (backup.is_symlink() or not backup.is_file()))
                        ):
                            raise OSError("recovery path changed after validation")
                        if replacement["existed"]:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(backup, target)
                        elif target.exists():
                            target.unlink()
                    except OSError:
                        outcomes.append({"turn_id": turn_id, "status": "recovery_failed"})
                        recovery_failed = True
                        break
                if recovery_failed:
                    journal._release_for_later_recovery()
                    continue
                if overlay is not None:
                    journal.discard_overlay(overlay)
                if "external_intent" in states and "external_outcome" not in states:
                    journal.record("indeterminate_external_effect")
                    outcomes.append({"turn_id": turn_id, "status": "IndeterminateExternalEffect"})
                elif "external_outcome" in states:
                    journal.record("external_effect_history_missing")
                    outcomes.append(
                        {"turn_id": turn_id, "status": "external_effect_history_missing"}
                    )
                else:
                    journal.record("rolled_back", recovery=True)
                    outcomes.append({"turn_id": turn_id, "status": "rolled_back"})
                cls._audit(
                    storage,
                    {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "after", "status": outcomes[-1]["status"], "dry_run": False},
                    audit_writer,
                )
                journal.close()
                continue
            if overlay is not None:
                journal.discard_overlay(overlay)
            outcomes.append({"turn_id": turn_id, "status": "committed_history_pending"})
            cls._audit(
                storage,
                {"turn_id": turn_id, "owner_run_id": journal_storage.run_id, "phase": "after", "status": "committed_history_pending", "dry_run": False},
                audit_writer,
            )
            journal._release_for_later_recovery()
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
        self.journal.record("transaction_opened", overlay=str(self._container), workspace=str(self.workspace))

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
            "commit_started",
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
            self.journal.record("committed", changed=changed)
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
                    self.journal.record("recovery_required", errors=restore_errors)
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
        self.journal.record("rolled_back", reason=reason)
