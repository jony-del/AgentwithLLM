"""Turn-scoped workspace transactions and durable execution journals.

The transaction presents built-in file tools with a private workspace snapshot.
Nothing reaches the real workspace until the authoritative model response has been
reconciled and the scheduler calls :meth:`WorkspaceTransaction.commit`.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    SCHEMA_VERSION = 2

    def __init__(
        self,
        root: str | Path,
        turn_id: str | None = None,
        *,
        _ownership: _JournalOwnership | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.turn_id = turn_id or uuid.uuid4().hex
        self.path = self.root / f"{self.turn_id}.jsonl"
        self._ownership = _ownership or _JournalOwnership(
            self.root / f"{self.turn_id}.lock"
        )
        if _ownership is None and not self._ownership.acquire():
            raise JournalWriteError(f"turn journal {self.turn_id!r} is already owned")
        self._sequence = 0
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
            record = {
                "v": self.SCHEMA_VERSION,
                "turn_id": self.turn_id,
                "sequence": self._sequence,
                "state": state,
                "ts": time.time(),
                "pid": os.getpid(),
                "process_token": _PROCESS_TOKEN,
                **payload,
            }
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", errors="strict") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._sequence += 1
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

    @staticmethod
    def load(path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return records
        for line in lines:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get("state"), str):
                records.append(value)
        return records

    @classmethod
    def recover_all(
        cls,
        root: str | Path,
        *,
        history_writer: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, str]]:
        """Recover unfinished journals without retrying external operations.

        Uncommitted overlays are discarded.  A committed round can be supplied to a
        transcript writer if its durable history payload was recorded.  An external
        intent with no outcome is explicitly marked indeterminate.
        """

        outcomes: list[dict[str, str]] = []
        base = Path(root)
        if not base.exists():
            return outcomes
        for path in sorted(base.glob("*.jsonl")):
            records = cls.load(path)
            if not records:
                continue
            states = [str(item["state"]) for item in records]
            turn_id = str(records[-1].get("turn_id") or path.stem)
            ownership = _JournalOwnership(base / f"{turn_id}.lock")
            if not ownership.acquire():
                # An active process, even one reusing a historical PID, still owns it.
                continue
            if states[-1] in {
                "history_persisted",
                "rolled_back",
                "indeterminate_external_effect",
                "external_effect_history_missing",
                "journal_closed",
            }:
                ownership.release()
                continue
            journal = cls(base, turn_id, _ownership=ownership)
            journal.path = path
            journal._sequence = max(int(item.get("sequence", 0)) for item in records) + 1
            history_payload = next(
                (item.get("history_payload") for item in reversed(records) if item.get("history_payload")),
                None,
            )
            history_recoverable = "committed" in states or "external_outcome" in states
            if (
                history_recoverable
                and history_writer is not None
                and isinstance(history_payload, dict)
            ):
                if history_writer(history_payload):
                    journal.record("history_persisted", recovery=True)
                    committed_overlay = next(
                        (
                            str(item.get("overlay"))
                            for item in reversed(records)
                            if item.get("overlay")
                        ),
                        "",
                    )
                    if "committed" in states and committed_overlay:
                        shutil.rmtree(Path(committed_overlay), ignore_errors=True)
                    outcomes.append({"turn_id": turn_id, "status": "history_persisted"})
                    journal.close()
                    continue
            overlay = next(
                (str(item.get("overlay")) for item in reversed(records) if item.get("overlay")),
                "",
            )
            if "committed" not in states:
                recovery_failed = False
                commit = next(
                    (item for item in reversed(records) if item.get("state") == "commit_started"),
                    None,
                )
                opened = next(
                    (item for item in reversed(records) if item.get("workspace")),
                    None,
                )
                if isinstance(commit, dict) and isinstance(opened, dict) and overlay:
                    workspace = Path(str(opened.get("workspace"))).resolve()
                    recovery = Path(overlay) / "recovery"
                    changed = commit.get("changed", [])
                    existed = commit.get("existed", {})
                    if isinstance(changed, list) and isinstance(existed, dict):
                        for raw_relative in reversed(changed):
                            relative = str(raw_relative)
                            target = workspace / Path(relative)
                            backup = recovery / Path(relative)
                            try:
                                if bool(existed.get(relative)) and backup.is_file():
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    os.replace(backup, target)
                                elif not bool(existed.get(relative)) and target.exists():
                                    target.unlink()
                            except OSError:
                                # Leave the journal unfinished for a later/operator retry.
                                outcomes.append({"turn_id": turn_id, "status": "recovery_failed"})
                                recovery_failed = True
                                break
                if recovery_failed:
                    journal._release_for_later_recovery()
                    continue
                if overlay:
                    shutil.rmtree(Path(overlay), ignore_errors=True)
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
                journal.close()
                continue
            if overlay:
                shutil.rmtree(Path(overlay), ignore_errors=True)
            outcomes.append({"turn_id": turn_id, "status": "committed_history_pending"})
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
        self._container = Path(tempfile.mkdtemp(prefix=f"polaris-turn-{turn_id[:8]}-"))
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
        shutil.rmtree(self._container, ignore_errors=True)
        return changed

    def rollback(self, reason: str) -> None:
        if self._closed:
            return
        if self._recovery_required:
            # Keep overlay/recovery backups for startup recovery; deleting them would
            # turn an observable indeterminate commit into unrecoverable data loss.
            return
        shutil.rmtree(self._container, ignore_errors=True)
        self._closed = True
        self.journal.record("rolled_back", reason=reason)
