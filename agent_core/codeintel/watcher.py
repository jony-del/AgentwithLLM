"""Optional bounded OS watches backed by mandatory incremental reconciliation."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from agent_core.codeintel.models import ChangeSet
from agent_core.codeintel.paths import excluded_path
from agent_core.permission_safety import is_secret_path

logger = logging.getLogger(__name__)


class WatchChanges:
    """Bounded coalescing queue; publication acknowledges only observed versions.

    Entries stay queued during IO, including failed/cancelled publication. A newer
    event for an in-flight path cannot be removed by an older acknowledgement.
    Overflow requests inventory reconciliation instead of growing the queue.
    """

    def __init__(self, workspace: Path, exclude_paths: tuple[str, ...], *,
                 capacity: int = 10000, batch_size: int = 256) -> None:
        self.workspace = workspace
        self.exclude_paths = exclude_paths
        self.capacity = capacity
        self.batch_size = batch_size
        self.changed: OrderedDict[str, int] = OrderedDict()
        self.guard = threading.Lock()
        self.rescan = threading.Event()
        self.sequence = 0

    def event(self, raw: str, directory: bool) -> None:
        try:
            relative = Path(raw).relative_to(self.workspace)
        except ValueError:
            return
        if ".." in relative.parts:
            return
        if excluded_path(relative.as_posix(), self.exclude_paths, directory=directory) or is_secret_path(relative):
            return
        if directory:
            self.rescan.set()
            return
        path = relative.as_posix()
        with self.guard:
            if path not in self.changed and len(self.changed) >= self.capacity:
                self.rescan.set()
                return
            self.sequence += 1
            self.changed[path] = self.sequence

    def batch(self) -> tuple[tuple[str, int], ...]:
        from itertools import islice
        with self.guard:
            return tuple(islice(self.changed.items(), self.batch_size))

    def acknowledge(self, batch: tuple[tuple[str, int], ...]) -> None:
        with self.guard:
            for path, sequence in batch:
                if self.changed.get(path) == sequence:
                    del self.changed[path]


async def publish_batch(service: Any, changed: WatchChanges) -> None:
    batch = changed.batch()
    if batch:
        await service.publish_changes(ChangeSet(uuid.uuid4().hex, tuple(path for path, _ in batch), "watcher"))
        changed.acknowledge(batch)


async def maintain(service: Any) -> None:
    changed = WatchChanges(service.workspace, service.config.exclude_paths)
    rescan = changed.rescan
    watched: set[str] = set()
    handler: Any = None

    if not service.config.watch:
        service.watch_state = "periodic_reconciliation_only"
    else:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class Handler(FileSystemEventHandler):
                def on_any_event(self, value: Any) -> None:
                    if value.event_type in {"created", "modified", "deleted", "moved"}:
                        changed.event(value.src_path, value.is_directory)
                        if getattr(value, "dest_path", ""):
                            changed.event(value.dest_path, value.is_directory)

            handler = Handler()
            service._observer = Observer()
            service._observer.schedule(handler, str(service.workspace), recursive=False)
            watched.add(str(service.workspace))
            service._observer.start()
            service.watch_state = "os_events (up to 512 directories) + periodic reconciliation"
        except (ImportError, OSError, RuntimeError) as exc:
            service.watch_state = "periodic_reconciliation_only"
            logger.info("code watcher unavailable (%s); periodic reconciliation active", exc)
    last_reconcile, last_audit = 0.0, 0.0
    quota_blocked_until = 0.0
    while not service._closed:
        try:
            await publish_batch(service, changed)
            now = time.monotonic()
            reconcile = rescan.is_set() or now - last_reconcile >= service.config.reconcile_seconds
            if reconcile:
                rescan.clear()
                last_reconcile = now
            # Low-frequency hash audit catches timestamp-preserving edits missed by OS events.
            audit = now - last_audit >= max(3600, service.config.reconcile_seconds * 60)
            if audit:
                last_audit = now
            if reconcile or audit or await service.has_pending_work():
                if now < quota_blocked_until and not reconcile:
                    pass  # quota backoff: idle slices retry only on reconcile ticks
                else:
                    status = await service.ensure_index(reconcile=reconcile, audit=audit, detailed=False)
                    if status.get("stop_reason") == "index_disk_quota":
                        quota_blocked_until = now + max(300.0, service.config.reconcile_seconds * 10)
                        service.watch_state = "degraded:index_disk_quota; index writes paused, queries unaffected"
                    elif quota_blocked_until:
                        quota_blocked_until = 0.0
            if reconcile and service._observer is not None and handler is not None and len(watched) < 512:
                def directories() -> list[str]:
                    with service.store.connect() as db:
                        rows = db.execute("SELECT path FROM files ORDER BY path LIMIT 2000").fetchall()
                    return sorted({str((service.workspace / row[0]).parent) for row in rows})
                for directory in await asyncio.to_thread(directories):
                    if directory not in watched and len(watched) < 512:
                        service._observer.schedule(handler, directory, recursive=False)
                        watched.add(directory)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            rescan.set()
            service.watch_state = f"degraded:{type(exc).__name__}; reconciliation pending"
            logger.warning("code index maintenance failed: %s", exc)
        await asyncio.sleep(1)
