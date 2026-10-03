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
from agent_core.codeintel.snapshots import IGNORED_DIRS
from agent_core.permission_safety import is_secret_path

logger = logging.getLogger(__name__)


async def maintain(service: Any) -> None:
    changed: OrderedDict[str, None] = OrderedDict()
    guard = threading.Lock()
    rescan = threading.Event()
    watched: set[str] = set()
    handler: Any = None

    def event(raw: str, directory: bool) -> None:
        try:
            relative = Path(raw).relative_to(service.workspace)
        except ValueError:
            return
        if any(p in IGNORED_DIRS for p in relative.parts) or is_secret_path(relative):
            return
        if directory:
            rescan.set()
            return
        with guard:
            if len(changed) >= 10000:
                rescan.set()
                return
            changed[relative.as_posix()] = None

    if not service.config.watch:
        service.watch_state = "periodic_reconciliation_only"
    else:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            class Handler(FileSystemEventHandler):
                def on_any_event(self, value: Any) -> None:
                    if value.event_type in {"created", "modified", "deleted", "moved"}:
                        event(value.src_path, value.is_directory)
                        if getattr(value, "dest_path", ""):
                            event(value.dest_path, value.is_directory)

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
            with guard:
                paths = tuple(changed)
                changed.clear()
            if paths:
                await service.publish_changes(ChangeSet(uuid.uuid4().hex, paths, "watcher"))
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
                    status = await service.ensure_index(reconcile=reconcile, audit=audit)
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
            service.watch_state = f"degraded:{type(exc).__name__}; reconciliation pending"
            logger.warning("code index maintenance failed: %s", exc)
        await asyncio.sleep(1)
