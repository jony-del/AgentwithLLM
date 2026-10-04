from __future__ import annotations

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from agent_core.codeintel.config import CodeIntelConfig
from agent_core.codeintel.service import CodeIntelligenceService
from agent_core.codeintel.working_set import TaskWorkingSet
from agent_core.codeintel.store import default_store

logger = logging.getLogger(__name__)
_SERVICES: WeakKeyDictionary[Any, dict[Any, tuple[CodeIntelligenceService, int]]] = WeakKeyDictionary()


def codeintel_enabled(session: Any) -> bool:
    """A missing config means defaults; defaults have the layer enabled."""
    config = getattr(session, "codeintel_config", None)
    return config is None or bool(config.enabled)


async def get_service(session: Any) -> CodeIntelligenceService:
    config = session.codeintel_config or CodeIntelConfig()
    if not config.enabled:
        raise RuntimeError("code intelligence is disabled")
    existing = session.codeintel
    rules = session.code_permission_rules() if callable(session.code_permission_rules) else None
    restricted = bool(rules and (rules.deny or rules.ask))
    policy_key = hashlib.sha256(repr(rules).encode()).hexdigest()[:16] if restricted else ""
    if existing is not None and existing.workspace == session.workspace.resolve() and existing.policy_key == policy_key:
        return existing
    if existing is not None:
        await release_service(session)
    loop = asyncio.get_running_loop()
    pool = _SERVICES.setdefault(loop, {})
    key = (str(session.workspace.resolve()), config, policy_key)
    def allowed(path: str) -> bool:
        arguments = ({"path": path}, {"path": str((session.workspace / path).resolve())})
        return rules is None or not any(rules.deny_match(name, argument) or rules.ask_match(name, argument)
                                       for argument in arguments for name in ("read_text_file", "search_text", "code_search"))
    database = default_store(session.workspace)
    if policy_key:
        database = database.with_name(f"code-{policy_key}.sqlite3")
    service, count = pool.get(key, (CodeIntelligenceService(session.workspace, config, database=database, allowed=allowed), 0))
    service.policy_key = policy_key
    pool[key] = (service, count + 1)
    session.codeintel = service
    if session.code_working_set is None:
        session.code_working_set = TaskWorkingSet()
    if service._background is None and config.maintain:
        from agent_core.codeintel.watcher import maintain
        service._background = asyncio.create_task(maintain(service), name="codeintel-maintenance")
    return service


async def release_service(session: Any) -> None:
    service = getattr(session, "codeintel", None)
    if service is None:
        return
    session.codeintel = None
    pool = _SERVICES.get(asyncio.get_running_loop(), {})
    key = (str(service.workspace), service.config, service.policy_key)
    _, count = pool.get(key, (service, 1))
    if count <= 1:
        pool.pop(key, None)
        await service.close()
    else:
        pool[key] = (service, count - 1)


def remember_version(session: Any, path: str, version: dict[str, Any]) -> None:
    key = str((session.workspace / path).resolve())
    session.code_read_versions.pop(key, None)
    session.code_read_versions[key] = version
    while len(session.code_read_versions) > 256:
        session.code_read_versions.pop(next(iter(session.code_read_versions)))


async def notify_committed(session: Any, paths: tuple[str, ...], event_id: str) -> None:
    from agent_core.codeintel.models import ChangeSet
    if session.code_working_set is not None:
        session.code_working_set.invalidate(paths)
    for path in paths:
        session.code_read_versions.pop(str((session.workspace / path).resolve()), None)
    # Do not initialize an unused index as a side effect of ordinary editing.
    if session.codeintel is not None:
        try:
            await session.codeintel.publish_changes(ChangeSet(event_id, paths))
        except Exception as exc:
            logger.warning("code index commit notification failed; reconciliation required: %s", exc)
            # Persisted transaction journal and startup reconciliation remain recovery sources.
            session.codeintel.watch_state = "commit_notification_failed; reconciliation required"


def edit_expected(session: Any, root: Path, raw: str, explicit: Any = None) -> Any:
    return explicit if explicit is not None else session.code_read_versions.get(str((root / raw).resolve()))
