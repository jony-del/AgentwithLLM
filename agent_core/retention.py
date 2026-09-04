"""Conservative automatic retention for resumable transcripts."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from agent_core.session import SessionRetentionConfig
from agent_core.transcript import load_transcript, project_dir


def _contained(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def prune_sessions(
    root: str | Path,
    workspace: str | Path,
    config: SessionRetentionConfig,
    *,
    current_session_id: str | None = None,
    apply: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    project = project_dir(root, workspace).resolve()
    report: dict[str, Any] = {
        "dry_run": not apply,
        "project": project.name,
        "selected": [],
        "deleted": 0,
        "bytes": 0,
        "protected": 0,
        "errors": 0,
    }
    if not config.enabled or not project.is_dir():
        return report
    eligible: list[tuple[Path, float, int]] = []
    for path in project.glob("*.jsonl"):
        if path.stem == current_session_id or not _contained(path, project):
            report["protected"] += 1
            continue
        try:
            stat = path.stat()
            loaded = load_transcript(path, skip_precompact=False)
        except (OSError, UnicodeError, ValueError, TypeError):
            report["protected"] += 1
            continue
        active = False
        for marker in project.glob(path.name + ".active.*"):
            try:
                if now - marker.stat().st_mtime <= 7 * 24 * 3600:
                    active = True
                    break
            except OSError:
                active = True
                break
        if active or loaded.diagnostics or (config.preserve_tagged and bool(loaded.tag)):
            report["protected"] += 1
            continue
        eligible.append((path, stat.st_mtime, stat.st_size))
    eligible.sort(key=lambda item: item[1], reverse=True)
    cutoff = now - config.transcript_days * 86_400
    selected = [
        item for index, item in enumerate(eligible)
        if item[1] < cutoff or index >= config.max_transcripts_per_project
    ]
    for path, _modified, size in selected:
        report["selected"].append(path.stem)
        report["bytes"] += size
        if not apply:
            continue
        try:
            if not _contained(path, project):
                raise OSError("retention target escaped project root")
            path.unlink()
            for sidecar in (
                path.with_suffix(path.suffix + ".round-index.json"),
            ):
                if sidecar.exists() and _contained(sidecar, project):
                    sidecar.unlink()
            for activity in project.glob(path.name + ".active.*"):
                if _contained(activity, project):
                    activity.unlink(missing_ok=True)
            sidechains = project / path.stem
            if sidechains.exists() and _contained(sidechains, project):
                shutil.rmtree(sidechains)
            report["deleted"] += 1
        except OSError:
            report["errors"] += 1
    return report


def maybe_prune_sessions(
    root: str | Path,
    workspace: str | Path,
    config: SessionRetentionConfig,
    *,
    current_session_id: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    now = time.time() if now is None else now
    project = project_dir(root, workspace)
    if not config.enabled or not project.is_dir():
        return None
    marker = project / ".retention-state.json"
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if now - float(state.get("last_scan", 0)) < config.scan_interval_seconds:
            return None
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    report = prune_sessions(
        root, workspace, config, current_session_id=current_session_id,
        apply=True, now=now,
    )
    temporary = marker.with_suffix(marker.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({"v": 1, "last_scan": now}), encoding="utf-8")
        os.replace(temporary, marker)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return report
