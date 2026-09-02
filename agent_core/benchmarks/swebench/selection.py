"""Safe instance selection manifests for full and manual smoke runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SWEbenchSelection


def load_selection(path: str | Path) -> SWEbenchSelection:
    file = Path(path).expanduser()
    if not file.exists():
        raise FileNotFoundError(file)
    if file.suffix.casefold() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML selection files require PyYAML") from exc
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    else:
        raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selection manifest must be an object")
    dataset = str(raw.get("dataset") or "SWE-bench/SWE-bench_Lite")
    relative_dataset = (file.parent / dataset).expanduser()
    if relative_dataset.exists():
        dataset = str(relative_dataset.resolve())
    split = str(raw.get("split") or "test")
    values = raw.get("instance_ids", raw.get("instances", []))
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("selection.instance_ids must be a list of IDs")
    ids = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not ids:
        raise ValueError("selection manifest contains no instance IDs")
    return SWEbenchSelection(dataset, split, ids, source=str(file.resolve()))


def make_selection(
    *,
    dataset: str,
    split: str,
    instance_ids: list[str] | tuple[str, ...] | None = None,
    selection_file: str | Path | None = None,
    limit: int | None = None,
    all_instances: bool = False,
    smoke: bool = False,
) -> SWEbenchSelection:
    if selection_file is not None:
        selection = load_selection(selection_file)
        # A manifest is an explicit, reviewable selection.  Its dataset/split are
        # authoritative so the same file works for both ``smoke`` (dev) and ``run``
        # (test) subcommands, whose parser defaults differ.
        selected = list(selection.instance_ids)
        dataset, split = selection.dataset, selection.split
    else:
        selected = [str(value).strip() for value in (instance_ids or []) if str(value).strip()]
    selected = list(dict.fromkeys(selected))
    if smoke and len(selected) > 5:
        raise ValueError("smoke mode accepts at most 5 manually selected instances")
    if smoke and all_instances:
        raise ValueError("smoke mode requires explicit --instance-id values or --selection; --all is disabled")
    if not selected and not all_instances:
        if limit is None:
            raise ValueError("choose --instance-id/--selection, or explicitly pass --all")
        if limit <= 0:
            raise ValueError("--limit must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    return SWEbenchSelection(dataset, split, tuple(selected), source="cli" if selection_file is None else str(selection_file))


def write_selection(path: str | Path, selection: SWEbenchSelection) -> Path:
    """Write IDs only; this file must never contain gold patch/test fields."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset": selection.dataset,
        "split": selection.split,
        "instance_ids": list(selection.instance_ids),
    }
    if target.suffix.casefold() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML selection files require PyYAML") from exc
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
