"""SWE-bench dataset loading with a dependency-free local-file fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .models import SWEbenchInstance


class SWEbenchDataset:
    """A small, deterministic view over one SWE-bench split.

    ``datasets`` is imported lazily.  This makes unit tests and local selection
    manifests work in a minimal installation while still using the official
    HuggingFace release by default.
    """

    def __init__(
        self,
        instances: Iterable[SWEbenchInstance],
        *,
        dataset: str,
        split: str,
        source: str = "memory",
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.source = source
        self._instances = tuple(instances)
        self._by_id = {item.instance_id: item for item in self._instances}
        if len(self._by_id) != len(self._instances):
            raise ValueError("SWE-bench dataset contains duplicate instance_id values")

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        dataset: str = "memory",
        split: str = "unknown",
        include_gold: bool = False,
        source: str = "memory",
    ) -> "SWEbenchDataset":
        return cls(
            (SWEbenchInstance.from_mapping(row, include_gold=include_gold) for row in rows),
            dataset=dataset,
            split=split,
            source=source,
        )

    @classmethod
    def load(
        cls,
        dataset: str = "SWE-bench/SWE-bench_Lite",
        split: str = "test",
        *,
        cache_dir: str | Path | None = None,
        include_gold: bool = False,
        trust_remote_code: bool = False,
    ) -> "SWEbenchDataset":
        path = Path(dataset).expanduser()
        if path.exists():
            rows = _read_local_rows(path)
            return cls.from_rows(
                rows,
                dataset=str(path.resolve()),
                split=split,
                include_gold=include_gold,
                source=str(path.resolve()),
            )

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "SWE-bench loading needs the optional benchmark dependencies. "
                "Install with `pip install -e '.[swebench]'`, or pass a local JSON/JSONL file."
            ) from exc

        kwargs: dict[str, Any] = {"split": split, "trust_remote_code": trust_remote_code}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        try:
            table = load_dataset(dataset, **kwargs)
        except TypeError:
            # Older datasets versions do not accept trust_remote_code for all builders.
            kwargs.pop("trust_remote_code", None)
            table = load_dataset(dataset, **kwargs)
        return cls.from_rows(
            (dict(row) for row in table),
            dataset=dataset,
            split=split,
            include_gold=include_gold,
            source=f"huggingface:{dataset}:{split}",
        )

    def __iter__(self) -> Iterator[SWEbenchInstance]:
        return iter(self._instances)

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, instance_id: str) -> SWEbenchInstance:
        return self._by_id[instance_id]

    def get(self, instance_id: str) -> SWEbenchInstance | None:
        return self._by_id.get(instance_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(item.instance_id for item in self._instances)

    def select(self, instance_ids: Iterable[str]) -> tuple[SWEbenchInstance, ...]:
        requested = tuple(dict.fromkeys(str(item).strip() for item in instance_ids if str(item).strip()))
        missing = [item for item in requested if item not in self._by_id]
        if missing:
            preview = ", ".join(missing[:8])
            more = " ..." if len(missing) > 8 else ""
            raise KeyError(f"instance_id not found in {self.dataset}/{self.split}: {preview}{more}")
        return tuple(self._by_id[item] for item in requested)

    def digest(self, instance_ids: Iterable[str] | None = None) -> str:
        selected = self._instances if instance_ids is None else self.select(instance_ids)
        payload = "\n".join(f"{item.instance_id}\t{item.base_commit}" for item in selected)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_rows(self) -> Iterator[dict[str, Any]]:
        for item in self._instances:
            yield item.public_dict()


def load_swebench_dataset(*args: Any, **kwargs: Any) -> SWEbenchDataset:
    """Convenience wrapper retained for callers that prefer a function API."""
    return SWEbenchDataset.load(*args, **kwargs)


def _read_local_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(value)
        return rows
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML selection/data files require PyYAML") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON dataset {path}: {exc}") from exc
    if isinstance(value, Mapping):
        # Support a common export shape: {"data": [...]} or {"instances": [...]}.
        for key in ("data", "instances", "rows"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                value = candidate
                break
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"local SWE-bench data must be a list of objects: {path}")
    return value
