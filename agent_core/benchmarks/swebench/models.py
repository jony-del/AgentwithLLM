"""Data contracts shared by the SWE-bench loader, runner and reporter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class InstanceState(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    SOLVING = "solving"
    PATCH_READY = "patch_ready"
    EVALUATION_PENDING = "evaluation_pending"
    EVALUATING = "evaluating"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class FailureKind(str, Enum):
    DATASET = "dataset"
    SELECTION = "selection"
    PREPARE = "prepare"
    RUNTIME = "runtime"
    AGENT = "agent"
    PATCH = "patch"
    HARNESS = "harness"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _string(value: object, default: str = "") -> str:
    return default if value is None else str(value)


@dataclass(frozen=True, slots=True)
class SWEbenchInstance:
    """The safe/public portion of one SWE-bench dataset row.

    Gold patch and test-oracle fields are deliberately kept in a private mapping
    only when ``from_mapping(..., include_gold=True)`` is explicitly requested by
    an evaluation component.  ``public_dict`` never exposes those fields.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str = ""
    issue_id: str = ""
    issue_url: str = ""
    pr_url: str = ""
    hints_text: str = ""
    FAIL_TO_PASS: tuple[str, ...] = ()
    PASS_TO_PASS: tuple[str, ...] = ()
    patch: str = field(default="", repr=False, compare=False)
    test_patch: str = field(default="", repr=False, compare=False)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        include_gold: bool = False,
        strict: bool = True,
    ) -> "SWEbenchInstance":
        required = ("instance_id", "repo", "base_commit", "problem_statement")
        missing = [name for name in required if not _string(_row_value(row, name)).strip()]
        if missing and strict:
            raise ValueError(f"SWE-bench row is missing required fields: {', '.join(missing)}")
        known = {
            "instance_id", "repo", "base_commit", "problem_statement", "version",
            "issue_id", "issue_url", "pr_url", "hints_text", "FAIL_TO_PASS",
            "PASS_TO_PASS", "patch", "test_patch",
        }
        extras = {
            str(k): v
            for k, v in row.items()
            if str(k) not in known and (include_gold or not _is_oracle_key(str(k)))
        }
        return cls(
            instance_id=_string(_row_value(row, "instance_id")).strip(),
            repo=_string(_row_value(row, "repo")).strip(),
            base_commit=_string(_row_value(row, "base_commit")).strip(),
            problem_statement=_string(_row_value(row, "problem_statement")),
            version=_string(_row_value(row, "version")).strip(),
            issue_id=_string(_row_value(row, "issue_id")).strip(),
            issue_url=_string(_row_value(row, "issue_url")).strip(),
            pr_url=_string(_row_value(row, "pr_url")).strip(),
            hints_text=_string(_row_value(row, "hints_text")),
            FAIL_TO_PASS=normalize_test_list(_row_value(row, "FAIL_TO_PASS")),
            PASS_TO_PASS=normalize_test_list(_row_value(row, "PASS_TO_PASS")),
            patch=_string(_row_value(row, "patch")) if include_gold else "",
            test_patch=_string(_row_value(row, "test_patch")) if include_gold else "",
            extra=extras,
        )

    @property
    def has_gold(self) -> bool:
        return bool(self.patch or self.test_patch or self.FAIL_TO_PASS or self.PASS_TO_PASS)

    @property
    def image(self) -> str:
        """Optional published solver image carried by newer dataset releases."""
        value = self.extra.get("image")
        return str(value).strip() if value else ""

    def public_dict(self) -> dict[str, Any]:
        """Return only fields safe to write into an Agent workspace or prompt."""
        value: dict[str, Any] = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "issue_id": self.issue_id,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "version": self.version,
            "issue_url": self.issue_url,
            "hints_text": self.hints_text,
        }
        difficulty = self.extra.get("difficulty")
        if difficulty:
            value["difficulty"] = str(difficulty)
        return value

    def evaluation_dict(self) -> dict[str, Any]:
        """Return the complete row for the Harness, never for the Agent."""
        value = self.public_dict()
        value.update(
            {
                "FAIL_TO_PASS": list(self.FAIL_TO_PASS),
                "PASS_TO_PASS": list(self.PASS_TO_PASS),
                "patch": self.patch,
                "test_patch": self.test_patch,
                "pr_url": self.pr_url,
                **dict(self.extra),
            }
        )
        return value


def normalize_test_list(value: object) -> tuple[str, ...]:
    """Normalize the several encodings used by released SWE-bench datasets."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return ()
    # Dataset parquet/JSON exports sometimes store a JSON array as a string.
    if text.startswith("[") and text.endswith("]"):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, list):
                return normalize_test_list(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return tuple(line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip())


def _is_oracle_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    return any(token in normalized for token in ("gold_patch", "test_patch", "test_diff", "fail_to_pass", "pass_to_pass", "expected_patch"))


def _row_value(row: Mapping[str, Any], name: str) -> object:
    if name in row:
        return row[name]
    aliases = {
        "base_commit": ("base_sha", "commit"),
        "problem_statement": ("issue", "description"),
        "FAIL_TO_PASS": ("fail_to_pass",),
        "PASS_TO_PASS": ("pass_to_pass",),
        "test_patch": ("test_diff",),
    }
    for alias in aliases.get(name, ()):
        if alias in row:
            return row[alias]
    return ""


@dataclass(frozen=True, slots=True)
class SWEbenchSelection:
    dataset: str
    split: str
    instance_ids: tuple[str, ...]
    source: str = "cli"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "instance_ids": list(self.instance_ids),
            "source": self.source,
        }


@dataclass(slots=True)
class SWEbenchRunConfig:
    dataset: str = "SWE-bench/SWE-bench_Lite"
    split: str = "test"
    output_dir: str = "swebench_runs"
    run_id: str = ""
    model: str = ""
    provider: str = ""
    image: str | None = None
    test_command: str | None = None
    solve_workers: int = 1
    evaluation_workers: int = 1
    max_wall_seconds: float = 1800.0
    max_steps: int | None = 80
    keep_workspaces: bool = False
    evaluate: bool = True
    resume: bool = False
    retry_failed: bool = False
    live: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
