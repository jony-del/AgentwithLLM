from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import hashlib
import json

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FileVersion:
    worktree_id: str
    path: str
    sha256: str
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CodeHit:
    path: str
    start_line: int | None
    end_line: int | None
    text: str
    version: FileVersion
    kind: str = "text"
    precision: str = "literal"
    module: str = "."
    source_symbol: str | None = None
    target_symbol: str | None = None
    # start/end identify the source fact (e.g. the entire function). These fields
    # describe only the source lines actually present in text, never marker lines.
    excerpt_start_line: int | None = None
    excerpt_end_line: int | None = None
    truncated: bool = False
    next_start_line: int | None = None
    symbol: dict[str, Any] | None = None


def symbol_reference(version: FileVersion, qualified: str, kind: str, start: int, end: int,
                     parser: str) -> dict[str, Any]:
    """Version-scoped definition identity, not a rename-stable semantic identity."""
    identity = [version.worktree_id, version.path, version.sha256, qualified, kind, start, end, parser]
    return {"schema_version": 1, "id": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
            "qualified_name": qualified, "container": qualified.rpartition(".")[0], "kind": kind,
            "parser": parser, "resolution": "syntax_definition", "identity_scope": "file_content_version"}


@dataclass(slots=True)
class SearchCoverage:
    requested: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Precision disclosures that do not affect completeness (e.g. syntactic-only
    # relations); kept separate so complete reflects coverage, not precision caveats.
    advisories: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=lambda: ["vcs/cache/dependency directories", "configured workspace-relative roots", "secrets", "symlinks"])
    complete: bool = False
    freshness: str = "observed; external changes may await reconciliation"

    def add_reason(self, reason: str) -> None:
        if reason in self.reasons:
            return
        if len(self.reasons) < 16:
            self.reasons.append(reason[:200])
        elif self.reasons[-1] != "additional_diagnostics_omitted":
            self.reasons.append("additional_diagnostics_omitted")


@dataclass(slots=True)
class SearchPage:
    hits: list[CodeHit] = field(default_factory=list)
    coverage: SearchCoverage = field(default_factory=SearchCoverage)
    generation: int = 0
    cursor: str | None = None
    usage: dict[str, int | float] = field(default_factory=dict)
    # Exhaustion of the selected indexed candidates, not proof of full repository
    # coverage. Pending/unindexable files can keep coverage.complete false.
    scan_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    kind: str = "auto"
    path: str = "."
    modules: tuple[str, ...] = ()
    language: str | None = None
    regex: bool = False
    ignore_case: bool = False
    limit: int = 100
    cursor: str | None = None
    direction: str = "incoming"
    expand_scope: bool = False


@dataclass(frozen=True, slots=True)
class ChangeSet:
    event_id: str
    paths: tuple[str, ...]
    source: str = "editor"


class StaleEvidence(RuntimeError):
    """The code that justified an operation is no longer the current version."""
