"""Durable task contracts and version-bound verification, independent of the LLM."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Literal
import uuid

from agent_core.permission_safety import is_secret_path
from agent_core.tools.transaction import JournalStorage, _open_state, _secure_mkdir

TaskStatus = Literal["running", "completed", "unverified", "blocked", "failed", "cancelled"]
_IGNORED = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
                      ".mypy_cache", ".ruff_cache"})
_ROOT_STATE = frozenset({".polaris", ".polaris-worktrees", "runs", "memory", "tmp", "swebench_runs"})


@dataclass(frozen=True)
class WorkspaceRevision:
    workspace: str
    digest: str
    files: dict[str, str]


def capture_revision(workspace: Path) -> WorkspaceRevision:
    """Bounded content snapshot; moving files fail closed instead of certifying stale code."""
    root = workspace.resolve()
    from agent_core.execution import current_execution_scope
    scope = current_execution_scope()
    files: dict[str, str] = {}
    total = 0
    def walk_error(error: OSError) -> None:
        raise error
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        if scope is not None:
            scope.raise_if_cancelled()
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED and
                         not (Path(directory) == root and name in _ROOT_STATE) and
                         not (Path(directory) / name).is_symlink() and not is_secret_path(name))
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("verification cannot certify symlinked files")
            if is_secret_path(name):
                continue
            before = path.stat()
            if len(files) >= 20_000 or total + before.st_size > 512 * 1024 * 1024:
                raise ValueError("workspace verification snapshot exceeds its budget")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(65536), b""):
                    if scope is not None:
                        scope.raise_if_cancelled()
                    digest.update(block)
                    total += len(block)
                    if total > 512 * 1024 * 1024:
                        raise ValueError("workspace verification snapshot exceeds its budget")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("workspace changed during verification snapshot")
            files[path.relative_to(root).as_posix()] = digest.hexdigest()
    workspace_digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return WorkspaceRevision(str(root), workspace_digest, files)


@dataclass(frozen=True)
class VerificationCheck:
    id: str
    argv: tuple[str, ...]
    kind: str = "test"

    def __post_init__(self) -> None:
        if not self.id or len(self.id) > 128 or not self.argv or len(self.argv) > 256 or sum(map(len, self.argv)) > 16_384:
            raise ValueError("verification requires a bounded id and nonempty argv")


@dataclass(frozen=True)
class TaskContract:
    goal: str
    constraints: tuple[str, ...] = ()
    acceptance: tuple[str, ...] = ()
    checks: tuple[VerificationCheck, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    review_required: bool = False
    verification_required: bool = False

    def __post_init__(self) -> None:
        if not self.goal.strip() or len(self.checks) > 32 or len({c.id for c in self.checks}) != len(self.checks):
            raise ValueError("task contract requires a goal and unique bounded checks")
        for raw in self.allowed_paths:
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts or path.drive:
                raise ValueError("allowed_paths must be workspace-relative")


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    depends_on: tuple[str, ...] = ()
    status: str = "pending"
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationEvidence:
    check_id: str
    revision: str
    workspace: str
    argv_digest: str
    exit_code: int | None
    state: str
    output_path: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    output_preview: str = ""


@dataclass
class TaskRun:
    contract: TaskContract
    baseline: WorkspaceRevision
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: TaskStatus = "running"
    plan: list[PlanStep] = field(default_factory=list)
    plan_revision: int = 0
    evidence: list[VerificationEvidence] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    mutation_seen: bool = False
    checks_locked: bool = False
    execution_workspace: str = ""
    current_revision: str = ""
    baseline_checkpoint: str = ""
    reviews: list[dict[str, Any]] = field(default_factory=list)
    workspace_binding: dict[str, Any] | None = None
    updated_at: float = field(default_factory=time.time)
    verifier_runs: list[dict[str, Any]] = field(default_factory=list)
    answer_reviews: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def replace_plan(self, steps: list[PlanStep]) -> None:
        if len(steps) > 128 or len({step.id for step in steps}) != len(steps):
            raise ValueError("plan ids must be unique and bounded")
        by_id = {step.id: step for step in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("task dependency cycle")
            if key in visited:
                return
            visiting.add(key)
            step = by_id[key]
            if not step.id or not step.description or step.status not in {"pending", "in_progress", "completed", "blocked"}:
                raise ValueError("invalid plan step")
            if len(step.paths) > 64 or any(Path(p).is_absolute() or Path(p).drive or ".." in Path(p).parts for p in step.paths):
                raise ValueError("plan paths must be bounded and workspace-relative")
            for dependency in step.depends_on:
                if dependency not in by_id:
                    raise ValueError("unknown task dependency")
                visit(dependency)
                if step.status in {"in_progress", "completed"} and by_id[dependency].status != "completed":
                    raise ValueError("dependencies must complete before starting a step")
            visiting.remove(key)
            visited.add(key)

        for key in by_id:
            visit(key)
        before = [(x.id, x.description, x.depends_on, x.paths) for x in self.plan]
        after = [(x.id, x.description, x.depends_on, x.paths) for x in steps]
        self.plan = steps
        if before != after:
            self.plan_revision += 1

    def record_failure(self, tool: str, arguments: dict, error_type: str, revision: str) -> None:
        self.current_revision = revision
        fingerprint = hashlib.sha256(json.dumps([tool, arguments, error_type], sort_keys=True).encode()).hexdigest()
        self.failures.append({"tool": tool, "error_type": error_type, "fingerprint": fingerprint,
                              "revision": revision, "plan_revision": self.plan_revision})
        self.failures = self.failures[-100:]

    def stalled(self) -> bool:
        if len(self.failures) < 3:
            return False
        recent = self.failures[-3:]
        return (recent[-1]["revision"] == self.current_revision and
                recent[-1]["plan_revision"] == self.plan_revision and
                len({(x["fingerprint"], x["revision"], x["plan_revision"]) for x in recent}) == 1)

    def context(self) -> str:
        # Only a bounded projection is pinned. The full contract stays in TaskStore
        # and is available through task_state paging rather than bloating every turn.
        selected = [x for x in self.plan if x.status != "completed"][:12]
        review = self.reviews[-1] if self.reviews else None
        review_summary = ({"status": review.get("status"), "revision": review.get("revision"),
                           "reason": str(review.get("reason", ""))[:256],
                           "findings": [{"path": str(f.get("path", ""))[:160],
                                         "severity": f.get("severity"), "message": str(f.get("message", ""))[:256]}
                                        for f in review.get("findings", [])[:3]],
                           "detail": "task_state(section=reviews)"} if review is not None else None)
        return json.dumps({"task_id": self.id, "goal": self.contract.goal[:4000],
                           "constraints": [x[:256] for x in self.contract.constraints[:8]],
                           "acceptance": [x[:256] for x in self.contract.acceptance[:8]],
                           "check_ids": [c.id[:128] for c in self.contract.checks],
                           "review_required": self.contract.review_required, "recent_review": review_summary,
                           "verification_required": self.contract.verification_required,
                           "recent_verifier": self._verifier_summary(self.verifier_runs),
                           "recent_answer_review": self._verifier_summary(self.answer_reviews),
                           "plan_revision": self.plan_revision,
                           "steps": [{"id": x.id[:128], "description": x.description[:256], "status": x.status,
                                      "depends_on": [d[:128] for d in x.depends_on[:8]],
                                      "paths": [p[:160] for p in x.paths[:4]],
                                      "path_count": len(x.paths)} for x in selected],
                           "step_count": len(self.plan), "completed_count": sum(x.status == "completed" for x in self.plan),
                           "recent_failures": self.failures[-3:], "detail_tool": "task_state"}, ensure_ascii=False)

    @staticmethod
    def _verifier_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not records:
            return None
        record = records[-1]
        return {"status": record.get("status"), "verdict": record.get("verdict"),
                "reason": str(record.get("reason", ""))[:512],
                "findings": record.get("findings", [])[:2], "detail": "task_state"}


@dataclass(frozen=True)
class VerificationReport:
    status: TaskStatus
    revision: str
    changed_paths: tuple[str, ...]
    issues: tuple[str, ...]
    evidence: tuple[VerificationEvidence, ...]


def verify_completion(task: TaskRun, current: WorkspaceRevision, *, running_processes: bool = False,
                      truncated: bool = False, termination_proven: bool = True) -> VerificationReport:
    changed = tuple(sorted(path for path in task.baseline.files.keys() | current.files.keys()
                           if task.baseline.files.get(path) != current.files.get(path)))
    issues: list[str] = []
    if truncated:
        issues.append("model response was truncated")
    if not termination_proven:
        issues.append("provider response termination was not proven")
    if running_processes:
        issues.append("background processes still running")
    if any(step.status != "completed" for step in task.plan):
        issues.append("plan has incomplete steps")
    if task.stalled():
        issues.append("three identical failed attempts without a code or plan change")
    if task.contract.allowed_paths:
        outside = [path for path in changed if not any(path == allowed or path.startswith(allowed.rstrip('/') + '/')
                                                      for allowed in task.contract.allowed_paths)]
        if outside:
            issues.append("changes outside task scope: " + ", ".join(outside[:20]))
    valid = tuple(e for e in task.evidence if e.revision == current.digest and e.workspace == current.workspace)
    for check in task.contract.checks:
        digest = hashlib.sha256(json.dumps(check.argv).encode()).hexdigest()
        matching = [e for e in valid if e.check_id == check.id and e.argv_digest == digest]
        if not matching or matching[-1].state != "completed" or matching[-1].exit_code != 0:
            issues.append(f"verification check has not passed on the final revision: {check.id}")
    if (changed or task.mutation_seen or task.contract.acceptance) and not task.contract.checks:
        issues.append("workspace changes require explicit verification checks")
    if task.contract.review_required:
        contract_hash = hashlib.sha256(json.dumps(asdict(task.contract), sort_keys=True).encode()).hexdigest()
        reviews = [r for r in task.reviews if r.get("revision") == current.digest and
                   r.get("workspace") == current.workspace and r.get("contract_hash") == contract_hash]
        if not reviews or reviews[-1].get("status") != "passed":
            issues.append("independent review has not passed on the final revision")
    if task.contract.verification_required:
        contract_hash = hashlib.sha256(json.dumps(asdict(task.contract), sort_keys=True).encode()).hexdigest()
        runs = [r for r in task.verifier_runs if r.get("revision") == current.digest and
                r.get("workspace") == current.workspace and r.get("contract_hash") == contract_hash]
        if not runs or runs[-1].get("status") != "passed":
            issue = "behavioral verifier has not passed on the final revision"
            failure_class = runs[-1].get("failure_class") if runs else None
            if isinstance(failure_class, str) and failure_class:
                issue = "[" + failure_class + "] " + issue
            issues.append(issue)
    status: TaskStatus = "blocked" if task.stalled() else ("unverified" if issues else "completed")
    return VerificationReport(status, current.digest, changed, tuple(issues), valid)


class TaskStore:
    """One private, versioned task record per session; permissions are never persisted here."""
    def __init__(self, storage: JournalStorage, *, isolated: bool = False) -> None:
        self.storage = storage
        self.path = (storage.run_root if isolated else storage.recovery_root) / "task-state.json"

    def save(self, task: TaskRun) -> None:
        self.storage.validate()
        _secure_mkdir(self.path.parent)
        temporary = self.path.with_name(f".task-{uuid.uuid4().hex}.tmp")
        task.updated_at = time.time()
        try:
            with _open_state(self.storage, temporary, "w") as handle:
                json.dump({"v": 3, "task": asdict(task)}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self.storage.validate(self.path)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self) -> TaskRun | None:
        self.storage.validate(self.path)
        if not self.path.exists():
            return None
        if self.path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("task-state record exceeds its budget")
        with _open_state(self.storage, self.path, "r") as handle:
            value = json.load(handle)
        if value.get("v") not in {1, 2, 3}:
            raise ValueError("unsupported task-state schema")
        data = value["task"]
        contract = data.pop("contract")
        for key in ("constraints", "acceptance", "allowed_paths"):
            contract[key] = tuple(contract[key])
        contract["checks"] = tuple(VerificationCheck(c["id"], tuple(c["argv"]), c["kind"]) for c in contract["checks"])
        data["contract"] = TaskContract(**contract)
        data["baseline"] = WorkspaceRevision(**data["baseline"])
        data["plan"] = [PlanStep(**{**step, "depends_on": tuple(step["depends_on"]), "paths": tuple(step.get("paths", ()))}) for step in data["plan"]]
        data["evidence"] = [VerificationEvidence(**e) for e in data["evidence"]]
        return TaskRun(**data)
