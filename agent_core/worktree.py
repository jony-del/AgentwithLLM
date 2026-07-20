"""Session-owned Git worktree isolation with fail-closed removal checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any


_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")


@dataclass(slots=True)
class WorktreeState:
    slug: str
    path: Path
    branch: str
    base_sha: str
    created_at: float


class WorktreeManager:
    def __init__(self, session: Any, registry: Any, sandbox: Any, config: Any) -> None:
        self.session = session
        self.registry = registry
        self.sandbox = sandbox
        self.config = config
        self.original_workspace = Path(session.workspace).resolve()
        self.active: WorktreeState | None = None
        self.owned: dict[str, WorktreeState] = {}

    async def _git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        process = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(cwd or self.session.workspace),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(process.communicate(), 30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"git {' '.join(args)} timed out") from None
        text = output.decode("utf-8", errors="replace")
        if check and process.returncode:
            detail = error.decode("utf-8", errors="replace").strip() or text.strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {detail[:4000]}")
        return text.strip()

    async def create_and_enter(self, name: str | None = None) -> WorktreeState:
        if self.active is not None:
            raise RuntimeError("this session is already inside a managed worktree")
        if self.session.process_supervisor is not None and self.session.process_supervisor.running():
            raise RuntimeError("stop all background tasks before switching workspace")
        slug = (name or f"session-{self.session.session_id[:8]}").strip().lower()
        if not _SLUG.fullmatch(slug):
            raise ValueError("worktree name must match [a-z0-9][a-z0-9_-]{0,47}")
        if slug in self.owned:
            state = self.owned[slug]
            if not state.path.is_dir():
                raise RuntimeError("session-owned worktree path disappeared; refusing to recreate it")
            await self._switch(state.path)
            self.active = state
            await self._event("workspace_switched", state, action="resume")
            return state
        repo_root = Path(await self._git("rev-parse", "--show-toplevel", cwd=self.original_workspace)).resolve()
        await self.cleanup_stale(repo_root)
        base_sha = await self._git("rev-parse", "HEAD", cwd=repo_root)
        root = (repo_root / self.config.root).resolve()
        if root != repo_root and repo_root not in root.parents:
            raise RuntimeError("configured worktree root escapes the repository")
        path = (root / slug).resolve()
        if path.parent != root or path.exists():
            raise RuntimeError(f"refusing unsafe or existing worktree path: {path}")
        branch = f"polaris/{self.session.session_id[:12]}/{slug}"
        root.mkdir(parents=True, exist_ok=True)
        await self._git("worktree", "add", "-b", branch, str(path), base_sha, cwd=repo_root)
        state = WorktreeState(slug, path, branch, base_sha, time.time())
        self.owned[slug] = state
        try:
            await self._switch(path)
        except Exception:
            await self._git("worktree", "remove", "--force", str(path), cwd=repo_root, check=False)
            self.owned.pop(slug, None)
            raise
        self.active = state
        await self._event("workspace_switched", state, action="enter")
        return state

    async def cleanup_stale(self, repo_root: Path | None = None) -> list[str]:
        """Remove only old, clean worktrees with Polaris's ephemeral branch namespace."""
        repo = repo_root or Path(
            await self._git("rev-parse", "--show-toplevel", cwd=self.original_workspace)
        ).resolve()
        root = (repo / self.config.root).resolve()
        listing = await self._git("worktree", "list", "--porcelain", cwd=repo)
        removed: list[str] = []
        cutoff = time.time() - max(1, int(self.config.stale_days)) * 86400
        for block in listing.split("\n\n"):
            fields = dict(
                line.split(" ", 1) for line in block.splitlines() if " " in line
            )
            raw_path = fields.get("worktree")
            branch = fields.get("branch", "")
            if not raw_path or not branch.startswith("refs/heads/polaris/ephemeral/"):
                continue
            path = Path(raw_path).resolve()
            if path.parent != root:
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                if await self._git("status", "--porcelain=v1", cwd=path):
                    continue
                await self._git("worktree", "remove", str(path), cwd=repo)
            except (OSError, RuntimeError):
                continue
            removed.append(str(path))
            if self.session.logger is not None:
                await self.session.logger.write(
                    "worktree_cleanup", {"path": str(path), "branch": branch, "action": "stale"}
                )
        return removed

    async def _switch(self, workspace: Path) -> None:
        manager = self.session.lsp_manager
        if manager is not None:
            await manager.close()
            self.session.lsp_manager = None
        self.session.workspace = workspace.resolve()
        self.registry.rebind_workspace(str(workspace))
        self.sandbox.workspace = workspace.resolve()
        setter = getattr(self.session, "permission_workspace_setter", None)
        if setter is not None:
            setter(workspace.resolve())

    async def summary(self, state: WorktreeState) -> dict[str, object]:
        status = await self._git("status", "--porcelain=v1", cwd=state.path)
        commits_text = await self._git("rev-list", "--count", f"{state.base_sha}..HEAD", cwd=state.path)
        commits = int(commits_text or "0")
        diff = await self._git("diff", "--stat", state.base_sha, cwd=state.path)
        return {
            "path": str(state.path), "branch": state.branch, "base_sha": state.base_sha,
            "dirty": bool(status), "uncommitted": status.splitlines()[:100],
            "new_commits": commits, "diff_summary": diff[:8000],
        }

    async def exit(self, action: str, *, discard_changes: bool) -> dict[str, object]:
        state = self.active
        if state is None or self.owned.get(state.slug) is not state:
            raise RuntimeError("no session-owned active worktree")
        if self.session.process_supervisor is not None and self.session.process_supervisor.running():
            raise RuntimeError("stop all background tasks before switching workspace")
        details = await self.summary(state)
        if action not in {"keep", "remove"}:
            raise ValueError("action must be keep or remove")
        if action == "remove" and (details["dirty"] or details["new_commits"]):
            if not discard_changes:
                raise RuntimeError("worktree has uncommitted files or new commits; set discard_changes=true after review")
        await self._switch(self.original_workspace)
        self.active = None
        if action == "remove":
            try:
                args = ["worktree", "remove"]
                if discard_changes:
                    args.append("--force")
                args.append(str(state.path))
                await self._git(*args, cwd=self.original_workspace)
            except Exception:
                await self._switch(state.path)
                self.active = state
                raise
            await self._git(
                "branch", "-D", state.branch, cwd=self.original_workspace, check=False
            )
            self.owned.pop(state.slug, None)
            await self._event("worktree_cleanup", state, action="remove", discard=discard_changes)
        else:
            await self._event("workspace_switched", state, action="keep")
        details.update({"action": action, "current_workspace": str(self.session.workspace)})
        return details

    async def _event(self, kind: str, state: WorktreeState, **extra: object) -> None:
        sink = getattr(self.session, "audit_event", None)
        if sink is not None:
            await sink(kind, {
                "path": str(state.path), "branch": state.branch, "base_sha": state.base_sha, **extra
            })
        elif self.session.logger is not None:
            await self.session.logger.write(kind, {
                "path": str(state.path), "branch": state.branch, "base_sha": state.base_sha, **extra
            })
