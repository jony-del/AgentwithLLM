import asyncio
import os
from pathlib import Path
import subprocess

import pytest

from agent_core.revisions import RevisionTracker, git_capture
from agent_core.task_runtime import capture_revision


async def test_hint_reuses_hashes_but_strict_capture_never_reuses(tmp_path):
    source = tmp_path / "f.py"
    source.write_text("x=1\n")
    tracker = RevisionTracker(tmp_path, git_aware=False)
    baseline = await tracker.capture()
    assert baseline == capture_revision(tmp_path)
    assert (await tracker.capture(strict=False)) == baseline
    assert tracker.last_metrics["bytes_hashed"] == 0 and tracker.last_metrics["files_reused"] == 1
    source.write_text("x=2\n")
    changed = await tracker.capture(strict=False)
    assert changed.digest != baseline.digest and tracker.last_metrics["bytes_hashed"] == source.stat().st_size
    strict = await tracker.capture()
    assert strict == changed and tracker.last_metrics["files_reused"] == 0


async def test_strict_capture_rehashes_even_when_metadata_hints_are_forged(tmp_path):
    source = tmp_path / "f.py"
    source.write_text("x=1\n")
    tracker = RevisionTracker(tmp_path, git_aware=False)
    baseline = await tracker.capture()
    source.write_text("x=2\n")
    stat = source.stat()
    tracker.cache["f.py"] = ((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino), baseline.files["f.py"])
    assert (await tracker.capture(strict=False)).digest == baseline.digest
    assert (await tracker.capture()).digest != baseline.digest


async def test_git_inventory_tracked_ignored_and_untracked_coverage(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "tracked.py").write_text("x=1\n")
    subprocess.run(["git", "add", "-f", "ignored/tracked.py"], cwd=tmp_path, check=True, timeout=10)
    (tmp_path / "ignored" / "artifact").write_text("ignored")
    (tmp_path / "new.py").write_text("new=1\n")
    (tmp_path / ".env").write_text("SECRET=private")
    tracker = RevisionTracker(tmp_path)
    revision = await tracker.capture()
    assert tracker.last_metrics["inventory"] == "git"
    assert set(revision.files) == {".gitignore", "ignored/tracked.py", "new.py"}
    (tmp_path / "ignored" / "tracked.py").unlink()
    assert "ignored/tracked.py" not in (await tracker.capture()).files


async def test_custom_runtime_paths_are_excluded_without_excluding_source(tmp_path):
    (tmp_path / "f.py").write_text("x=1\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.jsonl").write_text("event")
    tracker = RevisionTracker(tmp_path, excluded=(tmp_path / "logs",), git_aware=False)
    assert set((await tracker.capture()).files) == {"f.py"}


async def test_moving_inventory_fails_closed(tmp_path, monkeypatch):
    tracker = RevisionTracker(tmp_path, git_aware=False)
    (tmp_path / "f.py").write_text("x=1\n")
    calls = 0
    original = tracker._inventory
    async def inventory():
        nonlocal calls
        calls += 1
        if calls == 2:
            (tmp_path / "new.py").write_text("new=1\n")
        return await original()
    monkeypatch.setattr(tracker, "_inventory", inventory)
    with pytest.raises(ValueError, match="membership"):
        await tracker.capture()


async def test_git_pipe_limit_kills_and_awaits_process(tmp_path, monkeypatch):
    killed = waited = False
    class Process:
        returncode = None
        stdout = asyncio.StreamReader()
        stderr = asyncio.StreamReader()
        def kill(self):
            nonlocal killed
            killed = True
            self.returncode = -9
        async def wait(self):
            nonlocal waited
            waited = True
            return self.returncode
    process = Process()
    process.stdout.feed_data(b"x" * 200)
    process.stdout.feed_eof()
    async def spawn(*args, **kwargs):
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    async def terminate(process):
        process.kill()
        await process.wait()
    monkeypatch.setattr("agent_core.revisions.terminate_process_tree", terminate)
    with pytest.raises(ValueError, match="budget"):
        await git_capture(tmp_path, "ls-files", limit=100)
    assert killed and waited


async def test_linked_files_cannot_be_certified(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_text("private")
    try:
        os.symlink(outside, tmp_path / "link")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="escapes|redirected|non-regular"):
        await RevisionTracker(Path(tmp_path), git_aware=False).capture()
