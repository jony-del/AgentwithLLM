"""Repository preparation and isolated command runtimes for SWE-bench."""

from __future__ import annotations

import asyncio
import os
import stat
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import SWEbenchInstance


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def render(self, *, max_chars: int = 50_000) -> str:
        output = ((self.stdout or "") + ("\n" + self.stderr if self.stderr else "")).strip()
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n[output truncated at {max_chars} chars]"
        status = "timeout" if self.timed_out else f"exit {self.returncode}"
        return f"[{status}]\n{output or '(no output)'}"


class InstanceRuntime(Protocol):
    async def start(self) -> None: ...

    async def exec(self, command: str, *, timeout: float = 300.0, cwd: str = "/testbed") -> CommandResult: ...

    async def close(self) -> None: ...


class LocalRuntime:
    """A test/dev runtime; production runs should use :class:`DockerRuntime`."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def exec(self, command: str, *, timeout: float = 300.0, cwd: str = "/testbed") -> CommandResult:
        del cwd
        if not self.started:
            await self.start()
        started = time.monotonic()
        command_timeout = max(0.1, float(timeout))
        try:
            completed = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    command,
                    cwd=str(self.workspace),
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                    timeout=command_timeout,
                ),
                # Give subprocess.run a little time to clean up after its own
                # timeout before the outer coroutine guard fires.
                timeout=command_timeout + 5.0,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(command, -9, "", "command timed out", time.monotonic() - started, True)
        except asyncio.TimeoutError:
            return CommandResult(command, -9, "", "command timed out", time.monotonic() - started, True)
        return CommandResult(
            command,
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            time.monotonic() - started,
        )

    async def close(self) -> None:
        self.started = False


class DockerRuntime:
    """Persistent no-network Docker container with the workspace mounted at /testbed."""

    def __init__(
        self,
        workspace: str | Path,
        image: str,
        *,
        name: str | None = None,
        network_mode: str = "none",
        auto_pull: bool = True,
        activate_testbed: bool = True,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.image = image
        self.name = name or f"polaris-swebench-{uuid.uuid4().hex[:12]}"
        self.network_mode = network_mode
        self.auto_pull = auto_pull
        self.activate_testbed = activate_testbed
        self.client: Any = None
        self.container: Any = None

    async def start(self) -> None:
        if self.container is not None:
            return
        try:
            import docker
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Docker runtime is unavailable. Install `docker` and start Docker Desktop/Engine, "
                "or use --runtime local only for controlled development tests."
            ) from exc
        self.client = docker.from_env()
        kwargs = {
            "command": ["/bin/sh", "-lc", "while true; do sleep 3600; done"],
            "name": self.name,
            "detach": True,
            "working_dir": "/testbed",
            "volumes": {str(self.workspace): {"bind": "/testbed", "mode": "rw"}},
            "network_mode": self.network_mode,
            "environment": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TZ": "UTC"},
            "tty": False,
            "auto_remove": False,
        }
        try:
            self.container = await asyncio.to_thread(self.client.containers.run, self.image, **kwargs)
        except Exception as exc:
            if not self.auto_pull or not _looks_like_missing_image(exc):
                if self.client is not None:
                    self.client.close()
                self.client = None
                raise
            try:
                await asyncio.to_thread(self.client.images.pull, self.image)
                self.container = await asyncio.to_thread(self.client.containers.run, self.image, **kwargs)
            except Exception:
                if self.client is not None:
                    self.client.close()
                self.client = None
                raise

    async def exec(self, command: str, *, timeout: float = 300.0, cwd: str = "/testbed") -> CommandResult:
        await self.start()
        assert self.container is not None
        started = time.monotonic()

        def invoke() -> Any:
            command_to_run = "git config --global --add safe.directory /testbed >/dev/null 2>&1 || true; "
            if self.activate_testbed:
                # Official SWE-bench images install project dependencies in the
                # ``testbed`` conda environment, while their default process
                # often starts in the base environment.  Keep custom images
                # usable by falling back to their existing PATH when the env is
                # absent.
                command_to_run += (
                    "if [ -f /opt/miniconda3/bin/activate ]; then "
                    ". /opt/miniconda3/bin/activate testbed >/dev/null 2>&1 || true; "
                    f"fi; {command}"
                )
            else:
                command_to_run += command
            return self.container.exec_run(
                ["/bin/sh", "-lc", command_to_run],
                workdir=cwd,
                demux=True,
                stdout=True,
                stderr=True,
            )

        try:
            result = await asyncio.wait_for(asyncio.to_thread(invoke), timeout=max(0.1, float(timeout)))
        except asyncio.TimeoutError:
            # docker-py does not expose a portable exec-process kill API.  Killing
            # and restarting the disposable solver container is safer than allowing
            # a command to survive the Agent's timeout.
            await self._restart_after_timeout()
            return CommandResult(command, -9, "", "command timed out", time.monotonic() - started, True)
        output = getattr(result, "output", (b"", b""))
        if isinstance(output, tuple):
            stdout, stderr = output if len(output) == 2 else (output[0] if output else b"", b"")
        else:
            # Some docker-py versions/fakes return a single bytes object even
            # when ``demux=True`` was requested.
            stdout, stderr = output, b""
        return CommandResult(
            command,
            int(getattr(result, "exit_code", 1)),
            _decode_bytes(stdout),
            _decode_bytes(stderr),
            time.monotonic() - started,
        )

    async def _restart_after_timeout(self) -> None:
        if self.container is None:
            return
        try:
            await asyncio.to_thread(self.container.kill)
        except Exception:
            pass
        try:
            await asyncio.to_thread(self.container.remove, force=True)
        except Exception:
            pass
        self.container = None
        await self.start()

    async def close(self) -> None:
        container, client = self.container, self.client
        self.container = None
        self.client = None
        if container is not None:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception:
                pass
        if client is not None:
            try:
                await asyncio.to_thread(client.close)
            except Exception:
                pass


def _decode_bytes(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _looks_like_missing_image(exc: BaseException) -> bool:
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    return "imagenotfound" in name or "no such image" in text or "pull access denied" in text


def resolve_repo_url(repo: str) -> str:
    value = str(repo).strip()
    if value.startswith(("https://", "http://", "git@", "file://")):
        return value
    local = Path(value).expanduser()
    if local.exists():
        return str(local.resolve())
    if value.startswith("github.com/"):
        value = value[len("github.com/"):]
    value = value.rstrip("/")
    return f"https://github.com/{value if value.endswith('.git') else value + '.git'}"


def prepare_repository(
    instance: SWEbenchInstance,
    workspace: str | Path,
    *,
    source_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Materialize a repository with HEAD exactly at ``base_commit``."""
    target = Path(workspace).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    # The marker lives beside the workspace, never inside it: anything under the
    # mounted repository is model-visible and would otherwise become an untracked
    # prediction file.
    baseline_file = target.parent / f".{target.name}.baseline"
    if not force and baseline_file.exists() and (target / ".git").is_dir():
        try:
            if baseline_file.read_text(encoding="utf-8").strip() == instance.base_commit:
                return target
        except OSError:
            pass
    source = Path(source_dir).expanduser().resolve() if source_dir else None
    if source is not None and (source == target or target in source.parents):
        raise ValueError("source_dir must not be the workspace or a parent of it")
    if target.exists():
        _remove_tree(target)
    clone_source = str(source) if source is not None else resolve_repo_url(instance.repo)
    clone_path = target.parent / f".{target.name}.clone-{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-tags", clone_source, str(clone_path)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(clone_path), "checkout", "--quiet", "--detach", instance.base_commit],
            check=True,
            capture_output=True,
            timeout=300,
        )
        actual = _git_text(clone_path, ["rev-parse", "HEAD"]).strip()
        if actual != instance.base_commit:
            raise RuntimeError(f"checked out {actual}, expected {instance.base_commit}")
        # Remove the clone's remote refs before making it visible to the Agent.
        # This leaves HEAD exactly at the official base commit while preventing
        # ordinary ``git log --all`` inspection from exposing later solution
        # commits.  Expiring reflogs removes the other normal navigation path;
        # we intentionally avoid an expensive full ``git gc`` for every task.
        subprocess.run(
            ["git", "-C", str(clone_path), "remote", "remove", "origin"],
            check=False,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(clone_path), "reflog", "expire", "--expire=now", "--all"],
            check=False,
            capture_output=True,
            timeout=120,
        )
        shutil.move(str(clone_path), str(target))
        # Keep an immutable ref for patch export.  The Agent is allowed to use
        # git for inspection and may accidentally commit; diffing against HEAD
        # in that case would silently submit an empty patch.
        baseline_commit = _git_text(target, ["rev-parse", "HEAD"]).strip()
        subprocess.run(
            ["git", "-C", str(target), "config", "user.email", "swebench@localhost"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(target), "config", "user.name", "SWE-bench Runner"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "-C", str(target), "update-ref", "refs/polaris/swebench-baseline", baseline_commit],
            check=True,
            capture_output=True,
            timeout=60,
        )
        # Keep the official SHA outside the tracked tree so it can never become a
        # model prediction.
        baseline_file.write_text(instance.base_commit + "\n", encoding="utf-8")
        return target
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        _remove_tree(clone_path)
        _remove_tree(target)
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            detail = f"{detail}: {_decode_bytes(exc.stderr).strip()}"
        raise RuntimeError(f"could not prepare {instance.instance_id}: {detail}") from exc


def _git_text(cwd: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, timeout=120)
    return result.stdout.decode("utf-8", errors="replace")


def _remove_tree(path: Path) -> None:
    """Best-effort removal of disposable clones/workspaces on Windows and POSIX."""
    if not path.exists():
        return

    def onerror(function: Any, target: str, exc_info: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            function(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=onerror)


def resolve_official_image(instance: SWEbenchInstance, override: str | None = None) -> str | None:
    """Resolve a prebuilt official image without requiring the Harness at import time."""
    if override:
        return override
    env_image = os.getenv("SWEBENCH_INSTANCE_IMAGE")
    if env_image:
        return env_image
    if instance.image:
        return instance.image
    # Current SWE-bench exposes TestSpec.instance_image_key.  Keep this reflective
    # because the package changed import paths between releases.
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except (ImportError, ModuleNotFoundError):
        try:
            from swebench.harness.test_spec import make_test_spec
        except (ImportError, ModuleNotFoundError):
            try:
                from swebench.harness.utils import make_test_spec
            except (ImportError, ModuleNotFoundError):
                make_test_spec = None
    if make_test_spec is None:
        return _fallback_official_image(instance)
    try:
        spec = make_test_spec(instance.evaluation_dict())
    except Exception:
        spec = None
    if spec is None:
        # The current Harness naming contract is stable even when constructing a
        # full TestSpec fails because a repository-specific helper is unavailable.
        # This fallback lets an already-built official image be used offline.
        return _fallback_official_image(instance)
    # SWE-bench v5 exposes ``TestSpec.image``; older releases used
    # ``instance_image_key``.  Accept both so the runtime works across the
    # package versions supported by the optional dependency range.
    key = getattr(spec, "instance_image_key", None) or getattr(spec, "image", None)
    if not key:
        return _fallback_official_image(instance)
    key = str(key)
    # Harness defaults to the ``swebench`` namespace for local images.  Preserve a
    # fully-qualified key supplied by a custom namespace.
    return key if "/" in key.split("@", 1)[0] else f"swebench/{key}"


def _fallback_official_image(instance: SWEbenchInstance) -> str:
    namespace = os.getenv("SWEBENCH_IMAGE_NAMESPACE", "swebench").strip("/") or "swebench"
    arch = (os.getenv("SWEBENCH_ARCH", "x86_64") or "x86_64").strip().lower()
    if arch == "amd64":
        arch = "x86_64"
    tag = os.getenv("SWEBENCH_INSTANCE_IMAGE_TAG", "latest") or "latest"
    # Docker Hub replaces the double underscore separating owner/repository with
    # ``_1776_`` because dunders are not accepted in published image names.
    image_id = instance.instance_id.lower().replace("__", "_1776_")
    return f"{namespace}/sweb.eval.{arch}.{image_id}:{tag}"
