"""Budgeted filesystem fallback; no unbounded subprocess capture or traversal."""
from __future__ import annotations

import contextvars
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import sys
from contextlib import closing
from typing import Iterable, Iterator, Callable, Generator, Any, cast

from agent_core.codeintel.budget import BudgetExceeded, QueryBudget
from agent_core.codeintel.snapshots import IGNORED_DIRS
from agent_core.permission_safety import is_secret_path


def walk_files(root: Path, base: Path, budget: QueryBudget, *, ignored: frozenset[str] = IGNORED_DIRS,
               project: bool = True, observe_directory: Callable[[Path], None] | None = None) -> Iterator[Path]:
    from agent_core.tools.base import current_execution_context
    context = current_execution_context()
    transaction = context.workspace_view if context else None
    if project and transaction is not None and hasattr(transaction, "query_files"):
        yield from transaction.query_files(base.relative_to(root), budget, ignored)
        return
    if base.is_file():
        budget.consume(candidates=1)
        yield base
        return
    pending = [base]
    while pending:
        budget.check()
        directory = pending.pop()
        try:
            if observe_directory is not None:
                observe_directory(directory)
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError:
            continue  # directory vanished or became unreadable mid-walk
        for entry in entries:
            budget.consume(candidates=1)
            path = Path(entry.path)
            if entry.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or is_secret_path(path):
                continue
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in ignored:
                    pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                yield path


def streamed_lines(command: list[str], root: Path, budget: QueryBudget, data_in: bytes | None = None,
                 feed: Callable[[Any], None] | None = None) -> Generator[str, None, None]:
    """Bounded pipe queue; cancellation/timeout kills and reaps the actual child.

    ``feed`` (mutually exclusive with ``data_in``) writes to the child's stdin from a
    worker thread; a budget/cancel exception raised inside it is re-raised here after
    the pipe drains so partial output stays observable before the failure surfaces.
    """
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               stdin=subprocess.PIPE if (data_in is not None or feed is not None) else subprocess.DEVNULL)
    chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
    stop = threading.Event()
    feed_errors: list[BaseException] = []

    def reader() -> None:
        assert process.stdout is not None
        try:
            while not stop.is_set():
                data = cast(Any, process.stdout).read1(65536)
                while not stop.is_set():
                    try:
                        chunks.put(data if data else None, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if not data:
                    break
        except (OSError, ValueError):
            pass  # Main thread owns process status and error reporting.

    thread = threading.Thread(target=reader, name="code-search-pipe", daemon=True)
    thread.start()
    writer = None
    if feed is not None:
        def run_feed() -> None:
            assert process.stdin is not None
            try:
                feed(process.stdin)
            except (OSError, ValueError):
                pass  # Early result limit or timeout closes the pipe.
            except BaseException as exc:
                feed_errors.append(exc)
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        writer = threading.Thread(target=run_feed, name="code-search-input", daemon=True)
        writer.start()
    elif data_in is not None:
        def write_input() -> None:
            assert process.stdin is not None
            try:
                process.stdin.write(data_in)
                process.stdin.close()
            except (OSError, ValueError):
                pass  # Early result limit or timeout closes the pipe.
        writer = threading.Thread(target=write_input, name="code-search-input", daemon=True)
        writer.start()
    pending = b""
    try:
        while True:
            budget.check()
            try:
                data = chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            if data is None:
                if pending:
                    yield pending.decode("utf-8", errors="replace")
                break
            pending += data
            if len(pending) > budget.config.max_file_bytes:
                raise BudgetExceeded("output_line_bytes")
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                budget.check()
                yield line.decode("utf-8", errors="replace")
        code = process.wait(timeout=max(0.01, budget.deadline - time.monotonic()))
        if code not in (0, 1):
            raise OSError(f"search subprocess exited {code}")
        if feed_errors:
            raise feed_errors[0]
    finally:
        stop.set()
        if process.poll() is None:
            process.kill()
        process.wait()
        thread.join(timeout=1)
        if writer is not None:
            writer.join(timeout=1)
        if process.stdout is not None:
            process.stdout.close()


_REGEX_CHILD = (
    "import sys,re\n"
    "p=re.compile(sys.argv[1],re.I if sys.argv[2]=='1' else 0)\n"
    "r=sys.stdin.buffer\n"
    "o=sys.stdout\n"
    "while True:\n"
    "    h=r.readline()\n"
    "    if not h: break\n"
    "    name,size=h.rstrip(b'\\n').split(b'\\x00')\n"
    "    data=r.read(int(size))\n"
    "    for i,s in enumerate(data.decode('utf-8','replace').splitlines(),1):\n"
    "        if '\\x00' in s: continue\n"
    "        if p.search(s): o.write(name.decode('utf-8','replace')+'\\x00'+str(i)+'\\x00'+s+'\\n')\n"
)


def regex_files(pattern: str, producer: Callable[[], Iterable[tuple[str, bytes]]], root: Path,
                budget: QueryBudget, ignore_case: bool = False) -> Generator[tuple[str, int, str], None, None]:
    """Match a regex against many files through ONE killable child process.

    Input frames are ``name\\x00size\\n<bytes>``; matches come back as
    ``name\\x00lineno\\x00text``. Process isolation keeps pathological patterns
    killable; batching keeps the spawn count at one per call instead of one per file.
    The producer is pull-driven from the feed thread under the caller's context, so
    matching overlaps with reading and partial matches survive a dead budget.
    """
    command = [sys.executable, "-X", "utf8", "-c", _REGEX_CHILD, pattern, "1" if ignore_case else "0"]
    caller_context = contextvars.copy_context()

    def feed(stdin: Any) -> None:
        iterator = iter(producer())
        while True:
            try:
                name, data = caller_context.run(next, iterator)
            except StopIteration:
                return
            budget.check()
            stdin.write(name.encode("utf-8") + b"\x00" + str(len(data)).encode() + b"\n")
            stdin.write(data)

    with closing(streamed_lines(command, root, budget, feed=feed)) as output:
        for line in output:
            name, sep, rest = line.partition("\x00")
            if not sep:
                continue
            number, sep, content = rest.partition("\x00")
            if sep and number.isdigit():
                yield name, int(number), content
