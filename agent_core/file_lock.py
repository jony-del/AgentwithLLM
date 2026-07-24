from __future__ import annotations

import os
import time
from pathlib import Path
from typing import BinaryIO


class FileLock:
    """Dependency-free cross-platform advisory lock for sidecar lock files."""

    def __init__(self, path: str | Path, *, timeout: float | None = None) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._file: BinaryIO | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        self._file = file
        if file.seek(0, os.SEEK_END) == 0:
            file.write(b"\0")
            file.flush()
            os.fsync(file.fileno())
        file.seek(0)
        started = time.monotonic()
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if self.timeout is not None and time.monotonic() - started >= self.timeout:
                        file.close()
                        self._file = None
                        raise TimeoutError(f"timed out waiting for lock: {self.path}")
                    time.sleep(0.005)
        else:
            import importlib

            fcntl = importlib.import_module("fcntl")
            if self.timeout is None:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            else:
                while True:
                    try:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() - started >= self.timeout:
                            file.close()
                            self._file = None
                            raise TimeoutError(f"timed out waiting for lock: {self.path}")
                        time.sleep(0.005)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is None:
            return
        self._file.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import importlib

            fcntl = importlib.import_module("fcntl")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None
