from __future__ import annotations

import os
from pathlib import Path

import psutil


class RunLock:
    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "run.lock"
        self.fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = self._read_pid()
            if pid is not None and psutil.pid_exists(pid):
                raise RuntimeError(f"run is already locked by process {pid}: {self.path}")
            self.path.unlink(missing_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, str(os.getpid()).encode("ascii"))
        return self

    def _read_pid(self) -> int | None:
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def __exit__(self, *_args) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)
