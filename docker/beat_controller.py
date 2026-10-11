"""Run Beat and the lightweight control-queue worker in one container."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


children: list[subprocess.Popen[bytes]] = []
stopping = False


def request_stop(_signal: int, _frame: object) -> None:
    global stopping
    stopping = True


def stop_children() -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 20
    for child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log_level = os.environ.get("CELERY_LOG_LEVEL", "info")
    commands = (
        ["celery", "-A", "control_panel", "worker", "-l", log_level,
         "--queues=control", "--pool=solo", "--concurrency=1"],
        ["celery", "-A", "control_panel", "beat", "-l", log_level],
    )
    try:
        for command in commands:
            children.append(subprocess.Popen(command))
        while not stopping:
            if any(child.poll() is not None for child in children):
                return 1
            time.sleep(1)
        return 0
    finally:
        stop_children()


if __name__ == "__main__":
    sys.exit(main())
