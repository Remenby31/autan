"""ADR 0003 stage 2's timeout latch: run a command for at most BOUND_S seconds.

    python -m oracle.freeze.watchdog BOUND_S LATCH -- COMMAND...

The command runs in its own process group. When it is still running BOUND_S seconds after it
started, the watchdog first creates LATCH (exclusively, fsynced), then sends TERM to the group and
KILL 60 s later, and exits 124. Stage 2's collect counts an attempt with a latch as decisive,
whatever record it left. A TERM sent to the watchdog is forwarded to the group (rule E6)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

GRACE_S = 60


def run(bound_s: int, latch: str, command: list[str]) -> int:
    start = time.time()
    child = subprocess.Popen(command, start_new_session=True)

    def forward(signum: int, _frame: object) -> None:
        os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    try:
        return child.wait(timeout=max(0.0, start + bound_s - time.time()))
    except subprocess.TimeoutExpired:
        fd = os.open(latch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"bound_s": bound_s, "started_utc_s": start, "latched_utc_s": time.time()}, f)
            f.flush()
            os.fsync(f.fileno())
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        return 124


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__)
        return 2
    return run(int(argv[0]), argv[1], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
