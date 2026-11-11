"""ADR 0003 stage 2's timeout latch: run a command for at most BOUND_S seconds.

    python -m oracle.freeze.watchdog BOUND_S LATCH -- COMMAND...

The command runs in its own process group. When it is still running BOUND_S seconds after it
started (monotonic clock), the watchdog first tries to create LATCH (exclusively, fsynced, with
the UTC times for provenance), then, whatever that gave, sends TERM to the group and KILL 60 s
later, and exits 124. Stage 2's collect counts an attempt with a latch as decisive, whatever record
it left. However the command ends, every process left in its group is killed. A TERM sent to the
watchdog is forwarded to the group (rule E6), then the group is killed the same way. A command
ended by a signal gives 128 + its number, as a shell would."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

GRACE_S = 60


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _end_group(child: subprocess.Popen[bytes], grace_s: float) -> None:
    """TERM to the child's group, then KILL whatever is left of it after `grace_s`. The leader is
    reaped as it goes: an unreaped leader would keep the group alive for the whole grace."""
    pgid = child.pid
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        child.poll()
        if not _group_alive(pgid):
            return
        time.sleep(0.2)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _status(code: int) -> int:
    return 128 - code if code < 0 else code


def _latch(path: str, bound_s: int, started_utc: float) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"bound_s": bound_s, "started_utc_s": started_utc,
                       "latched_utc_s": time.time()}, f)  # fmt: skip
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:  # the group is ended all the same; the missing latch is reported
        print(f"watchdog: the latch {path} could not be written: {e}", file=sys.stderr)


def run(bound_s: int, latch: str, command: list[str], grace_s: float = GRACE_S) -> int:
    child: subprocess.Popen[bytes] | None = None
    terminated = False

    def forward(_signum: int, _frame: Any) -> None:
        nonlocal terminated
        terminated = True
        if child is not None:
            _end_group(child, grace_s)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        started_utc, started = time.time(), time.monotonic()
        child = subprocess.Popen(command, start_new_session=True)
        if terminated:  # a TERM that came while the command was being started
            _end_group(child, grace_s)
        try:
            code = child.wait(timeout=max(0.0, started + bound_s - time.monotonic()))
        except subprocess.TimeoutExpired:
            _latch(latch, bound_s, started_utc)
            _end_group(child, grace_s)
            child.wait()
            return 124
        _end_group(child, 0)  # whatever the command left in its group
        return _status(code)
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__)
        return 2
    return run(int(argv[0]), argv[1], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
