"""ADR 0003 stage 2's watchdog: run a command for at most BOUND_S seconds, and say how it ended.

    python -m oracle.freeze.watchdog BOUND_S MARKER -- COMMAND...

Before it starts the command, the watchdog creates MARKER (exclusively, fsynced) with
`{"status": "running"}`; if it cannot, the command does not run (exit 2). When the command ends,
MARKER is rewritten (atomically) with `"exited"`, its exit status and the elapsed time on a
monotonic clock. When it is still running BOUND_S seconds after it started, MARKER says
`"timeout"`, and the watchdog sends TERM to the command's process group and KILL 60 s later, and
exits 124. Stage 2's collect accepts an attempt only if its MARKER says `"exited"` with status 0
within the bound: a missing marker, or one that still says `"running"` or `"timeout"`, makes the
attempt decisive. Whatever the command does, every process left in its group is killed. A TERM
sent to the watchdog is forwarded to the group (rule E6). A command ended by a signal gives
128 + its number, as a shell would."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
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


def _create(marker: Path, doc: dict[str, Any]) -> None:
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
        f.flush()
        os.fsync(f.fileno())


def _replace(marker: Path, doc: dict[str, Any]) -> None:
    try:
        fd, tmp = tempfile.mkstemp(dir=marker.parent, prefix=f".{marker.name}.", suffix=".part")
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(marker)
    except OSError as e:  # the marker then still says "running": collect counts that as decisive
        print(f"watchdog: {marker} could not be updated: {e}", file=sys.stderr)


def run(bound_s: int, marker: str, command: list[str], grace_s: float = GRACE_S) -> int:
    path = Path(marker)
    started_utc, started = time.time(), time.monotonic()
    try:
        _create(path, {"status": "running", "bound_s": bound_s, "started_utc_s": started_utc})
    except OSError as e:
        print(
            f"watchdog: {marker} could not be created; the command is not run: {e}", file=sys.stderr
        )
        return 2
    child: subprocess.Popen[bytes] | None = None
    terminated = False

    def forward(_signum: int, _frame: Any) -> None:
        nonlocal terminated
        terminated = True
        if child is not None:
            _end_group(child, grace_s)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        child = subprocess.Popen(command, start_new_session=True)
        if terminated:  # a TERM that came while the command was being started
            _end_group(child, grace_s)
        try:
            code = child.wait(timeout=max(0.0, started + bound_s - time.monotonic()))
        except subprocess.TimeoutExpired:
            _replace(path, {"status": "timeout", "bound_s": bound_s, "started_utc_s": started_utc,
                            "elapsed_s": time.monotonic() - started})  # fmt: skip
            _end_group(child, grace_s)
            child.wait()
            return 124
        elapsed = time.monotonic() - started
        _end_group(child, 0)  # whatever the command left in its group
        status = _status(code)
        over = elapsed > bound_s  # an exit noticed after the bound is a timeout all the same
        _replace(path, {"status": "timeout" if over else "exited", "exit": status,
                        "bound_s": bound_s, "started_utc_s": started_utc,
                        "elapsed_s": elapsed})  # fmt: skip
        return 124 if over else status
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__)
        return 2
    return run(int(argv[0]), argv[1], argv[3:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
