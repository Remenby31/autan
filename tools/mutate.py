"""Run mutants against a CMake build: each mutant edits one source line, its test must fail.

    python3 tools/mutate.py [--min-timeout SECONDS] SRC BUILD MUTANTS.json [RESULTS.jsonl]

MUTANTS.json is a list of {"name", "file" (relative to SRC), "from", "to", "target" (CMake targets,
space-separated), "exe" (relative to BUILD), "test" (a Catch2 test name or wildcard pattern, taken
literally: a comma in it is part of the name)}. `from` must occur exactly once in the file; the
mutant is that occurrence replaced by `to`. BUILD must be a CMake build of SRC itself, linked
statically (autan's build refuses BUILD_SHARED_LIBS), so that the test's binaries hold all its
code. `exe` is a Catch2 binary, or a script that runs several with the arguments it gets and
names each as "$(dirname "$0")/<path>" so that the baseline can hash it.

Each verdict is one of:
- caught: the test ran and failed;
- caught by timeout: the test ran past max(--min-timeout, ten times the original's run) and was
  killed, as CI's timeout fails a test that hangs;
- survived: the test ran and passed;
- not built: the mutant does not compile;
- matches no test case.
With RESULTS, each verdict is appended as soon as it is known and the mutants already in it are
skipped, so a run stopped and started again repeats nothing; its exit status counts them too.

Guards, so that a verdict means what it says and no mutant outlives its run:
- one run at a time on BUILD (an exclusive lock), and BUILD's CMAKE_HOME_DIRECTORY is SRC;
- every file the mutants edit is touched and the targets rebuilt before the first run, so that no
  binary left by an interrupted run can be older than its source;
- before each mutant its test runs on the original build, must run a test case (Catch2's summary
  printed) and pass; that run is kept per SHA-256 of the binaries and repeated when they change;
- a test or build that is stopped, times out or is interrupted is killed with its process group;
- the original is restored with signals held, atomically, checked byte for byte, and rebuilt.

Exit status: 0 if every mutant was caught, 1 if one was not, 2 if a guard stopped the run.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

CAUGHT = ("caught", "caught by timeout")
HELD = {signal.SIGTERM, signal.SIGINT}


def stopped(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)  # unwinds through `finally`, which restores the source


def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binaries_sha(exe: pathlib.Path) -> str:
    data = exe.read_bytes()
    if not data.startswith(b"#!"):
        return sha(exe)
    names = re.findall(r'\$\(dirname "\$0"\)/([^"]+)"', data.decode())
    return sha(exe) + "".join(" " + sha(exe.parent / n) for n in names)


def write_atomic(path: pathlib.Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".mutate-tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def run(args: list[str], timeout: float) -> tuple[int | None, str, float]:
    """`args` in a process group of its own: its exit status (None when killed at `timeout`), its
    output and its seconds. The group is killed on a timeout and on any interruption."""
    start = time.monotonic()
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    def kill() -> str:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        try:
            return p.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:  # a descendant that left the group holds the pipe
            return ""

    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, kill(), time.monotonic() - start
    except BaseException:
        kill()
        raise
    return p.returncode, out, time.monotonic() - start


def build(build_dir: pathlib.Path, targets: str) -> bool:
    status, out, _ = run(["cmake", "--build", str(build_dir), "--target", *targets.split()], 3600)
    if status != 0:
        say(out[-1500:])
    return status == 0


def run_test(
    build_dir: pathlib.Path, m: dict, abort: bool, timeout: float
) -> tuple[bool, bool, float, bool]:
    """Whether the test ran a test case, whether it passed, its seconds, and whether it was killed
    at `timeout`."""
    name = m["test"].replace("\\", "\\\\").replace(",", "\\,")
    args = [str(build_dir / m["exe"]), name] + (["--abort"] if abort else [])
    status, out, seconds = run(args, timeout)
    if status is None:
        return True, False, seconds, True
    ran = re.search(r"All tests passed|test cases:", out) is not None
    ran = ran and "No test cases matched" not in out and "No tests ran" not in out
    return ran, status == 0, seconds, False


def restore(path: pathlib.Path, original: bytes) -> None:
    """`original` back in `path`, with SIGTERM and SIGINT held until it is: a signal cannot leave
    the source mutated or half written."""
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, HELD)
    try:
        write_atomic(path, original)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def main() -> int:
    signal.signal(signal.SIGTERM, stopped)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-timeout", type=float, default=60.0)
    parser.add_argument("src", type=pathlib.Path)
    parser.add_argument("build", type=pathlib.Path)
    parser.add_argument("mutants", type=pathlib.Path)
    parser.add_argument("results", type=pathlib.Path, nargs="?")
    a = parser.parse_args()
    src, build_dir, results = a.src, a.build, a.results

    lock = (build_dir / ".mutate.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say(f"another mutate.py runs on {build_dir}")
        return 2
    home = re.search(
        r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$",
        (build_dir / "CMakeCache.txt").read_text(),
        re.MULTILINE,
    )
    if home is None or pathlib.Path(home.group(1)).resolve() != src.resolve():
        say(f"{build_dir} builds {home.group(1) if home else 'no source'}, not {src}")
        return 2
    mutants = json.loads(a.mutants.read_text())
    if any(m["test"].startswith("-") for m in mutants):
        say("a test name cannot start with '-': Catch2 would read it as an option")
        return 2
    verdicts = {}
    if results is not None and results.exists():
        for line in results.read_text().splitlines():
            if line:
                verdicts[json.loads(line)["name"]] = json.loads(line)["verdict"]
    todo = [m for m in mutants if m["name"] not in verdicts]
    for file in {m["file"] for m in todo}:
        (src / file).touch()
    targets = " ".join(sorted({t for m in todo for t in m["target"].split()}))
    if todo and not build(build_dir, targets):
        say("the original does not build")
        return 2

    baseline_file = results.with_suffix(".baseline.json") if results is not None else None
    baseline = {}
    if baseline_file is not None and baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
    for m in todo:
        key = binaries_sha(build_dir / m["exe"]) + " " + m["test"]
        if key not in baseline:
            ran, passed, seconds, _ = run_test(build_dir, m, False, 3600.0)
            baseline[key] = [ran, passed, seconds]
            if baseline_file is not None:
                write_atomic(baseline_file, json.dumps(baseline, indent=1).encode())
        ran, passed, seconds = baseline[key]
        if not ran or not passed:
            reason = "runs no test case" if not ran else "fails"
            say(f"{m['name']}: on the original build its test {reason}")
            return 2
        path = src / m["file"]
        original = path.read_bytes()
        text = original.decode()
        if text.count(m["from"]) != 1:
            say(f"{m['name']}: '{m['from']}' found {text.count(m['from'])} times, not once")
            return 2
        mutated = text.replace(m["from"], m["to"]).encode()
        if mutated == original:
            say(f"{m['name']}: the mutant equals the original")
            return 2
        verdict = "not built"
        try:
            write_atomic(path, mutated)
            if build(build_dir, m["target"]):
                limit = max(a.min_timeout, 10 * seconds)
                ran, passed, _, killed = run_test(build_dir, m, True, limit)
                if killed:
                    verdict = "caught by timeout"
                elif not ran:
                    verdict = "matches no test case"
                else:
                    verdict = "survived" if passed else "caught"
        finally:
            restore(path, original)
        if path.read_bytes() != original:
            say(f"{m['name']}: the source was not restored")
            return 2
        if not build(build_dir, m["target"]):
            say(f"{m['name']}: the restored source does not build")
            return 2
        say(f"{m['name']}: {verdict}")
        verdicts[m["name"]] = verdict
        if results is not None:
            with results.open("a") as out:
                out.write(json.dumps({"name": m["name"], "verdict": verdict}) + "\n")
    names = {m["name"] for m in mutants}
    return 0 if all(v in CAUGHT for n, v in verdicts.items() if n in names) else 1


if __name__ == "__main__":
    sys.exit(main())
