"""Run mutants against a CMake build: each mutant edits one source line, its test must fail.

    python3 tools/mutate.py [--min-timeout SECONDS] SRC BUILD MUTANTS.json [RESULTS.jsonl]

MUTANTS.json is a list of {"name", "file" (relative to SRC), "from", "to", "target" (CMake targets,
space-separated), "exe" (relative to BUILD), "test" (a Catch2 test spec)}. `from` must occur exactly
once in the file; the mutant is that occurrence replaced by `to`. BUILD must be a CMake build of
SRC itself. `exe` is a Catch2 binary, or a script that runs several with the arguments it gets; a
script names each binary as "$(dirname "$0")/<path>" so that the baseline can hash it.

Each verdict is one of:
- caught: the test ran and failed;
- caught by timeout: the test ran past max(--min-timeout, ten times the original's run) and was
  killed, as CI's timeout fails a test that hangs;
- survived: the test ran and passed;
- not built: the mutant does not compile;
- matches no test case.
With RESULTS, each verdict is appended as soon as it is known and the mutants already in it are
skipped, so a run stopped and started again repeats nothing.

Guards, so that a verdict means what it says and no mutant outlives its run:
- BUILD's CMAKE_HOME_DIRECTORY is SRC: a build of another tree would never see the mutant;
- every file the mutants edit is touched and the targets rebuilt before the first run, so that no
  binary left by an interrupted run can be older than its source;
- before each mutant its test runs on the original build, must match a test case and pass; that
  run is kept per SHA-256 of the binaries (RESULTS' .baseline.json) and repeated when they change;
- the original is restored in `finally` whatever happens (SIGTERM included), touched, checked
  byte for byte, and rebuilt: the run stops if that rebuild fails;
- commas in the test spec are escaped: Catch2 reads an unescaped comma as a separator.

Exit status: 0 if every mutant was caught, 1 if one was not, 2 if a guard stopped the run.
"""

import argparse
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


def build(build_dir: pathlib.Path, targets: str) -> bool:
    r = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", *targets.split()],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if r.returncode:
        say(r.stdout[-1500:])
    return r.returncode == 0


def run_test(
    build_dir: pathlib.Path, m: dict, abort: bool, timeout: float
) -> tuple[bool, bool, float, bool]:
    """Whether the test matched a test case, whether it passed, its seconds, and whether it was
    killed at `timeout`."""
    spec = m["test"].replace("\\", "\\\\").replace(",", "\\,")
    args = [str(build_dir / m["exe"]), spec] + (["--abort"] if abort else [])
    start = time.monotonic()
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return True, False, time.monotonic() - start, True
    ran = "No test cases matched" not in out and "No tests ran" not in out
    return ran, p.returncode == 0, time.monotonic() - start, False


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

    home = re.search(
        r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$",
        (build_dir / "CMakeCache.txt").read_text(),
        re.MULTILINE,
    )
    if home is None or pathlib.Path(home.group(1)).resolve() != src.resolve():
        say(f"{build_dir} builds {home.group(1) if home else 'no source'}, not {src}")
        return 2
    mutants = json.loads(a.mutants.read_text())
    decided = set()
    if results is not None and results.exists():
        decided = {json.loads(line)["name"] for line in results.read_text().splitlines() if line}
    todo = [m for m in mutants if m["name"] not in decided]
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
    all_caught = True
    for m in todo:
        key = binaries_sha(build_dir / m["exe"]) + " " + m["test"]
        if key not in baseline:
            ran, passed, seconds, _ = run_test(build_dir, m, False, 3600.0)
            baseline[key] = [ran, passed, seconds]
            if baseline_file is not None:
                baseline_file.write_text(json.dumps(baseline, indent=1))
        ran, passed, seconds = baseline[key]
        if not ran or not passed:
            reason = "matches no test case" if not ran else "fails"
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
            path.write_bytes(mutated)
            if build(build_dir, m["target"]):
                ran, passed, _, killed = run_test(
                    build_dir, m, True, max(a.min_timeout, 10 * seconds)
                )
                if killed:
                    verdict = "caught by timeout"
                elif not ran:
                    verdict = "matches no test case"
                else:
                    verdict = "survived" if passed else "caught"
        finally:
            path.write_bytes(original)
            path.touch()
        if path.read_bytes() != original:
            say(f"{m['name']}: the source was not restored")
            return 2
        if not build(build_dir, m["target"]):
            say(f"{m['name']}: the restored source does not build")
            return 2
        say(f"{m['name']}: {verdict}")
        if results is not None:
            with results.open("a") as out:
                out.write(json.dumps({"name": m["name"], "verdict": verdict}) + "\n")
        all_caught = all_caught and verdict in CAUGHT
    return 0 if all_caught else 1


if __name__ == "__main__":
    sys.exit(main())
