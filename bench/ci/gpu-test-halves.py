#!/usr/bin/env python3
"""Splits the GPU tests (ctest label `gpu`) into two halves, each run in its own GPU-lock hold.

    bench/ci/gpu-test-halves.py BUILD_DIR OUT_DIR

The split is by label, matched as `ctest -L` matches them, so it is the same on every run:
half 1 is the GPU tests that also carry `mpi` (the multi-rank tests), half 2 the other GPU
tests. Each half is written to OUT_DIR/gpu-half-K as a ctest `-I` selection (`0,0,0,N,N,...`:
those test numbers only), and its tests' names to OUT_DIR/gpu-half-K.names.

Then the guard: ctest itself lists what each `-I` selection runs, and what `-L gpu` runs. The
script fails if a half is empty, if a GPU test is in neither half, if a test is in both halves,
or if a half selects a test that is not a GPU test.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def listed(build: str, *selection: str) -> list[dict]:
    """The tests ctest selects with these arguments, in test-number order."""
    run = subprocess.run(
        ["ctest", "--test-dir", build, "--show-only=json-v1", *selection],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        sys.stdout.write(run.stdout + run.stderr)
        what = " ".join(selection) or "(no selection)"
        sys.stdout.write(f"::error::GPU test split: ctest {what} could not list the tests\n")
        raise SystemExit(1)
    return json.loads(run.stdout)["tests"]


def labelled(test: dict, regex: str) -> bool:
    """Whether ctest -L REGEX selects this test: a label matches the regex (a search)."""
    for prop in test.get("properties", []):
        if prop["name"] == "LABELS":
            return any(re.search(regex, label) for label in prop["value"])
    return False


def main(build: str, out_dir: str) -> int:
    # Test numbers are positions in the unfiltered list, counted from 1.
    every = listed(build)
    halves: dict[int, list[int]] = {1: [], 2: []}
    for number, test in enumerate(every, start=1):
        if labelled(test, "gpu"):
            halves[1 if labelled(test, "mpi") else 2].append(number)

    gpu = [t["name"] for t in listed(build, "-L", "gpu")]
    selected: dict[int, list[str]] = {}
    errors = []
    for k, numbers in halves.items():
        if not numbers:
            # `-I 0,0,0` with no number would select every test, not none.
            errors.append(f"half {k} is empty")
            continue
        selection = "0,0,0," + ",".join(map(str, numbers))
        selected[k] = [t["name"] for t in listed(build, "-I", selection)]
        out = Path(out_dir)
        (out / f"gpu-half-{k}").write_text(selection + "\n")
        (out / f"gpu-half-{k}.names").write_text("".join(n + "\n" for n in selected[k]))

    one, two = set(selected.get(1, [])), set(selected.get(2, []))
    errors += [f"in neither half: {n}" for n in sorted(set(gpu) - one - two)]
    errors += [f"in both halves: {n}" for n in sorted(one & two)]
    errors += [f"not a GPU test: {n}" for n in sorted((one | two) - set(gpu))]
    if len(gpu) != len(set(gpu)):
        errors.append("two GPU tests share a name: the guard compares names")
    for e in errors:
        sys.stdout.write(f"::error::GPU test split: {e}\n")
    if errors:
        return 1
    sys.stdout.write(
        f"{len(gpu)} GPU tests: half 1 (multi-rank) {len(one)}, half 2 {len(two)}, "
        "each in exactly one half\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
