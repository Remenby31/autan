#!/usr/bin/env python3
"""Splits the CPU tests of a build into N shards, for gpu-tests' hosted arm64 jobs.

    bench/ci/cpu-test-shards.py BUILD_DIR OUT_DIR N

The CPU tests are the two sets gpu-tests ran on the Spark before they moved: `-LE 'gpu|mpi|certificate'`
(run with -j 4) and `-L mpi -LE gpu` (the multi-rank tests, run with -j 1). Labels are matched as
`ctest -L` matches them. Each set is dealt to the shards in the order of the tests' names, round
robin, so every shard, building on its own runner, deals the same tests to the same shard whatever
order its build lists them in. Shard K's part of each set is written to OUT_DIR/cpu-K-serial and
OUT_DIR/cpu-K-mpi as a ctest `-I` selection (`0,0,0,N,N,...`: those test numbers, in this build),
or as an empty file when the shard has no test of that set; the names of each whole set to
OUT_DIR/cpu-all-<set>.names, so that the workflow can check across runners that the shards ran
the same sets.

Then the guard: ctest itself lists what each selection runs, and what the two label selections run.
The script fails if a set is empty, if a shard has no test at all, if a test of a set is in no shard,
if a test is in two shards, or if a selection runs a test outside its set.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

SETS = {"serial": ("-LE", "gpu|mpi|certificate"), "mpi": ("-L", "mpi", "-LE", "gpu")}


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
        sys.stdout.write(f"::error::CPU test shards: ctest {what} could not list the tests\n")
        raise SystemExit(1)
    return json.loads(run.stdout)["tests"]


def labels(test: dict) -> list[str]:
    for prop in test.get("properties", []):
        if prop["name"] == "LABELS":
            return list(prop["value"])
    return []


def in_set(test: dict, name: str) -> bool:
    """Whether the ctest label selection of set `name` selects this test (a regex search)."""
    found = labels(test)
    gpu = any(re.search("gpu", label) for label in found)
    mpi = any(re.search("mpi", label) for label in found)
    certificate = any(re.search("certificate", label) for label in found)
    if name == "mpi":
        return mpi and not gpu
    return not (gpu or mpi or certificate)


def main(build: str, out_dir: str, n: int) -> int:
    if n < 1:
        sys.stdout.write("::error::CPU test shards: need at least one shard\n")
        return 1
    # Test numbers are positions in the unfiltered list, counted from 1.
    every = listed(build)
    by_name: dict[str, dict[str, int]] = {name: {} for name in SETS}
    for number, test in enumerate(every, start=1):
        for name in SETS:
            if in_set(test, name):
                by_name[name][test["name"]] = number
    # Dealt in name order: the same split on every runner that builds this commit.
    numbers = {name: [tests[t] for t in sorted(tests)] for name, tests in by_name.items()}

    out = Path(out_dir)
    errors: list[str] = []
    counts: dict[int, int] = {}
    for name, selection in SETS.items():
        expected = [t["name"] for t in listed(build, *selection)]
        if not expected:
            errors.append(f"no test in set {name} ({' '.join(selection)})")
        if len(expected) != len(set(expected)) or len(by_name[name]) != len(expected):
            errors.append(f"two tests of set {name} share a name: the guard compares names")
        (out / f"cpu-all-{name}.names").write_text("".join(e + "\n" for e in sorted(expected)))
        seen: dict[str, int] = {}
        for k in range(1, n + 1):
            mine = numbers[name][k - 1 :: n]
            path = out / f"cpu-{k}-{name}"
            if not mine:
                # `-I 0,0,0` with no number would select every test, not none.
                path.write_text("")
                continue
            sel = "0,0,0," + ",".join(map(str, mine))
            path.write_text(sel + "\n")
            got = [t["name"] for t in listed(build, "-I", sel)]
            (out / f"cpu-{k}-{name}.names").write_text("".join(g + "\n" for g in got))
            counts[k] = counts.get(k, 0) + len(got)
            for g in got:
                if g in seen:
                    errors.append(f"in shards {seen[g]} and {k}: {g}")
                seen[g] = k
            errors += [
                f"shard {k} runs a test outside set {name}: {g}" for g in got if g not in expected
            ]
        errors += [f"set {name}, in no shard: {e}" for e in expected if e not in seen]
        sys.stdout.write(f"set {name}: {len(expected)} tests over {n} shards\n")
    errors += [f"shard {k} has no test" for k in range(1, n + 1) if not counts.get(k)]
    for e in errors:
        sys.stdout.write(f"::error::CPU test shards: {e}\n")
    if errors:
        return 1
    sys.stdout.write(
        "each CPU test in exactly one shard: "
        + ", ".join(f"shard {k} {counts[k]}" for k in range(1, n + 1))
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
