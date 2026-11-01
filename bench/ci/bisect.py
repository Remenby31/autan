#!/usr/bin/env python3
"""Find the commit of main that introduced a confirmed benchmark regression.

    bench/ci/bisect.py --good SHA --bad SHA --out DIR --compare CMD -- MEASURE...

GOOD is the last commit of main that passed the gate, BAD the commit whose regression the gate
confirmed. The candidates are the first-parent commits after GOOD up to BAD. Each probe runs
`MEASURE... SHA DIR/SHA` (bench/ci/measure-point.sh: same Spark, same protocol, STREAM check
first), which exits 0 when the commit passes the gate, 1 on a confirmed regression and 4 when
the machine failed its STREAM check, in which case the bisection stops: such a point is never
interpreted. A culprit is reported only if `CMD CULPRIT_DIR PARENT_DIR` (autan-bench compare on
the tracked loops) also shows a regression from its parent; otherwise the shift is spread over
several commits and DIR/summary.json says so. Writes DIR/summary.json in every case.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GOOD, BAD, UNHEALTHY = 0, 1, 4
MAX_PROBES = 8


def candidates(good: str, bad: str) -> list[str]:
    out = subprocess.run(
        ["git", "rev-list", "--first-parent", "--reverse", f"{good}..{bad}"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return out.stdout.split()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--good", required=True)
    parser.add_argument("--bad", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", required=True, help="command run as CMD NEW_DIR OLD_DIR")
    args = parser.parse_args(argv[:split])
    measure = argv[split + 1 :]
    if not measure:
        parser.error("a measure command is required after --")
    args.out.mkdir(parents=True, exist_ok=True)

    commits = candidates(args.good, args.bad)
    probes: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "good": args.good, "bad": args.bad, "candidates": commits, "probes": probes,
    }  # fmt: skip

    def probe(sha: str) -> int:
        status = subprocess.run([*measure, sha, str(args.out / sha)], check=False).returncode
        verdict = {GOOD: "pass", BAD: "regression", UNHEALTHY: "machine-unhealthy"}.get(status)
        probes.append({"commit": sha, "verdict": verdict or f"error {status}"})
        return status

    def finish(status: str, **fields: object) -> int:
        summary.update(status=status, **fields)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0 if status in ("culprit", "spread") else 1

    if not commits or commits[-1] != args.bad:
        return finish("error", at=args.bad, reason="not a first-parent descendant of the good end")
    # log2(64) + 2: the search, then the culprit and its parent at most.
    if len(commits) > 2 ** (MAX_PROBES - 2):
        return finish("error", at=args.bad, reason=f"{len(commits)} commits to bisect")
    # Invariant: commits[lo] passes (lo = -1 stands for GOOD) and commits[hi] regressed.
    lo, hi = -1, len(commits) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        status = probe(commits[mid])
        if status == UNHEALTHY:
            return finish("machine-unhealthy", at=commits[mid])
        if status not in (GOOD, BAD):
            return finish("error", at=commits[mid])
        lo, hi = (mid, hi) if status == GOOD else (lo, mid)

    culprit = commits[hi]
    parent = commits[lo] if lo >= 0 else args.good
    # The ends of the bracket, if not measured during the search, must confirm their verdicts:
    # a passing culprit or a failing parent means the search rested on noise.
    for sha, expected in ((culprit, BAD), (parent, GOOD)):
        if not (args.out / sha).is_dir():
            status = probe(sha)
            if status == UNHEALTHY:
                return finish("machine-unhealthy", at=sha)
            if status != expected:
                return finish("error", at=sha, reason="the bracket did not reproduce")
    compare = subprocess.run(
        [*args.compare.split(), str(args.out / culprit), str(args.out / parent)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    evidence = {"parent": parent, "comparison": compare.stdout[-4000:]}
    if compare.returncode == 1:
        return finish("culprit", culprit=culprit, **evidence)
    if compare.returncode in (0, 3):
        return finish("spread", last_passing=parent, first_failing=culprit, **evidence)
    return finish("error", at=culprit, reason="the culprit and its parent could not be compared")


if __name__ == "__main__":
    sys.exit(main())
