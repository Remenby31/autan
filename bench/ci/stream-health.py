#!/usr/bin/env python3
"""Whether a STREAM result says the machine is fit to measure: every benchmark of the baseline at
least MIN of its median (default 0.97, the health criterion of #404), or exit 4.

    bench/ci/stream-health.py RESULT BASELINE [--min 0.97]

Not a regression gate (bench/ci/gate.sh tolerates 5 % within noise): a bisection on a machine
3-5 % slow "confirms" the regression of every commit of bandwidth-bound loops (#400). Exits 0
when fit, 4 when not, 2 when the files cannot be read or do not match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIT, UNREADABLE, UNFIT = 0, 2, 4


def medians(path: Path) -> dict[str, tuple[float, str]]:
    doc = json.loads(path.read_text())
    return {b["name"]: (float(b["stats"]["median"]), b["better"]) for b in doc["benchmarks"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--min", type=float, default=0.97, dest="minimum")
    args = parser.parse_args(argv)
    try:
        now, base = medians(args.result), medians(args.baseline)
    except (OSError, ValueError, KeyError, TypeError) as err:
        sys.stderr.write(f"stream-health: cannot read the results: {err}\n")
        return UNREADABLE
    fit = True
    for name, (reference, better) in base.items():
        if name not in now or reference <= 0 or now[name][0] <= 0:
            sys.stderr.write(f"stream-health: {name} is missing or not positive\n")
            return UNREADABLE
        value = now[name][0]
        share = value / reference if better == "higher" else reference / value
        ok = share >= args.minimum
        fit &= ok
        verdict = "ok" if ok else "LOW"
        sys.stdout.write(
            f"{name:<16} {value:>9.2f} of {reference:>9.2f}  {share:7.1%}  {verdict}\n"
        )
    if not fit:
        sys.stderr.write(f"stream-health: below {args.minimum:.0%} of {args.baseline}: the machine "
                         "is not fit to measure\n")  # fmt: skip
    return FIT if fit else UNFIT


if __name__ == "__main__":
    sys.exit(main())
