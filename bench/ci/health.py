#!/usr/bin/env python3
"""Decides whether the Spark is fit to measure: a reference STREAM against the GB10 baseline.

    bench/ci/health.py PROBE_RESULT BASELINE [--threshold 0.03]

The probe is STREAM built from the baseline's own commit (bench/ci/health.sh), never the binary
under test: a pull request that slowed STREAM must still fail its gate as a regression. The
machine is healthy when every probe median is at least (1 - threshold) times the baseline's
(declared before any measurement: 3 %, 2026-11-02). Otherwise it is degraded: shared memory
bandwidth taken by other tenants moved STREAM 5-6 % on both Sparks that day (#284), and a pull
request's bench would fail on the machine, not on its code. Exits 0 healthy, 3 degraded, 2 when
the files cannot be compared; prints one line per benchmark and the verdict.
"""

import argparse
import json
import sys
from pathlib import Path

HEALTHY, DEGRADED, UNUSABLE = 0, 3, 2


def medians(path: Path) -> dict[str, float]:
    doc = json.loads(path.read_text())
    return {b["name"]: b["stats"]["median"] for b in doc["benchmarks"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--threshold", type=float, default=0.03)
    args = parser.parse_args(argv)
    try:
        probe, base = medians(args.probe), medians(args.baseline)
    except (OSError, ValueError, KeyError) as exc:
        sys.stdout.write(f"health: cannot read the results: {exc}\n")
        return UNUSABLE
    missing = sorted(set(base) - set(probe))
    if missing:
        sys.stdout.write(f"health: the probe lacks {', '.join(missing)}\n")
        return UNUSABLE
    low = []
    for name in sorted(base):
        ratio = probe[name] / base[name]
        ok = ratio >= 1.0 - args.threshold
        low += [] if ok else [name]
        sys.stdout.write(f"{name:14s} probe {probe[name]:8.2f}  baseline {base[name]:8.2f}  "
                         f"{100 * (ratio - 1):+6.2f} %  {'ok' if ok else 'LOW'}\n")  # fmt: skip
    if low:
        sys.stdout.write(f"MACHINE DEGRADED: {', '.join(low)} more than "
                         f"{100 * args.threshold:g} % below the baseline\n")  # fmt: skip
        return DEGRADED
    sys.stdout.write(f"machine healthy: every probe median within {100 * args.threshold:g} %\n")
    return HEALTHY


if __name__ == "__main__":
    sys.exit(main())
