#!/usr/bin/env python3
"""Fails unless at least one test of a pytest JUnit report actually ran (not skipped).

    bench/ci/require-ran.py REPORT.xml

A run whose tests were all skipped (the GPU used by another tenant, a missing binary) prints
the skip reasons as an error annotation and exits 1: a green check must have run something.
"""

import sys
import xml.etree.ElementTree as ET


def main(path: str) -> int:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skipped = [c for c in cases if c.find("skipped") is not None]
    ran = len(cases) - len(skipped)
    if ran:
        sys.stdout.write(f"{ran} of {len(cases)} tests ran\n")
        return 0
    reasons = sorted({c.find("skipped").get("message", "") for c in skipped})
    sys.stdout.write(f"::error::no test ran ({len(cases)} skipped): {'; '.join(reasons)}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
