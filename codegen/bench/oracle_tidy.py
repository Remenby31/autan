"""Benchmark for tidy (ADR 0003): the time of a 3000-operation certified chain. Reported, not
asserted; run with AUTAN_SPIKE_LIB unset: `python codegen/bench/oracle_tidy.py`."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from oracle import certified as c

x = c.exact(0.1)
start = time.perf_counter()
for _ in range(1000):
    x = c.div(
        c.add(c.mul(x, c.exact(1.0 + 2.0**-20), c.BINARY64), c.exact(0.3), c.BINARY64),
        c.exact(3.0),
        c.BINARY64,
    )
sys.stdout.write(f"3000 certified operations in {time.perf_counter() - start:.3f} s\n")
