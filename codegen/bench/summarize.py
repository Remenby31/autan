"""Summarise autan-bench results of the loop benchmark against the GB10 STREAM peak.

The footprint is the kernel's analytic minimum of bytes per repetition; the spike criterion asks
for at least 10 times the L2.

python codegen/bench/summarize.py BASELINE RESULT.json [...]: one Markdown row per kernel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def peak(baseline: Path) -> float:
    doc = json.loads(baseline.read_text())
    return max(b["stats"]["median"] for b in doc["benchmarks"] if b["name"].startswith("stream."))


def main() -> None:
    p = peak(Path(sys.argv[1]))
    out = [f"Peak (best STREAM median of {sys.argv[1]}): {p:.2f} GB/s", ""]
    out.append(
        "| Mesh | Cells | Policy | Order | Kernel | Footprint / L2 | GB/s | % peak | ns/cell | CV |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[2:]:
        doc = json.loads(Path(path).read_text())
        cfg = doc["config"]
        mesh = Path(cfg["mesh"]).stem
        bench = {b["name"]: b for b in doc["benchmarks"]}
        l2 = doc["provenance"]["gpu"]["l2_bytes"]
        for name, b in bench.items():
            if name.endswith(".time_per_cell"):
                continue
            t = bench.get(name + ".time_per_cell")
            gbs = b["stats"]["median"]
            ns = t["stats"]["median"] if t else float("nan")
            out.append(
                f"| {mesh} | {cfg['n_cells']} | {cfg['policy']} | {cfg['order']} | "
                f"{name.removeprefix('codegen.')} | {b['bytes_per_rep'] / l2:.1f} | {gbs:.1f} | "
                f"{100 * gbs / p:.0f} % | {ns:.2f} | "
                f"{100 * b['stats']['cv']:.1f} % |"
            )
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
