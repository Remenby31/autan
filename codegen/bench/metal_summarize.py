"""Summarise autan-bench results of the Metal loop benchmark against the Mac's measured peaks.

Apple GPUs read faster than they copy, so a loop is measured against three references from the
Metal STREAM result: the copy peak, the read peak, and the gate reference, its own mix of reads
and writes. With B_c the copy and B_r the read medians, a read byte costs 1/B_r and a written
byte 1/B_w = 2/B_c - 1/B_r (copy moves one of each); a loop reading R and writing W bytes has the
reference (R + W) / (R/B_r + W/B_w).

python codegen/bench/metal_summarize.py STREAM.json RESULT.json [...]: one Markdown row per kernel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def peaks(stream: Path) -> tuple[float, float]:
    doc = json.loads(stream.read_text())
    med = {b["name"].split(".")[-1]: b["stats"]["median"] for b in doc["benchmarks"]}
    return med["copy"], med["read"]


def reference(total: float, written: float, copy: float, read: float) -> float:
    per_write = 2.0 / copy - 1.0 / read
    return total / ((total - written) / read + written * per_write)


def main() -> None:
    copy, read = peaks(Path(sys.argv[1]))
    out = [f"Copy peak {copy:.1f} GB/s, read peak {read:.1f} GB/s ({sys.argv[1]})", ""]
    out.append(
        "| Mesh | Cells | Policy | Kernel | GB/s | % copy | % read | Reference GB/s | % reference "
        "| ns/cell | CV |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[2:]:
        doc = json.loads(Path(path).read_text())
        cfg = doc["config"]
        mesh = Path(cfg["mesh"]).stem
        bench = {b["name"]: b for b in doc["benchmarks"]}
        for name, b in bench.items():
            if name.endswith(".time_per_cell"):
                continue
            t = bench.get(name + ".time_per_cell")
            gbs = b["stats"]["median"]
            ns = t["stats"]["median"] if t else float("nan")
            written = b.get("params", {}).get("write_bytes", 0.0)
            ref = reference(b["bytes_per_rep"], written, copy, read) if written else float("nan")
            out.append(
                f"| {mesh} | {cfg['n_cells']} | {cfg['policy']} | {name.split('.', 1)[-1]} | "
                f"{gbs:.1f} | {100 * gbs / copy:.0f} % | {100 * gbs / read:.0f} % | {ref:.1f} | "
                f"{100 * gbs / ref:.0f} % | {ns:.2f} | {100 * b['stats']['cv']:.1f} % |"
            )
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
