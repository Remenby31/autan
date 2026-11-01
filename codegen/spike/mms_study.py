"""MMS order study of the generated advection-diffusion scheme on V&V mesh families.

uv run python spike/mms_study.py FAMILIES_DIR OUT_DIR [policy,...] [strategy]

FAMILIES_DIR holds one directory per family as `autan-vv meshes` writes it (mesh files and
manifest.json); every level is read by the core readers. Writes one errors file per (family,
policy) in the input format of `autan-vv order`, with the mesh hash, case hash
(of mms.py), solver commit and precision of every level, and prints the observed orders. With a race strategy, the residual runs on the
device loops of the AUTAN_SPIKE_LIB driver (CUDA or Metal) and each level records it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import autan_spike as s
import mms


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    policies = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fp64", "fp32+acc64", "fp32"]
    strategy = sys.argv[4] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    # The manufactured case (solution, source, coefficients) is the content of mms.py.
    case_sha = hashlib.sha256(Path(mms.__file__).read_bytes()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    for fam_dir in sorted(d for d in src.iterdir() if (d / "manifest.json").exists()):
        manifest = json.loads((fam_dir / "manifest.json").read_text())
        family = fam_dir.name
        for policy in policies:
            levels = []
            for lv in manifest["levels"]:
                sha = lv["mesh_sha256"]
                m = s.read(str(fam_dir / lv["file"]))
                sol = mms.solve(m, policy, 1e-12 if policy == "fp64" else 1e-6, strategy)
                levels.append({
                    "n_cells": m.n_cells, "volume": float(m.cell_volume.sum()),
                    "errors": {"L1": sol.error_l1, "L2": sol.error_l2, "Linf": sol.error_linf},
                    "mesh_sha256": sha, "case_sha256": case_sha, "solver_commit": commit, "precision": policy,
                    "relative_residual": sol.residual, "strategy": strategy or "host",
                })  # fmt: skip
            label = f"{family} {policy}" + (f" {strategy}" if strategy else "")
            (out / f"{family}-{policy}{'-' + strategy if strategy else ''}.json").write_text(
                json.dumps({"label": label, "dim": 3, "levels": levels}, indent=1) + "\n"
            )
            h = [(lv["volume"] / lv["n_cells"]) ** (1 / 3) for lv in levels]
            row = [label]
            for norm in ("L2", "Linf"):
                fit, pair = mms.observed_order(h, [lv["errors"][norm] for lv in levels])
                row.append(f"{norm} {fit:.2f} / {pair:.2f}")
            sys.stdout.write(" | ".join(row) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
