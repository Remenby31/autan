"""MMS order study of the generated advection-diffusion scheme on exported V&V mesh families.

uv run python spike/mms_study.py FAMILY_DIR OUT_DIR [policy,...]

Writes one errors file per (family, policy) in the input format of `autan-vv order`, with the mesh
hash, solver commit and precision of every level, and prints the observed orders.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import autan_spike as s
import mms
import numpy as np
from meshes import FaceArrays


def load(path: Path) -> tuple[FaceArrays, str]:
    z = np.load(path)
    fa = FaceArrays(z["points"], z["face_offsets"], z["face_nodes"], z["cell0"], z["cell1"], int(z["n_cells"]))
    return fa, str(z["mesh_sha256"])


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    policies = sys.argv[3].split(",") if len(sys.argv) > 3 else ["fp64", "fp32+acc64", "fp32"]
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    families: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(src.glob("*.npz")):
        families[p.stem.rsplit("-n", 1)[0]].append(p)
    for family, paths in families.items():
        for policy in policies:
            levels = []
            for p in paths:
                fa, sha = load(p)
                m = s.mesh(fa)
                sol = mms.solve(m, policy, tol=1e-12 if policy == "fp64" else 1e-6)
                levels.append({
                    "n_cells": m.n_cells, "volume": float(m.cell_volume.sum()),
                    "errors": {"L1": sol.error_l1, "L2": sol.error_l2, "Linf": sol.error_linf},
                    "mesh_sha256": sha, "solver_commit": commit, "precision": policy,
                    "relative_residual": sol.residual,
                })  # fmt: skip
            label = f"{family} {policy}"
            (out / f"{family}-{policy}.json").write_text(json.dumps({"label": label, "dim": 3, "levels": levels}, indent=1))
            h = [(lv["volume"] / lv["n_cells"]) ** (1 / 3) for lv in levels]
            row = [label]
            for norm in ("L2", "Linf"):
                fit, pair = mms.observed_order(h, [lv["errors"][norm] for lv in levels])
                row.append(f"{norm} {fit:.2f} / {pair:.2f}")
            sys.stdout.write(" | ".join(row) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
