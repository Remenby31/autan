"""End-to-end verification: mesh family, reference solve, observed order."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from autan_vv.errors import VVError
from autan_vv.meshes.families import generate_family
from autan_vv.mms import advection_diffusion, dump_json
from autan_vv.reference_fv import solve
from autan_vv.study import order_study, write_report


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = "cannot record the V&V commit: not running from a git checkout"
        raise VVError(msg, code="V0502") from exc
    return out.stdout.strip()


def _provenance() -> dict[str, Any]:
    """SHA of the checkout holding this package, and whether it has uncommitted changes."""
    return {
        "commit": _git("rev-parse", "--verify", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def advection_diffusion_study(
    kind: str,
    levels: tuple[int, ...],
    out_dir: Path,
    *,
    perturbation: float = 0.0,
    seed: int = 0,
    precision: str = "fp64",
) -> dict[str, Any]:
    """Generate a family into `out_dir`, solve the manufactured advection-diffusion problem on every
    level and write `order.json` and `order.png` there. Returns the report.

    The report records the mesh hashes, the hash of the manufactured solution (its JSON export),
    the precision and the commit of the V&V code (V0502 outside a git checkout).
    """
    family, meshes = generate_family(kind, levels, out_dir, perturbation=perturbation, seed=seed)
    ms = advection_diffusion()
    rows = []
    for lv, mesh in zip(family.levels, meshes, strict=True):
        sol = solve(mesh, ms, precision)
        rows.append(
            {
                "n_cells": lv.n_cells,
                "volume": lv.volume,
                "mesh_sha256": lv.mesh_sha256,
                "file": lv.file,
                "gmres_iterations": sol.iterations,
                "relative_residual": sol.residual_norm,
                "errors": {"L1": sol.error_l1, "L2": sol.error_l2, "Linf": sol.error_linf},
            }
        )
    label = f"advection-diffusion MMS, {kind}{' perturbed ' + str(perturbation) if perturbation else ''}"
    report = order_study(label, 3, rows)
    report |= {
        "mesh_family": family.manifest(),
        "precision": precision,
        "solver": "autan_vv.reference_fv",
        "mms_sha256": hashlib.sha256(dump_json(ms).encode()).hexdigest(),
        **_provenance(),
    }
    write_report(report, out_dir / "order.json", out_dir / "order.png")
    return report
