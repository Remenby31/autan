"""End-to-end verification: mesh family, reference solve, observed order."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cfd_vv.meshes.families import generate_family
from cfd_vv.mms import advection_diffusion
from cfd_vv.reference_fv import solve
from cfd_vv.study import order_study, write_report

if TYPE_CHECKING:
    from pathlib import Path


def advection_diffusion_study(
    kind: str, levels: tuple[int, ...], out_dir: Path, *, perturbation: float = 0.0, seed: int = 0
) -> dict[str, Any]:
    """Generate a family into `out_dir`, solve the manufactured advection-diffusion problem on every
    level and write `order.json` and `order.png` there. Returns the report."""
    family, meshes = generate_family(kind, levels, out_dir, perturbation=perturbation, seed=seed)
    ms = advection_diffusion()
    rows = []
    for lv, mesh in zip(family.levels, meshes, strict=True):
        sol = solve(mesh, ms)
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
        "precision": "fp64",
        "solver": "cfd_vv.reference_fv",
    }
    write_report(report, out_dir / "order.json", out_dir / "order.png")
    return report
