"""Export V&V mesh family levels as face arrays for the spike driver (run in the autan-vv env).

uv run --project <vv> python codegen/spike/export_vv_family.py KIND N[,N...] PERTURBATION OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from autan_vv.meshes.families import generate_level


def main() -> None:
    kind, levels, perturbation, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    for n in (int(x) for x in levels.split(",")):
        mesh, version = generate_level(kind, n, perturbation=perturbation, seed=2026)
        cell1 = np.full(mesh.n_faces, -1, dtype=np.int32)
        cell1[: mesh.n_interior] = mesh.neighbour
        tag = f"{kind}{'-dirty' if perturbation else ''}-n{n:03d}"
        np.savez(
            out / f"{tag}.npz",
            points=mesh.points,
            face_offsets=mesh.face_offsets.astype(np.int64),
            face_nodes=mesh.face_nodes.astype(np.int32),
            cell0=mesh.owner.astype(np.int32),
            cell1=cell1,
            n_cells=mesh.n_cells,
            mesh_sha256=mesh.sha256(),
            gmsh=version,
        )
        sys.stdout.write(f"{tag} cells={mesh.n_cells}\n")


if __name__ == "__main__":
    main()
