"""One-layer hexahedral meshes from the 2D PLOT3D grids of the NASA Turbulence Modeling Resource.

Usage: python physics/tmr/plot3d.py GRID.p2dfmt[.gz] OUT.msh --zones flatplate

A 2D structured grid (i along the first index, j the second) becomes ni-1 x nj-1 hexahedra one cell
thick in z (z from 0 to `--span`), written as a Fluent ASCII mesh by autan-vv. Boundary zones come
from the grid's edges, split where the case needs it (`ZONES`), plus `sides` for the two
symmetry planes of the 2D problem. Zone types are placeholders: the case sets every condition.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
from autan_vv.meshes.facemesh import from_faces
from autan_vv.meshes.fluent import write_fluent


def read_plot3d_2d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """x and y of a formatted single-block 2D PLOT3D file, shape (nj, ni)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        values = f.read().split()
    if int(values[0]) != 1:
        msg = f"{path}: {values[0]} blocks, one expected"
        raise ValueError(msg)
    ni, nj = int(values[1]), int(values[2])
    data = np.array(values[3 : 3 + 2 * ni * nj], dtype=np.float64)
    if data.size != 2 * ni * nj:
        msg = f"{path}: {data.size} values for a {ni} x {nj} grid"
        raise ValueError(msg)
    return data[: ni * nj].reshape(nj, ni), data[ni * nj :].reshape(nj, ni)


def _flatplate(x: np.ndarray, side: str, index: np.ndarray) -> str:
    """TMR 2DZP: symmetry ahead of the plate (x < 0) and the plate on y = 0."""
    if side == "jmin":
        return "plate" if x[0, index].mean() >= 0.0 else "symmetry"
    return {"imin": "inflow", "imax": "outflow", "jmax": "top"}[side]


ZONES = {"flatplate": _flatplate}


def extrude(x: np.ndarray, y: np.ndarray, span: float, zone_of) -> object:
    nj, ni = x.shape
    nci, ncj = ni - 1, nj - 1
    pts = np.zeros((nj, ni, 2, 3))
    pts[..., 0] = x[..., None]
    pts[..., 1] = y[..., None]
    pts[:, :, 1, 2] = span
    points = pts.reshape(-1, 3)

    def node(j, i, k):
        return (j * ni + i) * 2 + k

    def cell(j, i):
        return j * nci + i

    centre = np.zeros((ncj, nci, 3))
    for dj in (0, 1):
        for di in (0, 1):
            centre[..., 0] += 0.25 * x[dj : dj + ncj, di : di + nci]
            centre[..., 1] += 0.25 * y[dj : dj + ncj, di : di + nci]
    centre[..., 2] = 0.5 * span
    centre = centre.reshape(-1, 3)

    def oriented(polys: np.ndarray, owner: np.ndarray, into: np.ndarray | None) -> np.ndarray:
        # Newell normal from the owner outward (towards the neighbour for interior faces).
        p = points[polys]
        n = np.cross(p[:, 0], p[:, 1]) + np.cross(p[:, 1], p[:, 2])
        n += np.cross(p[:, 2], p[:, 3]) + np.cross(p[:, 3], p[:, 0])
        target = p.mean(axis=1) if into is None else centre[into]
        flip = np.einsum("ij,ij->i", n, target - centre[owner]) < 0.0
        polys = polys.copy()
        polys[flip] = polys[flip][:, ::-1]
        return polys

    jj, ii = np.meshgrid(np.arange(ncj), np.arange(nci), indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()
    interior_polys, owners, neighbours = [], [], []
    boundary: dict[str, list] = {}

    def add_boundary(name: str, polys: np.ndarray, owner: np.ndarray) -> None:
        boundary.setdefault(name, []).append((polys, owner))

    # i faces (constant i), between (j, i-1) and (j, i).
    for i in range(ni):
        j = np.arange(ncj)
        polys = np.stack(
            [node(j, i, 0), node(j + 1, i, 0), node(j + 1, i, 1), node(j, i, 1)], axis=1
        )
        if 0 < i < ni - 1:
            o, nb = cell(j, i - 1), cell(j, i)
            interior_polys.append(oriented(polys, o, nb))
            owners.append(o)
            neighbours.append(nb)
        else:
            o = cell(j, 0 if i == 0 else nci - 1)
            add_boundary(
                zone_of(x, "imin" if i == 0 else "imax", np.array([i])), oriented(polys, o, None), o
            )
    # j faces (constant j), between (j-1, i) and (j, i); boundary edges split face by face.
    for j in range(nj):
        i = np.arange(nci)
        polys = np.stack(
            [node(j, i, 0), node(j, i + 1, 0), node(j, i + 1, 1), node(j, i, 1)], axis=1
        )
        if 0 < j < nj - 1:
            o, nb = cell(j - 1, i), cell(j, i)
            interior_polys.append(oriented(polys, o, nb))
            owners.append(o)
            neighbours.append(nb)
        else:
            o = cell(0 if j == 0 else ncj - 1, i)
            side = "jmin" if j == 0 else "jmax"
            names = np.array([str(zone_of(x, side, np.array([k, k + 1]))) for k in i])
            for name in dict.fromkeys(names):
                m = names == name
                add_boundary(name, oriented(polys[m], o[m], None), o[m])
    # The two symmetry planes of the 2D problem.
    c = cell(jj, ii)
    for k, name in ((0, "sides"), (1, "sides")):
        polys = np.stack(
            [node(jj, ii, k), node(jj, ii + 1, k), node(jj + 1, ii + 1, k), node(jj + 1, ii, k)],
            axis=1,
        )
        add_boundary(name, oriented(polys, c, None), c)

    mesh = from_faces(
        points,
        nci * ncj,
        (np.concatenate(interior_polys), np.concatenate(owners), np.concatenate(neighbours)),
        {
            name: (
                "wall" if name == "plate" else "pressure-far-field",
                np.concatenate([p for p, _ in parts]),
                np.concatenate([o for _, o in parts]),
            )
            for name, parts in boundary.items()
        },
        cell_kinds=("hex",) * (nci * ncj),
    )
    mesh.check()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--zones", choices=sorted(ZONES), required=True)
    parser.add_argument("--span", type=float, default=1.0)
    args = parser.parse_args()
    x, y = read_plot3d_2d(args.grid)
    mesh = extrude(x, y, args.span, ZONES[args.zones])
    write_fluent(mesh, args.out, "fluid")
    sys.stdout.write(f"{args.out}: {mesh.n_cells} cells, zones {[z.name for z in mesh.zones]}\n")


if __name__ == "__main__":
    main()
