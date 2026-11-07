"""One-layer hexahedral meshes from the 2D PLOT3D grids of the NASA Turbulence Modeling Resource.

Usage: python physics/tmr/plot3d.py GRID.p2dfmt[.gz] OUT.msh --zones flatplate

A 2D structured grid (i along the first index, j the second) becomes ni-1 x nj-1 hexahedra one cell
thick in z (z from 0 to `--span`), or in -y with the grid in the x-z plane (`--plane xz`, for
airfoils at incidence: a case's angle of attack turns the velocity in x-z), written as a Fluent ASCII mesh by autan-vv. Boundary zones come
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


def _bump(x: np.ndarray, side: str, index: np.ndarray) -> str:
    """TMR 2DBC: the wall from x = 0 to 1.5 (the bump in between), symmetry on y = 0 elsewhere."""
    if side == "jmin":
        return "wall" if 0.0 <= x[0, index].mean() <= 1.5 else "symmetry"
    return {"imin": "inflow", "imax": "outflow", "jmax": "top"}[side]


def _airfoil(x: np.ndarray, side: str, index: np.ndarray) -> str:
    """C-grid around an airfoil (TMR 2DN00): j = 0 is the airfoil and the wake cut, which `extrude`
    joins into interior faces; every other edge is farfield."""
    del x, index
    return "airfoil" if side == "jmin" else "farfield"


ZONES = {"flatplate": _flatplate, "bump": _bump, "airfoil": _airfoil}


def extrude(x: np.ndarray, y: np.ndarray, span: float, zone_of, plane: str = "xy") -> object:
    nj, ni = x.shape
    nci, ncj = ni - 1, nj - 1
    pts = np.zeros((nj, ni, 2, 3))
    pts[..., 0] = x[..., None]
    pts[..., 1] = y[..., None]
    pts[:, :, 1, 2] = span
    points = pts.reshape(-1, 3)

    # A C-grid's wake cut: nodes of j = 0 that coincide with their mirror i -> ni - 1 - i are
    # merged, so that the cut's faces become interior faces.
    merged = np.arange(points.shape[0])
    tol = 1e-9 * float(np.ptp(x))
    for i in range(ni // 2 + 1, ni):
        mirror = ni - 1 - i
        if abs(x[0, i] - x[0, mirror]) <= tol and abs(y[0, i] - y[0, mirror]) <= tol:
            for k in (0, 1):
                merged[i * 2 + k] = mirror * 2 + k

    def node(j, i, k):
        return merged[(j * ni + i) * 2 + k]

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
            keep = np.ones(i.size, dtype=bool)
            if j == 0:
                # Faces of the cut appear twice, once per side: interior faces between their cells.
                key = np.sort(polys, axis=1)
                for a in range(i.size):
                    b = i.size - 1 - a
                    if a < b and np.array_equal(key[a], key[b]):
                        interior_polys.append(
                            oriented(polys[a : a + 1], o[a : a + 1], o[b : b + 1])
                        )
                        owners.append(o[a : a + 1])
                        neighbours.append(o[b : b + 1])
                        keep[a] = keep[b] = False
            names = np.array([str(zone_of(x, side, np.array([k, k + 1]))) for k in i])
            for name in dict.fromkeys(names[keep]):
                m = keep & (names == name)
                add_boundary(name, oriented(polys[m], o[m], None), o[m])
    # The two symmetry planes of the 2D problem.
    c = cell(jj, ii)
    for k, name in ((0, "sides"), (1, "sides")):
        polys = np.stack(
            [node(jj, ii, k), node(jj, ii + 1, k), node(jj + 1, ii + 1, k), node(jj + 1, ii, k)],
            axis=1,
        )
        add_boundary(name, oriented(polys, c, None), c)

    used = np.unique(merged)
    renumber = np.full(points.shape[0], -1)
    renumber[used] = np.arange(used.size)
    interior_polys = [renumber[q] for q in interior_polys]
    boundary = {name: [(renumber[q], o) for q, o in parts] for name, parts in boundary.items()}
    points = points[used]
    if plane == "xz":
        # The grid's y becomes z and the layer extends towards -y: a rotation, which keeps every
        # face's orientation (the angle of attack of a case turns the velocity in x-z).
        points = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    mesh = from_faces(
        points,
        nci * ncj,
        (np.concatenate(interior_polys), np.concatenate(owners), np.concatenate(neighbours)),
        {
            name: (
                "wall" if name in ("plate", "wall", "airfoil") else "pressure-far-field",
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
    parser.add_argument(
        "--plane", choices=("xy", "xz"), default="xy", help="plane of the 2D grid (xz for lift)"
    )
    args = parser.parse_args()
    x, y = read_plot3d_2d(args.grid)
    mesh = extrude(x, y, args.span, ZONES[args.zones], args.plane)
    write_fluent(mesh, args.out, "fluid")
    sys.stdout.write(f"{args.out}: {mesh.n_cells} cells, zones {[z.name for z in mesh.zones]}\n")


if __name__ == "__main__":
    main()
