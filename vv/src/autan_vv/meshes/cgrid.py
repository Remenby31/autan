"""C-grids around 2D airfoils by hyperbolic marching, written as 2D PLOT3D grids.

J. L. Steger and R. L. Sorenson, Use of hyperbolic partial differential equations to generate body
fitted coordinates, NASA Langley Numerical Grid Generation Techniques, 1980, pp. 463-476 (NTRS
19810006209; page numbers below are theirs); the method is that of J. L. Steger and D. S. Chaussee,
SIAM J. Sci. Stat. Comput. 1(4):431-437, 1980, doi:10.1137/0901031. Grid lines of constant eta are
marched away from the airfoil and its wake cut (eta = 0) under the volume-orthogonality system
x_xi x_eta + y_xi y_eta = 0, x_xi y_eta - x_eta y_xi = V (p. 465), linearised about the known line as
A r_xi + B r_eta = (0, V + V~) (pp. 466-467) and solved by a non-iterative implicit step, central in
xi and one-sided in eta, a block-tridiagonal system per line (p. 468). The cell area V is the
prescribed normal step h_k times the local xi spacing, so that every line lies h_k from the previous
one; the paper takes V from a separate control-volume grid (p. 469). The implicit second-difference
smoothing of each step in xi, zero at the wall and growing over the first lines, is ours: the paper
gives none.

i runs along eta = 0 from the end of the wake below the cut, along the wake to the trailing edge,
around the airfoil (lower surface, leading edge, upper surface) and back along the wake above the
cut; the two wake halves coincide node for node (i and ni - 1 - i), which `physics/tmr/plot3d.py`
merges into interior faces. Levels nest: each is the finer one with every other line removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from autan_vv.cases import read_tape_1981
from autan_vv.errors import VVError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CGrid:
    """Node coordinates (nj, ni) and the i index of the trailing edge below the cut."""

    x: FloatArray
    y: FloatArray
    wake_cells: int

    def coarsened(self) -> CGrid:
        """Every other line in i and j: the next coarser nested level."""
        return CGrid(self.x[::2, ::2], self.y[::2, ::2], self.wake_cells // 2)


def _ratio(first: float, total: float, cells: int) -> float:
    """r of a geometric progression of `cells` steps starting at `first` that sums to `total`."""
    if first * cells >= total:
        msg = f"a first step of {first} over {cells} cells exceeds the length {total}"
        raise VVError(msg, code="V0310")
    lo, hi = 1.0 + 1e-12, 2.0
    while first * (hi**cells - 1.0) / (hi - 1.0) < total:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if first * (mid**cells - 1.0) / (mid - 1.0) < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _side(points: FloatArray, cells: int, relax: float) -> FloatArray:
    """`cells` + 1 nodes from the leading edge (first point) to the trailing edge (last) along a
    cubic spline in arc length: s = L [relax (1 - cos(pi t / 2)) + (1 - relax) (1 - cos(pi t)) / 2],
    clustered at the leading edge by both terms and at the trailing edge by the second only, so
    that `relax` sets the trailing-edge spacing (about relax pi L / (2 cells))."""
    chord = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    spline = CubicSpline(chord, points, bc_type="not-a-knot")
    fine = np.linspace(0.0, chord[-1], 20 * len(points))
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(spline(fine), axis=0).T))]
    t = np.arange(cells + 1) / cells
    target = arc[-1] * (
        relax * (1.0 - np.cos(0.5 * np.pi * t)) + (1.0 - relax) * 0.5 * (1.0 - np.cos(np.pi * t))
    )
    nodes = spline(np.interp(target, arc, fine))
    nodes[[0, -1]] = points[[0, -1]]  # the data's end points, bit for bit
    return nodes


def _surface(upper: FloatArray, lower: FloatArray, cells: int, relax: float) -> FloatArray:
    """Airfoil nodes from the trailing edge along the lower surface to the leading edge and along
    the upper surface back to the trailing edge; `upper` and `lower` run from the leading edge to
    the trailing edge and share both end points. Each surface is splined on its own: the leading
    edge is smooth in the data but the trailing edge is a corner."""
    if not (np.array_equal(upper[0], lower[0]) and np.array_equal(upper[-1], lower[-1])):
        msg = "the upper and lower surfaces must share the leading and trailing edges"
        raise VVError(msg, code="V0311")
    half = cells // 2
    low = _side(lower, half, relax)[::-1]
    up = _side(upper, half, relax)
    return np.vstack([low, up[1:]])


def _step(r: FloatArray, area: FloatArray, smoothing: float) -> FloatArray:
    """r^{k+1} - r^k for the cell areas `area` per node, Steger and Sorenson's linearised
    volume-orthogonality system, with a dissipation on the step of `smoothing` times the node's
    step-to-spacing ratio, the size of its coupling C (where a line's step is long against its
    xi spacing, as off a sharp trailing edge far from the wall, the march needs it most)."""
    ni = len(r)
    d = np.empty_like(r)
    d[1:-1] = 0.5 * (r[2:] - r[:-2])
    d[0], d[-1] = r[1] - r[0], r[-1] - r[-2]
    xx, yx = d[:, 0], d[:, 1]
    norm2 = xx * xx + yx * yx
    eps = smoothing * np.maximum(1.0, area / norm2)
    # The known state's eta derivative: the step that solves both equations for this line.
    xe, ye = -yx * area / norm2, xx * area / norm2
    # B^-1 A, symmetric (p. 467), and B^-1 (0, V + V~) with V~ = V.
    c11 = (xx * xe - yx * ye) / norm2
    c12 = (xx * ye + yx * xe) / norm2
    rhs = np.stack([-yx * 2.0 * area / norm2, xx * 2.0 * area / norm2], axis=1)
    # In delta form, dr = r^{k+1} - r^k: (I + C delta_xi - eps delta_xixi) dr = B^-1 f -
    # C delta_xi r^k, delta_xi central; the dissipation smooths the step, not the line (a
    # line's own second difference is large where xi is stretched, along the wake).
    dxi = np.zeros_like(r)
    dxi[1:-1] = 0.5 * (r[2:] - r[:-2])
    b = rhs - np.stack([c11 * dxi[:, 0] + c12 * dxi[:, 1], c12 * dxi[:, 0] - c11 * dxi[:, 1]], 1)
    inner = np.arange(1, ni - 1)
    rows, cols, vals = [], [], []
    for c in (0, 1):
        rows.append(2 * inner + c)
        cols.append(2 * inner + c)
        vals.append(1.0 + 2.0 * eps[inner])
        for e in (0, 1):
            coef = (c11, c12)[e] if c == 0 else (c12, -c11)[e]
            for side in (1, -1):
                rows.append(2 * inner + c)
                cols.append(2 * (inner + side) + e)
                vals.append(side * 0.5 * coef[inner] - (eps[inner] if c == e else 0.0))
        # The wake's ends: dr equal to their neighbour's, the outflow line kept straight.
        for i, j in ((0, 1), (ni - 1, ni - 2)):
            rows += [np.array([2 * i + c]), np.array([2 * i + c])]
            cols += [np.array([2 * i + c]), np.array([2 * j + c])]
            vals += [np.array([1.0]), np.array([-1.0])]
    b[[0, -1]] = 0.0
    matrix = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * ni, 2 * ni)
    ).tocsr()
    return spsolve(matrix, b.ravel()).reshape(ni, 2)


def _march(line: FloatArray, steps: FloatArray, smoothing: float) -> tuple[FloatArray, FloatArray]:
    """Nodes (len(steps) + 1, ni) of the lines marched from `line` (ni, 2), step k of height
    steps[k]. The cell area of a node starts at the step times its xi spacing on the known line and
    is rescaled twice by the step over the length obtained, the new line's spacing being the one
    the implicit system sees."""
    ni = len(line)
    xs = np.empty((len(steps) + 1, ni))
    ys = np.empty_like(xs)
    xs[0], ys[0] = line[:, 0], line[:, 1]
    r = line.copy()
    for k, h in enumerate(steps):
        d = np.empty_like(r)
        d[1:-1] = 0.5 * (r[2:] - r[:-2])
        d[0], d[-1] = r[1] - r[0], r[-1] - r[-2]
        spacing = np.hypot(d[:, 0], d[:, 1])
        if np.any(spacing <= 0.0):
            msg = f"line {k}: coincident neighbours along xi; the march is not hyperbolic there"
            raise VVError(msg, code="V0312")
        eps = smoothing * min(1.0, k / 8.0)
        area = h * spacing
        for _ in range(2):
            dr = _step(r, area, eps)
            area = area * h / np.hypot(dr[:, 0], dr[:, 1])
        dr = _step(r, area, eps)
        r = r + dr
        xs[k + 1], ys[k + 1] = r[:, 0], r[:, 1]
    return xs, ys


def c_grid(
    upper: FloatArray,
    lower: FloatArray,
    airfoil_cells: int,
    wake_cells: int,
    normal_cells: int,
    first: float,
    outer: float,
    smoothing: float = 0.3,
    relax: float = 0.5,
) -> CGrid:
    """The C-grid of an airfoil of unit chord along x (leading edge first in `upper` and `lower`,
    both ending at the trailing edge): `airfoil_cells` around the airfoil, `wake_cells` on each
    side of the cut from the trailing edge to `outer` chords downstream, `normal_cells` from the
    wall to `outer` chords away, the first of height `first`, growing geometrically. Raises
    VVError V0310-V0312 for an impossible spacing, open surfaces or a degenerate line."""
    if airfoil_cells % 2:
        msg = "an even number of cells around the airfoil is needed"
        raise VVError(msg, code="V0310")
    surface = _surface(upper, lower, airfoil_cells, relax)
    te = surface[0]
    te_step = float(np.hypot(*(surface[1] - surface[0])))
    wake_length = outer - (te[0] - min(upper[:, 0].min(), lower[:, 0].min()))
    ratio = _ratio(te_step, wake_length, wake_cells)
    offsets = te_step * (ratio ** np.arange(wake_cells) - 1.0) / (ratio - 1.0)
    offsets = np.r_[offsets, wake_length]
    wake = np.stack([te[0] + offsets[1:], np.full(wake_cells, te[1])], axis=1)
    line = np.vstack([wake[::-1], surface, wake])
    steps = first * _ratio(first, outer, normal_cells) ** np.arange(normal_cells, dtype=np.float64)
    x, y = _march(line, steps, smoothing)
    return CGrid(x, y, wake_cells)


def quality(grid: CGrid) -> dict[str, float]:
    """The declared criteria (issue #408): the smallest cell area (signed, positive when the cell
    is right-handed in i, j), the largest angle in degrees between the first j-line and the wall
    normal over the airfoil's nodes apart from the trailing edge's, and the largest ratio of
    consecutive steps along j over every i."""
    x, y = grid.x, grid.y
    ax = x[1:, 1:] - x[:-1, :-1]
    ay = y[1:, 1:] - y[:-1, :-1]
    bx = x[1:, :-1] - x[:-1, 1:]
    by = y[1:, :-1] - y[:-1, 1:]
    area = 0.5 * (ax * by - ay * bx)
    w = grid.wake_cells
    ni = x.shape[1]
    nodes = np.arange(w + 1, ni - w - 1)
    tx = x[0, nodes + 1] - x[0, nodes - 1]
    ty = y[0, nodes + 1] - y[0, nodes - 1]
    nx, ny = x[1, nodes] - x[0, nodes], y[1, nodes] - y[0, nodes]
    cos = (nx * tx + ny * ty) / (np.hypot(tx, ty) * np.hypot(nx, ny))
    angle = np.degrees(np.abs(np.arcsin(np.clip(cos, -1.0, 1.0))))
    step = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    return {
        "min_area": float(area.min()),
        "wall_angle_deg": float(angle.max()),
        "max_stretch": float((step[1:] / step[:-1]).max()),
    }


def write_plot3d(grid: CGrid, path: Path) -> None:
    """A formatted single-block 2D PLOT3D file (i fastest), as `physics/tmr/plot3d.py` reads."""
    nj, ni = grid.x.shape
    with path.open("w") as f:
        f.write(f"1\n{ni} {nj}\n")
        for a in (grid.x, grid.y):
            np.savetxt(f, a.reshape(-1, 1), fmt="%.17g")


def rae2822(tape: Path, corrupt: tuple[int, ...]) -> tuple[FloatArray, FloatArray]:
    """Upper and lower surfaces of the RAE 2822 from file 1121 of the 1981 tape (Bradshaw's
    f8621.txt): X/C, YUS/C, YLS/C as measured (the design columns are not used), the lower
    ordinates stored as magnitudes. The two surfaces' leading- and trailing-edge values differ by
    less than the tape's quantisation step and take their mean."""
    data = read_tape_1981(tape, corrupt)[1121]
    x, yu, yl = data[:, 0], data[:, 1], -data[:, 2]
    upper = np.stack([x, yu], axis=1)
    lower = np.stack([x, yl], axis=1)
    for k in (0, -1):
        mean = 0.5 * (upper[k] + lower[k])
        upper[k] = lower[k] = mean
    return upper, lower


LEVELS = {"coarse": (184, 32), "medium": (368, 64), "fine": (736, 128)}


def main(argv: list[str] | None = None) -> None:
    """Writes the three nested RAE 2822 levels (issue #408) and their criteria as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tape", type=Path, help="Bradshaw/d2/f8621.txt as fetched by autan-vv")
    parser.add_argument("out", type=Path)
    parser.add_argument("--corrupt", type=int, nargs="*", default=[1124])
    parser.add_argument("--first", type=float, default=1.0e-6)
    parser.add_argument("--outer", type=float, default=500.0)
    parser.add_argument("--smoothing", type=float, default=0.3)
    parser.add_argument("--relax", type=float, default=0.5)
    args = parser.parse_args(argv)
    upper, lower = rae2822(args.tape, tuple(args.corrupt))
    grid = c_grid(upper, lower, 480, 128, 128, args.first, args.outer, args.smoothing, args.relax)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ("fine", "medium", "coarse"):
        write_plot3d(grid, args.out / f"rae2822-{name}.p2dfmt")
        report[name] = {"cells": [grid.x.shape[1] - 1, grid.x.shape[0] - 1], **quality(grid)}
        grid = grid.coarsened()
    (args.out / "quality.json").write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
