"""RAE 2822 C-grids generated offline by Construct2D 2.1.4, measured against the criteria of #408.

Construct2D (D. Prosser, https://sourceforge.net/projects/construct2d/, GPL-3.0) is a V&V dev tool
here, like gmsh: never vendored, linked or shipped; vv/construct2d/Dockerfile builds the pinned
release. This module writes its inputs (the RAE 2822 surface of file 1121 of the 1981 tape and the
namelist selected on #449), runs that binary, reads its 2D PLOT3D grid, derives the nested medium
and coarse levels by removing every other line, measures each level and checks it against the
grid's pinned hash.

i runs along the wall line from the end of the wake below the cut, along the wake to the trailing
edge, around the airfoil (lower surface, leading edge, upper surface) and back along the wake above
the cut; the two wake halves coincide node for node, which `physics/tmr/plot3d.py` merges into
interior faces.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from autan_vv.cases import fetch, get_case, read_tape_1981
from autan_vv.errors import VVError

FloatArray = npt.NDArray[np.float64]

VERSION = "2.1.4"

NSRF, NWKE, JMAX = 481, 128, 129
RADIUS = 500.0
FIRST = 1.0e-6
REYNOLDS = 6.5e6
CFRAC = 0.5
NAMELIST: dict[str, dict[str, object]] = {
    "SOPT": {
        "nsrf": NSRF,
        "lesp": 5.0e-4,
        "tesp": 1.0e-3,
        "radi": RADIUS,
        "nwke": NWKE,
        "fdst": 1.0,
        "fwkl": 1.0,
        "fwki": 700.0,
    },
    "VOPT": {
        "name": "rae2822",
        "jmax": JMAX,
        "slvr": "ELLP",
        "topo": "CGRD",
        "recd": REYNOLDS,
        "cfrc": CFRAC,
        "stp1": 5000,
        "stp2": 20,
        "nrmt": 1,
        "nrmb": 1,
        "alfa": 1.0,
        "epsi": 15.0,
        "epse": 0.0,
        "funi": 0.2,
        "asmt": 20,
    },
    "OOPT": {"gdim": 2, "npln": 2, "dpln": 1.0},
}
GRIDS = {
    "fine": "b64a0447c067e0f22cf4b85e96d5b617ceff4f7d55fb24d53bfd2dab4adb1547",
    "medium": "9f2012bf7e16dc7a44b08dd4f19d8d8f6defcc9ecc47f51a16269eae2206b13d",
    "coarse": "06d946b9f79ba3aaa693e377e35cd941e3c4a631fd0e1cfda1bd7796b412da4e",
}
STRETCH = {"fine": 1.25, "medium": 1.56, "coarse": 2.44}
OUTER = (0.95 * RADIUS, 1.5 * RADIUS)


@dataclass(frozen=True)
class CGrid:
    """Node coordinates (nj, ni) and the number of wake cells on each side of the cut."""

    x: FloatArray
    y: FloatArray
    wake_cells: int

    def coarsened(self) -> CGrid:
        """Every other line in i and j: the next coarser nested level."""
        return CGrid(self.x[::2, ::2], self.y[::2, ::2], self.wake_cells // 2)


def tape(cache: Path) -> Path:
    """The 1981 tape (Bradshaw's f8621.txt) from the catalog's pinned TMR copy, CC0-1.0, verified by
    its SHA-256; the catalog's other `rae2822` files, from NPARC, are not fetched (#408)."""
    case = get_case("rae2822")
    only = dataclasses.replace(case, files=(case.file("f8621.txt"),))
    return fetch(only, cache, roles=(only.files[0].role,))[0]


def rae2822(tape: Path, corrupt: tuple[int, ...] = (1124,)) -> tuple[FloatArray, FloatArray]:
    """Upper and lower surfaces of the RAE 2822 from file 1121 of the 1981 tape (Bradshaw's
    f8621.txt): X/C, YUS/C, YLS/C as measured (the design columns are not used), the lower
    ordinates stored as magnitudes. The two surfaces' first and last points take their mean, so
    the trailing edge is closed and sharp."""
    data = read_tape_1981(tape, corrupt)[1121]
    x, yu, yl = data[:, 0], data[:, 1], -data[:, 2]
    upper = np.stack([x, yu], axis=1)
    lower = np.stack([x, yl], axis=1)
    for k in (0, -1):
        mean = 0.5 * (upper[k] + lower[k])
        upper[k] = lower[k] = mean
    return upper, lower


def yplus_for(first: float) -> float:
    """The `ypls` for which Construct2D's wall_distance (surface_util.f90) gives `first`:
    y0 = ypls cfrc / (Re sqrt(Cf / 2)), Cf = (2 log10 Re - 0.65)^-2.3."""
    cf = (2.0 * math.log10(REYNOLDS) - 0.65) ** -2.3
    return first * REYNOLDS * math.sqrt(0.5 * cf) / CFRAC


def _value(v: object) -> str:
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, float):
        return f"{v:.17g}"
    return str(v)


def write_inputs(upper: FloatArray, lower: FloatArray, out: Path) -> None:
    """`rae2822.dat` (a header line, then upper trailing edge to leading edge to lower trailing
    edge, as Construct2D reads it) and `grid_options.in`, the selected namelist."""
    points = np.vstack([upper[::-1], lower[1:]])
    lines = ["RAE 2822, file 1121"] + [f"{x:.17g} {y:.17g}" for x, y in points]
    (out / "rae2822.dat").write_text("\n".join(lines) + "\n")
    groups = {k: dict(v) for k, v in NAMELIST.items()}
    groups["VOPT"]["ypls"] = yplus_for(FIRST)
    text = "".join(
        f"&{group}\n" + "".join(f"  {k} = {_value(v)}\n" for k, v in keys.items()) + "/\n"
        for group, keys in groups.items()
    )
    (out / "grid_options.in").write_text(text)


def _areas(x: FloatArray, y: FloatArray) -> FloatArray:
    ax, ay = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    bx, by = x[1:, :-1] - x[:-1, 1:], y[1:, :-1] - y[:-1, 1:]
    return 0.5 * (ax * by - ay * bx)


def read_grid(path: Path) -> CGrid:
    """Construct2D's 2D PLOT3D file (`imax jmax`, then x and y with i fastest), oriented so that
    cells are right-handed in i, j. VVError V0311 if it is not such a file."""
    values = path.read_text().split()
    try:
        ni, nj = int(values[0]), int(values[1])
        data = np.array(values[2:], dtype=np.float64)
    except (IndexError, ValueError) as exc:
        msg = f"{path} is not a 2D PLOT3D grid of Construct2D"
        raise VVError(msg, code="V0311") from exc
    if ni < 3 or nj < 2 or data.size != 2 * ni * nj or not np.isfinite(data).all():
        msg = f"{path} holds {data.size} values for a {ni} x {nj} grid, or non-finite ones"
        raise VVError(msg, code="V0311")
    x = data[: ni * nj].reshape(nj, ni)
    y = data[ni * nj :].reshape(nj, ni)
    if _areas(x, y).min() < 0.0:
        x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
    return CGrid(x, y, (ni - NSRF) // 2)


def write_plot3d(grid: CGrid, path: Path) -> None:
    """A formatted single-block 2D PLOT3D file (i fastest), as `physics/tmr/plot3d.py` reads."""
    nj, ni = grid.x.shape
    with path.open("w") as f:
        f.write(f"1\n{ni} {nj}\n")
        for a in (grid.x, grid.y):
            np.savetxt(f, a.reshape(-1, 1), fmt="%.17g")


def criteria(grid: CGrid) -> dict[str, float]:
    """The measures of #408 and #449 on one level: the smallest signed cell area; the largest angle
    between the first j-line and the wall normal over the airfoil's nodes, trailing edge excluded;
    the largest ratio of consecutive j-steps; N1, the shortest edge over the grid's extent; N2, the
    outer line's shortest over its mean xi-edge; N3, the largest eta/xi edge ratio of the outer
    cells; N4, the smallest angle between the xi- and eta-lines at the nodes of rows j >= 1 (central
    xi difference, one-sided eta difference); the first step on the airfoil; and the outer line's
    distances from mid-chord."""
    x, y = grid.x, grid.y
    ni = x.shape[1]
    w = grid.wake_cells
    nodes = np.arange(w + 1, ni - w - 1)
    tx, ty = x[0, nodes + 1] - x[0, nodes - 1], y[0, nodes + 1] - y[0, nodes - 1]
    nx, ny = x[1, nodes] - x[0, nodes], y[1, nodes] - y[0, nodes]
    wall = (nx * tx + ny * ty) / (np.hypot(tx, ty) * np.hypot(nx, ny))
    dxi = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    deta = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    outer_xi = dxi[-1]
    outer_eta = 0.5 * (deta[-1, :-1] + deta[-1, 1:])
    xx, xy = x[1:, 2:] - x[1:, :-2], y[1:, 2:] - y[1:, :-2]
    ex = np.vstack([x[2:, 1:-1] - x[1:-1, 1:-1], x[-1:, 1:-1] - x[-2:-1, 1:-1]])
    ey = np.vstack([y[2:, 1:-1] - y[1:-1, 1:-1], y[-1:, 1:-1] - y[-2:-1, 1:-1]])
    cross = np.abs(xx * ex + xy * ey) / (np.hypot(xx, xy) * np.hypot(ex, ey))
    first = deta[0, w : ni - w]
    mid = 0.5 * (x[0, w] + x[0, ni // 2])
    radius = np.hypot(x[-1] - mid, y[-1])
    return {
        "min_area": float(_areas(x, y).min()),
        "wall_angle_deg": float(np.degrees(np.abs(np.arcsin(np.clip(wall, -1.0, 1.0)))).max()),
        "max_stretch": float((deta[1:] / deta[:-1]).max()),
        "n1_min_edge_over_extent": float(min(dxi.min(), deta.min()) / max(np.ptp(x), np.ptp(y))),
        "n2_outer_min_over_mean": float(outer_xi.min() / outer_xi.mean()),
        "n3_outer_aspect": float((outer_eta / outer_xi).max()),
        "n4_min_angle_deg": float(np.degrees(np.arccos(np.clip(cross, 0.0, 1.0))).min()),
        "first_step_min": float(first.min()),
        "first_step_max": float(first.max()),
        "outer_radius_min": float(radius.min()),
        "outer_radius_max": float(radius.max()),
    }


def passes(level: str, m: dict[str, float]) -> dict[str, bool]:
    """Pass/fail of each criterion of #449 on one level; the wall angle and the first step on the
    fine level only, where #408 makes them blocking."""
    checks = {
        "N1": m["n1_min_edge_over_extent"] > 1e-12,
        "N2": m["n2_outer_min_over_mean"] >= 0.1,
        "N3": m["n3_outer_aspect"] <= 100.0,
        "N4": m["n4_min_angle_deg"] >= 55.0,
        "areas": m["min_area"] > 0.0,
        "stretch": m["max_stretch"] <= STRETCH[level],
        "outer": OUTER[0] <= m["outer_radius_min"] and m["outer_radius_max"] <= OUTER[1],
    }
    if level == "fine":
        checks["wall_angle"] = m["wall_angle_deg"] <= 2.0
        error = max(abs(m["first_step_min"] / FIRST - 1.0), abs(m["first_step_max"] / FIRST - 1.0))
        checks["first_step"] = error <= 0.01
    return checks


def generate(exe: Path, tape: Path, out: Path) -> dict[str, object]:
    """Runs Construct2D in `out`, writes `rae2822-{fine,medium,coarse}.p2dfmt` and returns each
    level's measures. VVError V0311 if Construct2D writes no grid, V0312 if a level's SHA-256
    differs from its pinned value (a build or input other than the pinned ones)."""
    out.mkdir(parents=True, exist_ok=True)
    stale = ["rae2822.p3d", "rae2822_stats.p3d", "rae2822.nmf", "quality.json"]
    for name in stale + [f"rae2822-{level}.p2dfmt" for level in GRIDS]:
        (out / name).unlink(missing_ok=True)
    write_inputs(*rae2822(tape), out)
    run = subprocess.run(
        [str(exe), "rae2822.dat"],
        cwd=out,
        input="GRID\nSMTH\nn\nQUIT\n",
        text=True,
        capture_output=True,
        timeout=14400,
        check=False,
    )
    (out / "construct2d.log").write_text(run.stdout + run.stderr)
    if run.returncode != 0 or not (out / "rae2822.p3d").exists():
        msg = f"Construct2D wrote no grid (exit {run.returncode}), see {out / 'construct2d.log'}"
        raise VVError(msg, code="V0311")
    grid = read_grid(out / "rae2822.p3d")
    report: dict[str, object] = {
        "construct2d": VERSION,
        "namelist": NAMELIST,
        "ypls": yplus_for(FIRST),
    }
    for level in ("fine", "medium", "coarse"):
        path = out / f"rae2822-{level}.p2dfmt"
        write_plot3d(grid, path)
        m = criteria(grid)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report[level] = {
            "cells": [grid.x.shape[1] - 1, grid.x.shape[0] - 1],
            **m,
            "pass": passes(level, m),
            "sha256": digest,
        }
        if digest != GRIDS[level]:
            msg = f"{path.name}: SHA-256 {digest}, pinned {GRIDS[level]}"
            raise VVError(msg, code="V0312")
        grid = grid.coarsened()
    return report


def main(argv: list[str] | None = None) -> None:
    """Writes the three nested RAE 2822 levels (issues #408, #449) and their measures as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exe", type=Path, help=f"construct2d built from Construct2D {VERSION}")
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/autan-vv").expanduser())
    args = parser.parse_args(argv)
    report = generate(args.exe.resolve(), tape(args.cache), args.out)
    (args.out / "quality.json").write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
