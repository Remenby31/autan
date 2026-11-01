"""O-meshes of the sharp-trailing-edge NACA 0012 of Vassberg and Jameson (2010), as one-layer slabs.

J. C. Vassberg and A. Jameson, In pursuit of grid convergence for two-dimensional Euler solutions,
J. Aircraft 47(4):1152-1166, 2010, doi:10.2514/1.46737 (equation numbers below are theirs). The
airfoil is the NACA 0012 equation extended in chord to its root, so that the trailing edge is sharp
(eqs. 1, 4). The Karman-Trefftz transformation (eq. 9) with singular points at the trailing edge
and half the leading-edge radius inside the leading edge (eq. 10) maps it to a near circle. Surface
nodes lie on the analytic airfoil at constant angular spacing about the midpoint of the mapped chord
(eq. 12); field nodes lie on circles of radii growing as exp(2 pi j / NC) (eq. 13), blended from the
near circle to a circle (eq. 14). Cells have aspect ratio one and the far field lies about 150 chords
away. The grid is exactly symmetric about y = 0 (eqs. 15-16): the lower half mirrors the upper.

Level NC has NC x NC cells. R1, the radius of eq. 13, is computed from the finest grid of the
paper's family (NC = 4096), so that every level is the finest one with every other line removed,
as the paper builds its family. The slab is one hexahedron thick in z.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError
from autan_vv.meshes import families

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.meshes.facemesh import FaceMesh, FloatArray, IntArray

THICKNESS = 0.12
FINEST = 4096
SPAN = 1.0
ZONES = ("airfoil", "farfield", "back", "front")
ComplexArray = npt.NDArray[np.complex128]
_A = (0.2969, -0.1260, -0.3516, 0.2843, -0.1015)  # eq. 1, Abbott and von Doenhoff


def half_thickness(x: FloatArray) -> FloatArray:
    """Upper surface y(x) of eq. 1."""
    s = np.sqrt(x)
    a0, a1, a2, a3, a4 = _A
    return THICKNESS / 0.2 * (a0 * s + x * (a1 + x * (a2 + x * (a3 + x * a4))))


def _slope(x: float) -> float:
    a0, a1, a2, a3, a4 = _A
    return (
        THICKNESS / 0.2 * (0.5 * a0 / math.sqrt(x) + a1 + x * (2 * a2 + x * (3 * a3 + x * 4 * a4)))
    )


def trailing_edge() -> float:
    """Root of eq. 1 next to x = 1 (eq. 4), by Newton's method."""
    x = 1.0
    for _ in range(50):
        step = float(half_thickness(np.array(x))) / _slope(x)
        x -= step
        if abs(step) < 1e-16:
            break
    return x


class KarmanTrefftz:
    """Eq. 9: ((z - z1)/(z - z2))^(1/P) = (zeta - zeta1)/(zeta - zeta2), with zeta1 = 1, zeta2 = 0.

    Any real zeta1, zeta2 give the same mesh: they change the mapped plane by a similarity only, and
    the construction (angles about the chord midpoint, radii relative to R1) is invariant under it.
    """

    def __init__(self) -> None:
        self.x_te = trailing_edge()
        tau = 2.0 * math.atan(abs(_slope(self.x_te)))  # eq. 6
        self.power = 2.0 - tau / math.pi
        self.z1 = complex(self.x_te, 0.0)
        self.z2 = complex(0.5 * 1.1019 * THICKNESS**2, 0.0)  # eqs. 7 and 10

    def to_circle(self, z: ComplexArray) -> ComplexArray:
        w = np.power((z - self.z1) / (z - self.z2), 1.0 / self.power)
        return np.asarray(1.0 / (1.0 - w), dtype=np.complex128)

    def to_airfoil(self, zeta: ComplexArray) -> ComplexArray:
        v = np.power((zeta - 1.0) / zeta, self.power)
        return (self.z1 - self.z2 * v) / (1.0 - v)


def _upper_surface(kt: KarmanTrefftz, u: FloatArray) -> ComplexArray:
    """Upper surface point at parameter u in [0, 1]: x = x_TE u^2, smooth through the leading edge."""
    x = kt.x_te * u * u
    return x + 1j * half_thickness(x)


def _surface_nodes(kt: KarmanTrefftz, centre: float, theta: FloatArray) -> FloatArray:
    """Parameters u of the upper-surface points whose mapped angle about `centre` is theta in [0, pi].

    The mapped angle decreases monotonically from 0 at the trailing edge (u = 1) to pi at the
    leading edge (u = 0); bisection to round-off puts every node on the analytic airfoil.
    """
    lo = np.zeros_like(theta)
    hi = np.ones_like(theta)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        angle = np.angle(kt.to_circle(_upper_surface(kt, mid)) - centre)
        # angle(mid) > theta means mid is too close to the leading edge.
        above = angle > theta
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    u = 0.5 * (lo + hi)
    u[theta == 0.0] = 1.0
    u[theta == math.pi] = 0.0
    return u


def plane_grid(nc: int) -> tuple[ComplexArray, KarmanTrefftz]:
    """Nodes z[i, j] of the NC x NC O-mesh, i = 0..NC-1 around from the trailing edge through the
    upper surface, j = 0..NC from the airfoil to the far field."""
    if nc < 4 or nc % 4 or FINEST % nc:
        msg = f"NC must be a multiple of 4 dividing {FINEST}, got {nc}"
        raise VVError(msg, code="V0309")
    kt = KarmanTrefftz()
    zeta_le = complex(kt.to_circle(np.array(0j)))
    centre = 0.5 * (zeta_le.real + 1.0)  # eq. 12, zeta_TE = zeta1 = 1

    def quasi_circle(n: int) -> tuple[FloatArray, ComplexArray]:
        theta = 2.0 * math.pi * np.arange(n // 2 + 1) / n
        u = _surface_nodes(kt, centre, theta)
        upper = _upper_surface(kt, u)
        z = np.concatenate([upper, np.conj(upper[-2:0:-1])])
        return np.concatenate([theta, 2.0 * math.pi - theta[-2:0:-1]]), z

    _, z_finest = quasi_circle(FINEST)
    zeta_finest = kt.to_circle(z_finest)
    r1 = float(np.abs(np.roll(zeta_finest, -1) - zeta_finest).sum()) / (2.0 * math.pi)
    theta, z_surface = quasi_circle(nc)
    r_surface = np.abs(kt.to_circle(z_surface) - centre)
    radii = r1 * np.exp(2.0 * math.pi * np.arange(nc + 1) / nc)  # eq. 13
    r_far = radii[-1]
    # eq. 14, with r_i1 (R_far - R_j) where the paper prints r_i1 (R_far - R_1): as printed, the
    # far field would be r_i1 + R_far, not a circle, and a perfect circle would not give back the
    # radii of eq. 13. This form blends the near circle (j = 0) into the circle R_far (j = NC).
    r = (r_surface[:, None] * (r_far - radii[None, :]) + r_far * (radii[None, :] - r1)) / (
        r_far - r1
    )
    zeta = centre + r * np.exp(1j * theta)[:, None]
    z = kt.to_airfoil(zeta)
    z[:, 0] = z_surface
    # Exact symmetry about y = 0 (eqs. 15-16): row NC - i mirrors row i.
    half = nc // 2
    z[half + 1 :] = np.conj(z[1:half][::-1])
    z[0].imag = 0.0
    z[half].imag = 0.0
    return z, kt


def grid(nc: int) -> tuple[FloatArray, dict[str, IntArray], dict[str, IntArray]]:
    """Points, hexahedra and boundary faces (zones of ZONES) of the one-layer slab of level NC.

    The airfoil lies in the x-z plane (chord along +x, lift along +z, as case-schema.md orients the
    freestream) and the layer spans y in [0, SPAN]: `back` is y = 0 and `front` y = SPAN.
    """
    z, _ = plane_grid(nc)
    ni, nj = nc, nc + 1
    layer = ni * nj
    x, h = z.real.ravel(), z.imag.ravel()
    pts = np.concatenate([np.column_stack([x, np.full(layer, y), h]) for y in (0.0, SPAN)])

    def node(i: IntArray, j: IntArray) -> IntArray:
        return (i % ni) * nj + j

    i, j = (a.ravel() for a in np.meshgrid(np.arange(ni), np.arange(nc), indexing="ij"))
    # i runs counterclockwise in (x, z) and j outward; e_x x e_z = -e_y, so (i, j), (i+1, j),
    # (i+1, j+1), (i, j+1) turns positively about +y, the extrusion direction of a Gmsh hexahedron.
    bottom = np.stack([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)], axis=1)
    p = pts[bottom]
    turn = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])[:, 1]
    turn_opposite = np.cross(p[:, 3] - p[:, 2], p[:, 1] - p[:, 2])[:, 1]
    if np.any(turn <= 0.0) or np.any(turn_opposite <= 0.0):
        msg = "an O-mesh cell is inverted"
        raise VVError(msg, code="V0301")
    hexes = np.concatenate([bottom, bottom + layer], axis=1)
    ring = np.arange(ni)

    def side(jj: int) -> IntArray:
        a, b = node(ring, np.full(ni, jj)), node(ring + 1, np.full(ni, jj))
        return np.stack([a, b, b + layer, a + layer], axis=1)

    boundary = {"airfoil": side(0), "farfield": side(nc), "back": bottom, "front": bottom + layer}
    return pts, {"hex": hexes}, boundary


def generate_level(nc: int, path: Path | None = None) -> tuple[FaceMesh, str]:
    """Level NC as a FaceMesh, optionally written to `path` as MSH 4.1 ASCII; returns the Gmsh version.

    Coordinates are rounded to the 16 significant digits Gmsh writes, so file and memory agree.
    """
    points, cells, boundary = grid(nc)
    points = np.array([float(f"{v:.16g}") for v in points.ravel()]).reshape(-1, 3)
    mesh = families._to_facemesh(points, cells, boundary)
    mesh.check()
    gmsh = families._gmsh()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        version = str(gmsh.__version__)
        if path is not None:
            families._write_msh(gmsh, points, cells, boundary, path)
    finally:
        gmsh.finalize()
    return mesh, version


def generate_family(
    levels: tuple[int, ...], out_dir: Path
) -> tuple[families.Family, list[FaceMesh]]:
    """Write levels `naca0012-nNNNN.msh` and `manifest.json` into `out_dir`.

    The manifest's h is the in-plane size sqrt(area / cells), the measure of a 2D refinement study.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records, meshes, version = [], [], ""
    for nc in levels:
        path = out_dir / f"naca0012-n{nc:04d}.msh"
        mesh, version = generate_level(nc, path)
        record = families.level_record(nc, path, mesh)
        records.append(replace(record, h=math.sqrt(record.volume / SPAN / mesh.n_cells)))
        meshes.append(mesh)
    family = families.Family("naca0012", 0.0, 0, version, tuple(records))
    families.write_manifest(family, out_dir)
    return family, meshes
