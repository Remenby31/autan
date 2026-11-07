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

The same command writes the same bytes on every IEEE CPU (#536). The elementary functions whose
last bit belongs to the C library (the map's complex powers, angles and moduli, exp, cos, sin, atan)
are evaluated in mpmath at 120 bits, software arithmetic on integers, and each result is rounded
once to double; the sum giving R1 is math.fsum. Everything else is +, -, *, / and sqrt, which IEEE
rounds exactly, one NumPy operation at a time.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import mpmath
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
_MP = mpmath.MPContext()
_MP.prec = 120


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
        tau = 2 * _MP.atan(abs(_slope(self.x_te)))  # eq. 6
        self.power = float(2 - tau / _MP.pi)
        self.z1 = complex(self.x_te, 0.0)
        self.z2 = complex(0.5 * 1.1019 * THICKNESS**2, 0.0)  # eqs. 7 and 10

    def circle_point(self, z: complex) -> Any:
        """The mapped point of z, unrounded (an mpmath complex at 120 bits)."""
        w = _MP.power((_MP.mpc(z) - self.z1) / (_MP.mpc(z) - self.z2), 1 / _MP.mpf(self.power))
        return 1 / (1 - w)

    def to_circle(self, z: ComplexArray) -> ComplexArray:
        z = np.asarray(z, dtype=np.complex128)
        out = [complex(self.circle_point(v)) for v in z.ravel().tolist()]
        return np.array(out, dtype=np.complex128).reshape(z.shape)

    def to_airfoil(self, zeta: ComplexArray) -> ComplexArray:
        def one(v: complex) -> complex:
            w = _MP.mpc(v)
            p = _MP.power((w - 1) / w, self.power)
            return complex((self.z1 - self.z2 * p) / (1 - p))

        zeta = np.asarray(zeta, dtype=np.complex128)
        out = [one(v) for v in zeta.ravel().tolist()]
        return np.array(out, dtype=np.complex128).reshape(zeta.shape)

    def about(self, z: ComplexArray, centre: float) -> tuple[FloatArray, FloatArray]:
        """Angle and distance about `centre` of the mapped points of z, each rounded once."""
        mapped = [self.circle_point(v) - centre for v in z.ravel().tolist()]
        angle = np.array([float(_MP.arg(w)) for w in mapped]).reshape(z.shape)
        distance = np.array([float(abs(w)) for w in mapped]).reshape(z.shape)
        return angle, distance


def _upper_surface(kt: KarmanTrefftz, u: FloatArray) -> ComplexArray:
    """Upper surface point at parameter u in [0, 1]: x = x_TE u^2, smooth through the leading edge."""
    x = kt.x_te * u * u
    z = np.empty(x.shape, dtype=np.complex128)
    z.real, z.imag = x, half_thickness(x)
    return z


def _surface_nodes(kt: KarmanTrefftz, centre: float, theta: FloatArray) -> FloatArray:
    """Parameters u of the upper-surface points whose mapped angle about `centre` is theta in [0, pi].

    The mapped angle decreases monotonically from 0 at the trailing edge (u = 1) to pi at the
    leading edge (u = 0); bisection to round-off puts every node on the analytic airfoil.
    """
    lo = np.zeros_like(theta)
    hi = np.ones_like(theta)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        angle, _ = kt.about(_upper_surface(kt, mid), centre)
        # angle(mid) > theta means mid is too close to the leading edge.
        above = angle > theta
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    u = 0.5 * (lo + hi)
    u[theta == 0.0] = 1.0
    u[theta == math.pi] = 0.0
    return u


def _quasi_circle(kt: KarmanTrefftz, centre: float, n: int) -> tuple[FloatArray, ComplexArray]:
    """Angles theta and surface nodes z of the n surface points, about `centre`."""
    theta = 2.0 * math.pi * np.arange(n // 2 + 1) / n
    u = _surface_nodes(kt, centre, theta)
    upper = _upper_surface(kt, u)
    z = np.concatenate([upper, np.conj(upper[-2:0:-1])])
    return np.concatenate([theta, 2.0 * math.pi - theta[-2:0:-1]]), z


@functools.cache
def _construction() -> tuple[KarmanTrefftz, float, float]:
    """The map, the centre of eq. 12 and R1 of eq. 13, common to every level."""
    kt = KarmanTrefftz()
    zeta_le = complex(kt.circle_point(0j))
    centre = 0.5 * (zeta_le.real + 1.0)  # eq. 12, zeta_TE = zeta1 = 1
    _, z_finest = _quasi_circle(kt, centre, FINEST)
    zeta_finest = kt.to_circle(z_finest)
    step = (np.roll(zeta_finest, -1) - zeta_finest).tolist()
    r1 = math.fsum(float(abs(_MP.mpc(d))) for d in step) / (2.0 * math.pi)
    return kt, centre, r1


def plane_grid(nc: int) -> tuple[ComplexArray, KarmanTrefftz]:
    """Nodes z[i, j] of the NC x NC O-mesh, i = 0..NC-1 around from the trailing edge through the
    upper surface, j = 0..NC from the airfoil to the far field."""
    if nc < 4 or nc % 4 or FINEST % nc:
        msg = f"NC must be a multiple of 4 dividing {FINEST}, got {nc}"
        raise VVError(msg, code="V0309")
    kt, centre, r1 = _construction()
    theta, z_surface = _quasi_circle(kt, centre, nc)
    _, r_surface = kt.about(z_surface, centre)
    growth = 2.0 * math.pi * np.arange(nc + 1) / nc
    radii = r1 * np.array([float(_MP.exp(g)) for g in growth.tolist()])  # eq. 13
    r_far = radii[-1]
    # eq. 14, with r_i1 (R_far - R_j) where the paper prints r_i1 (R_far - R_1): as printed, the
    # far field would be r_i1 + R_far, not a circle, and a perfect circle would not give back the
    # radii of eq. 13. This form blends the near circle (j = 0) into the circle R_far (j = NC).
    r = (r_surface[:, None] * (r_far - radii[None, :]) + r_far * (radii[None, :] - r1)) / (
        r_far - r1
    )
    # Rows 0..NC/2 are mapped; the others mirror them (eqs. 15-16, below).
    half = nc // 2
    cos = np.array([float(_MP.cos(t)) for t in theta[: half + 1].tolist()])
    sin = np.array([float(_MP.sin(t)) for t in theta[: half + 1].tolist()])
    zeta = np.empty((half + 1, nc + 1), dtype=np.complex128)
    zeta.real = centre + r[: half + 1] * cos[:, None]
    zeta.imag = r[: half + 1] * sin[:, None]
    z = np.empty((nc, nc + 1), dtype=np.complex128)
    z[: half + 1] = kt.to_airfoil(zeta)
    z[:, 0] = z_surface
    # Exact symmetry about y = 0 (eqs. 15-16): row NC - i mirrors row i.
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


def write_msh(
    points: FloatArray, hexes: IntArray, boundary: dict[str, IntArray], path: Path
) -> None:
    """Write a level of this family as MSH 4.1 ASCII, without Gmsh (no linux-aarch64 build).

    For the slabs of `grid` the file is byte for byte the one Gmsh 4.15 writes for the same
    discrete model (one volume entity holding every node, one surface entity per zone, physical
    groups named after the zones, then `fluid`); the tests check it. It is not a general writer:
    Gmsh's bounding-box arithmetic, empty zones and long names are not reproduced, so other zones
    are refused (V0310).
    """
    if list(boundary) != list(ZONES) or any(f.shape[0] == 0 for f in boundary.values()):
        msg = f"write_msh writes NACA 0012 slabs only: zones {ZONES}, none empty"
        raise VVError(msg, code="V0310")

    def g(v: float) -> str:
        return f"{v:.16g}"

    def bbox(ids: IntArray) -> str:
        p = points[np.unique(ids)]
        return " ".join(g(float(v)) for v in (*p.min(axis=0), *p.max(axis=0)))

    names = list(boundary)
    quads = [boundary[name][:, :4] for name in names]
    n_nodes, n_hex = points.shape[0], hexes.shape[0]
    total = n_hex + sum(q.shape[0] for q in quads)
    with path.open("w") as f:
        f.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$PhysicalNames\n")
        f.write(f"{len(names) + 1}\n")
        f.writelines(f'2 {k + 1} "{name}"\n' for k, name in enumerate(names))
        f.write(f'3 {len(names) + 1} "{families.CELL_ZONE}"\n$EndPhysicalNames\n$Entities\n')
        f.write(f"0 0 {len(names)} 1\n")
        f.writelines(f"{k + 1} {bbox(q)} 1 {k + 1} 0 \n" for k, q in enumerate(quads))
        f.write(f"1 {bbox(hexes)} 1 {len(names) + 1} 0 \n$EndEntities\n$Nodes\n")
        f.write(f"{len(names) + 1} {n_nodes} 1 {n_nodes}\n")
        f.writelines(f"2 {k + 1} 0 0\n" for k in range(len(names)))
        f.write(f"3 1 0 {n_nodes}\n")
        f.writelines(f"{t}\n" for t in range(1, n_nodes + 1))
        f.writelines(f"{g(x)} {g(y)} {g(z)}\n" for x, y, z in points.tolist())
        f.write(f"$EndNodes\n$Elements\n{len(names) + 1} {total} 1 {total}\n")
        tag = n_hex + 1
        for k, q in enumerate(quads):
            f.write(f"2 {k + 1} 3 {q.shape[0]}\n")
            f.writelines(
                f"{tag + i} {a + 1} {b + 1} {c + 1} {d + 1} \n"
                for i, (a, b, c, d) in enumerate(q.tolist())
            )
            tag += q.shape[0]
        f.write(f"3 1 5 {n_hex}\n")
        f.writelines(
            f"{i + 1} " + " ".join(str(v + 1) for v in row) + " \n"
            for i, row in enumerate(hexes.tolist())
        )
        f.write("$EndElements\n")


def generate_level(nc: int, path: Path | None = None) -> FaceMesh:
    """Level NC as a FaceMesh, optionally written to `path` as MSH 4.1 ASCII (`write_msh`).

    Coordinates are rounded to the 16 significant digits the file holds, so file and memory agree.
    """
    points, cells, boundary = grid(nc)
    points = np.array([float(f"{v:.16g}") for v in points.ravel()]).reshape(-1, 3)
    mesh = families._to_facemesh(points, cells, boundary)
    mesh.check()
    if path is not None:
        write_msh(points, cells["hex"], boundary, path)
    return mesh


def generate_family(
    levels: tuple[int, ...], out_dir: Path
) -> tuple[families.Family, list[FaceMesh]]:
    """Write levels `naca0012-nNNNN.msh` and `manifest.json` into `out_dir`.

    The manifest's h is the in-plane size sqrt(area / cells), the measure of a 2D refinement study.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records, meshes = [], []
    for nc in levels:
        path = out_dir / f"naca0012-n{nc:04d}.msh"
        mesh = generate_level(nc, path)
        record = families.level_record(nc, path, mesh)
        records.append(replace(record, h=math.sqrt(record.volume / SPAN / mesh.n_cells)))
        meshes.append(mesh)
    # Written without Gmsh (`write_msh`).
    family = families.Family("naca0012", 0.0, 0, "none", tuple(records))
    families.write_manifest(family, out_dir)
    return family, meshes
