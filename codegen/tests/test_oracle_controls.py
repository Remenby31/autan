"""Defect controls of ADR 0003 that corrupt the oracle's inputs, not the kernels': 4-8 (check 2),
9 and the topology half of 3 (check 1), 10 (check 4). Each control first proves its failure at the
output, |y_c - centre| > R + r_c, then runs the check, which must fail on the corrupted entry."""

import math
from fractions import Fraction as Fr

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t

CONTROL_CASE = ("tet", 3, 0.15, 3)  # kind, N, perturbation, seed: a B case
# Face moments vanish on planar faces (every tet face): control 7 needs warped quads.
WARPED_CASE = ("hex", 3, 0.15, 3)


def _mesh(case: tuple[str, int, float, int] = CONTROL_CASE) -> tuple[s.Mesh, t.ReadMesh]:
    kind, n, p, seed = case
    m = s.mesh(meshes.build(kind, n, p, seed=seed))
    return m, t.read_mesh(m)


def _beyond(x: float, centre: Fr, radius: Fr, bound: Fr) -> float:
    """x moved one binary64 ulp at a time, away from the centre, until it is more than `bound`
    from the original value; refuses a step that crosses zero or leaves the normal range."""
    y, up = x, Fr(x) >= centre
    while abs(Fr(y) - Fr(x)) <= bound:
        y = math.nextafter(y, math.inf if up else -math.inf)
        if y == 0 or abs(y) < 2.0**-1022 or not math.isfinite(y) or (y > 0) != (x > 0):
            msg = "the corrupted value leaves the model"
            raise AssertionError(msg)
    assert abs(Fr(y) - centre) > radius  # the proof at the output (r_c = 0: a stored value)
    return y


def _target(stored: dict, name: str) -> tuple[int, int]:
    """The first entry of `name` the control may corrupt: |v| >= the smallest normal (ADR 0003,
    control 4), so its steps stay finite, normal and on one side of zero."""
    for i, row in enumerate(stored[name]):
        for j, v in enumerate(row):
            if abs(v) >= 2.0**-1022:
                return i, j
    msg = f"no entry of {name} can be corrupted: the control is not demonstrated"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("control", "name"),
    [(4, "face_area"), (6, "cell_volume"), (7, "face_moment"), (8, "cell_centroid")],
)
def test_a_stored_value_moved_beyond_twice_its_radius_fails_check_2(
    control: int, name: str
) -> None:
    # Controls 4, 6, 7, 8: K ulps, K the smallest with |y' - y| > 2 r (K = 1 when r = 0).
    m, read = _mesh(WARPED_CASE if name == "face_moment" else CONTROL_CASE)
    stored = q.stored_geometry(m)
    row, comp = _target(stored, name)
    cert = q.geometry_certified(read)[name][row][comp]
    values = list(stored[name][row])
    bound = 2 * cert.radius if cert.radius > 0 else Fr(0)
    values[comp] = _beyond(values[comp], cert.value, cert.radius, bound)
    stored[name][row] = values
    issues = q.check_geometry(read, stored)
    assert any(i.startswith(f"{name}[{row}][{comp}]: the stored value") for i in issues), control


def test_a_common_scaling_of_areas_moments_and_volumes_fails_check_2() -> None:
    # Control 5: closure would stay zero, the gradient unchanged; the nodes say otherwise.
    m, read = _mesh()
    certs = q.geometry_certified(read)
    stored = q.stored_geometry(m)
    proven = []
    for name in ("face_area", "face_moment", "cell_volume"):
        stored[name] = [[2 * v for v in row] for row in stored[name]]
        for i, row in enumerate(stored[name]):
            for j, v in enumerate(row):
                if abs(Fr(v) - certs[name][i][j].value) > certs[name][i][j].radius:
                    proven.append(f"{name}[{i}][{j}]")
    assert proven  # the proof at the output: some scaled entry lies outside its interval
    issues = q.check_geometry(read, stored)
    assert any(i.split(":")[0] in proven for i in issues)


def test_an_interior_face_with_owner_and_neighbour_swapped_fails_check_1() -> None:
    # Control 9: the area is not negated, so the face points from neighbour to owner.
    _, read = _mesh()
    kind, n, p, seed = CONTROL_CASE
    derived = t.derive(*meshes.cells(kind, n, p, seed=seed))  # type: ignore[attr-defined]
    corrupt = read.copy()
    corrupt.owner, corrupt.neighbour = read.owner.copy(), read.neighbour.copy()
    corrupt.owner[0], corrupt.neighbour[0] = read.neighbour[0], read.owner[0]
    assert t.compare(derived, read) == []
    assert t.compare(derived, corrupt)


def test_an_incidence_dropped_from_a_cell_fails_check_1() -> None:
    # Control 3, its check 1 half; the check 3 half needs the kernel to run on the dropped
    # incidence (see the PR: a test-only spike entry point, not in this file).
    _, read = _mesh()
    kind, n, p, seed = CONTROL_CASE
    derived = t.derive(*meshes.cells(kind, n, p, seed=seed))  # type: ignore[attr-defined]
    corrupt = read.copy()
    o = read.cell_face_offsets
    corrupt.cell_faces = np.delete(read.cell_faces, int(o[0]))
    corrupt.incidence_side = np.delete(read.incidence_side, int(o[0]))
    corrupt.cell_face_offsets = np.concatenate(([o[0]], o[1:] - 1))
    assert t.compare(derived, corrupt)


def test_the_kernel_output_offset_beyond_the_bound_fails_check_4() -> None:
    # Control 10: the next value of the gradient's format past the far side of G +- R.
    m, read = _mesh()
    geometry, exact = q.construct(read), q.geometry_exact(read)
    a, g = 1.0, (0.3, -1.2, 0.7)
    qv, qb = kr.field_floats(geometry, len(read.neighbour), a, g)
    out = s.gradients(m, "fp32", "lsq", np.array(qv), np.array(qb))[0].tolist()
    clean = kr.check_bound(read, geometry, exact, a, g, out, "fp32", "lsq")
    assert clean.issues == []
    bound = clean.bounds[(0, 0)]
    hi = Fr(g[0]) + bound
    y = np.float32(float(hi))
    while Fr(float(y)) <= hi:
        y = np.nextafter(y, np.float32(np.inf))
    assert abs(Fr(float(y)) - Fr(g[0])) > bound  # the proof at the output
    out[0][0] = float(y)
    bad = kr.check_bound(read, geometry, exact, a, g, out, "fp32", "lsq")
    assert any(i.startswith("fail: lsq cell 0 component 0") for i in bad.issues)
