"""Positive fixtures of ADR 0003: they must PASS.

Exact fixtures: a fixture is certified exact when the propagation gives R = 0 for every cell and
component; the check is then g_hat = G exactly. The approved set was selected on development (B)
sizes by that criterion among the declared candidates (all candidates and their outcomes are in the
freeze manifest); it must be non-empty, and every fixture in it must certify.

Boundary cases: a cell with only boundary faces, cells with a single boundary face, a zero gradient
component and the constant field must pass checks 1-4.
"""

from fractions import Fraction as Fr

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t

POLICIES = ("fp64", "fp32+acc64", "fp32")
METHODS = ("lsq", "gg")
# Every candidate tried for the exact set, on development (B) sizes: name -> (kind, N, perturbation,
# seed, a, G). The freeze manifest records each one's outcome; the approved set below must equal the
# candidates that certified there (make_manifest checks it).
CANDIDATES = {
    "uniform-hex-linear-N4": ("hex", 4, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),
    "uniform-hex-linear-N8": ("hex", 8, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),
    "uniform-hex-constant-N4": ("hex", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "uniform-hex-constant-N8": ("hex", 8, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "uniform-hex-zero-component-N4": ("hex", 4, 0.0, 0, 1.0, (0.5, 0.0, 1.0)),
    "tet-constant-N4": ("tet", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "hybrid-constant-N4": ("hybrid", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "poly-constant-N4": ("poly", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "hex-perturbed-constant-N4": ("hex", 4, 0.15, 3, 1.0, (0.0, 0.0, 0.0)),
    "single-cell-hex-linear": ("hex", 1, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),
    "hex-N3-linear": ("hex", 3, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),
}
# name: (kind, N, perturbation, seed, a, G)
APPROVED_EXACT = {
    "uniform-hex-constant-N4": ("hex", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "uniform-hex-constant-N8": ("hex", 8, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
    "poly-constant-N4": ("poly", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
}
BOUNDARY_CASES = {
    "single-cell-hex-linear": ("hex", 1, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),  # only boundary faces
    "hex-N3-linear": ("hex", 3, 0.0, 0, 1.0, (0.5, -0.25, 1.0)),  # cells with one boundary face
    "uniform-hex-zero-component-N4": ("hex", 4, 0.0, 0, 1.0, (0.5, 0.0, 1.0)),
    "uniform-hex-constant-N4": ("hex", 4, 0.0, 0, 1.0, (0.0, 0.0, 0.0)),
}


def _run(
    case: tuple, policy: str, method: str
) -> tuple[s.Mesh, t.ReadMesh, list, kr.D1, kr.Bound, np.ndarray]:
    kind, n, p, seed, a, g = case
    m = s.mesh(meshes.build(kind, n, p, seed=seed))
    read = t.read_mesh(m)
    pts, cells, ag = meshes.cells(kind, n, p, seed=seed)
    issues = t.compare(t.derive(pts, cells, ag), read)
    geometry, exact = q.construct(read), q.geometry_exact(read)
    certs = {k: [[pair.c for pair in row] for row in rows] for k, rows in geometry.items()}
    issues += q.check_geometry(read, q.stored_geometry(m), certs, exact)
    n_int = len(read.neighbour)
    qv, qb = kr.field_floats(geometry, n_int, a, g)
    if method == "lsq":
        out = s.gradients(m, policy, "lsq", np.array(qv), np.array(qb))[0]
    else:
        out = s.gradients(
            m, policy, "gg", np.array(qv), np.array(qb), 1, np.tile(np.array(g), (m.n_cells, 1))
        )[0]
    q_pairs = kr.linear_field(geometry["cell_centroid"], a, g)
    qb_pairs = [kr.given(0.0, kr.B64)] * n_int + kr.linear_field(
        geometry["face_centroid"][n_int:], a, g
    )
    grad0 = [[kr.given(v, kr.B64) for v in g]] * m.n_cells if method == "gg" else None
    d1 = kr.check_d1(
        read, kr.Inputs(geometry, q_pairs, qb_pairs, grad0), exact, out.tolist(), policy, method
    )
    b4 = kr.check_bound(read, geometry, exact, a, g, out.tolist(), policy, method)
    return m, read, issues, d1, b4, out


def test_the_approved_exact_set_is_not_empty() -> None:
    assert APPROVED_EXACT


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", sorted(APPROVED_EXACT))
def test_an_approved_exact_fixture_certifies_r_zero_and_gives_g_exactly(
    name: str, policy: str, method: str
) -> None:
    m, _, issues, d1, b4, out = _run(APPROVED_EXACT[name], policy, method)
    g = APPROVED_EXACT[name][5]
    assert issues == []
    assert d1.issues == []
    assert b4.issues == []
    assert m.n_cells > 0
    every = {(c, u) for c in range(m.n_cells) for u in range(3)}
    assert set(b4.bounds) == every  # every cell and component is covered
    assert set(d1.bounds) == every
    assert all(r == 0 for r in b4.bounds.values()), "the fixture does not certify R = 0"
    assert all(Fr(float(out[c, u])) == Fr(g[u]) for c in range(m.n_cells) for u in range(3))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", sorted(BOUNDARY_CASES))
def test_a_boundary_case_passes_checks_1_to_4(name: str, policy: str, method: str) -> None:
    m, read, issues, d1, b4, _ = _run(BOUNDARY_CASES[name], policy, method)
    n_int = len(read.neighbour)
    o = read.cell_face_offsets
    boundary = [
        sum(int(read.cell_faces[k]) >= n_int for k in range(int(o[c]), int(o[c + 1])))
        for c in range(m.n_cells)
    ]
    faces = [int(o[c + 1]) - int(o[c]) for c in range(m.n_cells)]
    if name == "single-cell-hex-linear":
        assert boundary == faces  # every face of the cell is a boundary face
    if name == "hex-N3-linear":
        assert 1 in boundary  # some cells have exactly one boundary face
    assert m.n_cells > 0
    every = {(c, u) for c in range(m.n_cells) for u in range(3)}
    assert set(b4.bounds) == every
    assert set(d1.bounds) == every
    assert issues == []
    assert d1.issues == []
    assert b4.issues == []
