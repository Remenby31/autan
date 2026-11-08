"""Defect controls of ADR 0003 that corrupt the oracle's inputs, not the kernels': 4-8 (check 2),
9 and the topology half of 3 (check 1), 10 (check 4). Each control first proves its failure at the
output, |y_c - centre| > R + r_c, then runs the check, which must fail on the corrupted entry."""

import math
from fractions import Fraction as Fr

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import certified as c
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


# ------------------------------------------------ controls run on the kernels (test-only entry points)


def _lsq_instance(m: s.Mesh, read: t.ReadMesh) -> tuple[kr.Inputs, dict, np.ndarray, np.ndarray]:
    """A field whose every neighbour difference is nonzero (checked exactly), on the kernels."""
    geometry = q.construct(read)
    rng = np.random.default_rng(11)
    qv = rng.standard_normal(m.n_cells)
    qb = rng.standard_normal(m.n_faces)
    n_int = len(read.neighbour)
    for cell in range(m.n_cells):
        for k in range(int(read.cell_face_offsets[cell]), int(read.cell_face_offsets[cell + 1])):
            f = int(read.cell_faces[k])
            other = (
                (qv[read.neighbour[f]] if read.owner[f] == cell else qv[read.owner[f]])
                if f < n_int
                else qb[f]
            )
            assert np.float32(other) != np.float32(qv[cell])  # nonzero in every policy's storage
    inp = kr.Inputs(
        geometry,
        [kr.given(float(v), c.BINARY64) for v in qv],
        [kr.given(float(v), c.BINARY64) for v in qb],
    )
    return inp, geometry, qv, qb


def _proof_d1(
    read: t.ReadMesh,
    clean: kr.Inputs,
    corrupt_read: t.ReadMesh,
    corrupt: kr.Inputs,
    exact: dict,
    policy: str,
    cell: int,
) -> tuple[list[int], list[list[Fr]]]:
    """Components u of `cell` whose corrupted output interval [y_c - r_c, y_c + r_c] lies wholly
    outside the clean acceptance interval [g_int - R_D1, g_int + R_D1] (ADR 0003, controls)."""
    data = kr.load_inputs(clean, policy)
    q_hat = [v.e for v in data.q]  # type: ignore[attr-defined]
    qb_hat = [v.e for v in data.q_b]  # type: ignore[attr-defined]
    g_int = kr.lsq_intended(read, exact, q_hat, qb_hat)[cell]
    accepted = kr.lsq(read, data, policy, kr.Rounded())[cell]
    bad_all = kr.lsq(corrupt_read, kr.load_inputs(corrupt, policy), policy, kr.Rounded())
    bad = bad_all[cell]
    out = []
    for u in range(3):
        v, r = accepted[u].c.value, accepted[u].c.radius  # type: ignore[attr-defined]
        r_d1 = r + abs(v - g_int[u])
        y, rc = bad[u].c.value, bad[u].c.radius  # type: ignore[attr-defined]
        if abs(y - g_int[u]) > r_d1 + rc:
            out.append(u)
    return out, [[v.e for v in row] for row in bad_all]  # type: ignore[attr-defined]


@pytest.mark.parametrize(("control", "factor"), [(1, 2.0), (2, -1.0)])
def test_a_corrupted_weight_fails_checks_2_and_3(control: int, factor: float) -> None:
    # Controls 1 (one weight component doubled) and 2 (one weight's sign flipped).
    policy = "fp64"
    m, read = _mesh()
    inp, geometry, qv, qb = _lsq_instance(m, read)
    exact = q.geometry_exact(read)
    certs = q.geometry_certified(read)["lsq_weight"]
    for k in range(len(read.cell_faces)):
        cell = int(np.searchsorted(read.cell_face_offsets, k, side="right")) - 1
        w = [float(p.e) for p in geometry["lsq_weight"][k]]
        u0 = max(range(3), key=lambda u: abs(w[u]))
        new = list(w) if control == 1 else [-x for x in w]
        if control == 1:
            new[u0] = factor * w[u0]
        # Check 2: the stored weight leaves its certified interval.
        stored = q.stored_geometry(m)
        stored["lsq_weight"][k] = new
        moved = [j for j in range(3) if abs(Fr(new[j]) - certs[k][j].value) > certs[k][j].radius]
        assert moved, "the corrupted weight stays in its interval"
        assert any(
            i.startswith(f"lsq_weight[{k}][{moved[0]}]: the stored value")
            for i in q.check_geometry(read, stored)
        )
        # Check 3, where the output proof holds.
        bad_geo = dict(geometry)
        bad_geo["lsq_weight"] = list(geometry["lsq_weight"])
        bad_geo["lsq_weight"][k] = [kr.given(x, c.BINARY64) for x in new]
        proven, expected = _proof_d1(
            read, inp, read, kr.Inputs(bad_geo, inp.q, inp.q_b), exact, policy, cell
        )
        if not proven:
            continue
        clean = s.gradients(m, policy, "lsq", qv, qb)[0]
        assert kr.check_d1(read, inp, exact, clean.tolist(), policy, "lsq").issues == []
        s.test_only_set_lsq_weight(m, k, tuple(new))
        g = s.gradients(m, policy, "lsq", qv, qb)[0]
        # The run executed exactly the modelled corruption, then it fails at the proven entry.
        assert all(Fr(float(g[i, u])) == expected[i][u] for i in range(m.n_cells) for u in range(3))
        issues = kr.check_d1(read, inp, exact, g.tolist(), policy, "lsq").issues
        assert any(i.startswith(f"fail: lsq cell {cell} component {proven[0]}") for i in issues)
        return
    pytest.fail(f"control {control}: no incidence where the output proof holds (not demonstrated)")


def test_an_incidence_dropped_from_a_cell_fails_check_3() -> None:
    # Control 3, its check 3 half: the kernels run without one incidence.
    policy = "fp64"
    m, read = _mesh()
    inp, geometry, qv, qb = _lsq_instance(m, read)
    exact = q.geometry_exact(read)
    o = read.cell_face_offsets
    for k in range(len(read.cell_faces)):
        cell = int(np.searchsorted(o, k, side="right")) - 1
        corrupt = read.copy()
        corrupt.cell_faces = np.delete(read.cell_faces, k)
        corrupt.incidence_side = np.delete(read.incidence_side, k)
        corrupt.cell_face_offsets = np.where(o > k, o - 1, o)
        bad_geo = dict(geometry)
        bad_geo["lsq_weight"] = [row for i, row in enumerate(geometry["lsq_weight"]) if i != k]
        proven, expected = _proof_d1(
            read, inp, corrupt, kr.Inputs(bad_geo, inp.q, inp.q_b), exact, policy, cell
        )
        if not proven:
            continue
        clean = s.gradients(m, policy, "lsq", qv, qb)[0]
        assert kr.check_d1(read, inp, exact, clean.tolist(), policy, "lsq").issues == []
        s.test_only_drop_incidence(m, k)
        g = s.gradients(m, policy, "lsq", qv, qb)[0]
        assert all(Fr(float(g[i, u])) == expected[i][u] for i in range(m.n_cells) for u in range(3))
        issues = kr.check_d1(read, inp, exact, g.tolist(), policy, "lsq").issues
        assert any(i.startswith(f"fail: lsq cell {cell} component {proven[0]}") for i in issues)
        return
    pytest.fail("control 3: no incidence where the output proof holds (not demonstrated)")


# Control 11 (ADR 0003, as amended on #43): demonstrated through the validity condition of checks 3
# and 4, "the replay's executed value equals the kernel's output, value for value" (+-0 identified).
# On the measured fixtures, the envelopes R_D1 and R4 alone did not separate binary32 from binary64
# accumulation; no claim is made beyond them.
CONTROL_11_CASE = ("hex", 4, 0.15, 3, 1.0)  # kind, N, perturbation, seed, a: a B fixture
G_B = (0.3, -1.2, 0.7)


@pytest.mark.parametrize("method", ["lsq", "gg"])
def test_a_narrowed_accumulation_makes_checks_3_and_4_invalid(method: str) -> None:
    # Exact emulation first: the narrowed sequence (accumulation in binary32 under fp32+acc64)
    # differs from the declared one on at least one entry. Then the run with the test-only policy
    # (the generator's loops, accumulation narrowed) must give exactly the narrowed emulation, and
    # checks 3 and 4 must report that entry invalid: the build does not execute the declared sequence.
    good, test = "fp32+acc64", "fp32+acc32-test-only"
    kind, n, p, seed, a = CONTROL_11_CASE
    m = s.mesh(meshes.build(kind, n, p, seed=seed))
    read = t.read_mesh(m)
    geometry, exact = q.construct(read), q.geometry_exact(read)
    n_int = len(read.neighbour)
    qv, qb = kr.field_floats(geometry, n_int, a, G_B)
    grad0 = [[kr.given(v, c.BINARY64) for v in G_B]] * read.n_cells
    field = kr.linear_field(geometry["cell_centroid"], a, G_B)
    field_b = [kr.given(0.0, c.BINARY64)] * n_int + kr.linear_field(
        geometry["face_centroid"][n_int:], a, G_B
    )
    inp = kr.Inputs(geometry, field, field_b, grad0 if method == "gg" else None)
    run = kr.lsq if method == "lsq" else kr.green_gauss
    narrowed = run(read, kr.load_inputs(inp, test), test, kr.Rounded())
    declared = run(read, kr.load_inputs(inp, good), good, kr.Rounded())
    differ = [
        (cell, u)
        for cell in range(read.n_cells)
        for u in range(3)
        if narrowed[cell][u].e != declared[cell][u].e  # type: ignore[attr-defined]
    ]
    assert differ, "control 11 not demonstrated: the narrowed sequence equals the declared one"
    g0 = np.tile(np.array(G_B), (m.n_cells, 1))
    # The nominal kernel on the same fixture passes both checks first.
    nominal = (
        s.gradients(m, good, "lsq", np.array(qv), np.array(qb))[0]
        if method == "lsq"
        else s.gradients(m, good, "gg", np.array(qv), np.array(qb), 1, g0)[0]
    )
    assert kr.check_d1(read, inp, exact, nominal.tolist(), good, method).issues == []
    assert (
        kr.check_bound(read, geometry, exact, a, G_B, nominal.tolist(), good, method).issues == []
    )
    out = (
        s.gradients(m, test, "lsq", np.array(qv), np.array(qb))[0]
        if method == "lsq"
        else s.gradients(m, test, "gg", np.array(qv), np.array(qb), 1, g0)[0]
    )
    for cell in range(read.n_cells):
        for u in range(3):
            assert Fr(float(out[cell, u])) == narrowed[cell][u].e, (cell, u)  # type: ignore[attr-defined]
    cell, u = differ[0]
    where = f"invalid: {method} cell {cell} component {u}: the build does not execute"
    d1 = kr.check_d1(read, inp, exact, out.tolist(), good, method)
    b4 = kr.check_bound(read, geometry, exact, a, G_B, out.tolist(), good, method)
    assert any(i.startswith(where) for i in d1.issues)
    assert any(i.startswith(where) for i in b4.issues)
