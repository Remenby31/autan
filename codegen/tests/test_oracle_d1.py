"""D1 (ADR 0003, check 3): the gradient kernels replayed from the generator's IR, and the kernel
against the intended operator evaluated exactly from the nodes."""

import functools
import json
import os
from fractions import Fraction as Fr
from pathlib import Path

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import certified as c
from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t

CASES = [(k, p) for k in ("hex", "tet", "hybrid", "poly") for p in (0.0, 0.15)]
POLICIES = ("fp64", "fp32+acc64", "fp32")


@functools.cache
def _case(kind: str, perturbation: float) -> tuple[s.Mesh, t.ReadMesh, dict, dict]:
    m = s.mesh(meshes.build(kind, 4, perturbation, seed=3))
    read = t.read_mesh(m)
    return m, read, q.construct(read), q.geometry_exact(read)


def _inputs(m: s.Mesh, geometry: dict) -> tuple[kr.Inputs, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    qv = rng.standard_normal(m.n_cells)
    qb = rng.standard_normal(m.n_faces)
    g0 = np.random.default_rng(6).standard_normal((m.n_cells, 3))
    inp = kr.Inputs(
        geometry,
        [kr.given(float(v), c.BINARY64) for v in qv],
        [kr.given(float(v), c.BINARY64) for v in qb],
        [[kr.given(float(v), c.BINARY64) for v in row] for row in g0],
    )
    return inp, qv, qb, g0


def _kernel(
    m: s.Mesh, policy: str, method: str, qv: np.ndarray, qb: np.ndarray, g0: np.ndarray
) -> np.ndarray:
    if method == "lsq":
        return s.gradients(m, policy, "lsq", qv, qb)[0]
    return s.gradients(m, policy, "gg", qv, qb, 1, g0)[0]


@pytest.mark.parametrize("method", ["lsq", "gg"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_the_replay_is_the_kernel_bit_for_bit(
    kind: str, perturbation: float, policy: str, method: str
) -> None:
    m, read, geometry, _ = _case(kind, perturbation)
    inp, qv, qb, g0 = _inputs(m, geometry)
    g = _kernel(m, policy, method, qv, qb, g0)
    data = kr.load_inputs(kr.module(), inp, policy)
    run = kr.lsq if method == "lsq" else kr.green_gauss
    replay = run(read, data, policy, kr.Rounded())
    for cell in range(m.n_cells):
        for u in range(3):
            assert replay[cell][u].e == Fr(float(g[cell, u])), (cell, u)  # type: ignore[attr-defined]


@pytest.mark.parametrize("method", ["lsq", "gg"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_d1_holds_on_the_b_cases(kind: str, perturbation: float, policy: str, method: str) -> None:
    m, read, geometry, exact = _case(kind, perturbation)
    inp, qv, qb, g0 = _inputs(m, geometry)
    g = _kernel(m, policy, method, qv, qb, g0)
    result = kr.check_d1(read, inp, exact, g.tolist(), policy, method)
    report = os.environ.get("AUTAN_D1_REPORT")
    if report:
        with Path(report).open("a") as f:
            f.write(json.dumps({
                "case": f"{kind}-{perturbation}", "policy": policy, "method": method,
                "issues": len(result.issues), "max_centre_ratio": float(result.max_centre_ratio),
                "max_use": float(result.max_use),
            }) + "\n")  # fmt: skip
    assert result.issues == [], result.issues[:5]


B_FIELDS = [(1.0, (0.3, -1.2, 0.7)), (1000.0, (0.3, -1.2, 0.7))]  # step B's a and G


@pytest.mark.parametrize("field", range(len(B_FIELDS)))
@pytest.mark.parametrize("method", ["lsq", "gg"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_the_bound_holds_on_the_b_cases(
    kind: str, perturbation: float, policy: str, method: str, field: int
) -> None:
    m, read, geometry, exact = _case(kind, perturbation)
    a, g = B_FIELDS[field]
    qv, qb = kr.field_floats(geometry, len(read.neighbour), a, g)
    qv_, qb_ = np.array(qv), np.array(qb)
    if method == "lsq":
        out = s.gradients(m, policy, "lsq", qv_, qb_)[0]
    else:
        out = s.gradients(m, policy, "gg", qv_, qb_, 1, np.tile(np.array(g), (m.n_cells, 1)))[0]
    result = kr.check_bound(read, geometry, exact, a, g, out.tolist(), policy, method)
    report = os.environ.get("AUTAN_D1_REPORT")
    if report:
        with Path(report.replace("d1report", "boundreport")).open("a") as f:
            f.write(json.dumps({
                "case": f"{kind}-{perturbation}", "policy": policy, "method": method, "a": a,
                "issues": len(result.issues), "first": result.issues[:2],
                "max_centre_ratio": float(result.max_centre_ratio), "max_use": float(result.max_use),
            }) + "\n")  # fmt: skip
    assert result.issues == [], result.issues[:5]
