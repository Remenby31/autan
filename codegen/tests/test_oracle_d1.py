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
    data = kr.load_inputs(inp, policy)
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


def test_the_generator_follows_the_declared_precision_contract() -> None:
    assert kr.check_contract(kr.module()) == []


def test_a_generator_storing_an_output_otherwise_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A generator-side departure on an output-only dataset: qmin stored in P::residual, which keeps
    # every selected binary32 minimum's value. The output comparison cannot see it; the contract does.
    real = kr.emit_cpp.storage_type

    def wrong(d: object, rel_read: set[int]) -> str:
        return "typename P::residual" if getattr(d, "name", "") == "qmin" else real(d, rel_read)  # type: ignore[arg-type]

    monkeypatch.setattr(kr.emit_cpp, "storage_type", wrong)
    assert any(i.startswith("qmin: stored in residual") for i in kr.check_contract(kr.module()))


def test_a_missing_finalise_is_reported() -> None:
    m = kr.module()
    lm = next(x for x in m.loops if x.name == "lsq_gradient_bounds")
    lm.fin_fn, lm.fin_sig = None, None
    assert any("include_own_value" in i for i in kr.check_contract(m))


def test_the_compiled_generated_code_is_the_frozen_text() -> None:
    assert kr.check_frozen() == []


def test_a_change_to_the_frozen_code_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(kr.FROZEN_SHA256, "codegen/generated/gradients_cpu.hpp", "0" * 64)
    assert any("gradients_cpu.hpp" in i for i in kr.check_frozen())


def test_a_missing_finalise_makes_both_checks_invalid_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Structural validity is checked before any replay: the public checks report it as invalid.
    m0 = kr.module()
    lm = next(x for x in m0.loops if x.name == "lsq_gradient_bounds")
    lm.fin_fn, lm.fin_sig = None, None
    monkeypatch.setattr(kr, "module", lambda: m0)
    m, read, geometry, exact = _case("hex", 0.0)
    inp, qv, qb, _ = _inputs(m, geometry)
    g = s.gradients(m, "fp64", "lsq", qv, qb)[0].tolist()
    d1 = kr.check_d1(read, inp, exact, g, "fp64", "lsq")
    assert any(i.startswith("invalid: precision contract") for i in d1.issues)
    a, gb = 1.0, (0.3, -1.2, 0.7)
    b4 = kr.check_bound(read, geometry, exact, a, gb, g, "fp64", "lsq")
    assert any(i.startswith("invalid: precision contract") for i in b4.issues)


def test_the_loaded_library_is_identified_by_its_digest() -> None:
    digest = kr.library_sha256()
    assert digest is not None
    assert kr.check_library(digest) == []


def test_a_stale_library_beside_matching_headers_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The headers next to it would pass check_frozen; the library itself is not the frozen one.
    frozen = kr.library_sha256()
    assert frozen is not None
    lib = Path(os.environ["AUTAN_SPIKE_LIB"])
    (tmp_path / "generated").mkdir()
    for name in ("gradients.hpp", "gradients_cpu.hpp"):
        (tmp_path / "generated" / name).write_bytes((lib.parent / "generated" / name).read_bytes())
    stale = tmp_path / lib.name
    stale.write_bytes(lib.read_bytes() + b"\0")
    monkeypatch.setenv("AUTAN_SPIKE_LIB", str(stale))
    assert kr.check_frozen() == []  # the headers match
    assert kr.check_library(frozen)  # the library does not
