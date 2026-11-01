"""Matrix blocks (Jac, JacAccum) read by generated loops: passive inputs of a linear operator."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import autan_spike as s
import meshes
import numpy as np
import pytest
from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]

MODULE = """
from autan_codegen.dsl import Inc, Jac, JacAccum, Residual, State, Write, cell_loop, cells, faces
from autan_codegen.dsl import face_loop, pointwise

x = cells.field("x", State.Vec5)
diag = cells.field("diag", JacAccum.Mat5)
shift = cells.field("shift", Jac.Mat5)
off = faces.field("off", Jac.Mat5)
y = cells.field("y", Residual.Vec5)


@pointwise
def apply(a: JacAccum.Mat5, s: Jac.Mat5, v: State.Vec5) -> Residual.Vec5:
    return a @ v + s @ v


@pointwise
def couple(b: Jac.Mat5, vl: State.Vec5, vr: State.Vec5) -> Residual.Vec5:
    return b @ (vr - vl)


diagonal = cell_loop(apply, a=diag, s=shift, v=x, out=Write(y))
coupling = face_loop(couple, b=off, vl=x.left, vr=x.right, out=Inc(y, left=+1, right=-1))
"""

# Three cells in a chain, faces 0 (0-1) and 1 (1-2), no boundary face: y_c = D_c x_c, then
# face f adds B_f (x_r - x_l) to its left cell and subtracts it from its right one. A shared
# library loaded by ctypes, as the spike drivers (no executable to launch).
DRIVER = r"""
#include <vector>
#include "blocks_cpu.hpp"
using namespace autan::kernels;
extern "C" void run(const double* in, double* out) {
  constexpr index_t nc = 3, nf = 2;
  std::vector<double> x(in, in + 5 * nc), diag(in + 5 * nc, in + 30 * nc),
      shift(in + 30 * nc, in + 55 * nc), off(in + 55 * nc, in + 55 * nc + 25 * nf),
      y(5 * nc, 0.0);
  std::vector<index_t> owner{0, 1}, neighbour{1, 2}, cell_faces{0, 0, 1, 1};
  std::vector<offset_t> offsets{0, 1, 3, 4};
  Topology t{};
  t.n_cells = t.n_owned_cells = nc;
  t.n_faces = t.n_interior_faces = t.n_owned_interior_faces = t.n_owned_faces = nf;
  t.n_incidences = 4;
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = offsets.data();
  t.cell_faces = cell_faces.data();
  blocks::Data<fp64> d{};
  d.topo = t;
  d.x = x.data();
  d.diag = diag.data();
  d.shift = shift.data();
  d.off = off.data();
  d.y = y.data();
  blocks::cpu::diagonal(d);
  blocks::cpu::coupling(d);
  for (std::size_t i = 0; i < y.size(); ++i) out[i] = y[i];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_blocks_are_passive_inputs_of_a_block_operator(tmp_path: Path) -> None:
    src = tmp_path / "blocks.py"
    src.write_text(MODULE)
    out = tmp_path / "gen"
    generate(src, out, ["cpu"])
    header = (out / "blocks.hpp").read_text()
    # Blocks are never differentiated, cell blocks included: no tangent or adjoint of them.
    for name in ("diag_tan", "diag_adj", "shift_tan", "shift_adj", "off_tan", "off_adj"):
        assert name not in header, name
    assert "typename P::accum* diag;" in header  # JacAccum: accumulation precision
    assert "typename P::jac* off;" in header
    (tmp_path / "driver.cpp").write_text(DRIVER)
    lib = tmp_path / "libblocks.so"
    subprocess.run(
        ["c++", "-std=c++20", "-O1", "-shared", "-fPIC", f"-I{ROOT / 'codegen' / 'include'}",
         f"-I{ROOT / 'core' / 'include'}", f"-I{out}", str(tmp_path / "driver.cpp"), "-o", str(lib)],
        check=True, capture_output=True,
    )  # fmt: skip
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 5))
    diag, shift = rng.standard_normal((3, 5, 5)), rng.standard_normal((3, 5, 5))
    off = rng.standard_normal((2, 5, 5))
    # Generated datasets are component-major: component k of entity i at k * n + i.
    inputs = np.concatenate(
        [
            x.T.ravel(),
            diag.reshape(3, 25).T.ravel(),
            shift.reshape(3, 25).T.ravel(),
            off.reshape(2, 25).T.ravel(),
        ]
    )
    y = np.zeros(15)
    ctypes.CDLL(str(lib)).run(
        inputs.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p)
    )
    expect = np.einsum("cij,cj->ci", diag, x) + np.einsum("cij,cj->ci", shift, x)
    for f, (left, right) in enumerate(((0, 1), (1, 2))):
        flow = off[f] @ (x[right] - x[left])
        expect[left] += flow
        expect[right] -= flow
    np.testing.assert_allclose(y.reshape(5, 3).T, expect, rtol=1e-14, atol=1e-14)


ASSEMBLY = """
from autan_codegen.dsl import Geom, Jac, JacAccum, Sum, cell_reduce, cells, faces, geometry, pointwise

jl = faces.field("jl", Jac.Mat5)
jr = faces.field("jr", Jac.Mat5)
diag = cells.field("diag", JacAccum.Mat5)


@pointwise
def diagonal_term(a: Jac.Mat5, b: Jac.Mat5, s: Geom) -> JacAccum.Mat5:
    return a if s > 0.0 else -b


assembled = cell_reduce(diagonal_term, a=jl, b=jr, s=geometry.incidence_side, acc=Sum(diag))
"""

# The chain of DRIVER plus boundary face 2 on cell 2: a cell's diagonal block sums, over its
# faces in incidence order, the face's left block where it is the left cell (or on a boundary
# face) and minus its right block where it is the right cell.
ASSEMBLY_DRIVER = r"""
#include <vector>
#include "assembly_cpu.hpp"
using namespace autan::kernels;
extern "C" void run(const double* in, double* out) {
  constexpr index_t nc = 3, nf = 3;
  std::vector<double> jl(in, in + 25 * nf), jr(in + 25 * nf, in + 50 * nf),
      side(in + 50 * nf, in + 50 * nf + 5), diag(25 * nc, 0.0);
  std::vector<index_t> owner{0, 1, 2}, neighbour{1, 2}, cell_faces{0, 0, 1, 1, 2};
  std::vector<offset_t> offsets{0, 1, 3, 5};
  Topology t{};
  t.n_cells = t.n_owned_cells = nc;
  t.n_faces = t.n_owned_faces = nf;
  t.n_interior_faces = t.n_owned_interior_faces = 2;
  t.n_incidences = 5;
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = offsets.data();
  t.cell_faces = cell_faces.data();
  assembly::Data<fp64> d{};
  d.topo = t;
  d.jl = jl.data();
  d.jr = jr.data();
  d.incidence_side = side.data();
  d.diag = diag.data();
  assembly::cpu::assembled(d);
  for (std::size_t i = 0; i < diag.size(); ++i) out[i] = diag[i];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_diagonal_blocks_assemble_from_face_blocks_by_incidence_side(tmp_path: Path) -> None:
    src = tmp_path / "assembly.py"
    src.write_text(ASSEMBLY)
    out = tmp_path / "gen"
    generate(src, out, ["cpu"])
    (tmp_path / "driver.cpp").write_text(ASSEMBLY_DRIVER)
    lib = tmp_path / "libassembly.so"
    subprocess.run(
        ["c++", "-std=c++20", "-O1", "-shared", "-fPIC", f"-I{ROOT / 'codegen' / 'include'}",
         f"-I{ROOT / 'core' / 'include'}", f"-I{out}", str(tmp_path / "driver.cpp"), "-o", str(lib)],
        check=True, capture_output=True,
    )  # fmt: skip
    rng = np.random.default_rng(4)
    jl, jr = rng.standard_normal((3, 5, 5)), rng.standard_normal((3, 5, 5))
    owner, neighbour = [0, 1, 2], [1, 2]
    incidences = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]  # (cell, face) in CSR order
    side = [1.0 if f >= len(neighbour) or owner[f] == c else -1.0 for c, f in incidences]
    inputs = np.concatenate(
        [jl.reshape(3, 25).T.ravel(), jr.reshape(3, 25).T.ravel(), np.array(side)]
    )
    diag = np.zeros(75)
    ctypes.CDLL(str(lib)).run(
        inputs.ctypes.data_as(ctypes.c_void_p), diag.ctypes.data_as(ctypes.c_void_p)
    )
    expect = np.zeros((3, 5, 5))
    for (c, f), sign in zip(incidences, side, strict=True):
        expect[c] += jl[f] if sign > 0 else -jr[f]
    np.testing.assert_array_equal(diag.reshape(25, 3).T.reshape(3, 5, 5), expect)


@pytest.mark.parametrize("kind", ["tet", "hex", "poly", "hybrid"])
def test_incidence_side_is_plus_one_on_the_left_cell_and_on_boundary_faces(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=7))
    for c in range(m.n_cells):
        for k in range(m.cell_face_offsets[c], m.cell_face_offsets[c + 1]):
            f = m.cell_faces[k]
            left = f >= m.n_interior or m.owner[f] == c
            assert m.incidence_side[k] == (1.0 if left else -1.0), (c, f)
