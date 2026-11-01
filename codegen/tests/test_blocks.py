"""Matrix blocks (Jac, JacAccum) read by generated loops: passive inputs of a linear operator."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]

MODULE = '''
from autan_codegen.dsl import Inc, Jac, JacAccum, Residual, State, Write, cell_loop, cells, faces
from autan_codegen.dsl import face_loop, pointwise

x = cells.field("x", State.Vec5)
diag = cells.field("diag", JacAccum.Mat5)
off = faces.field("off", Jac.Mat5)
y = cells.field("y", Residual.Vec5)


@pointwise
def apply(a: JacAccum.Mat5, v: State.Vec5) -> Residual.Vec5:
    return a @ v


@pointwise
def couple(b: Jac.Mat5, vl: State.Vec5, vr: State.Vec5) -> Residual.Vec5:
    return b @ (vr - vl)


diagonal = cell_loop(apply, a=diag, v=x, out=Write(y))
coupling = face_loop(couple, b=off, vl=x.left, vr=x.right, out=Inc(y, left=+1, right=-1))
'''

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
      off(in + 30 * nc, in + 30 * nc + 25 * nf), y(5 * nc, 0.0);
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
    # Blocks are never differentiated: no tangent or adjoint of diag or off.
    for name in ("diag_tan", "diag_adj", "off_tan", "off_adj"):
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
    diag, off = rng.standard_normal((3, 5, 5)), rng.standard_normal((2, 5, 5))
    # Generated datasets are component-major: component k of entity i at k * n + i.
    inputs = np.concatenate(
        [x.T.ravel(), diag.reshape(3, 25).T.ravel(), off.reshape(2, 25).T.ravel()]
    )
    y = np.zeros(15)
    ctypes.CDLL(str(lib)).run(
        inputs.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p)
    )
    expect = np.einsum("cij,cj->ci", diag, x)
    for f, (left, right) in enumerate(((0, 1), (1, 2))):
        flow = off[f] @ (x[right] - x[left])
        expect[left] += flow
        expect[right] -= flow
    np.testing.assert_allclose(y.reshape(5, 3).T, expect, rtol=1e-14, atol=1e-14)
