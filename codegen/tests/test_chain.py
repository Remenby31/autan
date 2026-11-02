"""A chain of two cell reductions (dsl.chain) against the two reductions run one after the other:
codegen/tests/kernels/chained.py, Euler's gradients and limiter pattern."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from typing import TYPE_CHECKING

import autan_spike as s
import meshes
import numpy as np
import pytest
from autan_codegen.generate import generate
from conftest import KINDS, PRECISIONS, ROOT, refused

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("policy", PRECISIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_a_chain_gives_the_bits_of_its_two_reductions(backend: str, kind: str, policy: str) -> None:
    # The chain keeps the first reduction's outputs in registers, converted as its own loop
    # stores them: the second reads the same values, so grad and phi carry the same bits. The
    # bounds it does not keep are never written.
    device = backend != "cpu"
    full = s.mesh(meshes.build(kind, 6, 0.15, seed=51))
    s.set_stencil(full, "face2")
    rng = np.random.default_rng(52)
    q, q_b = rng.standard_normal(full.n_cells), rng.standard_normal(full.n_faces)
    code = refused(backend, policy) if device else None
    if code is not None:
        with pytest.raises(s.SpikeError) as err:
            s.chained(full, policy, q, q_b, True, device)
        assert str(err.value).startswith(code)
        return
    two = s.chained(full, policy, q, q_b, False, device)
    one = s.chained(full, policy, q, q_b, True, device)
    assert two["grad"].tobytes() == one["grad"].tobytes()
    assert two["phi"].tobytes() == one["phi"].tobytes()
    assert np.isfinite(two["lo"]).all()
    assert np.isfinite(two["hi"]).all()
    assert np.isnan(one["lo"]).all()
    assert np.isnan(one["hi"]).all()
    if backend == "metal":
        return  # the global order is a DeviceMesh option (CUDA/HIP)

    # Ordered by the global numbering (stencil by global keys), 2 ranks give each owned cell the
    # bits 1 rank gives it.
    def owned(m: s.Mesh) -> dict[int, bytes]:
        s.set_global_order(m, "face2")
        cells, faces = s.global_ids(m)
        r = s.chained(m, policy, q[cells], q_b[faces], True, device)
        return {
            int(cells[c]): r["grad"][c].tobytes() + r["phi"][c].tobytes() for c in range(m.n_owned)
        }

    reference = owned(s.partition(full, np.zeros(full.n_cells, np.int32), 1, 0, layers=2))
    x = full.cell_centroid[:, 0]
    parts = (x > np.median(x)).astype(np.int32)
    for rank in (0, 1):
        got = owned(s.partition(full, parts, 2, rank, layers=2))
        assert got.keys() <= reference.keys()
        assert all(got[g] == reference[g] for g in got), (kind, rank)


# A policy whose storage is narrower than its compute type: no shipped policy is, so there the
# chain's conversion of its registers to their storage types is the only thing that gives the
# bits of the stored and reread values (each body converts its arguments to the compute type).
# Three cells in a row, two interior faces, a boundary face at each end, each cell's stencil its
# face neighbours and boundary faces.
NARROW_DRIVER = r"""
#include <vector>
#include "chained_cpu.hpp"
using namespace autan::kernels;
struct narrow {
  using state = float;
  using grad = float;
  using residual = double;
  using geom = double;
  using geom_value = double;
  using param = double;
  using jac = float;
  using compute = double;
  using accum = double;
  static constexpr const char* name = "narrow";
};
extern "C" void run(int fused, const double* q, const double* q_b, const double* w,
                    const double* xf, const double* xc, double* grad, double* phi) {
  std::vector<index_t> owner{0, 1, 0, 2}, neighbour{1, 2}, cell_faces{0, 2, 0, 1, 1, 3};
  std::vector<offset_t> offsets{0, 2, 4, 6}, stencil_offsets{0, 2, 4, 6};
  std::vector<index_t> stencil{1, -3, 0, 2, 1, -4};
  Topology t{};
  t.n_cells = t.n_owned_cells = 3;
  t.n_faces = t.n_owned_faces = 4;
  t.n_interior_faces = t.n_owned_interior_faces = 2;
  t.n_incidences = 6;
  t.owner = owner.data();
  t.neighbour = neighbour.data();
  t.cell_face_offsets = offsets.data();
  t.cell_faces = cell_faces.data();
  t.n_stencil = 6;
  t.stencil_offsets = stencil_offsets.data();
  t.stencil = stencil.data();
  std::vector<float> qs(q, q + 3), qbs(q_b, q_b + 4), g(9), lo(3), hi(3), ph(3);
  chained::Data<narrow> d{};
  d.topo = t;
  d.q = qs.data();
  d.q_b = qbs.data();
  d.stencil_lsq_weight = w;
  d.face_centroid = xf;
  d.cell_centroid = xc;
  d.grad = g.data();
  d.lo = lo.data();
  d.hi = hi.data();
  d.phi = ph.data();
  if (fused != 0) {
    chained::cpu::fused(d);
  } else {
    chained::cpu::gradients(d);
    chained::cpu::limiter(d);
  }
  for (int k = 0; k < 9; ++k) grad[k] = g[k];
  for (int c = 0; c < 3; ++c) phi[c] = ph[c];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_a_chain_rounds_its_registers_to_their_storage_types(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate(ROOT / "codegen" / "tests" / "kernels" / "chained.py", out, ["cpu"])
    (tmp_path / "narrow.cpp").write_text(NARROW_DRIVER)
    lib_path = tmp_path / "libnarrow.so"
    subprocess.run(
        ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
         f"-I{ROOT / 'codegen' / 'include'}", f"-I{out}", str(tmp_path / "narrow.cpp"),
         "-o", str(lib_path)],
        check=True, capture_output=True,
    )  # fmt: skip
    lib = ctypes.CDLL(str(lib_path))
    rng = np.random.default_rng(53)
    q, q_b = rng.standard_normal(3), rng.standard_normal(4)
    w = np.ascontiguousarray(rng.standard_normal(18) / 3.0)
    xf = np.ascontiguousarray(rng.standard_normal(12))
    xc = np.ascontiguousarray(rng.standard_normal(9))
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    got = {}
    for fused in (0, 1):
        grad, phi = np.zeros(9), np.zeros(3)
        lib.run(ctypes.c_int(fused), p(q), p(q_b), p(w), p(xf), p(xc), p(grad), p(phi))
        got[fused] = (grad, phi)
    assert got[0][0].tobytes() == got[1][0].tobytes()
    assert got[0][1].tobytes() == got[1][1].tobytes()
