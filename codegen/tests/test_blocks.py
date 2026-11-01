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
from autan_codegen.errors import Code, CodegenError
from autan_codegen.generate import generate
from conftest import PRECISIONS, eps

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

face_jac = faces.field("face_jac", Jac.shape(2, 5, 5))
bnd_jac = faces.field("bnd_jac", Jac.Mat5)
diag = cells.field("diag", JacAccum.Mat5)


@pointwise
def diagonal_term(j: Jac.Mat5, jb: Jac.Mat5, s: Geom) -> JacAccum.Mat5:
    return s * j + jb


assembled = cell_reduce(
    diagonal_term, j=face_jac.near, jb=bnd_jac.on_boundary, s=geometry.incidence_side,
    acc=Sum(diag),
)
"""

# The chain of DRIVER plus boundary face 2 on cell 2: a cell's diagonal block sums, over its
# faces in incidence order, the face's owner block face_jac[0] where it is the owner, minus its
# neighbour block face_jac[1] where it is the neighbour, and bnd_jac on a boundary face.
ASSEMBLY_DRIVER = r"""
#include <vector>
#include "assembly_cpu.hpp"
using namespace autan::kernels;
extern "C" void run(const double* in, double* out) {
  constexpr index_t nc = 3, nf = 3;
  std::vector<double> face_jac(in, in + 50 * nf), bnd_jac(in + 50 * nf, in + 75 * nf),
      side(in + 75 * nf, in + 75 * nf + 5), diag(25 * nc, 0.0);
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
  d.face_jac = face_jac.data();
  d.bnd_jac = bnd_jac.data();
  d.incidence_side = side.data();
  d.diag = diag.data();
  assembly::cpu::assembled(d);
  for (std::size_t i = 0; i < diag.size(); ++i) out[i] = diag[i];
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_diagonal_blocks_assemble_from_the_blocks_valid_on_each_face(tmp_path: Path) -> None:
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
    face_jac, bnd_jac = rng.standard_normal((3, 2, 5, 5)), rng.standard_normal((3, 5, 5))
    # Blocks a face does not have are never written by the physics: NaN must not reach a sum.
    face_jac[2] = np.nan
    bnd_jac[:2] = np.nan
    owner, neighbour = [0, 1, 2], [1, 2]
    incidences = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]  # (cell, face) in CSR order
    side = [1.0 if f >= len(neighbour) or owner[f] == c else -1.0 for c, f in incidences]
    inputs = np.concatenate(
        [face_jac.reshape(3, 50).T.ravel(), bnd_jac.reshape(3, 25).T.ravel(), np.array(side)]
    )
    diag = np.zeros(75)
    ctypes.CDLL(str(lib)).run(
        inputs.ctypes.data_as(ctypes.c_void_p), diag.ctypes.data_as(ctypes.c_void_p)
    )
    expect = np.zeros((3, 5, 5))
    for c, f in incidences:
        if f >= len(neighbour):
            expect[c] += bnd_jac[f]
        elif owner[f] == c:
            expect[c] += face_jac[f, 0]
        else:
            expect[c] -= face_jac[f, 1]
    np.testing.assert_array_equal(diag.reshape(25, 3).T.reshape(3, 5, 5), expect)


@pytest.mark.parametrize("kind", ["tet", "hex", "poly", "hybrid"])
def test_incidence_side_is_plus_one_on_the_left_cell_and_on_boundary_faces(kind: str) -> None:
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=7))
    for c in range(m.n_cells):
        for k in range(m.cell_face_offsets[c], m.cell_face_offsets[c + 1]):
            f = m.cell_faces[k]
            left = f >= m.n_interior or m.owner[f] == c
            assert m.incidence_side[k] == (1.0 if left else -1.0), (c, f)


# One driver for the host and the device: kernels/blocks.py inverts n blocks given as doubles
# (component-major), stored in the policy's accumulation type.
INVERSE_DRIVER = r"""
#include <vector>
#include "blocks.hpp"
#ifdef __CUDACC__
#include "blocks_gpu.cuh"
#else
#include "blocks_cpu.hpp"
#endif
using namespace autan::kernels;
using P = POLICY;
using A = P::accum;
extern "C" void run(int n, const double* in, double* out) {
  std::vector<A> a(in, in + 25 * n), inv(25 * n);
  Topology t{};
  t.n_cells = t.n_owned_cells = n;
  blocks::Data<P> d{};
  d.topo = t;
#ifdef __CUDACC__
  const std::size_t bytes = sizeof(A) * a.size();
  cudaMalloc(&d.diag, bytes);
  cudaMalloc(&d.diag_inv, bytes);
  cudaMemcpy(d.diag, a.data(), bytes, cudaMemcpyHostToDevice);
  blocks::gpu_loops::invert_diagonal(d, gpu::Schedule{});
  cudaMemcpy(inv.data(), d.diag_inv, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d.diag);
  cudaFree(d.diag_inv);
#else
  d.diag = a.data();
  d.diag_inv = inv.data();
  blocks::cpu::invert_diagonal(d);
#endif
  for (std::size_t i = 0; i < inv.size(); ++i) out[i] = double(inv[i]);
}
"""

POLICY_TYPES = {"fp64": "fp64", "fp32+acc64": "fp32_acc64", "fp32": "fp32"}
N_REGULAR = 48


def invert5(a: np.ndarray) -> np.ndarray:
    """Gauss-Jordan of the host BlockMatrix (physics/src/block_matrix.cpp), in Python floats."""
    m = [[float(a[i, j]) for j in range(5)] + [float(i == j) for j in range(5)] for i in range(5)]
    for k in range(5):
        p = k
        for i in range(k + 1, 5):
            if abs(m[i][k]) > abs(m[p][k]):
                p = i
        if m[p][k] == 0.0 or not np.isfinite(m[p][k]):
            return np.full((5, 5), np.nan)
        m[k], m[p] = m[p], m[k]
        d = 1.0 / m[k][k]
        m[k] = [x * d for x in m[k]]
        for i in range(5):
            if i != k:
                f = m[i][k]
                m[i] = [x - f * y for x, y in zip(m[i], m[k], strict=True)]
    return np.array([row[5:] for row in m])


def _test_blocks() -> np.ndarray:
    """Regular blocks (diagonally dominant, row-permuted so that pivoting swaps, row-scaled over
    six decades), then exactly singular ones (zero, a zero column, an infinite or a NaN entry)."""
    rng = np.random.default_rng(5)
    regular = rng.standard_normal((N_REGULAR, 5, 5)) + 6.0 * np.eye(5)
    for b in regular[16:32]:
        b[:] = b[rng.permutation(5)]
    regular[32:] *= 10.0 ** rng.uniform(-3, 3, (16, 5, 1))
    singular = np.repeat(rng.standard_normal((1, 5, 5)) + 6.0 * np.eye(5), 4, axis=0)
    singular[0] = 0.0
    singular[1][:, 2] = 0.0
    singular[2][3, 1] = np.inf
    singular[3][0, 4] = np.nan
    return np.concatenate([regular, singular])


def _build(tmp_path: Path, backend: str, policy: str) -> ctypes.CDLL:
    out = tmp_path / f"gen_{backend}"
    generate(ROOT / "kernels" / "blocks.py", out, [backend], policies=[policy])
    source = tmp_path / ("driver.cu" if backend == "gpu" else "driver.cpp")
    source.write_text(INVERSE_DRIVER)
    lib = tmp_path / f"libinverse_{backend}.so"
    flags = [f"-DPOLICY={POLICY_TYPES[policy]}", f"-I{ROOT / 'codegen' / 'include'}",
             f"-I{ROOT / 'core' / 'include'}", f"-I{out}", "-o", str(lib)]  # fmt: skip
    if backend == "gpu":
        nvcc = shutil.which("nvcc")
        assert nvcc is not None, "the GPU backend tests need nvcc on PATH"
        cmd = [nvcc, "-std=c++20", "-O2", "-shared", "-Xcompiler", "-fPIC", *flags,
               str(source), str(out / "blocks_gpu.cu")]  # fmt: skip
    else:
        cmd = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-shared", "-fPIC", *flags,
               str(source)]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


def _invert(lib: ctypes.CDLL, blocks: np.ndarray) -> np.ndarray:
    n = len(blocks)
    inputs = np.ascontiguousarray(blocks.reshape(n, 25).T)  # component-major
    inv = np.zeros(25 * n)
    lib.run(
        ctypes.c_int(n), inputs.ctypes.data_as(ctypes.c_void_p), inv.ctypes.data_as(ctypes.c_void_p)
    )
    return inv.reshape(25, n).T.reshape(n, 5, 5)


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("policy", PRECISIONS)
def test_diagonal_blocks_invert_as_the_host_block_matrix(
    tmp_path: Path, backend: str, policy: str
) -> None:
    if backend == "metal":
        with pytest.raises(CodegenError) as err:
            generate(ROOT / "kernels" / "blocks.py", tmp_path, ["metal"])
        assert err.value.code == Code.POLICY_UNSUPPORTED_BY_BACKEND
        return
    blocks = _test_blocks()
    a = blocks[:N_REGULAR]
    host = _invert(_build(tmp_path, "cpu", policy), blocks)
    # The inverse runs in the accumulation type: FP64 under fp32+acc64, as the host matrix.
    tol = 64 * float(np.finfo(np.float32 if policy == "fp32" else np.float64).eps)
    bound = (
        tol
        * np.linalg.cond(a)[:, None, None]
        * np.abs(np.linalg.inv(a)).max(axis=(1, 2))[:, None, None]
    )
    assert (np.abs(host[:N_REGULAR] - np.linalg.inv(a)) <= bound).all()
    if policy != "fp32":
        np.testing.assert_array_equal(host, np.array([invert5(b) for b in blocks]))
    # A zero or non-finite pivot, where the host matrix throws, makes the whole inverse NaN.
    assert np.isnan(host[N_REGULAR:]).all()
    if backend == "gpu":
        lib = _build(tmp_path, "gpu", policy)
        dev = _invert(lib, blocks)
        assert (np.abs(dev[:N_REGULAR] - host[:N_REGULAR]) <= bound).all()
        assert np.isnan(dev[N_REGULAR:]).all()
        np.testing.assert_array_equal(_invert(lib, blocks), dev)


# kernels/blocks.py `multiply` on a mesh's topology, every input given as doubles (component-major)
# and stored in the type the policy gives its dataset.
MULTIPLY_DRIVER = r"""
#include <type_traits>
#include <vector>
#include "blocks.hpp"
#ifdef __CUDACC__
#include "blocks_gpu.cuh"
#else
#include "blocks_cpu.hpp"
#endif
using namespace autan::kernels;
using P = POLICY;
using D = blocks::Data<P>;
template <class T> using Of = std::remove_cvref_t<T>;
#ifdef __CUDACC__
template <class T> T* place(const std::vector<T>& v) {
  T* p = nullptr;
  cudaMalloc(&p, sizeof(T) * v.size());
  cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice);
  return p;
}
#else
template <class T> T* place(std::vector<T>& v) { return v.data(); }
#endif
template <class T> std::vector<T> as(const double* p, std::size_t n) { return {p, p + n}; }
extern "C" void run(int nc, int nf, int ni, long long nk, const index_t* owner,
                    const index_t* neighbour, const offset_t* offsets, const index_t* cell_faces,
                    const double* side, const double* face_jac, const double* diag,
                    const double* x, double* y) {
  D d{};
  auto o = std::vector<index_t>(owner, owner + nf);
  auto n = std::vector<index_t>(neighbour, neighbour + ni);
  auto off = std::vector<offset_t>(offsets, offsets + nc + 1);
  auto cf = std::vector<index_t>(cell_faces, cell_faces + nk);
  auto s = as<Of<decltype(*d.incidence_side)>>(side, nk);
  auto fj = as<Of<decltype(*d.face_jac)>>(face_jac, 50 * nf);
  auto dg = as<Of<decltype(*d.diag)>>(diag, 25 * nc);
  auto xv = as<Of<decltype(*d.x)>>(x, 5 * nc);
  std::vector<Of<decltype(*d.y)>> yv(5 * nc);
  Topology t{};
  t.n_cells = t.n_owned_cells = nc;
  t.n_faces = t.n_owned_faces = nf;
  t.n_interior_faces = t.n_owned_interior_faces = ni;
  t.n_incidences = nk;
  t.owner = place(o);
  t.neighbour = place(n);
  t.cell_face_offsets = place(off);
  t.cell_faces = place(cf);
  d.topo = t;
  d.incidence_side = place(s);
  d.face_jac = place(fj);
  d.diag = place(dg);
  d.x = place(xv);
  d.y = place(yv);
#ifdef __CUDACC__
  blocks::gpu_loops::multiply(d, gpu::Schedule{});
  cudaMemcpy(yv.data(), d.y, sizeof(yv[0]) * yv.size(), cudaMemcpyDeviceToHost);
#else
  blocks::cpu::multiply(d);
#endif
  for (std::size_t i = 0; i < yv.size(); ++i) y[i] = double(yv[i]);
}
"""


def _build_multiply(tmp_path: Path, backend: str, policy: str) -> ctypes.CDLL:
    out = tmp_path / f"gen_{backend}"
    generate(ROOT / "kernels" / "blocks.py", out, [backend], policies=[policy])
    source = tmp_path / ("multiply.cu" if backend == "gpu" else "multiply.cpp")
    source.write_text(MULTIPLY_DRIVER)
    lib = tmp_path / f"libmultiply_{backend}.so"
    flags = [f"-DPOLICY={POLICY_TYPES[policy]}", f"-I{ROOT / 'codegen' / 'include'}",
             f"-I{ROOT / 'core' / 'include'}", f"-I{out}", "-o", str(lib)]  # fmt: skip
    if backend == "gpu":
        nvcc = shutil.which("nvcc")
        assert nvcc is not None, "the GPU backend tests need nvcc on PATH"
        cmd = [nvcc, "-std=c++20", "-O2", "-shared", "-Xcompiler", "-fPIC", *flags, str(source)]
    else:
        cmd = ["c++", "-std=c++20", "-O2", "-shared", "-fPIC", *flags, str(source)]
    subprocess.run(cmd, check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


def _multiply(lib: ctypes.CDLL, m: s.Mesh, face_jac: np.ndarray, diag: np.ndarray, x: np.ndarray):
    def p(a: np.ndarray) -> ctypes.c_void_p:
        return a.ctypes.data_as(ctypes.c_void_p)

    # face_jac is entity-major (layout="aos"); diag and x are component-major.
    soa = [np.ascontiguousarray(face_jac.reshape(len(face_jac), -1))]
    soa += [np.ascontiguousarray(a.reshape(len(a), -1).T) for a in (diag, x)]
    y = np.zeros(5 * m.n_cells)
    lib.run(
        ctypes.c_int(m.n_cells), ctypes.c_int(m.n_faces), ctypes.c_int(m.n_interior),
        ctypes.c_longlong(m.n_incidences), p(m.owner), p(m.neighbour), p(m.cell_face_offsets),
        p(m.cell_faces), p(m.incidence_side), *(p(a) for a in soa), p(y),
    )  # fmt: skip
    return y.reshape(5, m.n_cells).T


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("policy", PRECISIONS)
@pytest.mark.parametrize("kind", ["tet", "hex", "poly", "hybrid"])
def test_block_product_equals_the_assembled_matrix(
    tmp_path: Path, backend: str, policy: str, kind: str
) -> None:
    if backend == "metal":
        pytest.skip("kernels/blocks.py is not generated for Metal (E1013, see the inverse test)")
    m = s.mesh(meshes.build(kind, 4, 0.15, seed=9))
    nc, ni = m.n_cells, m.n_interior
    rng = np.random.default_rng(9)
    face_jac = rng.standard_normal((m.n_faces, 2, 5, 5))
    face_jac[ni:] = np.nan  # boundary faces have no face blocks: never read
    diag, x = rng.standard_normal((nc, 5, 5)), rng.standard_normal((nc, 5))
    if policy != "fp64":
        face_jac = face_jac.astype(np.float32).astype(np.float64)  # Jac is stored in FP32
    # The matrix of physics BlockMatrix: per interior face, A(owner, neighbour) = face_jac[1],
    # A(neighbour, owner) = -face_jac[0].
    a = np.zeros((nc, 5, nc, 5))
    for c in range(nc):
        a[c, :, c, :] = diag[c]
    for f in range(ni):
        o, nb = m.owner[f], m.neighbour[f]
        a[o, :, nb, :] += face_jac[f, 1]
        a[nb, :, o, :] -= face_jac[f, 0]
    a = a.reshape(5 * nc, 5 * nc)
    expect = (a @ x.ravel()).reshape(nc, 5)
    bound = 64 * eps(policy) * (np.abs(a) @ np.abs(x.ravel())).reshape(nc, 5)
    host = _multiply(_build_multiply(tmp_path, "cpu", policy), m, face_jac, diag, x)
    assert (np.abs(host - expect) <= bound).all()
    if backend == "gpu":
        lib = _build_multiply(tmp_path, "gpu", policy)
        dev = _multiply(lib, m, face_jac, diag, x)
        assert (np.abs(dev - expect) <= bound).all()
        np.testing.assert_array_equal(_multiply(lib, m, face_jac, diag, x), dev)
