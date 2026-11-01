"""Generated C++ compiles for shapes the frontend accepts, with tangents and adjoints."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]

MODULE = """
from autan_codegen.dsl import Geom, Grad, Max, Min, Residual, State, Sum, cell_reduce, cells, faces, geometry, outer, pointwise, rel, Vec3, Vec5, Write, boundary_loop, dot

u = cells.field("u", State.Vec5)
u_b = faces.field("u_b", State.Vec5)
grad = cells.field("grad", Grad.Mat53)
lo = cells.field("lo", State.Vec5)
hi = cells.field("hi", State.Vec5)


@pointwise
def term(uc: State.Vec5, un: State.Vec5, w: Geom.Vec3) -> tuple[Grad.Mat53, State.Vec5, State.Vec5]:
    return outer(un - uc, w), un, un


gradient = cell_reduce(term, uc=u.self_, un=u.other(boundary=u_b), w=geometry.lsq_weight,
                       acc=[Sum(grad), Min(lo), Max(hi)])

phi = cells.field("phi", State.Vec5)


@pointwise
def to_face(g: Grad.Mat53, xf: Geom.Vec3, xc: Geom.Vec3) -> State.Vec5:
    return g @ rel(xf, xc)


# Face data of the incidence's face inside a reduction (limiter: grad . (x_f - x_c)).
extrapolation = cell_reduce(to_face, g=grad.self_, xf=geometry.face_centroid,
                            xc=geometry.cell_centroid.self_, acc=[Min(phi)])


@pointwise
def mirror(ui: State.Vec5, n: Geom.Vec3) -> State.Vec5:
    m = Vec3(ui[1], ui[2], ui[3])
    mn = dot(m, n)
    return Vec5(ui[0], ui[1] - 2.0 * mn * n[0], ui[2] - 2.0 * mn * n[1], ui[3] - 2.0 * mn * n[2], ui[4])


# Boundary states written per face, then read by the reduction above through u.other(u_b).
boundary_states = boundary_loop(mirror, ui=u.left, n=geometry.face_normal, out=Write(u_b))
"""

UNIT = """
#include "state_gradient_cpu.hpp"
template void autan::kernels::state_gradient::cpu::gradient<autan::kernels::fp32_acc64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp32_acc64>&);
template void autan::kernels::state_gradient::cpu::gradient_tangent<autan::kernels::fp64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp64>&);
template void autan::kernels::state_gradient::cpu::gradient_adjoint<autan::kernels::fp32>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp32>&);
template void autan::kernels::state_gradient::cpu::extrapolation_adjoint<autan::kernels::fp64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp64>&);
template void autan::kernels::state_gradient::cpu::boundary_states<autan::kernels::fp32_acc64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp32_acc64>&);
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_state_gradients_face_data_and_boundary_states_compile_with_their_derivatives(
    tmp_path: Path,
) -> None:
    src = tmp_path / "state_gradient.py"
    src.write_text(MODULE)
    out = tmp_path / "gen"
    generate(src, out, ["cpu", "gpu"])
    header = (out / "state_gradient.hpp").read_text()
    assert "grad;  // cells, 15 component(s)" in header
    cpu = (out / "state_gradient_cpu.hpp").read_text()
    assert "d.face_centroid[f]" in cpu
    assert "d.u_b[f] = " in cpu
    assert "boundary_states_tangent" not in cpu
    unit = tmp_path / "unit.cpp"
    unit.write_text(UNIT)
    subprocess.run(
        ["c++", "-std=c++20", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT / 'codegen' / 'include'}", f"-I{out}", str(unit)],
        check=True,
    )  # fmt: skip


@pytest.mark.gpu
def test_state_gradient_module_compiles_for_cuda(tmp_path: Path) -> None:
    src = tmp_path / "state_gradient.py"
    src.write_text(MODULE)
    out = tmp_path / "gen"
    generate(src, out, ["gpu"])
    nvcc = shutil.which("nvcc")
    assert nvcc is not None, "the GPU backend tests need nvcc on PATH"
    subprocess.run(
        [nvcc, "-std=c++20", "-c", f"-I{ROOT / 'codegen' / 'include'}", f"-I{out}",
         str(out / "state_gradient_gpu.cu"), "-o", str(tmp_path / "unit.o")],
        check=True,
    )  # fmt: skip
