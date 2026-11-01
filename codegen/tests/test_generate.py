"""Generated C++ compiles for shapes the frontend accepts, with tangents and adjoints."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]

MODULE = '''
from autan_codegen.dsl import Geom, Grad, Max, Min, Residual, State, Sum, cell_reduce, cells, faces, geometry, outer, pointwise

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
'''

UNIT = """
#include "state_gradient_cpu.hpp"
template void autan::kernels::state_gradient::cpu::gradient<autan::kernels::fp32_acc64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp32_acc64>&);
template void autan::kernels::state_gradient::cpu::gradient_tangent<autan::kernels::fp64>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp64>&);
template void autan::kernels::state_gradient::cpu::gradient_adjoint<autan::kernels::fp32>(
    const autan::kernels::state_gradient::Data<autan::kernels::fp32>&);
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_a_reduction_of_5x3_gradients_compiles_with_its_derivatives(tmp_path: Path) -> None:
    src = tmp_path / "state_gradient.py"
    src.write_text(MODULE)
    out = tmp_path / "gen"
    generate(src, out, ["cpu", "gpu"])
    header = (out / "state_gradient.hpp").read_text()
    assert "grad;  // cells, 15 component(s)" in header
    unit = tmp_path / "unit.cpp"
    unit.write_text(UNIT)
    subprocess.run(
        ["c++", "-std=c++20", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT / 'codegen' / 'include'}", f"-I{out}", str(unit)],
        check=True,
    )  # fmt: skip
