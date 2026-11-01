"""Metal backend of the generator: what it refuses, and the Metal compiler accepting what it
emits for every kernel module and DSL shape. Execution on an Apple GPU: `pytest tests/test_gpu.py --backend metal`.
"""

from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING

import autan_spike as s
import numpy as np
import pytest
from autan_codegen import model
from autan_codegen.__main__ import main
from autan_codegen.derive import derive
from autan_codegen.emit_metal import POLICIES as METAL_POLICIES
from autan_codegen.emit_metal import emit_msl
from conftest import ROOT
from test_generate import MODULE

if TYPE_CHECKING:
    from pathlib import Path

KERNEL = "kernels/advection.py"


def test_metal_refuses_fp64_with_its_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [str(ROOT / KERNEL), "--out", str(tmp_path), "--backends", "metal"]
    assert main([*args, "--policies", "fp64"]) == 1
    assert "E1013" in capsys.readouterr().err
    assert main([*args, "--policies", "fp32,fp32+acc64"]) == 0
    assert (tmp_path / "advection_metal.hpp").exists()


# Every kernel module of the solver, and the module of test_generate.py (Mat53, outer, rel inside
# a reduction, a boundary loop writing face data), with all derivatives.
SOURCES = [*sorted(p.name for p in (ROOT / "kernels").glob("*.py")), "test_generate.MODULE"]


@pytest.mark.gpu
@pytest.mark.parametrize("source", SOURCES)
def test_every_kernel_module_compiles_to_metal(backend: str, tmp_path: Path, source: str) -> None:
    if backend != "metal":
        pytest.skip("needs --backend metal")
    if source == "test_generate.MODULE":
        path = tmp_path / "state_gradient.py"
        path.write_text(MODULE)
    else:
        path = ROOT / "kernels" / source
    m = model.load(path)
    derive(m)
    text = emit_msl(m, list(METAL_POLICIES))
    err = ctypes.create_string_buffer(4096)
    rc = s.lib().autan_spike_metal_compile(path.stem.encode(), text.encode(), err, len(err))
    assert rc == 0, err.value.decode()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "x",
    [
        [1e-38, 3e-39, 1e-45],  # subnormal FP32 entries: Apple GPUs flush them as operands
        [3e38, -3e38, 1e30],  # squares beyond FP32's range
        list(np.linspace(-1.0, 1.0, 70_001)),  # several chunks and partials
    ],
)
def test_metal_norm_of_fp32_data_is_exact_to_rounding(backend: str, x: list[float]) -> None:
    if backend != "metal":
        pytest.skip("needs --backend metal")
    xs = np.array(x, dtype=np.float32)
    out = ctypes.c_double()
    err = ctypes.create_string_buffer(512)
    rc = s.lib().autan_spike_metal_norm_f32(
        ctypes.c_void_p(xs.ctypes.data), ctypes.c_int64(xs.size), ctypes.byref(out), err, len(err)
    )
    assert rc == 0, err.value.decode()
    ref = math.sqrt(math.fsum(float(v) ** 2 for v in xs))
    # Double-float sums: relative error of order 2^-46 per operation.
    assert abs(out.value - ref) <= 1e-12 * ref
