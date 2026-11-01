"""Metal backend of the generator: what it refuses, and the Metal compiler accepting what it
emits for every DSL shape. Execution on an Apple GPU: `pytest tests/test_gpu.py --backend metal`.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import autan_spike as s
import pytest
from autan_codegen import model
from autan_codegen.__main__ import main
from autan_codegen.derive import derive
from autan_codegen.emit_metal import POLICIES as METAL_POLICIES
from autan_codegen.emit_metal import emit_msl
from conftest import ROOT

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


@pytest.mark.gpu
def test_every_dsl_shape_compiles_to_metal(backend: str, tmp_path: Path) -> None:
    # The module of test_generate.py (Mat53, outer, rel inside a reduction, a boundary loop
    # writing face data, all derivatives) compiled by the Metal compiler of the device.
    if backend != "metal":
        pytest.skip("needs --backend metal")
    from test_generate import MODULE

    src = tmp_path / "state_gradient.py"
    src.write_text(MODULE)
    m = model.load(src)
    derive(m)
    text = emit_msl(m, list(METAL_POLICIES))
    err = ctypes.create_string_buffer(4096)
    rc = s.lib().autan_spike_metal_compile(b"state_gradient", text.encode(), err, len(err))
    assert rc == 0, err.value.decode()
