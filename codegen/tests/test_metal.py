"""Metal backend of the generator: what it refuses, through the command line.

Execution on an Apple GPU is covered by `pytest tests/test_gpu.py --backend metal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen.__main__ import main
from conftest import ROOT

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

KERNEL = "kernels/advection.py"


def test_metal_refuses_fp64_with_its_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [str(ROOT / KERNEL), "--out", str(tmp_path), "--backends", "metal"]
    assert main([*args, "--policies", "fp64"]) == 1
    assert "E1013" in capsys.readouterr().err
    assert main([*args, "--policies", "fp32,fp32+acc64"]) == 0
    assert (tmp_path / "advection_metal.hpp").exists()
