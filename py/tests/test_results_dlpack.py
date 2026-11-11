"""Every array of the results API exports through DLPack without a copy, on run directories the
runtime wrote (docs/sdk/results.md; acceptance.md, M4)."""

import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any

import autan
import numpy as np
import pytest

from tests.test_solver_backend import CASE, MESH, MESH_INFO, RELAXATION

pytestmark = pytest.mark.skipif(
    None in (RELAXATION, MESH_INFO), reason="autan-run-relaxation and autan-mesh-info not built"
)


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, time: str, **knobs: str
) -> autan.Results:
    assert MESH_INFO is not None
    case = tmp_path / f"{name}.py"
    case.write_text(CASE + f"case.time = {time}\ncase.monitor(autan.forces('*'), every=1)\n")
    meshinfo = tmp_path / f"{name}.meshinfo.json"
    meshinfo.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
        ).stdout
    )
    with monkeypatch.context() as env:
        for knob in [k for k in os.environ if k.startswith("AUTAN_TEST_")]:
            env.delenv(knob)
        for knob, value in knobs.items():
            env.setenv(knob, value)
        runs = tmp_path / name
        # A diverged run is reported as an error; its directory is what is read.
        with contextlib.suppress(autan.CaseError):
            autan.load(case).run(
                autan.MeshInfo.load(meshinfo),
                backend=autan.SolverBackend(executable=RELAXATION),
                runs=runs,
            )
    [run] = runs.glob("run-*")
    return autan.Results.open(run)


def _shares(x: Any) -> None:
    assert isinstance(x, np.ndarray)
    assert x.flags.c_contiguous
    assert x.__dlpack_device__() == (1, 0)
    y = np.from_dlpack(x)
    assert y.__array_interface__["data"][0] == x.__array_interface__["data"][0]
    written = (not x[0]) if x.dtype == np.bool_ else x[0] + 1
    y[0] = written
    assert x[0] == written


def test_every_results_array_exports_without_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steady = _run(tmp_path, monkeypatch, "steady", "autan.Steady(max_iterations=5)")
    cell_zone = steady.mesh_info.cell_zones[0].name
    _, wall = steady.wall("Pressure")
    arrays: dict[str, Any] = {
        "field": steady.field("Density"),
        "wall faces": steady.wall("Pressure")[0],
        "wall values": wall,
        "cell zone": steady.cell_zone(cell_zone),
        "iterations": steady.history.iterations,
        "residual": steady.history.residual(steady.history.equations[0]),
        "provisional": steady.history.provisional,
        **{f"monitor {k}": v for k, v in steady.history.monitor("forces").items()},
        **{f"forces {q}": getattr(steady.forces, q) for q in steady.forces.quantities},
    }
    unsteady = _run(
        tmp_path,
        monkeypatch,
        "unsteady",
        "autan.Unsteady(steps=8, dt=1e-3 * autan.units.s, average_from=2)",
    )
    arrays["mean field"] = unsteady.field("Density", mean=True)
    diverged = _run(
        tmp_path,
        monkeypatch,
        "diverged",
        "autan.Steady(max_iterations=20, checkpoint=autan.Checkpoint(iterations=1))",
        AUTAN_TEST_FACTOR="1.0",
        AUTAN_TEST_DIVERGE_AT="6",
    )
    assert diverged.run.status == "diverged"
    diverged.checkpoint()
    arrays["checkpoint field"] = diverged.checkpoint_field("Density")
    for name, x in arrays.items():
        _shares(x)
        assert x.size, name
    # A field is read once and kept: a write through DLPack shows in later calls.
    field = steady.field("Pressure")
    np.from_dlpack(field)[0] = -1.0
    assert steady.field("Pressure")[0] == -1.0
