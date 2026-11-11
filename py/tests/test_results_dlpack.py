"""Every array of the results API exports through DLPack without a copy, on run directories the
shipped runtime wrote (docs/sdk/results.md; acceptance.md, M4)."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import autan
import numpy as np
import pytest

from tests.test_solver_backend import CASE, MESH, MESH_INFO, RUNTIME

pytestmark = pytest.mark.skipif(
    None in (RUNTIME, MESH_INFO), reason="autan-run and autan-mesh-info not built"
)

# results.md: the FlowSolution fields of a compressible run, the WallSolution fields, the
# History equations of a laminar run, and a forces monitor's quantities with ref_area and
# ref_length set.
FIELDS = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure", "Temperature")
WALL = ("Pressure", "WallShearStressX", "WallShearStressY", "WallShearStressZ")
EQUATIONS = ("density", "momentum-x", "momentum-y", "momentum-z", "energy")
FORCES = ("iteration", "fx", "fy", "fz", "mx", "my", "mz", "cd", "cs", "cl", "cmx", "cmy", "cmz")
MONITORED = (
    CASE.replace("temperature=300 * K)", "temperature=300 * K, ref_length=1.0 * autan.units.m)")
    + "case.monitor(autan.forces('*', ref_area=1.0 * autan.units.m**2), every=1)\n"
)
RUN = (
    "import sys, autan\nfrom pathlib import Path\n"
    "case, mesh = autan.load(sys.argv[1]), autan.MeshInfo.load(sys.argv[2])\n"
    "case.run(mesh, backend=autan.SolverBackend(executable=Path(sys.argv[3])), "
    "runs=Path(sys.argv[4]))\n"
)


def _inputs(tmp_path: Path, name: str, time_spec: str) -> tuple[Path, Path]:
    assert MESH_INFO is not None
    case = tmp_path / f"{name}.py"
    case.write_text(MONITORED + f"case.time = {time_spec}\n")
    meshinfo = tmp_path / f"{name}.meshinfo.json"
    meshinfo.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
        ).stdout
    )
    return case, meshinfo


def _run(tmp_path: Path, name: str, time_spec: str) -> autan.Results:
    case, meshinfo = _inputs(tmp_path, name, time_spec)
    return autan.load(case).run(
        autan.MeshInfo.load(meshinfo),
        backend=autan.SolverBackend(executable=RUNTIME),
        runs=tmp_path / name,
    )


# A run killed while it runs, once it has written a checkpoint: its directory keeps it.
def _killed(tmp_path: Path) -> autan.Results:
    case, meshinfo = _inputs(
        tmp_path,
        "killed",
        "autan.Steady(max_iterations=1000000, residual_orders=15, "
        "checkpoint=autan.Checkpoint(iterations=20))",
    )
    runs = tmp_path / "killed"
    writer = subprocess.Popen(
        [sys.executable, "-c", RUN, str(case), str(meshinfo), str(RUNTIME), str(runs)],
        start_new_session=True,
    )
    groups = [writer.pid]
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            found = list(runs.glob("run-*/run.json"))
            if found and json.loads(found[0].read_text()).get("checkpoint"):
                break
            time.sleep(0.05)
        children = subprocess.run(
            ["pgrep", "-P", str(writer.pid)], capture_output=True, text=True, check=False
        )
        groups += [int(pid) for pid in children.stdout.split()]
    finally:
        for group in groups:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
        writer.wait()
    [run] = runs.glob("run-*")
    return autan.Results.open(run)


def _shares(name: str, x: Any) -> None:
    assert isinstance(x, np.ndarray), name
    assert x.size, name
    assert x.flags.c_contiguous, name
    assert x.__dlpack_device__() == (1, 0), name
    y = np.from_dlpack(x)
    assert y.__array_interface__["data"][0] == x.__array_interface__["data"][0], name
    written = (not x[0]) if x.dtype == np.bool_ else x[0] + 1
    y[0] = written
    assert x[0] == written, name


def test_every_results_array_exports_without_a_copy(tmp_path: Path) -> None:
    steady = _run(tmp_path, "steady", "autan.Steady(max_iterations=5)")
    unsteady = _run(
        tmp_path, "unsteady", "autan.Unsteady(steps=8, dt=1e-4 * autan.units.s, average_from=2)"
    )
    killed = _killed(tmp_path)
    killed.checkpoint()
    monitor = steady.history.monitor("forces")
    assert set(monitor) == set(FORCES)
    assert set(steady.forces.quantities) == set(FORCES)
    arrays: dict[str, Any] = {
        "iterations": steady.history.iterations,
        "provisional": steady.history.provisional,
        "cell zone": steady.cell_zone(steady.mesh_info.cell_zones[0].name),
    }
    for field in FIELDS:
        arrays[f"field {field}"] = steady.field(field)
        arrays[f"mean field {field}"] = unsteady.field(field, mean=True)
        arrays[f"checkpoint field {field}"] = killed.checkpoint_field(field)
    for name in WALL:
        faces, values = steady.wall(name)
        arrays[f"wall faces {name}"] = faces
        arrays[f"wall {name}"] = values
    for equation in EQUATIONS:
        arrays[f"residual {equation}"] = steady.history.residual(equation)
    for quantity in FORCES:
        arrays[f"monitor {quantity}"] = monitor[quantity]
        arrays[f"forces {quantity}"] = getattr(steady.forces, quantity)
    for name, x in arrays.items():
        _shares(name, x)
