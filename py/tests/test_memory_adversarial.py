"""The memory estimate bounds what the runtime allocates for every input the contract accepts,
padded files included: docs/sdk/plan.md, #358 (memory gate, round 9)."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import autan
import autan.initial_state
import pytest

from tests.conftest import ROOT
from tests.test_solver_backend import HEX, MESH_INFO, RUNTIME

pytestmark = pytest.mark.skipif(
    None in (RUNTIME, MESH_INFO),
    reason="autan-run and autan-mesh-info not built (cmake --build build)",
)

HYBRID = ROOT / "tests" / "data" / "mesh" / "hybrid"
PAD = ROOT / "tests" / "io" / "pad_files.py"
PADDED = 64 * 2**20
VARIABLES = ["density", "velocity-x", "velocity-y", "velocity-z", "pressure"]
FREESTREAM = [1.2, 68.0, 0.0, 0.0, 101325.0]
FARFIELD = "case.boundary('*').farfield()\n"


def pad(kind: str, source: Path, out: Path, *extra: str) -> Path:
    subprocess.run(
        [sys.executable, str(PAD), kind, str(source), str(out), str(PADDED), *extra], check=True
    )
    return out


def mesh_info(mesh: Path, tmp_path: Path) -> autan.MeshInfo:
    assert MESH_INFO is not None
    path = tmp_path / "mesh.meshinfo.json"
    path.write_text(
        subprocess.run(
            [str(MESH_INFO), "--json", "--scale", "1", str(mesh)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return autan.MeshInfo.load(path)


def peak(run: autan.Results) -> int:
    """The run's host peak (VmHWM), from its `memory` log event (results.md#log)."""
    peaks = [
        entry["data"]["host_peak_bytes"]
        for entry in map(json.loads, (run.path / "log.jsonl").read_text().splitlines())
        if entry["source"] == "solver" and "host_peak_bytes" in (entry["data"] or {})
    ]
    assert len(peaks) == 1
    value: int = peaks[0]
    return value


def case_file(directory: Path, mesh: Path, boundaries: str) -> Path:
    case = directory / "case.py"
    case.write_text(
        "import autan\nfrom autan.units import K, Pa\n"
        f"case = autan.Case('padded', mesh={str(mesh)!r})\n"
        "case.physics = autan.Compressible(turbulence=autan.Inviscid())\n"
        "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
        "case.time = autan.Steady(max_iterations=3, linear_solver='gauss-seidel')\n" + boundaries
    )
    return case


def padded_gmsh(tmp_path: Path) -> tuple[Path, str]:
    return pad("gmsh", HYBRID / "hybrid.ascii.msh", tmp_path / "padded.msh"), FARFIELD


def padded_polymesh(tmp_path: Path) -> tuple[Path, str]:
    return pad("polymesh", HYBRID / "polyMesh", tmp_path / "polyMesh"), FARFIELD


def padded_cgns(tmp_path: Path) -> tuple[Path, str]:
    return pad("cgns", HYBRID / "hybrid.cgns", tmp_path / "padded.cgns"), FARFIELD


def padded_initial_state(tmp_path: Path) -> tuple[Path, str]:
    info = mesh_info(HEX, tmp_path)
    centroids = [
        ((i + 0.5) / 3, (j + 0.5) / 3, (k + 0.5) / 3)
        for k in range(3)
        for j in range(3)
        for i in range(3)
    ]
    autan.initial_state.write(
        tmp_path / "state.json",
        oracle="uniform",
        mesh_sha256=info.source.sha256,
        centroids=centroids,
        values=[FREESTREAM for _ in centroids],
    )
    padded = pad("json", tmp_path / "state.json", tmp_path / "padded.json")
    digest = hashlib.sha256(padded.read_bytes()).hexdigest()
    return HEX, FARFIELD + (
        f"case.initial_state = autan.StateFile(path='padded.json', sha256={digest!r})\n"
    )


def padded_face_states(tmp_path: Path) -> tuple[Path, str]:
    info = mesh_info(HEX, tmp_path)
    centroids = [[0.0, (j + 0.5) / 3, (k + 0.5) / 3] for k in range(3) for j in range(3)]
    document = {
        "schema": "autan.face-states/1",
        "oracle": "uniform",
        "mesh_sha256": info.source.sha256,
        "variables": VARIABLES,
        "zones": {"xmin": {"faces": 9, "centroids": centroids, "values": [FREESTREAM] * 9}},
    }
    (tmp_path / "states.json").write_text(json.dumps(document))
    padded = pad(
        "json", tmp_path / "states.json", tmp_path / "padded.json", "--empty-zone", "unused"
    )
    digest = hashlib.sha256(padded.read_bytes()).hexdigest()
    return HEX, (
        "for zone in ('xmax', 'ymin', 'ymax', 'zmin', 'zmax'):\n"
        "    case.boundary(zone).farfield()\n"
        f"case.boundary('xmin').oracle_state(path='padded.json', sha256={digest!r})\n"
    )


@pytest.mark.parametrize(
    "make",
    [padded_gmsh, padded_polymesh, padded_cgns, padded_initial_state, padded_face_states],
)
def test_the_estimate_bounds_a_run_on_padded_files(
    make: Callable[[Path], tuple[Path, str]], tmp_path: Path
) -> None:
    # plan.md: each file of the case counts at its size, whatever it carries beyond the mesh
    # and the states (skipped sections, unreferenced nodes, descriptors, whitespace, empty
    # zones): the measured peak (VmHWM, CPU) never exceeds the estimate.
    mesh, boundaries = make(tmp_path)
    info = mesh_info(mesh, tmp_path)
    case = autan.load(case_file(tmp_path, mesh, boundaries))
    estimate = case.plan(info, device_kind="cpu").memory.bytes_per_gpu
    run = case.run(info, backend=autan.SolverBackend(executable=RUNTIME), runs=tmp_path / "runs")
    assert run.run.status in ("converged", "completed")
    assert estimate >= peak(run), (estimate, peak(run))
