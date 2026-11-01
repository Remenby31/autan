"""The SDK through the real solver runtime: `autan prepare` output into `autan-run`, its
run directory back through `autan.Results`.

`autan-run` as shipped registers no physics; `autan-run-relaxation` (tests/runtime) is the
same runtime with the core's test relaxation physics, which runs every case to the end.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import autan
import pytest
from autan.cli import main

from tests.conftest import ROOT

MESH = ROOT / "tests" / "data" / "mesh" / "hybrid" / "polyMesh"


def _binary(variable: str, built: Path) -> Path | None:
    candidate = os.environ.get(variable) or (str(built) if built.is_file() else None)
    return Path(candidate) if candidate else None


RUNTIME = _binary("AUTAN_RUN", ROOT / "build" / "runtime" / "autan-run")
RELAXATION = _binary(
    "AUTAN_RUN_RELAXATION", ROOT / "build" / "tests" / "runtime" / "autan-run-relaxation"
)
MESH_INFO = _binary("AUTAN_MESH_INFO", ROOT / "build" / "io" / "autan-mesh-info")
pytestmark = pytest.mark.skipif(
    None in (RUNTIME, RELAXATION, MESH_INFO),
    reason="autan-run, autan-run-relaxation and autan-mesh-info not built "
    "(cmake --build build --target autan-run autan-run-relaxation autan-mesh-info)",
)
CASE = (
    "import autan\nfrom autan.units import K, Pa\n"
    f"case = autan.Case('hybrid', mesh={str(MESH)!r})\n"
    "case.physics = autan.Compressible(turbulence=autan.Laminar())\n"
    "case.freestream(mach=0.2, pressure=101325 * Pa, temperature=300 * K)\n"
    "for zone in ('xmin', 'xmax', 'ymax', 'zmin'):\n"
    "    case.boundary(zone).farfield()\n"
)


@pytest.fixture
def hybrid(tmp_path: Path) -> tuple[autan.Case, autan.MeshInfo]:
    assert MESH_INFO is not None
    info = subprocess.run(
        [str(MESH_INFO), "--json", str(MESH)], capture_output=True, text=True, check=True
    ).stdout
    path = tmp_path / "hybrid.py"
    path.write_text(CASE)
    meshinfo = tmp_path / "hybrid.meshinfo.json"
    meshinfo.write_text(info)
    return autan.load(path), autan.MeshInfo.load(meshinfo)


def test_the_shipped_runtime_reports_missing_physics(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    case, mesh = hybrid
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(executable=RUNTIME), runs=tmp_path / "runs")
    [found] = error.value.diagnostics
    assert found.code == "E0706"
    assert found.fix
    assert found.doc.endswith("#e0706")
    assert not any((tmp_path / "runs").iterdir())


def test_a_real_run_reads_back_through_the_results_contract(
    hybrid: tuple[autan.Case, autan.MeshInfo], tmp_path: Path
) -> None:
    case, mesh = hybrid
    backend = autan.SolverBackend(executable=RELAXATION, deterministic=True)
    result = case.run(mesh, backend=backend, runs=tmp_path / "runs")
    run = result.run
    assert run.status in ("converged", "completed")
    assert run.solver.backend == "cpu"
    assert run.case.sha256 == case.spec.hash()
    assert run.mesh.sha256 == mesh.source.sha256
    parts = [
        run.case.sha256,
        run.mesh.sha256,
        run.solver.commit,
        run.precision,
        run.solver.backend,
        run.devices.kind,
        str(run.devices.count),
    ]
    assert run.identity == hashlib.sha256("\n".join(parts).encode()).hexdigest()
    assert result.history.iterations[-1] == run.iterations
    assert result.field("Density").size == mesh.counts.cells


def test_divergence_and_interruption_come_back_as_the_contract_says(
    hybrid: tuple[autan.Case, autan.MeshInfo],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, mesh = hybrid
    monkeypatch.setenv("AUTAN_TEST_DIVERGE_AT", "3")
    result = case.run(
        mesh, backend=autan.SolverBackend(executable=RELAXATION), runs=tmp_path / "diverged"
    )
    assert result.run.status == "diverged"
    assert [d.code for d in result.run.diagnostics] == ["D0107"]


@pytest.mark.usefixtures("hybrid")
def test_cli_runs_the_runtime_and_stops_it_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path = tmp_path / "hybrid.py"
    info = tmp_path / "hybrid.meshinfo.json"
    monkeypatch.setenv("AUTAN_RUN", str(RELAXATION))
    argv = [
        "run",
        str(case_path),
        "--mesh-info",
        str(info),
        "--backend",
        "cpu",
        "--runs",
        str(tmp_path / "runs"),
        "--json",
    ]
    assert main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["solver"]["backend"] == "cpu"

    env = {**os.environ, "AUTAN_RUN": str(RELAXATION), "AUTAN_TEST_PAUSE_MS": "50"}
    command = [str(Path(sys.executable).with_name("autan")), *argv[:-1]]
    command[command.index(str(tmp_path / "runs"))] = str(tmp_path / "stopped")
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not list(
        (tmp_path / "stopped").glob("run-*/history.jsonl")
    ):
        time.sleep(0.02)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == 130
    [run_dir] = (tmp_path / "stopped").glob("run-*")
    assert autan.Results.open(run_dir).run.status == "stopped"


def test_missing_runtime_is_e0804(
    hybrid: tuple[autan.Case, autan.MeshInfo],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, mesh = hybrid
    monkeypatch.setenv("AUTAN_RUN", str(tmp_path / "nowhere"))
    with pytest.raises(autan.CaseError) as error:
        case.run(mesh, backend=autan.SolverBackend(), runs=tmp_path / "runs")
    assert error.value.codes == ["E0804"]
