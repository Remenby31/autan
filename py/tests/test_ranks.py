"""Runs on several devices: one rank each, started by an MPI launcher (solver-input.md,
``devices``)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import autan
import pytest

from tests.conftest import CRM_HL, mesh_document


def test_several_devices_start_one_rank_each(
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = tmp_path / "arguments"
    launcher = tmp_path / "mpiexec"
    launcher.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {arguments}\nexit 2\n')
    launcher.chmod(0o755)
    runtime = tmp_path / "autan-run"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("AUTAN_MPIEXEC", str(launcher))
    backend = autan.SolverBackend(devices=autan.Devices(kind="cuda", count=4), executable=runtime)
    with pytest.raises(autan.CaseError):
        autan.load(write_case(CRM_HL)).run(crm_mesh, backend=backend, runs=tmp_path / "runs")
    assert arguments.read_text().splitlines()[:3] == ["-n", "4", str(runtime)]


def test_several_devices_need_a_launcher(
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "autan-run"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    monkeypatch.delenv("AUTAN_MPIEXEC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no mpiexec, no mpirun
    backend = autan.SolverBackend(devices=autan.Devices(kind="cuda", count=2), executable=runtime)
    with pytest.raises(autan.CaseError) as error:
        autan.load(write_case(CRM_HL)).run(crm_mesh, backend=backend, runs=tmp_path / "runs")
    assert error.value.codes == ["E0808"]


def test_an_explicit_relative_launcher_is_the_one_in_the_working_directory(
    write_case: Callable[..., Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = tmp_path / "arguments"
    (tmp_path / "mpiexec").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {arguments}\nexit 2\n')
    (tmp_path / "mpiexec").chmod(0o755)
    runtime = tmp_path / "autan-run"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTAN_MPIEXEC", "./mpiexec")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # another mpiexec there must not be taken
    backend = autan.SolverBackend(devices=autan.Devices(kind="cuda", count=2), executable=runtime)
    with pytest.raises(autan.CaseError):
        autan.load(write_case(CRM_HL)).run(crm_mesh, backend=backend, runs=tmp_path / "runs")
    assert arguments.read_text().splitlines()[:2] == ["-n", "2"]


SCRIPT = """
import sys, autan, autan.solver_backend as backend
from pathlib import Path
backend.GRACE, backend.DRAIN = 0.2, 0.5
case, mesh = autan.load(sys.argv[1]), autan.MeshInfo.load(sys.argv[2])
devices = autan.Devices(kind="cuda", count=2)
try:
    case.run(mesh, backend=autan.SolverBackend(devices=devices, executable=Path(sys.argv[3])),
             runs=Path(sys.argv[4]))
except KeyboardInterrupt:
    sys.exit(130)
"""


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid (util-linux)")
def test_a_stuck_launcher_cannot_keep_the_sdk_waiting(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    started = tmp_path / "started"
    # Ignores SIGINT and SIGTERM, and leaves a child in another session holding the pipes.
    launcher = tmp_path / "mpiexec"
    launcher.write_text(
        "#!/bin/sh\ntrap '' INT TERM\n"
        f"setsid sleep 60 &\ntouch {started}\nwhile true; do sleep 1; done\n"
    )
    launcher.chmod(0o755)
    runtime = tmp_path / "autan-run"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    meshinfo = tmp_path / "crm.meshinfo.json"
    meshinfo.write_text(json.dumps(mesh_document("crm-hl")))
    env = {**os.environ, "AUTAN_MPIEXEC": str(launcher)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            SCRIPT,
            str(write_case(CRM_HL)),
            str(meshinfo),
            str(runtime),
            str(tmp_path / "runs"),
        ],
        env=env,
    )
    try:
        deadline = time.monotonic() + 20
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        begin = time.monotonic()
        assert process.wait(timeout=10) == 130
        assert time.monotonic() - begin < 3  # GRACE + DRAIN, not the child's 60 s
    finally:
        process.kill()
