"""Runs on several devices: one rank each, started by an MPI launcher (solver-input.md,
``devices``)."""

from collections.abc import Callable
from pathlib import Path

import autan
import pytest

from tests.conftest import CRM_HL


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
