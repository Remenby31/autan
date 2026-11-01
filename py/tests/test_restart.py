"""Restarting from a previous run (solver-input.md#start, results.md#runjson).

The oracle of `continue` is the uninterrupted run: continuing a stopped run must give
the history the run would have had without the stop.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import autan
import jsonschema
import pytest
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, FIXTURES


def history(result: autan.Results) -> list[str]:
    return (result.path / "history.jsonl").read_text().splitlines()


@pytest.fixture
def case(write_case: Callable[[str], Path]) -> autan.Case:
    return autan.load(write_case(CRM_HL))


def test_continue_extends_the_run_as_if_it_never_stopped(
    case: autan.Case, crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    whole = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "whole")
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path / "parts")
    assert stopped.run.status == "stopped"
    resumed = case.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path / "parts", start=stopped
    )
    assert history(stopped) + history(resumed) == history(whole)
    assert (resumed.run.status, resumed.run.iterations) == (whole.run.status, 100)

    start = resumed.run.start
    assert start is not None
    solution = (stopped.path / "solution.cgns").read_bytes()
    assert (start.run_id, start.identity, start.mode, start.iterations) == (
        stopped.run.run_id,
        stopped.run.identity,
        "continue",
        30,
    )
    assert start.solution_sha256 == hashlib.sha256(solution).hexdigest()
    run = resumed.run
    parts = [
        run.case.sha256,
        run.mesh.sha256,
        run.solver.commit,
        run.precision,
        run.solver.backend,
        run.devices.kind,
        str(run.devices.count),
        start.identity,
        "30",
        start.solution_sha256,
    ]
    assert run.identity == hashlib.sha256("\n".join(parts).encode()).hexdigest()


def test_warm_start_counts_from_one(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    previous = autan.load(write_case(CRM_HL)).run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path
    )
    other = autan.load(write_case(CRM_HL.replace("mach=0.20", "mach=0.25"), name="other.py"))
    warm = other.run(
        crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=previous, mode="warm"
    )
    assert history(warm)[0].startswith('{"iteration": 1,')
    assert warm.run.start is not None
    assert warm.run.start.mode == "warm"


@pytest.mark.parametrize(
    ("change", "mode", "code"),
    [
        (lambda body: body.replace("mach=0.20", "mach=0.25"), "continue", "E0708"),
        (lambda body: body + "case.time = autan.Steady(max_iterations=30)\n", "continue", "E0708"),
        (lambda body: body.replace("autan.SA()", "autan.SST()"), "warm", "E0708"),
    ],
)
def test_a_start_must_match_the_case(
    change: Callable[[str], str],
    mode: str,
    code: str,
    write_case: Callable[[str], Path],
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
) -> None:
    previous = autan.load(write_case(CRM_HL)).run(
        crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path
    )
    changed = autan.load(write_case(change(CRM_HL), name="changed.py"))
    with pytest.raises(autan.CaseError) as error:
        changed.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=previous, mode=mode)  # type: ignore[arg-type]
    assert error.value.codes == [code]


def test_only_finished_runs_on_the_same_mesh_can_be_resumed(
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    mesh_variant: Callable[..., autan.MeshInfo],
) -> None:
    diverged = case.run(crm_mesh, backend=autan.FakeBackend(diverge_at=10), runs=tmp_path)
    with pytest.raises(autan.CaseError) as error:
        case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=diverged)
    assert error.value.codes == ["E0707"]

    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=5), runs=tmp_path)

    def other_mesh(document: dict[str, dict[str, str]]) -> None:
        document["source"]["sha256"] = "1" * 64

    with pytest.raises(autan.CaseError) as error:
        case.run(
            mesh_variant("crm-hl", other_mesh),
            backend=autan.FakeBackend(),
            runs=tmp_path,
            start=stopped,
        )
    assert "E0203" in error.value.codes


def test_unsteady_continue_keeps_time_and_averaging(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    body = CRM_HL.replace("autan.SA()", "autan.DDES(base=autan.SA())") + (
        "from autan.units import s\n"
        "case.time = autan.Unsteady(steps=20, dt=1e-3 * s, average_from=5)\n"
    )
    case = autan.load(write_case(body))
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=8), runs=tmp_path)
    resumed = case.run(crm_mesh, backend=autan.FakeBackend(), runs=tmp_path, start=stopped)
    first = json.loads(history(resumed)[0])
    assert (first["iteration"], first["time"]) == (9, pytest.approx(9e-3))
    assert resumed.run.time == pytest.approx(20e-3)
    assert resumed.run.averaging is not None
    assert (resumed.run.averaging.from_step, resumed.run.averaging.to_step) == (6, 20)


def test_prepare_writes_the_start(
    case: autan.Case,
    crm_mesh: autan.MeshInfo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_case: Callable[[str], Path],
) -> None:
    stopped = case.run(crm_mesh, backend=autan.FakeBackend(iterations=30), runs=tmp_path)
    argv = [
        "prepare",
        str(write_case(CRM_HL)),
        "--mesh-info",
        str(FIXTURES / "crm-hl.meshinfo.json"),
        "--start",
        str(stopped.path),
    ]
    assert main(argv) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["start"]["run"] == str(stopped.path.absolute())
    assert (document["start"]["mode"], document["start"]["iterations"]) == ("continue", 30)
    assert main([*argv, "--warm"]) == 0
    assert json.loads(capsys.readouterr().out)["start"]["mode"] == "warm"
