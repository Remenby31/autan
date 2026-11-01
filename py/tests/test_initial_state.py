"""A case starting from an initial-state file (docs/contracts/initial-state.md)."""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import autan.initial_state
import autan.solver_input
import jsonschema
import pydantic
import pytest
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, mesh_document

LAMINAR = CRM_HL.replace("autan.SA()", "autan.Laminar()")
CELLS = 4
CENTROIDS = [[0.1 * i, 0.0, 0.0] for i in range(CELLS)]
VALUES = [[1.2, 30.0, 0.0, 0.0, 1e5] for _ in range(CELLS)]


def small(document: dict[str, Any]) -> None:
    """The CRM-HL MeshInfo with four fluid cells."""
    document["counts"]["cells"] = CELLS
    document["counts"]["cells_by_type"] = dict.fromkeys(document["counts"]["cells_by_type"], 0)
    document["counts"]["cells_by_type"]["hexa"] = CELLS
    document["cell_zones"] = [{"name": "fluid", "type": None, "native_type": None, "cells": CELLS}]
    document["interfaces"] = []


@pytest.fixture
def mesh(mesh_variant: Callable[..., autan.MeshInfo], tmp_path: Path) -> Path:
    path = tmp_path / "small.meshinfo.json"
    path.write_text(json.dumps(mesh_variant("crm-hl", small).document()))
    return path


def with_state(
    write_case: Callable[..., Path], body: str = LAMINAR, name: str = "case.py", **changes: Any
) -> Path:
    case = write_case(body, name=name)
    arguments: dict[str, Any] = {
        "oracle": "test",
        "mesh_sha256": mesh_document("crm-hl")["source"]["sha256"],
        "centroids": CENTROIDS,
        "values": VALUES,
    } | changes
    digest = autan.initial_state.write(case.parent / "state.json", **arguments)
    case.write_text(
        case.read_text()
        + f"case.initial_state = autan.StateFile(path='state.json', sha256={digest!r})\n"
    )
    return case


def test_write_returns_the_hash_of_a_valid_document(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    digest = autan.initial_state.write(
        path, oracle="o", mesh_sha256="0" * 64, centroids=CENTROIDS, values=VALUES
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    document = autan.initial_state.parse(path.read_bytes())
    assert not isinstance(document, str)
    assert (document.cells, document.values[0]) == (CELLS, (1.2, 30.0, 0.0, 0.0, 1e5))


@pytest.mark.parametrize(
    "values",
    [
        [[1.2, 30.0, 0.0, 0.0, float("nan")]] * CELLS,
        [[-1.2, 30.0, 0.0, 0.0, 1e5]] * CELLS,
        [[1.2, 30.0, 0.0, 1e5]] * CELLS,
        VALUES[:-1],
    ],
)
def test_write_refuses_what_the_contract_refuses(values: list[list[float]], tmp_path: Path) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 -- pydantic's ValidationError is a ValueError
        autan.initial_state.write(
            tmp_path / "s.json",
            oracle="o",
            mesh_sha256="0" * 64,
            centroids=CENTROIDS,
            values=values,
        )


def test_the_state_reaches_the_solver_input_and_the_run(
    write_case: Callable[..., Path], mesh: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = with_state(write_case)
    loaded = autan.load(case)
    assert loaded.plan(autan.MeshInfo.load(mesh)).ok
    assert main(["prepare", str(case), "--mesh-info", str(mesh)]) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    data = (case.parent / "state.json").read_bytes()
    assert document["initial_state"] == {
        "path": str((case.parent / "state.json").resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    document["initial_state"]["sha256"] = "0" * 64  # not the case's file
    with pytest.raises(pydantic.ValidationError):
        autan.solver_input.SolverInput.model_validate(document)

    result = loaded.run(
        autan.MeshInfo.load(mesh), backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs"
    )
    archived = result.path / "states" / f"{hashlib.sha256(data).hexdigest()}.json"
    assert archived.read_bytes() == data
    archived.write_text("{}")
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(result.path)
    assert error.value.codes == ["E0701"]


def _fifo(case: Path) -> None:
    (case.parent / "state.json").unlink()
    os.mkfifo(case.parent / "state.json")


def _raw(case: Path, text: str) -> None:
    """Replace the state file by ``text`` and point the case at its hash."""
    data = text.encode()
    (case.parent / "state.json").write_bytes(data)
    body = case.read_text().rsplit("case.initial_state", 1)[0]
    case.write_text(
        body + "case.initial_state = autan.StateFile(path='state.json', "
        f"sha256={hashlib.sha256(data).hexdigest()!r})\n"
    )


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        (lambda case: (case.parent / "state.json").write_text("{}"), "E0508"),  # another hash
        (lambda case: _raw(case, '{"schema": "autan.initial-state/1"}'), "E0508"),
        (
            lambda case: _raw(
                case, (case.parent / "state.json").read_text().replace("100000.0", "NaN", 1)
            ),
            "E0508",
        ),
        (lambda case: (case.parent / "state.json").unlink(), "E0508"),
        (_fifo, "E0508"),  # a FIFO with no writer: never a blocking read
    ],
)
def test_the_file_is_checked(
    damage: Callable[[Path], object], code: str, write_case: Callable[..., Path], mesh: Path
) -> None:
    case = with_state(write_case)
    damage(case)
    codes = [d.code for d in autan.load(case).plan(autan.MeshInfo.load(mesh)).diagnostics]
    assert codes == [code]


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"mesh_sha256": "1" * 64}, "E0509"),
        ({"centroids": CENTROIDS[:-1], "values": VALUES[:-1]}, "E0510"),
    ],
)
def test_the_file_fits_the_mesh(
    changes: dict[str, Any], code: str, write_case: Callable[..., Path], mesh: Path
) -> None:
    case = with_state(write_case, **changes)
    codes = [d.code for d in autan.load(case).plan(autan.MeshInfo.load(mesh)).diagnostics]
    assert codes == [code]


@pytest.mark.parametrize(
    "body",
    [
        CRM_HL,  # SA: the file has no turbulence variable
        LAMINAR + "from autan.units import K, Pa, m, s\n"
        "case.cell_zone('fluid').fluid(autan.gases.AIR, initial=autan.State(pressure=1e5 * Pa, "
        "temperature=300 * K, velocity=(0 * m / s, 0 * m / s, 0 * m / s)))\n",
    ],
)
def test_the_file_needs_a_case_it_can_start(body: str, write_case: Callable[..., Path]) -> None:
    with pytest.raises(autan.CaseError) as error:
        _ = autan.load(with_state(write_case, body)).spec
    assert error.value.codes == ["E0512"]


def test_the_file_and_a_start_exclude_each_other(
    write_case: Callable[..., Path], mesh: Path, tmp_path: Path
) -> None:
    info = autan.MeshInfo.load(mesh)
    previous = autan.load(write_case(LAMINAR, name="plain.py")).run(
        info, backend=autan.FakeBackend(iterations=3), runs=tmp_path / "runs"
    )
    with pytest.raises(autan.CaseError) as error:
        autan.load(with_state(write_case)).run(
            info, backend=autan.FakeBackend(), runs=tmp_path / "runs", start=previous
        )
    assert "E0512" in error.value.codes
