import json
import shutil
import sys
from pathlib import Path
from typing import Any

import autan
import pytest
from autan_ui.app import create_app
from fastapi.testclient import TestClient

PY = Path(__file__).parents[3] / "py"
CRM_HL = PY / "tests" / "fixtures" / "crm-hl.meshinfo.json"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    shutil.copy(CRM_HL, meshes / "crm-hl.meshinfo.json")
    (meshes / "notes.txt").write_text("not a MeshInfo")
    app = create_app(tmp_path / "runs", static=None, hosts=("testserver",), meshes_root=meshes)
    return TestClient(app)


def test_the_meshes_are_the_meshinfo_files_of_the_directory(client: TestClient) -> None:
    listed = client.get("/api/meshes").json()

    expected = json.loads(CRM_HL.read_text())
    assert [m["name"] for m in listed] == ["crm-hl"]
    assert listed[0]["boundary_zones"] == expected["boundary_zones"]
    assert listed[0]["counts"] == expected["counts"]


def _without_files(plan: dict[str, Any]) -> dict[str, Any]:
    # Diagnostics name the directory the case was planned in; what they say is their code.
    found = [(d["code"], d["severity"], d["location"]["path"]) for d in plan["diagnostics"]]
    return {**plan, "diagnostics": found}


def test_a_case_renders_as_the_sdk_reads_it(client: TestClient, tmp_path: Path) -> None:
    case = autan.load(PY / "examples" / "crm_hl.py")
    document = json.loads(case.to_json())
    mesh_info = autan.MeshInfo.load(CRM_HL)

    rendered = client.post("/api/cases/render", json={"case": document, "mesh": "crm-hl"}).json()

    # The SDK's own plan of the same document, read from a file next to nothing else.
    alone = tmp_path / "alone"
    alone.mkdir()
    (alone / "case.json").write_text(case.to_json())
    expected = autan.load(alone / "case.json").plan(mesh_info).document()
    assert rendered["ok"] is True
    assert rendered["python"] == case.to_python()
    assert rendered["toml"] == case.to_toml()
    assert _without_files(rendered["plan"]) == _without_files(expected)


OPENED: list[str] | None = None


def _record(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and OPENED is not None and isinstance(args[0], str | Path):
        OPENED.append(str(args[0]))


sys.addaudithook(_record)


@pytest.mark.parametrize("mesh", ["CRM_HL", "..", "../..", "/"])
def test_planning_reads_no_mesh_the_case_names(client: TestClient, mesh: str) -> None:
    document = json.loads(autan.load(PY / "examples" / "crm_hl.py").to_json())
    document["mesh"] = str(CRM_HL) if mesh == "CRM_HL" else mesh  # Paths the server can reach.

    rendered = client.post("/api/cases/render", json={"case": document, "mesh": "crm-hl"}).json()

    assert rendered["plan"]["mesh"]["path"] == document["mesh"]  # The case keeps its own.
    assert "W0205" in {d["code"] for d in rendered["plan"]["diagnostics"]}


def test_planning_reads_no_file_the_case_names(client: TestClient, tmp_path: Path) -> None:
    global OPENED  # noqa: PLW0603 -- the audit hook cannot be removed, only muted
    sentinel = tmp_path / "manufactured.json"
    sentinel.write_text("{}")
    document = json.loads(autan.load(PY / "examples" / "crm_hl.py").to_json())
    # A case verification allows: inviscid, every boundary given by the manufactured solution.
    document["physics"]["turbulence"] = {"model": "inviscid"}
    document["boundaries"] = []
    document["monitors"] = []
    document["verification"] = {"manufactured": {"path": str(sentinel), "sha256": "0" * 64}}

    OPENED = []
    try:
        rendered = client.post("/api/cases/render", json={"case": document, "mesh": "crm-hl"})
    finally:
        opened, OPENED = OPENED, None

    assert rendered.json()["ok"] is True
    assert str(sentinel) not in opened


def test_a_case_the_sdk_refuses_renders_as_its_coded_diagnostics(client: TestClient) -> None:
    document = json.loads(autan.load(PY / "examples" / "crm_hl.py").to_json())
    document["freestream"]["mach"] = -1

    rendered = client.post("/api/cases/render", json={"case": document}).json()

    assert rendered["ok"] is False
    assert all(d["code"].startswith("E") for d in rendered["diagnostics"])
    assert rendered["diagnostics"]


def test_a_mesh_not_listed_is_404(client: TestClient) -> None:
    document = json.loads(autan.load(PY / "examples" / "crm_hl.py").to_json())

    for mesh in ("../crm-hl", "notes.txt", "missing"):
        response = client.post("/api/cases/render", json={"case": document, "mesh": mesh})
        assert response.status_code == 404
