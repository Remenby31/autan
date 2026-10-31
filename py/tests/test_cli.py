"""The ``cfd`` command line, through its output and exit status."""

import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from cfd.cli import main

from tests.conftest import EXAMPLES, FIXTURES, mesh_document

CRM_INFO = str(FIXTURES / "crm-hl.meshinfo.json")


def run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any]:
    status = main([*argv, "--json"])
    return status, json.loads(capsys.readouterr().out)


def test_plan_json_reports_missing_boundary_condition_with_fix(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, example: Callable[[str], Path]
) -> None:
    document = mesh_document("crm-hl")
    for zone in document["boundary_zones"]:
        if zone["name"] == "empennage":
            zone["type"] = None
    info = tmp_path / "crm-hl.meshinfo.json"
    info.write_text(json.dumps(document))

    status, out = run_json(capsys, "plan", str(example("crm_hl.py")), "--mesh-info", str(info))

    assert status == 1
    assert out["ok"] is False
    [found] = out["diagnostics"]
    assert found["code"] == "E0412"
    assert found["severity"] == "error"
    assert found["data"] == {"zone": "empennage"}
    assert "case.boundary('empennage')" in found["fix"]
    assert found["doc"].endswith("#e0412")
    empennage = next(b for b in out["boundaries"] if b["zone"] == "empennage")
    assert empennage["condition"] is None
    assert empennage["provenance"] is None


def test_plan_text_and_json_agree_on_success(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    status, out = run_json(capsys, "plan", str(example("crm_hl.py")), "--mesh-info", CRM_INFO)
    assert status == 0
    assert out["ok"] is True
    assert out["diagnostics"] == []
    assert main(["plan", str(example("crm_hl.py")), "--mesh-info", CRM_INFO]) == 0
    text = capsys.readouterr().out
    assert "7/7 zones covered" in text
    assert "plan OK" in text


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (["plan", str(EXAMPLES / "crm_hl.py")], "E0201"),
        (["plan", "missing.py", "--mesh-info", CRM_INFO], "E0001"),
    ],
)
def test_plan_failures_are_coded(
    argv: list[str], code: str, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = run_json(capsys, *argv)
    assert status == 1
    assert [d["code"] for d in out["diagnostics"]] == [code]


def test_version_reports_build_commit_and_schemas(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = run_json(capsys, "version")
    assert status == 0
    assert re.fullmatch(r"[0-9a-f]{40}", out["commit"])
    assert out["schemas"] == {
        "case": "cfd.case/1",
        "mesh-info": "cfd.mesh-info/1",
        "results": "cfd.results/1",
    }


def test_doctor_reports_checks_with_registered_codes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status, out = run_json(capsys, "doctor")
    assert status == 0
    assert {c["name"] for c in out["checks"]} == {"python", "gpu", "mpi", "core"}
    assert "W0803" in {d["code"] for d in out["diagnostics"]}


def test_schema_command_prints_json_schema(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "case"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$id"] == "cfd.case/1"


def test_installed_entry_point_runs() -> None:
    cfd = Path(sys.executable).with_name("cfd")
    done = subprocess.run(
        [str(cfd), "version", "--json"], capture_output=True, text=True, check=True
    )
    assert json.loads(done.stdout)["schemas"]["case"] == "cfd.case/1"


def test_usage_error_exits_2() -> None:
    with pytest.raises(SystemExit) as exit_:
        main(["plan"])
    assert exit_.value.code == 2


def test_explain_prints_a_registered_code(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = run_json(capsys, "explain", "e0412")
    assert status == 0
    assert out["code"] == "E0412"
    assert out["severity"] == "error"
    assert main(["explain", "E9999"]) == 2


def test_case_prints_do_not_corrupt_json_output(
    capsys: pytest.CaptureFixture[str], example: Callable[[str], Path]
) -> None:
    path = example("crm_hl.py")
    path.write_text('print("loading")\n' + path.read_text())
    status, out = run_json(capsys, "plan", str(path), "--mesh-info", CRM_INFO)
    assert status == 0
    assert out["ok"] is True


def test_explain_all_json_lists_every_code(capsys: pytest.CaptureFixture[str]) -> None:
    status, out = run_json(capsys, "explain", "--all")
    assert status == 0
    assert "E0412" in {entry["code"] for entry in out["codes"]}
