"""Cases verified against a manufactured solution (case-schema.md#verification).

The solutions are real `autan-vv mms` exports (fixtures/*.mms.json); their declared
parameters come from docs/validation/mms.md (Euler: gamma 1.4, R 287; RANS-SA: also
mu 1e-3, Pr 0.72).
"""

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import autan
import autan.solver_input
import jsonschema
import pytest
from autan import cgns
from autan.cli import json_schema, main
from pydantic import ValidationError

from tests.conftest import FIXTURES

R_UNIVERSAL = 8.314462618

EULER = (
    "import autan\nfrom autan.units import K, Pa, kg, mol, s\n"
    "gas = autan.IdealGas(name='mms', molar_mass={molar_mass!r} * kg / mol, gamma=1.4,"
    " prandtl=0.72, viscosity=autan.ConstantViscosity(mu=1e-3 * Pa * s))\n"
    "case = autan.Case('mms', mesh='crm-hl.cgns')\n"
    "case.physics = autan.Compressible(turbulence={turbulence}, gas=gas)\n"
    "case.freestream(mach=0.3, pressure=101325 * Pa, temperature=300 * K)\n"
    "case.verification = autan.Manufactured(path='{name}.mms.json', sha256={digest!r})\n"
)


def digest(name: str) -> str:
    return hashlib.sha256((FIXTURES / f"{name}.mms.json").read_bytes()).hexdigest()


def verified(
    write_case: Callable[[str], Path],
    name: str = "euler",
    turbulence: str = "autan.Inviscid()",
    molar_mass: float = R_UNIVERSAL / 287.0,
    sha256: str | None = None,
    extra: str = "",
) -> Path:
    case = write_case(
        EULER.format(
            molar_mass=molar_mass, turbulence=turbulence, name=name, digest=sha256 or digest(name)
        )
        + extra
    )
    shutil.copy(FIXTURES / f"{name}.mms.json", case.parent / f"{name}.mms.json")
    return case


@pytest.mark.parametrize(
    ("name", "turbulence"), [("euler", "autan.Inviscid()"), ("rans_sa", "autan.SA()")]
)
def test_every_boundary_takes_the_manufactured_data(
    name: str,
    turbulence: str,
    write_case: Callable[[str], Path],
    mesh_variant: Callable[..., autan.MeshInfo],
) -> None:
    def untype(document: dict[str, list[dict[str, object]]]) -> None:
        for zone in document["boundary_zones"]:
            zone["type"] = None

    result = autan.load(verified(write_case, name, turbulence)).plan(mesh_variant("crm-hl", untype))
    assert result.diagnostics == ()
    assert {(b.condition or {}).get("kind") for b in result.boundaries} == {"manufactured"}
    assert {b.provenance for b in result.boundaries} == {"verification"}


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (lambda case: (case.parent / "euler.mms.json").unlink(), "E0207"),
        (lambda case: (case.parent / "euler.mms.json").write_text("{}"), "E0206"),
    ],
)
def test_the_solution_file_is_checked(
    setup: Callable[[Path], object],
    code: str,
    write_case: Callable[[str], Path],
    crm_mesh: autan.MeshInfo,
) -> None:
    case = verified(write_case)
    setup(case)
    assert [d.code for d in autan.load(case).plan(crm_mesh).diagnostics] == [code]


def test_a_file_of_another_format_is_e0208(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo
) -> None:
    other = (FIXTURES / "crm-hl.meshinfo.json").read_bytes()
    case = verified(write_case, sha256=hashlib.sha256(other).hexdigest())
    (case.parent / "euler.mms.json").write_bytes(other)
    assert [d.code for d in autan.load(case).plan(crm_mesh).diagnostics] == ["E0208"]


@pytest.mark.parametrize(
    ("kwargs", "extra", "mismatches"),
    [
        ({"molar_mass": 0.0289647}, "", 1),  # air: R = 287.05, not 287
        ({"turbulence": "autan.SA()"}, "", 1),  # SA case, inviscid solution
        ({"name": "rans_sa", "turbulence": "autan.Inviscid()"}, "", 1),  # SA solution
        ({"name": "rans_sa", "turbulence": "autan.Laminar()"}, "", 1),  # SA solution
        ({"name": "rans_sa_noft2", "turbulence": "autan.SA()"}, "", 1),  # no case counterpart
        ({}, "case.cell_zone('fluid').fluid(gas)\n", 1),
    ],
)
def test_the_solution_must_match_the_case_physics(
    kwargs: dict[str, object],
    extra: str,
    mismatches: int,
    write_case: Callable[[str], Path],
    crm_mesh: autan.MeshInfo,
) -> None:
    case = verified(write_case, extra=extra, **kwargs)  # type: ignore[arg-type]
    codes = [d.code for d in autan.load(case).plan(crm_mesh).diagnostics]
    assert codes == ["E0209"] * mismatches


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        ("case.boundary('farfield').farfield()\n", "E0416"),
        ("from autan.units import s\ncase.time = autan.Unsteady(steps=1, dt=1e-6 * s)\n", "E0417"),
    ],
)
def test_verification_cases_reject_contradictions(
    extra: str, code: str, write_case: Callable[[str], Path]
) -> None:
    with pytest.raises(autan.CaseError) as error:
        _ = autan.load(verified(write_case, extra=extra)).spec
    assert error.value.codes == [code]


def test_verification_round_trips_and_enters_the_hash(
    write_case: Callable[[str], Path], tmp_path: Path
) -> None:
    case = autan.load(verified(write_case))
    document = case.spec.document()
    assert document["verification"] == {
        "manufactured": {"path": "euler.mms.json", "sha256": digest("euler")}
    }
    validator = jsonschema.Draft202012Validator(json_schema("case"))
    validator.validate(document)
    other = autan.load(verified(write_case, sha256="0" * 64))
    assert other.spec.hash() != case.spec.hash()
    toml = tmp_path / "copy.toml"
    toml.write_text(case.to_toml())
    assert autan.load(toml).spec == case.spec
    rules = [{"zones": "farfield", "condition": {"kind": "farfield"}}]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**document, "boundaries": rules})
    validator.validate({**document, "verification": None, "boundaries": rules})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(
            {**document, "verification": {"manufactured": {"path": "x", "sha256": "0" * 64 + "\n"}}}
        )


def test_solver_input_carries_the_solution(
    write_case: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    case = verified(write_case)
    argv = ["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]
    assert main(argv) == 0
    document = json.loads(capsys.readouterr().out)
    validator = jsonschema.Draft202012Validator(json_schema("solver-input"))
    validator.validate(document)
    assert document["verification"] == {
        "path": str((case.parent / "euler.mms.json").resolve()),
        "sha256": digest("euler"),
    }
    assert {b["kind"] for b in document["boundaries"]} == {"manufactured"}
    for edit in (
        {"verification": None},
        {"verification": {"path": "relative.json", "sha256": digest("euler")}},
        {"verification": {"path": document["verification"]["path"], "sha256": "0" * 64}},
    ):
        altered = {**document, **edit}
        with pytest.raises(ValidationError):
            autan.solver_input.SolverInput.model_validate(altered)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**document, "verification": None})


def test_runs_archive_the_solution(
    write_case: Callable[[str], Path], crm_mesh: autan.MeshInfo, tmp_path: Path
) -> None:
    result = autan.load(verified(write_case)).run(
        crm_mesh, backend=autan.FakeBackend(iterations=2), runs=tmp_path / "runs"
    )
    bc_types = {
        bc.text()
        for bc in cgns.read(result.path / "solution.cgns")
        .child("Base")
        .child("Zone")
        .child("ZoneBC")
        .with_label("BC_t")
    }
    assert bc_types == {"UserDefined"}  # the SIDS string, not the C enumerator name
    assert result.field("Density").size > 0  # the reader accepts the solution
    archived = result.path / "manufactured.json"
    assert archived.read_bytes() == (FIXTURES / "euler.mms.json").read_bytes()
    archived.write_text("{}")
    with pytest.raises(autan.CaseError) as error:
        autan.Results.open(result.path)
    assert error.value.codes == ["E0701"]
