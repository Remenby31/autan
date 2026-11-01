"""Cases verified against a manufactured solution (case-schema.md#verification)."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import autan
import jsonschema
import pytest
from autan.cli import json_schema, main

from tests.conftest import CRM_HL, FIXTURES

SOLUTION = b'{"schema": "autan.mms/1", "name": "euler"}'
DIGEST = hashlib.sha256(SOLUTION).hexdigest()
BODY = CRM_HL.replace('case.boundary("wing.*").wall()\n', "").replace(
    'case.boundary("symmetry").symmetry()\n', ""
)


def verified(write_case: Callable[[str], Path], digest: str = DIGEST, extra: str = "") -> Path:
    return write_case(
        BODY
        + f"case.verification = autan.Manufactured(path='euler.mms.json', sha256={digest!r})\n"
        + extra
    )


def test_every_boundary_takes_the_manufactured_data(
    write_case: Callable[[str], Path], mesh_variant: Callable[..., autan.MeshInfo]
) -> None:
    case = verified(write_case)
    (case.parent / "euler.mms.json").write_bytes(SOLUTION)

    def untype(document: dict[str, object]) -> None:
        for zone in document["boundary_zones"]:  # type: ignore[attr-defined]
            zone["type"] = None

    result = autan.load(case).plan(mesh_variant("crm-hl", untype))
    assert result.ok
    assert result.diagnostics == ()
    assert {(b.condition or {}).get("kind") for b in result.boundaries} == {"manufactured"}
    assert {b.provenance for b in result.boundaries} == {"verification"}


@pytest.mark.parametrize(
    ("content", "digest", "code"),
    [(None, DIGEST, "E0207"), (b"{}", DIGEST, "E0206")],
)
def test_the_solution_file_is_checked(
    content: bytes | None,
    digest: str,
    code: str,
    write_case: Callable[[str], Path],
    crm_mesh: autan.MeshInfo,
) -> None:
    case = verified(write_case, digest)
    if content is not None:
        (case.parent / "euler.mms.json").write_bytes(content)
    assert [d.code for d in autan.load(case).plan(crm_mesh).diagnostics] == [code]


def test_boundary_rules_contradict_a_manufactured_solution(
    write_case: Callable[[str], Path],
) -> None:
    with pytest.raises(autan.CaseError) as error:
        _ = autan.load(verified(write_case, extra="case.boundary('farfield').farfield()\n")).spec
    assert error.value.codes == ["E0416"]


def test_verification_round_trips_and_enters_the_hash(
    write_case: Callable[[str], Path], tmp_path: Path
) -> None:
    case = autan.load(verified(write_case))
    document = case.spec.document()
    assert document["verification"] == {
        "manufactured": {"path": "euler.mms.json", "sha256": DIGEST}
    }
    jsonschema.Draft202012Validator(json_schema("case")).validate(document)
    other = autan.load(verified(write_case, "0" * 64))
    assert other.spec.hash() != case.spec.hash()
    toml = tmp_path / "case.toml"
    toml.write_text(case.to_toml())
    assert autan.load(toml).spec == case.spec
    document["boundaries"] = [{"zones": "farfield", "condition": {"kind": "farfield"}}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("case")).validate(document)


def test_solver_input_carries_the_solution(
    write_case: Callable[[str], Path], capsys: pytest.CaptureFixture[str]
) -> None:
    case = verified(write_case)
    (case.parent / "euler.mms.json").write_bytes(SOLUTION)
    argv = ["prepare", str(case), "--mesh-info", str(FIXTURES / "crm-hl.meshinfo.json")]
    assert main(argv) == 0
    document = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
    assert document["verification"] == {
        "path": str((case.parent / "euler.mms.json").resolve()),
        "sha256": DIGEST,
    }
    assert {b["kind"] for b in document["boundaries"]} == {"manufactured"}
    document["verification"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(json_schema("solver-input")).validate(document)
