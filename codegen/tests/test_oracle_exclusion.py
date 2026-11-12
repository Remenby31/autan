from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from oracle.freeze import vv_exclusion as vx_ex

CANDIDATES = {
    "expected_collisions": [{"n": 6, "perturbation": 0.0, "seed": 11}],
    "candidates": [
        {"n": 6, "perturbation": 0.0, "seed": 11, "file_sha256": "f6", "mesh_sha256": "m6"},
        {"n": 12, "perturbation": 0.0, "seed": 11, "file_sha256": "f12", "mesh_sha256": "m12"},
    ],
}
TEST6 = {"test": "t6", "kind": "agglomerated", "n": 6, "file_sha256": "f6", "mesh_sha256": "m6"}
UNIVERSE = {"records": [TEST6], "not_passed": [], "exitstatus": 0}
INST12 = {
    "id": "i12", "n": 12, "perturbation": 0.0, "seed": 11, "files": {"mesh": {"sha256": "f12"}},
}  # fmt: skip
MANIFEST = {"instances": [INST12], "excluded": {"excluded": ["N6 p0.0 s11"]}}


def _issues(manifest: object, candidates: dict = CANDIDATES) -> list[str]:
    return vx_ex.compare(copy.deepcopy(UNIVERSE), copy.deepcopy(candidates), manifest)  # type: ignore[arg-type]


def _with_instance(**changes: object) -> dict:
    m = copy.deepcopy(MANIFEST)
    m["instances"] = [{**INST12, **changes}]
    return m


def _without_n12_digest() -> tuple[dict, dict]:
    m = _with_instance(files={"mesh": {}})
    c = copy.deepcopy(CANDIDATES)
    c["candidates"][1]["file_sha256"] = None
    return m, c


def test_a_consistent_manifest_passes() -> None:
    assert _issues(copy.deepcopy(MANIFEST)) == []


@pytest.mark.parametrize(
    ("manifest", "candidates", "needles"),
    [
        pytest.param(
            {"instances": [{"id": "i6", "files": {"mesh": {"sha256": "f6"}}}],
             "excluded": {"excluded": ["N6 p0.0 s11"]}},
            CANDIDATES, ["no n, perturbation or seed", "collides with t6"],
            id="instance-without-identity-carrying-an-excluded-file",
        ),
        pytest.param(
            _with_instance(files={"mesh": {"sha256": "f6"}}), CANDIDATES,
            ["not that of candidate N12 p0.0 s11", "instance N12 p0.0 s11 collides with t6"],
            id="identity-and-file-of-different-candidates",
        ),
        pytest.param(
            copy.deepcopy(MANIFEST), {**CANDIDATES, "candidates": CANDIDATES["candidates"][1:]},
            ["is listed as an expected collision but is no candidate"],
            id="expected-collision-not-a-candidate",
        ),
        pytest.param(
            {"excluded": {"excluded": ["N6 p0.0 s11"]}}, CANDIDATES,
            ["the manifest has no instance"], id="manifest-without-instances",
        ),
        pytest.param(
            {"instances": [INST12]}, CANDIDATES, ["expected collision but not excluded"],
            id="manifest-without-exclusions",
        ),
        pytest.param(
            copy.deepcopy(MANIFEST),
            {**CANDIDATES, "candidates": [*CANDIDATES["candidates"], CANDIDATES["candidates"][1]]},
            ["listed twice among the candidates"], id="candidate-listed-twice",
        ),
        pytest.param(
            *_without_n12_digest(),
            ["a candidate without both digests", "its mesh file is no candidate"],
            id="instance-and-candidate-without-file-digest",
        ),
        pytest.param(None, CANDIDATES, [], id="no-manifest-mode-checks-the-universe-only"),
        pytest.param([], CANDIDATES, ["the manifest is not a JSON object"], id="manifest-not-an-object"),
    ],
)  # fmt: skip
def test_the_exclusion_check_fails_closed(
    manifest: object, candidates: dict, needles: list[str]
) -> None:
    found = _issues(manifest, candidates)
    assert all(any(needle in i for i in found) for needle in needles), (needles, found)
    assert bool(found) == bool(needles)


def test_the_cli_refuses_a_manifest_file_holding_null(tmp_path: Path) -> None:
    u, c, m = tmp_path / "u.json", tmp_path / "c.json", tmp_path / "m.json"
    u.write_text(json.dumps(UNIVERSE))
    c.write_text(json.dumps(CANDIDATES))
    m.write_text("null")
    cli = Path(vx_ex.__file__)
    r = subprocess.run([sys.executable, str(cli), str(u), str(c), str(m)], check=False)
    assert r.returncode == 2
