"""ADR 0003 stage 2's hosted exclusion check, fail-closed on malformed or inconsistent inputs (#43).

Kept apart from the frozen auxiliary suite: the exclusion check runs on hosted CI, not in the held-out
runs, and this file is not one of the manifest's hashed oracle files."""

from __future__ import annotations

import copy

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


def _issues(manifest: dict, candidates: dict = CANDIDATES) -> list[str]:
    return vx_ex.compare(copy.deepcopy(UNIVERSE), copy.deepcopy(candidates), manifest)


def test_a_consistent_manifest_passes() -> None:
    assert _issues(copy.deepcopy(MANIFEST)) == []


def _with_instance(**changes: object) -> dict:
    m = copy.deepcopy(MANIFEST)
    m["instances"] = [{**INST12, **changes}]
    return m


@pytest.mark.parametrize(
    ("manifest", "candidates", "needles"),
    [
        # An instance without its identity, carrying an excluded candidate's file: it collides.
        (
            {"instances": [{"id": "i6", "files": {"mesh": {"sha256": "f6"}}}],
             "excluded": {"excluded": ["N6 p0.0 s11"]}},
            CANDIDATES,
            ["no n, perturbation or seed", "collides with t6"],
        ),
        # An N12 identity carrying the N6 file: not its candidate's file, and it collides.
        (_with_instance(files={"mesh": {"sha256": "f6"}}), CANDIDATES,
         ["not that of candidate N12 p0.0 s11", "instance N12 p0.0 s11 collides with t6"]),
        # An expected collision missing from the candidates.
        (copy.deepcopy(MANIFEST),
         {**CANDIDATES, "candidates": CANDIDATES["candidates"][1:]},
         ["is listed as an expected collision but is no candidate"]),
        # A manifest without instances.
        ({"excluded": {"excluded": ["N6 p0.0 s11"]}}, CANDIDATES, ["the manifest has no instance"]),
        # A manifest without exclusions.
        ({"instances": [INST12]}, CANDIDATES, ["expected collision but not excluded"]),
        # A candidate listed twice.
        (copy.deepcopy(MANIFEST),
         {**CANDIDATES, "candidates": [*CANDIDATES["candidates"], CANDIDATES["candidates"][1]]},
         ["listed twice among the candidates"]),
    ],
)  # fmt: skip
def test_the_exclusion_check_fails_closed(
    manifest: dict, candidates: dict, needles: list[str]
) -> None:
    found = _issues(manifest, candidates)
    for needle in needles:
        assert any(needle in i for i in found), (needle, found)
