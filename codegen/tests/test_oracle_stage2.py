"""ADR 0003 stage 2's rules (#43: declaration, amended): an attempt over its bound or latched by the
watchdog is decisive, more attempts than the cap fail, a stage 2 manifest without its limits fails,
the auxiliary suite has its own bound, and stage 1 is unchanged. Also the watchdog itself, the
stage 2 instances of each branch, and the exclusion check's comparison."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
from oracle import heldout as h
from oracle import kernels as kr
from oracle.freeze import make_manifest as mm
from oracle.freeze import vv_exclusion as vx_ex
from oracle.freeze import watchdog

SPEC = {
    "id": "dev-hex-N2", "kind": "hex", "n": 2, "perturbation": 0.15, "seed": 3,
    "fields": [{"a": 1.0, "G": [0.3, -1.2, 0.7]}], "policies": ["fp64", "fp32"], "methods": ["lsq", "gg"],
}  # fmt: skip
NODES = ["tests/t.py::a", "tests/t.py::b"]
LIMITS = {"bounds_s": {"2": 600}, "aux_bound_s": 300, "retry_cap": 2}


def manifest(**over: object) -> tuple[bytes, str]:
    doc = {"stage": 2, "instances": [SPEC], "auxiliary_tests": {"node_ids": NODES}, **LIMITS}
    doc.update(over)
    raw = json.dumps(doc).encode()
    return raw, hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def library() -> str:
    digest = kr.library_sha256()
    assert digest is not None
    return digest


@pytest.fixture(scope="module")
def record(library: str) -> dict:
    _, sha = manifest()
    rec = h.run_instance(SPEC, library, sha)
    rec["started_utc_s"], rec["ended_utc_s"] = 1000.0, 1100.0
    return rec


def aux(library: str, sha: str, **over: object) -> dict:
    ev = {"schema": h.AUX_SCHEMA, "stage": "done", "manifest_sha256": sha,
          "library_sha256_expected": library, "library_sha256_loaded": library,
          "child_library_sha256": [library, library], "pytest_exit": 0,
          "outcomes": dict.fromkeys(NODES, "passed"),
          "started_utc_s": 2000.0, "ended_utc_s": 2100.0}  # fmt: skip
    ev.update(over)
    return ev


def collect(library: str, attempts: list, auxiliary: list | None = None, **over: object) -> dict:
    raw, sha = manifest(**over)
    return h.collect(raw, sha, library, {SPEC["id"]: attempts}, auxiliary or [aux(library, sha)])


def test_a_bound_attempt_within_its_bound_meets_the_acceptance_criteria(
    record: dict, library: str
) -> None:
    state = collect(library, [record])
    assert state["failures"] == []
    assert state["stage"] == 2
    assert state["state"].startswith("acceptance criteria met")


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"ended_utc_s": 1000.0 + 601}, "over its bound of 600 s"),
        ({"ended_utc_s": None}, "no valid start and end times"),
        ({"started_utc_s": 1200.0}, "no valid start and end times"),
        ({"timeout_latched": True}, "the run script's latch"),
    ],
)
def test_an_attempt_over_its_bound_or_latched_is_decisive(
    record: dict, library: str, change: dict, why: str
) -> None:
    late = {**copy.deepcopy(record), **change}
    state = collect(library, [late, record])
    assert any("decisive" in f for f in state["failures"]), state["failures"]
    assert state["state"] == "FAIL"
    assert why  # the screening reason, for the parametrize id


def test_more_attempts_than_the_cap_fail(record: dict, library: str) -> None:
    preflight = {"schema": h.SCHEMA, "id": SPEC["id"], "invalid": [], "stage": "manifest"}
    state = collect(library, [preflight, preflight, record])
    assert any("over the cap of 2" in f for f in state["failures"])
    assert collect(library, [preflight, record])["failures"] == []


def test_the_auxiliary_suite_has_its_own_bound(record: dict, library: str) -> None:
    _, sha = manifest()
    for bad in (aux(library, sha, ended_utc_s=2000.0 + 301), aux(library, sha, timeout_latched=True),
                aux(library, sha, started_utc_s=None)):  # fmt: skip
        assert collect(library, [record], [bad])["failures"]


@pytest.mark.parametrize(
    "over", [{"bounds_s": {}}, {"bounds_s": {"2": 0}}, {"aux_bound_s": None}, {"retry_cap": -1}]
)
def test_a_stage_2_manifest_without_valid_limits_fails(
    record: dict, library: str, over: dict
) -> None:
    state = collect(library, [record], **over)
    assert "the manifest freezes no valid stage 2 limits" in state["failures"]


def test_an_instance_without_a_frozen_bound_for_its_n_fails(record: dict, library: str) -> None:
    state = collect(library, [record], bounds_s={"12": 600})
    assert any("no bound for N = 2" in f for f in state["failures"])


def test_stage_1_is_unchanged(library: str) -> None:
    raw = json.dumps({"instances": [SPEC], "auxiliary_tests": {"node_ids": NODES}}).encode()
    sha = hashlib.sha256(raw).hexdigest()
    rec = h.run_instance(SPEC, library, sha)
    ev = aux(library, sha)
    state = h.collect(raw, sha, library, {SPEC["id"]: [rec]}, [ev])
    assert state["failures"] == []
    assert state["stage"] == 1
    assert state["state"].startswith("no failure observed in stage 1")


def test_the_watchdog_returns_the_command_s_status_within_its_bound(tmp_path: Path) -> None:
    latch = tmp_path / "a.json.timeout"
    assert watchdog.run(30, str(latch), [sys.executable, "-c", "raise SystemExit(3)"]) == 3
    assert not latch.exists()


def test_the_watchdog_latches_then_stops_the_command_at_its_bound(tmp_path: Path) -> None:
    latch = tmp_path / "a.json.timeout"
    t0 = time.time()
    code = watchdog.run(1, str(latch), [sys.executable, "-c", "import time; time.sleep(60)"])
    assert code == 124
    assert time.time() - t0 < 30
    assert json.loads(latch.read_text())["bound_s"] == 1


@pytest.mark.parametrize(("branch", "count"), [("a", 5)])
def test_a_branch_gives_its_instances_each_a_valid_spec(branch: str, count: int) -> None:
    out = mm.instances_stage2(branch, mm.STAGE2_FILES, mm.STAGE2_BOUNDS_S)
    assert len(out) == count
    assert all(h.spec_issues(i) == [] for i in out)
    assert {i["n"] for i in out} == {6, 12}
    assert all((i["n"], i["perturbation"], i["seed"]) not in mm.STAGE2_EXCLUDED for i in out)
    assert all(i["bound_s"] == mm.STAGE2_BOUNDS_S[str(i["n"])] for i in out)


def test_branch_a_prime_needs_its_n16_files_and_bound() -> None:
    with pytest.raises(SystemExit):
        mm.instances_stage2("a-prime", mm.STAGE2_FILES, mm.STAGE2_BOUNDS_S)


def _universe(*records: dict, not_passed: list[str] | None = None) -> dict:
    return {"records": list(records), "not_passed": not_passed or []}


CANDIDATES = {
    "expected_collisions": [{"n": 6, "perturbation": 0.0, "seed": 11}],
    "candidates": [
        {"n": 6, "perturbation": 0.0, "seed": 11, "file_sha256": "f6", "mesh_sha256": "m6"},
        {"n": 12, "perturbation": 0.0, "seed": 11, "file_sha256": "f12", "mesh_sha256": "m12"},
    ],
}
TEST6 = {"test": "t6", "kind": "agglomerated", "n": 6, "file_sha256": "f6", "mesh_sha256": "m6"}


def test_the_exclusion_check_passes_with_only_the_declared_collision() -> None:
    assert vx_ex.compare(_universe(TEST6), CANDIDATES) == []


@pytest.mark.parametrize(
    ("universe", "issue"),
    [
        (_universe(TEST6, {**TEST6, "test": "t12", "file_sha256": None, "mesh_sha256": "m12"}),
         "N12 p0.0 s11 collides"),
        (_universe({**TEST6, "file_sha256": "x", "mesh_sha256": "y"}), "expected to collide"),
        (_universe(TEST6, not_passed=["vv/tests/t.py::a: skipped"]), "vv test not passed"),
        (_universe(), "the universe is empty"),
    ],
)  # fmt: skip
def test_the_exclusion_check_refuses_a_collision_a_missing_one_a_skip_and_an_empty_universe(
    universe: dict, issue: str
) -> None:
    assert any(issue in i for i in vx_ex.compare(universe, CANDIDATES))
