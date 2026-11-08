"""The held-out runner (ADR 0003) is fail-closed: incomplete, malformed or inconsistent records never
pass; an exception or interruption never discards what was recorded; records and auxiliary evidence
are bound to the approved manifest bytes and library; an attempt that reported an issue or completed
without passing stays decisive; and stage 1 needs every instance and every frozen auxiliary test."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from oracle import heldout as h
from oracle import kernels as kr

SPEC = {
    "id": "dev-hex-N2", "kind": "hex", "n": 2, "perturbation": 0.15, "seed": 3,
    "fields": [{"a": 1.0, "G": [0.3, -1.2, 0.7]}], "policies": ["fp64", "fp32"], "methods": ["lsq", "gg"],
}  # fmt: skip
NODES = ["tests/t.py::a", "tests/t.py::b"]
MANIFEST = json.dumps({"instances": [SPEC], "auxiliary_tests": {"node_ids": NODES}}).encode()
MANIFEST_SHA = hashlib.sha256(MANIFEST).hexdigest()


@pytest.fixture(scope="module")
def library() -> str:
    digest = kr.library_sha256()
    assert digest is not None
    return digest


@pytest.fixture(scope="module")
def record(library: str) -> dict:
    return h.run_instance(SPEC, library, MANIFEST_SHA)


def aux(library: str, **over: object) -> dict:
    ev = {"schema": h.AUX_SCHEMA, "stage": "done", "manifest_sha256": MANIFEST_SHA,
          "library_sha256_expected": library, "library_sha256_loaded": library,
          "child_library_sha256": [library, library],
          "pytest_exit": 0, "outcomes": dict.fromkeys(NODES, "passed")}  # fmt: skip
    ev.update(over)
    return ev


def test_a_complete_clean_record_passes(record: dict) -> None:
    assert record["stage"] == "done"
    assert h.passed(record, SPEC)


@pytest.mark.parametrize(
    "spoil",
    [
        lambda r: r.update(runs=[]),
        lambda r: r["runs"].pop(),
        lambda r: r["runs"].append(r["runs"][0]),
        lambda r: r["runs"][0].update(d1_entries=r["runs"][0]["d1_entries"] - 1),
        lambda r: r["runs"][0].update(bound_covers_every_entry=False),
        lambda r: r["runs"][0].update(d1_issues=["fail: x"]),
        lambda r: r["runs"][0].update(d1_sha256="0" * 64),  # inconsistent with entries_sha256
        lambda r: r["runs"][0].update(entries_sha256="zz"),
        lambda r: r.update(entries_sha256="0" * 64),  # inconsistent with the runs
        lambda r: r.update(check1=["x"]),
        lambda r: r.update(invalid=["invalid: x"]),
        lambda r: r.update(stage="check 2"),
        lambda r: r.update(library_sha256_loaded="f" * 64),
        lambda r: r.update(library_sha256_loaded=None, library_sha256_expected=None),
        lambda r: r.update(cells=0),
        lambda r: r.update(cells=True),
        lambda r: r.update(schema="other"),
    ],
)
def test_an_incomplete_malformed_or_inconsistent_record_does_not_pass(
    record: dict, spoil: object
) -> None:
    bad = copy.deepcopy(record)
    spoil(bad)  # type: ignore[operator]
    assert not h.passed(bad, SPEC)


@pytest.mark.parametrize(
    "change",
    [
        {"fields": []},
        {"policies": []},
        {"methods": ["lsq", "other"]},
        {"kind": "agglomerated"},
        {"n": 0},
        {"n": 2.0},
        {"seed": -1},
        {"perturbation": math.inf},
        {"fields": [{"a": 1.0, "G": None}]},
        {"fields": [{"a": math.nan, "G": [0, 0, 0]}]},
        {"fields": [{"a": 10**400, "G": [0, 0, 0]}]},
    ],
)
def test_an_invalid_spec_is_refused_and_never_passes(
    library: str, record: dict, change: dict
) -> None:
    spec = {**SPEC, **change}
    assert h.spec_issues(spec)
    rec = h.run_instance(spec, library, MANIFEST_SHA)
    assert rec["invalid"]
    assert not h.passed({**copy.deepcopy(record), "spec": spec}, spec)
    assert not h.passed("not a record", SPEC)


def test_an_exception_in_check_4_keeps_d1_and_every_earlier_step(
    library: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        msg = "injected"
        raise RuntimeError(msg)

    monkeypatch.setattr(h.kr, "check_bound", boom)
    seen: list[str] = []
    rec = h.run_instance(SPEC, library, MANIFEST_SHA, checkpoint=lambda r: seen.append(r["stage"]))
    assert rec["check1"] == [] and rec["check2"] == []  # noqa: PT018
    assert "d1_issues" in rec["runs"][0] and "d1_sha256" in rec["runs"][0]  # noqa: PT018
    assert any("raised during" in i for i in rec["invalid"])
    assert "injected" in rec["traceback"]
    assert len(seen) >= 4
    assert not h.passed(rec, SPEC)


def test_a_wrong_library_digest_stops_the_instance() -> None:
    rec = h.run_instance(SPEC, "0" * 64, MANIFEST_SHA)
    assert any("frozen library" in i for i in rec["invalid"])
    assert "check1" not in rec


def test_an_existing_record_file_is_never_overwritten(tmp_path: Path, library: str) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_bytes(MANIFEST)
    out = tmp_path / "rec.json"
    out.write_text("earlier attempt")
    rc = h.main(["run", "--manifest", str(manifest), "--instance", SPEC["id"],
                 "--library-sha256", library, "--out", str(out)])  # fmt: skip
    assert rc == 2
    assert out.read_text() == "earlier attempt"


def test_collection_passes_only_a_complete_bound_clean_stage(record: dict, library: str) -> None:
    attempts = {SPEC["id"]: [record]}
    assert h.collect(MANIFEST, MANIFEST_SHA, library, attempts, [aux(library)])["failures"] == []
    assert h.collect(MANIFEST + b" ", MANIFEST_SHA, library, attempts, [aux(library)])[
        "failures"
    ]  # tampered
    assert h.collect(MANIFEST, MANIFEST_SHA, library, {}, [aux(library)])["failures"]  # no record
    assert h.collect(MANIFEST, MANIFEST_SHA, "1" * 64, attempts, [aux(library)])["failures"]
    assert h.collect(MANIFEST, MANIFEST_SHA, library, attempts, [])["failures"]  # no auxiliary
    assert h.collect(MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: ["junk"]}, [aux(library)])[
        "failures"
    ]
    for bad in (aux(library, pytest_exit=1), aux(library, library_sha256_loaded=None),
                aux(library, stage="started"),
                aux(library, manifest_sha256="0" * 64), aux(library, outcomes={NODES[0]: "passed"}),
                aux(library, outcomes={NODES[0]: "passed", NODES[1]: "skipped"})):  # fmt: skip
        assert h.collect(MANIFEST, MANIFEST_SHA, library, attempts, [aux(library), bad])["failures"]
    empty = json.dumps({"instances": [], "auxiliary_tests": {"node_ids": NODES}}).encode()
    assert h.collect(empty, hashlib.sha256(empty).hexdigest(), library, {}, [aux(library)])[
        "failures"
    ]


def test_an_attempt_that_reported_an_issue_or_completed_without_passing_is_decisive(
    record: dict, library: str
) -> None:
    reported = copy.deepcopy(record)
    reported["check1"] = ["a topology issue"]
    short = copy.deepcopy(record)
    short["runs"][0]["bound_entries"] -= 1  # completed, no issue listed, but not passing
    for first in (reported, short):
        state = h.collect(
            MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: [first, record]}, [aux(library)]
        )
        assert any("decisive" in f for f in state["failures"])
        assert state["state"].startswith("FAIL")


def test_a_clean_interruption_may_be_superseded(record: dict, library: str) -> None:
    preflight = {"schema": h.SCHEMA, "id": SPEC["id"], "invalid": [], "stage": "manifest"}
    mid = {k: copy.deepcopy(v) for k, v in record.items() if k != "entries_sha256"}
    mid["stage"] = "geometry"
    mid["runs"] = []
    for first in (preflight, mid):
        state = h.collect(
            MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: [first, record]}, [aux(library)]
        )
        assert state["failures"] == []


def test_node_outcomes_need_a_passing_body_and_keep_the_worst_phase() -> None:
    def line(node: str, when: str, outcome: str) -> str:
        return json.dumps({"nodeid": node, "when": when, "outcome": outcome})

    lines = [
        line("tests/t.py::C::a[1]", "setup", "passed"), line("tests/t.py::C::a[1]", "call", "passed"),
        line("tests/t.py::C::a[1]", "teardown", "failed"),
        line("tests/t.py::b", "call", "failed"), line("tests/t.py::b", "call", "passed"),
        line("tests/t.py::c", "setup", "skipped"),
        line("tests/t.py::d", "setup", "passed"), line("tests/t.py::d", "teardown", "passed"),
        line("tests/t.py::e", "setup", "passed"), line("tests/t.py::e", "call", "passed"),
        line("tests/t.py::e", "teardown", "passed"),
        json.dumps({"library_sha256": "a" * 64, "when": "sessionstart"}),
        json.dumps({"library_sha256": "a" * 64, "when": "sessionfinish", "exitstatus": 0}),
    ]  # fmt: skip
    outcomes, libraries = h.node_outcomes(lines)
    assert outcomes == {
        "tests/t.py::C::a[1]": "failed",
        "tests/t.py::b": "failed",
        "tests/t.py::c": "skipped",
        "tests/t.py::d": "not run",
        "tests/t.py::e": "passed",
    }
    assert libraries == ["a" * 64, "a" * 64]


def test_auxiliary_evidence_needs_the_library_the_pytest_process_loaded(
    record: dict, library: str
) -> None:
    attempts = {SPEC["id"]: [record]}
    for child in ([], [library], ["f" * 64, library], [library, None]):
        state = h.collect(
            MANIFEST, MANIFEST_SHA, library, attempts, [aux(library, child_library_sha256=child)]
        )
        assert state["failures"], child


def test_a_malformed_attempt_is_decisive_even_when_bound(record: dict, library: str) -> None:
    bare = {
        "manifest_sha256": MANIFEST_SHA,
        "library_sha256_expected": library,
        "library_sha256_loaded": library,
    }
    for junk in (bare, {**bare, "schema": h.SCHEMA, "id": SPEC["id"], "stage": 3}):
        state = h.collect(
            MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: [junk, record]}, [aux(library)]
        )
        assert any("decisive" in f for f in state["failures"])


def test_a_record_file_that_looks_temporary_is_refused(tmp_path: Path, library: str) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_bytes(MANIFEST)
    for name in ("rec.tmp", "rec.part", ".rec.json"):
        rc = h.main(["run", "--manifest", str(manifest), "--instance", SPEC["id"],
                     "--library-sha256", library, "--out", str(tmp_path / name)])  # fmt: skip
        assert rc == 2
        assert not (tmp_path / name).exists()


def test_a_preflight_attempt_with_a_recorded_mismatch_is_not_forgiven(
    record: dict, library: str
) -> None:
    for field in ("library_sha256_loaded", "manifest_sha256", "library_sha256_expected"):
        pre = {
            "schema": h.SCHEMA,
            "id": SPEC["id"],
            "invalid": [],
            "stage": "manifest",
            field: "f" * 64,
        }
        state = h.collect(
            MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: [pre, record]}, [aux(library)]
        )
        assert state["failures"], field


def test_collection_is_total_on_malformed_authenticated_input(record: dict, library: str) -> None:
    for raw in (
        b"{not json",
        b"[1, 2]",
        json.dumps({"instances": [{"id": ["x"]}], "auxiliary_tests": {"node_ids": NODES}}).encode(),
    ):
        state = h.collect(
            raw, hashlib.sha256(raw).hexdigest(), library, {SPEC["id"]: [record]}, [aux(library)]
        )
        assert state["failures"]
    assert h.collect(MANIFEST, MANIFEST_SHA, library, "junk", "junk")["failures"]  # type: ignore[arg-type]


def test_an_early_invalid_outcome_is_checkpointed(library: str) -> None:
    seen: list[dict] = []
    h.run_instance(
        {**SPEC, "n": 0}, library, MANIFEST_SHA, checkpoint=lambda r: seen.append(copy.deepcopy(r))
    )
    assert seen[-1]["invalid"]
    seen.clear()
    h.run_instance(SPEC, "0" * 64, MANIFEST_SHA, checkpoint=lambda r: seen.append(copy.deepcopy(r)))
    assert any("frozen library" in i for i in seen[-1]["invalid"])


def _aux_run(tmp_path: Path, library: str, name: str, *extra: str) -> dict:
    manifest = tmp_path / "m.json"
    manifest.write_bytes(MANIFEST)
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "def test_one():\n    assert 1 + 1 == 2\n\n\ndef test_two():\n    assert True\n"
    )
    out = tmp_path / name
    rc = h.main(["aux", "--manifest", str(manifest), "--library-sha256", library, "--out", str(out),
                 "--", "-q", "-p", "no:cacheprovider", *extra, str(probe)])  # fmt: skip
    assert rc == 0
    return json.loads(out.read_text())


def test_the_auxiliary_wrapper_records_what_pytest_itself_ran(tmp_path: Path, library: str) -> None:
    ev = _aux_run(tmp_path, library, "aux.json")
    assert ev["stage"] == "done"
    assert ev["pytest_exit"] == 0
    assert ev["child_library_sha256"] == [library, library]  # measured inside the pytest process
    probe = next(iter(ev["outcomes"])).split("::")[0]
    assert ev["outcomes"] == {f"{probe}::test_one": "passed", f"{probe}::test_two": "passed"}


def test_setup_alone_never_makes_a_node_pass(tmp_path: Path, library: str) -> None:
    ev = _aux_run(tmp_path, library, "aux-setup-only.json", "--setup-only")
    assert ev["stage"] == "done"
    assert ev["pytest_exit"] == 0
    assert ev["child_library_sha256"] == [library, library]
    probe = next(iter(ev["outcomes"])).split("::")[0]
    assert set(ev["outcomes"]) == {f"{probe}::test_one", f"{probe}::test_two"}
    assert set(ev["outcomes"].values()) == {"not run"}


def test_an_unrecognized_report_line_is_never_dropped() -> None:
    for junk in ({"when": "call", "outcome": "skipped"}, {"nodeid": 3, "when": "call", "outcome": "passed"},
                 {"nodeid": "t::a", "when": "elsewhere", "outcome": "passed"}, ["x"]):  # fmt: skip
        with pytest.raises(ValueError, match="unrecognized"):
            h.node_outcomes([json.dumps(junk)])


@pytest.mark.parametrize("stage", ["", "bogus", "a=9.0 G=[0, 0, 0] fp64 lsq"])
def test_an_attempt_in_an_unknown_stage_is_decisive(record: dict, library: str, stage: str) -> None:
    odd = {"schema": h.SCHEMA, "id": SPEC["id"], "stage": stage, "invalid": [], "runs": [],
           "manifest_sha256": MANIFEST_SHA, "library_sha256_expected": library, "library_sha256_loaded": library}  # fmt: skip
    state = h.collect(MANIFEST, MANIFEST_SHA, library, {SPEC["id"]: [odd, record]}, [aux(library)])
    assert any("decisive" in f for f in state["failures"])
