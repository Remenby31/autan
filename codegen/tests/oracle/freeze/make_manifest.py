"""Write the stage 1 manifest of ADR 0003's held-out validation (#43), deterministically.

Usage: python make_manifest.py FIXTURES_JSONL OUT_JSON NODE_IDS
NODE_IDS is `pytest --collect-only -q` of the oracle suite: every node id it lists is frozen as an
auxiliary test that must pass for stage 1.
FIXTURES_JSONL is the evaluation of every declared positive-fixture candidate on development (B)
sizes (fixtures_eval output); the manifest records all candidates and their outcomes, not only the
approved ones. The spike library's digest is NOT in this file: the library is built from the commit
that holds it; its SHA-256 goes into the freeze record written after that build.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ORACLE = Path(__file__).resolve().parents[1]  # codegen/tests/oracle
TESTS = ORACLE.parent
FILES = [
    "oracle/certified.py", "oracle/sequences.py", "oracle/topology.py", "oracle/kernels.py",
    "oracle/heldout.py", "test_oracle_certified.py", "test_oracle_roots.py", "test_oracle_tidy.py",
    "test_oracle_topology.py", "test_oracle_geometry.py", "test_oracle_d1.py",
    "test_oracle_controls.py", "test_oracle_fixtures.py", "test_spike_isolation.py",
    "test_oracle_heldout.py", "oracle/freeze/make_manifest.py", "oracle/freeze/fixtures_eval.py",
    "oracle/freeze/equivalence.py",
    "oracle/freeze/nodeid_report.py",
]  # fmt: skip
POLICIES = ["fp64", "fp32+acc64", "fp32"]
METHODS = ["lsq", "gg"]
FIELDS = [{"a": -7.5, "G": [-0.9, 0.4, 1.3]}, {"a": 250.0, "G": [-0.9, 0.4, 1.3]}]
# Minutes for one instance at N = 8, measured on B (one field, one policy, scaled to the declared
# work), times (N / 8)^3, times a margin of 1.5. Upper bounds for the queue, not a protocol item.
N8_MINUTES = {"hex": 3.3, "poly": 6.7, "hybrid": 10.0, "tet": 18.0}
PLACEMENT = {
    "rule": "every stage 1 instance on spark-5512 (no deadline); spark-5359 is not used for the held-out",
    "machine_independent": "an instance's record does not depend on the machine: the same library bytes (one build, its SHA-256 checked by the runner on every instance), CPU kernels only, the oracle in exact rational arithmetic (Python fractions)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instances() -> list[dict]:
    out = []
    for kind in ("hex", "tet", "hybrid", "poly"):
        for n in (6, 12, 24):
            # Clean meshes do not depend on the seed (meshes.py: the generator draws only when the
            # perturbation is nonzero); the clean instance takes seed 11, a held-out seed.
            for p, seeds in ((0.0, [11]), (0.15, [11, 13])):
                for seed in seeds:
                    est = max(5, round(N8_MINUTES[kind] * (n / 8) ** 3 * 1.5))
                    machine = "spark-5512"
                    out.append({
                        "id": f"s1-{kind}-N{n}-p{p}-s{seed}", "kind": kind, "n": n,
                        "perturbation": p, "seed": seed, "fields": FIELDS,
                        "policies": POLICIES, "methods": METHODS, "est_min": est,
                        "machine": machine,
                    })  # fmt: skip
    return out


# Why the candidates that do not certify do not (observed on B; the mechanism is a lead, not a
# confirmed cause). Uniform dyadic hex and poly, N = 4: face_area and face_moment certify R = 0, but
# face_centroid (480 of 720 entries on hex), cell_centroid, cell_volume and every lsq_weight carry
# R > 0 (at most 4.9e-13 relative on hex, 1.3e-11 on poly).
WHY_NOT = {
    "uniform-hex-linear-N4": "R > 0: a linear field carries the geometry radii (face/cell centroids, volumes, weights) into the gradient; g_hat = G exactly for LSQ and for fp32 GG, not for fp64 GG",
    "uniform-hex-linear-N8": "as uniform-hex-linear-N4",
    "uniform-hex-zero-component-N4": "as uniform-hex-linear-N4 (a zero component does not remove the radii of the others)",
    "single-cell-hex-linear": "as uniform-hex-linear-N4 (boundary case: it passes checks 1-4, it does not certify R = 0)",
    "hex-N3-linear": "as uniform-hex-linear-N4 with N = 3 (non-dyadic 1/3): neither R = 0 nor g_hat = G exactly (boundary case: it passes checks 1-4)",
    "tet-constant-N4": "LSQ certifies R = 0 (zero differences); Green-Gauss has R > 0 (q times the area and moment sums, whose geometry radii are nonzero), with g_hat = G exactly",
    "hybrid-constant-N4": "as tet-constant-N4",
    "hex-perturbed-constant-N4": "LSQ certifies R = 0; Green-Gauss has R > 0 and g_hat != G exactly (perturbed geometry)",
}  # fmt: skip


def fixtures(path: Path) -> dict:
    from test_oracle_fixtures import APPROVED_EXACT, CANDIDATES  # noqa: PLC0415

    by = defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        by[r["candidate"]].append(r)
    if set(by) != set(CANDIDATES):
        msg = (
            f"the evaluation covers {sorted(by)}, the declared candidates are {sorted(CANDIDATES)}"
        )
        raise SystemExit(msg)
    combos = {(p, m) for p in POLICIES for m in METHODS}
    candidates = []
    for name, rows in by.items():
        got = [(r["policy"], r["method"]) for r in rows]
        if len(got) != len(combos) or set(got) != combos:
            msg = f"{name}: rows {sorted(got)}, expected every policy x method once"
            raise SystemExit(msg)
        clean = all(
            r["check1"] + r["check2"] + r["d1_issues"] + r["bound_issues"] == 0 for r in rows
        )
        covered = all(r.get("covers_every_entry") is True for r in rows)
        certifies = (
            clean and covered and all(r["certifies_R0"] and r["g_hat_equals_G"] for r in rows)
        )
        kind, n, pert, seed, a, g = CANDIDATES[name]
        candidates.append({
            "name": name,
            "spec": {"kind": kind, "n": n, "perturbation": pert, "seed": seed, "a": a, "G": list(g)},
            "certifies_R0_and_g_hat_equals_G_everywhere": certifies,
            "per_policy_method": {f"{r['policy']}/{r['method']}": {"R0": r["certifies_R0"], "g_hat_equals_G": r["g_hat_equals_G"]} for r in rows},
            "geometry_radius_zero": rows[0]["geometry_radius_zero"],
            "covers_every_entry": covered,
            "checks_1_to_4_issues": sum(r["check1"] + r["check2"] + r["d1_issues"] + r["bound_issues"] for r in rows),
            "why_not_certified": None if certifies else WHY_NOT[name],
        })  # fmt: skip
    approved = sorted(
        c["name"] for c in candidates if c["certifies_R0_and_g_hat_equals_G_everywhere"]
    )
    if not approved:
        msg = "the approved exact set is empty: stop and report (ADR 0003)"
        raise SystemExit(msg)
    if approved != sorted(APPROVED_EXACT):
        msg = f"certified {approved}, but the tests assert {sorted(APPROVED_EXACT)}"
        raise SystemExit(msg)
    return {
        "criterion": "certified exact: R = 0 on every cell and component, g_hat = G exactly, and checks 1-4 clean, for every policy and method (ADR 0003, positive fixtures); selected on B sizes among the declared candidates",
        "candidates": candidates,
        "approved": approved,
        "test": "codegen/tests/test_oracle_fixtures.py",
    }


def main() -> int:
    sys.path.insert(0, str(TESTS))
    from oracle import kernels as kr  # noqa: PLC0415

    manifest = {
        "schema": "autan.adr0003-manifest/1",
        "stage": 1,
        "adr": "docs/adr/0003-derived-linear-exactness-bound.md",
        "oracle_sha256": {f"codegen/tests/{f}": sha256(TESTS / f) for f in FILES},
        "frozen_code_sha256": dict(kr.FROZEN_SHA256),
        "precision_contract": {"storage": kr.DECLARED_STORAGE, "compute": kr.DECLARED_COMPUTE, "roles": kr.DECLARED_ROLES, "loops": {k: list(v) for k, v in kr.DECLARED_LOOPS.items()}},
        "library": "the SHA-256 of the spike library built from this commit is in the freeze record, not here",
        "instances": instances(),
        "placement": PLACEMENT,
        "agglomerated": "stage 2: vv's agglomerated family, frozen and approved separately; nothing run on it before",
        "positive_fixtures": fixtures(Path(sys.argv[1])),
        "boundary_cases": {"test": "codegen/tests/test_oracle_fixtures.py::test_a_boundary_case_passes_checks_1_to_4"},
        "auxiliary_tests": {
            "rule": "the oracle suite on the frozen build: every node id below must pass (none skipped, deselected or missing); it holds the positive fixtures, the boundary cases, every control, the contract and frozen-code checks and the runner's own tests",
            "node_ids": [n for n in Path(sys.argv[3]).read_text().split() if "::" in n],
        },
        "controls": {"tests": ["codegen/tests/test_oracle_controls.py", "codegen/tests/test_oracle_geometry.py"], "rule": "every control demonstrated and failing as stated, with its proof at the output"},
        "reporting": {"per_class_radii": "a separate pass after the stage results, on the same frozen instances; reporting only, never read by the gate"},
        "verdict": "fail-closed (heldout.collect): every instance's record present and passing with full coverage, and the fixtures, boundary cases and controls passing; stage 1 results are recorded, not a verdict; no PASS until stage 2; a stage 1 failure stays a failure",
    }  # fmt: skip
    Path(sys.argv[2]).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
