"""Write the stage 1 manifest of ADR 0003's held-out validation (#43), deterministically.

Usage: python make_manifest.py FIXTURES_JSONL OUT_JSON NODE_IDS
NODE_IDS is `pytest --collect-only -q` of AUX_FILES (the oracle suite the auxiliary run executes):
every node id it lists, one per line, is frozen as an auxiliary test that must pass for stage 1.
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
    "oracle/freeze/nodeid_report.py", "test_oracle_freeze_tooling.py",
]  # fmt: skip
# The files the auxiliary run collects and runs (codegen/tests/...), in its order.
AUX_FILES = [
    "tests/test_oracle_certified.py", "tests/test_oracle_roots.py", "tests/test_oracle_tidy.py",
    "tests/test_oracle_topology.py", "tests/test_oracle_geometry.py", "tests/test_oracle_controls.py",
    "tests/test_oracle_d1.py", "tests/test_oracle_fixtures.py", "tests/test_oracle_heldout.py",
    "tests/test_spike_isolation.py",
]  # fmt: skip
# Stage 2 (#43: declaration 6786763697, amended 6787504754 and 6799616988): the oracle's own files
# and the auxiliary suite gain the vv-export reader, its tests, the watchdog and the driver.
FILES_STAGE2 = [
    *FILES, "oracle/vvexport.py", "oracle/freeze/watchdog.py", "oracle/freeze/collect_stage2.py",
    "test_oracle_vvexport.py", "test_oracle_stage2.py",
]  # fmt: skip
AUX_FILES_STAGE2 = [
    *AUX_FILES, "tests/test_oracle_vvexport.py", "tests/test_oracle_stage2.py",
]  # fmt: skip
# The hosted vv-meshes runs on main d7a8714c (export_cells, gmsh 4.15.2), copied verbatim to
# spark-5512 ~/cfd/m1-stage2/data/<family>/ (SHA256SUMS 0f55c4ae...).
STAGE2_RUNS = {
    (0.0, 11): ("meshes-agglomerated-p0.0-s11", 42952304018),
    (0.15, 11): ("meshes-agglomerated-p0.15-s11", 42952307240),
    (0.15, 13): ("meshes-agglomerated-p0.15-s13", 42952310343),
}
# N16 (branch a'), generated after the N16 measurement picked a' by the same workflow at the same
# commit d7a8714c (branch hpc/vv-meshes-d7a8714c), one level per run, copied verbatim to
# ~/cfd/m1-stage2/data/<family>-n16/. (n, perturbation, seed) -> (family directory, run).
STAGE2_RUNS_BY_LEVEL = {
    (16, 0.15, 11): ("meshes-agglomerated-p0.15-s11-n16", 43115495266),
    (16, 0.15, 13): ("meshes-agglomerated-p0.15-s13-n16", 43115499012),
}
# (n, perturbation, seed) -> (mesh file, its SHA-256, the export's SHA-256): the definitive hashes
# (#43, reports/stage2-definitive-hashes-20261111.txt; N16 from runs 43115495266 and 43115499012).
STAGE2_FILES: dict[tuple[int, float, int], tuple[str, str, str]] = {
    (12, 0.0, 11): ("agglomerated-n012.msh", "3b6110d8ba18971343975982d3bccf795715dcff60e68ef6a469876a5f37ca1c", "5785d4e0e1740878f2383c198983206d23406b8e0de8e00b104b77a099f1a809"),
    (24, 0.0, 11): ("agglomerated-n024.msh", "75f201d181aba27141fafe4e1aacdfd583d7410152cc205b7540e624e977563b", "304cc706cbed0ca5d496b8ec94583eb1baa2dd51051ffeee3fd4710977dc50f4"),
    (6, 0.15, 11): ("agglomerated-dirty-n006.msh", "3bbbaae2c7974c45fb47f3f1a142ebef9ca888e9a028d7d6fe1abbd83f5d9a38", "3f6ede7623e52462b41b516597fe747c8733950e25d028d794bb8b47e0ff52da"),
    (12, 0.15, 11): ("agglomerated-dirty-n012.msh", "c92b4654324882ae39c28df2a214139bcdb116d8c80b0f54adceff27775e09f9", "1b59af7eba19fd46c28bfc33501ce6469660ab000b9034c2e81a23351c831527"),
    (24, 0.15, 11): ("agglomerated-dirty-n024.msh", "efc898ede087bfafcc2b8712b1bccc4ee5e130d3c380f7582f3c797a9900df9f", "13a8db369da7949f2bad966f5153b87887af89eeb40ec2045cff9874141bac44"),
    (6, 0.15, 13): ("agglomerated-dirty-n006.msh", "8bf8290b21ea368806f86d1e643464d6ba1b7541e896fe54bb675a9e7d188268", "cd6aea6579731dbd924b8824209fb81bec31c95cffeff2dae6ca853e13aa019e"),
    (12, 0.15, 13): ("agglomerated-dirty-n012.msh", "a225c0b1c359926a8a62d35b1d115a6d79d47b728583674e50790e3c846121ff", "c1d5a818049edf6868e217e486d416c492ee006b410aaf0571cb323121e97596"),
    (24, 0.15, 13): ("agglomerated-dirty-n024.msh", "5fae1f4ad111f90a18ae0348cf3b90fb7fa7f361fa4a5f383a59456294648c1e", "c56f570cf1ab7aebc37b185ee4b647350b2f837dd027c62f5358951db76d9f3d"),
    (16, 0.15, 11): ("agglomerated-dirty-n016.msh", "c5d273a688573aa7b580fb73bfb4b1cabe3529baf2f68ed70fc461e9018bb411", "c02c8dbc6b9fecccc3c5f4421a32850de8f228e142cd954aeabaf4b4219a64ca"),
    (16, 0.15, 13): ("agglomerated-dirty-n016.msh", "44761629a94dfd1d5fa91662a70798057acacba3435c8c235001211ac6841d8a", "f01af8b4c962abbd20a8e1ab10745e1c0caa80d938056537fc9cbcf12270d9cc"),
}  # fmt: skip
STAGE2_LEVELS = {"a": (6, 12), "a-prime": (6, 12, 16)}
# Excluded candidates, each with its reason. N16 p0 s11 (#43, 6809100033, the orchestrator's decision
# of 2026-11-11 before the N16 result): the oracle ran this very mesh before the freeze, as the
# N16 resource measurement (file 69d58408..., mesh 5dfb6d5e...: at p0 the seed has no effect).
STAGE2_EXCLUDED = {
    (6, 0.0, 11): "byte-identical to test_meshes' N6 p0 s3, a mesh a vv test generates (declaration section 3)",
    (16, 0.0, 11): "byte-identical to the N16 resource measurement's mesh (N16 p0 s1), which the oracle ran before the freeze (#43, 6809100033)",
}  # fmt: skip
# 3 x the measured N6 (576 s), N12 (12,909 s) and N16 (29,061 s; s2t2-n16, branch a').
STAGE2_BOUNDS_S = {"6": 1728, "12": 38727, "16": 87183}
STAGE2_AUX_BOUND_S = 5799  # 3 x 1,933 s, #668's GB10 run on spark-5512
STAGE2_RETRY_CAP = 2
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


def instances_stage2(branch: str, files: dict, bounds_s: dict[str, int]) -> list[dict]:
    """Stage 2's instances for the branch the N16 measurement picked (#43, 6799616988): agglomerated,
    N in the branch's levels x (p0 s11, p0.15 s11, p0.15 s13), minus the excluded ones. Each names
    its two files by SHA-256 and its family directory under ~/cfd/m1-stage2/data/."""
    out = []
    for n in STAGE2_LEVELS[branch]:
        if str(n) not in bounds_s:
            msg = f"no frozen bound for N = {n}"
            raise SystemExit(msg)
        for p, seed in ((0.0, 11), (0.15, 11), (0.15, 13)):
            if (n, p, seed) in STAGE2_EXCLUDED:
                continue
            if (n, p, seed) not in files:
                msg = f"no definitive hashes for N{n} p{p} s{seed}"
                raise SystemExit(msg)
            mesh, mesh_sha, cells_sha = files[(n, p, seed)]
            family, run = STAGE2_RUNS_BY_LEVEL.get((n, p, seed)) or STAGE2_RUNS[(p, seed)]
            out.append({
                "id": f"s2-agglomerated-N{n}-p{p}-s{seed}", "kind": "agglomerated", "n": n,
                "perturbation": p, "seed": seed, "source": "vv-export",
                "files": {
                    "mesh": {"name": mesh, "sha256": mesh_sha},
                    "cells": {"name": mesh.removesuffix(".msh") + ".cells.json", "sha256": cells_sha},
                },
                "data": family, "provenance": {"vv_meshes_run": run, "attempt": 1},
                "fields": FIELDS, "policies": POLICIES, "methods": METHODS,
                "bound_s": bounds_s[str(n)], "machine": "spark-5512",
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


def node_ids(collected: str) -> list[str]:
    """The node ids of `pytest --collect-only -q` output: one per line, whole. A node id may hold
    spaces (parametrize ids such as `[a=9.0 G=[0, 0, 0] fp64 lsq]`), so lines are never split on
    whitespace (stage 1 froze a truncated id that way, #43)."""
    return [line.strip() for line in collected.splitlines() if "::" in line]


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

    if "--stage" in sys.argv:
        return main_stage2(kr)

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
            "node_ids": node_ids(Path(sys.argv[3]).read_text()),
        },
        "controls": {"tests": ["codegen/tests/test_oracle_controls.py", "codegen/tests/test_oracle_geometry.py"], "rule": "every control demonstrated and failing as stated, with its proof at the output"},
        "reporting": {"per_class_radii": "a separate pass after the stage results, on the same frozen instances; reporting only, never read by the gate"},
        "verdict": "fail-closed (heldout.collect): every instance's record present and passing with full coverage, and the fixtures, boundary cases and controls passing; stage 1 results are recorded, not a verdict; no PASS until stage 2; a stage 1 failure stays a failure",
    }  # fmt: skip
    Path(sys.argv[2]).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


def main_stage2(kr: object) -> int:
    """make_manifest.py FIXTURES OUT COLLECTED --stage 2 --branch a|a-prime: stage 2's manifest,
    written once the branch is known."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures")
    ap.add_argument("out")
    ap.add_argument("collected")
    ap.add_argument("--stage", type=int, choices=[2], required=True)
    ap.add_argument("--branch", choices=sorted(STAGE2_LEVELS), required=True)
    args = ap.parse_args()
    files = dict(STAGE2_FILES)
    bounds = {str(n): STAGE2_BOUNDS_S[str(n)] for n in STAGE2_LEVELS[args.branch]}
    instances = instances_stage2(args.branch, files, bounds)
    runs = {f"p{p} s{seed}": run for (p, seed), (_, run) in STAGE2_RUNS.items()}
    runs |= {
        f"N{n} p{p} s{seed}": run
        for (n, p, seed), (_, run) in STAGE2_RUNS_BY_LEVEL.items()
        if n in STAGE2_LEVELS[args.branch]
    }
    aux_ids = node_ids(Path(args.collected).read_text())
    files_seen = {i.split("::")[0] for i in aux_ids}
    if files_seen != set(AUX_FILES_STAGE2):
        msg = f"the collected node ids come from {sorted(files_seen)}, not from {AUX_FILES_STAGE2}"
        raise SystemExit(msg)
    manifest = {
        "schema": "autan.adr0003-manifest/1",
        "stage": 2,
        "adr": "docs/adr/0003-derived-linear-exactness-bound.md",
        "declaration": "https://github.com/Remenby31/autan/issues/43#issuecomment-6786763697, amended 6787504754 (vertices) and 6799616988 (levels, bounds, window)",
        "branch": args.branch,
        "oracle_sha256": {f"codegen/tests/{f}": sha256(TESTS / f) for f in FILES_STAGE2},
        "frozen_code_sha256": dict(kr.FROZEN_SHA256),  # type: ignore[attr-defined]
        "library": "the SHA-256 of the spike library built from this commit is in the freeze record, not here",
        "instances": instances,
        "bounds_s": bounds,
        "aux_bound_s": STAGE2_AUX_BOUND_S,
        "retry_cap": STAGE2_RETRY_CAP,
        "data": {"root": "spark-5512 ~/cfd/m1-stage2/data", "runs": runs},
        "excluded": {"rule": "exclude any candidate whose file SHA-256 or mesh SHA-256 equals that of a mesh generated by a vv test, the list re-hashed at the frozen commit", "excluded": [f"N{n} p{p} s{s}" for n, p, s in sorted(STAGE2_EXCLUDED) if n in STAGE2_LEVELS[args.branch]], "reasons": {f"N{n} p{p} s{s}": why for (n, p, s), why in sorted(STAGE2_EXCLUDED.items()) if n in STAGE2_LEVELS[args.branch]}},
        "positive_fixtures": fixtures(Path(args.fixtures)),
        "auxiliary_tests": {
            "rule": "the oracle suite on the frozen build, the stage 2 tests included: every node id below must pass (none skipped, deselected or missing), within aux_bound_s",
            "files": AUX_FILES_STAGE2,
            "node_ids": aux_ids,
        },
        "verdict": "fail-closed (heldout.collect, stage 2): every instance's attempt bound, done and passing all its combinations within its bound, at most retry_cap attempts, and the auxiliary suite passing within aux_bound_s; stated as 'acceptance criteria met' on these frozen exports only",
    }  # fmt: skip
    Path(args.out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
