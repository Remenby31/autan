"""Equivalence of the build-once runner (ADR 0003 freeze), on development (B) instances only.

For each instance:
- the certificates projected from the single construction equal, Cert by Cert, those check 2
  rebuilds (sequences.geometry_certified), and two evaluations of the exact geometry are equal;
- on corrupted stored geometry (areas scaled by 2, one centroid moved), check 2 gives the same,
  non-empty, issues with the geometry reused and rebuilt;
- the instance's records with the geometry reused and rebuilt are identical, value for value.
Usage (in codegen/, the spike built): python tests/oracle/freeze/equivalence.py KIND[,KIND...]"""

import copy
import json
import math
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[3] / "spike"),
]

import autan_spike as s
import meshes
from oracle import heldout as h
from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t

library = kr.library_sha256()
assert library is not None
failures = 0
fields = [{"a": 1.0, "G": [0.3, -1.2, 0.7]}, {"a": 1000.0, "G": [0.3, -1.2, 0.7]}]
for kind in sys.argv[1].split(","):
    for n in (4, 8):
        for p in (0.0, 0.15):
            m = s.mesh(meshes.build(kind, n, p, seed=3))
            read = t.read_mesh(m)
            geometry = q.construct(read)
            once = {k: [[pair.c for pair in row] for row in rows] for k, rows in geometry.items()}
            certs_equal = once == q.geometry_certified(read)
            exact_once = q.geometry_exact(read)
            exact_equal = exact_once == q.geometry_exact(read)
            bad = q.stored_geometry(m)
            bad["face_area"] = [[2 * v for v in row] for row in bad["face_area"]]
            row = list(bad["cell_centroid"][0])
            row[0] = math.nextafter(math.nextafter(row[0], math.inf), math.inf) + 1e-9
            bad["cell_centroid"][0] = row
            reused = q.check_geometry(read, copy.deepcopy(bad), once, exact_once)
            rebuilt = q.check_geometry(read, copy.deepcopy(bad))
            corrupted_equal = reused == rebuilt and bool(reused)
            spec = {"id": f"B-{kind}-N{n}-p{p}-s3", "kind": kind, "n": n, "perturbation": p, "seed": 3,
                    "fields": fields, "policies": ["fp64", "fp32+acc64", "fp32"], "methods": ["lsq", "gg"]}  # fmt: skip
            a = h.run_instance(spec, library, "0" * 64, reuse_geometry=True)
            b = h.run_instance(spec, library, "0" * 64, reuse_geometry=False)
            records_equal = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            ok = (
                certs_equal
                and exact_equal
                and corrupted_equal
                and records_equal
                and h.passed(a, spec)
            )
            failures += not ok
            print(json.dumps({
                "id": spec["id"], "certs_equal": certs_equal, "exact_equal": exact_equal,
                "corrupted_check2_equal_and_nonempty": corrupted_equal, "corrupted_issues": len(reused),
                "records_identical": records_equal, "passed": h.passed(a, spec),
                "entries_sha256": a.get("entries_sha256"), "cells": a.get("cells"),
            }), flush=True)  # fmt: skip
if failures:
    sys.exit(f"{failures} instance(s) failed the equivalence")
