"""Evaluate every declared positive-fixture candidate (test_oracle_fixtures.CANDIDATES) on
development (B) sizes: one JSON line per candidate x policy x method, read by make_manifest.py.
Usage (in codegen/, the spike built): python tests/oracle/freeze/fixtures_eval.py > fixtures.jsonl"""

import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[3] / "spike"),
]

import autan_spike as s
import meshes
import numpy as np
from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t
from test_oracle_fixtures import CANDIDATES

POLICIES = ("fp64", "fp32+acc64", "fp32")

for name, (kind, n, p, seed, a, g) in CANDIDATES.items():
    m = s.mesh(meshes.build(kind, n, p, seed=seed))
    read = t.read_mesh(m)
    pts, cells, ag = meshes.cells(kind, n, p, seed=seed)
    check1 = t.compare(t.derive(pts, cells, ag), read)
    geometry, exact = q.construct(read), q.geometry_exact(read)
    certs = {k: [[pair.c for pair in row] for row in rows] for k, rows in geometry.items()}
    check2 = q.check_geometry(read, q.stored_geometry(m), certs, exact)
    geo_r0 = all(cert.c.radius == 0 for rows in geometry.values() for row in rows for cert in row)
    n_int = len(read.neighbour)
    qv, qb = kr.field_floats(geometry, n_int, a, g)
    for pol in POLICIES:
        for method in ("lsq", "gg"):
            if method == "lsq":
                out = s.gradients(m, pol, "lsq", np.array(qv), np.array(qb))[0]
            else:
                out = s.gradients(
                    m,
                    pol,
                    "gg",
                    np.array(qv),
                    np.array(qb),
                    1,
                    np.tile(np.array(g), (m.n_cells, 1)),
                )[0]
            b = kr.check_bound(read, geometry, exact, a, g, out.tolist(), pol, method)
            field = kr.linear_field(geometry["cell_centroid"], a, g)
            fb = [kr.given(0.0, kr.B64)] * n_int + kr.linear_field(
                geometry["face_centroid"][n_int:], a, g
            )
            grad0 = [[kr.given(v, kr.B64) for v in g]] * m.n_cells if method == "gg" else None
            d1 = kr.check_d1(
                read, kr.Inputs(geometry, field, fb, grad0), exact, out.tolist(), pol, method
            )
            every = {(c, u) for c in range(m.n_cells) for u in range(3)}
            covered = m.n_cells > 0 and set(b.bounds) == every and set(d1.bounds) == every
            r0 = covered and all(v == 0 for v in b.bounds.values())
            exact_g = all(
                Fr(float(out[c, u])) == Fr(g[u]) for c in range(m.n_cells) for u in range(3)
            )
            print(json.dumps({
                "candidate": name, "policy": pol, "method": method, "cells": m.n_cells,
                "check1": len(check1), "check2": len(check2), "geometry_radius_zero": geo_r0,
                "d1_issues": len(d1.issues), "bound_issues": len(b.issues),
                "covers_every_entry": covered, "certifies_R0": r0, "g_hat_equals_G": exact_g,
            }), flush=True)  # fmt: skip
