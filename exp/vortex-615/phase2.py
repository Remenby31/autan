"""#615 vortex-order study, phase 2 (declared at
https://github.com/Remenby31/autan/issues/615#issuecomment-6783705879): runs R0, RP, RH, RA on
vortex-hex n = 64, 128, 256, 512 at dt = 2/512, the 2 x 2 of initial state x reference, the
fail-closed control against #615, and the frozen checks of the manifest.

    python phase2.py prepare   FAMILY CASES WORK         # quadrature checks, the variants' cases
    python phase2.py roundtrip WORK AUTAN_RUN            # --stop-file before the first step
    python phase2.py run       WORK AUTAN_RUN VARIANT [N] # one variant's four levels, or level N
    python phase2.py measure   WORK REPORTS OUT.json     # readings (needs every step above)

FAMILY is `autan-vv meshes vortex-hex --levels 64,128,256,512`; CASES is `autan-vv vortex-cases`
on it (R0's cases, finest = 512); REPORTS holds #615's verify-v0-up.json and verify-k-n512.json.
WORK must not exist before `prepare`. Every step needs OMP_NUM_THREADS=1, and every run and
roundtrip the same AUTAN_RUN (its SHA-256 is recorded and checked). Exit 0: the step is done and
all its checks passed; 1: a check failed or the step could not be measured; 2: usage. The rhow
preservation test is a reading (it enters each band, as the gate's), not an instrument check.
"""

import dataclasses
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import autan
import numpy as np
from autan.initial_state import write as write_state
from autan.meshinfo import MeshInfo, mesh_sha256
from autan.results import Devices
from autan.solver_input import build
from autan_vv.harness import read_geometry
from autan_vv.meshes.families import generate_level
from autan_vv.oracles import conserved
from autan_vv.oracles.inviscid import GAMMA, isentropic_vortex

LEVELS = (64, 128, 256, 512)
VARS = ("rho", "rhou", "rhov", "rhoE")
ALL = (*VARS, "rhow")
PRIM = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")
VARIANTS = {
    "R0": ("centroid", "roe"),
    "RP": ("average", "roe"),
    "RH": ("centroid", "hllc"),
    "RA": ("centroid", "ausm+up"),
}
T_END = 40.0
EPS = float(np.finfo(np.float64).eps)
FLOOR_FACTOR = 100.0  # gates.FLOOR_FACTOR
MIN_LEVELS = 4  # gates.MIN_LEVELS
PAIRS = ("64->128", "128->256", "256->512")
CPU = Devices(kind="cpu", count=1)


class Failed(Exception):
    pass


def fail(msg):
    raise Failed(msg)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, document):
    with Path(path).open("w") as f:
        json.dump(document, f, indent=1)


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def gauss(m):
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w  # on [0, 1], weights summing to 1


def oracle_conserved(points, t):
    f = isentropic_vortex().fields(points, t)
    return conserved(f, GAMMA)


def primitives_to_conserved(values):
    v = np.asarray(values, dtype=float)
    f = dict(zip(PRIM, v.T, strict=True))
    return conserved(f, GAMMA)


def cell_boxes(mesh):
    """Each cell's x and y bounds from the mesh's own nodes (vortex-hex cells are axis-aligned)."""
    counts = np.diff(mesh.face_offsets)
    lo = np.full((mesh.n_cells, 2), np.inf)
    hi = np.full((mesh.n_cells, 2), -np.inf)
    nodes = mesh.points[mesh.face_nodes][:, :2]
    owner = np.repeat(mesh.owner, counts)
    neighbour = np.repeat(mesh.neighbour, counts[: mesh.n_interior])
    for cells, vals in ((owner, nodes), (neighbour, nodes[: neighbour.size])):
        np.minimum.at(lo, cells, vals)
        np.maximum.at(hi, cells, vals)
    return lo, hi


def averages(lo, hi, z, t, m):
    """Cell averages of the conserved variables by an m x m Gauss-Legendre tensor rule."""
    s, w = gauss(m)
    acc = {k: np.zeros(lo.shape[0]) for k in ALL}
    for (si, wi), (sj, wj) in itertools.product(zip(s, w, strict=True), zip(s, w, strict=True)):
        x = lo[:, 0] + si * (hi[:, 0] - lo[:, 0])
        y = lo[:, 1] + sj * (hi[:, 1] - lo[:, 1])
        u = oracle_conserved(np.column_stack([x, y, z]), t)
        for k in ALL:
            acc[k] += wi * wj * u[k]
    return acc


def scales(fields):
    return {k: float(np.max(np.abs(fields[k]))) for k in ALL}


def close(a, b, scale_k, k):
    if k == "rhow":
        return bool(float(np.max(np.abs(a - b))) <= 1e-14)
    return bool(float(np.max(np.abs(a - b))) <= 1e-12 * scale_k)


def l2(weight, d):
    return float(np.sqrt(np.sum(weight * d * d)))


def verify_errors(u, ref, weight):
    """The custom verifier: volume-weighted L2 error of each conserved variable against `ref`."""
    return {k: l2(weight, u[k] - ref[k]) for k in ALL}


def require_env():
    if os.environ.get("OMP_NUM_THREADS") != "1":
        fail("OMP_NUM_THREADS must be 1")


def prepare(family, cases, work):
    require_env()
    if work.exists():
        fail(f"{work} exists: prepare starts from an empty WORK")
    work = work.resolve()
    work.mkdir(parents=True)
    cases = cases.resolve()
    manifest = read_json(family / "manifest.json")
    s, w = gauss(6)
    moments = True
    for a in range(12):
        for b in range(12):
            q = sum(
                float(wi * wj * si**a * sj**b)
                for si, wi in zip(s, w, strict=True)
                for sj, wj in zip(s, w, strict=True)
            )
            moments = moments and abs(q - 1.0 / ((a + 1) * (b + 1))) <= 1e-13
    checks = {"moments": bool(moments), "levels": {}}
    ok = checks["moments"]
    for level in manifest["levels"]:
        n = level["n"]
        mesh, _ = generate_level("vortex-hex", n, seed=manifest["seed"])
        mesh_file = (family / level["file"]).resolve()
        if mesh.sha256() != level["mesh_sha256"]:
            fail(f"n = {n} does not regenerate from its manifest")
        geo = mesh.geometry
        weight = geo.cell_volume / geo.cell_volume.sum()
        lo, hi = cell_boxes(mesh)
        area = (hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1])
        c = {"area": bool(np.all(np.abs(area - geo.cell_volume) <= 1e-13 * geo.cell_volume))}
        z = geo.cell_centroid[:, 2]
        for t in (0.0, T_END):
            a6, a12 = averages(lo, hi, z, t, 6), averages(lo, hi, z, t, 12)
            sc = scales(a12)
            c[f"6v12 t={t:g}"] = all(close(a6[k], a12[k], sc[k], k) for k in ALL)
            c[f"6v12 L2 t={t:g}"] = {k: l2(weight, a6[k] - a12[k]) for k in ALL}
            np.savez(work / f"avg-n{n:03d}-t{t:g}.npz", **a6)
        a = dict(np.load(work / f"avg-n{n:03d}-t0.npz"))
        rho = a["rho"]
        m = np.column_stack([a["rhou"], a["rhov"], a["rhow"]])
        u = m / rho[:, None]
        p = (GAMMA - 1.0) * (a["rhoE"] - 0.5 * np.sum(m * m, axis=1) / rho)
        origin = mesh.points[0]
        centroids = (
            dataclasses.replace(mesh, points=mesh.points - origin).geometry.cell_centroid + origin
        )
        state = work / f"vortex-hex-n{n:03d}.average.initial.json"
        digest = write_state(
            state,
            oracle="isentropic-vortex",
            mesh_sha256=mesh_sha256(mesh_file),
            centroids=centroids.tolist(),
            values=np.column_stack([rho, u, p]).tolist(),
        )
        stem = Path(level["file"]).stem
        base = (cases / f"{stem}.toml").read_text()
        old_state = base.split("[initial_state]")[1].split('path = "')[1].split('"')[0]
        old_sha = base.split("[initial_state]")[1].split('sha256 = "')[1].split('"')[0]
        old_mesh = base.split('\nmesh = "')[1].split('"')[0]
        for name, (init, flux) in VARIANTS.items():
            text = base.replace(f'\nmesh = "{old_mesh}"', f'\nmesh = "{mesh_file}"', 1)
            if init == "average":
                text = text.replace(f'path = "{old_state}"', f'path = "{state}"').replace(
                    old_sha, digest
                )
            else:
                text = text.replace(
                    f'path = "{old_state}"', f'path = "{(cases / old_state).resolve()}"'
                )
            if flux != "roe":
                if text.count('flux = "roe"') != 1:
                    fail("the case does not name its flux once")
                text = text.replace('flux = "roe"', f'flux = "{flux}"')
            (work / f"{name}-n{n:03d}.toml").write_text(text)
        checks["levels"][str(n)] = c
        ok = ok and all(v for v in c.values() if isinstance(v, bool))
    checks["pass"] = bool(ok)
    write_json(work / "prepare.json", checks)
    return 0 if ok else 1


def mesh_info(work, n):
    out = work / f"meshinfo-n{n:03d}.json"
    if not out.exists():
        text = (work / f"R0-n{n:03d}.toml").read_text()
        path = text.split('\nmesh = "')[1].split('"')[0]
        doc = subprocess.run(
            ["autan-mesh-info", "--json", path], check=True, capture_output=True, text=True
        ).stdout
        out.write_text(doc)
    return MeshInfo.load(out)


def archived_initial(run_dir):
    """The conserved initial state the run archived (states/<sha256>.json)."""
    case = autan.load(run_dir / "case.toml")
    sha = case.initial_state.sha256
    doc = read_json(run_dir / "states" / f"{sha}.json")
    return primitives_to_conserved(doc["values"])


def intended(work, name, n, points):
    if VARIANTS[name][0] == "centroid":
        return oracle_conserved(points, 0.0)
    return dict(np.load(work / f"avg-n{n:03d}-t0.npz"))


def run_fields(run_dir):
    r = autan.Results.open(run_dir)
    f = {k: np.asarray(r.field(k)) for k in PRIM}
    return r, f | conserved(f, GAMMA)


def resolve_exe(exe):
    found = shutil.which(exe) if os.sep not in exe else exe
    if not found or not Path(found).is_file():
        fail(f"{exe} is not an executable file")
    return str(Path(found).resolve())


def record_binary(work, exe, step):
    path = work / "binary.json"
    sha = sha_file(exe)
    known = read_json(path) if path.exists() else {}
    if known and known["sha256"] != sha:
        fail(f"{step}: AUTAN_RUN changed ({sha} != {known['sha256']})")
    write_json(path, {"sha256": sha, "steps": sorted({*known.get("steps", []), step})})


def roundtrip(work, exe):
    require_env()
    record_binary(work, exe, "roundtrip")
    report, ok = {}, True
    for name in ("R0", "RP"):
        for n in LEVELS:
            case = autan.load(work / f"{name}-n{n:03d}.toml")
            src = work / f"rt-{name}-n{n:03d}.json"
            write_json(src, build(case, mesh_info(work, n), devices=CPU).document())
            stop = work / "stop"
            stop.write_text("")
            runs = work / "rt-runs" / f"{name}-n{n:03d}"
            runs.mkdir(parents=True)
            p = subprocess.run(
                [exe, "--input", str(src), "--runs", str(runs), "--stop-file", str(stop)],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = p.stdout.strip().splitlines()
            res = {"exit": p.returncode}
            if lines:
                r, u = run_fields(Path(lines[-1]))
                want = intended(work, name, n, read_geometry(r).geometry.cell_centroid)
                sc = scales(want)
                res |= {"status": r.run.status, "iterations": r.run.iterations, "time": r.run.time}
                res["state"] = all(close(u[k], want[k], sc[k], k) for k in ALL)
            res["pass"] = bool(
                res["exit"] == 130
                and res.get("status") == "stopped"
                and res.get("iterations") == 0
                and res.get("time") == 0.0
                and res.get("state")
            )
            ok = ok and res["pass"]
            report[f"{name}-n{n}"] = res
    report["pass"] = bool(ok)
    write_json(work / "roundtrip.json", report)
    return 0 if ok else 1


def run(work, exe, name, levels=LEVELS):
    require_env()
    record_binary(work, exe, f"run {name}")
    backend = autan.SolverBackend(executable=Path(exe))
    ok = True
    for n in levels:
        case = autan.load(work / f"{name}-n{n:03d}.toml")
        results = case.run(
            mesh_info(work, n), backend=backend, runs=work / "runs" / f"{name}-n{n:03d}"
        )
        sys.stdout.write(f"{name} n={n} {results.run.status} {results.path}\n")
        ok = ok and results.run.status == "completed"
    return 0 if ok else 1


def eligible_fit(errs, hs, level_scales):
    """The gate's fit: levels whose error >= 100 eps times that level's scale; at least four, else none."""
    keep = [
        (h, e)
        for h, e, sc in zip(hs, errs, level_scales, strict=True)
        if math.isfinite(e) and e >= FLOOR_FACTOR * EPS * sc
    ]
    if len(keep) < MIN_LEVELS:
        return None
    p, _ = np.polyfit(np.log([h for h, _ in keep]), np.log([e for _, e in keep]), 1)
    return float(p)


def pair_orders(errs, hs):
    out = []
    for i in range(len(errs) - 1):
        e1, e2 = errs[i], errs[i + 1]
        ok = e1 > 0 and e2 > 0 and math.isfinite(e1) and math.isfinite(e2)
        out.append(math.log(e1 / e2) / math.log(hs[i] / hs[i + 1]) if ok else None)
    return out


def readings_of(data, name, ref, levels):
    hs = [data[(name, n)]["h"] for n in levels]
    entry = {}
    for k in VARS:
        errs = [data[(name, n)][ref][k] for n in levels]
        level_scales = [data[(name, n)]["scale"][ref][k] for n in levels]
        entry[k] = {
            "errors": errs,
            "pairwise": pair_orders(errs, hs),
            "fit": eligible_fit(errs, hs, level_scales),
        }
    return entry


def measure(work, reports, out):
    require_env()
    work = work.resolve()
    if out.exists():
        out.unlink()
    report = {"checks": {}}
    prep, rt, binary = (
        read_json(work / "prepare.json"),
        read_json(work / "roundtrip.json"),
        read_json(work / "binary.json"),
    )
    report["checks"]["prepare"] = bool(prep["pass"])
    report["checks"]["roundtrip"] = bool(rt["pass"])
    report["checks"]["binary"] = binary
    report["checks"]["binary_all_steps"] = sorted(binary["steps"]) == sorted(
        ["roundtrip", *(f"run {v}" for v in VARIANTS)]
    )
    archived = {r["n_cells"]: r for r in read_json(reports / "verify-v0-up.json")["runs"]} | {
        r["n_cells"]: r for r in read_json(reports / "verify-k-n512.json")["runs"]
    }
    data, runs_report, ok = {}, {}, True
    for name in VARIANTS:
        for n in LEVELS:
            found = sorted((work / "runs" / f"{name}-n{n:03d}").glob("run-*"))
            if len(found) != 1:
                fail(f"{name}-n{n}: {len(found)} runs")
            r, u = run_fields(found[0])
            g = read_geometry(r).geometry
            weight = g.cell_volume / g.cell_volume.sum()
            ref_c = oracle_conserved(g.cell_centroid, T_END)
            ref_a = dict(np.load(work / f"avg-n{n:03d}-t40.npz"))
            initial = archived_initial(found[0])
            want = intended(work, name, n, g.cell_centroid)
            sc0 = scales(want)
            init_ok = all(close(initial[k], want[k], sc0[k], k) for k in ALL)
            cons = {}
            for k in ALL:
                d = abs(
                    float(np.sum(g.cell_volume * u[k])) - float(np.sum(g.cell_volume * initial[k]))
                )
                tol = (1e-14 if k == "rhow" else 1e-12 * sc0[k]) * float(g.cell_volume.sum())
                cons[k] = bool(d <= tol)
            valid = bool(
                r.run.status == "completed"
                and abs(r.run.time - T_END) <= 1e-9
                and r.run.precision == "fp64"
            )
            e_c, e_a = verify_errors(u, ref_c, weight), verify_errors(u, ref_a, weight)
            # The quadrature budget: its L2 difference at t = 40 below 1e-3 of the error it enters.
            budget = all(prep["levels"][str(n)]["6v12 L2 t=40"][k] <= 1e-3 * e_a[k] for k in VARS)
            if VARIANTS[name][0] == "average":
                q0 = prep["levels"][str(n)]["6v12 L2 t=0"]
                budget = budget and all(q0[k] <= 1e-3 * min(e_a[k], e_c[k]) for k in VARS)
            mom = max(float(np.max(np.abs(ref_c[k]))) for k in ("rhou", "rhov", "rhow"))
            pres = l2(weight, u["rhow"] - ref_c["rhow"]) / mom / EPS

            def group_scale(ref):
                return {
                    k: max(
                        float(np.max(np.abs(ref[q])))
                        for q in (("rhou", "rhov", "rhow") if k in ("rhou", "rhov") else (k,))
                    )
                    or 1.0
                    for k in VARS
                }

            data[(name, n)] = {
                "h": 20.0 / n,
                "centroid": e_c,
                "average": e_a,
                "scale": {"centroid": group_scale(ref_c), "average": group_scale(ref_a)},
                "preserved": bool(pres <= 64.0),
            }
            runs_report[f"{name}-n{n}"] = {
                "valid": valid,
                "initial_state": init_ok,
                "conservation": cons,
                "quadrature_budget": budget,
                "errors_vs_centroid": e_c,
                "errors_vs_average": e_a,
                "rhow_preservation_in_eps": pres,
                "rhow_preserved": bool(pres <= 64.0),
            }
            ok = ok and valid and init_ok and all(cons.values()) and budget
            if name == "R0" and n == LEVELS[0]:
                zero = verify_errors(ref_a, ref_a, weight)
                via = verify_errors(ref_c, ref_a, weight)
                direct = {
                    k: float(np.sqrt(np.sum(weight * (ref_c[k] - ref_a[k]) ** 2))) for k in ALL
                }
                report["checks"]["verifier_zero"] = all(v <= 1e-14 for v in zero.values())
                report["checks"]["verifier_direct"] = all(
                    abs(via[k] - direct[k]) <= 1e-15 + 1e-12 * direct[k] for k in ALL
                )
    report["runs"] = runs_report
    ctrl = {}
    for n in LEVELS:
        a = archived.get(n * n)
        e = data[("R0", n)]["centroid"]
        ctrl[str(n)] = (
            max(abs(e[k] - a["errors"][k]["L2"]) / a["errors"][k]["L2"] for k in VARS)
            if a
            else float("inf")
        )
    report["checks"]["control_max_rel"] = ctrl
    report["checks"]["control"] = all(v <= 1e-3 for v in ctrl.values())
    # Stock verify on R0 against centroids: the custom path must agree to 1e-12 relative.
    r0 = [str(min((work / "runs" / f"R0-n{n:03d}").glob("run-*"))) for n in LEVELS]
    stock = {}
    for direction, ordered in (("up", r0), ("down", r0[::-1])):
        target = work / f"verify-R0-{direction}.json"
        if target.exists():
            target.unlink()
        p = subprocess.run(
            [
                "autan-vv",
                "verify",
                *ordered,
                "--against",
                "isentropic-vortex",
                "--out",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0 or not target.exists():
            fail(f"autan-vv verify ({direction}) failed (exit {p.returncode})")
        stock[direction] = {
            r["run_id"]: {k: r["errors"][k]["L2"] for k in VARS} for r in read_json(target)["runs"]
        }
    agree = True
    for n in LEVELS:
        run_id = autan.Results.open(Path(r0[LEVELS.index(n)])).run.run_id
        for k in VARS:
            ref = stock["up"][run_id][k]
            agree = agree and abs(data[("R0", n)]["centroid"][k] - ref) <= (
                1e-12 * ref if ref > 1e-14 else 1e-14
            )
    report["checks"]["custom_vs_stock"] = bool(agree)
    # Permutation: stock errors by run identity, and the custom readings computed in both orders.
    perm = stock["up"] == stock["down"]
    for name in VARIANTS:
        for ref in ("centroid", "average"):
            up = readings_of(data, name, ref, LEVELS)
            down = readings_of(data, name, ref, LEVELS[::-1])
            for k in VARS:
                fu, fd = up[k]["fit"], down[k]["fit"]
                same_fit = (fu is None and fd is None) or (
                    fu is not None and fd is not None and abs(fu - fd) <= 1e-12
                )
                perm = perm and same_fit
                perm = perm and all(
                    (a is None and b is None)
                    or (a is not None and b is not None and abs(a - b) <= 1e-12)
                    for a, b in zip(up[k]["pairwise"], down[k]["pairwise"][::-1], strict=True)
                )
    report["checks"]["permutation"] = bool(perm)
    gate = (
        "prepare",
        "roundtrip",
        "binary_all_steps",
        "verifier_zero",
        "verifier_direct",
        "control",
        "custom_vs_stock",
        "permutation",
    )
    read = bool(ok and all(report["checks"][c] for c in gate))
    report["checks"]["read"] = read
    if read:
        readings = {}
        for name in VARIANTS:
            for ref in ("centroid", "average"):
                entry = readings_of(data, name, ref, LEVELS)
                kept = [data[(name, n)]["preserved"] for n in LEVELS]
                for i, pair in enumerate(PAIRS):
                    values = [entry[k]["pairwise"][i] for k in VARS]
                    entry[f"band {pair}"] = (
                        all(v is not None and 1.9 <= v <= 2.1 for v in values)
                        and kept[i]
                        and kept[i + 1]
                    )
                fits = [entry[k]["fit"] for k in VARS]
                entry["band fit"] = (
                    "unavailable"
                    if any(f is None for f in fits)
                    else all(1.9 <= f <= 2.1 for f in fits) and all(kept)
                )
                readings[f"{name} vs {ref}"] = entry

        def diff(a, b):
            return None if a is None or b is None else a - b

        compare, two = {}, {}
        for k in VARS:
            for i, pair in enumerate(PAIRS):
                p = {
                    (v, rf): readings[f"{v} vs {rf}"][k]["pairwise"][i]
                    for v in VARIANTS
                    for rf in ("centroid", "average")
                }
                for v in ("RH", "RA"):
                    compare[f"{v}-R0 {k} {pair}"] = diff(p[(v, "centroid")], p[("R0", "centroid")])
                two[f"{k} {pair}"] = {
                    "init effect @centroid": diff(p[("RP", "centroid")], p[("R0", "centroid")]),
                    "init effect @average": diff(p[("RP", "average")], p[("R0", "average")]),
                    "ref effect @R0": diff(p[("R0", "average")], p[("R0", "centroid")]),
                    "ref effect @RP": diff(p[("RP", "average")], p[("RP", "centroid")]),
                    "interaction": diff(
                        diff(p[("RP", "average")], p[("R0", "average")]),
                        diff(p[("RP", "centroid")], p[("R0", "centroid")]),
                    ),
                }
        report["readings"], report["flux_vs_R0"], report["two_by_two"] = readings, compare, two
    write_json(out, report)
    sys.stdout.write(f"read {read}\n")
    return 0 if read else 1


def main():
    args = sys.argv[1:]
    try:
        if args[:1] == ["prepare"] and len(args) == 4:
            return prepare(Path(args[1]), Path(args[2]), Path(args[3]))
        if args[:1] == ["roundtrip"] and len(args) == 3:
            return roundtrip(Path(args[1]).resolve(), resolve_exe(args[2]))
        if args[:1] == ["run"] and len(args) in (4, 5) and args[3] in VARIANTS:
            levels = (int(args[4]),) if len(args) == 5 else LEVELS
            if any(n not in LEVELS for n in levels):
                return 2
            return run(Path(args[1]).resolve(), resolve_exe(args[2]), args[3], levels)
        if args[:1] == ["measure"] and len(args) == 4:
            return measure(Path(args[1]), Path(args[2]), Path(args[3]))
    except Exception as exc:  # noqa: BLE001 - any failure of a step means not measured
        sys.stdout.write(f"NOT MEASURED: {type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(__doc__ + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
