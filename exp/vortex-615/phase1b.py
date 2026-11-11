"""#615 vortex-order study, phase 1b: phase 1 with control 4's bound replaced by control 2's declared
recovery tolerance, which is also each fit's resolution scale (amendment on #615); otherwise phase 1
as declared at
https://github.com/Remenby31/autan/issues/615#issuecomment-6783705879, with the note
https://github.com/Remenby31/autan/issues/615#issuecomment-6783739791): the translation/residual
split of #615's saved solutions, with the frozen instrument checks of the manifest first.

    python phase1.py RUNS_DIR REPORTS_DIR OUT.json

RUNS_DIR holds vx-615-{c-n256,c-n512,c-n1024,k-n512}/out/run-*; REPORTS_DIR holds #615's
verify-c.json and verify-k-n512.json. Exit 0: every instrument check passed and the readings were
written; 1: a check failed or a run is not a valid study run (phase 1 is not read); 2: usage.
"""

import itertools
import json
import math
import sys
from pathlib import Path

import autan
import numpy as np
from autan.units import si
from autan_vv.harness import read_geometry
from autan_vv.oracles import conserved
from autan_vv.oracles.inviscid import GAMMA, isentropic_vortex
from scipy.optimize import least_squares

T = 40.0
P_INF = 1.0 / GAMMA
EPS = float(np.finfo(np.float64).eps)
PRESERVED = 64.0  # gates.PRESERVED
VARS = ("rho", "rhou", "rhov", "rhoE")
PRIM = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")
FAMILIES = {
    "c": {
        "runs": {"vx-615-c-n256": 256, "vx-615-c-n512": 512, "vx-615-c-n1024": 1024},
        "dt": 2.0 / 1024,
        "report": "verify-c.json",
    },
    "k": {"runs": {"vx-615-k-n512": 512}, "dt": 2.0 / 512, "report": "verify-k-n512.json"},
}


class Invalid(Exception):
    pass


FIT = {"method": "trf", "xtol": 1e-14, "ftol": 1e-14, "gtol": 1e-14, "max_nfev": 200}


def wrap(v):
    """Shortest periodic displacement on the period-20 square."""
    v = np.asarray(v, dtype=float)
    return (v + 10.0) % 20.0 - 10.0


def dist(a, b=(0.0, 0.0)):
    return float(np.hypot(*wrap(np.asarray(a) - np.asarray(b))))


def oracle_at(points, centre=(0.0, 0.0), strength=5.0):
    f = isentropic_vortex(centre=tuple(centre), strength=strength).fields(points, T)
    return f | conserved(f, GAMMA)


class Level:
    def __init__(self, run_dir, dt, n):
        found = sorted((run_dir / "out").glob("run-*"))
        if len(found) != 1:
            msg = f"{run_dir.name}: {len(found)} run directories, expected one"
            raise Invalid(msg)
        path = found[0]
        results = autan.Results.open(path)
        info = results.run
        self.run_id = info.run_id
        steps = int(results.case.time.steps)
        step = si(results.case.time.dt, "s")
        self.valid = {
            "status": info.status == "completed",
            "time": abs(info.time - T) <= 1e-9,
            "precision": info.precision == "fp64",
            "dt": abs(step - dt) <= 1e-15 and abs(steps * step - T) <= 1e-9,
        }
        if not all(self.valid.values()):
            msg = f"{run_dir.name}: not a valid study run {self.valid}"
            raise Invalid(msg)
        mesh = read_geometry(results)
        if mesh.n_cells != n * n:
            msg = f"{run_dir.name}: {mesh.n_cells} cells, expected {n * n}"
            raise Invalid(msg)
        g = mesh.geometry
        self.n_cells = mesh.n_cells
        self.points = g.cell_centroid
        self.volume = g.cell_volume
        self.weight = g.cell_volume / g.cell_volume.sum()
        self.h = 20.0 / math.sqrt(mesh.n_cells)
        run = {name: np.asarray(results.field(name)) for name in PRIM}
        self.U = run | conserved(run, GAMMA)

    def norm(self, diff):
        return float(np.sqrt(np.sum(self.weight * diff**2)))


def recovery_tolerance(magnitude):
    """Control 2's declared recovery tolerance for a displacement of `magnitude`."""
    return 1e-3 * magnitude + 1e-9


def fit_centre(level, pressure, start, log=None):
    """The translation (x0, y0) of the oracle whose pressure is closest to `pressure`, reduced to the
    shortest periodic displacement; resolved False when the fit failed or exhausted its evaluations.
    `log`, if given, receives the fit's termination record."""
    sv = np.sqrt(level.volume)

    def residual(q):
        p = isentropic_vortex(centre=(q[0] * level.h, q[1] * level.h)).fields(level.points, T)
        return sv * (pressure - p["Pressure"]) / P_INF

    r = least_squares(residual, np.asarray(start, dtype=float) / level.h, **FIT)
    centre = wrap(np.asarray(r.x) * level.h)
    finite = bool(
        np.all(np.isfinite(centre))
        and math.isfinite(float(r.cost))
        and math.isfinite(float(r.optimality))
    )
    resolved = bool(r.status > 0 and r.nfev < FIT["max_nfev"] and finite)
    if log is not None:
        log.append(
            {
                "status": int(r.status),
                "nfev": int(r.nfev),
                "cost": float(r.cost),
                "optimality": float(r.optimality),
                "centre": centre.tolist(),
                "resolved": resolved,
                "tolerance": recovery_tolerance(dist(centre)),
            }
        )
    return centre, resolved


def split(level, U, centre):
    exact = oracle_at(level.points)
    shifted = oracle_at(level.points, centre)
    out = {}
    for k in VARS:
        e, t, s = U[k] - exact[k], shifted[k] - exact[k], U[k] - shifted[k]
        out[k] = {
            "e": level.norm(e),
            "t": level.norm(t),
            "s": level.norm(s),
            "cross": float(2 * np.sum(level.weight * t * s)),
        }
    return out


def preservation(level):
    """The gate's preservation test of rhow: L2 error / momentum scale / eps <= 64."""
    exact = oracle_at(level.points)
    scale = max(float(np.max(np.abs(exact[k]))) for k in ("rhou", "rhov", "rhow")) or 1.0
    worst = level.norm(level.U["rhow"] - exact["rhow"]) / scale / EPS
    return {"rhow_L2_in_eps": worst, "pass": worst <= PRESERVED}


def order(e1, e2, h1, h2):
    if not (e1 > 0 and e2 > 0 and math.isfinite(e1) and math.isfinite(e2)):
        return None
    return math.log(e1 / e2) / math.log(h1 / h2)


def synthetic(level, d, h):
    mapped = level.points.copy()
    mapped[:, 0] = (level.points[:, 0] - d[0]) / (1 + 0.3 * h * h)
    mapped[:, 1] = (level.points[:, 1] - d[1]) / (1 + 0.3 * h * h)
    return oracle_at(mapped, strength=5.0 * (1 + 0.5 * h * h))


def label(hold):
    return "set" if len(hold) == 4 else "partial: " + (",".join(hold) if hold else "(none)")


def main():
    if len(sys.argv) != 4:
        sys.stdout.write(__doc__ + "\n")
        return 2
    runs, reports, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_path.unlink(missing_ok=True)
    try:
        return study(runs, reports, out_path)
    except Exception as exc:  # noqa: BLE001 - any reader or validity failure means not measured
        with out_path.open("w") as f:
            json.dump(
                {"instrument_ok": False, "not_measured": f"{type(exc).__name__}: {exc}"},
                f,
                indent=1,
            )
        sys.stdout.write(f"NOT MEASURED: {type(exc).__name__}: {exc}\n")
        return 1


def study(runs, reports, out_path):
    report = {"checks": {}, "validity": {}, "levels": {}, "pairs": {}}
    ok = True
    levels = {}
    for spec in FAMILIES.values():
        with (reports / spec["report"]).open() as f:
            archive = json.load(f)
        archived = {r["run_id"]: {k: r["errors"][k]["L2"] for k in VARS} for r in archive["runs"]}
        for rid, n in spec["runs"].items():
            lv = Level(runs / rid, spec["dt"], n)
            levels[rid] = lv
            report["validity"][rid] = lv.valid
            h = lv.h
            checks = {
                "fits": {"control2": [], "starts": [], "control4": [], "control5": [], "solver": []}
            }
            fits = checks["fits"]
            # 1. Refresh against this family's own archived full-precision report.
            e0 = split(lv, lv.U, (0.0, 0.0))
            a = archived.get(lv.run_id)
            rel = max(abs(e0[k]["e"] - a[k]) / a[k] for k in VARS) if a else float("inf")
            checks["refresh_max_rel"] = rel
            checks["refresh"] = rel <= 1e-6
            # 2. Known shifts on the oracle sampled at the centroids: the resolution proxy u_delta.
            offsets = [(0.0, 0.0)]
            for s in (0.1 * h, h):
                offsets += [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)]
            for c in (0.02, 0.05):
                offsets += [(c * h * h, 0.0), (c * h * h / math.sqrt(2), c * h * h / math.sqrt(2))]
            worst, passed = 0.0, True
            for d in offsets:
                p = isentropic_vortex(centre=d).fields(lv.points, T)["Pressure"]
                got, res = fit_centre(lv, p, (0.0, 0.0), fits["control2"])
                err = dist(got, d)
                worst = max(worst, err)
                passed = passed and res and err <= 1e-3 * dist(d) + 1e-9
            checks["known_shifts"] = passed
            u = worst
            checks["u_delta"] = u
            # 4. Synthetic distortion with a known shift; 3. five starts on that synthetic field.
            d4 = (0.05 * h * h, 0.0)
            syn = synthetic(lv, d4, h)
            got, res = fit_centre(lv, syn["Pressure"], (0.0, 0.0), fits["control4"])
            starts = [
                fit_centre(lv, syn["Pressure"], s, fits["starts"])
                for s in ((0, 0), (h, 0), (-h, 0), (0, h), (0, -h))
            ]
            checks["starts"] = all(r and dist(x, got) <= 1e-3 * dist(got) + 1e-9 for x, r in starts)
            checks["distortion"] = {
                "recovered": got.tolist(),
                "resolved": res,
                "bound": recovery_tolerance(dist(d4)),
                "within": res and dist(got, d4) <= recovery_tolerance(dist(d4)),
                "s_exact": {k: lv.norm(syn[k] - oracle_at(lv.points, d4)[k]) for k in VARS},
                "s_fit": {k: lv.norm(syn[k] - oracle_at(lv.points, got)[k]) for k in VARS},
            }
            # 5. The same distortion with no shift.
            syn0 = synthetic(lv, (0.0, 0.0), h)
            got0, res0 = fit_centre(lv, syn0["Pressure"], (0.0, 0.0), fits["control5"])
            checks["distortion_unshifted"] = {
                "recovered": got0.tolist(),
                "resolved": res0,
                "within": res0 and dist(got0) <= 1e-9,
            }
            checks["preservation"] = preservation(lv)
            report["checks"][rid] = checks
            ok = ok and checks["refresh"] and checks["known_shifts"] and checks["starts"]
            ok = ok and checks["distortion"]["within"] and checks["distortion_unshifted"]["within"]

            # The readings of this level.
            centre, res = fit_centre(lv, lv.U["Pressure"], (0.0, 0.0), fits["solver"])
            for group in fits.values():  # each fit's declared u_fit, now that u_delta is known
                for record in group:
                    record["u_fit"] = max(u, record["tolerance"])
            ul = max(u, recovery_tolerance(dist(centre)))  # this fit's resolution scale
            sp = split(lv, lv.U, centre)
            moved = {
                m: split(lv, lv.U, centre + np.asarray(m))
                for m in ((ul, 0), (-ul, 0), (0, ul), (0, -ul))
            }
            carries = {}
            for k in VARS:
                for part in ("t", "s"):
                    changes = [abs(moved[m][k][part] - sp[k][part]) for m in moved]
                    finite = all(math.isfinite(c) for c in changes) and math.isfinite(sp[k][part])
                    carries[f"{k}.{part}"] = bool(
                        res and finite and sp[k][part] > 10 * max(changes)
                    )
            report["levels"][rid] = {
                "h": h,
                "n_cells": lv.n_cells,
                "centre": centre.tolist(),
                "displacement": dist(centre),
                "resolution": ul,
                "displacement_resolved": bool(
                    res and math.isfinite(ul) and dist(centre) >= 10 * ul
                ),
                "norms": sp,
                "carries_order": carries,
            }

    # Control 4's order agreement on the c pairs.
    ids = list(FAMILIES["c"]["runs"])
    for a, b in itertools.pairwise(ids):
        ca, cb = report["checks"][a]["distortion"], report["checks"][b]["distortion"]
        ha, hb = report["levels"][a]["h"], report["levels"][b]["h"]
        diffs = []
        for k in VARS:
            pf = order(ca["s_fit"][k], cb["s_fit"][k], ha, hb)
            pe = order(ca["s_exact"][k], cb["s_exact"][k], ha, hb)
            diffs.append(abs(pf - pe) if pf is not None and pe is not None else float("inf"))
        agree = all(x <= 0.05 for x in diffs)
        report["checks"][f"{a}->{b}.distortion_order"] = {"max_diff": max(diffs), "pass": agree}
        ok = ok and agree
    report["instrument_ok"] = ok

    # Pairs of the c family (read only if the instrument passed).
    for a, b in itertools.pairwise(ids):
        la, lb = report["levels"][a], report["levels"][b]
        pair = {"orders": {}, "flags": {}, "four_level_fit": "unavailable (three levels)"}
        for k in VARS:
            pe = order(la["norms"][k]["e"], lb["norms"][k]["e"], la["h"], lb["h"])
            pair["orders"][k] = {"e": pe}
            for part in ("t", "s"):
                eligible = la["carries_order"][f"{k}.{part}"] and lb["carries_order"][f"{k}.{part}"]
                pair["orders"][k][part] = (
                    order(la["norms"][k][part], lb["norms"][k][part], la["h"], lb["h"])
                    if eligible
                    else None
                )
        pair["band"] = all(
            pair["orders"][k]["e"] is not None and 1.9 <= pair["orders"][k]["e"] <= 2.1
            for k in VARS
        )
        both_resolved = la["displacement_resolved"] and lb["displacement_resolved"]
        pair["orders"]["displacement"] = (
            order(la["displacement"], lb["displacement"], la["h"], lb["h"])
            if both_resolved
            else None
        )
        if not both_resolved:
            pair["flags"] = dict.fromkeys(
                ("large fitted translation", "small residual", "faster residual"), "unavailable"
            )
        else:
            pair["flags"]["large fitted translation"] = label(
                [
                    k
                    for k in VARS
                    if all(lv["norms"][k]["t"] >= 0.5 * lv["norms"][k]["e"] for lv in (la, lb))
                ]
            )
            pair["flags"]["small residual"] = label(
                [
                    k
                    for k in VARS
                    if all(lv["norms"][k]["s"] <= 0.3 * lv["norms"][k]["e"] for lv in (la, lb))
                ]
            )
            if any(pair["orders"][k]["s"] is None or pair["orders"][k]["e"] is None for k in VARS):
                pair["flags"]["faster residual"] = "unavailable"
            else:
                pair["flags"]["faster residual"] = label(
                    [k for k in VARS if pair["orders"][k]["s"] - pair["orders"][k]["e"] >= 0.3]
                )
        report["pairs"][f"{a}->{b}"] = pair
    # The n = 512 time comparison, descriptive.
    report["time_n512"] = {
        k: {
            "c(dt=2/1024)": report["levels"]["vx-615-c-n512"]["norms"][k],
            "k(dt=2/512)": report["levels"]["vx-615-k-n512"]["norms"][k],
        }
        for k in VARS
    }
    with out_path.open("w") as f:
        json.dump(report, f, indent=1)
    sys.stdout.write(f"instrument_ok {ok}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
