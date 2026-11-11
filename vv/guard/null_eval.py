"""#611 gate revision: the evaluator of the declaration (issue 611, comment 6785712335).

Usage: python null_eval.py MANIFEST OUT [--stage F|H|C] [--unseal]

MANIFEST lists runs: [{"set": "S_F"|"S_H"|"S_C"|"B4"|"p657", "label": str, "run": dir,
"commit": sha}]. Stage F computes every candidate on S_F, derives the bounds and checks
permissiveness and the positive controls; stage H adds the selection on S_H; stage C the
confirmation on S_C. #657's runs ("p657") are read only with --unseal, after adoption is frozen.
A run that cannot be measured (missing, truncated, non-finite, zero where a logarithm is taken)
counts as a violation wherever it would be tested.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from autan_vv import guard

STEPS = guard.STEPS
W1 = (10001, 20000)
W2 = (18001, 20000)
MEDIAN_M = (15001, 20000)
DROPS = list(guard.DROP_MIN)  # (equation, reference): peak, first or shared
DELTA = {"T": 0.05, "M": 0.3, "G_P": 0.3}
LIMIT = {"T": 0.5, "M": -9.0, "energy_drop": 9.0}
CANDIDATES = {"d": ["T", "M", "withdrawn"], "c": ["median_drop", "energy_ungated"]}


def _slice(series: list[float], window: tuple[int, int]) -> list[float]:
    return series[window[0] - 1 : window[1]]


def _log_ratio(num: float, den: float) -> float | None:
    if not (math.isfinite(num) and math.isfinite(den)) or num <= 0:
        return None
    return math.inf if den == 0 else math.log10(num / den)


def candidates(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Every candidate measure of one history (declaration notation)."""
    res = {k: [line["residuals"][k] for line in lines] for k in lines[0]["residuals"]}
    e = res["energy"]
    p, e1 = max(e), e[0]
    level = 1e-10 * p
    w1 = _slice(e, W1)
    momentum = max(res[k][0] for k in guard.MOMENTUM if k in res)
    out: dict[str, Any] = {
        "T": sum(1 for r in w1 if r > level) / len(w1),
        "M": _log_ratio(statistics.median(_slice(e, MEDIAN_M)), p),
        "G_P": _log_ratio(p, e1),
        "median_drop": {},
    }
    for k, ref in DROPS:
        r = res[k]
        reference = {"peak": max(r), "first": r[0], "shared": momentum}[ref]
        out["median_drop"][f"{k}/{ref}"] = _log_ratio(reference, statistics.median(_slice(r, W2)))
    m = guard.measure(lines)
    out["m3f"], out["m3s"] = m["m3f"], m["m3s"]
    out["b_pass"] = all(g["pass"] for g in guard.gates(m, has_544=False) if g["gate"] == "b")
    return out


def _round_out(value: float, upper: bool, sig_digits: int | None) -> float:
    if sig_digits is not None:
        if value == 0:
            return 0.0
        scale = 10 ** (sig_digits - 1 - math.floor(math.log10(abs(value))))
        f = math.ceil if upper else math.floor
        return f(value * scale) / scale
    return (math.ceil(value * 100) if upper else math.floor(value * 100)) / 100


def bounds(train: list[dict[str, Any]]) -> dict[str, Any]:
    """The declaration's bound rule on S_F."""
    b: dict[str, Any] = {}
    for key in ("T", "M", "G_P"):
        vals = [c[key] for c in train]
        hi = max(vals) + max(DELTA[key], max(vals) - min(vals))
        b[key] = _round_out(hi, upper=True, sig_digits=2 if key == "T" else None)
    b["median_drop"] = {}
    for name in train[0]["median_drop"]:
        vals = [c["median_drop"][name] for c in train]
        lo = min(vals) - max(0.3, max(vals) - min(vals))
        b["median_drop"][name] = _round_out(lo, upper=False, sig_digits=None)
    return b


def permissive(b: dict[str, Any]) -> dict[str, bool]:
    """True where a candidate's bound is too weak to be adopted."""
    energy = [v for k, v in b["median_drop"].items() if k.startswith("energy/")]
    return {
        "T": b["T"] >= LIMIT["T"],
        "M": b["M"] >= LIMIT["M"],
        "median_drop": any(v < LIMIT["energy_drop"] for v in energy),
    }


def passes(c: dict[str, Any] | None, b: dict[str, Any], gate: str, cand: str) -> bool:
    """One run against one candidate; an unmeasurable run fails."""
    if c is None:
        return False
    cap = c["G_P"] is not None and c["G_P"] <= b["G_P"]
    if gate == "d":
        if cand == "T":
            return c["T"] <= b["T"] and cap
        if cand == "M":
            return c["M"] is not None and c["M"] <= b["M"] and cap
        return True  # withdrawn: m3s reported only
    if cand == "median_drop":
        drops = all(
            c["median_drop"][k] is not None and c["median_drop"][k] >= v
            for k, v in b["median_drop"].items()
        )
        return drops and cap
    return all(  # energy_ungated: the non-energy drops keep the last-step gate
        g["pass"] for g in c["_gates_c"] if g["equation"] != "energy"
    )


def full_guard(c: dict[str, Any], b: dict[str, Any], cd: str, cc: str) -> dict[str, bool]:
    """Which parts of the full guard catch a history: gate (b), m3f, candidate (d), candidate (c)."""
    return {
        "b": not c["b_pass"],
        "m3f": c["m3f"] is None or c["m3f"] > guard.M3F_MAX,
        "d": not passes(c, b, "d", cd),
        "c": not passes(c, b, "c", cc),
    }


def _transform(lines: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """The declaration's synthetic regressions of one history."""
    res = {k: [line["residuals"][k] for line in lines] for k in lines[0]["residuals"]}
    e = res["energy"]
    p = max(e)
    if kind == "stretch":
        out = {}
        for k, r in res.items():
            logs = [math.log10(v) if v > 0 else None for v in r]
            new = []
            for n in range(1, STEPS + 1):
                x = max(1.0, n / 1.3)
                i, f = math.floor(x), x - math.floor(x)
                a, z = logs[i - 1], logs[min(i, STEPS - 1)]
                new.append(0.0 if a is None or z is None else 10 ** (a + f * (z - a)))
            out[k] = new
        res = out
    elif kind == "plateau":
        res["energy"] = [v if n < 12000 else 1e-9 * p for n, v in enumerate(e, start=1)]
    elif kind == "resurgence":
        level = 1e-10 * p
        res["energy"] = [
            10 * level if W1[0] <= n <= W1[1] and ((n - W1[0]) // 200) % 10 in (0, 3, 6) else v
            for n, v in enumerate(e, start=1)
        ]
    elif kind == "peak":
        i = e.index(p)
        res["energy"] = [v * 10 if j == i else v for j, v in enumerate(e)]
    return [{**line, "residuals": {k: res[k][j] for k in res}} for j, line in enumerate(lines)]


def _load(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str]:
    runs = sorted(Path(entry["run"]).glob("run-*")) if Path(entry["run"]).is_dir() else []
    if len(runs) != 1:
        return None, None, f"{len(runs)} run dirs"
    try:
        _, lines = guard.read_run(runs[0], entry["commit"])
        c = candidates(lines)
        c["_gates_c"] = [g for g in guard.gates(guard.measure(lines), False) if g["gate"] == "c"]
        c["_lines"] = lines
        return c, guard.canonical_sha256(lines), "ok"
    except (guard.GuardError, OSError, KeyError, ValueError, ZeroDivisionError) as err:
        return None, None, f"unmeasurable: {err}"


def main() -> None:
    manifest, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "F"
    unseal = "--unseal" in sys.argv
    entries = json.loads(manifest.read_text())
    report: dict[str, Any] = {"stage": stage, "runs": []}
    loaded: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}
    for entry in entries:
        if entry["set"] == "p657" and not unseal:
            continue
        if (entry["set"] == "S_H" and stage == "F") or (entry["set"] == "S_C" and stage != "C"):
            continue
        c, sha, status = _load(entry)
        loaded.setdefault(entry["set"], []).append((entry["label"], c))
        row = {"set": entry["set"], "label": entry["label"], "status": status, "canonical": sha}
        if c is not None:
            row.update({k: c[k] for k in ("T", "M", "G_P", "median_drop", "m3f", "m3s")})
        report["runs"].append(row)
    train = [c for _, c in loaded.get("S_F", [])]
    if not train or any(c is None for c in train):
        report["error"] = "S_F incomplete or unmeasurable: no bounds"
        out_path.write_text(json.dumps(report, indent=1, default=str))
        return
    b = bounds(train)
    report["bounds"], report["permissive"] = b, permissive(b)
    controls: dict[str, Any] = {}
    for gate, cands in CANDIDATES.items():
        for cand in cands:
            caught = {}
            for kind in ("stretch", "plateau", "resurgence", "peak"):
                hits = []
                for c in train:
                    t = candidates(_transform(c["_lines"], kind))
                    t["_gates_c"] = [
                        g
                        for g in guard.gates(guard.measure(_transform(c["_lines"], kind)), False)
                        if g["gate"] == "c"
                    ]
                    cd = cand if gate == "d" else "T"
                    cc = cand if gate == "c" else "median_drop"
                    hits.append(full_guard(t, b, cd, cc))
                caught[kind] = {
                    "all_caught": all(any(h.values()) for h in hits),
                    "by_part": list(hits),
                }
            controls[f"{gate}:{cand}"] = caught
    report["positive_controls"] = controls
    selection: dict[str, Any] = {}
    for gate, cands in CANDIDATES.items():
        chosen = None
        for cand in cands:
            fallback = cand in ("withdrawn", "energy_ungated")
            ok_perm = fallback or not report["permissive"].get(cand, False)
            ok_ctrl = all(v["all_caught"] for v in controls[f"{gate}:{cand}"].values())
            held = loaded.get("S_H", [])
            viol_h = [lab for lab, c in held if not passes(c, b, gate, cand)]
            status = {
                "permissive_ok": ok_perm,
                "controls_ok": ok_ctrl,
                "S_H_runs": len(held),
                "S_H_violations": viol_h,
            }
            if stage == "C":
                conf = loaded.get("S_C", [])
                status["S_C_runs"] = len(conf)
                status["S_C_violations"] = [lab for lab, c in conf if not passes(c, b, gate, cand)]
            selection[f"{gate}:{cand}"] = status
            if chosen is None and ok_perm and ok_ctrl and stage != "F" and held and not viol_h:
                chosen = cand
        if chosen is None and stage != "F":
            chosen = f"{cands[-1]} (fallback: no candidate passed)"
        selection[f"{gate}:selected"] = chosen
    report["selection"] = selection
    for row in report["runs"]:
        row.pop("_lines", None)
    out_path.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
