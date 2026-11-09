"""Nightly convergence guard on the TMR flat plate (#611): measures and verdict of one night.

A night runs the guard case twice from one binary (docs/validation/guard.md). `verdict` reads both
runs, checks they are one fresh 20 000-step run each of the night's commit with identical numbers,
and applies the gates declared on #611 against the values of `main` 179e316c (#597's B4 history,
sha256 b65468a7...). It returns `green`, `red` or `void` (void only with infrastructure evidence the
job names). `night_status` reports `MISSING` for a night without a verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

VERDICT_SCHEMA = "autan.vv-guard-night/1"
STEPS = 20000
ORDERS = 10.0
# case-schema.md#time: identically zero at iteration 1 below this times the largest residual.
ZERO = 1e-12
MOMENTUM = ("momentum-x", "momentum-y", "momentum-z")

# #611, gate (b): tonight's final residuals x 10^0.5 (comment 6731472376).
FINAL_BOUND = {
    "density": 4.579e-15,
    "momentum-x": 1.2719e-12,
    "momentum-y": 5.355e-13,
    "energy": 1.8778e-9,
    "nu-tilde": 3.625e-18,
}
# The equation the slab's span makes zero: it must stay at or below its identically-zero floor.
ZERO_EQUATION = "momentum-z"
# #611, gate (c): orders, tonight's value - 0.5. Keys: (equation, reference).
DROP_MIN = {
    ("energy", "peak"): 11.36,
    ("energy", "first"): 9.82,
    ("momentum-x", "first"): 12.53,
    ("momentum-y", "shared"): 12.91,
    ("nu-tilde", "first"): 13.27,
}
# #611, gate (d): floor(1.10 x tonight's) for the first and the sustained crossing of energy below
# 1e-10 x its peak.
M3F_MAX = 2174
M3S_MAX = 13295


class GuardError(Exception):
    """A run the guard cannot measure: the night is red."""


def _history(run: Path) -> list[dict[str, Any]]:
    lines = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines() if x.strip()]
    if not lines:
        msg = f"{run}: empty history"
        raise GuardError(msg)
    return lines


def canonical_sha256(lines: list[dict[str, Any]]) -> str:
    """sha256 of the history's numbers: iteration, residuals, provisional flag and monitors, without
    times or timings, so two runs of one binary compare bit for bit."""
    keep = [
        {k: line.get(k) for k in ("iteration", "residuals", "provisional", "monitors")}
        for line in lines
    ]
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def read_run(run: Path, commit: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """run.json and history of one fresh STEPS-step run of `commit`, all residuals finite; else
    GuardError."""
    info = json.loads((run / "run.json").read_text())
    if info["solver"]["commit"] != commit:
        msg = f"{run}: commit {info['solver']['commit']}, not the night's {commit}"
        raise GuardError(msg)
    if info.get("start") is not None:
        msg = f"{run}: not a fresh run"
        raise GuardError(msg)
    lines = _history(run)
    its = [line["iteration"] for line in lines]
    if its != list(range(1, STEPS + 1)) or info["iterations"] != STEPS:
        msg = f"{run}: history is not steps 1 to {STEPS} ({len(its)} lines)"
        raise GuardError(msg)
    if not info["solver"].get("binary_sha256"):
        msg = f"{run}: no binary_sha256 in run.json"
        raise GuardError(msg)
    for line in lines:
        values = [*line["residuals"].values(), *_numbers(line.get("monitors", {}))]
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            msg = f"{run}: non-finite or non-numeric value at step {line['iteration']}"
            raise GuardError(msg)
        if not all(v >= 0.0 for v in line["residuals"].values()):
            msg = f"{run}: negative residual at step {line['iteration']}"
            raise GuardError(msg)
    return info, lines


def _numbers(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _numbers(v)]
    if isinstance(value, list):
        return [x for v in value for x in _numbers(v)]
    return [value] if not isinstance(value, str | bool) and value is not None else []


def _rule_step(lines: list[dict[str, Any]], shared: bool) -> int | None:
    """First step at which case-schema's rule holds with `ORDERS` (#544's shared momentum
    reference if `shared`): every equation `ORDERS` below its reference, an equation identically
    zero at step 1 exempt while at or below the floor and, once above it, measured from the
    largest of its kind."""
    first = lines[0]["residuals"]
    floor = ZERO * max(first.values())
    momentum = max(first[k] for k in MOMENTUM if k in first)
    woken: set[str] = set()
    for line in lines:
        ok = True
        for k, r in line["residuals"].items():
            exempt = first[k] < floor
            if exempt and k not in woken:
                if r <= floor:
                    continue
                woken.add(k)
            if k in MOMENTUM and (shared or exempt):
                ref = momentum
            elif exempt:
                ref = max(first.values())
            else:
                ref = first[k]
            ok = ok and r <= ref * 10.0**-ORDERS
        if ok:
            return int(line["iteration"])
    return None


def measure(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """The guard's measures of one history (#611)."""
    first, last = lines[0]["residuals"], lines[-1]["residuals"]
    floor = ZERO * max(first.values())
    momentum = max(first[k] for k in MOMENTUM if k in first)
    drops: dict[str, dict[str, float | None]] = {}
    for k in first:
        r = [line["residuals"][k] for line in lines]
        peak = max(r)
        drops[k] = {
            "peak": math.log10(peak / r[-1]) if r[-1] > 0 else math.inf,
            "first": (math.log10(first[k] / r[-1]) if r[-1] > 0 else math.inf)
            if first[k] > 0
            else None,
            "shared": (math.log10(momentum / r[-1]) if r[-1] > 0 else math.inf)
            if k in MOMENTUM
            else None,
        }
    energy = [line["residuals"]["energy"] for line in lines]
    threshold = max(energy) * 10.0**-ORDERS
    its = [int(line["iteration"]) for line in lines]
    m3f = next((its[i] for i, r in enumerate(energy) if r <= threshold), None)
    above = [i for i, r in enumerate(energy) if r > threshold]
    m3s = (
        its[above[-1] + 1]
        if above and above[-1] + 1 < len(its)
        else (its[0] if not above else None)
    )
    return {
        "final": dict(last),
        "zero_floor": floor,
        "zero_equation_max": max(line["residuals"][ZERO_EQUATION] for line in lines),
        "drops": drops,
        "m3f": m3f,
        "m3s": m3s,
        "rule_own": _rule_step(lines, shared=False),
        "rule_shared": _rule_step(lines, shared=True),
    }


def gates(m: dict[str, Any], has_544: bool) -> list[dict[str, Any]]:
    """Each gate of #611 on the measures `m` of run 1, with its value, bound and pass."""
    out: list[dict[str, Any]] = []
    rule = m["rule_shared"] if has_544 else m["rule_own"]
    out.append({"gate": "a", "rule": "shared" if has_544 else "own", "value": rule,
                "pass": rule is not None if has_544 else True})  # fmt: skip
    for k, bound in FINAL_BOUND.items():
        v = m["final"].get(k)
        out.append({"gate": "b", "equation": k, "value": v, "bound": bound,
                    "pass": v is not None and v <= bound})  # fmt: skip
    z = m["zero_equation_max"]
    out.append({"gate": "b", "equation": ZERO_EQUATION, "value": z, "bound": m["zero_floor"],
                "pass": z is not None and z <= m["zero_floor"]})  # fmt: skip
    for (k, ref), low in DROP_MIN.items():
        v = m["drops"].get(k, {}).get(ref)
        out.append({"gate": "c", "equation": k, "reference": ref, "value": v, "min": low,
                    "pass": v is not None and v >= low})  # fmt: skip
    out.append({"gate": "d", "measure": "m3f", "value": m["m3f"], "max": M3F_MAX,
                "pass": m["m3f"] is not None and m["m3f"] <= M3F_MAX})  # fmt: skip
    out.append({"gate": "d", "measure": "m3s", "value": m["m3s"], "max": M3S_MAX,
                "pass": m["m3s"] is not None and m["m3s"] <= M3S_MAX})  # fmt: skip
    return out


def _first_difference(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int | None:
    for x, y in zip(a, b, strict=False):
        if x["residuals"] != y["residuals"]:
            return int(x["iteration"])
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def verdict(
    run1: Path,
    run2: Path,
    commit: str,
    *,
    has_544: bool,
    previous: dict[str, Any] | None = None,
    previous_history: Path | None = None,
    exit_codes: tuple[int, int] = (0, 0),
    infra: str | None = None,
) -> dict[str, Any]:
    """One night's verdict (schema autan.vv-guard-night/1). `infra` names evidenced infrastructure
    failure (a cgroup OOM kill the job measured as new): the night is then `void`. A run that exits
    non-zero (`exit_codes`) or cannot be measured, two runs that differ, or a failing gate is
    `red`. `previous` is the last green night's verdict (or the reference run's, before the first
    green night): the same binary must give the same canonical history (gate e); another binary's
    difference is reported, with the first differing step against `previous_history`."""
    night: dict[str, Any] = {"schema": VERDICT_SCHEMA, "commit": commit, "has_544": has_544,
                             "exit_codes": list(exit_codes)}  # fmt: skip
    if infra:
        return {**night, "outcome": "void", "infra": infra}
    try:
        info1, lines1 = read_run(run1, commit)
        info2, lines2 = read_run(run2, commit)
        sha1, sha2 = canonical_sha256(lines1), canonical_sha256(lines2)
        binary = info1["solver"]["binary_sha256"]
        night |= {"binary_sha256": binary, "history_sha256": sha1, "status": info1["status"]}
        m = measure(lines1)
        checks = gates(m, has_544)
    except (GuardError, OSError, KeyError, ValueError, TypeError) as err:
        return {**night, "outcome": "red", "reason": f"unmeasurable run: {err}"}
    reasons = [c for c in checks if not c["pass"]]
    if any(rc != 0 for rc in exit_codes):
        reasons.append({"gate": "exit", "exit_codes": list(exit_codes), "pass": False})
    if sha1 != sha2 or binary != info2["solver"]["binary_sha256"]:
        reasons.append({"gate": "repeat", "run1": sha1, "run2": sha2, "pass": False})
    across: dict[str, Any] = {"previous": None}
    if previous is not None:
        across = {"previous": previous.get("commit"),
                  "same_binary": previous.get("binary_sha256") == binary}  # fmt: skip
        if across["same_binary"]:
            across["same_history"] = previous.get("history_sha256") == sha1
            if not across["same_history"]:
                reasons.append({"gate": "e", "previous": previous.get("history_sha256"),
                                "tonight": sha1, "pass": False})  # fmt: skip
        if previous_history is not None and previous.get("history_sha256") != sha1:
            try:
                across["first_differing_step"] = _first_difference(
                    _history_lines(previous_history), lines1
                )
            except (OSError, ValueError) as err:
                across["first_differing_step"] = f"unreadable: {err}"
        pm = previous.get("measures") or {}
        across["deltas"] = {
            "m3f": _delta(m["m3f"], pm.get("m3f")),
            "m3s": _delta(m["m3s"], pm.get("m3s")),
            "drops": {k: {r: _delta(v, (pm.get("drops") or {}).get(k, {}).get(r))
                          for r, v in d.items()} for k, d in m["drops"].items()},
        }  # fmt: skip
    return {**night, "outcome": "red" if reasons else "green", "measures": m, "gates": checks,
            "failing": reasons, "across_nights": across}  # fmt: skip


def _history_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _delta(now: Any, before: Any) -> float | None:
    numbers = isinstance(now, int | float) and isinstance(before, int | float)
    if numbers and math.isfinite(now) and math.isfinite(before):
        return float(now - before)
    return None


def night_status(night: Path) -> str:
    """`MISSING` if the night's directory holds no verdict.json, else its outcome."""
    path = night / "verdict.json"
    if not path.exists():
        return "MISSING"
    return str(json.loads(path.read_text()).get("outcome", "MISSING"))
