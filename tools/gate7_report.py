"""D2 gate 7's sensitivity report, part (c): each face's flux against the positive-control scale.

    python3 tools/gate7_report.py FACES [FACES ...]

Each FACES is a file the conservation check writes on a run's start event with
AUTAN_CONSERVATION_FACES (docs/design/euler.md): a JSON line (schema
`autan.conservation-faces/1`, the event's five `equations`, E, D and `max_faces`, the counts of
interior `faces` and `boundary_faces`), the CSV header `face,boundary,x,y,z,<equations>`, then
every face of the mesh once, with its five signed fluxes F_{f,k}. It is read as tools/gate7.py
reads it (a file off that format cannot be read), and every flux, of boundary faces too, must be
finite. The population is every interior face (boundary 0), zero-flux faces included; tau is
tools/gate7.py's for d = `max_faces`. Per file, as the 2026-11-09 amendment of D2's gate 7
declaration on #259 states:
- per equation k, the median and the minimum over the faces of |F_{f,k}| / D_k, the median of an
  even count being the mean of its two middle values;
- the median over the faces of max_k |F_{f,k}| / D_k over the active equations, N/A if none is;
- equation k is N/A when D_k = 0 and E_k = 0, left out of max_k; D_k < 0, D_k = 0 with E_k != 0, a
  non-finite value anywhere, read or derived, makes the report invalid;
- an equation whose median is below 100 tau gets the declared note: its median single-incidence
  signal is below the positive-control scale, which neither means the gate cannot detect it (the
  gate rejects at tau) nor changes gate 7's rule.

It prints one strict JSON line per file. Exit status: 0 every report is valid, 1 one is invalid,
2 a file it cannot read.
"""

import argparse
import math
import pathlib
import statistics
import sys

from gate7 import UNREADABLE, dumps, faces_rows, finite, tau_of

NOTE = (
    "{name}: the median single-incidence signal is below the positive-control scale (100 tau). "
    "That does not mean the gate cannot detect it, since the gate rejects at tau, and it does not "
    "change gate 7's rule."
)


def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def report(path: pathlib.Path) -> dict:
    head, _, rows = faces_rows(path)
    fluxes = [f for _, boundary, f in rows if not boundary]
    names, e_sums, d_sums = head["equations"], head["E"], head["D"]
    tau = tau_of(head["max_faces"])
    invalid = []
    if any(not math.isfinite(v) for _, _, f in rows for v in f):
        invalid.append("a face flux is not finite")
    active = []
    for k, name in enumerate(names):
        if not (finite(e_sums[k]) and finite(d_sums[k])):
            invalid.append(f"{name}: E or D is not finite")
        elif d_sums[k] < 0:
            invalid.append(f"{name}: D = {d_sums[k]!r} < 0")
        elif d_sums[k] == 0 and e_sums[k] != 0:
            invalid.append(f"{name}: D = 0 with E = {e_sums[k]!r}")
        elif d_sums[k] != 0:
            active.append(k)
    out: dict = {"file": str(path), "valid": False, "invalid": invalid, "tau": tau}
    out |= {"scale": 100 * tau, "faces": len(fluxes)}
    if invalid:
        return out
    ratios = {k: [abs(f[k]) / d_sums[k] for f in fluxes] for k in active}
    if any(not math.isfinite(r) for values in ratios.values() for r in values):
        invalid.append("a ratio |F| / D is not finite")
        return out
    equations, notes = [], []
    for k, name in enumerate(names):
        if k not in active:
            equations.append({"name": name, "status": "N/A"})
            continue
        median = statistics.median(ratios[k])
        least = min(ratios[k])
        equations.append({"name": name, "status": "active", "median": median, "min": least})
        if median < 100 * tau:
            notes.append(NOTE.format(name=name))
    largest = [max(ratios[k][i] for k in active) for i in range(len(fluxes))] if active else []
    of_max = statistics.median(largest) if largest else None
    derived = [e[s] for e in equations if e["status"] == "active" for s in ("median", "min")]
    if not all(math.isfinite(v) for v in [*derived, *([of_max] if largest else [])]):
        invalid.append("a median or minimum is not finite")
        return out
    out |= {"valid": True, "equations": equations, "notes": notes}
    out["median_of_max"] = of_max if largest else "N/A"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("faces", type=pathlib.Path, nargs="+")
    a = parser.parse_args()
    valid = True
    for path in a.faces:
        try:
            r = report(path)
        except UNREADABLE as exc:
            sys.stderr.write(f"gate7_report: {path}: {exc!r}\n")
            return 2
        valid = valid and r["valid"]
        say(dumps(r))
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
