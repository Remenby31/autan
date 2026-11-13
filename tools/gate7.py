"""D2 gate 7 on one run: a `conservation` event of autan-run's log against its threshold.

    python3 tools/gate7.py RUN [--at start|last] [--evidence FILE]

RUN is a run directory whose log.jsonl holds the `conservation` events of a run made with
AUTAN_CONSERVATION=1 (docs/design/euler.md, Conservation check). `--at start` reads the one event on
the state the run started from (the positive controls). `--at last`, the default, reads the last
event after a step (the campaign): it is the written solution's state only if run.json says
`converged` and its `iterations` equal the event's `iteration`, so either mismatch fails. The
rules are the D2 gate 7 row of docs/validation/acceptance.md:
- tau = max(64 * 2^-52, 2 gamma_(d+2)), gamma_m = m u / (1 - m u), u = 2^-53, d the event's
  `max_faces`;
- per equation k, q_k = |E_k| / D_k <= tau; D_k = 0 passes only with E_k = 0; D_k < 0 fails;
- a non-finite value of E, D, `boundary` or `faces` (null in the log), or a non-finite q_k, fails;
- `max_terms` > `max_faces` fails;
- `flux_mismatch` > 0 makes the case VOID, neither PASS nor FAIL, whatever else holds.

It prints one strict JSON line {verdict, q, tau, d, at, reasons}, q_k null where it is not a
finite number. With `--evidence FILE` it first writes to FILE the event, the q's, the outcome of
each rule, the run's run.json and the SHA-256 of the face rows of the event's `faces_file`
(AUTAN_CONSERVATION_FACES), its first line, which holds E, left out. That file is read in full
first: its JSON line (schema `autan.conservation-faces/1`) must repeat the event's equations, E, D
and `max_faces`; the CSV header must be face,boundary,x,y,z and the equations; every row must
hold ten numbers, a boundary flag of 0 or 1 and a finite centroid, with face ids strictly
ascending; the interior rows (at least one) and boundary rows must match the declared counts,
which must add up to the faces of the run's mesh-info.json. tools/gate7_report.py reads it
the same way.

Exit status: 0 PASS, 1 FAIL, 3 VOID, 2 a usage error or a run it cannot read.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import pathlib
import sys
from typing import NoReturn

FACES_SCHEMA = "autan.conservation-faces/1"
UNREADABLE = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, csv.Error)
U = 2.0**-53
EXIT = {"PASS": 0, "FAIL": 1, "VOID": 3}


def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def no_constant(name: str) -> NoReturn:
    msg = f"{name} is not JSON"
    raise ValueError(msg)


def loads(text: str) -> object:
    return json.loads(text, parse_constant=no_constant)


def dumps(value: object, **kwargs: object) -> str:
    return json.dumps(value, allow_nan=False, **kwargs)


def tau_of(d: int) -> float:
    m = (d + 2) * U
    return max(64 * 2.0**-52, 2 * m / (1 - m))


def finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def count(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{name} is not a count: {value!r}"
        raise ValueError(msg)
    return value


def unreadable(why: str) -> NoReturn:
    raise ValueError(why)


def faces_rows(path: pathlib.Path) -> tuple[dict, str, list[tuple[int, bool, list[float]]]]:
    head_line, body = path.read_text().split("\n", 1)
    head = loads(head_line)
    if not isinstance(head, dict) or head.get("schema") != FACES_SCHEMA:
        unreadable(f"{path}: not {FACES_SCHEMA}")
    names = head["equations"]
    if not (isinstance(names, list) and len(names) == 5 and all(isinstance(n, str) for n in names)):
        unreadable(f"{path}: five equation names expected")
    for key in ("E", "D"):
        if not isinstance(head[key], list) or len(head[key]) != 5:
            unreadable(f"{path}: {key} must hold five values")
    count("max_faces", head["max_faces"])
    interior = count("faces", head["faces"])
    boundary = count("boundary_faces", head["boundary_faces"])
    lines = list(csv.reader(io.StringIO(body)))
    if not lines or lines[0] != ["face", "boundary", "x", "y", "z", *names]:
        unreadable(f"{path}: the CSV header is not face,boundary,x,y,z and the equations")
    rows: list[tuple[int, bool, list[float]]] = []
    for line in lines[1:]:
        if len(line) != 10 or line[1] not in ("0", "1"):
            unreadable(f"{path}: row {line!r}")
        face, values = int(line[0]), [float(v) for v in line[2:]]
        if (rows and face <= rows[-1][0]) or face < 0:
            unreadable(f"{path}: face ids not strictly ascending at {face}")
        if not all(math.isfinite(v) for v in values[:3]):
            unreadable(f"{path}: face {face} has a non-finite centroid")
        rows.append((face, line[1] == "1", values[3:]))
    found = (sum(not b for _, b, _ in rows), sum(b for _, b, _ in rows))
    if found != (interior, boundary) or interior == 0:
        declared = (interior, boundary)
        unreadable(f"{path}: {found} interior and boundary rows, {declared} declared")
    return head, body, rows


def select(run: pathlib.Path, at: str) -> dict:
    events = []
    for line in (run / "log.jsonl").read_text().splitlines():
        if line:
            e = loads(line)
            if e.get("source") == "solver" and e.get("message") == "conservation":
                events.append(e["data"])
    wanted = [e for e in events if e.get("at") == ("start" if at == "start" else "step")]
    if at == "start" and len(wanted) != 1:
        msg = f"{len(wanted)} conservation events at the start, not one"
        raise ValueError(msg)
    if not wanted:
        msg = "no conservation event after a step"
        raise ValueError(msg)
    return wanted[-1]


def judge(event: dict) -> dict:
    d = count("max_faces", event["max_faces"])
    terms = count("max_terms", event["max_terms"])
    mismatch = count("flux_mismatch", event["flux_mismatch"])
    sums = {key: event[key] for key in ("E", "D", "boundary", "faces")}
    if any(not isinstance(v, list) or len(v) != 5 for v in sums.values()):
        msg = "E, D, boundary and faces must each hold five values"
        raise ValueError(msg)
    tau = tau_of(d)
    reasons = [
        f"{key}[{k}] is not finite"
        for key, values in sums.items()
        for k, v in enumerate(values)
        if not finite(v)
    ]
    q: list[float | None] = []
    closes = []
    above = 0
    for k, (e, dk) in enumerate(zip(sums["E"], sums["D"], strict=True)):
        ratio = None
        if finite(dk) and dk < 0:
            reasons.append(f"D[{k}] = {dk!r} < 0")
        elif finite(e) and finite(dk):
            ratio = (0.0 if e == 0 else math.inf) if dk == 0 else abs(e) / dk
            if dk == 0 and e != 0:
                reasons.append(f"D[{k}] = 0 with E[{k}] = {e!r}")
            elif not math.isfinite(ratio):
                reasons.append(f"q[{k}] = |E| / D is not finite")
            elif ratio > tau:
                above += 1
                reasons.append(f"q[{k}] = {ratio!r} > tau")
        q.append(ratio if ratio is not None and math.isfinite(ratio) else None)
        closes.append(ratio is not None and ratio <= tau)
    if terms > d:
        reasons.append(f"max_terms {terms} > max_faces {d}")
    verdict = "PASS" if not reasons else "FAIL"
    if mismatch > 0:
        verdict = "VOID"
        first = event.get("first_mismatch")
        reasons.append(f"{mismatch} interface faces differ between their ranks, first {first}")
    checks = {
        "finite": all(finite(v) for values in sums.values() for v in values),
        "max_terms": terms <= d,
        "q_le_tau": closes,
        "flux_mismatch": mismatch,
    }
    only_q = 0 < above == len(reasons)
    out = {"verdict": verdict, "q": q, "tau": tau, "d": d, "checks": checks, "reasons": reasons}
    return out | {"fails_only_at_q": only_q}


def terminal(info: dict | None, event: dict) -> list[str]:
    if info is None:
        return ["no run.json"]
    reasons = []
    if info.get("status") != "converged":
        reasons.append(f"run.json status {info.get('status')!r}, not 'converged'")
    last, written = event.get("iteration"), info.get("iterations")
    if last is None or last != written:
        reasons.append(f"the event is at iteration {last!r}, run.json at {written!r}")
    return reasons


def evidence(run: pathlib.Path, at: str, event: dict, info: dict | None, outcome: dict) -> dict:
    faces = event.get("faces_file")
    faces_sha = None
    if faces is not None:
        head, body, rows = faces_rows(pathlib.Path(faces))
        for key in ("equations", "E", "D", "max_faces"):
            if head[key] != event[key]:
                unreadable(f"{faces}: its {key} is not the event's")
        mesh_faces = loads((run / "mesh-info.json").read_text())["counts"]["faces"]
        if len(rows) != mesh_faces:
            unreadable(f"{faces}: {len(rows)} faces, the mesh has {mesh_faces}")
        faces_sha = hashlib.sha256(body.encode()).hexdigest()
    return {
        "run": str(run),
        "at": at,
        "event": event,
        "run_json": info,
        "face_rows_sha256": faces_sha,
        "tau": outcome["tau"],
        "d": outcome["d"],
        "q": outcome["q"],
        "checks": outcome["checks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--at", choices=("start", "last"), default="last")
    parser.add_argument("--evidence", type=pathlib.Path)
    a = parser.parse_args()
    try:
        event = select(a.run, a.at)
        info_file = a.run / "run.json"
        info = loads(info_file.read_text()) if info_file.is_file() else None
        outcome = judge(event)
        if a.at == "last":
            late = terminal(info, event)
            outcome["checks"]["terminal"] = not late
            outcome["reasons"] += late
            if late and outcome["verdict"] == "PASS":
                outcome["verdict"] = "FAIL"
        if a.evidence is not None:
            a.evidence.write_text(dumps(evidence(a.run, a.at, event, info, outcome), indent=1))
    except UNREADABLE as exc:
        sys.stderr.write(f"gate7: {a.run}: {exc!r}\n")
        return 2
    line = {k: outcome[k] for k in ("verdict", "q", "tau", "d", "reasons")}
    say(dumps({**line, "at": a.at}))
    return EXIT[outcome["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
