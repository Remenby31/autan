"""D2 gate 7's positive control, part (b): a mutant's evidence against the intact build's.

    python3 tools/gate7_control.py INTACT MUTANT [--components K ...] [--interface] [--face ID]
        --case-sha256 SHA --solution-sha256 SHA|none

INTACT and MUTANT are evidence files tools/gate7.py wrote (`--evidence`) on one configuration's
start event, with a face selected by AUTAN_CONSERVATION_FACE and every face's flux written by
AUTAN_CONSERVATION_FACES (docs/design/euler.md), the intact build's and the mutant's. With F the
selected face's flux in its global orientation and s = +1, the dropped incidence being the
global-left one, it requires (the 2026-11-09 amendment of D2's gate 7 declaration on #259):
- gate 7's rules (tools/gate7.py's judge) recomputed on both events, each record's tau, d, q
  and rule outcomes equal to them, and every check below on the recomputed values: the intact
  event PASSes; the mutant one FAILs at q_k <= tau only, every other rule holding;
- both runs ended `converged` or `completed`, each on its own: neither diverged, failed nor
  stopped;
- one state and configuration: the same run.json identity, case, mesh, precision, devices, cell
  order and start, the solver binary aside, and the frozen pins: the case's SHA-256
  (`--case-sha256`) and the source solution's (`--solution-sha256`, `none` for a run without a
  `start`, such as the box's);
- exactly one selected face in each, with the same global id, global-left cell and rank;
- unchanged bit for bit: the face's flux, D, the boundary sums, tau and every face's flux,
  interior and boundary, by the SHA-256 of the face rows, which both must record;
- per component k of `--components` (default all five): |F_k| / D_k >= 101 tau, then
  q_k of the mutant >= 100 tau, and |E_k(mutant) - E_k(intact) + s F_k| <= 2 tau D_k;
- with `--interface`, the face on a partition interface (the 4-rank configurations); with
  `--face`, that global id.

It prints one strict JSON line {pass, failed, checks}. Exit status: 0 every check passes, 1 one
fails, 2 an evidence file it cannot read.
"""

import argparse
import math
import pathlib
import re
import sys

from gate7 import UNREADABLE, dumps, judge, loads

S = 1.0
ENDED = ("converged", "completed")
SAME = ("identity", "case", "mesh", "precision", "devices", "cell_order", "start")


def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def same(a: object, b: object) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a).hex() == float(b).hex()
    return a == b


def at_most(value: float, bound: float) -> bool:
    return math.isfinite(value) and value <= bound


def at_least(value: float, bound: float) -> bool:
    return math.isfinite(value) and value >= bound


def compare(intact: dict, mutant: dict, components: list[int], a: argparse.Namespace) -> dict:
    checks: dict[str, bool] = {}
    i, m = intact["event"], mutant["event"]
    checks["both read at the start"] = intact["at"] == "start" and mutant["at"] == "start"
    oi, o = judge(i), judge(m)
    checks["records match their events"] = all(
        same(r["tau"], j["tau"])
        and r["d"] == j["d"]
        and same(r["q"], j["q"])
        and r["checks"] == j["checks"]
        for r, j in ((intact, oi), (mutant, o))
    )
    checks["intact passes gate 7"] = oi["verdict"] == "PASS"
    checks["mutant fails gate 7 at q <= tau only"] = o["verdict"] == "FAIL" and o["fails_only_at_q"]
    info = [e.get("run_json") or {} for e in (intact, mutant)]
    for name, run in zip(("intact", "mutant"), info, strict=True):
        checks[f"{name} run ended converged or completed"] = run.get("status") in ENDED
    for key in SAME:
        checks[f"same {key}"] = key in info[0] and key in info[1] and info[0][key] == info[1][key]
    checks["same solver commit"] = (info[0].get("solver") or {}).get("commit") == (
        info[1].get("solver") or {}
    ).get("commit")
    pin = None if a.solution_sha256 == "none" else a.solution_sha256
    checks["the frozen source solution"] = all(
        (run.get("start") or {}).get("solution_sha256") == pin
        and (pin is not None or run.get("start") is None)
        for run in info
    )
    checks["the frozen case"] = all(
        (run.get("case") or {}).get("sha256") == a.case_sha256 for run in info
    )
    shas = [e.get("face_rows_sha256") for e in (intact, mutant)]
    checks["every face flux unchanged"] = (
        all(isinstance(s, str) and re.fullmatch("[0-9a-f]{64}", s) for s in shas)
        and shas[0] == shas[1]
    )
    fi, fm = i.get("selected_face") or {}, m.get("selected_face") or {}
    checks["one selected face"] = fi.get("matches") == 1 and fm.get("matches") == 1
    if not checks["one selected face"]:
        return checks
    for key in ("id", "left_cell", "rank"):
        checks[f"same {key}"] = fi[key] == fm[key]
    if a.face is not None:
        checks["the named face"] = fm["id"] == a.face
    if a.interface:
        checks["on a partition interface"] = fm["interface"] is True
    checks["face flux unchanged"] = same(fi["flux"], fm["flux"])
    checks["D unchanged"] = same(i["D"], m["D"])
    checks["boundary unchanged"] = same(i["boundary"], m["boundary"])
    checks["tau unchanged"] = same(oi["tau"], o["tau"])
    tau = o["tau"]
    for k in components:
        f, d = fm["flux"][k], m["D"][k]
        e_mut, e_intact, q = m["E"][k], i["E"][k], o["q"][k]
        if any(v is None for v in (f, d, e_mut, e_intact, q)) or not d > 0:
            checks[f"[{k}] finite, D > 0"] = False
            continue
        checks[f"[{k}] |F| / D >= 101 tau"] = at_least(abs(f) / d, 101 * tau)
        checks[f"[{k}] q >= 100 tau"] = at_least(q, 100 * tau)
        defect = abs(e_mut - e_intact + S * f)
        checks[f"[{k}] signed defect <= 2 tau D"] = at_most(defect, 2 * tau * d)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("intact", type=pathlib.Path)
    parser.add_argument("mutant", type=pathlib.Path)
    parser.add_argument("--components", type=int, nargs="+", choices=range(5), default=range(5))
    parser.add_argument("--interface", action="store_true")
    parser.add_argument("--face", type=int)
    parser.add_argument("--case-sha256", required=True)
    parser.add_argument("--solution-sha256", required=True)
    a = parser.parse_args()
    try:
        intact = loads(a.intact.read_text())
        mutant = loads(a.mutant.read_text())
        checks = compare(intact, mutant, list(a.components), a)
    except UNREADABLE as exc:
        sys.stderr.write(f"gate7_control: {exc!r}\n")
        return 2
    failed = [name for name, ok in checks.items() if not ok]
    say(dumps({"pass": not failed, "failed": failed, "checks": checks}))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
