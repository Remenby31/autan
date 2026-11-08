"""The held-out runner of ADR 0003 (#43): one instance of the frozen manifest per call.

An instance is one mesh (generator kind, N, perturbation, seed) with its declared fields (a, G),
policies and methods. The runner, in this order:
1. checks that the loaded spike library (AUTAN_SPIKE_LIB) is the frozen one, by SHA-256: on a
   mismatch the instance is invalid at once and nothing else runs;
2. check 1 (topology from the cells' own polygons);
3. builds the certified geometry ONCE (sequences.construct) and the exact geometry, then check 2
   on them;
4. for every field, policy and method: the kernel run, check 3 (D1) and check 4, both given the
   frozen library digest.

Records. Each job attempt writes its own file, created exclusively (an existing one is never
overwritten), checkpointed atomically through a unique temporary file after every step, with its provenance (the expected and loaded library digests, the
manifest's) from the first checkpoint on. Each check's outcome (issues, and whether it covered
exactly every cell and component) is persisted before its digest of every entry's centre and
certified radius; an exception adds an invalid entry naming its stage and discards nothing.

Acceptance (fail-closed). `passed` needs a valid spec and a valid, complete, clean record: exactly
the declared field x policy x method runs, each check covering exactly every cell and component,
per-run digests consistent with the record's. `collect` gives stage 1's state from EVERY attempt of
every instance and of the auxiliary suite, against the approved manifest bytes and library digest
(and never raises: malformed input is a failure):
- any attempt that reported an issue, or that completed without passing, is decisive;
- only an attempt interrupted without any of these may be superseded;
- an attempt interrupted before any check may be superseded only if its provenance is missing or
  matches; a recorded mismatch is never forgiven;
- every instance needs a passing attempt; every auxiliary attempt (`aux` runs the oracle suite on
  the frozen build itself: fixtures, boundary cases, controls, with exact node ids from its pytest
  plugin) must have finished, exited 0, and passed every frozen node id.
Even then it is "no failure observed", not a verdict (ADR 0003: no PASS until stage 2).

`reuse_geometry=False` rebuilds the geometry inside check 2, as the unit tests do; the two paths
must give identical records (the equivalence declared for the freeze).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import autan_spike as s
import meshes
import numpy as np

from oracle import kernels as kr
from oracle import sequences as q
from oracle import topology as t

if TYPE_CHECKING:
    from collections.abc import Callable
    from fractions import Fraction

SCHEMA = "autan.adr0003-instance/3"
AUX_SCHEMA = "autan.adr0003-auxiliary/1"
SPEC_KEYS = ("id", "kind", "n", "perturbation", "seed", "fields", "policies", "methods")
KINDS = ("hex", "tet", "hybrid", "poly")
POLICIES = ("fp64", "fp32+acc64", "fp32")
METHODS = ("lsq", "gg")
PREFLIGHT = ("manifest", "start", "library")  # stages before any check has run
SHA = re.compile(r"[0-9a-f]{64}")


def _frac(x: Fraction) -> str:
    """Exact and without Python's limit on decimal conversion of huge integers: in hexadecimal."""
    return f"{x.numerator:x}/{x.denominator:x}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _kernel(
    m: s.Mesh, pol: str, method: str, qv: list[float], qb: list[float], g: tuple[float, ...]
) -> np.ndarray:
    if method == "lsq":
        return s.gradients(m, pol, "lsq", np.array(qv), np.array(qb))[0]
    grad0 = np.tile(np.array(g), (m.n_cells, 1))
    return s.gradients(m, pol, "gg", np.array(qv), np.array(qb), 1, grad0)[0]


def combinations(spec: dict[str, Any]) -> list[tuple[float, tuple[float, ...], str, str]]:
    """The declared runs of an instance: every field x policy x method."""
    fields = [(float(f["a"]), tuple(float(v) for v in f["G"])) for f in spec["fields"]]
    return [
        (a, g, pol, meth)
        for (a, g), pol, meth in product(fields, spec["policies"], spec["methods"])
    ]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def spec_issues(spec: Any) -> list[str]:
    """Why a spec is not a valid instance (empty if it is). Total: never raises."""
    try:
        if not isinstance(spec, dict):
            return ["invalid: the instance is not a mapping"]
        issues = [f"invalid: the instance has no {key}" for key in SPEC_KEYS if key not in spec]
        if issues:
            return issues
        if not isinstance(spec["id"], str) or not spec["id"]:
            issues.append("invalid: the instance id is not a non-empty string")
        if spec["kind"] not in KINDS:
            issues.append(f"invalid: unsupported kind {spec['kind']!r}")
        if not _count(spec["n"]) or spec["n"] <= 0:
            issues.append(f"invalid: N must be a positive integer, got {spec['n']!r}")
        if not _finite(spec["perturbation"]) or spec["perturbation"] < 0:
            issues.append(
                f"invalid: perturbation must be finite and >= 0, got {spec['perturbation']!r}"
            )
        if not _count(spec["seed"]) or spec["seed"] < 0:
            issues.append(f"invalid: seed must be a non-negative integer, got {spec['seed']!r}")
        for key, allowed in (("policies", POLICIES), ("methods", METHODS)):
            values = spec[key]
            if not isinstance(values, list) or not values or any(v not in allowed for v in values):
                issues.append(f"invalid: {key} must be a non-empty list of {allowed}")
        fields = spec["fields"]
        if not isinstance(fields, list) or not fields:
            issues.append("invalid: the instance declares no field")
        else:
            issues.extend(
                f"invalid: malformed field {f!r}"
                for f in fields
                if not isinstance(f, dict)
                or not _finite(f.get("a"))
                or not isinstance(f.get("G"), list)
                or len(f["G"]) != 3
                or not all(_finite(v) for v in f["G"])
            )
        if not issues and len(set(combinations(spec))) != len(combinations(spec)):
            issues.append("invalid: duplicate field, policy or method")
        return issues  # noqa: TRY300  inside the try on purpose: any error is an invalid spec
    except Exception:  # noqa: BLE001  total: a validator that raises would fail open upstream
        return ["invalid: the instance could not be validated"]


def _ignore(_record: dict[str, Any]) -> None:
    return


def run_instance(
    spec: dict[str, Any],
    library_sha256: str,
    manifest_sha256: str,
    reuse_geometry: bool = True,
    checkpoint: Callable[[dict[str, Any]], None] = _ignore,
) -> dict[str, Any]:
    """Run the declared checks on one instance and return its record; never raises.
    `checkpoint(record)` is called after every step."""
    record: dict[str, Any] = {
        "schema": SCHEMA, "id": spec.get("id") if isinstance(spec, dict) else None, "spec": spec,
        "stage": "start", "library_sha256_expected": library_sha256,
        "library_sha256_loaded": kr.library_sha256(), "manifest_sha256": manifest_sha256,
        "invalid": [], "runs": [],
    }  # fmt: skip
    checkpoint(record)
    try:
        record["invalid"] += spec_issues(spec)
        if record["invalid"]:
            checkpoint(record)
            return record
        record["stage"] = "library"
        library = kr.check_library(library_sha256)
        if library:
            record["invalid"] += [f"invalid: frozen library: {x}" for x in library]
            checkpoint(record)
            return record
        record["stage"] = "check 1"
        kind, n, p, seed = spec["kind"], spec["n"], spec["perturbation"], spec["seed"]
        m = s.mesh(meshes.build(kind, n, p, seed=seed))
        read = t.read_mesh(m)
        pts, cells, ag = meshes.cells(kind, n, p, seed=seed)
        record["cells"] = m.n_cells
        record["check1"] = t.compare(t.derive(pts, cells, ag), read)
        checkpoint(record)
        record["stage"] = "geometry"
        geometry = q.construct(read)
        exact = q.geometry_exact(read)
        stored = q.stored_geometry(m)
        record["stage"] = "check 2"
        if reuse_geometry:
            certs = {k: [[pair.c for pair in row] for row in rows] for k, rows in geometry.items()}
            record["check2"] = q.check_geometry(read, stored, certs, exact)
        else:
            record["check2"] = q.check_geometry(read, stored)
        checkpoint(record)
        n_int = len(read.neighbour)
        every = {(c, u) for c in range(m.n_cells) for u in range(3)}
        for a, g, pol, method in combinations(spec):
            key = f"a={a} G={list(g)} {pol} {method}"
            record["stage"] = key
            qv, qb = kr.field_floats(geometry, n_int, a, g)  # type: ignore[arg-type]
            q_pairs = kr.linear_field(geometry["cell_centroid"], a, g)  # type: ignore[arg-type]
            qb_pairs = [kr.given(0.0, kr.B64)] * n_int + kr.linear_field(
                geometry["face_centroid"][n_int:],
                a,
                g,  # type: ignore[arg-type]
            )
            out = _kernel(m, pol, method, qv, qb, g)
            grad0 = [[kr.given(v, kr.B64) for v in g]] * m.n_cells if method == "gg" else None
            inp = kr.Inputs(geometry, q_pairs, qb_pairs, grad0)
            d1 = kr.check_d1(read, inp, exact, out.tolist(), pol, method, library=library_sha256)
            run: dict[str, Any] = {
                "a": a, "G": list(g), "policy": pol, "method": method,
                "d1_issues": d1.issues, "d1_entries": len(d1.bounds),
                "d1_covers_every_entry": bool(every) and set(d1.bounds) == every,
                "d1_max_centre_ratio": float(d1.max_centre_ratio), "d1_max_use": float(d1.max_use),
            }  # fmt: skip
            record["runs"].append(run)  # the outcome first, then its digest
            checkpoint(record)
            d1_digest = hashlib.sha256()
            for (c, u), (centre, radius) in sorted(d1.bounds.items()):
                line = f"{key} d1 {c} {u} {_frac(centre)} {_frac(radius)} {float(out[c, u])!r}\n"
                d1_digest.update(line.encode())
            run["d1_sha256"] = d1_digest.hexdigest()
            checkpoint(record)
            b4 = kr.check_bound(
                read, geometry, exact, a, g, out.tolist(), pol, method,  # type: ignore[arg-type]
                library=library_sha256,
            )  # fmt: skip
            run.update({
                "bound_issues": b4.issues, "bound_entries": len(b4.bounds),
                "bound_covers_every_entry": bool(every) and set(b4.bounds) == every,
                "bound_max_centre_ratio": float(b4.max_centre_ratio),
                "bound_max_use": float(b4.max_use),
            })  # fmt: skip
            checkpoint(record)
            b4_digest = hashlib.sha256()
            for (c, u), radius in sorted(b4.bounds.items()):
                b4_digest.update(f"{key} bound {c} {u} {_frac(radius)}\n".encode())
            run["bound_sha256"] = b4_digest.hexdigest()
            run["entries_sha256"] = _sha(run["d1_sha256"] + run["bound_sha256"])
            checkpoint(record)
        record["entries_sha256"] = _sha("".join(r["entries_sha256"] for r in record["runs"]))
        record["stage"] = "done"
    except Exception:  # noqa: BLE001  fail-closed: recorded as invalid, nothing discarded
        record["invalid"].append(f"invalid: the runner raised during {record['stage']}")
        record["traceback"] = traceback.format_exc()
    checkpoint(record)
    return record


def _run_ok(r: dict[str, Any], cells: int) -> bool:
    shas = (r["d1_sha256"], r["bound_sha256"], r["entries_sha256"])
    return (
        r["d1_issues"] == [] and r["bound_issues"] == []
        and r["d1_covers_every_entry"] is True and r["bound_covers_every_entry"] is True
        and r["d1_entries"] == 3 * cells and r["bound_entries"] == 3 * cells
        and all(isinstance(h, str) and SHA.fullmatch(h) for h in shas)
        and r["entries_sha256"] == _sha(r["d1_sha256"] + r["bound_sha256"])
    )  # fmt: skip


def passed(record: Any, spec: Any) -> bool:
    """Fail-closed: a valid spec and a valid, complete, clean, self-consistent record."""
    try:
        if spec_issues(spec) or not isinstance(record, dict):
            return False
        if (
            record.get("schema") != SCHEMA
            or record.get("id") != spec["id"]
            or record.get("spec") != spec
        ):
            return False
        if record.get("invalid") or record.get("stage") != "done":
            return False
        loaded = record.get("library_sha256_loaded")
        if (
            not isinstance(loaded, str)
            or not SHA.fullmatch(loaded)
            or loaded != record.get("library_sha256_expected")
        ):
            return False
        if record["check1"] != [] or record["check2"] != []:
            return False
        cells = record["cells"]
        if not _count(cells) or cells <= 0:
            return False
        expected = {(a, tuple(g), pol, meth) for a, g, pol, meth in combinations(spec)}
        runs = record["runs"]
        got = {(r["a"], tuple(r["G"]), r["policy"], r["method"]) for r in runs}
        if len(runs) != len(expected) or got != expected:
            return False
        if not all(_run_ok(r, cells) for r in runs):
            return False
        return record.get("entries_sha256") == _sha("".join(r["entries_sha256"] for r in runs))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


def reported_issues(record: Any) -> bool:
    """Whether an attempt reported anything: an invalidity, or an issue of any check, or a check
    that did not cover every entry. Such an attempt is decisive."""
    if not isinstance(record, dict):
        return True
    try:
        if record.get("invalid") or record.get("check1") or record.get("check2"):
            return True
        for r in record.get("runs", []):
            if r.get("d1_issues") or r.get("bound_issues"):
                return True
            if (
                r.get("d1_covers_every_entry") is False
                or r.get("bound_covers_every_entry") is False
            ):
                return True
        return False  # noqa: TRY300  inside the try on purpose: malformed means decisive
    except (AttributeError, TypeError):
        return True


def decisive(record: Any, spec: Any) -> bool:
    """An attempt that counts against the stage whatever comes after: it reported something, or it
    completed without passing. Only a clean interruption (neither) may be superseded."""
    if reported_issues(record):
        return True
    return isinstance(record, dict) and record.get("stage") == "done" and not passed(record, spec)


def _bound(rec: Any, manifest_sha256: str, library_sha256: str) -> bool:
    return isinstance(rec, dict) and (
        rec.get("manifest_sha256") == manifest_sha256
        and rec.get("library_sha256_expected") == library_sha256
        and rec.get("library_sha256_loaded") == library_sha256
    )


def clean_preflight(rec: Any, sid: str, manifest_sha256: str, library_sha256: str) -> bool:
    """An attempt interrupted before any check, which may be superseded: this instance's schema and
    id, a preflight stage, no invalid entry, no check output at all, and provenance that is missing
    or matches the approved digests. A recorded mismatch is never forgiven."""
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA or rec.get("id") != sid:
        return False
    if rec.get("stage") not in PREFLIGHT or rec.get("invalid"):
        return False
    if any(k in rec for k in ("check1", "check2", "cells", "entries_sha256")) or rec.get("runs"):
        return False
    wanted = {"manifest_sha256": manifest_sha256, "library_sha256_expected": library_sha256,
              "library_sha256_loaded": library_sha256}  # fmt: skip
    return all(rec.get(k) in (None, v) for k, v in wanted.items())


def unfinished_stages(spec: dict[str, Any]) -> set[str]:
    """The stages an attempt of this instance can be in after preflight and before "done"."""
    runs = {f"a={a} G={list(g)} {pol} {method}" for a, g, pol, method in combinations(spec)}
    return {"check 1", "geometry", "check 2", *runs}


def clean_interruption(rec: Any, spec: dict[str, Any]) -> bool:
    """An attempt interrupted after preflight, which may be superseded: well formed (this schema and
    id, one of this instance's known unfinished stages, valid field shapes) and without any
    reported issue."""
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA or rec.get("id") != spec.get("id"):
        return False
    stage = rec.get("stage")
    if spec_issues(spec) or not isinstance(stage, str) or stage not in unfinished_stages(spec):
        return False
    if not isinstance(rec.get("invalid"), list) or not isinstance(rec.get("runs"), list):
        return False
    return not reported_issues(rec)


def _collect_instances(
    instances: list[Any], manifest_sha256: str, library_sha256: str, attempts: dict[str, list[Any]]
) -> list[str]:
    failures = []
    for spec in instances:
        sid = spec.get("id") if isinstance(spec, dict) else None
        if not isinstance(sid, str) or not sid:
            failures.append(f"a manifest instance has no valid id: {sid!r}")
            continue
        tries = attempts.get(sid) or []
        if not isinstance(tries, list) or not tries:
            failures.append(f"{sid}: no record")
            continue
        for k, rec in enumerate(tries):
            if clean_preflight(rec, sid, manifest_sha256, library_sha256):
                continue
            if not _bound(rec, manifest_sha256, library_sha256):
                failures.append(
                    f"{sid} attempt {k}: not bound to the approved library and manifest"
                )
            elif passed(rec, spec):
                continue
            elif decisive(rec, spec) or not clean_interruption(rec, spec):
                failures.append(
                    f"{sid} attempt {k}: reported an issue, completed without passing, or is malformed (decisive)"
                )
            # otherwise a clean, well-formed interruption: superseded if a later attempt passes
        if not any(
            passed(rec, spec) and _bound(rec, manifest_sha256, library_sha256) for rec in tries
        ):
            failures.append(f"{sid}: no attempt passed")
    return failures


def _collect_auxiliary(
    required: list[str], manifest_sha256: str, library_sha256: str, auxiliary: list[Any]
) -> list[str]:
    failures = []
    if not auxiliary:
        failures.append("no auxiliary evidence")
    for k, ev in enumerate(auxiliary):
        ok = (
            isinstance(ev, dict) and ev.get("schema") == AUX_SCHEMA and ev.get("stage") == "done"
            and ev.get("manifest_sha256") == manifest_sha256
            and ev.get("library_sha256_expected") == library_sha256
            and ev.get("library_sha256_loaded") == library_sha256
            and ev.get("pytest_exit") == 0
            and isinstance(ev.get("child_library_sha256"), list)
            and len(ev["child_library_sha256"]) == 2  # measured by pytest at session start and end
            and all(d == library_sha256 for d in ev["child_library_sha256"])
            and isinstance(ev.get("outcomes"), dict)
            and all(ev["outcomes"].get(n) == "passed" for n in required)
        )  # fmt: skip
        if not ok:
            failures.append(
                f"auxiliary attempt {k}: unfinished, not bound, not exit 0, or a frozen node id did not pass (decisive)"
            )
    return failures


def collect(
    manifest_bytes: bytes,
    manifest_sha256: str,
    library_sha256: str,
    attempts: dict[str, list[Any]],
    auxiliary: list[Any],
) -> dict[str, Any]:
    """Stage 1's state, fail-closed and total (malformed input is a failure, never an exception),
    from every attempt of every instance and of the auxiliary suite.

    `manifest_bytes` are verified against `manifest_sha256`, the approved digest; the instances and
    the frozen auxiliary node ids come from those verified bytes. `library_sha256` is the approved
    library digest (freeze record). `attempts[id]` holds every record written for that instance (one
    file per job attempt). `auxiliary` holds every auxiliary attempt record (`aux` subcommand)."""
    failures: list[str] = []
    try:
        if (
            not isinstance(manifest_bytes, bytes)
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256
        ):
            failures.append("the manifest is not the approved one")
        else:
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                failures.append("the manifest is not a mapping")
            else:
                instances = manifest.get("instances")
                if not isinstance(instances, list) or not instances:
                    failures.append("the manifest declares no instance")
                else:
                    failures += _collect_instances(
                        instances,
                        manifest_sha256,
                        library_sha256,
                        attempts if isinstance(attempts, dict) else {},
                    )
                aux = manifest.get("auxiliary_tests")
                required = aux.get("node_ids") if isinstance(aux, dict) else None
                if (
                    not isinstance(required, list)
                    or not required
                    or not all(isinstance(n, str) for n in required)
                ):
                    failures.append("the manifest freezes no valid auxiliary test list")
                else:
                    failures += _collect_auxiliary(
                        required,
                        manifest_sha256,
                        library_sha256,
                        auxiliary if isinstance(auxiliary, list) else [],
                    )
    except Exception:  # noqa: BLE001  total: an error in collection is a failure of the stage
        failures.append("collection failed: " + traceback.format_exc(limit=2))
    return {
        "stage": 1,
        "failures": failures,
        "state": "FAIL (stays FAIL whatever stage 2 gives)"
        if failures
        else "no failure observed in stage 1; not a verdict until stage 2",
    }


def node_outcomes(lines: list[str]) -> tuple[dict[str, str], list[str | None]]:
    """The nodeid_report plugin's lines -> ({node id: outcome}, [library digests the pytest process
    measured]). Any unrecognized line raises (malformed evidence is never dropped). A node is "passed" only if its test body ran and passed (a "call" phase) and no phase
    of it, in any repeat, did not pass; setup and teardown alone never make a node pass."""
    phases: dict[str, list[tuple[Any, Any]]] = {}
    libraries: list[str | None] = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if (
            isinstance(r, dict)
            and isinstance(r.get("nodeid"), str)
            and r.get("when") in ("setup", "call", "teardown")
        ):
            phases.setdefault(r["nodeid"], []).append((r["when"], r.get("outcome")))
        elif (
            isinstance(r, dict)
            and r.get("when") in ("sessionstart", "sessionfinish")
            and "nodeid" not in r
        ):
            libraries.append(r.get("library_sha256"))
        else:
            msg = f"an unrecognized report line: {line[:200]!r}"
            raise ValueError(msg)  # the auxiliary attempt is then recorded as malformed
    out = {}
    for node, seen in phases.items():
        if any(outcome not in ("passed", "skipped") for _, outcome in seen):
            out[node] = "failed"
        elif any(outcome == "skipped" for _, outcome in seen):
            out[node] = "skipped"
        elif any(when == "call" and outcome == "passed" for when, outcome in seen):
            out[node] = "passed"
        else:
            out[node] = "not run"  # setup/teardown only: the body never ran
    return out, libraries


TEMP_SUFFIXES = (".tmp", ".part")


def _create(path: Path, data: dict[str, Any]) -> bool:
    """Create the attempt file exclusively (False if it exists): it belongs to this attempt only."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return True


def _replace(path: Path, data: dict[str, Any]) -> None:
    """Checkpoint an attempt file this process created: a unique temporary file, then an atomic
    replace (a reader sees the previous checkpoint or this one)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    Path(tmp).replace(path)


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one instance")
    aux = sub.add_parser("aux", help="run the auxiliary suite and record its outcome")
    for p in (run, aux):
        p.add_argument("--manifest", required=True)
        p.add_argument(
            "--library-sha256", required=True, help="the frozen library digest (freeze record)"
        )
        p.add_argument(
            "--out", required=True, help="this attempt's record file, which must not exist"
        )
    run.add_argument("--instance", required=True)
    run.add_argument("--rebuild-geometry", action="store_true", help="equivalence check only")
    aux.add_argument("pytest_args", nargs=argparse.REMAINDER, help="-- then the pytest arguments")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.suffix in TEMP_SUFFIXES or out.name.startswith("."):
        print(f"{out}: a record file must not look like a temporary file")
        return 2
    raw = Path(args.manifest).read_bytes() if Path(args.manifest).is_file() else b""
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    base = {"manifest_sha256": manifest_sha256, "library_sha256_expected": args.library_sha256,
            "library_sha256_loaded": kr.library_sha256()}  # fmt: skip
    if args.command == "aux":
        record: dict[str, Any] = {"schema": AUX_SCHEMA, "stage": "started", **base}
        if not _create(out, record):
            print(f"{out} exists: one file per attempt, never overwritten")
            return 2
        fd, name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".nodeids")
        os.close(fd)
        report = Path(name)
        # The plugin is loaded before any conftest: the tests directory (the oracle package's parent)
        # must be importable in the pytest process itself.
        tests_dir = str(Path(__file__).resolve().parents[1])
        path = os.pathsep.join(p for p in (tests_dir, os.environ.get("PYTHONPATH", "")) if p)
        env = {**os.environ, "AUTAN_NODEID_REPORT": str(report), "PYTHONPATH": path}
        pytest_args = [a for a in args.pytest_args if a != "--"]
        cmd = [sys.executable, "-m", "pytest", "-p", "oracle.freeze.nodeid_report", *pytest_args]
        done = subprocess.run(cmd, env=env, check=False)
        record.update({"command": cmd, "pytest_exit": done.returncode})
        try:
            outcomes, libraries = node_outcomes(report.read_text().splitlines())
        except (ValueError, OSError):  # malformed or missing report: recorded, never "done"
            record.update({"stage": "malformed report", "traceback": traceback.format_exc()})
        else:
            record.update(
                {"stage": "done", "outcomes": outcomes, "child_library_sha256": libraries}
            )
        _replace(out, record)
        report.unlink(missing_ok=True)
        return 0
    record = {"schema": SCHEMA, "id": args.instance, "invalid": [], "stage": "manifest", **base}
    if not _create(out, record):
        print(f"{out} exists: one file per attempt, never overwritten")
        return 2
    try:
        manifest = json.loads(raw)
        spec = next(i for i in manifest["instances"] if i["id"] == args.instance)
        record = run_instance(
            spec, args.library_sha256, manifest_sha256,
            reuse_geometry=not args.rebuild_geometry, checkpoint=lambda r: _replace(out, r),
        )  # fmt: skip
        record["passed"] = passed(record, spec)
    except Exception:  # noqa: BLE001  a manifest or selection error is an invalid record too
        record.setdefault("invalid", []).append(
            f"invalid: the runner raised during {record.get('stage')}"
        )
        record["traceback"] = traceback.format_exc()
        record["passed"] = False
    _replace(out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
