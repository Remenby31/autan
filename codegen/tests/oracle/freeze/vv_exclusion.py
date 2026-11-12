"""ADR 0003 stage 2's exclusion check (#43, section 3): every mesh any vv test generates, hashed.

As a pytest plugin on the vv suite (`pytest -p vv_exclusion vv/tests`, with this directory on
PYTHONPATH and VV_EXCLUSION_OUT set), it wraps `generate_level` (through which `generate_family`
and the CLI pass) and records, for every mesh a test generates: the kind, n, perturbation and
seed, the mesh SHA-256 (`FaceMesh.sha256()`) and the file SHA-256 (the file written, or, for the
Fluent kinds, the bytes the same writer produces). It wraps `cartesian` too (kind "cartesian",
the mesh SHA-256 and the bytes the Fluent writer produces). Two vv tests build a mesh by hand
instead (test_meshes' deliberately broken FaceMesh, test_su2's limiter slab of quads); they are
not recorded. It also records every skipped or failed test.

As a script (`python vv_exclusion.py OUT CANDIDATES MANIFEST`, or `--no-manifest-yet` in place of
MANIFEST before the stage 2 manifest exists), it compares that universe with
the candidates' file and mesh SHA-256s and exits 1 on any collision not declared, on any skipped
or failed vv test (any phase), on a pytest session that did not exit 0, or on an empty universe.
Given a stage 2 manifest, it also requires every instance's mesh file to be among the candidates,
so that no instance escapes the check."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

RECORDS: list[dict[str, Any]] = []
NOT_PASSED: list[str] = []
EXIT: dict[str, int] = {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install() -> None:
    from autan_vv.meshes import families as fam  # noqa: PLC0415

    orig = getattr(fam.generate_level, "__wrapped_original__", fam.generate_level)

    def generate_level(kind: str, n: int, *args: Any, **kw: Any) -> Any:
        mesh, version = orig(kind, n, *args, **kw)
        path = kw.get("path")
        if path is not None and Path(path).exists():
            file_sha = _sha(Path(path).read_bytes())
        elif kind in ("poly", "agglomerated"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "level.msh"
                fam.write_fluent(mesh, out, fam.CELL_ZONE)
                file_sha = _sha(out.read_bytes())
        else:
            file_sha = None  # a Gmsh kind not written: its mesh SHA-256 still counts
        RECORDS.append({
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""), "kind": kind, "n": n,
            "perturbation": kw.get("perturbation", 0.0), "seed": kw.get("seed", 0),
            "mesh_sha256": mesh.sha256(), "file_sha256": file_sha,
        })  # fmt: skip
        return mesh, version

    generate_level.__wrapped_original__ = orig  # type: ignore[attr-defined]
    fam.generate_level = generate_level

    orig_cart = getattr(fam.cartesian, "__wrapped_original__", fam.cartesian)

    def cartesian(cells: Any, *args: Any, **kw: Any) -> Any:
        mesh = orig_cart(cells, *args, **kw)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "box.msh"
            fam.write_fluent(mesh, out, fam.CELL_ZONE)
            file_sha = _sha(out.read_bytes())
        RECORDS.append({
            "test": os.environ.get("PYTEST_CURRENT_TEST", ""), "kind": "cartesian",
            "n": "x".join(str(c) for c in cells), "perturbation": 0.0, "seed": 0,
            "mesh_sha256": mesh.sha256(), "file_sha256": file_sha,
        })  # fmt: skip
        return mesh

    cartesian.__wrapped_original__ = orig_cart  # type: ignore[attr-defined]
    fam.cartesian = cartesian
    # Modules that imported the names before: rebind them too.
    for name, mod in list(sys.modules.items()):
        if name.startswith(("autan_vv", "test_")):
            if getattr(mod, "generate_level", None) is orig:
                mod.generate_level = generate_level
            if getattr(mod, "cartesian", None) is orig_cart:
                mod.cartesian = cartesian


def pytest_configure(config: Any) -> None:  # noqa: ARG001
    _install()


def pytest_collection_modifyitems(session: Any, config: Any, items: Any) -> None:  # noqa: ARG001
    _install()


def pytest_runtest_logreport(report: Any) -> None:
    if report.outcome != "passed":  # failed or skipped, in setup, call or teardown
        NOT_PASSED.append(f"{report.nodeid} ({report.when}): {report.outcome}")


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # noqa: ARG001
    out = os.environ["VV_EXCLUSION_OUT"]
    doc = {"records": RECORDS, "not_passed": NOT_PASSED, "exitstatus": int(exitstatus)}
    Path(out).write_text(json.dumps(doc, indent=1))


def _digest(x: Any) -> bool:
    return isinstance(x, str) and x != ""


def _hit_text(hit: dict[str, Any]) -> str:
    return f"{hit['test']} ({hit['kind']} n{hit['n']})"


def compare(
    universe: dict[str, Any], candidates: dict[str, Any], manifest: dict[str, Any] | None = None
) -> list[str]:
    """Every issue with the universe against the candidates (and the manifest, when given).

    Fail-closed (#43, 6809100033 and its follow-up approval 6810000919): every expected collision
    must be a candidate and must collide; given the manifest, it must also be excluded and never an
    instance. Every instance must name its candidate (n, perturbation, seed) and carry exactly that
    candidate's mesh file; an instance that collides, by its own file hash or its candidate's mesh
    hash, fails whether or not it is listed."""
    issues = [f"vv test not passed: {t}" for t in universe["not_passed"]]
    if universe.get("exitstatus") != 0:
        issues.append(f"the vv pytest session exited {universe.get('exitstatus')!r}, not 0")
    if not universe["records"]:
        issues.append("the vv suite generated no mesh: the universe is empty")
    files = {r["file_sha256"]: r for r in universe["records"] if r["file_sha256"]}
    meshes = {r["mesh_sha256"]: r for r in universe["records"]}
    cand: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for c in candidates["candidates"]:
        key = (c["n"], c["perturbation"], c["seed"])
        if not (_digest(c.get("file_sha256")) and _digest(c.get("mesh_sha256"))):
            issues.append(f"N{key[0]} p{key[1]} s{key[2]}: a candidate without both digests")
        if key in cand:
            issues.append(f"N{key[0]} p{key[1]} s{key[2]} is listed twice among the candidates")
        cand[key] = c
    known = {c["file_sha256"] for c in candidates["candidates"] if _digest(c.get("file_sha256"))}
    expected = {(c["n"], c["perturbation"], c["seed"]) for c in candidates["expected_collisions"]}
    for n, p, s in sorted(expected, key=str):
        if (n, p, s) not in cand:
            issues.append(f"N{n} p{p} s{s} is listed as an expected collision but is no candidate")
    inst: set[tuple[Any, Any, Any]] = set()
    if manifest is not None:
        if not isinstance(manifest, dict):
            issues.append("the manifest is not a JSON object")
            manifest = {}
        instances = manifest.get("instances")
        if not instances:
            issues.append("the manifest has no instance")
        for i in instances or []:
            key = (i.get("n"), i.get("perturbation"), i.get("seed"))
            fsha = i.get("files", {}).get("mesh", {}).get("sha256")
            c = cand.get(key)
            if None in key:
                issues.append(f"instance {i.get('id')}: no n, perturbation or seed")
            if not _digest(fsha) or fsha not in known:
                issues.append(
                    f"instance {i.get('id')}: its mesh file is no candidate of the exclusion check"
                )
            elif c is None or c["file_sha256"] != fsha:
                issues.append(
                    f"instance {i.get('id')}: its mesh file is not that of candidate N{key[0]} p{key[1]} s{key[2]}"
                )
            inst.add(key)
            hit = files.get(fsha) or (meshes.get(c["mesh_sha256"]) if c else None)
            if hit:
                issues.append(
                    f"instance N{key[0]} p{key[1]} s{key[2]} collides with {_hit_text(hit)}"
                )
        excluded = set(manifest.get("excluded", {}).get("excluded", []))
        for n, p, s in sorted(expected, key=str):
            if (n, p, s) in inst:
                issues.append(
                    f"N{n} p{p} s{s} is listed as an expected collision but is an instance"
                )
            if f"N{n} p{p} s{s}" not in excluded:
                issues.append(f"N{n} p{p} s{s} is listed as an expected collision but not excluded")
    for key, c in cand.items():
        hit = files.get(c["file_sha256"]) or meshes.get(c["mesh_sha256"])
        if hit and key not in expected and key not in inst:
            issues.append(f"N{key[0]} p{key[1]} s{key[2]} collides with {_hit_text(hit)}")
        if not hit and key in expected:
            issues.append(f"N{key[0]} p{key[1]} s{key[2]} was expected to collide and does not")
    return issues


if __name__ == "__main__":
    args = sys.argv[1:]
    no_manifest_yet = "--no-manifest-yet" in args
    args = [a for a in args if a != "--no-manifest-yet"]
    if len(args) == 2 and not no_manifest_yet:
        print("::error::the stage 2 manifest is required (or --no-manifest-yet before it exists)")
        sys.exit(2)
    universe = json.loads(Path(args[0]).read_text())
    candidates = json.loads(Path(args[1]).read_text())
    manifest = json.loads(Path(args[2]).read_text()) if len(args) > 2 else None
    if len(args) > 2 and not isinstance(manifest, dict):
        print("::error::the stage 2 manifest given is not a JSON object")
        sys.exit(2)
    found = compare(universe, candidates, manifest)
    print(f"{len(universe['records'])} meshes generated by the vv tests")
    for i in found:
        print(f"::error::{i}")
    sys.exit(1 if found else 0)
