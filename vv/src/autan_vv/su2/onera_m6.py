"""The ONERA M6 wing in Euler flow (run 2308 of Schmitt and Charpin) for SU2: meshes, configuration,
results.

The meshes are the `onera-m6` family (`autan_vv.meshes.onera_m6`) converted to SU2's native 3D format:
tetrahedra, and triangle markers named as SU2's published `inv_ONERAM6.cfg` names them (`WING`,
`FARFIELD`, `SYMMETRY`), so that the configuration runs with only its mesh, references and outputs
changed. Results: CL and CD per level and their continuum, and Cp at the experiment's seven span
stations, cut and compared with the functions of Autan's M2 study (`autan_vv.cases.onera_m6`).
Convergence is the absolute criterion of #493 with each run as its own baseline: run to its floor,
converged once CL and CD have stayed settled over the window.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autan_vv.errors import VVError
from autan_vv.su2 import history, msh
from autan_vv.su2.config import options
from autan_vv.su2.mesh import _write

if TYPE_CHECKING:
    from pathlib import Path

    from autan_vv.su2.msh import FloatArray, IntArray

RECORD_SCHEMA = "autan.su2-onera-m6-mesh/1"
LEVEL_SCHEMA = "autan.su2-onera-m6-level/1"
REPORT_SCHEMA = "autan.su2-onera-m6/1"
# SU2 v8.6.0, TestCases/euler/oneram6/inv_ONERAM6.cfg.
PUBLISHED_SHA256 = "f0f81240d05730b51c36575991c864733ed2243e7a69008b07203ebe119d9f58"
MARKERS = {"wing": "WING", "farfield": "FARFIELD", "symmetry": "SYMMETRY"}
VTK_TRIANGLE, VTK_TETRA = 5, 10
OUTPUTS = ("cl", "cd")
SURFACE = "surface_flow"
# What `configure` changes in SU2's published configuration; nothing that sets the flow or scheme.
ADOPTED_KEYS = ("MESH_FILENAME", "MESH_FORMAT", "REF_AREA", "REF_LENGTH", "ITER", "CONV_FIELD",
                "CONV_RESIDUAL_MINVAL", "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME",
                "OUTPUT_FILES", "OUTPUT_WRT_FREQ", "SURFACE_FILENAME", "VOLUME_OUTPUT",
                "WRT_RESTART_COMPACT", "SCREEN_OUTPUT")  # fmt: skip


@dataclass(frozen=True)
class Record:
    """What `convert` wrote and from what (schema autan.su2-onera-m6-mesh/1)."""

    schema: str
    source: str
    source_sha256: str
    file: str
    file_sha256: str
    points: int
    tetrahedra: int
    markers: dict[str, int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def convert(source: Path, out: Path) -> Record:
    """Write the MSH 4.1 ONERA M6 mesh `source` to `out` in SU2 format, with `out`.json holding its
    Record: the points in the source's order, its one 3D group of tetrahedra, and its triangle
    zones `wing`, `farfield` and `symmetry` as the markers of MARKERS. V0901 for a file the reader
    refuses, V0902 for one that is not such a mesh."""
    m = msh.read(source)
    volumes = [n for n, d in m.dims.items() if d == 3]
    zones = {n for n, d in m.dims.items() if d == 2}
    if len(volumes) != 1 or m.types[volumes[0]] != msh.TET:
        msg = f"{source}: expected one 3D group of tetrahedra, got {volumes}"
        raise VVError(msg, code="V0902")
    if zones != set(MARKERS) or any(m.types[z] != msh.TRIANGLE for z in zones):
        msg = f"{source}: expected triangle zones {sorted(MARKERS)}, got {sorted(zones)}"
        raise VVError(msg, code="V0902")
    _write(
        out,
        3,
        m.points,
        (VTK_TETRA, m.groups[volumes[0]]),
        {MARKERS[z]: (VTK_TRIANGLE, m.groups[z]) for z in MARKERS},
    )
    record = Record(
        schema=RECORD_SCHEMA,
        source=source.name,
        source_sha256=_sha256(source.read_bytes()),
        file=out.name,
        file_sha256=_sha256(out.read_bytes()),
        points=len(m.points),
        tetrahedra=len(m.groups[volumes[0]]),
        markers={MARKERS[z]: len(m.groups[z]) for z in MARKERS},
    )
    out.with_suffix(".json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    return record


def configure(published: str, mesh_file: str, max_iterations: int = 50000) -> str:
    """SU2's published ONERA M6 configuration with only ADOPTED_KEYS changed: the mesh, Schmitt and
    Charpin's reference area and mean aerodynamic chord (those of Autan's case), the iteration cap,
    SU2's own residual stop moved past the floor (the runner decides), and the outputs the results
    are read from. V0903 if the published file does not mark the wing, far field and symmetry
    plane as MARKERS names them."""
    from autan_vv.meshes import onera_m6 as geometry  # noqa: PLC0415

    theirs = options(published)
    for key, marker in (("MARKER_EULER", "WING"), ("MARKER_FAR", "FARFIELD"),
                        ("MARKER_SYM", "SYMMETRY")):  # fmt: skip
        if marker not in theirs.get(key, ""):
            msg = f"published {key} = {theirs.get(key)!r} does not name {marker!r}"
            raise VVError(msg, code="V0903")
    ours = {
        "MESH_FILENAME": mesh_file,
        "MESH_FORMAT": "SU2",
        "REF_AREA": repr(geometry.REFERENCE_AREA),
        "REF_LENGTH": repr(geometry.MEAN_AERODYNAMIC_CHORD),
        "ITER": str(max_iterations),
        "CONV_FIELD": "( REL_RMS_DENSITY, REL_RMS_MOMENTUM-X, REL_RMS_MOMENTUM-Y, REL_RMS_MOMENTUM-Z,"
        " REL_RMS_ENERGY )",
        "CONV_RESIDUAL_MINVAL": "-16.0",
        "HISTORY_OUTPUT": "( ITER, RMS_RES, AERO_COEFF, WALL_TIME )",
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "history",
        "OUTPUT_FILES": "( RESTART_ASCII, SURFACE_CSV )",
        "OUTPUT_WRT_FREQ": str(max_iterations),
        "SURFACE_FILENAME": SURFACE,
        "VOLUME_OUTPUT": "( COORDINATES, SOLUTION, PRIMITIVE )",
        "WRT_RESTART_COMPACT": "NO",
        "SCREEN_OUTPUT": "( INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG )",
    }
    kept = [
        line for line in published.splitlines()
        if line.split("%", 1)[0].split("=", 1)[0].strip() not in ADOPTED_KEYS
    ]  # fmt: skip
    return "\n".join([*kept, *(f"{k}= {ours[k]}" for k in ADOPTED_KEYS)]) + "\n"


def prepare(
    family: Path,
    published: Path,
    out: Path,
    max_iterations: int = 50000,
    overrides: dict[str, str] | None = None,
    levels: tuple[int, ...] | None = None,
) -> list[Path]:
    """Convert the levels of the onera-m6 family `family` (its manifest.json; `levels`, all by
    default) to OUT/meshes/ and write OUT/nNNN/{case.cfg, level.json}; returns the case
    directories, coarsest first. `overrides` (a declared diagnostic) replaces the value of keys the
    configuration already has, and level.json records them. V0805 if `family` is not an onera-m6
    family, V0903 if `published` is not SU2's file (PUBLISHED_SHA256) or an override names a key
    the configuration does not have."""
    manifest = json.loads((family / "manifest.json").read_text())
    if manifest.get("schema") != "autan.mesh-family/1" or manifest.get("kind") != "onera-m6":
        msg = f"{family} is not an onera-m6 mesh family"
        raise VVError(msg, code="V0805")
    text = published.read_bytes()
    if _sha256(text) != PUBLISHED_SHA256:
        msg = f"{published} is not SU2's published inv_ONERAM6.cfg"
        raise VVError(msg, code="V0903")
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    dirs = []
    chosen = [lv for lv in manifest["levels"] if levels is None or lv["n"] in levels]
    for level in sorted(chosen, key=lambda lv: lv["n_cells"]):
        source = family / level["file"]
        if _sha256(source.read_bytes()) != level["file_sha256"]:
            msg = f"{source} is not the manifest's file"
            raise VVError(msg, code="V0805")
        record = convert(source, out / "meshes" / source.with_suffix(".su2").name)
        cfg = apply_overrides(
            configure(text.decode(), f"../meshes/{record.file}", max_iterations), overrides or {}
        )
        run = out / f"n{level['n']:03d}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "case.cfg").write_text(cfg)
        info = {
            "schema": LEVEL_SCHEMA, "n": level["n"], "cells": level["n_cells"], "h": level["h"],
            "published_sha256": PUBLISHED_SHA256, "case_sha256": _sha256(cfg.encode()),
            "mesh": asdict(record), "overrides": overrides or {},
        }  # fmt: skip
        (run / "level.json").write_text(json.dumps(info, indent=2) + "\n")
        dirs.append(run)
    return dirs


def apply_overrides(cfg: str, overrides: dict[str, str]) -> str:
    """`cfg` with the value of each key of `overrides` replaced, the line written `KEY= value`.
    V0903 if a key is not in `cfg` exactly once."""
    lines = cfg.splitlines()
    keys = [line.split("%", 1)[0].split("=", 1)[0].strip() for line in lines]
    for key, value in overrides.items():
        if keys.count(key) != 1:
            msg = f"override {key}: the configuration has it {keys.count(key)} times, not once"
            raise VVError(msg, code="V0903")
        lines[keys.index(key)] = f"{key}= {value}"
    return "\n".join(lines) + "\n"


def read_su2_3d(path: Path) -> tuple[FloatArray, dict[str, IntArray]]:
    """Points and marker elements of an SU2 native 3D mesh (SU2's format: NDIME, NELEM, NPOIN with
    an optional trailing index, NMARK). V0902 if it is not 3D."""
    lines = iter(path.read_text().splitlines())

    def value(key: str) -> str:
        for line in lines:
            if line.split("%", 1)[0].strip().startswith(key + "="):
                return line.split("=", 1)[1].strip()
        msg = f"{path}: no {key}"
        raise VVError(msg, code="V0902")

    if int(value("NDIME")) != 3:
        msg = f"{path}: not a 3D SU2 mesh"
        raise VVError(msg, code="V0902")
    for _ in range(int(value("NELEM"))):
        next(lines)
    npoin = int(value("NPOIN").split()[0])
    points = np.array([[float(t) for t in next(lines).split()[:3]] for _ in range(npoin)])
    markers = {}
    for _ in range(int(value("NMARK"))):
        tag = value("MARKER_TAG")
        n = int(value("MARKER_ELEMS"))
        markers[tag] = np.array([[int(t) for t in next(lines).split()[1:]] for _ in range(n)])
    return points, markers


def surface_cp(surface: Path, n_points: int) -> FloatArray:
    """Per mesh point, SU2's pressure coefficient from its surface file (`PointID`,
    `Pressure_Coefficient`), NaN off the plotted markers. V0904 if a column is missing."""
    with surface.open(newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    head = [h.strip().strip('"') for h in rows[0]]
    try:
        ip, icp = head.index("PointID"), head.index("Pressure_Coefficient")
    except ValueError:
        msg = f"{surface}: no PointID or Pressure_Coefficient column"
        raise VVError(msg, code="V0904") from None
    cp = np.full(n_points, np.nan)
    for r in rows[1:]:
        cp[int(r[ip])] = float(r[icp])
    return cp


def sections(
    points: FloatArray, wing: IntArray, cp: FloatArray, eta: list[float]
) -> dict[float, Any]:
    """Cp of the wing triangles `wing`, nodal values `cp`, cut at each span station eta, in x/c of
    the original local chord, upper (z >= 0) and lower surface: the cut of Autan's M2 study."""
    from autan_vv.cases import onera_m6 as case  # noqa: PLC0415
    from autan_vv.meshes import onera_m6 as geometry  # noqa: PLC0415

    faces = list(wing)
    out = {}
    for station in eta:
        y = station * geometry.SEMISPAN
        xzv = case.cut(points, faces, cp, y)
        xc = (xzv[:, 0] - geometry.leading_edge(y)) / geometry.chord(y)
        zc = xzv[:, 1] / geometry.chord(y)
        up, lo = zc >= 0.0, zc < 0.0
        upper = np.column_stack([xc[up], xzv[up, 2]])
        lower = np.column_stack([xc[lo], xzv[lo, 2]])
        out[station] = case.Section(
            station, upper[np.argsort(upper[:, 0])], lower[np.argsort(lower[:, 0])]
        )
    return out


def _level(run: Path) -> tuple[dict[str, Any], tuple[FloatArray, IntArray, FloatArray]]:
    level = json.loads((run / "level.json").read_text())
    prov: dict[str, Any] = history.provenance(run)["provenance"]  # type: ignore[assignment]
    mesh = run / options((run / "case.cfg").read_text())["MESH_FILENAME"]
    on_disk = (_sha256((run / "case.cfg").read_bytes()), _sha256(mesh.read_bytes()))
    prepared = (level["case_sha256"], level["mesh"]["file_sha256"])
    if not on_disk == prepared == (prov["case_sha256"], prov["mesh_sha256"]):
        msg = f"{run}: the run's configuration or mesh is not the prepared one"
        raise VVError(msg, code="V0904")
    if prov["exit_status"] != 0 or level["published_sha256"] != PUBLISHED_SHA256:
        msg = f"{run}: SU2 failed or the configuration is not the published one"
        raise VVError(msg, code="V0904")
    runs = history.read_su2(run / "history.csv")
    cfg = options((run / "case.cfg").read_text())
    # A 3D residual integrates fluxes over faces: its scale is a flux times an area.
    area = float(cfg["REF_LENGTH"]) ** 2
    scales = history.flux_scales(history.su2_freestream((run / "log.txt").read_text()), area)
    scales = {eq: scales[eq] for eq in runs.residuals}
    row: dict[str, Any] = {
        "n": level["n"], "cells": level["cells"], "h": level["h"], "su2": prov["su2"],
        "image_id": prov["image_id"], "case_sha256": level["case_sha256"],
        "mesh_sha256": level["mesh"]["file_sha256"],
        "source_sha256": level["mesh"]["source_sha256"],
        "precision": "fp64", "overrides": level.get("overrides", {}),
        "last_iteration": int(runs.iterations[-1]), "cl": float(runs.cl[-1]),
        "cd": float(runs.cd[-1]),
        "drop": {eq: float(np.log10(r[0] / r[-1])) for eq, r in runs.residuals.items()},
    }  # fmt: skip
    points, markers = read_su2_3d(mesh)
    cut = (points, markers["WING"], surface_cp(run / f"{SURFACE}.csv", len(points)))
    try:
        eps = history.epsilon(runs, scales, OUTPUTS)
    except VVError:
        row.update(converged=False, iterations=None, epsilon=None)
        return row, cut
    at = history.absolute_converged_at(runs, {eq: eps * s for eq, s in scales.items()}, OUTPUTS)
    row.update(
        converged=at is not None, epsilon=eps,
        iterations=None if at is None else int(runs.iterations[at] - runs.iterations[0]) + 1,
    )  # fmt: skip
    return row, cut


def _richardson(h: list[float], values: list[float]) -> dict[str, Any]:
    from autan_vv.order import richardson  # noqa: PLC0415

    try:
        r = richardson(h, values)
    except VVError as err:
        return {"error": err.code}
    return {"estimate": r.extrapolated, "order": r.order, "gci_fine": r.gci_fine}


def results(root: Path, cache: Path) -> dict[str, Any]:
    """The report (autan.su2-onera-m6/1) on the runs `prepare` wrote under `root`: per level,
    the absolute criterion of #493 with the run as its own baseline, which with no independent
    baseline establishes that CL and CD stayed settled within RTOL for WINDOW iterations, not
    that the residuals fell (their drop from the first iteration is reported beside it), CL and CD at the last iteration and, per span
    station of `case_2308.dat`, the comparison of Autan's M2 study (upper-surface shocks against
    the experiment's, rear-shock gap, Cp RMS away from shocks); CL and CD in the continuum from the
    three levels (h of the manifest), none unless all three converged. V0904 if a run is not the
    prepared one, SU2 failed or an output is missing; V0905 if the levels mix SU2 builds."""
    from autan_vv.cases import fetch, get_case  # noqa: PLC0415
    from autan_vv.cases import onera_m6 as case  # noqa: PLC0415

    c = case.conditions()
    data = get_case("onera-m6")
    paths = dict(zip(
        [f.name for f in data.files if f.role == "experiment"],
        fetch(data, cache, roles=("experiment",)), strict=True,
    ))  # fmt: skip
    measured = case.experiment(paths["case_2308.dat"])
    read = sorted((_level(run) for run in root.iterdir() if (run / "level.json").exists()),
                  key=lambda item: item[0]["cells"])  # fmt: skip
    levels = [lv for lv, _ in read]
    if len({(lv["su2"], lv["image_id"]) for lv in levels}) != 1:
        msg = f"{root}: the levels mix SU2 builds"
        raise VVError(msg, code="V0905")
    for lv, (points, wing, cp) in read:
        computed = sections(points, wing, cp, list(measured))
        lv["sections"] = case.compare(computed, measured, c["mach"])
    converged = len(levels) == 3 and all(lv["converged"] for lv in levels)
    continuum = {
        k: _richardson([lv["h"] for lv in levels], [lv[k] for lv in levels])
        if converged else {"error": "not converged"}
        for k in OUTPUTS
    }  # fmt: skip
    return {"schema": REPORT_SCHEMA, "levels": levels, "continuum": continuum}
