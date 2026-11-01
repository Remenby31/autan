"""SU2 on the NACA 0012 inviscid cases of Vassberg and Jameson (2010): preparation and study.

`prepare` converts each level of a `naca0012` mesh family and writes, per condition and level, the
SU2 configuration of the very Autan case `autan_vv.cases.naca0012.case` builds. `study` reads the
runs of one condition, applies the protocol's convergence criterion to each history and checks the
continuum estimates against the paper with the gates of `autan_vv.cases.naca0012`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from autan_vv.cases import naca0012 as cases
from autan_vv.errors import VVError
from autan_vv.meshes.families import MANIFEST_SCHEMA
from autan_vv.su2 import config, history, mesh

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

STUDY_SCHEMA = "autan.su2-study/1"
LEVEL_SCHEMA = "autan.su2-level/1"


def prepare(
    family: Path,
    out: Path,
    *,
    dimension: mesh.Dimension = 2,
    conditions: Sequence[str] = tuple(cases.CONDITIONS),
    max_iterations: int = 20000,
) -> list[Path]:
    """Write OUT/meshes/<level>.su2 and OUT/<condition>/n<NC>/{case.cfg, level.json}; returns the
    case directories. V0805 if `family` is not a naca0012 family."""
    manifest = json.loads((family / "manifest.json").read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("kind") != "naca0012":
        msg = f"{family} is not a naca0012 mesh family ({MANIFEST_SCHEMA})"
        raise VVError(msg, code="V0805")
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    dirs = []
    for level in manifest["levels"]:
        source = family / level["file"]
        target = out / "meshes" / source.with_suffix(f".{dimension}d.su2").name
        record = mesh.convert(source, target, dimension)
        slab = mesh.read_slab(source)
        venkat = config.venkat_coefficient(slab)
        for condition in conditions:
            case = cases.case(condition, source.name)
            text = config.config(
                case,
                record,
                venkat,
                span=slab.y1 - slab.y0,
                max_iterations=max_iterations,
                residual_orders=cases.RESIDUAL_ORDERS,
            )
            text = text.replace(
                f"MESH_FILENAME= {record.file}", f"MESH_FILENAME= ../../meshes/{record.file}"
            )
            run = out / condition / f"n{level['n']:04d}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "case.cfg").write_text(text)
            info = {
                "schema": LEVEL_SCHEMA,
                "condition": condition,
                "n": level["n"],
                "venkat_coefficient": venkat,
                "case_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "autan_case_sha256": hashlib.sha256(
                    json.dumps(
                        case.spec.model_dump(mode="json", by_alias=True), sort_keys=True
                    ).encode()
                ).hexdigest(),
                "source_mesh_sha256": level["mesh_sha256"],
                "mesh": asdict(record),
            }
            (run / "level.json").write_text(json.dumps(info, indent=2) + "\n")
            dirs.append(run)
    return dirs


@dataclass(frozen=True)
class Level:
    """One SU2 run of the study: its convergence and the provenance of its numbers."""

    run: str
    n: int
    points: int
    venkat_coefficient: float
    converged: bool
    iterations: int
    wall_seconds: float | None
    ranks: int
    cl: float
    cd: float
    cmy: float
    drop: dict[str, float]
    case_sha256: str
    plane_sha256: str
    su2: str
    image_id: str


def read_level(run: Path) -> tuple[Level, str]:
    """A prepared and run case directory, and its condition. V0904 if a file is missing."""
    try:
        info = json.loads((run / "level.json").read_text())
        prov = json.loads((run / "provenance.json").read_text())
    except FileNotFoundError as err:
        msg = f"{run}: {err.filename} is missing"
        raise VVError(msg, code="V0904") from None
    if prov.get("exit_status") != 0:
        msg = f"{run}: SU2 exited with status {prov.get('exit_status')}"
        raise VVError(msg, code="V0904")
    text = (run / "case.cfg").read_text()
    if hashlib.sha256(text.encode()).hexdigest() != info["case_sha256"]:
        msg = f"{run}: case.cfg is not the prepared one"
        raise VVError(msg, code="V0904")
    conv = history.convergence(history.read_su2(run / "history.csv"), cases.RESIDUAL_ORDERS)
    level = Level(
        run=run.name,
        n=info["n"],
        points=info["mesh"]["points"],
        venkat_coefficient=info["venkat_coefficient"],
        converged=conv.converged,
        iterations=conv.iterations,
        wall_seconds=conv.wall_seconds,
        ranks=prov["ranks"],
        cl=conv.cl,
        cd=conv.cd,
        cmy=conv.cmy,
        drop=conv.drop,
        case_sha256=info["case_sha256"],
        plane_sha256=info["mesh"]["plane_sha256"],
        su2=prov["su2"],
        image_id=prov["image_id"],
    )
    return level, info["condition"]


def _as_forces(level: Level) -> cases.LevelForces:
    return cases.LevelForces(
        run=level.run,
        n_cells=level.n * level.n,
        h=1.0 / level.n,
        cl=level.cl,
        cd=level.cd,
        cmy=level.cmy,
        status="converged" if level.converged else "completed",
        backend="su2",
        commit=level.su2,
        precision="fp64",
        case_sha256=level.case_sha256,
        mesh_sha256=level.plane_sha256,
    )


def study(runs: Sequence[Path]) -> dict[str, Any]:
    """Every level of one condition with its iterations and forces at the criterion; continuum
    CL, CD, CM from the three finest converged levels and the paper's gates when at least
    cases.MIN_LEVELS levels converged. V0905 if the runs mix conditions, SU2 builds or levels."""
    read = [read_level(r) for r in runs]
    levels = sorted((lv for lv, _ in read), key=lambda lv: lv.n)
    conditions = {c for _, c in read}
    if len(conditions) != 1:
        msg = f"the runs mix conditions {sorted(conditions)}"
        raise VVError(msg, code="V0905")
    if len({lv.image_id for lv in levels}) != 1 or len({lv.n for lv in levels}) != len(levels):
        msg = "the runs mix SU2 images or repeat a level"
        raise VVError(msg, code="V0905")
    condition = conditions.pop()
    converged = [_as_forces(lv) for lv in levels if lv.converged]
    out: dict[str, Any] = {
        "schema": STUDY_SCHEMA,
        "case": "naca0012",
        "condition": condition,
        "su2": levels[0].su2,
        "image_id": levels[0].image_id,
        "levels": [asdict(lv) for lv in levels],
    }
    if len(converged) < cases.MIN_LEVELS:
        out["continuum"] = None
        out["verdicts"] = []
        out["not_estimated_because"] = f"{len(converged)} converged levels"
        return out
    estimates = cases.continuum(converged)
    out["continuum"] = estimates
    out["verdicts"] = [
        cases.check(g, converged, estimates)
        for g in cases.GATES.values()
        if g.condition == condition
    ]
    return out
