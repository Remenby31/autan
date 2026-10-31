"""Run directories ``autan.results/1``: writing them and reading them back (docs/contracts/results.md)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autan import cgns
from autan.errors import CaseError, Diagnostic, Location, diagnostic
from autan.meshinfo import MeshInfo
from autan.schema import DDES, SA, SST, CaseSpec, Compressible, Steady

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from autan.schema import Forces as ForcesSpec

SCHEMA_ID = "autan.results/1"
Status = Literal["running", "converged", "completed", "stopped", "diverged", "failed"]
SOLUTION_REQUIRED: frozenset[str] = frozenset({"converged", "completed", "stopped"})
Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseIdentity(_Strict):
    name: str
    sha256: Hex


class MeshIdentity(_Strict):
    path: str
    sha256: Hex


class Solver(_Strict):
    version: str
    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    backend: Literal["cuda", "hip", "cpu", "fake"]


class Devices(_Strict):
    kind: Literal["cuda", "hip", "cpu"]
    count: Annotated[int, Field(ge=1)]


class Averaging(_Strict):
    from_step: int
    to_step: int
    from_time: float
    to_time: float


class RunInfo(_Strict):
    """The content of ``run.json``."""

    schema_: Literal["autan.results/1"] = Field(default="autan.results/1", alias="schema")
    run_id: Annotated[str, Field(pattern=r"^run-\d{8}T\d{6}Z-[0-9a-f]{8}$")]
    identity: Hex
    status: Status
    case: CaseIdentity
    mesh: MeshIdentity
    solver: Solver
    precision: Literal["fp32+acc64", "fp64"]
    deterministic: bool
    devices: Devices
    started: str
    finished: str | None
    iterations: Annotated[int, Field(ge=0)]
    time: float | None
    averaging: Averaging | None
    diagnostics: tuple[Diagnostic, ...]

    model_config = ConfigDict(populate_by_name=True)

    def document(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def identity(case: str, mesh: str, solver: Solver, precision: str, devices: Devices) -> str:
    """The reproducibility fingerprint of a run (results.md, ``identity``)."""
    parts = [case, mesh, solver.commit, precision, solver.backend, devices.kind, str(devices.count)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


class RunWriter:
    """Creates a run directory and keeps ``run.json`` and ``history.jsonl`` up to date.

    The directory is prepared under a hidden name and renamed once ``run.json``,
    ``case.toml`` and ``mesh-info.json`` are written, so a reader listing ``runs``
    never sees a run without them.
    """

    def __init__(self, runs: Path, info: RunInfo, spec: CaseSpec, mesh_info: MeshInfo) -> None:
        runs.mkdir(parents=True, exist_ok=True)
        staging = runs / f".{info.run_id}.tmp"
        staging.mkdir()
        self.path = staging
        self.info = info
        (staging / "case.toml").write_text(_toml(spec))
        (staging / "mesh-info.json").write_text(json.dumps(mesh_info.document(), indent=2))
        self._write_info()
        final = runs / info.run_id
        if final.exists():
            message = f"run directory {final} already exists"
            raise FileExistsError(message)
        staging.rename(final)
        self.path = final

    @staticmethod
    def new_run_id(started: datetime) -> str:
        return f"run-{started.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"

    def _write_info(self) -> None:
        temporary = self.path / "run.json.tmp"
        temporary.write_text(json.dumps(self.info.document(), indent=2) + "\n")
        temporary.replace(self.path / "run.json")

    def append(self, record: dict[str, Any]) -> None:
        with (self.path / "history.jsonl").open("a") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()

    def finish(self, **changes: Any) -> None:
        self.info = self.info.model_copy(update={**changes, "finished": iso(utc_now())})
        self._write_info()


def _toml(spec: CaseSpec) -> str:
    import tomli_w  # noqa: PLC0415 -- only writers need it

    return tomli_w.dumps(spec.document())


class History:
    """Residual and monitor time series of a run, as NumPy arrays."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.iterations: NDArray[np.int64] = np.array(
            [r["iteration"] for r in records], dtype=np.int64
        )

    def residual(self, equation: str) -> NDArray[np.float64]:
        return np.array([r["residuals"][equation] for r in self._records], dtype=np.float64)

    @property
    def equations(self) -> list[str]:
        return list(self._records[0]["residuals"]) if self._records else []

    def monitor(self, name: str) -> dict[str, NDArray[Any]]:
        """Every quantity of monitor ``name``, with the iterations it was recorded at."""
        rows = [
            (r["iteration"], r["monitors"][name]) for r in self._records if name in r["monitors"]
        ]
        if not rows:
            raise CaseError([diagnostic("E0702", f"no monitor named {name!r} in the history")])
        keys = rows[0][1].keys()
        series: dict[str, NDArray[Any]] = {
            "iteration": np.array([i for i, _ in rows], dtype=np.int64)
        }
        for key in keys:
            series[key] = np.array([values[key] for _, values in rows], dtype=np.float64)
        return series

    @property
    def monitors(self) -> list[str]:
        names: dict[str, None] = {}
        for record in self._records:
            names.update(dict.fromkeys(record["monitors"]))
        return list(names)


class Forces:
    """A forces monitor: ``fx`` … ``mz`` in SI and, when available, coefficients."""

    def __init__(self, series: dict[str, NDArray[Any]]) -> None:
        self._series = series

    def __getattr__(self, name: str) -> NDArray[Any]:
        try:
            return self._series[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def quantities(self) -> list[str]:
        return list(self._series)


class Results:
    """A run directory read back and checked against ``autan.results/1``."""

    def __init__(
        self, path: Path, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo, history: History
    ) -> None:
        self.path, self.run, self.case, self.mesh_info, self.history = (
            path,
            run,
            case,
            mesh_info,
            history,
        )
        self._solution: cgns.Node | None = None

    @classmethod
    def open(cls, path: str | Path) -> Results:
        """Read the run directory at ``path``; any departure from the contract is ``E0701``."""
        path = Path(path)

        def broken(message: str, file: str | None = None) -> NoReturn:
            where = Location(file=str(path / file) if file else str(path))
            raise CaseError([diagnostic("E0701", message, location=where)])

        try:
            document = json.loads((path / "run.json").read_text())
        except (OSError, ValueError) as error:
            broken(f"cannot read run.json: {error}", "run.json")
        if not isinstance(document, dict) or document.get("schema") != SCHEMA_ID:
            broken(f"run.json schema is not {SCHEMA_ID!r}", "run.json")
        try:
            run = RunInfo.model_validate(document)
        except ValidationError as error:
            first = error.errors(include_url=False)[0]
            broken(f"run.json: {first['msg']} at {first['loc']}", "run.json")
        try:
            case_document = tomllib.loads((path / "case.toml").read_text())
            mesh_document = json.loads((path / "mesh-info.json").read_text())
        except (OSError, ValueError) as error:
            broken(str(error))
        try:
            case = CaseSpec.from_document(case_document, Location(file=str(path / "case.toml")))
            mesh_info = MeshInfo.from_document(
                mesh_document, Location(file=str(path / "mesh-info.json"))
            )
        except CaseError as error:
            broken(f"stored case or MeshInfo is invalid: {error.codes}")
        for problem in run_problems(path, run, case, mesh_info):
            broken(problem, "run.json")
        records = _read_history(path / "history.jsonl", run.status == "running", broken)
        for problem in history_problems(records, run, case):
            broken(problem, "history.jsonl")
        results = cls(path, run, case, mesh_info, History(records))
        solution = path / "solution.cgns"
        if run.status != "running" and solution.exists():
            try:
                results._solution = cgns.read(solution)
            except (OSError, ValueError, KeyError) as error:
                broken(f"cannot read solution.cgns: {error}", "solution.cgns")
            for problem in solution_problems(results._solution, run, case, mesh_info):
                broken(problem, "solution.cgns")
        elif run.status in SOLUTION_REQUIRED:
            broken(f"a {run.status} run must have solution.cgns", "solution.cgns")
        return results

    @property
    def forces(self) -> Forces:
        """The forces monitor of the case, when it has exactly one."""
        names = [m.name for m in self.case.monitors if m.kind == "forces"]
        if len(names) != 1:
            raise CaseError(
                [
                    diagnostic(
                        "E0702",
                        f"the case has {len(names)} forces monitors ({', '.join(names)}); "
                        "use results.history.monitor(name)",
                    )
                ]
            )
        return Forces(self.history.monitor(names[0]))

    def _tree(self) -> cgns.Node:
        if self._solution is None:
            path = self.path / "solution.cgns"
            reason = "is still running" if self.run.status == "running" else "has no solution"
            message = f"run {self.run.run_id} ({self.run.status}) {reason}"
            raise CaseError([diagnostic("E0703", message, location=Location(file=str(path)))])
        return self._solution

    def _zone(self) -> cgns.Node:
        return self._tree().child("Base").child("Zone")

    def field(self, name: str, *, mean: bool = False) -> NDArray[np.float64]:
        """Cell-centred field ``name`` (SIDS name, SI units), in mesh cell order."""
        name_of_solution = "FlowSolutionMean" if mean else "FlowSolution"
        zone = self._zone()
        if name_of_solution not in zone.children:
            message = f"run {self.run.run_id} has no time average"
            raise CaseError([diagnostic("E0703", message)])
        solution = zone.child(name_of_solution)
        if name not in solution.children:
            raise CaseError(
                [
                    diagnostic(
                        "E0703",
                        f"no field {name!r}; fields: "
                        f"{', '.join(k for k in solution.children if k != 'GridLocation')}",
                    )
                ]
            )
        return np.asarray(solution.child(name).data, dtype=np.float64)

    def wall(self, name: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Wall field ``name`` with the 0-based indices of the faces it is given on."""
        region = self._zone().child("WallSolution")
        faces = np.asarray(region.child("PointList").data, dtype=np.int64).ravel() - 1
        return faces, np.asarray(region.child(name).data, dtype=np.float64)

    def cell_zone(self, zone: str) -> NDArray[np.int64]:
        """0-based indices of the cells of cell zone ``zone``."""
        names = json.loads(self._tree().child("Base").child("ZoneNames").text())["cell"]
        node_name = next(k for k, v in names.items() if v == zone)
        zone_node = self._zone()
        offset = int(zone_node.child("NFaceElements").child("ElementRange").data[0])
        points = np.asarray(zone_node.child(node_name).child("PointList").data).ravel()
        return points.astype(np.int64) - offset


def residual_names(spec: CaseSpec) -> list[str]:
    """The equations of the history, in order (results.md#history)."""
    names = (
        ["density", "momentum-x", "momentum-y", "momentum-z", "energy"]
        if isinstance(spec.physics, Compressible)
        else ["pressure", "momentum-x", "momentum-y", "momentum-z"]
    )
    base = _base(spec)
    if isinstance(base, SA):
        return [*names, "nu-tilde"]
    return [*names, "k", "omega"] if isinstance(base, SST) else names


def field_names(spec: CaseSpec) -> list[str]:
    """The FlowSolution fields a run of ``spec`` writes (results.md#solutioncgns)."""
    names = ["Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"]
    if isinstance(spec.physics, Compressible):
        names.append("Temperature")
    base = _base(spec)
    if isinstance(base, SA):
        return [*names, "TurbulentSANuTilde"]
    if isinstance(base, SST):
        return [*names, "TurbulentEnergyKinetic", "TurbulentDissipationRate"]
    return names


def force_keys(monitor: ForcesSpec, spec: CaseSpec) -> set[str]:
    keys = {"fx", "fy", "fz", "mx", "my", "mz"}
    if monitor.ref_area is not None:
        keys |= {"cd", "cs", "cl"}
        if spec.freestream.ref_length is not None:
            keys |= {"cmx", "cmy", "cmz"}
    return keys


def _base(spec: CaseSpec) -> object:
    turbulence = spec.physics.turbulence
    return turbulence.base if isinstance(turbulence, DDES) else turbulence


_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


def run_problems(path: Path, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo) -> list[str]:
    """Invariants of run.json against its directory, case and MeshInfo."""
    found = []
    if path.name != run.run_id:
        found.append(f"directory {path.name!r} is not named after run_id {run.run_id!r}")
    if not _TIMESTAMP.fullmatch(run.started):
        found.append(f"started {run.started!r} is not a UTC ISO 8601 timestamp")
    elif run.run_id[4:20] != run.started.replace("-", "").replace(":", ""):
        found.append("run_id does not carry the started timestamp")
    if (run.finished is None) != (run.status == "running"):
        found.append("finished must be null exactly while running")
    if run.finished is not None and not _TIMESTAMP.fullmatch(run.finished):
        found.append(f"finished {run.finished!r} is not a UTC ISO 8601 timestamp")
    expected = identity(
        case.hash(), mesh_info.source.sha256, run.solver, run.precision, run.devices
    )
    if run.identity != expected:
        found.append("identity does not match its definition")
    if case.hash() != run.case.sha256 or case.name != run.case.name:
        found.append("case.toml does not match run.json case")
    if mesh_info.source.sha256 != run.mesh.sha256:
        found.append("mesh-info.json does not match run.json mesh.sha256")
    if not Path(run.mesh.path).is_absolute():
        found.append("mesh.path must be absolute")
    if run.precision != case.physics.precision:
        found.append("precision differs from the case's")
    steady = isinstance(case.time, Steady)
    if steady != (run.time is None):
        found.append("time must be null exactly for steady runs")
    if run.averaging is not None and steady:
        found.append("averaging is only for unsteady runs")
    if run.status == "diverged" and not any(d.code.startswith("D") for d in run.diagnostics):
        found.append("a diverged run must carry a D diagnostic")
    return found


def _read_history(path: Path, live: bool, broken: Any) -> list[dict[str, Any]]:
    """Parse history.jsonl; while the run is live, its last line may still be being written."""
    if not path.exists():
        return []
    text = path.read_text()
    complete, _, partial = text.rpartition("\n")
    if partial and not live:
        broken("the last line is not terminated", "history.jsonl")
    records = []
    for number, line in enumerate(
        complete.splitlines() if complete or not partial else [], start=1
    ):
        try:
            record = json.loads(line)
        except ValueError as error:
            broken(f"line {number}: {error}", "history.jsonl")
        if not isinstance(record, dict):
            broken(f"line {number}: not an object", "history.jsonl")
        records.append(record)
    return records


def history_problems(records: list[dict[str, Any]], run: RunInfo, case: CaseSpec) -> list[str]:
    """Every line complete and finite, iterations increasing, the last one the run's."""
    found: list[str] = []
    equations = residual_names(case)
    monitors = {m.name: m for m in case.monitors}
    steady = isinstance(case.time, Steady)
    previous = 0
    for record in records:
        where = f"iteration {record.get('iteration')!r}"
        if set(record) != {"iteration", "time", "residuals", "monitors"}:
            return [f"{where}: keys must be iteration, time, residuals, monitors"]
        iteration = record["iteration"]
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration <= previous:
            return [f"{where}: iterations must be increasing integers"]
        previous = iteration
        if steady != (record["time"] is None) or not (steady or _finite(record["time"])):
            found.append(f"{where}: time must be null for steady runs, a number otherwise")
        residuals = record["residuals"]
        if not isinstance(residuals, dict) or list(residuals) != equations:
            found.append(f"{where}: residuals must be {equations}")
        elif not all(_finite(v) and v >= 0 for v in residuals.values()):
            found.append(f"{where}: residuals must be finite and non-negative")
        recorded = record["monitors"]
        due = {name for name, m in monitors.items() if iteration % m.every == 0}
        if not isinstance(recorded, dict) or set(recorded) != due:
            found.append(f"{where}: monitors must be exactly {sorted(due)}")
            continue
        for name, values in recorded.items():
            if not isinstance(values, dict) or set(values) != force_keys(monitors[name], case):
                found.append(f"{where}: monitor {name!r} has the wrong quantities")
            elif not all(_finite(v) for v in values.values()):
                found.append(f"{where}: monitor {name!r} has non-finite values")
    if run.status != "running":
        last = records[-1]["iteration"] if records else 0
        if last != run.iterations:
            found.append(f"history ends at iteration {last}, run.json says {run.iterations}")
    return found


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


UNITS = ("Kilogram", "Meter", "Second", "Kelvin", "Radian")


def solution_problems(
    tree: cgns.Node, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo
) -> list[str]:
    """The solution.cgns layout of results.md: labels, sizes, connectivity, fields, names."""
    try:
        return _solution_problems(tree, run, case, mesh_info)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return [f"missing or malformed node: {type(error).__name__}: {error}"]


def _solution_problems(
    tree: cgns.Node, run: RunInfo, case: CaseSpec, mesh_info: MeshInfo
) -> list[str]:
    found: list[str] = []
    base = tree.child("Base")
    zone = base.child("Zone")
    for node, label in ((base, "CGNSBase_t"), (zone, "Zone_t")):
        if node.label != label:
            found.append(f"{node.name} is not a {label}")
    if base.child("DimensionalUnits").text_rows() != list(UNITS):
        found.append(f"DimensionalUnits must be {list(UNITS)}")
    if json.loads(base.child("RunMetadata").text()).get("identity") != run.identity:
        found.append("RunMetadata does not match run.json")
    names = json.loads(base.child("ZoneNames").text())
    if sorted(names["boundary"].values()) != sorted(z.name for z in mesh_info.boundary_zones):
        found.append("ZoneNames boundary zones differ from the MeshInfo")
    if sorted(names["cell"].values()) != sorted(z.name for z in mesh_info.cell_zones):
        found.append("ZoneNames cell zones differ from the MeshInfo")

    nodes, cells, _ = (int(v) for v in np.asarray(zone.data).ravel())
    if zone.child("ZoneType").text() != "Unstructured":
        found.append("ZoneType must be Unstructured")
    coordinates = zone.child("GridCoordinates")
    found.extend(
        f"Coordinate{axis} must have one value per node"
        for axis in "XYZ"
        if np.asarray(coordinates.child(f"Coordinate{axis}").data).size != nodes
    )
    ngon, nface = zone.child("NGonElements"), zone.child("NFaceElements")
    faces = int(ngon.child("ElementRange").data[1])
    found += _elements_problems(ngon, 22, 1, faces, 1, nodes, signed=False)
    found += _elements_problems(nface, 23, faces + 1, faces + cells, 1, faces, signed=True)

    solution = zone.child("FlowSolution")
    if solution.child("GridLocation").text() != "CellCenter":
        found.append("FlowSolution must be at CellCenter")
    for name in field_names(case):
        values = np.asarray(solution.children[name].data) if name in solution.children else None
        if values is None or values.size != cells or not np.all(np.isfinite(values)):
            found.append(f"FlowSolution {name} must hold {cells} finite values")
    if ("FlowSolutionMean" in zone.children) != (run.averaging is not None):
        found.append("FlowSolutionMean must exist exactly when run.json has averaging")
    for bc in zone.child("ZoneBC").with_label("BC_t"):
        if not _ids_in(bc.child("PointList").data, 1, faces):
            found.append(f"BC {bc.name} PointList must hold face ids")
    for region in zone.with_label("ZoneSubRegion_t"):
        low, high = (1, faces) if region.name == "WallSolution" else (faces + 1, faces + cells)
        if not _ids_in(region.child("PointList").data, low, high):
            found.append(f"ZoneSubRegion {region.name} PointList is out of range")
    return found


def _elements_problems(
    node: cgns.Node, kind: int, first: int, last: int, low: int, high: int, *, signed: bool
) -> list[str]:
    found = []
    if list(np.asarray(node.data)) != [kind, 0]:
        found.append(f"{node.name} must be element type {kind}")
    if list(np.asarray(node.child("ElementRange").data)) != [first, last]:
        found.append(f"{node.name} ElementRange must be [{first}, {last}]")
    offsets = np.asarray(node.child("ElementStartOffset").data)
    connectivity = np.asarray(node.child("ElementConnectivity").data)
    if (
        offsets.size != last - first + 2
        or offsets[0] != 0
        or np.any(np.diff(offsets) < (3 if not signed else 4))
        or offsets[-1] != connectivity.size
    ):
        found.append(f"{node.name} ElementStartOffset is inconsistent")
    ids = np.abs(connectivity) if signed else connectivity
    if not _ids_in(ids, low, high) or (signed and np.any(connectivity == 0)):
        found.append(f"{node.name} ElementConnectivity is out of range")
    return found


def _ids_in(ids: Any, low: int, high: int) -> bool:
    array = np.asarray(ids)
    return bool(array.size == 0 or (array.min() >= low and array.max() <= high))
