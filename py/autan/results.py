"""Run directories ``autan.results/1``: writing them and reading them back (docs/contracts/results.md)."""

from __future__ import annotations

import hashlib
import json
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
from autan.schema import CaseSpec

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    """Creates a run directory and keeps ``run.json`` and ``history.jsonl`` up to date."""

    def __init__(self, runs: Path, info: RunInfo, spec: CaseSpec, mesh_info: MeshInfo) -> None:
        self.path = runs / info.run_id
        runs.mkdir(parents=True, exist_ok=True)
        self.path.mkdir()
        self.info = info
        (self.path / "case.toml").write_text(_toml(spec))
        (self.path / "mesh-info.json").write_text(json.dumps(mesh_info.document(), indent=2))
        self._write_info()

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
            broken(
                f"run.json: {error.errors(include_url=False)[0]['msg']} at "
                f"{error.errors()[0]['loc']}",
                "run.json",
            )
        try:
            case_document = tomllib.loads((path / "case.toml").read_text())
            mesh_document = json.loads((path / "mesh-info.json").read_text())
        except (OSError, ValueError) as error:
            broken(str(error))
        case = CaseSpec.from_document(case_document, Location(file=str(path / "case.toml")))
        mesh_info = MeshInfo.from_document(
            mesh_document, Location(file=str(path / "mesh-info.json"))
        )
        if case.hash() != run.case.sha256:
            broken("case.toml does not match run.json case.sha256", "case.toml")
        if mesh_info.source.sha256 != run.mesh.sha256:
            broken("mesh-info.json does not match run.json mesh.sha256", "mesh-info.json")
        history_path = path / "history.jsonl"
        records = (
            [json.loads(line) for line in history_path.read_text().splitlines()]
            if history_path.exists()
            else []
        )
        iterations = [r["iteration"] for r in records]
        if iterations != sorted(set(iterations)):
            broken("history.jsonl iterations are not strictly increasing", "history.jsonl")
        if run.status in SOLUTION_REQUIRED and not (path / "solution.cgns").exists():
            broken(f"a {run.status} run must have solution.cgns", "solution.cgns")
        return cls(path, run, case, mesh_info, History(records))

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
            if not path.exists():
                raise CaseError(
                    [
                        diagnostic(
                            "E0703",
                            f"run {self.run.run_id} ({self.run.status}) has no solution",
                            location=Location(file=str(path)),
                        )
                    ]
                )
            self._solution = cgns.read(path)
        return self._solution

    def _zone(self) -> cgns.Node:
        return self._tree().child("Base").child("Zone")

    def field(self, name: str, *, mean: bool = False) -> NDArray[np.float64]:
        """Cell-centred field ``name`` (SIDS name, SI units), in mesh cell order."""
        solution = self._zone().child("FlowSolutionMean" if mean else "FlowSolution")
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
