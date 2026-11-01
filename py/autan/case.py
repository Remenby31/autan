"""``autan.Case``: build a case in Python, load it from a file, plan it."""

from __future__ import annotations

import json
import runpy
import tomllib
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import tomli_w

from autan.errors import CaseError, Location, diagnostic
from autan.plan import Plan, plan
from autan.restart import resolve as resolve_start
from autan.schema import (
    BoundaryRule,
    CaseSpec,
    CellZoneRule,
    Compressible,
    ConstantDensityFluid,
    Farfield,
    Fluid,
    Forces,
    Freestream,
    IdealGas,
    Incompressible,
    Inlet,
    Manufactured,
    OracleState,
    Outlet,
    PointVortex,
    Slip,
    Solid,
    SolidZone,
    State,
    StateFile,
    Steady,
    Symmetry,
    Unsteady,
    VelocityInlet,
    Verification,
    Wall,
    _caller_location,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pint

    from autan.fake import FakeBackend
    from autan.meshinfo import MeshInfo
    from autan.results import Results
    from autan.solver_backend import SolverBackend

_collectors: list[list[Case]] = []


class Case:
    """A case under construction: ``autan.Case("crm-hl", mesh="crm-hl.cgns")``.

    Builder methods validate their arguments immediately and raise ``autan.CaseError``
    with a coded diagnostic that points at the calling line. ``spec`` returns the
    complete, validated document (``autan.case/1``).
    """

    def __init__(self, name: str, mesh: str | Path) -> None:
        self._origin = _caller_location()
        self._name = name
        self._mesh = str(mesh)
        self._physics: Compressible | Incompressible | None = None
        self._freestream: Freestream | None = None
        self._time: Steady | Unsteady = Steady()
        self._boundaries: list[BoundaryRule] = []
        self._cell_zones: list[CellZoneRule] = []
        self._monitors: list[Forces] = []
        self._verification: Verification | None = None
        self._initial_state: StateFile | None = None
        self._locations: dict[str, Location] = {}
        self._document_file: str | None = None
        if _collectors:
            _collectors[-1].append(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mesh(self) -> str:
        return self._mesh

    @property
    def mesh_path(self) -> Path:
        """Absolute path of the mesh: ``mesh`` relative to the case file's directory."""
        return self.resolve(self._mesh)

    @property
    def physics(self) -> Compressible | Incompressible | None:
        return self._physics

    @physics.setter
    def physics(self, physics: Compressible | Incompressible) -> None:
        if not isinstance(physics, Compressible | Incompressible):
            raise CaseError(
                [
                    diagnostic(
                        "E0003",
                        f"physics must be autan.Compressible or autan.Incompressible, "
                        f"got {type(physics).__name__}",
                        location=_caller_location(),
                    )
                ]
            )
        self._physics = physics
        self._locations["physics"] = _caller_location()

    @property
    def initial_state(self) -> StateFile | None:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: StateFile) -> None:
        """Start every fluid cell from a file: ``autan.StateFile(path=..., sha256=...)``."""
        if not isinstance(state, StateFile):
            raise CaseError(
                [
                    diagnostic(
                        "E0003",
                        f"initial_state must be autan.StateFile, got {type(state).__name__}",
                        location=_caller_location(),
                    )
                ]
            )
        self._initial_state = state
        self._locations["initial_state"] = _caller_location()

    @property
    def verification(self) -> Verification | None:
        return self._verification

    @verification.setter
    def verification(self, manufactured: Manufactured) -> None:
        """Verify against a manufactured solution: ``autan.Manufactured(path=..., sha256=...)``."""
        if not isinstance(manufactured, Manufactured):
            raise CaseError(
                [
                    diagnostic(
                        "E0003",
                        f"verification must be autan.Manufactured, got "
                        f"{type(manufactured).__name__}",
                        location=_caller_location(),
                    )
                ]
            )
        self._verification = Verification(manufactured=manufactured)
        self._locations["verification"] = _caller_location()

    def resolve(self, relative: str) -> Path:
        """Absolute path of a file the case names, relative to the case file's directory."""
        source = self._document_file or self._origin.file
        base = Path(source).parent if source and Path(source).is_file() else Path.cwd()
        return (base / relative).resolve()

    @property
    def time(self) -> Steady | Unsteady:
        return self._time

    @time.setter
    def time(self, time: Steady | Unsteady) -> None:
        if not isinstance(time, Steady | Unsteady):
            raise CaseError(
                [
                    diagnostic(
                        "E0003",
                        f"time must be autan.Steady or autan.Unsteady, got {type(time).__name__}",
                        location=_caller_location(),
                    )
                ]
            )
        self._time = time
        self._locations["time"] = _caller_location()

    def freestream(
        self,
        *,
        mach: float | None = None,
        velocity: pint.Quantity | None = None,
        aoa: pint.Quantity | None = None,
        sideslip: pint.Quantity | None = None,
        temperature: pint.Quantity | None = None,
        pressure: pint.Quantity | None = None,
        re: float | None = None,
        ref_length: pint.Quantity | None = None,
    ) -> None:
        """Set the freestream state; which parameters apply depends on the physics."""
        given = {
            key: value for key, value in locals().items() if key != "self" and value is not None
        }
        self._freestream = Freestream(**given)
        self._locations["freestream"] = _caller_location()

    def boundary(self, zones: str) -> BoundarySelector:
        """Select boundary zones by glob pattern; call a condition method on the result."""
        return BoundarySelector(self, zones)

    def cell_zone(self, zones: str) -> CellZoneSelector:
        """Select cell zones by glob pattern; call ``fluid`` or ``solid`` on the result."""
        return CellZoneSelector(self, zones)

    def monitor(self, monitor: Forces, *, every: int | None = None) -> None:
        """Record ``monitor`` every ``every`` iterations."""
        if not isinstance(monitor, Forces):
            raise CaseError(
                [
                    diagnostic(
                        "E0003",
                        f"monitor must be built by autan.forces(...), got {type(monitor).__name__}",
                        location=_caller_location(),
                    )
                ]
            )
        if every is not None:
            monitor = Forces(**{**monitor.model_dump(), "every": every})
        self._locations[f"monitors[{len(self._monitors)}]"] = _caller_location()
        self._monitors.append(monitor)

    def _add_boundary(self, rule: BoundaryRule) -> None:
        self._locations[f"boundaries[{len(self._boundaries)}]"] = _caller_location()
        self._boundaries.append(rule)

    def _add_cell_zone(self, rule: CellZoneRule) -> None:
        self._locations[f"cell_zones[{len(self._cell_zones)}]"] = _caller_location()
        self._cell_zones.append(rule)

    @property
    def spec(self) -> CaseSpec:
        """The validated case document."""
        missing = [
            key
            for key, value in (("physics", self._physics), ("freestream", self._freestream))
            if value is None
        ]
        if missing:
            raise CaseError(
                diagnostic(
                    "E0003",
                    f"case.{key} is not set",
                    location=self._origin,
                    fix=(
                        "case.physics = autan.Compressible(turbulence=autan.SA())"
                        if key == "physics"
                        else "case.freestream(mach=..., re=..., ref_length=...)"
                    ),
                )
                for key in missing
            )
        try:
            return CaseSpec(
                name=self._name,
                mesh=self._mesh,
                physics=self._physics,
                freestream=self._freestream,
                time=self._time,
                boundaries=tuple(self._boundaries),
                cell_zones=tuple(self._cell_zones),
                monitors=tuple(self._monitors),
                verification=self._verification,
                initial_state=self._initial_state,
            )
        except CaseError as error:
            raise CaseError(
                d.model_copy(update={"location": self.locate(d.location.path or "")})
                if d.location is not None
                else d
                for d in error.diagnostics
            ) from None

    def locate(self, path: str) -> Location:
        """Location of the document path ``path`` (``"boundaries[2]"``) in the case source."""
        if self._document_file is not None:
            return Location(file=self._document_file, path=path or None)
        parts = path.split(".")
        for end in range(len(parts), 0, -1):
            found = self._locations.get(".".join(parts[:end]))
            if found is not None:
                return found.model_copy(update={"path": path or None})
        return self._origin.model_copy(update={"path": path or None})

    @classmethod
    def from_spec(cls, spec: CaseSpec, *, file: str | None = None) -> Case:
        """A case holding ``spec``; diagnostics point at ``file`` and document paths."""
        case = cls.__new__(cls)
        case._origin = Location(file=file)
        case._name, case._mesh = spec.name, spec.mesh
        case._physics, case._freestream, case._time = spec.physics, spec.freestream, spec.time
        case._boundaries = list(spec.boundaries)
        case._cell_zones = list(spec.cell_zones)
        case._monitors = list(spec.monitors)
        case._verification = spec.verification
        case._initial_state = spec.initial_state
        case._locations = {}
        case._document_file = file
        return case

    def to_toml(self) -> str:
        return tomli_w.dumps(self.spec.document())

    def to_json(self) -> str:
        return json.dumps(self.spec.document(), indent=2) + "\n"

    def run(
        self,
        mesh_info: MeshInfo,
        *,
        backend: FakeBackend | SolverBackend | None = None,
        runs: str | Path = "runs",
        start: Results | None = None,
        mode: Literal["continue", "warm"] = "continue",
    ) -> Results:
        """Run the case with ``backend`` into a new directory under ``runs``; with ``start``,
        from the state of that previous run (``continue`` it, or ``warm``-start a case)."""
        if backend is None:
            missing = diagnostic(
                "E0804",
                "case.run() needs a backend: autan.SolverBackend() or autan.FakeBackend()",
                location=_caller_location(),
            )
            raise CaseError([missing])
        checked = resolve_start(self.spec, mesh_info, start, mode)
        return backend.run(self, mesh_info, Path(runs), checked)

    def plan(self, mesh_info: MeshInfo, *, gpus: int = 1) -> Plan:
        """Resolve boundaries and cell zones against the mesh and estimate memory."""
        return plan(self, mesh_info, gpus=gpus)


class BoundarySelector:
    """The boundary zones matching a pattern; each method adds one rule to the case."""

    def __init__(self, case: Case, zones: str) -> None:
        self._case, self._zones = case, zones

    def _add(self, condition: Any) -> None:
        self._case._add_boundary(BoundaryRule(zones=self._zones, condition=condition))

    def wall(self, *, temperature: pint.Quantity | None = None) -> None:
        """No-slip wall, adiabatic unless ``temperature`` is given."""
        self._add(Wall() if temperature is None else Wall(temperature=temperature))

    def slip(self) -> None:
        """Inviscid (slip) wall."""
        self._add(Slip())

    def symmetry(self) -> None:
        self._add(Symmetry())

    def farfield(self, *, point_vortex: PointVortex | None = None) -> None:
        """Characteristic farfield at the freestream state, corrected by ``point_vortex`` for
        a 2D lifting body."""
        self._add(Farfield(point_vortex=point_vortex))

    def inlet(self, *, total_pressure: pint.Quantity, total_temperature: pint.Quantity) -> None:
        """Subsonic inlet at total conditions (compressible)."""
        self._add(Inlet(total_pressure=total_pressure, total_temperature=total_temperature))

    def velocity_inlet(self, *, velocity: pint.Quantity) -> None:
        """Inlet at a given normal velocity (incompressible)."""
        self._add(VelocityInlet(velocity=velocity))

    def outlet(self, *, static_pressure: pint.Quantity) -> None:
        """Subsonic outlet at a static pressure."""
        self._add(Outlet(static_pressure=static_pressure))

    def oracle_state(self, *, path: str, sha256: str) -> None:
        """Impose the per-face states of an ``autan.face-states/1`` file (compressible)."""
        self._add(OracleState(path=path, sha256=sha256))


class CellZoneSelector:
    """The cell zones matching a pattern; each method adds one rule to the case."""

    def __init__(self, case: Case, zones: str) -> None:
        self._case, self._zones = case, zones

    def fluid(
        self, material: IdealGas | ConstantDensityFluid, *, initial: State | None = None
    ) -> None:
        """A fluid zone of ``material``, starting from ``initial`` (default: the freestream)."""
        medium = (
            Fluid(material=material)
            if initial is None
            else Fluid(material=material, initial=initial)
        )
        self._case._add_cell_zone(CellZoneRule(zones=self._zones, medium=medium))

    def solid(self, material: Solid) -> None:
        self._case._add_cell_zone(
            CellZoneRule(zones=self._zones, medium=SolidZone(material=material))
        )


def forces(
    surfaces: str,
    *,
    name: str = "forces",
    ref_area: pint.Quantity | None = None,
    moment_center: tuple[pint.Quantity, pint.Quantity, pint.Quantity] | None = None,
) -> Forces:
    """A forces monitor over the boundary zones matching ``surfaces``."""
    given: dict[str, Any] = {"surfaces": surfaces, "name": name}
    if ref_area is not None:
        given["ref_area"] = ref_area
    if moment_center is not None:
        given["moment_center"] = moment_center
    return Forces(**given)


@contextmanager
def _collect() -> Iterator[list[Case]]:
    _collectors.append([])
    try:
        yield _collectors[-1]
    finally:
        _collectors.pop()


def load(path: str | Path) -> Case:
    """Load a case from a ``.py``, ``.toml`` or ``.json`` file, raising coded diagnostics."""
    path = Path(path)
    where = Location(file=str(path))
    if not path.is_file():
        raise CaseError([diagnostic("E0001", f"{path}: no such file", location=where)])
    if path.suffix in (".toml", ".json"):
        try:
            text = path.read_text()
            document = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
        except (tomllib.TOMLDecodeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CaseError([diagnostic("E0001", f"{path}: {error}", location=where)]) from None
        if not isinstance(document, dict):
            raise CaseError([diagnostic("E0003", f"{path}: not a table", location=where)])
        return Case.from_spec(CaseSpec.from_document(document, where), file=str(path))
    if path.suffix != ".py":
        raise CaseError(
            [diagnostic("E0001", f"{path}: case files are .py, .toml or .json", location=where)]
        )
    with _collect() as cases:
        try:
            runpy.run_path(str(path), run_name="__autan_case__")
        except CaseError:
            raise
        except SyntaxError as error:
            raise CaseError(
                [
                    diagnostic(
                        "E0001",
                        f"syntax error: {error.msg}",
                        location=Location(file=str(path), line=error.lineno),
                    )
                ]
            ) from None
        except Exception as error:  # noqa: BLE001 -- user code may raise anything
            frames = [
                f
                for f in traceback.extract_tb(error.__traceback__)
                if Path(f.filename) == path.resolve() or f.filename == str(path)
            ]
            line = frames[-1].lineno if frames else None
            raise CaseError(
                [
                    diagnostic(
                        "E0001",
                        f"{type(error).__name__}: {error}",
                        location=Location(file=str(path), line=line),
                    )
                ]
            ) from None
    if len(cases) != 1:
        raise CaseError(
            [
                diagnostic(
                    "E0002",
                    f"{path} creates {len(cases)} autan.Case objects, expected 1",
                    location=where,
                )
            ]
        )
    return cases[0]
