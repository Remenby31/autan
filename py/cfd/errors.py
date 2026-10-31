"""Diagnostic codes: one registry, rendered for humans, JSON and the reference page.

A code is ``E####`` (error: blocks the run), ``W####`` (warning) or ``D####``
(divergence detected during a run). ``docs/sdk/errors.md`` is generated from
``REGISTRY`` by ``cfd explain --all``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

DOC_URL = "https://github.com/Remenby31/cfd-solver/blob/main/docs/sdk/errors.md"

Severity = Literal["error", "warning", "divergence"]
_SEVERITY: dict[str, Severity] = {"E": "error", "W": "warning", "D": "divergence"}


@dataclass(frozen=True)
class Code:
    code: str
    title: str
    explanation: str
    fix: str

    @property
    def severity(self) -> Severity:
        return _SEVERITY[self.code[0]]


_CODES = [
    Code(
        "E0001",
        "The case file cannot be read",
        "The file does not exist, is not valid Python or TOML, or raised an exception while "
        "running. Python case files run as scripts; the traceback points at the failing line.",
        "Fix the file at the reported location.",
    ),
    Code(
        "E0002",
        "The case file does not define exactly one case",
        "A Python case file must create exactly one `cfd.Case`, at module level.",
        'Create one case: `case = cfd.Case("name", mesh="mesh.cgns")`.',
    ),
    Code(
        "E0003",
        "The case document does not match the schema",
        "A field is missing, has the wrong type, or is not part of `cfd.case/1`. The location "
        "gives the path of the field in the document. See docs/contracts/case-schema.md.",
        "Correct the field named in the message.",
    ),
    Code(
        "E0004",
        "Unsupported case schema version",
        "The document's `schema` key is not `cfd.case/1`.",
        'Set `schema = "cfd.case/1"`.',
    ),
    Code(
        "E0201",
        "No mesh information",
        "Planning a case needs the MeshInfo of its mesh (docs/contracts/mesh-info.md). The "
        "C++ mesh readers are not part of this build, so it must be passed explicitly.",
        "Pass the MeshInfo JSON: `cfd plan case.py --mesh-info mesh.meshinfo.json`.",
    ),
    Code(
        "E0202",
        "The MeshInfo document is invalid",
        "The JSON does not match `cfd.mesh-info/1`, or its counts are inconsistent (zone "
        "counts that do not sum to the cell count, duplicate names, unknown interface zones).",
        "Regenerate the MeshInfo with the mesh reader of the same solver version.",
    ),
    Code(
        "E0203",
        "The MeshInfo describes another mesh file",
        "The file name in the MeshInfo `source.path` differs from the case's `mesh`.",
        "Pass the MeshInfo of the case's mesh, or change `mesh` in the case.",
    ),
    Code(
        "W0204",
        "Highly non-orthogonal cells",
        "The maximum face non-orthogonality exceeds 70°, the threshold above which "
        "OpenFOAM's checkMesh reports severely non-orthogonal faces. Gradients and viscous "
        "fluxes lose accuracy and implicit solves may stall.",
        "Improve the mesh near the worst cells, or lower the CFL number.",
    ),
    Code(
        "E0301",
        "Quantity of the wrong dimension",
        "The value has a dimension other than the one the field expects, for example a speed "
        "given for the Mach number.",
        "Pass a value of the expected dimension, with a unit from `cfd.units`.",
    ),
    Code(
        "E0302",
        "Missing unit",
        "A dimensional field received a bare number. Angles also need a unit, because "
        "degrees and radians cannot be told apart.",
        "Multiply by a unit: `19.57 * deg`, `7.0 * m`.",
    ),
    Code(
        "E0303",
        "Unknown unit",
        "The text could not be parsed as `<number> <unit>` with Pint's default registry.",
        'Write the quantity as `"<number> <unit>"`, for example `"101325 Pa"`.',
    ),
    Code(
        "E0304",
        "Value out of range",
        "The value must be finite, and strictly positive where the schema says `> 0` "
        "(temperatures are compared in kelvin).",
        "Use a physically valid value.",
    ),
    Code(
        "E0305",
        "Turbulence model needs an unsteady run",
        "DDES and WMLES resolve turbulent structures in time and cannot run steady.",
        "Set `case.time = cfd.Unsteady(steps=..., dt=...)`.",
    ),
    Code(
        "E0306",
        "Freestream does not match the physics",
        "Compressible flow needs `mach` and exactly one of `pressure` or `re` (with "
        "`ref_length`); incompressible flow needs `velocity` and takes no `mach`, `re`, "
        "`pressure` or `temperature`.",
        "Set the freestream parameters required by the physics.",
    ),
    Code(
        "E0410",
        "Boundary rule matches no zone",
        "The pattern of `case.boundary(...)` matches no boundary zone of the mesh.",
        "Use one of the zone names listed by `cfd plan`, or a glob that matches some.",
    ),
    Code(
        "E0411",
        "Boundary zone matched by several rules",
        "Two `case.boundary(...)` patterns match the same zone; neither takes precedence.",
        "Narrow the patterns so each zone matches one rule.",
    ),
    Code(
        "E0412",
        "Boundary zone without a condition",
        "The zone is not typed in the mesh file and no rule of the case matches it.",
        'Give it a condition: `case.boundary("name").wall()` (or `slip()`, `symmetry()`, '
        "`farfield()`, `inlet(...)`, `outlet(...)`).",
    ),
    Code(
        "E0413",
        "Mesh boundary type needs values",
        "The mesh types the zone as a thermal wall, an inlet or an outlet, which needs values "
        "only the case can provide.",
        'Set them: `case.boundary("name").outlet(static_pressure=101325 * Pa)`.',
    ),
    Code(
        "E0414",
        "Boundary condition not available for this physics",
        "`inlet` (total conditions) is compressible only, `velocity-inlet` incompressible "
        "only, and a no-slip `wall` needs viscous flow.",
        "Use the condition matching the physics: `slip()` for inviscid walls, `inlet(...)` or "
        "`velocity_inlet(...)`.",
    ),
    Code(
        "W0415",
        "Case overrides the mesh boundary type",
        "The case applies a condition of another kind than the type stored in the mesh file. "
        "This is allowed; the warning makes the override visible.",
        "Check that the override is intended.",
    ),
    Code(
        "E0501",
        "Cell zone rule matches no zone",
        "The pattern of `case.cell_zone(...)` matches no cell zone of the mesh.",
        "Use one of the cell zone names listed by `cfd plan`.",
    ),
    Code(
        "E0502",
        "Cell zone matched by several rules",
        "Two `case.cell_zone(...)` patterns match the same zone.",
        "Narrow the patterns so each zone matches one rule.",
    ),
    Code(
        "E0503",
        "Solid zone without a material",
        "The mesh declares the zone solid and the case gives it no material.",
        'Set it: `case.cell_zone("name").solid(cfd.materials.COPPER)`.',
    ),
    Code(
        "E0504",
        "Material does not match the physics",
        "Compressible fluid zones need an ideal gas; incompressible fluid zones need a "
        "constant-density fluid.",
        "Use a material of the right kind for the physics.",
    ),
    Code(
        "E0505",
        "Fluids of different materials in contact",
        "Two fluid zones with different materials share an interface. Mixing them needs "
        "species transport, which `cfd.case/1` does not provide.",
        "Use the same material on both sides, or separate them by a solid zone.",
    ),
    Code(
        "E0506",
        "Solid zone with incompressible physics",
        "Conjugate heat transfer needs an energy equation in the fluid, which incompressible "
        "flow does not solve in `cfd.case/1`.",
        "Use `cfd.Compressible(...)`, or remove the solid zone from the mesh.",
    ),
    Code(
        "E0601",
        "Monitor surfaces match no boundary zone",
        "The `surfaces` pattern of a monitor matches no boundary zone of the mesh.",
        "Use a pattern that matches the zones to integrate over.",
    ),
    Code(
        "E0602",
        "Duplicate monitor name",
        "Two monitors share a name; results are keyed by monitor name.",
        'Name them: `cfd.forces("wing.*", name="wing")`.',
    ),
    Code(
        "E0701",
        "The run directory does not conform to cfd.results/1",
        "A file is missing or unreadable, `run.json` has another schema, the stored case or "
        "MeshInfo does not match the hashes of `run.json`, or the history is out of order. "
        "See docs/contracts/results.md.",
        "Open a directory written by a backend of this solver version.",
    ),
    Code(
        "E0702",
        "No such monitor",
        "The history has no monitor of that name, or `results.forces` was used on a case "
        "with zero or several forces monitors.",
        "Use `results.history.monitor(name)` with a name from `results.history.monitors`.",
    ),
    Code(
        "E0703",
        "No such solution data",
        "The run has no `solution.cgns` (it failed or diverged before writing one), or the "
        "requested field is not in it.",
        "Check `results.run.status`, and use a field name listed in the message.",
    ),
    Code(
        "D0107",
        "Divergence",
        "The solution diverged: a residual or a state became non-finite or non-physical. "
        "`data` gives the iteration, the cell (0-based) and its cell zone.",
        "Lower the CFL number or use a more dissipative limiter; check the mesh quality "
        "near the cell.",
    ),
    Code(
        "W0801",
        "No NVIDIA GPU found",
        "`nvidia-smi` is missing or lists no GPU. CUDA runs are not possible on this machine.",
        "Install the NVIDIA driver, or run on a GPU machine.",
    ),
    Code(
        "W0802",
        "No MPI launcher found",
        "Neither `mpirun` nor `mpiexec` is on the PATH. Multi-GPU runs need MPI.",
        "Install a GPU-aware MPI (Open MPI or MPICH built with CUDA support).",
    ),
    Code(
        "W0803",
        "Solver core not built",
        "This installation contains the SDK only. Cases can be planned and run with the "
        "`fake` backend, which writes synthetic results.",
        "Build the C++ core (see README) to run real computations.",
    ),
    Code(
        "E0804",
        "No solver backend",
        "`case.run()` needs a backend. The C++ core is not part of this build, so only the "
        "`fake` backend exists; it writes synthetic results.",
        "Pass one explicitly: `case.run(mesh_info, backend=cfd.FakeBackend())`.",
    ),
]

REGISTRY: dict[str, Code] = {code.code: code for code in _CODES}


def doc_url(code: str) -> str:
    return f"{DOC_URL}#{code.lower()}"


class Location(BaseModel):
    """Where a diagnostic points: a file, a line in it, and/or a path in the case document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str | None = None
    line: int | None = None
    path: str | None = None

    def __str__(self) -> str:
        where = self.file or "<case>"
        if self.line is not None:
            where += f":{self.line}"
        if self.path:
            where += f" ({self.path})"
        return where


class Diagnostic(BaseModel):
    """One problem, with its registered code, location, suggested fix and documentation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Severity
    message: str
    location: Location | None = None
    fix: str
    doc: str
    data: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        lines = [f"{self.severity}[{self.code}]: {self.message}"]
        if self.location is not None:
            lines.append(f"  --> {self.location}")
        lines.append(f"   = fix : {self.fix}")
        lines.append(f"   = doc : {self.doc}")
        return "\n".join(lines)


def diagnostic(
    code: str,
    message: str,
    *,
    location: Location | None = None,
    fix: str | None = None,
    **data: Any,
) -> Diagnostic:
    """Build a diagnostic for a registered code; ``fix`` overrides the generic fix."""
    entry = REGISTRY[code]
    return Diagnostic(
        code=code,
        severity=entry.severity,
        message=message,
        location=location,
        fix=fix or entry.fix,
        doc=doc_url(code),
        data=data,
    )


class CaseError(Exception):
    """Raised when a case cannot be built, loaded, planned or run; carries its diagnostics."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(d.render() for d in self.diagnostics))

    @property
    def codes(self) -> list[str]:
        return [d.code for d in self.diagnostics]


def reference_page() -> str:
    """The Markdown reference of every registered code."""
    lines = [
        "# Diagnostic codes",
        "",
        "Generated from `py/cfd/errors.py` by `cfd explain --all`; do not edit by hand.",
        "",
        (
            "`E` codes are errors and block the run, `W` codes are warnings, `D` codes report "
            "a divergence detected during a run. Codes `E0101` to `E0199` belong to the mesh "
            "readers ([mesh.md](../contracts/mesh.md))."
        ),
        "",
    ]
    for entry in sorted(REGISTRY.values(), key=lambda c: c.code[1:]):
        lines += [explain(entry.code), ""]
    return "\n".join(lines)


def explain(code: str) -> str:
    """The Markdown documentation of one registered code."""
    entry = REGISTRY[code]
    return "\n".join(
        [
            f"## {entry.code}",
            "",
            f"**{entry.title}** ({entry.severity})",
            "",
            entry.explanation,
            "",
            f"Fix: {entry.fix}",
        ]
    )
