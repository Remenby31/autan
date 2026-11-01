"""GPU-native unstructured finite-volume CFD solver: Python SDK.

Build a case with ``autan.Case``, plan it against its mesh with ``case.plan``;
the case document is specified in docs/contracts/case-schema.md.
"""

from autan import gases, materials, units
from autan.case import Case, forces, load
from autan.errors import CaseError, Diagnostic, Location
from autan.fake import FakeBackend
from autan.meshinfo import MeshInfo
from autan.plan import Plan
from autan.results import Devices, Results, RunInfo
from autan.schema import (
    DDES,
    SA,
    SST,
    WMLES,
    CaseSpec,
    Compressible,
    ConstantDensityFluid,
    ConstantViscosity,
    IdealGas,
    Incompressible,
    Inviscid,
    Laminar,
    Manufactured,
    Solid,
    State,
    Steady,
    Sutherland,
    Unsteady,
)
from autan.solver_backend import SolverBackend
from autan.version import __version__

__all__ = [
    "DDES",
    "SA",
    "SST",
    "WMLES",
    "Case",
    "CaseError",
    "CaseSpec",
    "Compressible",
    "ConstantDensityFluid",
    "ConstantViscosity",
    "Devices",
    "Diagnostic",
    "FakeBackend",
    "IdealGas",
    "Incompressible",
    "Inviscid",
    "Laminar",
    "Location",
    "Manufactured",
    "MeshInfo",
    "Plan",
    "Results",
    "RunInfo",
    "Solid",
    "SolverBackend",
    "State",
    "Steady",
    "Sutherland",
    "Unsteady",
    "__version__",
    "forces",
    "gases",
    "load",
    "materials",
    "units",
]
