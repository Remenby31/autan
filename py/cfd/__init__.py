"""GPU-native unstructured finite-volume CFD solver: Python SDK.

Build a case with ``cfd.Case``, plan it against its mesh with ``case.plan``;
the case document is specified in docs/contracts/case-schema.md.
"""

from cfd import gases, materials, units
from cfd.case import Case, forces, load
from cfd.errors import CaseError, Diagnostic, Location
from cfd.fake import FakeBackend
from cfd.meshinfo import MeshInfo
from cfd.plan import Plan
from cfd.results import Results
from cfd.schema import (
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
    Solid,
    Steady,
    Sutherland,
    Unsteady,
)
from cfd.version import __version__

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
    "Diagnostic",
    "FakeBackend",
    "IdealGas",
    "Incompressible",
    "Inviscid",
    "Laminar",
    "Location",
    "MeshInfo",
    "Plan",
    "Results",
    "Solid",
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
