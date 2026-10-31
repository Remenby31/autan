"""Exact and reference solutions that solver runs are verified against.

An oracle maps cell positions (and a physical time for unsteady flows) to the primitive fields of
docs/contracts/results.md, in SI: `Density`, `VelocityX`, `VelocityY`, `VelocityZ`, `Pressure`,
`Temperature` and, for Spalart-Allmaras, `TurbulentSANuTilde`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from autan_vv.mms.solutions import ManufacturedSolution

FloatArray = npt.NDArray[np.float64]
FieldFunction = Callable[[FloatArray, float | None], dict[str, FloatArray]]

PRIMITIVES = ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure")


@dataclass(frozen=True)
class Oracle:
    """An exact or reference solution and the conditions under which it holds.

    `dim` is the number of directions the solution varies in (2 for planar flows computed on one
    layer of cells), which sets the mesh size h = (A/N)^(1/2) or (V/N)^(1/3). `physics` names the
    equations it solves (`euler`, `navier-stokes`, `rans-sa`, `incompressible-navier-stokes`).
    `reference_only` marks a numerical reference, usable for regression but not as an exact oracle.
    """

    name: str
    description: str
    physics: str
    dim: int
    unsteady: bool
    gamma: float
    gas_constant: float
    evaluate: FieldFunction
    parameters: Mapping[str, float] = field(default_factory=dict)
    references: tuple[str, ...] = ()
    reference_only: bool = False

    def fields(self, points: FloatArray, time: float | None = None) -> dict[str, FloatArray]:
        """Primitive fields at `points` (n, 3), with `Temperature` from the ideal-gas law."""
        if self.unsteady and time is None:
            msg = f"oracle {self.name} is unsteady: a time is required"
            raise VVError(msg, code="V0701")
        out = self.evaluate(np.asarray(points, dtype=np.float64), time)
        if "Temperature" not in out and np.isfinite(self.gas_constant):
            out["Temperature"] = out["Pressure"] / (out["Density"] * self.gas_constant)
        return out


def conserved(fields: Mapping[str, FloatArray], gamma: float) -> dict[str, FloatArray]:
    """Conserved variables rho, rho u, rho v, rho w, rho E of an ideal gas from primitive fields."""
    rho = fields["Density"]
    u, v, w = fields["VelocityX"], fields["VelocityY"], fields["VelocityZ"]
    energy = fields["Pressure"] / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
    return {"rho": rho, "rhou": rho * u, "rhov": rho * v, "rhow": rho * w, "rhoE": energy}


def from_mms(ms: ManufacturedSolution) -> Oracle:
    """The manufactured solution's fields as an oracle (the solver adds its source term)."""
    names = {
        "rho": "Density",
        "u": "VelocityX",
        "v": "VelocityY",
        "w": "VelocityZ",
        "p": "Pressure",
        "nu_hat": "TurbulentSANuTilde",
    }
    evaluator = ms.evaluator("fields")
    physics = {
        "compressible Euler": "euler",
        "compressible laminar Navier-Stokes": "navier-stokes",
    }.get(ms.equations, "rans-sa")

    def evaluate(points: FloatArray, _time: float | None) -> dict[str, FloatArray]:
        return {names[k]: v for k, v in evaluator(points).items()}

    return Oracle(
        name=f"mms-{ms.name.replace('_', '-')}",
        description=ms.description,
        physics=physics,
        dim=3,
        unsteady=False,
        gamma=ms.parameters["gamma"],
        gas_constant=ms.parameters["gas_constant"],
        evaluate=evaluate,
        parameters=dict(ms.parameters),
        references=ms.references,
    )
