"""Machine-readable form of the gates of docs/validation/acceptance.md that runs can be checked against.

Each gate names its oracle, what is measured and the accepted interval. acceptance.md states every
gate with its source and its identifier; `test_harness.py` checks the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

CONSERVED = ("rho", "rhou", "rhov", "rhow", "rhoE")


@dataclass(frozen=True)
class Gate:
    """`measure` is `order` (fitted L-norm order and finest-pair order of each field must lie in
    [lower, upper]) or `max-error` (relative L-infinity error of each field at most `upper`
    machine epsilons of the run precision, on every run)."""

    id: str
    oracle: str
    measure: str
    norm: str
    fields: tuple[str, ...]
    lower: float
    upper: float
    milestone: str


GATES: dict[str, Gate] = {
    g.id: g
    for g in (
        Gate(
            "freestream-preservation",
            "freestream",
            "max-error",
            "Linf",
            ("Density", "VelocityX", "VelocityY", "VelocityZ", "Pressure"),
            0.0,
            64.0,
            "M2",
        ),
        Gate("euler-mms-order", "mms-euler", "order", "L2", CONSERVED, 1.9, 2.1, "M2"),
        Gate("ringleb-order", "ringleb", "order", "L2", ("Density", "Entropy"), 1.9, 2.1, "M2"),
        Gate("vortex-order", "isentropic-vortex", "order", "L2", CONSERVED, 1.9, 2.1, "M2"),
        Gate("sod-order", "sod", "order", "L1", ("Density",), 0.6, 1.1, "M2"),
        Gate("ns-mms-order", "mms-navier-stokes", "order", "L2", CONSERVED, 1.9, 2.1, "M3"),
        Gate(
            "rans-sa-mms-order",
            "mms-rans-sa",
            "order",
            "L2",
            (*CONSERVED, "TurbulentSANuTilde"),
            1.9,
            2.1,
            "M3",
        ),
        Gate(
            "couette-velocity-exact",
            "couette",
            "max-error",
            "Linf",
            ("VelocityX",),
            0.0,
            64.0,
            "M3",
        ),
        Gate(
            "couette-temperature-order", "couette", "order", "L2", ("Temperature",), 1.9, 2.1, "M3"
        ),
    )
}
