"""Machine-readable form of the gates of docs/validation/acceptance.md that runs can be checked against.

Each gate names its oracles, what is measured and the accepted interval. acceptance.md states every
gate with its source and its identifier; `test_harness.py` checks the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

CONSERVED = ("rho", "rhou", "rhov", "rhow", "rhoE")
# acceptance.md: orders are fitted over at least four levels, using only levels whose error is at
# least 100 times the rounding floor of the precision.
MIN_LEVELS = 4
FLOOR_FACTOR = 100.0


@dataclass(frozen=True)
class Gate:
    """A gate of acceptance.md.

    `measure` is `order` (over at least four levels above the rounding floor, the fitted and the
    finest-pair order of each field's `norm` error must lie in [lower, upper]; a field the oracle
    makes identically zero must instead stay zero to `PRESERVED` epsilons) or `max-error` (the
    relative `norm` error of each field at most `upper` epsilons of the run precision, on every
    run). `fields` None means every field the run writes. `time` and `domain` are prerequisites:
    every run must reach that physical time, and the mesh must span `domain` (per axis, min and max).
    `statuses` are the run statuses (results.md, run.json) the gate accepts: `converged` for a steady
    gate, `completed` for an unsteady one that runs a fixed number of steps; `floor` (a steady
    fp32+acc64 run stopped at its round-off floor, not converged) only where the gate measures
    round-off itself. A run of another status fails the gate's `status` prerequisite. With
    `upper_blocks` False, an order above `upper` is a warning, and above `investigate_above` the
    verdict asks for an explanation; neither fails the gate. With `monotone`, each field's error
    must decrease strictly from every level to the next finer one.
    """

    id: str
    oracles: tuple[str, ...]
    measure: str
    norm: str
    fields: tuple[str, ...] | None
    lower: float
    upper: float
    milestone: str
    time: float | None = None
    domain: tuple[tuple[float, float], ...] | None = None
    upper_blocks: bool = True
    investigate_above: float | None = None
    monotone: bool = False
    statuses: tuple[str, ...] = ("converged",)


PRESERVED = 64.0

GATES: dict[str, Gate] = {
    g.id: g
    for g in (
        Gate(
            "freestream-preservation",
            ("freestream",),
            "max-error",
            "Linf",
            None,
            0.0,
            64.0,
            "M2",
            statuses=("converged", "floor"),
        ),
        Gate(
            "euler-mms-order",
            ("mms-euler",),
            "order",
            "L2",
            CONSERVED,
            1.9,
            2.1,
            "M2",
            upper_blocks=False,
            investigate_above=2.6,
            monotone=True,
        ),
        Gate(
            "ringleb-order",
            ("ringleb",),
            "order",
            "L2",
            ("Density", "Pressure", "VelocityX", "VelocityY", "Entropy"),
            1.9,
            2.1,
            "M2",
            upper_blocks=False,
            investigate_above=2.6,
            monotone=True,
        ),
        Gate(
            "vortex-order",
            ("isentropic-vortex",),
            "order",
            "L2",
            CONSERVED,
            1.9,
            2.1,
            "M2",
            time=40.0,
            domain=((-10.0, 10.0), (-10.0, 10.0)),
            statuses=("completed",),
        ),
        Gate(
            "sod-order",
            ("sod",),
            "order",
            "L1",
            ("Density",),
            0.6,
            1.1,
            "M2",
            statuses=("completed",),
        ),
        Gate(
            "ns-mms-order",
            ("mms-navier-stokes",),
            "order",
            "L2",
            CONSERVED,
            1.9,
            2.1,
            "M3",
            upper_blocks=False,
            investigate_above=2.6,
            monotone=True,
        ),
        Gate(
            "rans-sa-mms-order",
            ("mms-rans-sa",),
            "order",
            "L2",
            (*CONSERVED, "TurbulentSANuTilde"),
            1.9,
            2.1,
            "M3",
            upper_blocks=False,
            investigate_above=2.6,
            monotone=True,
        ),
        Gate(
            "couette-velocity-exact",
            ("couette",),
            "max-error",
            "Linf",
            ("VelocityX", "VelocityY", "VelocityZ"),
            0.0,
            64.0,
            "M3",
            statuses=("converged", "floor"),
        ),
        Gate(
            "couette-temperature-order",
            ("couette",),
            "order",
            "L2",
            ("Temperature",),
            1.9,
            2.1,
            "M3",
        ),
    )
}
