"""`autan-vv verify` recomputes a steady run's stop verdict from its history (case-schema.md#time).

The histories are written so that their verdict follows from that text by hand: residuals of 1
at iteration 1 and exact powers of ten after it, floors of 1e-8, plateaus of constant values.
"""

from collections.abc import Callable
from pathlib import Path

import autan
import numpy as np
import pytest
from autan.errors import diagnostic
from autan.results import Convergence, StartInfo
from autan.units import K, Pa
from autan_vv.errors import VVError
from autan_vv.harness import verify
from autan_vv.meshes.families import cartesian
from autan_vv.meshes.fluent import write_fluent
from autan_vv.runs import mesh_info, write_run

Lines = list[dict[str, float]]


@pytest.fixture
def freestream_run(tmp_path: Path) -> Callable[..., Path]:
    """Writes a run of a steady Euler case on a 3^3 hex mesh holding the freestream state."""
    mesh, path = cartesian((3, 3, 3)), tmp_path / "box.msh"
    write_fluent(mesh, path, "fluid")

    def write(precision: str = "fp64", max_iterations: int = 10000, **run: object) -> Path:
        case = autan.Case("stop", mesh=path)
        case.physics = autan.Compressible(turbulence=autan.Inviscid(), precision=precision)  # type: ignore[arg-type]
        case.freestream(mach=0.3, pressure=1e5 * Pa, temperature=300 * K)
        case.time = autan.Steady(max_iterations=max_iterations)
        case.boundary("*").farfield()
        speed = 0.3 * np.sqrt(1.4 * 287.0 * 300.0)
        state = {
            "Density": 1e5 / (287.0 * 300.0),
            "VelocityX": speed,
            "VelocityY": 0.0,
            "VelocityZ": 0.0,
            "Pressure": 1e5,
            "Temperature": 300.0,
        }
        fields = {k: np.full(mesh.n_cells, v) for k, v in state.items()}
        info = mesh_info(mesh, path, "fluent")
        return write_run(tmp_path / "runs", case, mesh, info, fields, **run).path  # type: ignore[arg-type]

    return write


# results.md#history, compressible.
EQUATIONS = ["density", "momentum-x", "momentum-y", "momentum-z", "energy"]


def lines(*values: float | dict[str, float]) -> Lines:
    """One history line per value: every equation at that value, or the equations a dict names
    at theirs and the others at 1e-9."""
    return [
        {e: v.get(e, 1e-9) for e in EQUATIONS}
        if isinstance(v, dict)
        else dict.fromkeys(EQUATIONS, v)
        for v in values
    ]


def refused(run: Path) -> dict[str, object]:
    with pytest.raises(VVError) as err:
        verify([run], "freestream")
    assert err.value.code == "V0730"
    return err.value.data


def test_a_converged_run_stops_where_every_residual_has_dropped(
    freestream_run: Callable[..., Path],
) -> None:
    verify([freestream_run(residuals=lines(1.0, 1e-4, 1e-9))], "freestream")
    # momentum-y dropped seven orders only.
    run = freestream_run(residuals=lines(1.0, {"momentum-y": 1e-7}))
    assert refused(run) == {"iteration": 2, "equation": "momentum-y"}
    # Iteration 2 met the rule: the run had to stop there.
    assert refused(freestream_run(residuals=lines(1.0, 1e-9, 1e-10))) == {
        "iteration": 2,
        "equation": None,
    }


def test_a_completed_run_never_meets_the_rule_and_ends_at_its_budget(
    freestream_run: Callable[..., Path],
) -> None:
    verify(
        [freestream_run(max_iterations=3, status="completed", residuals=lines(1.0, 0.1, 0.01))],
        "freestream",
    )
    run = freestream_run(max_iterations=3, status="completed", residuals=lines(1.0, 0.1, 1e-9))
    assert refused(run) == {"iteration": 3, "equation": None}
    run = freestream_run(max_iterations=3, status="completed", residuals=lines(1.0, 0.1))
    assert refused(run) == {"iteration": 2, "equation": None}


def test_a_continued_run_drops_from_the_original_reference(
    freestream_run: Callable[..., Path],
) -> None:
    def start(woken: list[str]) -> StartInfo:
        return StartInfo(
            run_id="run-20261106T000000Z-00000000",
            identity="0" * 64,
            mode="continue",
            iterations=5,
            time=None,
            solution_sha256="0" * 64,
            file="solution",
            residual_reference=dict.fromkeys(EQUATIONS, 1.0),
            woken=woken,
        )

    # Five orders below its own first line, nine below the original run's iteration 1.
    verify([freestream_run(start=start([]), residuals=lines(1e-4, 1e-9))], "freestream")
    run = freestream_run(start=start([]), residuals=lines(1e-4, {"energy": 1e-7}))
    assert refused(run) == {"iteration": 7, "equation": "energy"}
    # max_iterations counts the whole chain: a continuation stops at 7, unconverged.
    run = freestream_run(max_iterations=7, start=start([]), residuals=lines(1e-4, 1e-6, 1e-9))
    assert refused(run) == {"iteration": 8, "equation": None}


def test_an_identically_zero_equation_is_exempt_until_it_leaves_its_floor(
    freestream_run: Callable[..., Path],
) -> None:
    zero = {"momentum-z": 0.0}
    verify(
        [freestream_run(residuals=lines({**dict.fromkeys(EQUATIONS, 1.0), **zero}, zero))],
        "freestream",
    )
    # At iteration 2 momentum-z rises above 1e-12 of the largest residual: W0710, and from then on
    # it has to drop from its reference of 0 like the others, which it never does.
    first = {**dict.fromkeys(EQUATIONS, 1.0), **zero}
    woke = lines(first, {"momentum-z": 1e-6})
    run = freestream_run(max_iterations=2, status="completed", residuals=woke)
    assert refused(run) == {"iteration": 2, "equation": "momentum-z"}
    w0710 = diagnostic("W0710", "momentum-z left its floor", equation="momentum-z", iteration=2)
    verify(
        [
            freestream_run(
                max_iterations=2, status="completed", residuals=woke, diagnostics=(w0710,)
            )
        ],
        "freestream",
    )


def test_an_fp32_run_may_converge_on_a_round_off_plateau_within_ten_floors(
    freestream_run: Callable[..., Path],
) -> None:
    floors = dict.fromkeys(EQUATIONS, 1e-8)

    def run(stop: int, value: float, plateau: list[str]) -> Path:
        # A new reference at iteration 2, then the same value: on a plateau from iteration 102.
        return freestream_run(
            precision="fp32+acc64",
            residuals=lines(1.0, *[value] * (stop - 1)),
            convergence=Convergence(floor=floors, plateau=plateau),
        )

    verify([run(102, 5e-8, EQUATIONS)], "freestream")
    # Later: a first-order start of 49 iterations puts it there.
    verify([run(150, 5e-8, EQUATIONS)], "freestream")
    assert refused(run(102, 5e-8, [])) == {"iteration": 102, "equation": None}
    # Above ten floors, or before 100 iterations without a new reference.
    assert refused(run(102, 2e-7, EQUATIONS)) == {"iteration": 102, "equation": "density"}
    assert refused(run(60, 5e-8, EQUATIONS)) == {"iteration": 60, "equation": "density"}
    # Density sits in its band, energy above it: energy is the one that fails.
    mixed = freestream_run(
        precision="fp32+acc64",
        residuals=lines(1.0, *[{"density": 5e-8, "energy": 2e-7}] * 101),
        convergence=Convergence(floor=floors, plateau=["density"]),
    )
    assert refused(mixed) == {"iteration": 102, "equation": "energy"}
