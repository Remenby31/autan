"""`autan-vv verify` recomputes a steady run's stop verdict from its history (case-schema.md#time).

The histories are written so that their verdict follows from that text by hand: residuals of 1
at iteration 1 and exact powers of ten after it, floors of 1e-8, plateaus of constant values.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import autan
import numpy as np
import pytest
from autan.errors import diagnostic
from autan.results import Convergence, StartInfo, field_names
from autan.units import K, Pa, kg, mol, s
from autan_vv.errors import VVError
from autan_vv.harness import UNIVERSAL_GAS_CONSTANT, verify
from autan_vv.meshes.families import cartesian
from autan_vv.meshes.fluent import write_fluent
from autan_vv.oracles.registry import get_oracle
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
    # it has to drop 8 orders from the largest first momentum residual, 1: 1e-6 is above 1e-8.
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
    # Once at 1e-9 it has: the run converges at iteration 3, and saying so at 1e-7 is refused.
    fell = lines(first, {"momentum-z": 1e-6}, {"momentum-z": 1e-9})
    verify([freestream_run(residuals=fell, diagnostics=(w0710,))], "freestream")
    early = lines(first, {"momentum-z": 1e-6}, {"momentum-z": 1e-7})
    run = freestream_run(residuals=early, diagnostics=(w0710,))
    assert refused(run) == {"iteration": 3, "equation": "momentum-z"}


def test_an_fp32_run_may_converge_on_a_round_off_plateau_within_ten_floors(
    freestream_run: Callable[..., Path],
) -> None:
    floors = dict.fromkeys(EQUATIONS, 1e-8)

    def run(stop: int, value: float, plateau: list[str], provisional: int = 0) -> Path:
        # 1.0, then the same value: the level (median of the last 100 counted residuals) is that
        # value from the 100th counted iteration, its first and so its reference; on a plateau
        # 100 counted iterations later.
        return freestream_run(
            precision="fp32+acc64",
            residuals=lines(1.0, *[value] * (stop - 1)),
            convergence=Convergence(floor=floors, plateau=plateau),
            provisional=provisional,
        )

    verify([run(200, 5e-8, EQUATIONS)], "freestream")
    # 49 provisional lines: counting starts at iteration 50, the first level is at 149 and the
    # plateau holds at 249, not before; unmarked, at 200.
    verify([run(249, 5e-8, EQUATIONS, provisional=49)], "freestream")
    assert refused(run(248, 5e-8, EQUATIONS, provisional=49)) == {
        "iteration": 248,
        "equation": "density",
    }
    assert refused(run(249, 5e-8, EQUATIONS)) == {"iteration": 200, "equation": None}
    assert refused(run(200, 5e-8, [])) == {"iteration": 200, "equation": None}
    # Above ten floors, or before 100 counted iterations without a new reference of the level.
    assert refused(run(200, 2e-7, EQUATIONS)) == {"iteration": 200, "equation": "density"}
    assert refused(run(199, 5e-8, EQUATIONS)) == {"iteration": 199, "equation": "density"}
    # Density sits in its band, energy above it: energy is the one that fails.
    mixed = freestream_run(
        precision="fp32+acc64",
        residuals=lines(1.0, *[{"density": 5e-8, "energy": 2e-7}] * 199),
        convergence=Convergence(floor=floors, plateau=["density"]),
    )
    assert refused(mixed) == {"iteration": 200, "equation": "energy"}


def test_a_plateau_follows_the_residual_level_not_its_lowest_draw(
    freestream_run: Callable[..., Path],
) -> None:
    # The sequences of the runtime's test (test_runtime.cpp): stationary noise uniform on 0.65 to
    # 1.35 times p (coefficient of variation 0.2) along the golden-ratio sequence, with one draw
    # at 0.85 times its lowest value at iteration 150, whose level (median of the last 100) is the
    # reference from iteration 100, so the run converges at 200; then the same noise falling 15 %
    # every 100 iterations, which never reaches a plateau.
    floors = dict.fromkeys(EQUATIONS, 1e-8)
    p = 5e-8

    def noise(k: int) -> float:
        return p * (1.0 + 0.35 * (2.0 * math.fmod(k * 0.6180339887498949, 1.0) - 1.0))

    series = [noise(k) for k in range(1, 201)]
    series[149] = 0.85 * min(series[:149])

    def stationary(stop: int) -> Path:
        return freestream_run(
            precision="fp32+acc64",
            residuals=lines(*series[:stop]),
            convergence=Convergence(floor=floors, plateau=EQUATIONS),
        )

    verify([stationary(200)], "freestream")
    # The value rule stopped this sequence at 105; the level rule refuses it there.
    assert refused(stationary(105)) == {"iteration": 105, "equation": "density"}
    falling = [noise(k) * 0.85 ** (k / 100.0) for k in range(1, 1001)]
    verify(
        [
            freestream_run(
                precision="fp32+acc64",
                max_iterations=1000,
                status="completed",
                residuals=lines(*falling),
                convergence=Convergence(floor=floors, plateau=[]),
            )
        ],
        "freestream",
    )


def test_degraded_mesh_convergence_judges_each_equations_drop_whatever_the_status(
    tmp_path: Path,
) -> None:
    oracle = get_oracle("mms-euler")
    gas = autan.IdealGas(
        name="manufactured",
        molar_mass=UNIVERSAL_GAS_CONSTANT / oracle.gas_constant * kg / mol,
        gamma=oracle.gamma,
        prandtl=0.72,
        viscosity=autan.ConstantViscosity(mu=1e-5 * Pa * s),
    )
    meshes = []
    for n in (3, 4, 5):
        mesh, path = cartesian((n, n, n)), tmp_path / f"box-{n}.msh"
        write_fluent(mesh, path, "fluid")
        meshes.append((mesh, path))

    def run(level: int, history: Lines, precision: str = "fp64", **options: Any) -> Path:
        mesh, path = meshes[level]
        case = autan.Case("degraded", mesh=path)
        case.physics = autan.Compressible(
            turbulence=autan.Inviscid(),
            precision=precision,  # type: ignore[arg-type]
            gas=gas,
        )
        case.freestream(mach=0.3, pressure=1e5 * Pa, temperature=300 * K)
        case.time = autan.Steady(max_iterations=3, residual_orders=8.0)
        case.boundary("*").farfield()
        exact = oracle.fields(mesh.geometry.cell_centroid, None)
        fields = {k: exact[k] for k in field_names(case.spec)}
        info = mesh_info(mesh, path, "fluent")
        return write_run(
            tmp_path / "runs", case, mesh, info, fields, residuals=history, **options
        ).path

    gate = "degraded-mesh-convergence"
    # Converged: 8.5 orders on every equation but momentum-y, identically zero.
    converged = run(
        0, [{e: 0.0 if e == "momentum-y" else v for e in EQUATIONS} for v in (1.0, 10.0**-8.5)]
    )
    # Completed at max_iterations with density 7 orders down; diverged, residuals rising.
    completed = run(1, lines(1.0, 1e-5, {"density": 1e-7}), status="completed")
    diverged = run(
        2,
        lines(1e-200, 1e200),
        status="diverged",
        diagnostics=(diagnostic("D0107", "divergence", iteration=2),),
    )
    report = verify([converged, completed, diverged], "mms-euler", gate)
    checks = {c["status"]: c for c in report["verdict"]["checks"]}
    assert checks["converged"]["equations"]["density"]["drop"] == pytest.approx(8.5)
    assert checks["converged"]["equations"]["momentum-y"] == {
        "drop": None,
        "identically_zero": True,
        "pass": True,
    }
    assert checks["converged"]["pass"]
    assert checks["completed"]["equations"]["density"]["drop"] == pytest.approx(7.0)
    assert checks["completed"]["equations"]["energy"]["drop"] == pytest.approx(9.0)
    assert not checks["completed"]["pass"]
    assert checks["diverged"]["equations"]["density"]["drop"] == pytest.approx(-400.0)
    assert not checks["diverged"]["pass"]
    # Nine orders, but on a fourth line past max_iterations = 3.
    late = run(
        1,
        lines(1.0, 1e-3, 1e-7, 1e-9),
        status="diverged",
        diagnostics=(diagnostic("D0107", "divergence", iteration=4),),
    )
    (check,) = verify([late], "mms-euler", gate)["verdict"]["checks"]
    assert check["equations"]["density"]["drop"] == pytest.approx(9.0)
    assert not check["pass"]
    assert verify([converged, completed], "mms-euler", gate)["verdict"]["pass"] is False
    assert verify([converged], "mms-euler", gate)["verdict"]["pass"] is True
    # Outside fp64 the drops are reported, not judged.
    fp32 = run(0, lines(1.0, 1e-3, 1e-4), "fp32+acc64", status="completed")
    verdict = verify([fp32], "mms-euler", gate)["verdict"]
    assert verdict["pass"] is None
    assert verdict["checks"][0]["equations"]["density"]["drop"] == pytest.approx(4.0)
    # A run of another physics is still refused.
    with pytest.raises(VVError) as err:
        verify([converged], "mms-navier-stokes", gate)
    assert err.value.code == "V0724"
