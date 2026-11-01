"""SU2 configuration of an Autan case, under the fairness protocol of docs/validation/su2.md.

The physical setup (gas, freestream, boundaries, force reference) is read from the Autan case, so
both codes solve one problem. SU2 runs in two schemes, each with the solver controls SU2 publishes
for this case at the pinned tag: `roe`, the closest to Autan's (docs/design/euler.md): Roe, MUSCL
on primitive variables with weighted least-squares gradients, Venkatakrishnan's limiter with its
threshold matched to Autan's (`venkat_coefficient`), controls of
TestCases/euler/naca0012/inv_NACA0012_Roe.cfg; and `jst`, SU2's reference scheme for this case
(TestCases/euler/naca0012/inv_NACA0012.cfg and QuickStart/inv_NACA0012.cfg).
"""

from __future__ import annotations

import fnmatch
import math

import autan
import numpy as np
from autan.solver_input import freestream_state, gas_constant
from autan.units import si

from autan_vv.errors import VVError
from autan_vv.su2.mesh import Record, Slab, _bottom_and_top

# Autan's Venkatakrishnan constant (docs/design/euler.md, Residual): eps_k^2 = (K / L)^3 V q_k^2.
AUTAN_VENKAT_K = 10.0
# Region whose median cell sets SU2's uniform threshold: within one chord of the mid-chord point,
# where the shocks of the NACA 0012 conditions stand.
MATCH_CENTRE = (0.5, 0.0)
MATCH_RADIUS = 1.0
STOP_MARGIN = 2.0

# SU2's published settings for the inviscid NACA 0012 (SU2 v8.6.0, TestCases/euler/naca0012).
SCHEMES: dict[str, dict[str, str]] = {
    # inv_NACA0012_Roe.cfg; the limiter coefficient is set by `config`.
    "roe": {
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
        "CONV_NUM_METHOD_FLOW": "ROE",
        "ENTROPY_FIX_COEFF": "0.001",
        "MUSCL_FLOW": "YES",
        "MUSCL_KAPPA_FLOW": "0.0",
        "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN",
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": "30.0",
        "CFL_ADAPT": "NO",
        "LINEAR_SOLVER": "BCGSTAB",
        "LINEAR_SOLVER_ERROR": "1E-1",
        "LINEAR_SOLVER_ITER": "3",
        "MGLEVEL": "3",
        "MGCYCLE": "W_CYCLE",
        "MG_PRE_SMOOTH": "( 1, 2, 2, 2 )",
        "MG_POST_SMOOTH": "( 1, 1, 1, 1 )",
        "MG_CORRECTION_SMOOTH": "( 1, 1, 1, 1 )",
        "MG_DAMP_RESTRICTION": "0.80",
        "MG_DAMP_PROLONGATION": "0.80",
    },
    # inv_NACA0012.cfg (full multigrid start).
    "jst": {
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
        "CONV_NUM_METHOD_FLOW": "JST",
        "JST_SENSOR_COEFF": "( 0.5, 0.02 )",
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": "1000.0",
        "CFL_ADAPT": "NO",
        "LINEAR_SOLVER": "FGMRES",
        "LINEAR_SOLVER_PREC": "LU_SGS",
        "LINEAR_SOLVER_ERROR": "1E-3",
        "LINEAR_SOLVER_ITER": "5",
        "MGLEVEL": "3",
        "MGCYCLE": "FULLMG_CYCLE",
        "MG_STARTUP_STAGNATION": "0.99",
        "MG_STARTUP_ITER": "10",
        "MG_STARTUP_CONVERGENCE": "-2",
        "MG_PRE_SMOOTH": "( 1, 2, 3, 4 )",
        "MG_POST_SMOOTH": "( 0, 0, 0, 0 )",
        "MG_CORRECTION_SMOOTH": "( 0, 0, 0, 0 )",
        "MG_DAMP_RESTRICTION": "1.0",
        "MG_DAMP_PROLONGATION": "1.0",
    },
}


def venkat_coefficient(slab: Slab, ref_length: float = 1.0) -> float:
    """SU2's VENKAT_LIMITER_COEFF that gives Autan's threshold on the median cell of the match
    region.

    SU2 v8.6.0 (CLimiterDetails.hpp) uses one threshold eps^2 = (K_SU2 L_e)^3 with L_e = 1, on
    variables made dimensionless by the freestream (REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE:
    rho, p, T = 1 there). Autan's is eps_k^2 = (K V^(1/3) / L)^3 q_k^2 in each cell, q_k the
    freestream scale of variable k: in units of q_k the two agree where V^(1/3) = K_SU2 L / K.
    """
    bottom, _ = _bottom_and_top(slab)
    xz = slab.points[bottom][..., [0, 2]]
    x, z = xz[..., 0], xz[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(z, -1, axis=1) - np.roll(x, -1, axis=1) * z, axis=1))
    volume = area * (slab.y1 - slab.y0)
    centre = xz.mean(axis=1)
    near = np.hypot(centre[:, 0] - MATCH_CENTRE[0], centre[:, 1] - MATCH_CENTRE[1]) <= MATCH_RADIUS
    if not near.any():
        msg = "no cell lies in the region where the limiter threshold is matched"
        raise VVError(msg, code="V0903")
    return AUTAN_VENKAT_K * float(np.median(np.cbrt(volume[near]))) / ref_length


def _matching(pattern: str, zones: list[str]) -> list[str]:
    return [z for z in zones if fnmatch.fnmatchcase(z, pattern)]


def config(
    case: autan.Case,
    mesh: Record,
    venkat: float,
    *,
    span: float,
    max_iterations: int = 20000,
    residual_orders: float = 10.0,
    stop_margin: float = STOP_MARGIN,
    scheme: str = "roe",
) -> str:
    """The SU2 configuration of `case` on the converted `mesh`, in `scheme` (a key of SCHEMES;
    `venkat` is used by `roe` only).

    The case must be inviscid compressible with slip, far-field and symmetry boundaries and one
    forces monitor (V0903). In 2D, Autan's (x, z) is SU2's (x, y), the symmetry zones are not in the
    mesh and SU2's coefficients are per unit span, so its reference area is Autan's divided by the
    slab's `span`. SU2 measures a residual's drop from its largest value so far (REL_RMS_*), the
    protocol from its first: `run.sh` stops SU2 when the protocol's criterion is met, SU2's own
    rule is a backstop `stop_margin` orders beyond, and the criterion's iteration is read from the
    history.
    """
    if scheme not in SCHEMES:
        msg = f"unknown SU2 scheme {scheme!r}; expected one of {sorted(SCHEMES)}"
        raise VVError(msg, code="V0903")
    spec = case.spec
    physics = spec.physics
    if not isinstance(physics, autan.Compressible) or not isinstance(
        physics.turbulence, autan.Inviscid
    ):
        msg = f"case {case.name}: SU2 is configured for inviscid compressible cases only"
        raise VVError(msg, code="V0903")
    fs = freestream_state(spec)
    if fs.sideslip != 0.0 or fs.mach is None or fs.temperature is None:
        msg = f"case {case.name}: needs a Mach number, a temperature and no sideslip"
        raise VVError(msg, code="V0903")
    zones = [*mesh.markers, *mesh.dropped_zones]
    markers: dict[str, list[str]] = {"slip": [], "farfield": [], "symmetry": []}
    for rule in spec.boundaries:
        kind = rule.condition.kind
        if kind not in markers:
            msg = f"case {case.name}: boundary kind {kind!r} is not configured"
            raise VVError(msg, code="V0903")
        markers[kind] += _matching(rule.zones, zones)
    if mesh.dimension == 2 and sorted(markers["symmetry"]) != sorted(mesh.dropped_zones):
        msg = f"case {case.name}: the planes dropped in 2D must be its symmetry zones"
        raise VVError(msg, code="V0903")
    if len(spec.monitors) != 1 or spec.monitors[0].kind != "forces":
        msg = f"case {case.name}: expected one forces monitor"
        raise VVError(msg, code="V0903")
    forces = spec.monitors[0]
    if forces.ref_area is None or fs.ref_length is None:
        msg = f"case {case.name}: the forces monitor needs a reference area and length"
        raise VVError(msg, code="V0903")
    centre = [float(si(c, "m")) for c in forces.moment_center]
    ref_area = float(si(forces.ref_area, "m**2"))
    surfaces = _matching(forces.surfaces, list(mesh.markers))
    if mesh.dimension == 2:
        if centre[1] != 0.0:
            msg = f"case {case.name}: the moment centre must lie in the plane y = 0"
            raise VVError(msg, code="V0903")
        centre = [centre[0], centre[2], 0.0]
        ref_area /= span
    lines = {
        "SOLVER": "EULER",
        "MATH_PROBLEM": "DIRECT",
        "RESTART_SOL": "NO",
        "FLUID_MODEL": "IDEAL_GAS",
        "GAMMA_VALUE": repr(physics.gas.gamma),
        "GAS_CONSTANT": repr(gas_constant(physics.gas)),
        "MACH_NUMBER": repr(fs.mach),
        "AOA": repr(math.degrees(fs.aoa)),
        "SIDESLIP_ANGLE": "0.0",
        "FREESTREAM_PRESSURE": repr(fs.pressure),
        "FREESTREAM_TEMPERATURE": repr(fs.temperature),
        "REF_DIMENSIONALIZATION": "FREESTREAM_PRESS_EQ_ONE",
        "REF_ORIGIN_MOMENT_X": repr(centre[0]),
        "REF_ORIGIN_MOMENT_Y": repr(centre[1]),
        "REF_ORIGIN_MOMENT_Z": repr(centre[2]),
        "REF_LENGTH": repr(fs.ref_length),
        "REF_AREA": repr(ref_area),
        "MARKER_EULER": f"( {', '.join(markers['slip'])} )",
        "MARKER_FAR": f"( {', '.join(markers['farfield'])} )",
        **(
            {"MARKER_SYM": f"( {', '.join(markers['symmetry'])} )"}
            if mesh.dimension == 3 and markers["symmetry"]
            else {}
        ),
        "MARKER_MONITORING": f"( {', '.join(surfaces)} )",
        "MARKER_PLOTTING": f"( {', '.join(surfaces)} )",
        **SCHEMES[scheme],
        **({"VENKAT_LIMITER_COEFF": repr(venkat)} if scheme == "roe" else {}),
        "ITER": str(max_iterations),
        # The slab's transverse momentum (y) is zero in exact arithmetic and sits at round-off.
        "CONV_FIELD": "( "
        + ", ".join(
            f"REL_RMS_{eq}"
            for eq in equations(mesh.dimension)
            if eq != "MOMENTUM-Y" or mesh.dimension == 2
        )
        + " )",
        "CONV_RESIDUAL_MINVAL": repr(-residual_orders - stop_margin),
        "CONV_STARTITER": "10",
        "MESH_FILENAME": mesh.file,
        "MESH_FORMAT": "SU2",
        "OUTPUT_FILES": "( RESTART_ASCII, SURFACE_CSV )",
        "OUTPUT_WRT_FREQ": str(max_iterations),
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "history",
        "HISTORY_OUTPUT": "( ITER, RMS_RES, REL_RMS_RES, AERO_COEFF, WALL_TIME )",
        "SCREEN_OUTPUT": "( INNER_ITER, WALL_TIME, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG )",
        "SCREEN_WRT_FREQ_INNER": "100",
    }
    return "".join(f"{k}= {v}\n" for k, v in lines.items())


def equations(dimension: int) -> tuple[str, ...]:
    """SU2's names of the residuals of the Euler equations in `dimension`."""
    momentum = ("MOMENTUM-X", "MOMENTUM-Y", "MOMENTUM-Z")[:dimension]
    return ("DENSITY", *momentum, "ENERGY")


# What `adopt` changes in a published SU2 configuration: the mesh, the flight condition and the
# outputs and stop rule the protocol reads; nothing that sets the scheme or the solver.
ADOPTED_KEYS = ("MESH_FILENAME", "MESH_FORMAT", "MACH_NUMBER", "AOA", "FREESTREAM_PRESSURE",
                "FREESTREAM_TEMPERATURE", "ITER", "CONV_FIELD", "CONV_RESIDUAL_MINVAL",
                "HISTORY_OUTPUT", "TABULAR_FORMAT", "CONV_FILENAME", "OUTPUT_FILES",
                "OUTPUT_WRT_FREQ")  # fmt: skip


def options(text: str) -> dict[str, str]:
    """The options of an SU2 configuration (`%` starts a comment), keys stripped."""
    out = {}
    for line in text.splitlines():
        body = line.split("%", 1)[0]
        if "=" in body:
            key, value = body.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def adopt(
    published: str,
    case: autan.Case,
    mesh: Record,
    *,
    span: float,
    max_iterations: int = 20000,
    residual_orders: float = 10.0,
    stop_margin: float = STOP_MARGIN,
) -> str:
    """A configuration SU2 publishes, run on the converted `mesh` at the flight condition of
    `case`: only ADOPTED_KEYS change. The published markers must name the mesh's zones with the
    case's conditions, and its force reference must be the case's (V0903 otherwise)."""
    ours = options(config(case, mesh, 0.0, span=span, max_iterations=max_iterations,
                          residual_orders=residual_orders, stop_margin=stop_margin))  # fmt: skip
    theirs = options(published)
    checked = ("MARKER_EULER", "MARKER_FAR", "MARKER_SYM", "MARKER_MONITORING", "REF_LENGTH",
               "REF_AREA", "REF_ORIGIN_MOMENT_X", "REF_ORIGIN_MOMENT_Y", "REF_ORIGIN_MOMENT_Z")  # fmt: skip
    for key in checked:
        if key not in ours and key not in theirs:
            continue
        same = theirs.get(key, "").replace(" ", "") == ours.get(key, "").replace(" ", "")
        numeric = key.startswith("REF_") and key in theirs and key in ours
        if not same and not (numeric and float(theirs[key]) == float(ours[key])):
            msg = f"published {key} = {theirs.get(key)!r}, the case needs {ours.get(key)!r}"
            raise VVError(msg, code="V0903")
    kept = [
        line for line in published.splitlines()
        if line.split("%", 1)[0].split("=", 1)[0].strip() not in ADOPTED_KEYS
    ]  # fmt: skip
    return "\n".join([*kept, *(f"{k}= {ours[k]}" for k in ADOPTED_KEYS)]) + "\n"
