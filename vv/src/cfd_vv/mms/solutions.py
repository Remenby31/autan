"""Manufactured solutions and their source terms, derived symbolically.

Every solution is steady and lives on the unit cube [0, 1]^3. For a system written as
dU/dt + div(F_c(U) - F_v(U, grad U)) = Q(U, grad U) + S, the manufactured source is
S = div(F_c - F_v) - Q evaluated on the manufactured fields; Q holds the model's own
source terms (Spalart-Allmaras production and destruction). The solver adds S as a
cell source, integrated with the same quadrature as its other volume terms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import sympy as sp

from cfd_vv.errors import VVError

x, y, z = X = sp.symbols("x y z", real=True)
DOMAIN: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

FloatArray = npt.NDArray[np.float64]


def R(value: str) -> sp.Rational:
    """Exact rational from a decimal literal, so exported coefficients carry no binary rounding."""
    return sp.Rational(value)


@dataclass(frozen=True)
class Validity:
    """A condition the derivation assumes, checked by sampling the domain: `lower < expr < upper`."""

    name: str
    expr: sp.Expr
    lower: float = -np.inf
    upper: float = np.inf
    region: str = "domain"


@dataclass(frozen=True)
class ManufacturedSolution:
    """A steady manufactured solution: primitive fields, conserved variables and per-equation sources."""

    name: str
    equations: str
    description: str
    parameters: Mapping[str, float]
    fields: Mapping[str, sp.Expr]
    conserved: Mapping[str, sp.Expr]
    sources: Mapping[str, sp.Expr]
    auxiliary: Mapping[str, sp.Expr] = field(default_factory=dict)
    validity: tuple[Validity, ...] = ()
    references: tuple[str, ...] = ()

    def gradients(self) -> dict[str, tuple[sp.Expr, sp.Expr, sp.Expr]]:
        """Exact gradients of the primitive fields."""
        return {k: (sp.diff(v, x), sp.diff(v, y), sp.diff(v, z)) for k, v in self.fields.items()}

    def evaluator(self, kind: str) -> Callable[[FloatArray], dict[str, FloatArray]]:
        """Vectorised FP64 evaluator of `fields`, `conserved`, `sources` or `auxiliary` at points (n, 3)."""
        group: Mapping[str, sp.Expr] = getattr(self, kind)
        names = list(group)
        fns = [sp.lambdify(X, group[n], "numpy") for n in names]

        def evaluate(points: FloatArray) -> dict[str, FloatArray]:
            p = np.asarray(points, dtype=np.float64)
            return {n: np.broadcast_to(f(p[:, 0], p[:, 1], p[:, 2]), p.shape[:1]).astype(np.float64)
                    for n, f in zip(names, fns, strict=True)}

        return evaluate

    def check_validity(self, samples_per_axis: int = 24) -> None:
        """Sample the domain and raise V0201 if an assumption of the derivation is violated."""
        if not self.validity:
            return
        t = (np.arange(samples_per_axis) + 0.5) / samples_per_axis
        gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
        # Wall-bounded checks also probe the first cell layer of a 1/1024 mesh.
        near_wall = np.geomspace(1.0 / 2048.0, t[0], 6)
        wx, wy, wz = np.meshgrid(t, near_wall, t, indexing="ij")
        px, py, pz = (np.concatenate([a.ravel(), b.ravel()]) for a, b in ((gx, wx), (gy, wy), (gz, wz)))
        for cond in self.validity:
            values = np.broadcast_to(sp.lambdify(X, cond.expr, "numpy")(px, py, pz), px.shape)
            bad = ~((values > cond.lower) & (values < cond.upper)) | ~np.isfinite(values)
            if np.any(bad):
                i = int(np.argmax(bad))
                raise VVError("V0201", f"{self.name}: assumption '{cond.name}' fails at "
                                       f"({px[i]:.4g}, {py[i]:.4g}, {pz[i]:.4g}): value {values[i]:.6g}")


def _div(flux: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.diff(flux[0], x) + sp.diff(flux[1], y) + sp.diff(flux[2], z)


def _grad(f: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (sp.diff(f, x), sp.diff(f, y), sp.diff(f, z))


def trig_field(base: str, *terms: tuple[str, str, str, str]) -> sp.Expr:
    """Roy et al. (2004), eq. 11 form: phi0 + sum_k a_k f_k(c_k pi x_k), f_k in {sin, cos}, extended to 3D.

    Each term is (axis, function, amplitude, wavenumber factor), e.g. ("x", "sin", "0.15", "1").
    """
    fns = {"sin": sp.sin, "cos": sp.cos}
    axes = {"x": x, "y": y, "z": z}
    expr: sp.Expr = R(base)
    for axis, fn, amplitude, wavenumber in terms:
        expr += R(amplitude) * fns[fn](R(wavenumber) * sp.pi * axes[axis])
    return expr


# --- scalar advection-diffusion ------------------------------------------------------------------------------


def advection_diffusion(
    velocity: tuple[str, str, str] = ("1", "0.7", "0.4"),
    diffusivity: str = "0.1",
    solution: sp.Expr | None = None,
) -> ManufacturedSolution:
    """Steady div(a phi) - div(nu grad phi) = S with constant velocity a and diffusivity nu."""
    a = tuple(R(c) for c in velocity)
    nu = R(diffusivity)
    phi = solution if solution is not None else (
        1 + R("0.5") * sp.sin(R("1.1") * sp.pi * x + R("0.3")) * sp.cos(R("0.9") * sp.pi * y + R("0.1"))
        * sp.sin(R("0.7") * sp.pi * z + R("0.6"))
    )
    gphi = _grad(phi)
    flux = tuple(a[i] * phi - nu * gphi[i] for i in range(3))
    return ManufacturedSolution(
        name="advection_diffusion",
        equations="scalar advection-diffusion",
        description="div(a phi) - div(nu grad phi) = S, a and nu constant, smooth non-polynomial phi.",
        parameters={"ax": float(a[0]), "ay": float(a[1]), "az": float(a[2]), "nu": float(nu)},
        fields={"phi": phi},
        conserved={"phi": phi},
        sources={"phi": _div(flux)},
        references=("roache2002",),
    )


# --- compressible flow ----------------------------------------------------------------------------------------

GAMMA = R("1.4")
GAS_CONSTANT = R("287")


@dataclass(frozen=True)
class Flow:
    """Primitive compressible fields: density, velocity and pressure."""

    rho: sp.Expr
    u: tuple[sp.Expr, sp.Expr, sp.Expr]
    p: sp.Expr

    @property
    def temperature(self) -> sp.Expr:
        return self.p / (self.rho * GAS_CONSTANT)

    @property
    def total_energy(self) -> sp.Expr:
        return self.p / (GAMMA - 1) + self.rho * sum(c**2 for c in self.u) / 2

    def conserved(self) -> dict[str, sp.Expr]:
        return {"rho": self.rho, "rhou": self.rho * self.u[0], "rhov": self.rho * self.u[1],
                "rhow": self.rho * self.u[2], "rhoE": self.total_energy}

    def primitive(self) -> dict[str, sp.Expr]:
        return {"rho": self.rho, "u": self.u[0], "v": self.u[1], "w": self.u[2], "p": self.p}


def _convective_fluxes(f: Flow) -> list[tuple[sp.Expr, sp.Expr, sp.Expr]]:
    rho, u, p = f.rho, f.u, f.p
    h0 = f.total_energy + p
    mass = tuple(rho * u[j] for j in range(3))
    mom = [tuple(rho * u[i] * u[j] + (p if i == j else 0) for j in range(3)) for i in range(3)]
    energy = tuple(h0 * u[j] for j in range(3))
    return [mass, *mom, energy]


def _viscous_fluxes(f: Flow, mu: sp.Expr, conductivity: sp.Expr) -> list[tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """Newtonian stress with Stokes' hypothesis and Fourier heat flux (TMR, 'Implementing turbulence models')."""
    u = f.u
    grad_u = [[sp.diff(u[i], X[j]) for j in range(3)] for i in range(3)]
    div_u = sum(grad_u[k][k] for k in range(3))
    tau = [[mu * (grad_u[i][j] + grad_u[j][i] - (sp.Rational(2, 3) * div_u if i == j else 0)) for j in range(3)]
           for i in range(3)]
    grad_t = _grad(f.temperature)
    zero = (sp.Integer(0),) * 3
    mom = [tuple(tau[i][j] for j in range(3)) for i in range(3)]
    energy = tuple(sum(u[i] * tau[i][j] for i in range(3)) + conductivity * grad_t[j] for j in range(3))
    return [zero, *mom, energy]


def _sources(conv: list[tuple[sp.Expr, sp.Expr, sp.Expr]], visc: list[tuple[sp.Expr, sp.Expr, sp.Expr]] | None,
             names: list[str]) -> dict[str, sp.Expr]:
    out = {}
    for k, name in enumerate(names):
        flux = conv[k] if visc is None else tuple(conv[k][j] - visc[k][j] for j in range(3))
        out[name] = _div(flux)
    return out


ROY_SUBSONIC_3D = Flow(
    rho=trig_field("1", ("x", "sin", "0.15", "1"), ("y", "cos", "-0.1", "0.5"), ("z", "sin", "0.08", "0.75")),
    u=(
        trig_field("70", ("x", "sin", "4", "5/3"), ("y", "cos", "-12", "3/2"), ("z", "cos", "7", "1/2")),
        trig_field("90", ("x", "cos", "-20", "3/2"), ("y", "sin", "4", "1"), ("z", "sin", "-11", "4/3")),
        trig_field("80", ("x", "sin", "8", "5/4"), ("y", "cos", "-11", "1"), ("z", "cos", "6", "3/4")),
    ),
    p=trig_field("100000", ("x", "cos", "-30000", "1"), ("y", "sin", "20000", "5/4"), ("z", "cos", "10000", "3/4")),
)


def _flow_validity(f: Flow) -> tuple[Validity, ...]:
    mach2 = sum(c**2 for c in f.u) * f.rho / (GAMMA * f.p)
    return (Validity("density positive", f.rho, lower=0.0), Validity("pressure positive", f.p, lower=0.0),
            Validity("subsonic", mach2, upper=1.0))


def euler(flow: Flow = ROY_SUBSONIC_3D) -> ManufacturedSolution:
    """Steady compressible Euler, calorically perfect air (gamma 1.4, R 287 J/(kg K))."""
    names = ["rho", "rhou", "rhov", "rhow", "rhoE"]
    ms = ManufacturedSolution(
        name="euler",
        equations="compressible Euler",
        description="Subsonic 3D trigonometric fields in the form of Roy et al. (2004).",
        parameters={"gamma": float(GAMMA), "gas_constant": float(GAS_CONSTANT)},
        fields=flow.primitive(),
        conserved=flow.conserved(),
        sources=_sources(_convective_fluxes(flow), None, names),
        validity=_flow_validity(flow),
        references=("roy2004", "salari2000"),
    )
    ms.check_validity()
    return ms


def navier_stokes(
    flow: Flow = ROY_SUBSONIC_3D, viscosity: str = "10", prandtl: str = "0.72",
) -> ManufacturedSolution:
    """Steady laminar compressible Navier-Stokes with constant viscosity and Prandtl number.

    The default viscosity gives a Reynolds number of order 10 on the unit cube so that viscous and
    convective terms are both exercised, as in Roy et al. (2004).
    """
    mu, pr = R(viscosity), R(prandtl)
    cp = GAMMA * GAS_CONSTANT / (GAMMA - 1)
    names = ["rho", "rhou", "rhov", "rhow", "rhoE"]
    ms = ManufacturedSolution(
        name="navier_stokes",
        equations="compressible laminar Navier-Stokes",
        description="Subsonic 3D trigonometric fields in the form of Roy et al. (2004), constant viscosity.",
        parameters={"gamma": float(GAMMA), "gas_constant": float(GAS_CONSTANT), "viscosity": float(mu),
                    "prandtl": float(pr)},
        fields=flow.primitive(),
        conserved=flow.conserved(),
        sources=_sources(_convective_fluxes(flow), _viscous_fluxes(flow, mu, cp * mu / pr), names),
        validity=_flow_validity(flow),
        references=("roy2004", "salari2000"),
    )
    ms.check_validity()
    return ms


# --- RANS with Spalart-Allmaras -------------------------------------------------------------------------------

SA_CONSTANTS: dict[str, sp.Expr] = {
    "cb1": R("0.1355"), "sigma": sp.Rational(2, 3), "cb2": R("0.622"), "kappa": R("0.41"), "cw2": R("0.3"),
    "cw3": sp.Integer(2), "cv1": R("7.1"), "ct3": R("1.2"), "ct4": R("0.5"), "c2": R("0.7"), "c3": R("0.9"),
    "prandtl_turbulent": R("0.9"),
}


def _sa_wall_flow(u_edge: str, thickness: str, nu_hat_max: str) -> tuple[Flow, sp.Expr]:
    """Wall-bounded fields: no-slip velocity and nu_hat = 0 at the wall y = 0, nu_hat ~ y near it.

    Shapes follow the principles of Eca et al. (2007) and Oliver et al. (2012): a boundary-layer-like
    velocity, an eddy-viscosity variable linear in the wall distance near the wall, smooth elsewhere.
    """
    bl = 1 - sp.exp(-y / R(thickness))
    u = R(u_edge) * bl * (1 + R("0.1") * sp.sin(R("0.8") * sp.pi * x) + R("0.05") * sp.cos(R("0.6") * sp.pi * z))
    v = 5 * bl**2 * sp.sin(R("0.5") * sp.pi * x + R("0.2"))
    w = 10 * bl * sp.cos(R("0.7") * sp.pi * x) * sp.sin(R("0.5") * sp.pi * z + R("0.3"))
    rho = trig_field("1", ("x", "sin", "0.1", "0.75"), ("y", "cos", "0.05", "1"), ("z", "sin", "0.05", "0.5"))
    p = trig_field("100000", ("x", "cos", "2000", "1"), ("y", "sin", "1000", "1"), ("z", "cos", "1000", "0.5"))
    eta = y / R("0.25")
    nu_hat = R(nu_hat_max) * eta * sp.exp(1 - eta) * (1 + R("0.2") * sp.sin(R("0.5") * sp.pi * x)
                                                  * sp.cos(R("0.4") * sp.pi * z))
    return Flow(rho=rho, u=(u, v, w), p=p), nu_hat


def rans_sa(
    viscosity: str = "0.001", prandtl: str = "0.72", ft2: bool = True,
    wall_flow: tuple[str, str, str] = ("50", "0.15", "0.01"),
) -> ManufacturedSolution:
    """Steady compressible RANS closed by Spalart-Allmaras, conservative form of Allmaras et al. (2012).

    With nu_hat >= 0 the SA-neg model equals standard SA with the S_tilde of Allmaras et al. (2012),
    Note 1(c) of the TMR page; `ft2=False` gives SA-noft2. The wall distance is d = y (wall at y = 0,
    the mesh boundary named "ymin"). The fields are chosen so that the clipping branches never activate
    (S_bar >= -c2 Omega, r < 10, S_tilde > 0), which `check_validity` verifies; the source is therefore
    smooth and the solver must use the same branches.
    """
    c = SA_CONSTANTS
    flow, nu_hat = _sa_wall_flow(*wall_flow)
    mu, pr = R(viscosity), R(prandtl)
    cp = GAMMA * GAS_CONSTANT / (GAMMA - 1)
    d = y
    rho = flow.rho
    nu = mu / rho
    chi = nu_hat / nu
    fv1 = chi**3 / (chi**3 + c["cv1"] ** 3)
    mu_t = rho * nu_hat * fv1
    fv2 = 1 - chi / (1 + chi * fv1)
    u = flow.u
    vort = (sp.diff(u[2], y) - sp.diff(u[1], z), sp.diff(u[0], z) - sp.diff(u[2], x),
            sp.diff(u[1], x) - sp.diff(u[0], y))
    omega = sp.sqrt(sum(w_**2 for w_ in vort))
    s_bar = nu_hat * fv2 / (c["kappa"] ** 2 * d**2)
    s_tilde = omega + s_bar
    r = nu_hat / (s_tilde * c["kappa"] ** 2 * d**2)
    g = r + c["cw2"] * (r**6 - r)
    fw = g * ((1 + c["cw3"] ** 6) / (g**6 + c["cw3"] ** 6)) ** sp.Rational(1, 6)
    cw1 = c["cb1"] / c["kappa"] ** 2 + (1 + c["cb2"]) / c["sigma"]
    f_t2 = c["ct3"] * sp.exp(-c["ct4"] * chi**2) if ft2 else sp.Integer(0)
    production = c["cb1"] * (1 - f_t2) * s_tilde * nu_hat
    destruction = (cw1 * fw - c["cb1"] / c["kappa"] ** 2 * f_t2) * (nu_hat / d) ** 2
    g_nu = _grad(nu_hat)
    g_rho = _grad(rho)
    sa_conv = tuple(rho * u[j] * nu_hat for j in range(3))
    sa_diff = tuple(rho * (nu + nu_hat) * g_nu[j] / c["sigma"] for j in range(3))
    # ICCFD7-1902 conservative form (TMR, SA-neg section): the cb2 and density-gradient terms are sources.
    sa_q = rho * production - rho * destruction + (
        rho * c["cb2"] * sum(gi**2 for gi in g_nu) - (nu + nu_hat) * sum(g_rho[i] * g_nu[i] for i in range(3))
    ) / c["sigma"]
    conductivity = cp * (mu / pr + mu_t / c["prandtl_turbulent"])
    names = ["rho", "rhou", "rhov", "rhow", "rhoE"]
    sources = _sources(_convective_fluxes(flow), _viscous_fluxes(flow, mu + mu_t, conductivity), names)
    sources["rhonu"] = _div(tuple(sa_conv[j] - sa_diff[j] for j in range(3))) - sa_q
    fields = flow.primitive() | {"nu_hat": nu_hat}
    conserved = flow.conserved() | {"rhonu": rho * nu_hat}
    params = {"gamma": float(GAMMA), "gas_constant": float(GAS_CONSTANT), "viscosity": float(mu),
              "prandtl": float(pr), "ft2": 1.0 if ft2 else 0.0} | {k: float(v) for k, v in c.items()}
    ms = ManufacturedSolution(
        name="rans_sa" if ft2 else "rans_sa_noft2",
        equations="compressible RANS, Spalart-Allmaras (SA-neg)" if ft2 else "compressible RANS, SA-noft2",
        description="Wall-bounded 3D fields, wall at y = 0, d = y, constant laminar viscosity.",
        parameters=params,
        fields=fields,
        conserved=conserved,
        sources=sources,
        auxiliary={"wall_distance": d, "mu_t": mu_t, "chi": chi, "r": r, "s_tilde": s_tilde},
        validity=(*_flow_validity(flow),
                  Validity("nu_hat positive", nu_hat, lower=0.0),
                  Validity("S_bar >= -c2 Omega (no S_tilde clipping)", s_bar + c["c2"] * omega, lower=0.0),
                  Validity("S_tilde positive", s_tilde, lower=0.0),
                  Validity("r below its limit 10", r, upper=10.0)),
        references=("spalart1994", "allmaras2012", "eca2007", "oliver2012", "tmr_sa"),
    )
    ms.check_validity()
    return ms


CATALOG: dict[str, Callable[[], ManufacturedSolution]] = {
    "advection_diffusion": advection_diffusion,
    "euler": euler,
    "navier_stokes": navier_stokes,
    "rans_sa": rans_sa,
    "rans_sa_noft2": lambda: rans_sa(ft2=False),
}
