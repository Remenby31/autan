"""Observed order of accuracy, Richardson extrapolation and Roache's grid convergence index."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats

from cfd_vv.errors import VVError


def mesh_size(n_cells: int, volume: float, dim: int) -> float:
    """Return the representative mesh size h = (V / N)^(1/d) of a mesh with N cells filling volume V."""
    if n_cells <= 0 or volume <= 0.0 or dim not in (1, 2, 3):
        raise VVError("V0101", f"invalid mesh size inputs: n_cells={n_cells}, volume={volume}, dim={dim}")
    return float((volume / n_cells) ** (1.0 / dim))


@dataclass(frozen=True)
class OrderFit:
    """Least-squares fit of log(e) = log(C) + p log(h) over a mesh family."""

    order: float
    order_stderr: float
    order_ci95: tuple[float, float]
    constant: float
    r_squared: float
    pairwise_orders: tuple[float, ...]


def fit_order(h: Sequence[float], errors: Sequence[float]) -> OrderFit:
    """Fit e = C h^p by least squares in log space and return p with its uncertainty.

    `pairwise_orders[i]` is the order between levels i and i+1 after sorting from coarse to fine,
    so its last entry is the order on the finest pair.
    """
    h_arr = np.asarray(h, dtype=np.float64)
    e_arr = np.asarray(errors, dtype=np.float64)
    if h_arr.shape != e_arr.shape or h_arr.ndim != 1 or h_arr.size < 2:
        raise VVError("V0102", "need at least two (h, error) pairs of equal length")
    if np.any(h_arr <= 0.0) or np.any(e_arr <= 0.0) or not np.all(np.isfinite(e_arr)):
        raise VVError("V0103", "mesh sizes and errors must be finite and strictly positive")
    order_idx = np.argsort(h_arr)[::-1]
    h_arr, e_arr = h_arr[order_idx], e_arr[order_idx]
    if np.any(np.diff(h_arr) >= 0.0):
        raise VVError("V0104", "mesh sizes must be distinct")
    log_h, log_e = np.log(h_arr), np.log(e_arr)
    pairwise = tuple(float(v) for v in np.diff(log_e) / np.diff(log_h))
    if h_arr.size == 2:
        return OrderFit(pairwise[0], math.nan, (math.nan, math.nan), float(e_arr[0] / h_arr[0] ** pairwise[0]),
                        1.0, pairwise)
    fit = stats.linregress(log_h, log_e)
    dof = h_arr.size - 2
    half_width = float(stats.t.ppf(0.975, dof)) * float(fit.stderr)
    return OrderFit(
        order=float(fit.slope),
        order_stderr=float(fit.stderr),
        order_ci95=(float(fit.slope) - half_width, float(fit.slope) + half_width),
        constant=float(math.exp(fit.intercept)),
        r_squared=float(fit.rvalue**2),
        pairwise_orders=pairwise,
    )


class Convergence(StrEnum):
    """Convergence condition of three solutions, from the ratio R = (f2 - f1) / (f3 - f2)."""

    MONOTONIC = "monotonic"
    OSCILLATORY = "oscillatory"
    DIVERGENT = "divergent"


@dataclass(frozen=True)
class RichardsonResult:
    """Three-grid Richardson extrapolation and GCI following Celik et al. (2008), level 1 being the finest."""

    order: float
    extrapolated: float
    gci_fine: float
    gci_medium: float
    relative_gci_fine: float
    asymptotic_ratio: float
    refinement_ratios: tuple[float, float]


GCI_SAFETY_THREE_GRIDS = 1.25
GCI_SAFETY_TWO_GRIDS = 3.0


def richardson(h: Sequence[float], values: Sequence[float], *, max_iterations: int = 200) -> RichardsonResult:
    """Extrapolate a functional from its values on three grids and bound its discretisation error.

    Inputs are in any order; they are sorted so that level 1 is the finest. The observed order solves
    Celik et al. (2008), eqs. 3a-3c, by fixed-point iteration, which handles non-constant refinement ratios.
    GCI uses Roache's safety factor 1.25. `asymptotic_ratio` is GCI_32 / (r21^p GCI_21), close to 1 in the
    asymptotic range (Roache 1998). An oscillatory or divergent triplet raises: its order is meaningless.
    """
    if len(h) != 3 or len(values) != 3:
        raise VVError("V0105", "Richardson extrapolation needs exactly three grids")
    pairs = sorted(zip(h, values, strict=True), key=lambda hv: hv[0])
    (h1, f1), (h2, f2), (h3, f3) = pairs
    if not (0.0 < h1 < h2 < h3):
        raise VVError("V0104", "mesh sizes must be distinct and positive")
    r21, r32 = h2 / h1, h3 / h2
    eps21, eps32 = f2 - f1, f3 - f2
    if eps21 == 0.0 or eps32 == 0.0:
        raise VVError("V0106", "two grids give identical values: the order is undefined")
    condition = convergence_condition((f3, f2, f1))
    if condition is Convergence.OSCILLATORY:
        raise VVError("V0107", "oscillatory convergence: observed order undefined")
    if condition is Convergence.DIVERGENT:
        raise VVError("V0108", "divergent solutions: observed order undefined")
    s = 1.0
    p = abs(math.log(abs(eps32 / eps21))) / math.log(r21)
    for _ in range(max_iterations):
        q = math.log((r21**p - s) / (r32**p - s))
        p_new = abs(math.log(abs(eps32 / eps21)) + q) / math.log(r21)
        if abs(p_new - p) <= 1e-12 * max(1.0, abs(p)):
            p = p_new
            break
        p = p_new
    else:
        raise VVError("V0109", "observed order iteration did not converge")
    extrapolated = (r21**p * f1 - f2) / (r21**p - 1.0)
    gci21 = GCI_SAFETY_THREE_GRIDS * abs(eps21) / (r21**p - 1.0)
    gci32 = GCI_SAFETY_THREE_GRIDS * abs(eps32) / (r32**p - 1.0)
    return RichardsonResult(
        order=p,
        extrapolated=extrapolated,
        gci_fine=gci21,
        gci_medium=gci32,
        relative_gci_fine=gci21 / abs(f1) if f1 != 0.0 else math.inf,
        asymptotic_ratio=gci32 / (r21**p * gci21),
        refinement_ratios=(r21, r32),
    )


def gci_two_grids(h: Sequence[float], values: Sequence[float], order: float) -> float:
    """Return Roache's two-grid GCI on the finer grid, assuming `order`, with safety factor 3."""
    if len(h) != 2 or len(values) != 2 or order <= 0.0:
        raise VVError("V0110", "two-grid GCI needs two grids and a positive assumed order")
    (h1, f1), (h2, f2) = sorted(zip(h, values, strict=True), key=lambda hv: hv[0])
    if not 0.0 < h1 < h2:
        raise VVError("V0104", "mesh sizes must be distinct and positive")
    return GCI_SAFETY_TWO_GRIDS * abs(f2 - f1) / ((h2 / h1) ** order - 1.0)


def convergence_condition(values_coarse_to_fine: Sequence[float]) -> Convergence:
    """Classify three solutions ordered coarse to fine as monotonic, oscillatory or divergent."""
    f3, f2, f1 = values_coarse_to_fine
    ratio = (f2 - f1) / (f3 - f2)
    if ratio < 0.0:
        return Convergence.OSCILLATORY
    if ratio >= 1.0:
        return Convergence.DIVERGENT
    return Convergence.MONOTONIC
