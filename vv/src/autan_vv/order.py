"""Observed order of accuracy, Richardson extrapolation and Roache's grid convergence index."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import stats

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from collections.abc import Sequence


def mesh_size(n_cells: int, volume: float, dim: int) -> float:
    """Return the representative mesh size h = (V / N)^(1/d) of a mesh with N cells filling volume V."""
    if n_cells <= 0 or not math.isfinite(volume) or volume <= 0.0 or dim not in (1, 2, 3):
        msg = f"invalid mesh size inputs: n_cells={n_cells}, volume={volume}, dim={dim}"
        raise VVError(msg, code="V0101")
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


def fit_order(h: npt.ArrayLike, errors: npt.ArrayLike) -> OrderFit:
    """Fit e = C h^p by least squares in log space and return p with its uncertainty.

    `pairwise_orders[i]` is the order between levels i and i+1 after sorting from coarse to fine,
    so its last entry is the order on the finest pair.
    """
    h_arr = np.asarray(h, dtype=np.float64)
    e_arr = np.asarray(errors, dtype=np.float64)
    if h_arr.shape != e_arr.shape or h_arr.ndim != 1 or h_arr.size < 2:
        msg = "need at least two (h, error) pairs of equal length"
        raise VVError(msg, code="V0102")
    finite = np.all(np.isfinite(h_arr)) and np.all(np.isfinite(e_arr))
    if not finite or np.any(h_arr <= 0.0) or np.any(e_arr <= 0.0):
        msg = "mesh sizes and errors must be finite and strictly positive"
        raise VVError(msg, code="V0103")
    order_idx = np.argsort(h_arr)[::-1]
    h_arr, e_arr = h_arr[order_idx], e_arr[order_idx]
    if np.any(np.diff(h_arr) >= 0.0):
        msg = "mesh sizes must be distinct"
        raise VVError(msg, code="V0104")
    log_h, log_e = np.log(h_arr), np.log(e_arr)
    pairwise = tuple(float(v) for v in np.diff(log_e) / np.diff(log_h))
    if h_arr.size == 2:
        return OrderFit(
            pairwise[0],
            math.nan,
            (math.nan, math.nan),
            float(np.exp(min(np.log(e_arr[0]) - pairwise[0] * np.log(h_arr[0]), 709.0))),
            1.0,
            pairwise,
        )
    fit = stats.linregress(log_h, log_e)
    dof = h_arr.size - 2
    half_width = float(stats.t.ppf(0.975, dof)) * float(fit.stderr)
    return OrderFit(
        order=float(fit.slope),
        order_stderr=float(fit.stderr),
        order_ci95=(float(fit.slope) - half_width, float(fit.slope) + half_width),
        # exp overflows for far-from-asymptotic data; the order stays meaningful.
        constant=math.exp(fit.intercept) if fit.intercept < 709.0 else math.inf,
        r_squared=float(fit.rvalue**2),
        pairwise_orders=pairwise,
    )


@dataclass(frozen=True)
class RichardsonResult:
    """Three-grid Richardson extrapolation and GCI following Celik et al. (2008), level 1 being the finest."""

    order: float
    extrapolated: float
    gci_fine: float
    gci_medium: float
    relative_gci_fine: float
    refinement_ratios: tuple[float, float]


GCI_SAFETY_THREE_GRIDS = 1.25
GCI_SAFETY_TWO_GRIDS = 3.0


def _log_expm1(x: float) -> float:
    """log(e^x - 1) for x > 0 without overflow."""
    return x + math.log(-math.expm1(-x)) if x > 1.0 else math.log(math.expm1(x))


def _log_ratio_model(p: float, ln21: float, ln32: float) -> float:
    """log of (f3 - f2)/(f2 - f1) for f = f0 + a h^p: p ln r21 + log((r32^p - 1)/(r21^p - 1))."""
    return p * ln21 + _log_expm1(p * ln32) - _log_expm1(p * ln21)


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        msg = "mesh sizes, values and orders must be finite"
        raise VVError(msg, code="V0103")


def richardson(h: Sequence[float], values: Sequence[float]) -> RichardsonResult:
    """Extrapolate a functional from its values on three grids and bound its discretisation error.

    Inputs are in any order; level 1 is the finest. The observed order solves Celik et al. (2008),
    eqs. 3a-3c, written for f = f0 + a h^p as (f3 - f2)/(f2 - f1) = r21^p (r32^p - 1)/(r21^p - 1),
    by bisection in log form, which holds for any refinement ratios. GCI uses Roache's safety
    factor 1.25. Differences of opposite signs (oscillatory, V0107) or a ratio no positive order
    explains (V0108, V0109) raise: the order is then meaningless.
    """
    if len(h) != 3 or len(values) != 3:
        msg = "Richardson extrapolation needs exactly three grids"
        raise VVError(msg, code="V0105")
    _require_finite(*h, *values)
    pairs = sorted(zip(h, values, strict=True), key=lambda hv: hv[0])
    (h1, f1), (h2, f2), (h3, f3) = pairs
    if not (0.0 < h1 < h2 < h3):
        msg = "mesh sizes must be distinct and positive"
        raise VVError(msg, code="V0104")
    r21, r32 = h2 / h1, h3 / h2
    eps21, eps32 = f2 - f1, f3 - f2
    if eps21 == 0.0 or eps32 == 0.0:
        msg = "two grids give identical values: the order is undefined"
        raise VVError(msg, code="V0106")
    if (eps21 > 0.0) != (eps32 > 0.0):
        msg = "oscillatory convergence: observed order undefined"
        raise VVError(msg, code="V0107")
    ln21, ln32 = math.log(r21), math.log(r32)
    target = math.log(eps32 / eps21)
    # The model ratio increases with p from ln r32 / ln r21 (p -> 0).
    p_lo, p_hi = 1e-8, 64.0
    if target <= _log_ratio_model(p_lo, ln21, ln32):
        msg = "the solutions do not converge as h^p for any positive p"
        raise VVError(msg, code="V0108")
    if target >= _log_ratio_model(p_hi, ln21, ln32):
        msg = "observed order above 64: the grids are not in a power-law range"
        raise VVError(msg, code="V0109")
    for _ in range(200):
        p = 0.5 * (p_lo + p_hi)
        if _log_ratio_model(p, ln21, ln32) < target:
            p_lo = p
        else:
            p_hi = p
    p = 0.5 * (p_lo + p_hi)
    # eps / (r^p - 1) evaluated as eps * exp(-log(expm1(p ln r))) so that large r^p cannot overflow.
    scale21 = math.exp(-_log_expm1(p * ln21))
    scale32 = math.exp(-_log_expm1(p * ln32))
    extrapolated = f1 - eps21 * scale21
    gci21 = GCI_SAFETY_THREE_GRIDS * abs(eps21) * scale21
    gci32 = GCI_SAFETY_THREE_GRIDS * abs(eps32) * scale32
    return RichardsonResult(
        order=p,
        extrapolated=extrapolated,
        gci_fine=gci21,
        gci_medium=gci32,
        relative_gci_fine=gci21 / abs(f1) if f1 != 0.0 else math.inf,
        refinement_ratios=(r21, r32),
    )


def gci_two_grids(h: Sequence[float], values: Sequence[float], order: float) -> float:
    """Return Roache's two-grid GCI on the finer grid, assuming `order`, with safety factor 3."""
    if len(h) != 2 or len(values) != 2 or order <= 0.0:
        msg = "two-grid GCI needs two grids and a positive assumed order"
        raise VVError(msg, code="V0110")
    _require_finite(*h, *values, order)
    (h1, f1), (h2, f2) = sorted(zip(h, values, strict=True), key=lambda hv: hv[0])
    if not 0.0 < h1 < h2:
        msg = "mesh sizes must be distinct and positive"
        raise VVError(msg, code="V0104")
    return float(GCI_SAFETY_TWO_GRIDS * abs(f2 - f1) / ((h2 / h1) ** order - 1.0))
