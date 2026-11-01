"""Manufactured advection-diffusion problem and its steady solve through the generated residual.

phi = 1 + 1/2 sin(1.1 pi x + 0.3) cos(0.9 pi y + 0.1) sin(0.7 pi z + 0.6), a = (1, 0.7, 0.4),
nu = 0.1 (the `advection_diffusion` solution of the V&V catalogue). The source
s = a . grad(phi) - nu lap(phi) is derived by hand below; lap(phi) = -(A^2 + B^2 + C^2)(phi - 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import autan_spike as s
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

F64 = npt.NDArray[np.float64]
A, B, C = 1.1 * np.pi, 0.9 * np.pi, 0.7 * np.pi
VELOCITY = (1.0, 0.7, 0.4)
NU = 0.1


def exact(x: F64) -> F64:
    return 1.0 + 0.5 * np.sin(A * x[:, 0] + 0.3) * np.cos(B * x[:, 1] + 0.1) * np.sin(
        C * x[:, 2] + 0.6
    )


def source(x: F64) -> F64:
    sx, cx = np.sin(A * x[:, 0] + 0.3), np.cos(A * x[:, 0] + 0.3)
    sy, cy = np.sin(B * x[:, 1] + 0.1), np.cos(B * x[:, 1] + 0.1)
    sz, cz = np.sin(C * x[:, 2] + 0.6), np.cos(C * x[:, 2] + 0.6)
    gx = 0.5 * A * cx * cy * sz
    gy = -0.5 * B * sx * sy * sz
    gz = 0.5 * C * sx * cy * cz
    lap = -(A * A + B * B + C * C) * 0.5 * sx * cy * sz
    a = VELOCITY
    return a[0] * gx + a[1] * gy + a[2] * gz - NU * lap


def first_order_operator(m: s.Mesh) -> sp.csr_matrix:
    """Upwind convection and two-point diffusion: the preconditioner of the steady solve."""
    a = np.array(VELOCITY)
    ni = m.n_interior
    o, n = m.owner[:ni], m.neighbour
    S = m.face_area
    un = S @ a
    d = m.cell_centroid[n] - m.cell_centroid[o]
    dif = NU * np.einsum("ij,ij->i", S[:ni], S[:ni]) / np.abs(np.einsum("ij,ij->i", d, S[:ni]))
    rows, cols, vals = [], [], []

    def add(r: npt.NDArray[np.int32], c: npt.NDArray[np.int32], v: F64) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    up = un[:ni]
    add(o, o, np.maximum(up, 0) + dif)
    add(o, n, np.minimum(up, 0) - dif)
    add(n, n, -np.minimum(up, 0) + dif)
    add(n, o, -np.maximum(up, 0) - dif)
    ob = m.owner[ni:]
    db = m.face_centroid[ni:] - m.cell_centroid[ob]
    Sb = S[ni:]
    add(
        ob,
        ob,
        np.maximum(un[ni:], 0)
        + NU * np.einsum("ij,ij->i", Sb, Sb) / np.abs(np.einsum("ij,ij->i", db, Sb)),
    )
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(m.n_cells, m.n_cells),
    )


@dataclass
class Solution:
    q: F64
    error_l1: float
    error_l2: float
    error_linf: float
    residual: float
    iterations: int


def solve(
    m: s.Mesh, policy: str = "fp64", tol: float = 1e-12, strategy: str | None = None
) -> Solution:
    """Solve R(q) = 0 by GMRES; R is affine, so J v = R(v) - R(0) with zero data.

    The residual runs on the host loops, or on the device loops in race `strategy` if given.
    """

    def residual(*args: object) -> tuple[F64, F64]:
        if strategy is None:
            return s.advdiff_residual(m, policy, *args)  # type: ignore[arg-type]
        res, grad, _norm = s.gpu_advdiff_residual(m, policy, strategy, 128, *args)  # type: ignore[arg-type]
        return res, grad

    q_b = np.zeros(m.n_faces)
    q_b[m.n_interior :] = exact(m.face_centroid[m.n_interior :])
    src = source(m.cell_centroid)
    zero_b = np.zeros(m.n_faces)
    zero_s = np.zeros(m.n_cells)
    r0, _ = residual(np.zeros(m.n_cells), q_b, src, VELOCITY, NU)

    def matvec(v: F64) -> F64:
        return residual(v, zero_b, zero_s, VELOCITY, NU)[0]

    J = spla.LinearOperator((m.n_cells, m.n_cells), matvec=matvec)  # type: ignore[call-arg]
    ilu = spla.spilu(first_order_operator(m).tocsc(), drop_tol=1e-5, fill_factor=20)
    M = spla.LinearOperator((m.n_cells, m.n_cells), matvec=ilu.solve)  # type: ignore[call-arg]
    count = [0]

    def cb(_: object) -> None:
        count[0] += 1

    # Defect correction around GMRES, so that the final residual is the one of the policy.
    q = np.zeros(m.n_cells)
    for _ in range(4):
        r, _g = residual(q, q_b, src, VELOCITY, NU)
        dq, _info = spla.gmres(
            J,
            -r,
            M=M,
            rtol=tol,
            atol=0.0,
            restart=60,
            maxiter=400,
            callback=cb,
            callback_type="pr_norm",
        )
        q = q + dq
    r, _g = residual(q, q_b, src, VELOCITY, NU)
    e = q - exact(m.cell_centroid)
    v = m.cell_volume
    return Solution(
        q,
        float(np.sum(np.abs(e) * v) / v.sum()),
        float(np.sqrt(np.sum(e * e * v) / v.sum())),
        float(np.abs(e).max()),
        float(np.linalg.norm(r) / max(np.linalg.norm(r0), 1e-300)),
        count[0],
    )


def observed_order(h: list[float], e: list[float]) -> tuple[float, float]:
    """Least-squares slope of log e against log h over all levels, and the finest-pair order."""
    lh, le = np.log(h), np.log(e)
    fit = float(np.polyfit(lh, le, 1)[0])
    pair = float((le[-1] - le[-2]) / (lh[-1] - lh[-2]))
    return fit, pair
