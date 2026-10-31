"""Reference cell-centred finite-volume solve of steady scalar advection-diffusion, in FP64.

It exists to prove the order tool and the mesh families end to end, independently of the C++
solver, and to serve as a second-order baseline its results can be compared with. Scheme:

- gradients: weighted least squares over face neighbours and boundary face centroids, weights
  1/|d|^2 (Mavriplis, AIAA 2003-3986);
- convection: upwind cell value reconstructed with the upwind cell gradient and integrated over
  the face's triangle fan; exact Dirichlet value on boundary faces (planar box faces);
- diffusion: face gradient grad_f = mean(grad) + alpha (dphi - mean(grad).d) d/|d|^2, alpha = 4/3
  inside (alpha-damping, Nishikawa, AIAA 2010-5093) and 2 on boundary faces; correcting along d
  keeps the coefficients bounded on non-convex cells where d.n can be negative;
- source: the manufactured source at the cell centroid times the cell volume.

All Dirichlet data come from the manufactured solution. The operator is affine in phi; it is
assembled as a sparse matrix and solved with GMRES preconditioned by the LU factors of the
compact first-order operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from autan_vv.errors import VVError

if TYPE_CHECKING:
    from autan_vv.meshes.facemesh import FaceMesh, FloatArray
    from autan_vv.mms.solutions import ManufacturedSolution

ALPHA = 4.0 / 3.0
# Boundary faces: the quadratic through phi_P, its derivative along d and phi_b gives
# d(phi)/dd at the face = 2 (phi_b - phi_P)/|d| - grad_P.d/|d|, i.e. alpha = 2.
ALPHA_BOUNDARY = 2.0


@dataclass(frozen=True)
class Solution:
    """Cell values, discretisation errors and solver diagnostics of one solve."""

    phi: FloatArray
    error_l1: float
    error_l2: float
    error_linf: float
    residual_norm: float
    iterations: int


def _lsq_gradient(mesh: FaceMesh, phi_b: FloatArray) -> tuple[list[sps.csr_matrix], FloatArray]:
    """Return sparse (Gx, Gy, Gz) and the boundary contribution g0 with grad = G phi + g0."""
    g = mesh.geometry
    nc, ni = mesh.n_cells, mesh.n_interior
    xc, xf = g.cell_centroid, g.face_centroid
    own, nb = mesh.owner, mesh.neighbour
    bf = np.arange(ni, mesh.n_faces)
    # Links: (cell, other cell or -1 for boundary, displacement).
    src = np.concatenate([own[:ni], nb, own[bf]])
    dst = np.concatenate([nb, own[:ni], np.full(bf.size, -1)])
    d = np.concatenate([xc[nb] - xc[own[:ni]], xc[own[:ni]] - xc[nb], xf[bf] - xc[own[bf]]])
    w = 1.0 / np.einsum("ij,ij->i", d, d)
    m = np.zeros((nc, 3, 3))
    np.add.at(m, src, w[:, None, None] * d[:, :, None] * d[:, None, :])
    k = w[:, None] * np.einsum("nij,nj->ni", np.linalg.inv(m)[src], d)
    interior = dst >= 0
    mats = []
    g0 = np.zeros((nc, 3))
    np.add.at(g0, src[~interior], k[~interior] * phi_b[:, None])
    for axis in range(3):
        rows = np.concatenate([src[interior], np.arange(nc)])
        cols = np.concatenate([dst[interior], np.arange(nc)])
        diag = np.zeros(nc)
        np.add.at(diag, src, -k[:, axis])
        vals = np.concatenate([k[interior, axis], diag])
        mats.append(sps.csr_matrix((vals, (rows, cols)), shape=(nc, nc)))
    return mats, g0


PRECISIONS: dict[str, type[np.floating[Any]]] = {"fp64": np.float64, "fp32": np.float32}


def solve(
    mesh: FaceMesh, ms: ManufacturedSolution, precision: str = "fp64", rtol: float | None = None
) -> Solution:
    """Solve the advection-diffusion problem `ms` (from `autan_vv.mms.advection_diffusion`) on `mesh`.

    Geometry and coefficients are computed in FP64 (the solver's geometry policy); with
    `precision="fp32"` the operator, right-hand side, preconditioner and Krylov solve are FP32.
    `rtol` defaults to 64 machine epsilons of the precision. Requires a positive diffusivity: with
    every boundary value prescribed, pure advection has a singular operator (V0403).
    """
    if ms.equations != "scalar advection-diffusion":
        msg = f"the reference solver handles scalar advection-diffusion, not {ms.equations}"
        raise VVError(msg, code="V0401")
    if precision not in PRECISIONS:
        msg = f"precision must be one of {sorted(PRECISIONS)}, not {precision!r}"
        raise VVError(msg, code="V0404")
    real = PRECISIONS[precision]
    tol = 64 * float(np.finfo(real).eps) if rtol is None else rtol
    p = ms.parameters
    a = np.array([p["ax"], p["ay"], p["az"]])
    nu = p["nu"]
    if not nu > 0.0:
        msg = f"diffusivity must be positive, not {nu}"
        raise VVError(msg, code="V0403")
    g = mesh.geometry
    nc, ni, nf = mesh.n_cells, mesh.n_interior, mesh.n_faces
    own, nb = mesh.owner, mesh.neighbour
    bf = np.arange(ni, nf)
    exact = ms.evaluator("fields")
    phi_b = exact(g.face_centroid[bf])["phi"]
    source = ms.evaluator("sources")(g.cell_centroid)["phi"] * g.cell_volume

    (gx, gy, gz), g0 = _lsq_gradient(mesh, phi_b)
    sel_own = sps.csr_matrix((np.ones(ni), (np.arange(ni), own[:ni])), shape=(ni, nc))
    sel_nb = sps.csr_matrix((np.ones(ni), (np.arange(ni), nb)), shape=(ni, nc))
    area = g.face_area

    # Convection, interior faces: phi_U + grad_U.(x - x_U) integrated over the face's fan triangles,
    # = mdot phi_U + grad_U.q with q = M_f a - mdot x_U, exact for linear fields on warped faces.
    mdot = area[:ni] @ a
    up = np.where(mdot >= 0.0, own[:ni], nb)
    sel_up = sps.csr_matrix((np.ones(ni), (np.arange(ni), up)), shape=(ni, nc))
    q = g.face_moment[:ni] @ a - mdot[:, None] * g.cell_centroid[up]
    grad_up = [sel_up @ gx, sel_up @ gy, sel_up @ gz]
    conv_rows = sps.diags(mdot) @ sel_up + sum(sps.diags(q[:, k]) @ grad_up[k] for k in range(3))
    conv_const = np.einsum("ij,ij->i", q, g0[up])

    # Diffusion, interior faces: alpha-damping face gradient dotted with the area vector.
    d = g.cell_centroid[nb] - g.cell_centroid[own[:ni]]
    w = ALPHA * np.einsum("ij,ij->i", d, area[:ni]) / np.einsum("ij,ij->i", d, d)
    mean_grad = [0.5 * (sel_own + sel_nb) @ gk for gk in (gx, gy, gz)]
    coef = area[:ni] - w[:, None] * d
    diff_rows = sum(sps.diags(coef[:, k]) @ mean_grad[k] for k in range(3)) + sps.diags(w) @ (
        sel_nb - sel_own
    )
    mean_g0 = 0.5 * (g0[own[:ni]] + g0[nb])
    diff_const = np.einsum("ij,ij->i", coef, mean_g0)

    flux_rows = conv_rows - nu * diff_rows
    flux_const = conv_const - nu * diff_const
    div_int = sps.csr_matrix(
        (
            np.r_[np.ones(ni), -np.ones(ni)],
            (np.r_[own[:ni], nb], np.r_[np.arange(ni), np.arange(ni)]),
        ),
        shape=(nc, ni),
    )
    matrix = div_int @ flux_rows
    rhs = source - div_int @ flux_const

    # Boundary faces: exact Dirichlet value for convection; alpha-damping with the boundary value.
    ob = own[bf]
    db = g.face_centroid[bf] - g.cell_centroid[ob]
    wb = ALPHA_BOUNDARY * np.einsum("ij,ij->i", db, area[bf]) / np.einsum("ij,ij->i", db, db)
    coef_b = area[bf] - wb[:, None] * db
    sel_b = sps.csr_matrix((np.ones(bf.size), (np.arange(bf.size), ob)), shape=(bf.size, nc))
    diff_b = sum(sps.diags(coef_b[:, k]) @ (sel_b @ gk) for k, gk in enumerate((gx, gy, gz)))
    diff_b = diff_b - sps.diags(wb) @ sel_b
    diff_b_const = np.einsum("ij,ij->i", coef_b, g0[ob]) + wb * phi_b
    div_b = sps.csr_matrix((np.ones(bf.size), (ob, np.arange(bf.size))), shape=(nc, bf.size))
    matrix = (matrix - nu * (div_b @ diff_b)).tocsc()
    rhs = rhs - div_b @ ((area[bf] @ a) * phi_b - nu * diff_b_const)

    # Preconditioner: first-order upwind convection and two-point diffusion (compact stencil).
    up_coef = np.maximum(mdot, 0.0)
    dn_coef = np.minimum(mdot, 0.0)
    tp = nu * np.abs(w)
    rows = np.r_[own[:ni], own[:ni], nb, nb]
    cols = np.r_[own[:ni], nb, own[:ni], nb]
    vals = np.r_[up_coef + tp, dn_coef - tp, -up_coef - tp, -dn_coef + tp]
    diag_b = np.zeros(nc)
    np.add.at(diag_b, ob, np.maximum(area[bf] @ a, 0.0) + nu * np.abs(wb))
    low = sps.csc_matrix(
        (np.r_[vals, diag_b], (np.r_[rows, np.arange(nc)], np.r_[cols, np.arange(nc)])),
        shape=(nc, nc),
    )
    matrix = matrix.astype(real)
    rhs = rhs.astype(real)
    lu = spla.splu(low.astype(real))
    precond = spla.LinearOperator((nc, nc), matvec=lu.solve)
    iterations = 0

    def count(_: object) -> None:
        nonlocal iterations
        iterations += 1

    phi, info = spla.gmres(
        matrix,
        rhs,
        M=precond,
        rtol=tol,
        atol=0.0,
        restart=200,
        maxiter=50,
        callback=count,
        callback_type="pr_norm",
    )
    if info != 0:
        msg = f"GMRES did not converge (info={info})"
        raise VVError(msg, code="V0402")
    phi64 = phi.astype(np.float64)
    residual = float(np.linalg.norm(matrix.astype(np.float64) @ phi64 - rhs) / np.linalg.norm(rhs))
    err = phi64 - exact(g.cell_centroid)["phi"]
    vol = g.cell_volume
    return Solution(
        phi=phi64,
        error_l1=float(np.sum(vol * np.abs(err)) / vol.sum()),
        error_l2=float(np.sqrt(np.sum(vol * err**2) / vol.sum())),
        error_linf=float(np.max(np.abs(err))),
        residual_norm=residual,
        iterations=iterations,
    )
