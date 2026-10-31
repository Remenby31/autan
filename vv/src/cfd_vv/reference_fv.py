"""Reference cell-centred finite-volume solve of steady scalar advection-diffusion, in FP64.

It exists to prove the order tool and the mesh families end to end, independently of the C++
solver, and to serve as a second-order baseline its results can be compared with. Scheme:

- gradients: weighted least squares over face neighbours and boundary face centroids, weights
  1/|d|^2 (Mavriplis, AIAA 2003-3986);
- convection: upwind face value reconstructed with the upwind cell gradient; exact Dirichlet value
  on boundary faces;
- diffusion: alpha-damping face gradient, grad_f = mean(grad) + alpha (dphi - mean(grad).d)/(d.n) n,
  alpha = 4/3 (Nishikawa, AIAA 2010-5093);
- source: the manufactured source at the cell centroid times the cell volume.

All Dirichlet data come from the manufactured solution. The operator is affine in phi; it is
assembled as a sparse matrix and solved with GMRES preconditioned by the LU factors of the
compact first-order operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from cfd_vv.errors import VVError

if TYPE_CHECKING:
    from cfd_vv.meshes.facemesh import FaceMesh, FloatArray
    from cfd_vv.mms.solutions import ManufacturedSolution

ALPHA = 4.0 / 3.0


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


def solve(mesh: FaceMesh, ms: ManufacturedSolution, rtol: float = 1e-12) -> Solution:
    """Solve the advection-diffusion problem `ms` (from `cfd_vv.mms.advection_diffusion`) on `mesh`."""
    if ms.equations != "scalar advection-diffusion":
        msg = f"the reference solver handles scalar advection-diffusion, not {ms.equations}"
        raise VVError(msg, code="V0401")
    p = ms.parameters
    a = np.array([p["ax"], p["ay"], p["az"]])
    nu = p["nu"]
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
    mag = np.linalg.norm(area, axis=1)
    n_hat = area / mag[:, None]

    # Convection, interior faces: upwind value phi_U + grad_U . (x_f - x_U).
    mdot = area[:ni] @ a
    up_is_owner = mdot >= 0.0
    up = np.where(up_is_owner, own[:ni], nb)
    sel_up = sps.csr_matrix((np.ones(ni), (np.arange(ni), up)), shape=(ni, nc))
    r = g.face_centroid[:ni] - g.cell_centroid[up]
    grad_up = [sel_up @ gx, sel_up @ gy, sel_up @ gz]
    conv = sel_up + sum(sps.diags(r[:, k]) @ grad_up[k] for k in range(3))
    conv_rows = sps.diags(mdot) @ conv
    conv_const = mdot * np.einsum("ij,ij->i", r, g0[up])

    # Diffusion, interior faces: alpha-damping face gradient dotted with the area vector.
    d = g.cell_centroid[nb] - g.cell_centroid[own[:ni]]
    dn = np.einsum("ij,ij->i", d, n_hat[:ni])
    mean_grad = [0.5 * (sel_own + sel_nb) @ gk for gk in (gx, gy, gz)]
    coef = area[:ni] - (ALPHA * mag[:ni] / dn)[:, None] * d
    diff_rows = sum(sps.diags(coef[:, k]) @ mean_grad[k] for k in range(3)) + sps.diags(
        ALPHA * mag[:ni] / dn
    ) @ (sel_nb - sel_own)
    mean_g0 = 0.5 * (g0[own[:ni]] + g0[nb])
    diff_const = np.einsum("ij,ij->i", coef, mean_g0)

    flux_rows = conv_rows - nu * diff_rows
    flux_const = conv_const - nu * diff_const
    div_int = sps.csr_matrix(
        (np.r_[np.ones(ni), -np.ones(ni)], (np.r_[own[:ni], nb], np.r_[np.arange(ni), np.arange(ni)])),
        shape=(nc, ni),
    )
    matrix = div_int @ flux_rows
    rhs = source - div_int @ flux_const

    # Boundary faces: exact Dirichlet value for convection; alpha-damping with the boundary value.
    ob = own[bf]
    db = g.face_centroid[bf] - g.cell_centroid[ob]
    dnb = np.einsum("ij,ij->i", db, n_hat[bf])
    coef_b = area[bf] - (ALPHA * mag[bf] / dnb)[:, None] * db
    sel_b = sps.csr_matrix((np.ones(bf.size), (np.arange(bf.size), ob)), shape=(bf.size, nc))
    diff_b = sum(sps.diags(coef_b[:, k]) @ (sel_b @ gk) for k, gk in enumerate((gx, gy, gz)))
    diff_b = diff_b - sps.diags(ALPHA * mag[bf] / dnb) @ sel_b
    diff_b_const = np.einsum("ij,ij->i", coef_b, g0[ob]) + ALPHA * mag[bf] / dnb * phi_b
    div_b = sps.csr_matrix((np.ones(bf.size), (ob, np.arange(bf.size))), shape=(nc, bf.size))
    matrix = (matrix - nu * (div_b @ diff_b)).tocsc()
    rhs = rhs - div_b @ ((area[bf] @ a) * phi_b - nu * diff_b_const)

    # Preconditioner: first-order upwind convection and two-point diffusion (compact stencil).
    up_coef = np.maximum(mdot, 0.0)
    dn_coef = np.minimum(mdot, 0.0)
    tp = nu * mag[:ni] / dn
    rows = np.r_[own[:ni], own[:ni], nb, nb]
    cols = np.r_[own[:ni], nb, own[:ni], nb]
    vals = np.r_[up_coef + tp, dn_coef - tp, -up_coef - tp, -dn_coef + tp]
    diag_b = np.zeros(nc)
    np.add.at(diag_b, ob, np.maximum(area[bf] @ a, 0.0) + nu * mag[bf] / dnb)
    low = sps.csc_matrix((np.r_[vals, diag_b], (np.r_[rows, np.arange(nc)], np.r_[cols, np.arange(nc)])),
                         shape=(nc, nc))
    lu = spla.splu(low)
    precond = spla.LinearOperator((nc, nc), matvec=lu.solve)
    iterations = 0

    def count(_: object) -> None:
        nonlocal iterations
        iterations += 1

    phi, info = spla.gmres(matrix, rhs, M=precond, rtol=rtol, atol=0.0, restart=200, maxiter=50,
                           callback=count, callback_type="pr_norm")
    if info != 0:
        msg = f"GMRES did not converge (info={info})"
        raise VVError(msg, code="V0402")
    residual = float(np.linalg.norm(matrix @ phi - rhs) / np.linalg.norm(rhs))
    err = phi - exact(g.cell_centroid)["phi"]
    vol = g.cell_volume
    return Solution(
        phi=phi,
        error_l1=float(np.sum(vol * np.abs(err)) / vol.sum()),
        error_l2=float(np.sqrt(np.sum(vol * err**2) / vol.sum())),
        error_linf=float(np.max(np.abs(err))),
        residual_norm=residual,
        iterations=iterations,
    )
