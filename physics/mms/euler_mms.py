"""Order of accuracy of the Euler residual against the `euler` manufactured solution of autan-vv.

uv run --project vv python physics/mms/euler_mms.py FAMILIES OUT [--policies ...] [--fluxes ...]

FAMILIES holds one directory per mesh family as `autan-vv meshes` writes it. Every level is read by
the core readers, the steady discrete solution is found by Newton-Krylov on the generated residual
(boundary faces take the exact state, cells the exact source), and the volume-weighted errors of
the conserved variables at cell centroids are written per (family, policy, flux) in the input
format of `autan-vv order`, with mesh hash, solver commit and precision. The library path comes from
AUTAN_PHYSICS_MMS_LIB.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from autan_vv.mms.export import to_json
from autan_vv.mms.solutions import euler
from autan_vv.order import fit_order, mesh_size

F64 = npt.NDArray[np.float64]
FLUXES = {"roe": 0, "hllc": 1, "ausm-up": 2}
STENCILS = {"face2": 0, "vertex": 1}
CONSERVED = ("rho", "rhou", "rhov", "rhow", "rhoE")
PRIMITIVE = ("rho", "u", "v", "w", "p")


@cache
def lib() -> ctypes.CDLL:
    return ctypes.CDLL(os.environ["AUTAN_PHYSICS_MMS_LIB"])


def _p(a: np.ndarray) -> ctypes.c_void_p:  # type: ignore[type-arg]
    assert a.flags.c_contiguous
    return ctypes.c_void_p(a.ctypes.data)


def _check(rc: int, err: ctypes.Array[ctypes.c_char]) -> None:
    if rc != 0:
        raise RuntimeError(err.value.decode())


@dataclass
class Mesh:
    handle: ctypes.c_void_p
    n_cells: int
    n_faces: int
    n_interior: int
    owner: npt.NDArray[np.int32]
    neighbour: npt.NDArray[np.int32]
    cell_centroid: F64
    cell_volume: F64
    face_centroid: F64
    face_normal: F64

    def __del__(self) -> None:
        lib().autan_mms_mesh_free(self.handle)


def read(path: Path) -> Mesh:
    err = ctypes.create_string_buffer(512)
    h = ctypes.c_void_p()
    _check(lib().autan_mms_mesh_read(str(path).encode(), ctypes.byref(h), err, len(err)), err)
    counts = np.zeros(3, dtype=np.int64)
    lib().autan_mms_mesh_counts(h, _p(counts))
    nc, nf, ni = (int(c) for c in counts)
    owner = np.zeros(nf, dtype=np.int32)
    neighbour = np.zeros(ni, dtype=np.int32)
    cc = np.zeros((nc, 3))
    cv = np.zeros(nc)
    fc = np.zeros((nf, 3))
    fn = np.zeros((nf, 3))
    lib().autan_mms_mesh_arrays(h, _p(owner), _p(neighbour), _p(cc), _p(cv), _p(fc), _p(fn))
    return Mesh(h, nc, nf, ni, owner, neighbour, cc, cv, fc, fn)


class Euler:
    """The residual of autan::physics::Euler on the host; arrays are 5 x n_cells."""

    def __init__(
        self,
        m: Mesh,
        policy: str,
        flux: str,
        limiter: bool,
        mach_ref: float,
        reference: F64,
        stencil: str = "face2",
        venkat_k: float = 5.0,
        neighbour_boundary_faces: bool = True,
    ) -> None:
        self.n = m.n_cells
        err = ctypes.create_string_buffer(512)
        h = ctypes.c_void_p()
        scheme = np.array(
            [FLUXES[flux], 1, int(limiter), STENCILS[stencil], int(neighbour_boundary_faces)],
            dtype=np.int32,
        )
        gamma, rgas = 1.4, 287.0
        real = np.concatenate(
            [[gamma, rgas, venkat_k, 1.0, mach_ref, 1.0, 100.0, 100.0, 100.0, 1e5], reference]
        )
        _check(
            lib().autan_mms_euler_create(
                m.handle, policy.encode(), _p(scheme), _p(real), ctypes.byref(h), err, len(err)
            ),
            err,
        )
        self.handle = h
        self.mesh = m  # the native residual borrows the mesh's topology

    def __del__(self) -> None:
        lib().autan_mms_euler_free(self.handle)

    def set_boundary(self, kind: npt.NDArray[np.uint8], data: F64) -> None:
        err = ctypes.create_string_buffer(512)
        data = np.ascontiguousarray(data)
        _check(
            lib().autan_mms_euler_set_boundary(
                self.handle, _p(kind), _p(data), ctypes.c_int64(kind.size), err, len(err)
            ),
            err,
        )

    def set_source(self, s: F64) -> None:
        err = ctypes.create_string_buffer(512)
        s = np.ascontiguousarray(s)
        _check(lib().autan_mms_euler_set_source(self.handle, _p(s), err, len(err)), err)

    def residual(self, u: F64) -> F64:
        err = ctypes.create_string_buffer(512)
        u = np.ascontiguousarray(u)
        r = np.zeros_like(u)
        _check(lib().autan_mms_euler_residual(self.handle, _p(u), _p(r), err, len(err)), err)
        return r

    def first_order_tangent(self, u: F64, du: F64) -> F64:
        err = ctypes.create_string_buffer(512)
        u, du = np.ascontiguousarray(u), np.ascontiguousarray(du)
        r = np.zeros_like(u)
        _check(
            lib().autan_mms_euler_first_order_tangent(
                self.handle, _p(u), _p(du), _p(r), err, len(err)
            ),
            err,
        )
        return r


def distance2_colouring(m: Mesh) -> list[npt.NDArray[np.int64]]:
    """Greedy colouring of cells such that no two cells of a colour share a face neighbour."""
    adj: list[list[int]] = [[] for _ in range(m.n_cells)]
    for o, n in zip(m.owner[: m.n_interior].tolist(), m.neighbour.tolist(), strict=True):
        adj[o].append(n)
        adj[n].append(o)
    colour = np.full(m.n_cells, -1, dtype=np.int64)
    for c in range(m.n_cells):
        used = set()
        for n in adj[c]:
            used.add(colour[n])
            for nn in adj[n]:
                used.add(colour[nn])
        k = 0
        while k in used:
            k += 1
        colour[c] = k
    return [np.flatnonzero(colour == k) for k in range(int(colour.max()) + 1)]


def first_order_jacobian(
    e: Euler, m: Mesh, u: F64, colours: list[npt.NDArray[np.int64]]
) -> sp.csr_matrix:
    """dR1/dU in cell-major ordering (row 5 i + k), from the generated tangent probed by colour."""
    nc = m.n_cells
    o, nb = m.owner[: m.n_interior].astype(np.int64), m.neighbour.astype(np.int64)
    # Owner of each probed cell for every row it touches: itself and its face neighbours.
    pair_row = np.concatenate([np.arange(nc), o, nb])
    pair_col = np.concatenate([np.arange(nc), nb, o])
    colour_of = np.empty(nc, dtype=np.int64)
    for k, cells in enumerate(colours):
        colour_of[cells] = k
    rows, cols, vals = [], [], []
    for k, cells in enumerate(colours):
        sel = colour_of[pair_col] == k
        pr, pc = pair_row[sel], pair_col[sel]
        for j in range(5):
            du = np.zeros((5, nc))
            du[j, cells] = 1.0
            dr = e.first_order_tangent(u, du)
            for i in range(5):
                rows.append(5 * pr + i)
                cols.append(5 * pc + j)
                vals.append(dr[i, pr])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(5 * nc, 5 * nc),
    )


@dataclass
class Solution:
    u: F64
    relative_residual: float
    newton: int
    gmres: int


def solve(
    m: Mesh,
    policy: str,
    flux: str,
    limiter: bool,
    tol: float,
    stencil: str = "face2",
    venkat_k: float = 5.0,
    neighbour_boundary_faces: bool = True,
) -> Solution:
    """Newton-Krylov for R_P(u) = 0 from the exact solution; Jacobian-vector products by FD on
    the FP64 residual, preconditioned by ILU of the first-order Jacobian, so that the converged u
    is the discrete solution of policy P."""
    ms = euler()
    ex_f = ms.evaluator("fields")
    ex_c = ms.evaluator("conserved")
    ex_s = ms.evaluator("sources")
    b = slice(m.n_interior, m.n_faces)
    wb = ex_f(m.face_centroid[b])
    cc = ex_c(m.cell_centroid)
    u = np.array([cc[k] for k in CONSERVED])
    sc = ex_s(m.cell_centroid)
    src = np.array([sc[k] for k in CONSERVED])
    speed = np.sqrt(
        np.mean(cc["rhou"] ** 2 + cc["rhov"] ** 2 + cc["rhow"] ** 2) / np.mean(cc["rho"]) ** 2
    )
    sound = np.sqrt(1.4 * np.mean(wb["p"]) / np.mean(wb["rho"]))
    mach = float(speed / sound)
    kind = np.zeros(m.n_faces - m.n_interior, dtype=np.uint8)
    data = np.array([wb[k] for k in PRIMITIVE])
    # The cells store differences from the mean state, as autan-run's from the freestream.
    reference = u.mean(axis=1)
    ep = Euler(
        m, policy, flux, limiter, mach, reference, stencil, venkat_k, neighbour_boundary_faces
    )
    e64 = (
        ep
        if policy == "fp64"
        else Euler(
            m, "fp64", flux, limiter, mach, reference, stencil, venkat_k, neighbour_boundary_faces
        )
    )
    # The preconditioner is the first-order Roe Jacobian whatever the flux: its ILU stays stable
    # where HLLC's switches between wave branches make pivots vanish.
    pre = Euler(
        m, "fp64", "roe", limiter, mach, reference, stencil, venkat_k, neighbour_boundary_faces
    )
    for e in {id(ep): ep, id(e64): e64, id(pre): pre}.values():
        e.set_boundary(kind, data)
        e.set_source(src)
    scale = np.sqrt(np.mean(u * u, axis=1))[:, None]
    colours = distance2_colouring(m)

    def to_x(a: F64) -> F64:  # component-major (5, n) / scale -> cell-major vector
        return np.ascontiguousarray((a / scale).T).ravel()

    def from_x(x: F64) -> F64:
        return x.reshape(-1, 5).T * scale

    r = ep.residual(u)
    r0 = float(np.linalg.norm(to_x(r)))
    best, it_gmres, newton = r0, 0, 0
    # The preconditioner is built once, at the exact solution the iteration starts from.
    jac = first_order_jacobian(pre, m, u, colours)
    d = np.repeat(scale[:, 0][None, :], m.n_cells, axis=0).ravel()
    js = sp.diags(1.0 / d) @ jac @ sp.diags(d)
    ilu = spla.spilu(js.tocsc(), drop_tol=1e-4, fill_factor=8)
    for newton in range(1, 31):  # noqa: B007 (reported after the loop)
        rx = to_x(r)
        r64 = e64.residual(u)
        ux = to_x(u)

        def matvec(v: F64, ux: F64 = ux, r64: F64 = r64) -> F64:
            nv = float(np.linalg.norm(v))
            if nv == 0.0:
                return np.zeros_like(v)
            h = 1e-7 * (1.0 + float(np.linalg.norm(ux)) / np.sqrt(ux.size)) / nv * np.sqrt(v.size)
            return (to_x(e64.residual(from_x(ux + h * v))) - to_x(r64)) / h

        n5 = 5 * m.n_cells
        jop = spla.LinearOperator((n5, n5), matvec=matvec)  # type: ignore[call-arg]
        pre = spla.LinearOperator((n5, n5), matvec=ilu.solve)  # type: ignore[call-arg]
        count = [0]

        def cb(_: object, count: list[int] = count) -> None:
            count[0] += 1

        dx, _info = spla.gmres(
            jop, -rx, M=pre, rtol=1e-3, atol=0.0, restart=60, maxiter=5, callback=cb,
            callback_type="pr_norm",
        )  # fmt: skip
        it_gmres += count[0]
        u = from_x(ux + dx)
        r = ep.residual(u)
        rn = float(np.linalg.norm(to_x(r)))
        if rn <= tol * r0:
            best = rn
            break
        if rn > 0.5 * best and best < 1e-3 * r0:
            best = min(best, rn)
            break  # rounding floor of the policy
        best = min(best, rn)
    return Solution(u, best / r0, newton, it_gmres)


def errors(m: Mesh, u: F64) -> dict[str, dict[str, float]]:
    cc = euler().evaluator("conserved")(m.cell_centroid)
    v = m.cell_volume
    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(CONSERVED):
        e = u[k] - cc[name]
        out[name] = {
            "L1": float(np.sum(np.abs(e) * v) / v.sum()),
            "L2": float(np.sqrt(np.sum(e * e * v) / v.sum())),
            "Linf": float(np.abs(e).max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("families", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--policies", default="fp64,fp32+acc64")
    ap.add_argument("--fluxes", default="roe,hllc,ausm-up")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--limiter", action="store_true")
    ap.add_argument("--stencil", default="face2", choices=sorted(STENCILS))
    ap.add_argument("--venkat-k", type=float, default=5.0)
    ap.add_argument(
        "--own-boundary-faces",
        action="store_true",
        help="gradient fits take only the cell's own boundary faces when that suffices",
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lib().autan_mms_commit.restype = ctypes.c_char_p
    commit = lib().autan_mms_commit().decode()  # the library's build, not the checkout's
    solution = json.dumps(to_json(euler()), sort_keys=True)
    kinds = a.kinds.split(",") if a.kinds else None
    for fam in sorted(d for d in a.families.iterdir() if (d / "manifest.json").exists()):
        if kinds and fam.name not in kinds:
            continue
        manifest = json.loads((fam / "manifest.json").read_text())
        meshes = [(lv, read(fam / lv["file"])) for lv in manifest["levels"]]
        for policy in a.policies.split(","):
            for flux in a.fluxes.split(","):
                config = {"solution": solution, "precision": policy, "flux": flux,
                          "limiter": a.limiter, "venkat_k": a.venkat_k, "stencil": a.stencil, "own_boundary_faces": a.own_boundary_faces}  # fmt: skip
                case_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                levels = []
                for lv, m in meshes:
                    t0 = time.perf_counter()
                    s = solve(
                        m,
                        policy,
                        flux,
                        a.limiter,
                        1e-10 if policy == "fp64" else 1e-5,
                        a.stencil,
                        a.venkat_k,
                        not a.own_boundary_faces,
                    )
                    levels.append({
                        "n_cells": m.n_cells, "volume": float(m.cell_volume.sum()),
                        "errors": errors(m, s.u), "mesh_sha256": lv["mesh_sha256"],
                        "solver_commit": commit, "case_sha256": case_sha,
                        "precision": policy, "flux": flux,
                        "limiter": a.limiter, "venkat_k": a.venkat_k, "stencil": a.stencil, "relative_residual": s.relative_residual,
                        "newton": s.newton, "gmres": s.gmres,
                        "seconds": time.perf_counter() - t0,
                    })  # fmt: skip
                label = f"{fam.name} {policy} {flux} {a.stencil}" + (
                    f" limited K={a.venkat_k:g}" if a.limiter else ""
                )
                name = label.replace(" ", "-").replace("+", "")
                # One study per conserved variable, in the input format of `autan-vv order`.
                for var in CONSERVED:
                    per_var = [{**lv, "errors": lv["errors"][var]} for lv in levels]
                    (a.out / f"{name}-{var}.json").write_text(
                        json.dumps(
                            {"label": f"{label} {var}", "dim": 3, "levels": per_var}, indent=1
                        )
                        + "\n"
                    )
                h = [mesh_size(lv["n_cells"], lv["volume"], 3) for lv in levels]
                row = [label]
                for var in CONSERVED:
                    f = fit_order(h, [lv["errors"][var]["L2"] for lv in levels])
                    row.append(f"{var} {f.order:.2f}/{f.pairwise_orders[-1]:.2f}")
                res = max(lv["relative_residual"] for lv in levels)
                sys.stdout.write(" | ".join(row) + f" | res {res:.1e}\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
