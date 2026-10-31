"""ctypes bindings to the spike library (codegen/spike/capi.cpp): meshes through the core and the
generated host loops. The library path comes from AUTAN_SPIKE_LIB."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from meshes import FaceArrays

F64 = npt.NDArray[np.float64]


class SpikeError(Exception):
    def __init__(self, message: str) -> None:
        self.code = message.split(":", 1)[0]
        super().__init__(message)


@cache
def lib() -> ctypes.CDLL:
    return ctypes.CDLL(os.environ["AUTAN_SPIKE_LIB"])


def _p(a: np.ndarray) -> ctypes.c_void_p:  # type: ignore[type-arg]
    assert a.flags.c_contiguous
    return ctypes.c_void_p(a.ctypes.data)


def _check(rc: int, err: ctypes.Array[ctypes.c_char]) -> None:
    if rc != 0:
        raise SpikeError(err.value.decode())


@dataclass
class Mesh:
    handle: ctypes.c_void_p
    n_cells: int
    n_faces: int
    n_interior: int
    n_incidences: int
    owner: npt.NDArray[np.int32]
    neighbour: npt.NDArray[np.int32]
    cell_face_offsets: npt.NDArray[np.int64]
    cell_faces: npt.NDArray[np.int32]
    face_area: F64  # (n_faces, 3)
    face_centroid: F64
    face_moment: F64  # (n_faces, 3, 3)
    cell_centroid: F64
    cell_volume: F64
    lsq_weight: F64

    def __del__(self) -> None:
        if self.handle:
            lib().autan_spike_mesh_free(self.handle)

    @property
    def h(self) -> float:
        return float((self.cell_volume.sum() / self.n_cells) ** (1.0 / 3.0))


def mesh(fa: FaceArrays) -> Mesh:
    err = ctypes.create_string_buffer(512)
    h = ctypes.c_void_p()
    xyz = np.ascontiguousarray(fa.points, dtype=np.float64)
    rc = lib().autan_spike_mesh_from_faces(
        ctypes.c_int64(len(xyz)), _p(xyz), ctypes.c_int64(len(fa.cell0)), _p(fa.face_offsets),
        _p(fa.face_nodes), _p(fa.cell0), _p(fa.cell1), ctypes.c_int32(fa.n_cells),
        ctypes.byref(h), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    counts = np.zeros(5, dtype=np.int64)
    lib().autan_spike_mesh_counts(h, _p(counts))
    nc, _, nf, ni, nk = (int(c) for c in counts)
    a = {
        "owner": np.zeros(nf, np.int32),
        "neighbour": np.zeros(ni, np.int32),
        "cell_face_offsets": np.zeros(nc + 1, np.int64),
        "cell_faces": np.zeros(nk, np.int32),
        "face_area": np.zeros(3 * nf),
        "face_centroid": np.zeros(3 * nf),
        "face_moment": np.zeros(9 * nf),
        "cell_centroid": np.zeros(3 * nc),
        "cell_volume": np.zeros(nc),
        "lsq_weight": np.zeros(3 * nk),
    }
    lib().autan_spike_mesh_arrays(h, *(_p(v) for v in a.values()))
    soa = {k: a[k].reshape(3, n).T.copy() for k, n in
           (("face_area", nf), ("face_centroid", nf), ("cell_centroid", nc), ("lsq_weight", nk))}  # fmt: skip
    moment = a["face_moment"].reshape(9, nf).T.reshape(nf, 3, 3).copy()
    return Mesh(h, nc, nf, ni, nk, a["owner"], a["neighbour"], a["cell_face_offsets"], a["cell_faces"],
                soa["face_area"], soa["face_centroid"], moment, soa["cell_centroid"], a["cell_volume"],
                soa["lsq_weight"])  # fmt: skip


def soa(v: F64) -> F64:
    """(n, k) -> SoA flat array (component-major), the layout of generated datasets."""
    return np.ascontiguousarray(v.T).ravel()


def advdiff_residual(
    m: Mesh,
    policy: str,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
) -> tuple[F64, F64]:
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    grad = np.zeros(3 * m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source))
    rc = lib().autan_spike_advdiff_residual(
        m.handle, policy.encode(), _p(q), _p(q_b), _p(source), _p(vel), ctypes.c_double(nu),
        _p(res), _p(grad), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res, grad.reshape(3, -1).T


def gradients(
    m: Mesh,
    policy: str,
    method: str,
    q: F64,
    q_b: F64,
    iterations: int = 1,
    grad0: F64 | None = None,
) -> tuple[F64, F64, F64]:
    """LSQ gradient and bounds, or `iterations` of corrected Green-Gauss starting from `grad0`."""
    err = ctypes.create_string_buffer(512)
    grad = np.zeros(3 * m.n_cells) if grad0 is None else soa(grad0)
    qmin = np.zeros(m.n_cells)
    qmax = np.zeros(m.n_cells)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    rc = lib().autan_spike_gradients(
        m.handle, policy.encode(), method.encode(), _p(q), _p(q_b), ctypes.c_int(iterations),
        _p(grad), _p(qmin), _p(qmax), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return grad.reshape(3, -1).T, qmin, qmax


def advection(m: Mesh, policy: str, q: F64, grad: F64, velocity: tuple[float, float, float]) -> F64:
    """Residual of the section 3.3 upwind face loop on the host."""
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, g = np.ascontiguousarray(q, dtype=np.float64), soa(grad)
    rc = lib().autan_spike_advection(
        m.handle, policy.encode(), _p(q), _p(g), _p(vel), _p(res), err, ctypes.c_size_t(len(err))
    )
    _check(rc, err)
    return res


STRATEGIES = ("atomic", "warp-aggregated", "coloured", "store-gather", "gather-recompute")
DETERMINISTIC = ("coloured", "store-gather", "gather-recompute")


def gpu_advdiff_residual(
    m: Mesh,
    policy: str,
    strategy: str,
    block: int,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
) -> tuple[F64, F64]:
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    grad = np.zeros(3 * m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source))
    rc = lib().autan_spike_gpu_advdiff_residual(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b), _p(source),
        _p(vel), ctypes.c_double(nu), _p(res), _p(grad), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res, grad.reshape(3, -1).T


def gpu_gradients(
    m: Mesh,
    policy: str,
    method: str,
    strategy: str,
    block: int,
    q: F64,
    q_b: F64,
    iterations: int = 1,
    grad0: F64 | None = None,
) -> tuple[F64, F64, F64]:
    err = ctypes.create_string_buffer(512)
    grad = np.zeros(3 * m.n_cells) if grad0 is None else soa(grad0)
    qmin = np.zeros(m.n_cells)
    qmax = np.zeros(m.n_cells)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    rc = lib().autan_spike_gpu_gradients(
        m.handle, policy.encode(), method.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(q_b),
        ctypes.c_int(iterations), _p(grad), _p(qmin), _p(qmax), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return grad.reshape(3, -1).T, qmin, qmax


def gpu_advection(
    m: Mesh,
    policy: str,
    strategy: str,
    block: int,
    q: F64,
    grad: F64,
    velocity: tuple[float, float, float],
) -> F64:
    err = ctypes.create_string_buffer(512)
    res = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, g = np.ascontiguousarray(q, dtype=np.float64), soa(grad)
    rc = lib().autan_spike_gpu_advection(
        m.handle, policy.encode(), strategy.encode(), ctypes.c_int(block), _p(q), _p(g), _p(vel), _p(res),
        err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return res


def advdiff_derivative(
    m: Mesh,
    policy: str,
    mode: str,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
    seed: F64,
) -> F64:
    """mode "tangent": dR for dq = seed; "adjoint": q-bar for R-bar = seed (dR/dq only)."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source, seed = (
        np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source, seed)
    )
    rc = lib().autan_spike_advdiff_derivative(
        m.handle, policy.encode(), ctypes.c_int(0 if mode == "tangent" else 1), _p(q), _p(q_b), _p(source),
        _p(vel), ctypes.c_double(nu), _p(seed), _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def advdiff_complex_step(
    m: Mesh,
    q: F64,
    q_b: F64,
    source: F64,
    velocity: tuple[float, float, float],
    nu: float,
    dq: F64,
    h: float = 1e-30,
) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(m.n_cells)
    vel = np.array(velocity, dtype=np.float64)
    q, q_b, source, dq = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b, source, dq))
    rc = lib().autan_spike_advdiff_complex_step(
        m.handle, _p(q), _p(q_b), _p(source), _p(vel), ctypes.c_double(nu), _p(dq), ctypes.c_double(h),
        _p(out), err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out


def roe(m: Mesh, policy: str, mode: str, u: F64, seed: F64 | None = None) -> F64:
    """Roe face loop on U (n, 5): residual, tangent (seed dU) or adjoint (seed R-bar); (n, 5)."""
    err = ctypes.create_string_buffer(512)
    out = np.zeros(5 * m.n_cells)
    us = soa(u)
    sd = soa(seed) if seed is not None else np.zeros(5 * m.n_cells)
    code = {"residual": 0, "tangent": 1, "adjoint": 2}[mode]
    rc = lib().autan_spike_roe(
        m.handle,
        policy.encode(),
        ctypes.c_int(code),
        _p(us),
        _p(sd),
        _p(out),
        err,
        ctypes.c_size_t(len(err)),
    )
    _check(rc, err)
    return out.reshape(5, -1).T


def roe_complex_step(m: Mesh, u: F64, du: F64, h: float = 1e-30) -> F64:
    err = ctypes.create_string_buffer(512)
    out = np.zeros(5 * m.n_cells)
    us, ds = soa(u), soa(du)
    rc = lib().autan_spike_roe_complex_step(
        m.handle, _p(us), _p(ds), ctypes.c_double(h), _p(out), err, ctypes.c_size_t(len(err))
    )
    _check(rc, err)
    return out.reshape(5, -1).T


def lsq_bounds_derivative(m: Mesh, policy: str, mode: str, q: F64, q_b: F64, seed: F64) -> F64:
    """Tangent: seed dq (n,) -> (n, 5) = (dgrad, dqmin, dqmax). Adjoint: seed (n, 5) -> q-bar (n,)."""
    err = ctypes.create_string_buffer(512)
    q, q_b = (np.ascontiguousarray(x, dtype=np.float64) for x in (q, q_b))
    if mode == "tangent":
        sd, out = np.ascontiguousarray(seed, dtype=np.float64), np.zeros(5 * m.n_cells)
    else:
        sd, out = soa(seed), np.zeros(m.n_cells)
    rc = lib().autan_spike_lsq_bounds_derivative(
        m.handle, policy.encode(), ctypes.c_int(1 if mode == "tangent" else 2), _p(q), _p(q_b), _p(sd), _p(out),
        err, ctypes.c_size_t(len(err)),
    )  # fmt: skip
    _check(rc, err)
    return out.reshape(5, -1).T if mode == "tangent" else out
