"""Hash the store-gather outputs of every spike face loop, per mesh kind and policy.

AUTAN_SPIKE_LIB=.../libautan_spike_metal.dylib PYTHONPATH=codegen/spike \
    uv run --project codegen python codegen/spike/experiments/178/store_gather_hashes.py
Two builds print the same lines if and only if their outputs are byte-identical.
"""

from __future__ import annotations

import hashlib
import sys

import autan_spike as s
import meshes
import mms
import numpy as np

SG = "store-gather"


def main() -> None:
    for kind in ("tet", "hex", "poly", "hybrid"):
        m = s.mesh(meshes.build(kind, 8, 0.15, seed=11))
        rng = np.random.default_rng(3)
        q = rng.standard_normal(m.n_cells)
        qb = rng.standard_normal(m.n_faces)
        src = rng.standard_normal(m.n_cells)
        g = rng.standard_normal((m.n_cells, 3))
        rho = 1.0 + 0.2 * rng.random(m.n_cells)
        vel = 0.3 * rng.standard_normal((m.n_cells, 3))
        u = np.column_stack([rho, rho[:, None] * vel, 2.5 + 0.5 * rho * np.sum(vel * vel, axis=1)])
        seed = rng.standard_normal(u.shape)
        y = rng.standard_normal(m.n_cells)
        for pol in ("fp32", "fp32+acc64"):
            out = [
                s.gpu_advdiff_residual(m, pol, SG, 128, q, qb, src, mms.VELOCITY, mms.NU)[0],
                s.gpu_advection(m, pol, SG, 128, q, g, mms.VELOCITY),
                s.gpu_gradients(m, pol, "gg", SG, 128, q, qb, 3, g)[0],
                s.gpu_advdiff_adjoint(m, pol, SG, 128, q, qb, src, mms.VELOCITY, mms.NU, y),
            ]
            out += [
                s.gpu_roe(m, pol, SG, 128, mode, u, seed)
                for mode in ("residual", "tangent", "adjoint")
            ]
            digest = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in out))
            sys.stdout.write(f"{kind} {pol} {digest.hexdigest()[:16]}\n")


if __name__ == "__main__":
    main()
