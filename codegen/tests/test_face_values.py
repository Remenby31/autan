"""Per-face values of a face loop (opt-in `--face-values`, #43): the body's outputs on each face,
before any accumulation. Signed and accumulated per cell in incidence order, they are the residual
of the gather strategies bit for bit, so a conservation check can sum exactly what the loop sums."""

import autan_spike as s
import meshes
import numpy as np
import pytest
from conftest import KINDS, PRECISIONS

VELOCITY = (1.0, 0.5, -0.25)


@pytest.mark.parametrize("policy", PRECISIONS)
@pytest.mark.parametrize("kind", KINDS)
def test_signed_face_values_accumulate_to_the_residual(kind: str, policy: str) -> None:
    m = s.mesh(meshes.build(kind, 6, 0.15, seed=9))
    rng = np.random.default_rng(1)
    q = rng.standard_normal(m.n_cells)
    grad = rng.standard_normal((m.n_cells, 3))
    values = s.advection_face_values(m, policy, q, grad, VELOCITY)
    residual = s.advection(m, policy, q, grad, VELOCITY, "store-gather")
    accum = np.float32 if policy == "fp32" else np.float64
    for c in range(m.n_owned):
        acc = accum(0)
        for k in range(int(m.cell_face_offsets[c]), int(m.cell_face_offsets[c + 1])):
            f = int(m.cell_faces[k])
            if f >= m.n_interior:
                continue  # the advection residual is one face loop, over interior faces
            v = accum(values[f])
            acc = acc + v if int(m.owner[f]) == c else acc - v
        assert float(acc) == residual[c], (c, float(acc), residual[c])
