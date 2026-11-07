"""Geometry against the nodes (ADR 0003, check 2): the stored values the kernels read, and the
per-instance hypot check of the face area magnitudes, with no assumption on the library."""

import autan_spike as s
import meshes
import pytest
from oracle import certified as c

CASES = [(k, p) for k in ("hex", "tet", "hybrid", "poly") for p in (0.0, 0.15)]
HYPOT_K = 2  # eps = HYPOT_K u: the tolerance declared in the freeze


@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_stored_face_magnitudes_pass_the_exact_hypot_check(kind: str, perturbation: float) -> None:
    # The magnitude the kernels read, against the rational sum of squares of the stored area
    # components it was computed from: certified within (1 +- eps) of their root, or invalid.
    m = s.mesh(meshes.build(kind, 4, perturbation, seed=3))
    magnitude = s.face_area_magnitude(m)
    assert magnitude.shape == (m.n_faces,)
    for f in range(m.n_faces):
        area = tuple(float(v) for v in m.face_area[f])
        c.hypot_checked(
            area, tuple(c.exact(a) for a in area), float(magnitude[f]), HYPOT_K, c.BINARY64
        )
