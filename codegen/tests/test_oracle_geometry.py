"""Geometry against the nodes (ADR 0003, check 2): the face area magnitudes the kernels read are
the declared sequence of libstdc++'s three-argument hypot (`sequences.hypot3`), replayed bit for bit
from the stored area components, with a certified radius around the exact root."""

import math
from fractions import Fraction as Fr

import autan_spike as s
import meshes
import numpy as np
import pytest
from oracle import certified as c
from oracle import sequences as q

CASES = [(k, p) for k in ("hex", "tet", "hybrid", "poly") for p in (0.0, 0.15)]


def covers(cert: c.Cert, exact_square: Fr, y: float) -> bool:
    lo = max(Fr(y) - cert.radius, Fr(0))
    hi = Fr(y) + cert.radius
    return lo * lo <= exact_square <= hi * hi


@pytest.mark.parametrize(("kind", "perturbation"), CASES)
def test_stored_magnitudes_are_the_declared_hypot_bit_for_bit(
    kind: str, perturbation: float
) -> None:
    m = s.mesh(meshes.build(kind, 4, perturbation, seed=3))
    magnitude = s.face_area_magnitude(m)
    assert magnitude.shape == (m.n_faces,)
    for f in range(m.n_faces):
        area = tuple(float(v) for v in m.face_area[f])
        assert q.hypot3_emulated(*area) == float(magnitude[f]), (kind, f)
        cert = q.hypot3_cert(tuple(c.exact(a) for a in area), area, float(magnitude[f]))
        assert covers(cert, sum(Fr(a) ** 2 for a in area), float(magnitude[f]))


def test_a_magnitude_shifted_by_one_ulp_is_invalid() -> None:
    m = s.mesh(meshes.build("tet", 4, 0.15, seed=3))
    magnitude = s.face_area_magnitude(m)
    area = tuple(float(v) for v in m.face_area[0])
    shifted = math.nextafter(float(magnitude[0]), math.inf)
    with pytest.raises(c.Invalid):
        q.hypot3_cert(tuple(c.exact(a) for a in area), area, shifted)


def test_the_radius_covers_the_exact_root_of_uncertain_arguments() -> None:
    rng = np.random.default_rng(29)
    for _ in range(2000):
        exact = [Fr(float(v)) for v in rng.standard_normal(3) * 10.0 ** rng.integers(-5, 5, 3)]
        certs = tuple(c.Cert(v, abs(v) / 2**40) for v in exact)
        executed = tuple(float(v * (1 + Fr(1, 2**41))) for v in exact)
        h = q.hypot3_emulated(*executed)
        cert = q.hypot3_cert(certs, executed, h)
        assert covers(cert, sum(v * v for v in exact), h)


def test_a_zero_face_is_an_exact_zero_or_invalid() -> None:
    zero = (0.0, 0.0, 0.0)
    assert q.hypot3_emulated(*zero) == 0.0
    cert = q.hypot3_cert(tuple(c.exact(a) for a in zero), zero, 0.0)
    assert cert.value == 0
    assert cert.radius == 0
    with pytest.raises(c.Invalid):
        q.hypot3_cert(tuple(c.Cert(Fr(0), Fr(1, 2**60)) for _ in zero), zero, 0.0)
