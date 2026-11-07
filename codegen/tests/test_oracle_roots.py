"""The two operations beyond the four rules (ADR 0003 amendment): sqrt, correctly rounded, with
witnesses validated by exact squaring; hypot, checked per instance exactly against the rational sum
of squares of its executed arguments, with no libm assumption."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from oracle import certified as c

B64 = c.BINARY64
K = 2  # the hypot tolerance in units of u used by these tests


def covers(cert: c.Cert, exact_square: Fr, y: float) -> bool:
    """|y - sqrt(S)| <= radius, decided exactly: (y - R)^2 <= S <= (y + R)^2, with y - R clipped
    at 0."""
    lo = max(Fr(y) - cert.radius, Fr(0))
    hi = Fr(y) + cert.radius
    return lo * lo <= exact_square <= hi * hi


def test_sqrt_bounds_the_hardware_root() -> None:
    rng = np.random.default_rng(5)
    for _ in range(3000):
        exact = Fr(int(rng.integers(1, 2**60)), int(rng.integers(1, 2**40)))
        x = c.cast(c.Cert(exact, Fr(0)), B64)
        executed = float(exact)  # within x's radius by the cast rule
        assert abs(Fr(executed) - exact) <= x.radius
        root = c.sqrt(x, B64)
        assert covers(root, exact, math.sqrt(executed))
        assert covers(root, exact, float(root.value))  # the centre itself is within


def test_sqrt_of_operands_at_their_interval_ends() -> None:
    exact = Fr(2)
    x = c.Cert(exact, Fr(1, 2**20))
    root = c.sqrt(x, B64)
    for executed in (2.0, 2.0 + 2.0**-20, 2.0 - 2.0**-20):
        assert covers(root, exact, math.sqrt(executed))


def test_sqrt_refuses_an_interval_reaching_zero() -> None:
    with pytest.raises(c.Invalid):
        c.sqrt(c.Cert(Fr(1, 4), Fr(1, 4)), B64)
    with pytest.raises(c.Invalid):
        c.sqrt(c.Cert(Fr(-1), Fr(0)), B64)


def test_hypot_is_checked_against_the_sum_of_squares_of_its_arguments() -> None:
    rng = np.random.default_rng(11)
    for _ in range(3000):
        args = [float(v) for v in rng.standard_normal(3) * 10.0 ** rng.integers(-3, 4)]
        certs = [c.exact(a) for a in args]
        h = math.hypot(*args)  # any implementation: the check is per instance
        out = c.hypot_checked(tuple(args), tuple(certs), h, K, B64)
        assert covers(out, sum(Fr(a) ** 2 for a in args), h)


def test_hypot_propagates_the_uncertainty_of_its_arguments() -> None:
    exact = [Fr(3), Fr(4), Fr(12)]  # sqrt(169) = 13
    certs = [c.Cert(v, Fr(1, 2**30)) for v in exact]
    executed = (3.0 + 2.0**-30, 4.0, 12.0 - 2.0**-30)
    h = math.hypot(*executed)
    out = c.hypot_checked(executed, tuple(certs), h, K, B64)
    assert covers(out, Fr(169), h)
    assert out.radius >= abs(Fr(h) - 13)


def test_a_hypot_result_off_by_two_epsilon_is_invalid() -> None:
    # The control the amendment asks for: h perturbed by +2 eps on one instance must be invalid.
    args = (0.3, -1.7, 2.9)
    h = math.hypot(*args)
    eps = K * B64.u
    bad = float(Fr(h) * (1 + 2 * eps))
    s = sum(Fr(a) ** 2 for a in args)
    while (Fr(bad) * (1 - eps)) ** 2 <= s:  # make sure it is outside the band after rounding
        bad = math.nextafter(bad, math.inf)
    with pytest.raises(c.Invalid):
        c.hypot_checked(args, tuple(c.exact(a) for a in args), bad, K, B64)


def test_hypot_refuses_an_argument_outside_its_certificate() -> None:
    args = (1.0, 2.0, 2.0)
    certs = (c.exact(1.0), c.exact(2.0), c.Cert(Fr(5, 2), Fr(1, 4)))
    with pytest.raises(c.Invalid):
        c.hypot_checked(args, certs, 3.0, K, B64)
