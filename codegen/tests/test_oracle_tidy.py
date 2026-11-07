"""tidy (ADR 0003): every certificate is re-centred on a dyadic of CENTRE_BITS significant bits and
its radius rounded up, exactly, to RADIUS_BITS significant bits, so that values and radii keep a
bounded size. The interval never shrinks."""

import time
from fractions import Fraction as Fr

import numpy as np
import pytest
from oracle import certified as c

B64 = c.BINARY64


def contains(outer: c.Cert, lo: Fr, hi: Fr) -> bool:
    return outer.value - outer.radius <= lo and hi <= outer.value + outer.radius


def test_tidy_never_shrinks_an_interval() -> None:
    rng = np.random.default_rng(31)
    for _ in range(5000):
        v = Fr(
            int(rng.integers(-(2**62), 2**62)) * 3 ** int(rng.integers(0, 60)),
            int(rng.integers(1, 2**62)) * 7 ** int(rng.integers(0, 60)),
        )
        r = Fr(
            int(rng.integers(0, 2**62)), int(rng.integers(1, 2**62)) * 5 ** int(rng.integers(0, 60))
        )
        t = c.tidy(c.Cert(v, r))
        assert contains(t, v - r, v + r)  # both endpoints, so the whole interval
        assert (
            t.value.denominator.bit_length()
            <= 1 + max(0, -c.dyadic_exponent(t.value)) + c.CENTRE_BITS
        )
        assert t.radius >= r


def test_exact_dyadics_keep_their_value_and_radius_zero() -> None:
    for v in (Fr(0), Fr(1), Fr(-3, 8), Fr(2**127 + 1, 2**60), Fr(1, 2**1074)):
        t = c.tidy(c.Cert(v, Fr(0)))
        assert t.value == v
        assert t.radius == 0


def test_the_centre_ties_to_even() -> None:
    # 2^128 + 1 needs 129 bits: halfway between 2^128 and 2^128 + 2, whose last kept bit is even
    # for 2^128.
    t = c.tidy(c.Cert(Fr(2**128 + 1), Fr(0)))
    assert t.value == 2**128
    assert t.radius == 1
    t = c.tidy(c.Cert(Fr(2**128 + 3), Fr(0)))
    assert t.value == 2**128 + 4
    assert t.radius == 1


def test_the_radius_is_rounded_up_exactly() -> None:
    r = Fr(2**64 + 1, 2**64)  # 65 significant bits
    t = c.tidy(c.Cert(Fr(0), r))
    assert t.radius == 1 + Fr(2, 2**64)
    assert t.radius >= r


def test_a_divisor_whose_tidied_radius_reaches_it_is_refused() -> None:
    y = c.tidy(c.Cert(Fr(1), 1 - Fr(1, 2**80)))  # the radius rounds up to 1
    assert y.radius == 1
    with pytest.raises(c.Invalid):
        c.div(c.exact(1.0), y, B64)


def test_long_chains_keep_a_bounded_size() -> None:
    # Before tidy, the denominators grew by about 53 bits per operation; a chain of 3000 operations
    # with divisions by 3 now stays small and fast.
    x = c.exact(0.1)
    start = time.perf_counter()
    for k in range(1000):
        x = c.div(
            c.add(c.mul(x, c.exact(1.0 + 2.0**-20), B64), c.exact(0.3), B64), c.exact(3.0), B64
        )
        assert x.value.denominator.bit_length() < 1200, k
        assert x.radius.denominator.bit_length() < 1200, k
    assert time.perf_counter() - start < 5.0
