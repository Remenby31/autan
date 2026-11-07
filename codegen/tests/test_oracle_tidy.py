"""tidy (ADR 0003): every certificate a rule returns is re-centred on a dyadic of CENTRE_BITS
significant bits (ties to even) and its radius rounded up, exactly, to RADIUS_BITS significant bits.
The interval never shrinks, the enlargement per operation is at most 2^-63 r + 2^-127 |v| for the
pre-tidy radius r and centre v, and both fields stay dyadic within their bit budgets."""

from fractions import Fraction as Fr

import numpy as np
import pytest
from oracle import certified as c

B64 = c.BINARY64
Q = Fr(1, 2**63)


def contains(outer: c.Cert, lo: Fr, hi: Fr) -> bool:
    return outer.value - outer.radius <= lo and hi <= outer.value + outer.radius


def significant_bits(x: Fr) -> int:
    """Significant bits of a dyadic; fails for anything else."""
    if x == 0:
        return 0
    d = x.denominator
    assert d & (d - 1) == 0, f"{x} is not dyadic"
    n = abs(x.numerator)
    while n % 2 == 0:
        n //= 2
    return n.bit_length()


def check(x: c.Cert) -> c.Cert:
    """tidy(x), with every property the ADR states asserted exactly."""
    t = c.tidy(x)
    assert contains(t, x.value - x.radius, x.value + x.radius)
    assert t.radius - x.radius <= Q * x.radius + Fr(1, 2**127) * abs(x.value)
    assert significant_bits(t.value) <= c.CENTRE_BITS
    assert significant_bits(t.radius) <= c.RADIUS_BITS
    return t


def test_tidy_never_shrinks_and_respects_its_bound() -> None:
    rng = np.random.default_rng(31)
    for _ in range(5000):
        v = Fr(
            int(rng.integers(-(2**62), 2**62)) * 3 ** int(rng.integers(0, 60)),
            int(rng.integers(1, 2**62)) * 7 ** int(rng.integers(0, 60)),
        )
        r = Fr(
            int(rng.integers(0, 2**62)), int(rng.integers(1, 2**62)) * 5 ** int(rng.integers(0, 60))
        )
        check(c.Cert(v, r))


def test_exact_dyadics_keep_their_value_and_radius_zero() -> None:
    for v in (Fr(0), Fr(1), Fr(-3, 8), Fr(2**127 + 1, 2**60), Fr(1, 2**1074)):
        t = check(c.Cert(v, Fr(0)))
        assert t.value == v
        assert t.radius == 0


def test_a_non_dyadic_value_with_radius_zero_gets_a_radius() -> None:
    t = check(c.Cert(Fr(1, 3), Fr(0)))
    assert t.radius > 0


def test_ties_go_to_even_on_both_signs() -> None:
    for sign in (1, -1):
        t = check(c.Cert(Fr(sign * (2**128 + 1)), Fr(0)))
        assert t.value == sign * 2**128
        t = check(c.Cert(Fr(sign * (2**128 + 3)), Fr(0)))
        assert t.value == sign * (2**128 + 4)


def test_rounding_carries_into_the_next_binade() -> None:
    t = check(c.Cert(Fr(2**129 - 1, 2), Fr(0)))  # 2^128 - 1/2: the tie rounds up to 2^128
    assert t.value == 2**128
    t = check(c.Cert(Fr(0), Fr(2**65 - 1, 2)))  # radius 2^64 - 1/2 rounds up to 2^64
    assert t.radius == 2**64


@pytest.mark.parametrize("exponent", [-5000, -1100, -1074, -60, 0, 60, 1100, 5000])
def test_extreme_scales(exponent: int) -> None:
    scale = Fr(2) ** exponent
    for v, r in (
        ((2**200 + 1) * scale, scale / 3),
        (-(3**90) * scale, Fr(0)),
        (scale / 7, scale * Fr(5, 11)),
    ):
        check(c.Cert(v, r))


def test_the_radius_is_rounded_up_exactly() -> None:
    r = Fr(2**64 + 1, 2**64)  # 65 significant bits
    t = check(c.Cert(Fr(0), r))
    assert t.radius == 1 + Fr(2, 2**64)


def test_every_rule_returns_a_tidied_certificate() -> None:
    third = c.Cert(Fr(1, 3), Fr(1, 3 * 2**70))
    for out in (
        c.add(third, third, B64),
        c.sub(third, c.exact(0.1), B64),
        c.mul(third, third, B64),
        c.div(c.exact(1.0), third, B64),
        c.cast(third, c.BINARY32),
        c.cast(c.cast(third, c.BINARY32), B64, narrowing=False),
    ):
        assert out == c.tidy(out)  # already tidy: tidy is idempotent on its own output
        assert significant_bits(out.value) <= c.CENTRE_BITS
        assert significant_bits(out.radius) <= c.RADIUS_BITS


def test_the_exactness_refinement_still_gives_radius_zero() -> None:
    total = c.exact(0.0)
    for k in range(10):
        total = c.add(total, c.mul(c.exact(2.0**-k), c.exact(3.0), B64), B64)
    assert total.radius == 0
    # A non-dyadic exact value is not representable: its tidied certificate has a radius, and the
    # refinement is not taken downstream.
    assert c.add(c.tidy(c.Cert(Fr(1, 3), Fr(0))), c.exact(1.0), B64).radius > 0


def test_eta_and_invalid_after_tidy() -> None:
    z = c.sub(c.Cert(Fr(1), Fr(1, 2**60)), c.Cert(Fr(1), Fr(0)), B64)
    assert z.radius >= (1 + B64.u) * Fr(1, 2**60) + B64.eta
    y = c.tidy(c.Cert(Fr(1), 1 - Fr(1, 2**80)))  # the radius rounds up to 1
    assert y.radius == 1
    with pytest.raises(c.Invalid):
        c.div(c.exact(1.0), y, B64)


def test_long_chains_keep_a_bounded_size() -> None:
    x = c.exact(0.1)
    for _ in range(1000):
        x = c.div(
            c.add(c.mul(x, c.exact(1.0 + 2.0**-20), B64), c.exact(0.3), B64), c.exact(3.0), B64
        )
        assert significant_bits(x.value) <= c.CENTRE_BITS
        assert significant_bits(x.radius) <= c.RADIUS_BITS
