"""The square root beyond the four rules (ADR 0003 amendment): the certified rule, with witnesses
validated by exact squaring; and the exact emulation of round-to-nearest-even and of the correctly
rounded square root that declared sequences are replayed with, against the hardware."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from oracle import certified as c

B64 = c.BINARY64


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


@pytest.mark.parametrize("fmt", ["binary64", "binary32"])
def test_round_to_reproduces_the_hardware(fmt: str) -> None:
    rng = np.random.default_rng(17)
    f = B64 if fmt == "binary64" else c.BINARY32
    cast = float if fmt == "binary64" else (lambda v: float(np.float32(v)))
    ops = (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b)
    for _ in range(3000):
        a, b = (cast(float(v)) for v in rng.standard_normal(2) * 10.0 ** rng.integers(-30, 30, 2))
        for op in ops:
            with np.errstate(over="ignore"):
                hw = op(a, b) if fmt == "binary64" else float(op(np.float32(a), np.float32(b)))
            if not math.isfinite(hw):  # overflow: refused by design, never a value
                with pytest.raises(c.Invalid):
                    c.round_to(op(Fr(a), Fr(b)), f)
                continue
            assert c.round_to(op(Fr(a), Fr(b)), f) == Fr(hw), (a, b)


def test_round_to_breaks_ties_to_even_and_handles_subnormals() -> None:
    assert c.round_to(1 + Fr(1, 2**53), B64) == 1
    assert c.round_to(1 + Fr(3, 2**53), B64) == 1 + Fr(1, 2**51)
    tiny = Fr(1, 2**1074)
    assert c.round_to(tiny / 2, B64) == 0
    assert c.round_to(3 * tiny / 2, B64) == 2 * tiny
    assert c.round_to(3 * tiny / 4, B64) == tiny
    with pytest.raises(c.Invalid):
        c.round_to(Fr(2) ** 1024, B64)


@pytest.mark.parametrize("fmt", ["binary64", "binary32"])
def test_round_sqrt_reproduces_the_hardware(fmt: str) -> None:
    rng = np.random.default_rng(23)
    for _ in range(3000):
        v = abs(float(rng.standard_normal() * 10.0 ** rng.integers(-30, 30)))
        if fmt == "binary64":
            assert c.round_sqrt(Fr(v), B64) == Fr(math.sqrt(v)), v
        else:
            x = np.float32(v)
            assert c.round_sqrt(Fr(float(x)), c.BINARY32) == Fr(float(np.sqrt(x))), v
    assert c.round_sqrt(Fr(4), B64) == 2
    assert c.round_sqrt(Fr(0), B64) == 0
