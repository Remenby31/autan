"""Certified error propagation of ADR 0003 (codegen/tests/oracle/certified.py).

Every bound is checked against what IEEE hardware computes, never against the module's own
formula: binary64 by Python floats, binary32 by NumPy float32 operations on float32 operands
(direct rounding; no rational -> binary64 -> binary32 double rounding as a reference). Cases are
deterministic: ties of both parities, subnormals and the normal boundary, the largest finite value,
cancellation, zero centres, negative divisors, divisor radii near |y|, interval endpoints and
chains of operations."""

import operator
import sys
from collections.abc import Callable
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest
from oracle import certified as c

B64, B32 = c.BINARY64, c.BINARY32
U64 = Fr(1, 2**53)
TINY64 = 2.0**-1074  # smallest binary64 subnormal
NORMAL64 = sys.float_info.min
MAX64 = sys.float_info.max
OPS: dict[
    str, tuple[Callable[[c.Cert, c.Cert, c.Format], c.Cert], Callable[[float, float], float]]
] = {
    "add": (c.add, operator.add),
    "sub": (c.sub, operator.sub),
    "mul": (c.mul, operator.mul),
    "div": (c.div, operator.truediv),
}


def f32(x: float) -> np.float32:
    """A binary32 value that must be exact: refuses any rounding."""
    y = np.float32(x)
    assert float(y) == x, x
    return y


def within(cert: c.Cert, computed: float) -> bool:
    return abs(Fr(computed) - cert.value) <= cert.radius


def check64(name: str, x: c.Cert, y: c.Cert, xh: float, yh: float) -> c.Cert:
    """The hardware result of xh op yh, with |xh - x| <= r_x and |yh - y| <= r_y, lies within the
    propagated radius."""
    assert within(x, xh)
    assert within(y, yh)
    rule, hw = OPS[name]
    out = rule(x, y, B64)
    assert within(out, hw(xh, yh)), (name, xh, yh, hw(xh, yh), out)
    return out


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 2.0**-53),  # tie, rounds to the even 1
        (1.0 + 2.0**-52, 2.0**-53),  # tie, rounds up to the even 1 + 2^-51
        (-1.0, -(2.0**-53)),
        (2.0**52, 0.5),  # tie at the integer scale
        (2.0**52 + 1, 0.5),
    ],
)
def test_ties_of_both_parities(a: float, b: float) -> None:
    out = check64("add", c.exact(a), c.exact(b), a, b)
    assert out.radius > 0


def test_ties_of_both_parities_in_binary32() -> None:
    for a, b in ((1.0, 2.0**-24), (1.0 + 2.0**-23, 2.0**-24)):
        xa, xb = f32(a), f32(b)
        out = c.add(c.exact(float(xa)), c.exact(float(xb)), B32)
        assert within(out, float(xa + xb))
        assert out.radius > 0


@pytest.mark.parametrize(
    ("a", "b", "name"),
    [
        (TINY64, 0.5, "mul"),  # half the smallest subnormal: a tie that rounds to 0
        (TINY64, 0.75, "mul"),  # above half: rounds to the smallest subnormal
        (TINY64, 0.25, "mul"),  # below half: rounds to 0
        (3 * TINY64, 0.5, "mul"),  # an inexact subnormal tie
        (4 * TINY64, 0.5, "mul"),  # an exact subnormal
        (-TINY64, 0.5, "mul"),
        (NORMAL64, 0.5, "mul"),  # from the normal boundary into the subnormals
        (NORMAL64, 1 - 2.0**-52, "mul"),  # just below the boundary
        (NORMAL64, 2.0**-53, "add"),  # the smallest normal plus less than its spacing
        (3 * TINY64, 2.0, "div"),
    ],
)
def test_subnormals_against_the_hardware(a: float, b: float, name: str) -> None:
    check64(name, c.exact(a), c.exact(b), a, b)


def test_subnormals_in_binary32() -> None:
    tiny32 = float(np.float32(2.0**-149))
    for b in (0.5, 0.75, 0.25):
        xa, xb = np.float32(tiny32), f32(b)
        out = c.mul(c.exact(float(xa)), c.exact(float(xb)), B32)
        assert within(out, float(xa * xb))


def test_exact_subnormals_keep_radius_zero() -> None:
    assert c.mul(c.exact(4 * TINY64), c.exact(0.5), B64).radius == 0
    assert c.add(c.exact(TINY64), c.exact(TINY64), B64).radius == 0


def test_a_zero_centre_with_a_radius_is_sound_where_results_are_subnormal() -> None:
    # x has exact value 0 and radius 2^-1070: its computed value may be any binary64 number
    # within that, among them subnormals; each product by 0.5 the hardware gives is covered.
    x = c.Cert(Fr(0), Fr(2) ** -1070)
    for xh in (0.0, TINY64, 3 * TINY64, -(2.0**-1070), 2.0**-1071 + TINY64):
        assert within(x, xh)
        out = c.mul(x, c.exact(0.5), B64)
        assert within(out, xh * 0.5), xh


def test_the_largest_finite_value_and_the_overflow_boundary() -> None:
    assert c.add(c.exact(MAX64), c.exact(0.0), B64).radius == 0
    assert c.mul(c.exact(MAX64), c.exact(0.5), B64).radius == 0
    # MAX + half its spacing rounds to infinity: refused.
    assert float("inf") == MAX64 + 2.0**970
    with pytest.raises(c.Invalid):
        c.add(c.exact(MAX64), c.exact(2.0**970), B64)
    with pytest.raises(c.Invalid):
        c.cast(c.exact(1.0e39), B32)
    max32 = float(np.finfo(np.float32).max)
    assert c.cast(c.exact(max32), B32).radius == 0


def test_catastrophic_cancellation() -> None:
    a = 1.0 + 2.0**-52
    assert check64("sub", c.exact(a), c.exact(1.0), a, 1.0).radius == 0
    # Operands known only within their radii: the hardware difference of any endpoints is covered.
    x, y = c.Cert(Fr(a), U64), c.Cert(Fr(1), U64)
    for xh in (a, a - 2.0**-53 * 0, 1.0 + 2.0**-51):
        for yh in (1.0, 1.0 - 2.0**-53):
            if within(x, xh) and within(y, yh):
                check64("sub", x, y, xh, yh)


def test_negative_divisors() -> None:
    for a, b in ((1.0, -3.0), (-1.0, -3.0), (7.0, -0.1)):
        check64("div", c.exact(a), c.exact(b), a, b)


def test_divisor_radius_near_its_value() -> None:
    x = c.exact(1.0)
    near = c.Cert(Fr(3), Fr(3) * (1 - Fr(1, 2**10)))
    out = c.div(x, near, B64)
    assert out.radius > 0
    for yh in (3.0, 3.0 * 2.0**-10 + 2.0**-60, 5.9):
        if within(near, yh):
            assert within(out, 1.0 / yh), yh
    for ry in (Fr(3), Fr(4)):
        with pytest.raises(c.Invalid):
            c.div(x, c.Cert(Fr(3), ry), B64)


def test_interval_endpoints_and_chains() -> None:
    # Operands at the ends of their intervals, then chains: every hardware value stays within.
    rng = np.random.default_rng(43)
    for _ in range(2000):
        xs = [Fr(int(rng.integers(-(2**40), 2**40)), int(rng.integers(1, 2**30))) for _ in range(3)]
        hats = [float(v) for v in xs]
        certs = [c.Cert(v, abs(Fr(h) - v)) for v, h in zip(xs, hats, strict=True)]
        acc, acc_h = certs[0], hats[0]
        for k, name in enumerate(rng.permutation(list(OPS))):
            other, other_h = certs[1 + k % 2], hats[1 + k % 2]
            if name == "div" and abs(other.value) <= other.radius:
                continue
            acc = check64(str(name), acc, other, acc_h, other_h)
            acc_h = OPS[str(name)][1](acc_h, other_h)


def test_chains_in_binary32() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        vals = [np.float32(rng.standard_normal()) for _ in range(3)]
        certs = [c.exact(float(v)) for v in vals]
        acc, acc_h = certs[0], vals[0]
        for k, name in enumerate(("mul", "add", "sub", "div")):
            other, other_h = certs[1 + k % 2], vals[1 + k % 2]
            acc = OPS[name][0](acc, other, B32)
            acc_h = OPS[name][1](acc_h, other_h)
            assert within(acc, float(acc_h)), (name, vals)


def test_inputs_given_exactly_have_radius_zero() -> None:
    x = c.exact(0.1)
    assert x.value == Fr(0.1)
    assert x.radius == 0


def test_widening_only_from_a_declared_narrower_format() -> None:
    # A widening cast is exact only for a value of the narrower format: anything else is refused.
    third = c.cast(c.Cert(Fr(1, 3), Fr(0)), B32)
    widened = c.cast(third, B64, narrowing=False)
    assert widened.radius == third.radius
    with pytest.raises(c.Invalid):
        c.cast(c.Cert(Fr(1, 3), Fr(0)), B64, narrowing=False)
    with pytest.raises(c.Invalid):
        c.cast(c.exact(0.1), B32, narrowing=False)


def test_nothing_outside_the_tests_imports_the_oracle() -> None:
    # The oracle is verification tooling: the generator and the kernels never depend on it.
    root = Path(__file__).resolve().parents[2]
    sources = [*(root / "codegen" / "src").rglob("*.py"), *(root / "kernels").rglob("*.py")]
    offenders = [p for p in sources if "oracle" in p.read_text()]
    assert sources
    assert not offenders, offenders
