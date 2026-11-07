"""Certified error propagation (ADR 0003, docs/adr/0003-derived-linear-exactness-bound.md).

Every quantity is a `Cert`: the exact value of the declared operation sequence evaluated without
rounding, and a radius proven to bound the distance from it to the value that any evaluation of
that sequence produces in IEEE 754 arithmetic, round to nearest, without FMA contraction.

For an operation whose exact result on the computed operands is z, with exact value v and
operand error |z - v| <= D, the computed fl(z) satisfies |fl(z) - v| <= (1 + u) D + u |v|, plus
eta = 2^(emin - p) when [v - D, v + D] reaches the subnormal band, where u |z| alone does not bound
the rounding (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., eq. (2.8)): an exact
rational, 2^-1075 in binary64 and 2^-150 in binary32. It is kept for + and - too, conservative:
dropping it there would need a proof of the operands' format. D is
r_x + r_y for + and -, |x| r_y + |y| r_x + r_x r_y for *, and (r_x + |x/y| r_y) / (|y| - r_y) for
/, refused when r_y >= |y|. When D = 0 and v is representable, fl(z) = v: the radius is 0.

Model and limits. IEEE 754 round to nearest even with gradual underflow: flush-to-zero and
denormals-are-zero are excluded, and so are evaluation changes such as x87 extended precision or an
FMA the compiler contracts unless the sequence declares it as one operation. `Invalid` (overflow,
a divisor its radius may reach, a widening from an undeclared format) never becomes a pass. The
certificate bounds a distance: the sign of a zero result is outside it.

Values are exact rationals throughout; no floating-point arithmetic enters a value or a radius.
Verification tooling only: nothing outside codegen/tests imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


class Invalid(ValueError):
    """The instance leaves the arithmetic model (overflow, or a divisor its radius may reach)."""


@dataclass(frozen=True)
class Format:
    """An IEEE 754 binary format: precision p (bits, hidden bit included), exponent range."""

    name: str
    p: int
    emin: int
    emax: int

    @property
    def u(self) -> Fraction:
        """Unit roundoff 2^-p."""
        return Fraction(1, 2**self.p)

    @property
    def normal_min(self) -> Fraction:
        return Fraction(2) ** self.emin

    @property
    def max(self) -> Fraction:
        return (2 - Fraction(2) ** (1 - self.p)) * Fraction(2) ** self.emax

    @property
    def eta(self) -> Fraction:
        """Half the spacing of the subnormals: the absolute rounding error there."""
        return Fraction(2) ** (self.emin - self.p)


BINARY64 = Format("binary64", 53, -1022, 1023)
BINARY32 = Format("binary32", 24, -126, 127)


@dataclass(frozen=True)
class Cert:
    """Exact value, radius, and the format of the computed value when it is declared (set by
    every rule; a float given exactly is binary64; an exact rational has none)."""

    value: Fraction
    radius: Fraction
    fmt: Format | None = None

    def __post_init__(self) -> None:
        if self.radius < 0:
            msg = "a radius is never negative"
            raise ValueError(msg)


def exact(x: float | Fraction) -> Cert:
    """An input given exactly: a float is its exact binary64 value; a rational has no format."""
    if isinstance(x, float):
        return Cert(Fraction(x), Fraction(0), BINARY64)
    return Cert(Fraction(x), Fraction(0))


def _contains(wide: Format, narrow: Format) -> bool:
    return narrow.p <= wide.p and narrow.emin >= wide.emin and narrow.emax <= wide.emax


def representable(v: Fraction, fmt: Format) -> bool:
    """Whether v is a finite value of `fmt` (zero, subnormal or normal)."""
    if v == 0:
        return True
    a = abs(v)
    if a > fmt.max:
        return False
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, fmt.emin)
    scaled = a * Fraction(2) ** (fmt.p - 1 - e)
    return scaled.denominator == 1


def _rounded(v: Fraction, d: Fraction, fmt: Format) -> Cert:
    """The result of rounding, to `fmt`, an exact operation result within d of v."""
    if abs(v) + d > fmt.max:
        msg = f"{fmt.name}: the result may overflow (|v| + D exceeds the largest finite value)"
        raise Invalid(msg)
    if d == 0 and representable(v, fmt):
        return Cert(v, Fraction(0), fmt)
    r = (1 + fmt.u) * d + fmt.u * abs(v)
    if abs(v) - d < fmt.normal_min:
        r += fmt.eta
    return Cert(v, r, fmt)


def add(x: Cert, y: Cert, fmt: Format) -> Cert:
    return _rounded(x.value + y.value, x.radius + y.radius, fmt)


def sub(x: Cert, y: Cert, fmt: Format) -> Cert:
    return _rounded(x.value - y.value, x.radius + y.radius, fmt)


def mul(x: Cert, y: Cert, fmt: Format) -> Cert:
    d = abs(x.value) * y.radius + abs(y.value) * x.radius + x.radius * y.radius
    return _rounded(x.value * y.value, d, fmt)


def div(x: Cert, y: Cert, fmt: Format) -> Cert:
    if y.radius >= abs(y.value):
        msg = f"{fmt.name}: the divisor's radius reaches it (r_y >= |y|)"
        raise Invalid(msg)
    q = x.value / y.value
    d = (x.radius + abs(q) * y.radius) / (abs(y.value) - y.radius)
    return _rounded(q, d, fmt)


def cast(x: Cert, fmt: Format, *, narrowing: bool = True) -> Cert:
    """Conversion to `fmt`: rounding when it narrows; exact when it widens, which requires the
    computed value to be a finite value of a declared format that `fmt` contains."""
    if not narrowing:
        if x.fmt is None or not _contains(fmt, x.fmt):
            msg = f"widening to {fmt.name} needs a value of a declared narrower format"
            raise Invalid(msg)
        return Cert(x.value, x.radius, fmt)
    return _rounded(x.value, x.radius, fmt)
