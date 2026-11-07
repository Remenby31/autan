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

import math
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


# The two operations beyond the four rules (ADR 0003 amendment, decided on #43).

ROOT_BITS = 80  # significant bits of the rational root witnesses, whatever the magnitude


def _scale_bits(q: Fraction) -> int:
    """k such that the root of q * 4^k has about ROOT_BITS bits before the binary point."""
    e = q.numerator.bit_length() - q.denominator.bit_length()
    return ROOT_BITS - e // 2


def _root_floor(q: Fraction) -> Fraction:
    """A rational l >= 0 with l^2 <= q: proposed by an integer square root, validated exactly."""
    if q <= 0:
        return Fraction(0)
    k = _scale_bits(q)
    scaled = q * Fraction(4) ** k
    out = Fraction(math.isqrt(scaled.numerator // scaled.denominator)) / Fraction(2) ** k
    if out * out > q:
        msg = "a lower root witness failed its exact check"
        raise Invalid(msg)
    return out


def _root_ceil(q: Fraction) -> Fraction:
    """A rational U >= 0 with U^2 >= q: proposed by an integer square root, validated exactly."""
    if q <= 0:
        return Fraction(0)
    k = _scale_bits(q)
    scaled = q * Fraction(4) ** k
    n = -(-scaled.numerator // scaled.denominator)
    root = math.isqrt(n)
    if root * root < n:
        root += 1
    out = Fraction(root) / Fraction(2) ** k
    if out * out < q:
        msg = "an upper root witness failed its exact check"
        raise Invalid(msg)
    return out


def _enclose_root(q: Fraction) -> tuple[Fraction, Fraction]:
    """A rational centre mu and radius rho with (mu - rho)^2 <= q <= (mu + rho)^2, mu - rho >= 0."""
    lo, hi = _root_floor(q), _root_ceil(q)
    mu, rho = (lo + hi) / 2, (hi - lo) / 2
    if mu - rho < 0 or not ((mu - rho) ** 2 <= q <= (mu + rho) ** 2):
        msg = "a root enclosure failed its exact check"
        raise Invalid(msg)
    return mu, rho


def sqrt(x: Cert, fmt: Format) -> Cert:
    """Correctly rounded square root of an executed x with |X - x| <= r, X - r > 0:
    |y - sqrt(X)| <= r / (l + m) + u U (+ eta), with l^2 <= X - r, m^2 <= X, U^2 >= X + r; the
    certified value is a rational centre mu with rho >= |mu - sqrt(X)|."""
    big_x, r = x.value, x.radius
    if big_x - r <= 0:
        msg = f"{fmt.name}: sqrt of an interval that reaches zero or below"
        raise Invalid(msg)
    ell, m, big_u = _root_floor(big_x - r), _root_floor(big_x), _root_ceil(big_x + r)
    if ell <= 0 or m <= 0:
        msg = f"{fmt.name}: sqrt witnesses are not positive"
        raise Invalid(msg)
    mu, rho = _enclose_root(big_x)
    radius = rho + r / (ell + m) + fmt.u * big_u
    if mu - radius < fmt.normal_min:
        radius += fmt.eta
    return Cert(mu, radius, fmt)


# Exact emulation of IEEE round to nearest even (gradual underflow), to replay declared sequences
# bit for bit; never a reference for direct rounding through another format.


def _exponent(a: Fraction) -> int:
    """floor(log2 a) for a > 0."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    return e


def round_to(v: Fraction, fmt: Format) -> Fraction:
    """v rounded to the nearest value of `fmt`, ties to even; Invalid if it overflows."""
    if v == 0:
        return Fraction(0)
    a = abs(v)
    e = max(_exponent(a), fmt.emin)
    scale = Fraction(2) ** (fmt.p - 1 - e)  # one unit in the last place is 1 after scaling
    scaled = a * scale
    n = scaled.numerator // scaled.denominator
    rest = scaled - n
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    r = Fraction(n) / scale
    if r > fmt.max:
        msg = f"{fmt.name}: the rounded result overflows"
        raise Invalid(msg)
    return r if v > 0 else -r


def _ulp(a: Fraction, fmt: Format) -> Fraction:
    """The spacing of `fmt` at a > 0 (above a, within a's binade)."""
    return Fraction(2) ** (max(_exponent(a), fmt.emin) - fmt.p + 1)


def round_sqrt(v: Fraction, fmt: Format) -> Fraction:
    """The correctly rounded square root of a rational v >= 0 (round to nearest even)."""
    if v < 0:
        msg = "round_sqrt of a negative value"
        raise Invalid(msg)
    if v == 0:
        return Fraction(0)
    r = round_to(_root_floor(v), fmt)
    for _ in range(4):
        up = r + _ulp(r, fmt)
        below = r - (
            _ulp(r, fmt) / 2
            if r == Fraction(2) ** _exponent(r) and _exponent(r) > fmt.emin
            else _ulp(r, fmt)
        )
        hi, lo = (r + up) / 2, (r + below) / 2  # the midpoints with the neighbours
        if hi * hi < v or (hi * hi == v and (r / _ulp(r, fmt)) % 2 == 1):
            r = up
        elif lo * lo > v or (lo * lo == v and (r / _ulp(r, fmt)) % 2 == 1):
            r = below
        else:
            return r
    msg = "round_sqrt did not settle"
    raise Invalid(msg)


def fl_add(a: Fraction, b: Fraction, fmt: Format) -> Fraction:
    return round_to(a + b, fmt)


def fl_mul(a: Fraction, b: Fraction, fmt: Format) -> Fraction:
    return round_to(a * b, fmt)


def fl_div(a: Fraction, b: Fraction, fmt: Format) -> Fraction:
    if b == 0:
        msg = "division by zero"
        raise Invalid(msg)
    return round_to(a / b, fmt)
