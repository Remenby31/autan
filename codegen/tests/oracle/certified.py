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

ROOT_BITS = 80  # precision of the rational root witnesses, beyond the radicand's scale


def _root_floor(q: Fraction) -> Fraction:
    """A rational l >= 0 with l^2 <= q: proposed by an integer square root, validated exactly."""
    if q <= 0:
        return Fraction(0)
    n = q.numerator * 4**ROOT_BITS // q.denominator
    out = Fraction(math.isqrt(n), 2**ROOT_BITS)
    if out * out > q:
        msg = "a lower root witness failed its exact check"
        raise Invalid(msg)
    return out


def _root_ceil(q: Fraction) -> Fraction:
    """A rational U >= 0 with U^2 >= q: proposed by an integer square root, validated exactly."""
    if q <= 0:
        return Fraction(0)
    n = -(-q.numerator * 4**ROOT_BITS // q.denominator)
    root = math.isqrt(n)
    if root * root < n:
        root += 1
    out = Fraction(root, 2**ROOT_BITS)
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


def hypot_checked(
    executed: tuple[float, ...], args: tuple[Cert, ...], h: float, k: int, fmt: Format
) -> Cert:
    """An executed hypot result h of the executed arguments, with no assumption on the library:
    h^2 (1 - eps)^2 <= S <= h^2 (1 + eps)^2 is checked exactly, S the rational sum of squares of
    the executed arguments and eps = k u, which certifies |h - sqrt(S)| <= eps h. The arguments'
    uncertainty is propagated to the exact value sqrt(V), V the sum of squares of their exact
    values. Any failed check makes the instance invalid."""
    if len(executed) != len(args):
        msg = "hypot: as many executed arguments as certificates"
        raise ValueError(msg)
    for a, cert in zip(executed, args, strict=True):
        if not math.isfinite(a) or abs(Fraction(a) - cert.value) > cert.radius:
            msg = "hypot: an executed argument lies outside its certificate"
            raise Invalid(msg)
    if not math.isfinite(h) or h < 0:
        msg = "hypot: the executed result is not finite and nonnegative"
        raise Invalid(msg)
    hh, eps = Fraction(h), k * fmt.u
    s_exec = sum((Fraction(a) ** 2 for a in executed), Fraction(0))
    if not (hh * hh * (1 - eps) ** 2 <= s_exec <= hh * hh * (1 + eps) ** 2):
        msg = f"hypot: the executed result is not within (1 +- {k}u) of the root of its arguments' sum of squares"
        raise Invalid(msg)
    big_v = sum((cert.value**2 for cert in args), Fraction(0))
    spread = sum((2 * abs(cert.value) * cert.radius + cert.radius**2 for cert in args), Fraction(0))
    lower = _root_floor(big_v - spread) if big_v - spread > 0 else Fraction(0)
    m = _root_floor(big_v)
    d = (
        spread / (lower + m) if lower + m > 0 else _root_ceil(spread)
    )  # |sqrt a - sqrt b| <= sqrt|a - b|
    mu, rho = _enclose_root(big_v) if big_v > 0 else (Fraction(0), Fraction(0))
    return Cert(mu, rho + d + eps * hh, fmt)
