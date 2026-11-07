"""Declared operation sequences of ADR 0003: each reproduces, operation by operation, a piece of the
build it certifies. Its source and version are pinned in FREEZE.md; a replay that does not match the
stored result bit for bit means the build does not run the declared sequence, and the instance is
invalid (fail-closed)."""

from __future__ import annotations

from fractions import Fraction

from oracle import certified as c

# libstdc++ of GCC 13 (the autan-dev image), /usr/include/c++/13/cmath:3550-3563:
#   __x = abs(__x); __y = abs(__y); __z = abs(__z);
#   if (_Tp __a = __x < __y ? __y < __z ? __z : __y : __x < __z ? __z : __x)
#     return __a * sqrt((__x / __a) * (__x / __a) + (__y / __a) * (__y / __a)
#                       + (__z / __a) * (__z / __a));
#   else return {};
HYPOT3_SOURCE = "GCC 13 libstdc++ /usr/include/c++/13/cmath:3550-3563 (std::__hypot3)"


def _pick(x: Fraction, y: Fraction, z: Fraction) -> int:
    """Index of __a among (x, y, z), by the source's comparisons."""
    if x < y:
        return 2 if y < z else 1
    return 2 if x < z else 0


def hypot3_emulated(x: float, y: float, z: float) -> float:
    """std::hypot(x, y, z) of the build, replayed exactly in binary64."""
    f = c.BINARY64
    v = [abs(Fraction(t)) for t in (x, y, z)]
    a = v[_pick(*v)]
    if a == 0:
        return 0.0
    squares = [c.fl_mul(q, q, f) for q in (c.fl_div(t, a, f) for t in v)]
    total = c.fl_add(c.fl_add(squares[0], squares[1], f), squares[2], f)
    return float(c.fl_mul(a, c.round_sqrt(total, f), f))


def hypot3_cert(
    args: tuple[c.Cert, c.Cert, c.Cert], executed: tuple[float, float, float], stored: float
) -> c.Cert:
    """The certified magnitude of a face: the stored value must be the declared sequence of the
    executed arguments bit for bit (else Invalid); its radius around the exact root sqrt(sum X^2)
    comes from the rules on that sequence. The exact identity a sqrt(sum (X/a)^2) = sqrt(sum X^2)
    holds for any a > 0, so the executed branch need only certify a > 0."""
    if hypot3_emulated(*executed) != stored:
        msg = "hypot: the stored magnitude is not the declared sequence of its arguments"
        raise c.Invalid(msg)
    f = c.BINARY64
    for t, cert in zip(executed, args, strict=True):
        if abs(Fraction(t) - cert.value) > cert.radius:
            msg = "hypot: an executed argument lies outside its certificate"
            raise c.Invalid(msg)
    absolute = [c.Cert(abs(cert.value), cert.radius, f) for cert in args]
    k = _pick(*(abs(Fraction(t)) for t in executed))
    a = absolute[k]
    if abs(Fraction(executed[k])) == 0:
        if any(cert.value != 0 or cert.radius != 0 for cert in args):
            msg = "hypot: a zero face whose exact area is not certainly zero"
            raise c.Invalid(msg)
        return c.Cert(Fraction(0), Fraction(0), f)
    if a.value - a.radius <= 0:
        msg = "hypot: the scale a is not certainly positive"
        raise c.Invalid(msg)
    squares = [c.mul(q, q, f) for q in (c.div(t, a, f) for t in absolute)]
    total = c.add(c.add(squares[0], squares[1], f), squares[2], f)
    return c.mul(a, c.sqrt(total, f), f)
