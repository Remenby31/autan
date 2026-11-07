"""Certified error propagation of ADR 0003 (codegen/tests/oracle/certified.py): the four rules
and the exactness refinement on hand-checkable cases, the refusals, and soundness against IEEE
arithmetic itself (binary64 by Python floats, binary32 by NumPy)."""

from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest
from oracle import certified as c

B64, B32 = c.BINARY64, c.BINARY32
U64, U32 = Fr(1, 2**53), Fr(1, 2**24)


def test_inputs_given_exactly_have_radius_zero() -> None:
    x = c.exact(0.1)
    assert x.value == Fr(0.1)
    assert x.radius == 0


def test_representable_results_of_exact_operands_stay_exact() -> None:
    # 1 + 2, 0.5 * 0.25, 3 / 4 and dyadic chains: R = 0 (the exactness refinement).
    one, two = c.exact(1.0), c.exact(2.0)
    assert c.add(one, two, B64).radius == 0
    assert c.mul(c.exact(0.5), c.exact(0.25), B32).radius == 0
    assert c.div(c.exact(3.0), c.exact(4.0), B64).radius == 0
    total = c.exact(0.0)
    for k in range(10):
        total = c.add(total, c.mul(c.exact(2.0**-k), c.exact(3.0), B64), B64)
    assert total.radius == 0
    assert total.value == 3 * sum(Fr(1, 2**k) for k in range(10))


def test_a_rounded_sum_costs_u_times_its_value() -> None:
    s = c.add(c.exact(1.0), c.exact(2.0**-60), B64)
    assert s.radius == U64 * (1 + Fr(1, 2**60))


def test_the_product_rule_by_hand() -> None:
    x, y = c.Cert(Fr(3), Fr(1, 8)), c.Cert(Fr(5), Fr(1, 16))
    p = c.mul(x, y, B64)
    assert p.value == 15
    assert p.radius == (1 + U64) * (Fr(3, 16) + Fr(5, 8) + Fr(1, 128)) + U64 * 15


def test_the_quotient_rule_by_hand_and_its_refusal() -> None:
    x, y = c.Cert(Fr(1), Fr(1, 1024)), c.Cert(Fr(3), Fr(1, 2))
    q = c.div(x, y, B64)
    assert q.value == Fr(1, 3)
    assert q.radius == (1 + U64) * (Fr(1, 1024) + Fr(1, 3) * Fr(1, 2)) / (3 - Fr(1, 2)) + U64 * Fr(
        1, 3
    )
    for ry in (Fr(3), Fr(4)):
        with pytest.raises(c.Invalid):
            c.div(x, c.Cert(Fr(3), ry), B64)


def test_casts_narrowing_and_widening() -> None:
    third = c.Cert(Fr(1, 3), Fr(0))
    assert c.cast(third, B32).radius == U32 * Fr(1, 3)
    assert c.cast(c.exact(0.5), B32).radius == 0
    widened = c.cast(c.Cert(Fr(1, 3), Fr(1, 10)), B64, narrowing=False)
    assert widened.radius == Fr(1, 10)


def test_overflow_is_refused() -> None:
    big = c.exact(1.0e308)
    with pytest.raises(c.Invalid):
        c.mul(big, c.exact(10.0), B64)
    with pytest.raises(c.Invalid):
        c.cast(c.exact(1.0e39), B32)


def test_results_near_underflow_carry_the_absolute_term() -> None:
    # An exact zero with a nonzero radius may round in the subnormal band, where u|z| alone does
    # not bound the error: the rule adds eta = 2^(emin - p).
    z = c.sub(c.Cert(Fr(1), Fr(1, 2**60)), c.Cert(Fr(1), Fr(0)), B64)
    assert z.value == 0
    assert z.radius == (1 + U64) * Fr(1, 2**60) + B64.eta


@pytest.mark.parametrize("fmt", ["binary64", "binary32"])
def test_radii_bound_ieee_arithmetic(fmt: str) -> None:
    # Operands are exact values rounded to the format (so |x_hat - x| <= their radius); every
    # result computed by the hardware must lie within the propagated radius of the exact result.
    rng = np.random.default_rng(43)
    f = B64 if fmt == "binary64" else B32
    to = float if fmt == "binary64" else (lambda v: float(np.float32(v)))
    ops = {
        "add": (c.add, lambda a, b: a + b),
        "sub": (c.sub, lambda a, b: a - b),
        "mul": (c.mul, lambda a, b: a * b),
        "div": (c.div, lambda a, b: a / b),
    }
    for _ in range(4000):
        xs = [
            Fr(int(rng.integers(1, 2**40)), int(rng.integers(1, 2**40)))
            * (-1) ** int(rng.integers(2))
            for _ in range(2)
        ]
        hats = [to(float(v)) for v in xs]
        certs = [c.cast(c.Cert(v, Fr(0)), f) for v in xs]
        assert all(abs(Fr(h) - v) <= ce.radius for h, v, ce in zip(hats, xs, certs, strict=True))
        for name, (rule, hw) in ops.items():
            out = rule(certs[0], certs[1], f)
            if fmt == "binary64":
                got = hw(hats[0], hats[1])
            else:
                got = float(hw(np.float32(hats[0]), np.float32(hats[1])))
            assert abs(Fr(got) - out.value) <= out.radius, (name, xs, hats, got)


def test_nothing_outside_the_tests_imports_the_oracle() -> None:
    # The oracle is verification tooling: the generator and the kernels never depend on it.
    root = Path(__file__).resolve().parents[2]
    sources = [*(root / "codegen" / "src").rglob("*.py"), *(root / "kernels").rglob("*.py")]
    offenders = [p for p in sources if "oracle" in p.read_text()]
    assert sources
    assert not offenders, offenders
