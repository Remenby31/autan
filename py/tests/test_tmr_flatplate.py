"""The TMR 2DZP flat-plate script (physics/tmr/flatplate.py) against the case's definition at
tmbwg.github.io/turbmodels: M 0.2, Re 5e6 per metre, T 300 K, isentropic totals at the inflow,
freestream static pressure at the outflow, adiabatic plate."""

import math
from pathlib import Path

import autan
import pytest
from autan.units import Quantity

SCRIPT = Path(__file__).parents[2] / "physics" / "tmr" / "flatplate.py"
GAMMA = 1.4
GAS_CONSTANT = 8.314462618 / 0.0289647


def test_flat_plate_script_defines_the_tmr_case() -> None:
    document = autan.load(SCRIPT).spec.document()
    conditions = {rule["zones"]: rule["condition"] for rule in document["boundaries"]}
    assert {zone: c["kind"] for zone, c in conditions.items()} == {
        "sides": "symmetry",
        "plate": "wall",
        "symmetry": "symmetry",
        "inflow": "inlet",
        "outflow": "outlet",
        "top": "farfield",
    }
    assert "temperature" not in conditions["plate"]
    assert document["physics"]["turbulence"]["model"] == "sa"
    assert document["freestream"]["mach"] == 0.2
    assert Quantity(document["freestream"]["temperature"]).to("K").magnitude == 300.0

    # The outflow static pressure gives Re = rho V / mu = 5e6 per metre (Sutherland's law).
    p = Quantity(conditions["outflow"]["static_pressure"]).to("Pa").magnitude
    mu = 1.716e-5 * (300.0 / 273.15) ** 1.5 * (273.15 + 110.4) / (300.0 + 110.4)
    speed = 0.2 * math.sqrt(GAMMA * GAS_CONSTANT * 300.0)
    assert p / (GAS_CONSTANT * 300.0) * speed / mu == pytest.approx(5.0e6, rel=1e-9)

    # Isentropic totals of a Mach 0.2 stream.
    ratio = 1.0 + 0.5 * (GAMMA - 1.0) * 0.2**2
    total_pressure = Quantity(conditions["inflow"]["total_pressure"]).to("Pa").magnitude
    total_temperature = Quantity(conditions["inflow"]["total_temperature"]).to("K").magnitude
    assert total_pressure == pytest.approx(p * ratio ** (GAMMA / (GAMMA - 1.0)), rel=1e-9)
    assert total_temperature == pytest.approx(300.0 * ratio, rel=1e-9)

    (monitor,) = document["monitors"]
    assert monitor["surfaces"] == "plate"
    assert Quantity(monitor["ref_area"]).to("m**2").magnitude == pytest.approx(2.0)
