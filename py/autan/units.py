"""Physical units for case values.

Multiply a number by a unit to build a quantity: ``7.0 * m``, ``19.57 * deg``,
``101325 * Pa``. Every dimensional field of the case checks the dimension of
the quantity it receives; see ``docs/contracts/case-schema.md#quantities``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import pint
from pydantic import PlainSerializer
from pydantic_core import PydanticCustomError, core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue

ureg = pint.UnitRegistry()
Quantity = ureg.Quantity

m = ureg.meter
mm = ureg.millimeter
s = ureg.second
kg = ureg.kilogram
mol = ureg.mole
K = ureg.kelvin
Pa = ureg.pascal
kPa = ureg.kilopascal
J = ureg.joule
W = ureg.watt
deg = ureg.degree
rad = ureg.radian

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_QUANTITY_PATTERN = rf"^\s*{_NUMBER}\s+\S.*$"
_QUANTITY = re.compile(rf"\s*({_NUMBER})\s+(\S.*?)\s*")
_BARE_NUMBER = re.compile(rf"\s*{_NUMBER}\s*")


def fail(code: str, template: str, **context: object) -> NoReturn:
    """Raise a pydantic error whose type carries a diagnostic code (``autan.E0301``)."""
    kind = f"autan.{code}"
    raise PydanticCustomError(kind, template, context)


@dataclass(frozen=True)
class Dimension:
    """Field marker: a quantity of ``dimension`` (a Pint dimensionality string).

    ``"[angle]"`` is special: Pint treats angles as dimensionless, so an angle
    is a quantity whose root unit is the radian.
    """

    dimension: str
    positive: bool = False

    def __get_pydantic_core_schema__(
        self, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(self.validate)

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        bound = " > 0" if self.positive else ""
        return {
            "type": "string",
            "pattern": _QUANTITY_PATTERN,
            "description": f'Quantity "<number> <unit>" of dimension {self.dimension}{bound}.',
            "x-autan-dimension": self.dimension,
        }

    def validate(self, value: object) -> pint.Quantity:
        quantity = _to_quantity(value, self.dimension)
        if not _has_dimension(quantity, self.dimension):
            fail(
                "E0301",
                "expected a quantity of dimension {expected}, got {got} ({actual})",
                expected=self.dimension,
                got=repr(format_quantity(quantity)),
                actual=_describe_dimension(quantity),
            )
        magnitude = float(quantity.magnitude)
        if not math.isfinite(magnitude):
            fail("E0304", "{got} is not finite", got=repr(format_quantity(quantity)))
        if self.positive:
            absolute = quantity.to(ureg.kelvin) if self.dimension == "[temperature]" else quantity
            if float(absolute.magnitude) <= 0:
                fail("E0304", "{got} must be > 0", got=repr(format_quantity(quantity)))
        return quantity


def _to_quantity(value: object, dimension: str) -> pint.Quantity:
    if isinstance(value, pint.Quantity):
        if value._REGISTRY is not ureg:
            fail(
                "E0303",
                "{got} comes from another unit registry; build it with autan.units",
                got=repr(str(value)),
            )
        return Quantity(value.magnitude, value.units)
    if isinstance(value, str):
        return parse(value, dimension)
    if isinstance(value, int | float) and not isinstance(value, bool):
        fail(
            "E0302",
            "{got} has no unit; expected a quantity of dimension {expected}",
            got=repr(value),
            expected=dimension,
        )
    fail(
        "E0301",
        "expected a quantity of dimension {expected}, got {got}",
        expected=dimension,
        got=repr(value),
    )


def _has_dimension(quantity: pint.Quantity, dimension: str) -> bool:
    if dimension == "[angle]":
        return bool(ureg.get_root_units(quantity.units)[1] == ureg.radian)
    return bool(quantity.check(dimension)) and ureg.get_root_units(quantity.units)[1] != ureg.radian


def _describe_dimension(quantity: pint.Quantity) -> str:
    if ureg.get_root_units(quantity.units)[1] == ureg.radian:
        return "[angle]"
    return str(quantity.dimensionality) or "dimensionless"


def parse(text: str, dimension: str) -> pint.Quantity:
    """Read ``"<number> <unit>"``; the unit part may only multiply, divide and raise units."""
    if _BARE_NUMBER.fullmatch(text):
        fail("E0302", "{got} has no unit; expected {expected}", got=repr(text), expected=dimension)
    match = _QUANTITY.fullmatch(text)
    if match is None:
        fail("E0303", "{got} is not written as '<number> <unit>'", got=repr(text))
    try:
        unit = ureg.parse_units(match.group(2))
    except Exception as error:  # noqa: BLE001 -- Pint's parser raises many unrelated types
        fail(
            "E0303",
            "cannot read the unit of {got}: {reason}",
            got=repr(text),
            reason=f"{type(error).__name__}: {error}",
        )
    return Quantity(float(match.group(1)), unit)


def quantity(dimension: str, *, positive: bool = False) -> Any:
    """Annotation metadata of a quantity field: dimension check, and canonical text in JSON."""
    return (Dimension(dimension, positive), PlainSerializer(format_quantity, when_used="json"))


def format_quantity(quantity: pint.Quantity) -> str:
    """Canonical text of a quantity: shortest round-trip number, a space, Pint's short unit."""
    return f"{float(quantity.magnitude)!r} {quantity.units:~D}"


def validate_number(value: object, *, positive: bool) -> float:
    """Validate a dimensionless number; a quantity with a dimension is rejected (E0301)."""
    if isinstance(value, pint.Quantity):
        if not value.dimensionless or ureg.get_root_units(value.units)[1] == ureg.radian:
            fail(
                "E0301",
                "expected a dimensionless number, got {got}",
                got=repr(format_quantity(value)),
            )
        value = float(value.to(ureg.dimensionless).magnitude)
    if isinstance(value, bool) or not isinstance(value, int | float):
        fail("E0301", "expected a dimensionless number, got {got}", got=repr(value))
    number = float(value)
    if not math.isfinite(number):
        fail("E0304", "{got} is not finite", got=repr(number))
    if positive and number <= 0:
        fail("E0304", "{got} must be > 0", got=repr(number))
    return number


def si(quantity: pint.Quantity, unit: str) -> float:
    """Magnitude of ``quantity`` in ``unit``."""
    return float(quantity.to(unit).magnitude)
