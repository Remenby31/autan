"""The canonical Python rendering of a case: SDK code that ``autan.load`` turns back into the
same case."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pint
from pydantic import BaseModel

from autan import gases, materials, units
from autan.schema import (
    CaseSpec,
    Compressible,
    Fluid,
    Freestream,
    Spec,
    Steady,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

WIDTH = 100
INDENT = "    "
# The temperature CaseSpec gives a compressible freestream that has none.
_FILLED_TEMPERATURE = 288.15 * units.K
_UNIT_NAMES = frozenset(name for name, value in vars(units).items() if isinstance(value, pint.Unit))
_LIBRARY = [
    (f"autan.{module.__name__.rsplit('.', 1)[1]}.{name}", value)
    for module in (gases, materials)
    for name, value in sorted(vars(module).items())
    if name.isupper() and isinstance(value, BaseModel)
]


class _Imports:
    def __init__(self) -> None:
        self.units: set[str] = set()

    def lines(self) -> list[str]:
        lines = ["import autan"]
        if self.units:
            lines.append(f"from autan.units import {', '.join(_sorted_names(self.units))}")
        return lines


def _sorted_names(names: Iterable[str]) -> list[str]:
    """ruff's isort order within one import: constants (uppercase, longer than one character),
    then classes, then the rest, each case-insensitively."""

    def kind(name: str) -> int:
        if len(name) > 1 and name.isupper():
            return 0
        return 1 if name[0].isupper() else 2

    return sorted(names, key=lambda name: (kind(name), name.lower(), name))


def to_python(spec: CaseSpec) -> str:
    """Render ``spec`` as SDK code (see ``autan.Case.to_python``)."""
    imports = _Imports()
    body = _statements(spec, imports)
    return "\n".join([*imports.lines(), "", *body]) + "\n"


def _statements(spec: CaseSpec, imports: _Imports) -> list[str]:
    value = _Values(imports)
    out = [
        _statement(
            "case", _Call("autan.Case", (value(spec.name), _Keyword("mesh", value(spec.mesh))))
        )
    ]
    out.append(_statement("case.physics", value(spec.physics)))
    out.append(_statement("", _Call("case.freestream", value.keywords(_freestream(spec)))))
    if spec.time != Steady():
        out.append(_statement("case.time", value(spec.time)))
    if spec.initial_state is not None:
        out.append(_statement("case.initial_state", value(spec.initial_state)))
    if spec.verification is not None:
        out.append(_statement("case.verification", value(spec.verification.manufactured)))
    for zone in spec.cell_zones:
        medium = zone.medium
        selector = f"case.cell_zone({_flat(value(zone.zones))})"
        arguments: tuple[_Expr | _Keyword, ...] = (value(medium.material),)
        if isinstance(medium, Fluid):
            if medium.initial is not None:
                arguments += (_Keyword("initial", value(medium.initial)),)
            out.append(_statement("", _Call(f"{selector}.fluid", arguments)))
        else:
            out.append(_statement("", _Call(f"{selector}.solid", arguments)))
    for rule in spec.boundaries:
        condition = rule.condition
        method = f"case.boundary({_flat(value(rule.zones))}).{condition.kind.replace('-', '_')}"
        out.append(_statement("", _Call(method, value.keywords(condition))))
    for monitor in spec.monitors:
        given = value.keywords(monitor, skip=frozenset({"surfaces", "every"}))
        forces = _Call("autan.forces", (value(monitor.surfaces), *given))
        every = (_Keyword("every", value(monitor.every)),) if monitor.every != 1 else ()
        out.append(_statement("", _Call("case.monitor", (forces, *every))))
    return out


def _freestream(spec: CaseSpec) -> Freestream:
    """The freestream as the case gave it: without the temperature CaseSpec filled in."""
    fs = spec.freestream
    if isinstance(spec.physics, Compressible) and fs.temperature == _FILLED_TEMPERATURE:
        return fs.model_copy(update={"temperature": None})
    return fs


class _Values:
    """Python expressions of case values."""

    def __init__(self, imports: _Imports) -> None:
        self.imports = imports

    def __call__(self, value: Any) -> _Expr:
        if isinstance(value, pint.Quantity):
            return self.quantity(value)
        if isinstance(value, bool | int | float) or value is None:
            return repr(value)
        if isinstance(value, str):
            return _string(value)
        if isinstance(value, tuple | list):
            return _Call("", tuple(self(item) for item in value))
        if isinstance(value, Spec):
            return self.model(value)
        message = f"cannot render {type(value).__name__}"
        raise TypeError(message)

    def model(self, model: Spec) -> _Expr:
        for name, library in _LIBRARY:
            if model == library:
                return name
        return _Call(f"autan.{type(model).__name__}", self.keywords(model))

    def keywords(
        self, model: BaseModel, skip: frozenset[str] = frozenset()
    ) -> tuple[_Keyword, ...]:
        """``key=value`` for each field of ``model`` that differs from its default."""
        found = []
        for key, field in type(model).model_fields.items():
            if key in skip or key in ("kind", "model"):
                continue
            mine = getattr(model, key)
            if not field.is_required() and mine == field.get_default(call_default_factory=True):
                continue
            found.append(_Keyword(key, self(mine)))
        return tuple(found)

    def quantity(self, quantity: pint.Quantity) -> str:
        magnitude = repr(float(quantity.magnitude))
        text = f"{quantity.units:~D}"
        symbols = set(re.findall(r"[A-Za-z_]\w*", text))
        if not symbols <= _UNIT_NAMES:
            return f"autan.units.Quantity({magnitude}, {_string(text)})"
        self.imports.units |= symbols
        expression = text.replace(" ** ", "**")
        if expression.startswith("1 / "):
            return f"{magnitude} / {expression[4:]}"
        return f"{magnitude} * {expression}"


def _string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


@dataclass(frozen=True)
class _Keyword:
    name: str
    value: _Expr


@dataclass(frozen=True)
class _Call:
    """``head(arguments)``; a tuple when ``head`` is empty."""

    head: str
    arguments: tuple[_Expr | _Keyword, ...]


_Expr = str | _Call


def _flat(expr: _Expr | _Keyword) -> str:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, _Keyword):
        return f"{expr.name}={_flat(expr.value)}"
    inner = ", ".join(_flat(argument) for argument in expr.arguments)
    if not expr.head and len(expr.arguments) == 1:
        inner += ","
    return f"{expr.head}({inner})"


def _layout(expr: _Expr | _Keyword, indent: int, column: int, tail: int) -> str:
    """``expr`` starting at ``column`` of a line indented by ``indent``, followed by ``tail``
    characters on its last line, broken the way ruff format breaks it."""
    flat = _flat(expr)
    if isinstance(expr, str) or column + len(flat) + tail <= WIDTH:
        return flat
    if isinstance(expr, _Keyword):
        prefix = f"{expr.name}="
        return prefix + _layout(expr.value, indent, column + len(prefix), tail)
    inner = indent + len(INDENT)
    joined = ", ".join(_flat(argument) for argument in expr.arguments)
    if not expr.head and len(expr.arguments) == 1:
        joined += ","
    if inner + len(joined) <= WIDTH:
        return f"{expr.head}(\n{' ' * inner}{joined}\n{' ' * indent})"
    if expr.head and len(expr.arguments) == 1:
        # One argument has no comma to split at: ruff breaks inside it, without a trailing comma.
        argument = _layout(expr.arguments[0], inner, inner, 0)
        return f"{expr.head}(\n{' ' * inner}{argument}\n{' ' * indent})"
    lines = "".join(
        f"{' ' * inner}{_layout(argument, inner, inner, 1)},\n" for argument in expr.arguments
    )
    return f"{expr.head}(\n{lines}{' ' * indent})"


def _statement(target: str, expr: _Expr) -> str:
    """``target = expr``, or the bare call ``expr`` when ``target`` is empty."""
    prefix = f"{target} = " if target else ""
    return prefix + _layout(expr, 0, len(prefix), 0)
