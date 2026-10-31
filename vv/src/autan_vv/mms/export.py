"""Export manufactured solutions to a C++20 header and to JSON for the solver and its kernel generator."""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, Any

import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter

from autan_vv.errors import VVError
from autan_vv.mms.solutions import ManufacturedSolution, X

if TYPE_CHECKING:
    from collections.abc import Mapping

JSON_SCHEMA = "autan.mms/1"
REFERENCES: dict[str, str] = {
    "roache2002": "P. J. Roache, Code verification by the method of manufactured solutions, "
    "J. Fluids Eng. 124(1):4-10, 2002. doi:10.1115/1.1436090",
    "salari2000": "K. Salari and P. Knupp, Code verification by the method of manufactured solutions, "
    "Sandia report SAND2000-1444, 2000. doi:10.2172/759450",
    "roy2004": "C. J. Roy, C. C. Nelson, T. M. Smith and C. C. Ober, Verification of Euler/Navier-Stokes codes "
    "using the method of manufactured solutions, Int. J. Numer. Meth. Fluids 44(6):599-620, 2004. "
    "doi:10.1002/fld.660",
    "spalart1994": "P. R. Spalart and S. R. Allmaras, A one-equation turbulence model for aerodynamic flows, "
    "La Recherche Aerospatiale 1:5-21, 1994.",
    "allmaras2012": "S. R. Allmaras, F. T. Johnson and P. R. Spalart, Modifications and clarifications for the "
    "implementation of the Spalart-Allmaras turbulence model, ICCFD7-1902, 2012.",
    "eca2007": "L. Eca, M. Hoekstra, A. Hay and D. Pelletier, On the construction of manufactured solutions for "
    "one and two-equation eddy-viscosity models, Int. J. Numer. Meth. Fluids 54(2):119-154, 2007. "
    "doi:10.1002/fld.1387",
    "oliver2012": "T. A. Oliver, K. C. Estacio-Hiroms, N. Malaya and G. F. Carey, Manufactured solutions for the "
    "Favre-averaged Navier-Stokes equations with eddy-viscosity turbulence models, AIAA 2012-0080. "
    "doi:10.2514/6.2012-80",
    "tmr_sa": "NASA Turbulence Modeling Resource, The Spalart-Allmaras turbulence model, "
    "https://tmbwg.github.io/turbmodels/spalart.html",
}


def _outputs(ms: ManufacturedSolution) -> list[tuple[str, str, sp.Expr]]:
    """Every exported expression as (group, name, expr), in a fixed order."""
    out: list[tuple[str, str, sp.Expr]] = []
    out += [("fields", k, v) for k, v in ms.fields.items()]
    for k, g in ms.gradients().items():
        out += [("gradients", f"{k}_{axis}", gi) for axis, gi in zip("xyz", g, strict=True)]
    out += [("conserved", k, v) for k, v in ms.conserved.items()]
    out += [("sources", k, v) for k, v in ms.sources.items()]
    out += [("auxiliary", k, v) for k, v in ms.auxiliary.items()]
    return out


def _cse(exprs: list[sp.Expr]) -> tuple[list[tuple[sp.Symbol, sp.Expr]], list[sp.Expr]]:
    subs, reduced = sp.cse(
        exprs, symbols=sp.numbered_symbols("t"), optimizations="basic", order="none"
    )
    return list(subs), list(reduced)


class _CxxPrinter(CXX17CodePrinter):  # type: ignore[misc]
    """Prints literals as T(...) so one template body serves float and double without promotion to double."""

    def _print_Pi(self, _expr: sp.Expr) -> str:
        return "std::numbers::pi_v<T>"

    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"(T({expr.p}) / T({expr.q}))"

    def _print_Integer(self, expr: sp.Integer) -> str:
        return f"T({expr.p})"

    def _print_Float(self, expr: sp.Float) -> str:
        return f"T({float(expr)!r})"

    def _print_Pow(self, expr: sp.Pow) -> str:
        if expr.exp.is_Integer and 1 <= abs(int(expr.exp)) <= 4:
            base = self.parenthesize(expr.base, 1000)
            product = "*".join([base] * abs(int(expr.exp)))
            return f"({product})" if expr.exp > 0 else f"(T(1) / ({product}))"
        if expr.exp == sp.Rational(1, 2):
            return f"std::sqrt({self._print(expr.base)})"
        return f"std::pow({self._print(expr.base)}, {self._print(expr.exp)})"


class _JsonPrinter(C99CodePrinter):  # type: ignore[misc]
    """C99 syntax with the constant pi spelt `pi` and rationals as exact decimal quotients."""

    def _print_Pi(self, _expr: sp.Expr) -> str:
        return "pi"

    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"({expr.p}.0/{expr.q}.0)"


def _cxx_identifier(name: str) -> str:
    return (
        name if name.isidentifier() else "_" + "".join(ch if ch.isalnum() else "_" for ch in name)
    )


def to_cpp(ms: ManufacturedSolution, namespace: str = "autan::mms") -> str:
    """Return a self-contained C++20 header evaluating the solution at a point, for float or double.

    Each group (`fields`, `gradients`, `conserved`, `sources`, `auxiliary`) is one function
    `template <typename T> void group(T x, T y, T z, T* out)` writing the entries in the order of the
    matching `*_names` array. Functions are `__host__ __device__` under CUDA or HIP.
    """
    printer = _CxxPrinter()
    outputs = _outputs(ms)
    groups: dict[str, list[tuple[str, sp.Expr]]] = {}
    for group, name, expr in outputs:
        groups.setdefault(group, []).append((name, expr))
    lines = [
        "// Generated by autan-vv from a symbolic derivation (vv/src/autan_vv/mms). Do not edit.",
        f"// Manufactured solution '{ms.name}': {ms.description}",
        f"// Equations: {ms.equations}. References: {', '.join(ms.references)} (see docs/validation/mms.md).",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cmath>",
        "#include <numbers>",
        "#include <string_view>",
        "",
        "#if defined(__CUDACC__) || defined(__HIPCC__)",
        "#define AUTAN_MMS_HD __host__ __device__",
        "#else",
        "#define AUTAN_MMS_HD",
        "#endif",
        "",
        f"namespace {namespace}::{ms.name} {{",
        "",
        f'inline constexpr std::string_view name = "{ms.name}";',
        "inline constexpr std::array<double, 3> domain_lower = {0.0, 0.0, 0.0};",
        "inline constexpr std::array<double, 3> domain_upper = {1.0, 1.0, 1.0};",
        "",
        "struct Parameters {",
    ]
    lines += [
        f"  static constexpr double {_cxx_identifier(k)} = {v!r};" for k, v in ms.parameters.items()
    ]
    lines += ["};", ""]
    for group, items in groups.items():
        names = ", ".join(f'"{n}"' for n, _ in items)
        lines.append(
            f"inline constexpr std::array<std::string_view, {len(items)}> {group}_names = {{{names}}};"
        )
    lines.append("")
    for group, items in groups.items():
        subs, reduced = _cse([e for _, e in items])
        lines += [
            "template <typename T>",
            f"AUTAN_MMS_HD inline void {group}(T x, T y, T z, T* out) {{",
            "  (void)x; (void)y; (void)z;",
        ]
        lines += [f"  const T {s} = {printer.doprint(e)};" for s, e in subs]
        lines += [
            f"  out[{i}] = {printer.doprint(e)};  // {n}"
            for i, ((n, _), e) in enumerate(zip(items, reduced, strict=True))
        ]
        lines += ["}", ""]
    lines += [f"}}  // namespace {namespace}::{ms.name}", ""]
    return "\n".join(lines)


def to_json(ms: ManufacturedSolution) -> dict[str, Any]:
    """Return the solution as a JSON-ready document (schema `autan.mms/1`, see docs/validation/mms.md)."""
    printer = _JsonPrinter()
    groups: dict[str, dict[str, Any]] = {}
    outputs = _outputs(ms)
    for group in dict.fromkeys(g for g, _, _ in outputs):
        items = [(n, e) for g, n, e in outputs if g == group]
        subs, reduced = _cse([e for _, e in items])
        groups[group] = {
            "subexpressions": [[str(s), printer.doprint(e)] for s, e in subs],
            "outputs": {n: printer.doprint(e) for (n, _), e in zip(items, reduced, strict=True)},
        }
    return {
        "schema": JSON_SCHEMA,
        "name": ms.name,
        "equations": ms.equations,
        "description": ms.description,
        "coordinates": [str(c) for c in X],
        "domain": {"lower": [0.0, 0.0, 0.0], "upper": [1.0, 1.0, 1.0]},
        "parameters": dict(ms.parameters),
        "validity": [
            {"name": v.name, "lower": _finite_or_none(v.lower), "upper": _finite_or_none(v.upper)}
            for v in ms.validity
        ],
        "references": {k: REFERENCES[k] for k in ms.references},
        **groups,
    }


def _finite_or_none(v: float) -> float | None:
    return v if abs(v) != float("inf") else None


def dump_json(ms: ManufacturedSolution) -> str:
    """Serialise `to_json` deterministically."""
    return json.dumps(to_json(ms), indent=1, allow_nan=False) + "\n"


_FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pow": sp.Pow,
}
_BINARY: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def _expression(text: str, names: dict[str, sp.Expr]) -> sp.Expr:
    """Build a sympy expression from the restricted grammar of docs/validation/mms.md, never eval."""

    def build(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](build(node.left), build(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            inner = build(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return (
                sp.Rational(str(node.value))
                if isinstance(node.value, float)
                else sp.Integer(node.value)
            )
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(build(a) for a in node.args))
        msg = f"forbidden syntax in an MMS expression: {ast.dump(node)[:80]}"
        raise VVError(msg, code="V0202")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        msg = f"unparsable MMS expression: {text[:80]!r}"
        raise VVError(msg, code="V0202") from exc
    return build(tree.body)


def parse_json_expressions(doc: Mapping[str, Any], group: str) -> dict[str, sp.Expr]:
    """Rebuild the sympy expressions of one group from a JSON document (a reference reader).

    Only the grammar of the format is accepted (V0202 otherwise); nothing in the document is
    executed.
    """
    names: dict[str, sp.Expr] = {"pi": sp.pi} | {str(c): c for c in X}
    for name, text in doc[group]["subexpressions"]:
        names[name] = _expression(text, names)
    return {name: _expression(text, names) for name, text in doc[group]["outputs"].items()}
