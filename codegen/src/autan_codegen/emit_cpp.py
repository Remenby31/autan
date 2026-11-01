"""C++20 emission shared by every backend: bodies, the data struct, loads and stores.

Bodies are `template <class P> AUTAN_HD` functions of the precision policy P. Arguments arrive in
their storage type, are converted to `P::compute` on first use, and `rel` subtracts raw Geom values
in their storage type before converting. Backends emit loops around these bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.model import ROLE_TYPE, Module

if TYPE_CHECKING:
    from autan_codegen.frontend import Signature
    from autan_codegen.ir import Function

INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "and": "&&", "or": "||"}
CALL = {"sqrt": "sqrt", "exp": "exp", "log": "log", "sin": "sin", "cos": "cos"}
CMP = {"lt", "le", "gt", "ge"}


def ctype(role: dsl.Role, policy: str = "P") -> str:
    return f"typename {policy}::{ROLE_TYPE[role]}"


def literal(x: float) -> str:
    r = repr(float(x))
    if r in ("inf", "-inf", "nan"):
        raise ValueError(r)
    return r if ("." in r or "e" in r) else r + ".0"


@dataclass
class Emitter:
    """Collects lines; `line_directives` maps body statements to the DSL source with #line."""

    header_name: str
    line_directives: bool = True

    def __post_init__(self) -> None:
        self.lines: list[str] = []

    def w(self, s: str = "") -> None:
        self.lines.extend(s.split("\n"))

    def source_line(self, line: int, path: str) -> None:
        if self.line_directives and line > 0:
            self.lines.append(f'#line {line} "{path}"')

    def restore_line(self) -> None:
        if self.line_directives:
            self.lines.append(f'#line {len(self.lines) + 2} "{self.header_name}"')

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def arg_name(k: int, c: int) -> str:
    return f"a{k}_{c}"


def emit_function(
    e: Emitter, fn: Function, sig_types: list[dsl.Type], n_out: int, name: str | None = None
) -> None:
    """Emit `fn` as `template <class P> AUTAN_HD inline void name(args..., C* out)`."""
    params = [
        f"[[maybe_unused]] const {ctype(t.role)} {arg_name(k, c)}"
        for k, t in enumerate(sig_types)
        for c in range(t.size)
    ]
    params.append("typename P::compute* __restrict__ out" if n_out else "typename P::compute*")
    e.w("template <class P>")
    e.w(f"AUTAN_HD inline void {name or fn.name}({', '.join(params)}) {{")
    e.w("  using C = typename P::compute;")
    body_lines(e, fn, indent="  ")
    for k, o in enumerate(fn.outputs):
        e.w(f"  out[{k}] = v{o};")
    e.restore_line()
    e.w("}")
    e.w()


def body_lines(e: Emitter, fn: Function, indent: str) -> None:
    users_non_rel: set[int] = set()
    for ins in fn.instrs:
        if ins.op != "rel":
            users_non_rel.update(ins.args)
    users_non_rel.update(fn.outputs)
    last_line = -1
    for v, ins in enumerate(fn.instrs):
        if ins.line != last_line:
            e.source_line(ins.line, fn.source)
            last_line = ins.line
        a = [f"v{x}" for x in ins.args]
        if ins.op == "input":
            if v not in users_non_rel:
                continue
            e.w(f"{indent}const C v{v} = C({arg_name(*ins.attr)});")  # type: ignore[misc]
            continue
        if ins.op == "const":
            e.w(f"{indent}const C v{v} = C({literal(ins.attr)});")  # type: ignore[arg-type]
            continue
        if ins.op == "rel":
            x, y = (fn.instrs[i].attr for i in ins.args)
            e.w(f"{indent}const C v{v} = C({arg_name(*x)} - {arg_name(*y)});")  # type: ignore[misc]
            continue
        if ins.op in CMP:
            e.w(f"{indent}const bool v{v} = ::autan::kernels::{ins.op}({a[0]}, {a[1]});")
            continue
        if ins.op in ("and", "or"):
            e.w(f"{indent}const bool v{v} = {a[0]} {INFIX[ins.op]} {a[1]};")
            continue
        if ins.op == "not":
            e.w(f"{indent}const bool v{v} = !{a[0]};")
            continue
        if ins.op in INFIX:
            e.w(f"{indent}const C v{v} = {a[0]} {INFIX[ins.op]} {a[1]};")
        elif ins.op == "neg":
            e.w(f"{indent}const C v{v} = -{a[0]};")
        elif ins.op in CALL:
            e.w(f"{indent}const C v{v} = {CALL[ins.op]}({a[0]});")
        elif ins.op == "select":
            e.w(f"{indent}const C v{v} = {a[0]} ? {a[1]} : {a[2]};")
        else:
            raise AssertionError(ins.op)


# ---------------------------------------------------------------------------- data access


def stride(d: dsl.Dataset) -> str:
    return {
        dsl.MeshSet.CELLS: "d.topo.n_cells",
        dsl.MeshSet.FACES: "d.topo.n_faces",
        dsl.MeshSet.INCIDENCES: "d.topo.n_incidences",
    }[d.set]


def at(d: dsl.Dataset | dsl.Parameter, index: str, c: int, layout: str = "soa") -> str:
    if isinstance(d, dsl.Parameter):
        return f"d.{d.name}[{c}]"
    if d.type.size == 1:
        return f"d.{d.name}[{index}]"
    if layout == "aos":
        return f"d.{d.name}[{d.type.size} * ({index}) + {c}]"
    return f"d.{d.name}[{c} * {stride(d)} + ({index})]" if c else f"d.{d.name}[{index}]"


def emit_data_struct(e: Emitter, m: Module) -> None:
    e.w(
        "/// Pointers to every dataset the module's loops use (host or device), and parameter values."
    )
    e.w("/// Datasets are structure-of-arrays: component c of entity i is at c * n + i.")
    e.w("template <class P>")
    e.w("struct Data {")
    e.w("  ::autan::kernels::Topology topo;")
    for d in m.datasets:
        const = "const " if d.geometry else ""
        e.w(
            f"  {const}{ctype(d.type.role)}* {d.name};  // {d.set.value}, {d.type.size} component(s)"
        )
    for p in m.params:
        e.w(f"  {ctype(p.type.role)} {p.name}[{p.type.size}];")
    e.w("};")
    e.w()


def call_args(
    sig: Signature,
    bindings: dict[str, dsl.Binding],
    index_of: dict[str, str],
    skip: int = 0,
    layout: str = "soa",
) -> list[str]:
    """Loads of each scalar argument; `index_of[via]` is the index expression of that binding."""
    out = []
    for name, t in zip(sig.names[skip:], sig.types[skip:], strict=True):
        b = bindings[name]
        for c in range(t.size):
            if b.via == "other":
                bnd = (
                    at(b.boundary, "f", c, layout)
                    if b.boundary is not None
                    else f"{ctype(t.role)}(0)"
                )
                out.append(
                    f"(f < d.topo.n_interior_faces ? {at(b.dataset, 'nb', c, layout)} : {bnd})"
                )
            else:
                out.append(at(b.dataset, index_of[b.via], c, layout))
    return out


def unique_bodies(m: Module) -> list[tuple[Function, Signature]]:
    seen: set[str] = set()
    out = []
    for fn, sig in m.bodies:
        if fn.name not in seen:
            seen.add(fn.name)
            out.append((fn, sig))
    return out


def emit_common_header(m: Module, line_directives: bool = True) -> str:
    """`<module>.hpp`: the Data struct and every body of the module."""
    e = Emitter(f"{m.name}.hpp", line_directives)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w("#pragma once")
    e.w('#include "autan/kernels/runtime.hpp"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name} {{")
    e.w("using std::cos;")
    e.w("using std::exp;")
    e.w("using std::log;")
    e.w("using std::sin;")
    e.w("using std::sqrt;")
    e.w()
    emit_data_struct(e, m)
    for fn, sig in unique_bodies(m):
        emit_function(e, fn, sig.types, len(fn.outputs))
    for fn, types in m.extra_bodies:
        emit_function(e, fn, types, len(fn.outputs))
    e.w(f"}}  // namespace autan::kernels::{m.name}")
    return e.text()
