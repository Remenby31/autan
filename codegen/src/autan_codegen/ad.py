"""Tangent and adjoint of IR functions by source transformation.

Branches are linearised on the side the primal selected (research section 4.3, non-smooth
operations): the derivative of select(c, a, b) is select(c, da, db). `rel` differences of Geom
inputs are passive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen.ir import Function, Instr, optimise

if TYPE_CHECKING:
    from collections.abc import Sequence

ZERO = None  # a structurally zero derivative


class _Out:
    """Appends instructions to a function, folding the structural zeros of derivatives."""

    def __init__(self, fn: Function) -> None:
        self.fn = fn
        self.consts: dict[float, int] = {}

    def const(self, x: float) -> int:
        if x not in self.consts:
            self.consts[x] = self.fn.add(Instr("const", (), x))
        return self.consts[x]

    def op(self, op: str, *args: int, line: int = 0) -> int:
        return self.fn.add(Instr(op, tuple(args), None, line))

    def add(self, a: int | None, b: int | None, line: int = 0) -> int | None:
        if a is None:
            return b
        if b is None:
            return a
        return self.op("add", a, b, line=line)

    def mul(self, a: int | None, b: int, line: int = 0) -> int | None:
        return None if a is None else self.op("mul", a, b, line=line)

    def neg(self, a: int | None, line: int = 0) -> int | None:
        return None if a is None else self.op("neg", a, line=line)

    def real(self, a: int | None) -> int:
        return self.const(0.0) if a is None else a


def _copy_primal(src: Function, dst: Function) -> None:
    for ins in src.instrs:
        dst.add(ins)


def _active(fn: Function, active_args: set[int]) -> set[int]:
    act: set[int] = set()
    for v, ins in enumerate(fn.instrs):
        if ins.op == "input":
            if ins.attr[0] in active_args:  # type: ignore[index]
                act.add(v)
        elif ins.op not in ("rel", "const") and not ins.is_bool and any(a in act for a in ins.args):
            act.add(v)
    return act


def tangent(fn: Function, active_args: Sequence[int]) -> Function:
    """Inputs: the primal arguments, then one tangent argument per active argument (same size).
    Outputs: the primal outputs, then their tangents."""
    n_args = len(fn.arg_sizes)
    out = Function(
        f"{fn.name}_tangent",
        [*fn.arg_names, *(f"{fn.arg_names[k]}_tan" for k in active_args)],
        [*fn.arg_sizes, *(fn.arg_sizes[k] for k in active_args)],
        source=fn.source,
    )
    _copy_primal(fn, out)
    o = _Out(out)
    tan_arg = {k: n_args + j for j, k in enumerate(active_args)}
    act = _active(fn, set(active_args))
    d: dict[int, int | None] = {}
    for v, ins in enumerate(fn.instrs):
        if v not in act:
            continue
        a = ins.args
        da = [d.get(x) for x in a]
        ln = ins.line
        match ins.op:
            case "input":
                attr = ins.attr
                assert isinstance(attr, tuple)
                d[v] = out.add(Instr("input", (), (tan_arg[attr[0]], attr[1]), ln))
            case "add":
                d[v] = o.add(da[0], da[1], ln)
            case "sub":
                d[v] = o.add(da[0], o.neg(da[1], ln), ln)
            case "mul":
                d[v] = o.add(o.mul(da[0], a[1], ln), o.mul(da[1], a[0], ln), ln)
            case "div":
                # d(x / y) = (dx - v dy) / y
                num = o.add(da[0], o.neg(o.mul(da[1], v, ln), ln), ln)
                d[v] = None if num is None else o.op("div", num, a[1], line=ln)
            case "neg":
                d[v] = o.neg(da[0], ln)
            case "sqrt":
                two_v = o.op("mul", o.const(2.0), v, line=ln)
                d[v] = o.op("div", da[0], two_v, line=ln)  # type: ignore[arg-type]
            case "exp":
                d[v] = o.mul(da[0], v, ln)
            case "log":
                d[v] = o.op("div", da[0], a[0], line=ln)  # type: ignore[arg-type]
            case "sin":
                d[v] = o.mul(da[0], o.op("cos", a[0], line=ln), ln)
            case "cos":
                d[v] = o.neg(o.mul(da[0], o.op("sin", a[0], line=ln), ln), ln)
            case "select":
                t, f = da[1], da[2]
                d[v] = (
                    None
                    if t is None and f is None
                    else o.op("select", a[0], o.real(t), o.real(f), line=ln)
                )
            case _:
                raise AssertionError(ins.op)
    out.outputs = [*fn.outputs, *(o.real(d.get(y)) for y in fn.outputs)]
    return optimise(out)


def adjoint(fn: Function, active_args: Sequence[int], seeds: Sequence[Sequence[float]]) -> Function:
    """Inputs: the primal arguments, then one seed argument per row of `seeds`, each of the size of
    the outputs; the output adjoint is sum_i seeds[i][o] * seed_i[o]. Outputs: the adjoint of each
    active argument, in order, flattened."""
    n_out = len(fn.outputs)
    n_args = len(fn.arg_sizes)
    out = Function(
        f"{fn.name}_adjoint",
        [*fn.arg_names, *(f"seed{i}" for i in range(len(seeds)))],
        [*fn.arg_sizes, *(n_out for _ in seeds)],
        source=fn.source,
    )
    _copy_primal(fn, out)
    o = _Out(out)
    act = _active(fn, set(active_args))
    bar: dict[int, int | None] = {}
    # Adjoints that a select may have zeroed (the branch it did not take). The primal of that
    # branch may be NaN or infinite (sqrt of a negative, division by zero), so their products
    # with partial derivatives are guarded: a zero adjoint contributes zero, never 0 * inf.
    masked: set[int] = set()

    def acc(x: int, contribution: int | None, mask: bool = False) -> None:
        if x in act and contribution is not None:
            bar[x] = o.add(bar.get(x), contribution)
            if mask or contribution in masked or bar.get(x) in masked:
                masked.add(bar[x])  # type: ignore[arg-type]

    def scaled(b: int, product: int) -> int:
        """`product` (b times a partial) with b == 0 giving 0 when b may be a zeroed branch."""
        if b not in masked:
            return product
        z = o.const(0.0)
        nonzero = o.op("or", o.op("lt", b, z), o.op("gt", b, z))
        r = o.op("select", nonzero, product, z)
        masked.add(r)
        return r

    for k, y in enumerate(fn.outputs):
        total: int | None = None
        for i, row in enumerate(seeds):
            if row[k] == 0.0:
                continue
            s = out.add(Instr("input", (), (n_args + i, k)))
            term = s if row[k] == 1.0 else o.op("mul", o.const(row[k]), s)
            total = o.add(total, term)
        acc(y, total)
    inputs: dict[tuple[int, int], int] = {}
    for v in range(len(fn.instrs) - 1, -1, -1):
        ins = fn.instrs[v]
        if ins.op == "input":
            inputs[ins.attr] = v  # type: ignore[index]
            continue
        b = bar.get(v)
        if b is None or v not in act:
            continue
        a = ins.args
        ln = ins.line
        m = b in masked
        match ins.op:
            case "add":
                acc(a[0], b, m)
                acc(a[1], b, m)
            case "sub":
                acc(a[0], b, m)
                acc(a[1], o.neg(b, ln), m)
            case "mul":
                acc(a[0], scaled(b, o.op("mul", b, a[1], line=ln)))
                acc(a[1], scaled(b, o.op("mul", b, a[0], line=ln)))
            case "div":
                q = scaled(b, o.op("div", b, a[1], line=ln))
                acc(a[0], q)
                acc(a[1], o.neg(scaled(b, o.op("mul", q, v, line=ln)), ln), b in masked)
            case "neg":
                acc(a[0], o.neg(b, ln), m)
            case "sqrt":
                two_v = o.op("mul", o.const(2.0), v, line=ln)
                acc(a[0], scaled(b, o.op("div", b, two_v, line=ln)))
            case "exp":
                acc(a[0], scaled(b, o.op("mul", b, v, line=ln)))
            case "log":
                acc(a[0], scaled(b, o.op("div", b, a[0], line=ln)))
            case "sin":
                acc(a[0], scaled(b, o.op("mul", b, o.op("cos", a[0], line=ln), line=ln)))
            case "cos":
                acc(
                    a[0],
                    o.neg(scaled(b, o.op("mul", b, o.op("sin", a[0], line=ln), line=ln)), ln),
                    b in masked,
                )
            case "select":
                z = o.const(0.0)
                acc(a[1], o.op("select", a[0], b, z, line=ln), True)
                acc(a[2], o.op("select", a[0], z, b, line=ln), True)
            case _:
                raise AssertionError(ins.op)
    outputs = []
    for k in active_args:
        for c in range(fn.arg_sizes[k]):
            iv = inputs.get((k, c))
            outputs.append(o.real(None if iv is None else bar.get(iv)))
    out.outputs = outputs
    return optimise(out)
