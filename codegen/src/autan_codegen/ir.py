"""Scalar SSA IR of a pointwise body.

Vectors and matrices are scalarised by the frontend, so every value is a real or a boolean scalar
and every instruction defines exactly one value, numbered by its position. A function has inputs
(one per scalar component of each argument), instructions, and outputs (value numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# op -> number of operands; result kind is "bool" for comparisons, "real" otherwise.
ARITY = {
    "input": 0,
    "const": 0,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    "sqrt": 1,
    "exp": 1,
    "log": 1,
    "sin": 1,
    "cos": 1,
    "rel": 2,  # difference of two raw Geom inputs, in their storage type, then converted
    "lt": 2,
    "le": 2,
    "gt": 2,
    "ge": 2,
    "and": 2,
    "or": 2,
    "not": 1,
    "select": 3,  # select(cond, if_true, if_false)
}
BOOL_OPS = frozenset({"lt", "le", "gt", "ge", "and", "or", "not"})


@dataclass(frozen=True)
class Instr:
    op: str
    args: tuple[int, ...] = ()
    attr: object = None  # const: float; input: (argument index, component index)
    line: int = 0

    @property
    def is_bool(self) -> bool:
        return self.op in BOOL_OPS


@dataclass
class Function:
    """A scalarised pointwise function. `arg_sizes[i]` scalars belong to argument i."""

    name: str
    arg_names: list[str]
    arg_sizes: list[int]
    instrs: list[Instr] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    source: str = ""

    def add(self, instr: Instr) -> int:
        self.instrs.append(instr)
        return len(self.instrs) - 1

    def inputs(self) -> dict[tuple[int, int], int]:
        return {i.attr: v for v, i in enumerate(self.instrs) if i.op == "input"}  # type: ignore[misc]


def optimise(fn: Function) -> Function:
    """Constant folding of trivial identities, value numbering and dead-code elimination."""
    out = Function(fn.name, fn.arg_names, fn.arg_sizes, source=fn.source)
    remap: dict[int, int] = {}
    table: dict[tuple[object, ...], int] = {}

    def const_of(v: int) -> float | None:
        i = out.instrs[v]
        return i.attr if i.op == "const" else None  # type: ignore[return-value]

    def emit(instr: Instr) -> int:
        key = (instr.op, instr.args, instr.attr)
        if instr.op in ("add", "mul", "and", "or"):
            key = (instr.op, tuple(sorted(instr.args)), instr.attr)
        if key in table:
            return table[key]
        v = out.add(instr)
        table[key] = v
        return v

    for old, ins in enumerate(fn.instrs):
        args = tuple(remap[a] for a in ins.args)
        new = Instr(ins.op, args, ins.attr, ins.line)
        c = [const_of(a) for a in args]
        if ins.op == "mul" and (c[0] == 1.0 or c[1] == 1.0):
            remap[old] = args[1] if c[0] == 1.0 else args[0]
            continue
        if ins.op == "add" and (c[0] == 0.0 or c[1] == 0.0):
            remap[old] = args[1] if c[0] == 0.0 else args[0]
            continue
        if ins.op == "sub" and c[1] == 0.0:
            remap[old] = args[0]
            continue
        if ins.op == "div" and c[1] == 1.0:
            remap[old] = args[0]
            continue
        if ins.op == "select" and args[1] == args[2]:
            remap[old] = args[1]
            continue
        remap[old] = emit(new)
    out.outputs = [remap[o] for o in fn.outputs]
    return dce(out)


def dce(fn: Function) -> Function:
    live = set(fn.outputs)
    for v in range(len(fn.instrs) - 1, -1, -1):
        if v in live:
            live.update(fn.instrs[v].args)
    # Inputs are kept so that the function signature never changes.
    keep = [v for v, i in enumerate(fn.instrs) if v in live or i.op == "input"]
    index = {v: k for k, v in enumerate(keep)}
    out = Function(fn.name, fn.arg_names, fn.arg_sizes, source=fn.source)
    for v in keep:
        i = fn.instrs[v]
        out.add(Instr(i.op, tuple(index[a] for a in i.args), i.attr, i.line))
    out.outputs = [index[o] for o in fn.outputs]
    return out
