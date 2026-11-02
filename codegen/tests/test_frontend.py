"""DSL parsing: SSA semantics of names, branches and unrolled loops, and coded rejections."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from autan_codegen import dsl
from autan_codegen.ad import adjoint, jacobian, tangent
from autan_codegen.dsl import (
    Geom,
    Grad,
    Param,
    Residual,
    State,
    cells,
    dot,
    identity,
    inverse,
    outer,
    pointwise,
    rel,
    sqrt,
)
from autan_codegen.errors import CodegenError
from autan_codegen.frontend import parse
from autan_codegen.generate import generate
from conftest import ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from autan_codegen.ir import Function
from autan_codegen.model import build_loop

OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "rel": lambda a, b: a - b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "not": lambda a: not a,
    "select": lambda c, a, b: a if c else b,
}


def run(fn: Function, *args: float | tuple[float, ...]) -> list[float]:
    flat = [list(a) if isinstance(a, tuple) else [a] for a in args]
    v: list[object] = []
    for ins in fn.instrs:
        if ins.op == "input":
            k, c = ins.attr  # type: ignore[misc]
            v.append(np.float64(flat[k][c]))
        elif ins.op == "const":
            v.append(np.float64(ins.attr))  # type: ignore[arg-type]
        else:
            v.append(OPS[ins.op](*(v[a] for a in ins.args)))
    return [v[o] for o in fn.outputs]  # type: ignore[misc]


@pointwise
def reassigned_before_predicate(x: State) -> State:
    y = x
    y = y * 2.0
    return y - 1.0 if y > 1.0 else -y


@pointwise
def unrolled(v: State.Vec3) -> State:
    acc = 0.0
    for i in range(3):
        acc += v[i] * (i + 1)
    return acc


@pointwise
def helper(a: State, b: State) -> State:
    return a * b if a > b else a + b


@pointwise
def inlined_and_vectors(
    u: State.Vec3, x: Geom.Vec3, y: Geom.Vec3, k: Param
) -> tuple[State, State.Vec3]:
    d = rel(x, y)
    return helper(dot(u, d), k), abs(k) * u - d


@pointwise
def minmax_ties(a: State, b: State) -> tuple[State, State]:
    return min(a, b), max(a, b)


def test_branches_see_the_value_their_predicate_was_computed_from() -> None:
    fn, _ = parse(reassigned_before_predicate)
    assert run(fn, 0.75) == [0.5]
    assert run(fn, 0.25) == [-0.5]


def test_loops_with_constant_bounds_are_unrolled() -> None:
    fn, _ = parse(unrolled)
    assert run(fn, (1.0, 10.0, 100.0)) == [321.0]
    assert all(i.op != "select" for i in fn.instrs)


def test_calls_are_inlined_and_vectors_scalarised() -> None:
    fn, sig = parse(inlined_and_vectors)
    assert sig.names == ["u", "x", "y", "k"]
    # d = (1, 1, -1); u . d = 2 > k = -3 -> 2 * -3; abs(-3) u - d
    assert run(fn, (1.0, 2.0, 1.0), (2.0, 3.0, 0.0), (1.0, 2.0, 1.0), -3.0) == [-6.0, 2.0, 5.0, 4.0]


def test_min_and_max_select_the_first_argument_on_ties() -> None:
    fn, _ = parse(minmax_ties)
    a_min, a_max = fn.instrs[fn.outputs[0]], fn.instrs[fn.outputs[1]]
    assert a_min.op == a_max.op == "select"
    assert fn.instrs[a_min.args[1]].attr == (0, 0)  # first argument when the comparison holds
    assert fn.instrs[a_max.args[1]].attr == (0, 0)
    assert run(fn, 2.0, 2.0) == [2.0, 2.0]


@pointwise
def uses_while(x: State) -> State:
    while x > 1.0:
        x = x / 2.0
    return x


@pointwise
def one_sided(x: State) -> State:
    if x > 0.0:
        y = x
    return y


@pointwise
def rel_after_arithmetic(x: Geom.Vec3, y: Geom.Vec3) -> Geom.Vec3:
    return rel(x * 2.0, y)


@pointwise
def calls_python(x: State) -> State:
    return round(x)


@pointwise(compute="accum")
def accum_helper(x: State) -> State:
    return x


@pointwise
def calls_accum_helper(x: State) -> State:
    return accum_helper(x)


@pointwise
def identity_of_a_value(x: State) -> Grad.Mat3:
    return identity(x)


@pointwise
def inverse_of_a_rectangle(m: Grad.Mat53) -> Grad.Mat53:
    return inverse(m)


@pointwise
def guarded_sqrt(x: State) -> State:
    return sqrt(x) if x > 0.0 else x


def test_the_branch_not_taken_does_not_poison_the_adjoint() -> None:
    # d/dx at x = -1 is 1 (the else branch); the sqrt branch evaluates to NaN and must not leak.
    fn, _ = parse(guarded_sqrt)
    adj = adjoint(fn, [0], [[1.0]])
    tan = tangent(fn, [0])
    with np.errstate(invalid="ignore", divide="ignore"):
        assert run(adj, -1.0, 1.0) == [1.0]
        assert run(tan, -1.0, 1.0) == [-1.0, 1.0]
        assert run(adj, 4.0, 1.0) == [0.25]


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (uses_while, "E1001"),
        (one_sided, "E1005"),
        (rel_after_arithmetic, "E1007"),
        (calls_python, "E1001"),
        (inverse_of_a_rectangle, "E1006"),
        (identity_of_a_value, "E1006"),
        (calls_accum_helper, "E1006"),
    ],
)
def test_unsupported_constructs_are_rejected_with_a_code_and_line(
    body: dsl.Body, code: str
) -> None:
    with pytest.raises(CodegenError) as info:
        parse(body)
    assert info.value.code == code
    assert info.value.where is not None
    assert info.value.where.endswith(tuple(f":{n}" for n in range(1, 500)))


q = cells.field("q", State)
r = cells.field("r", Residual)


@pointwise
def flux(ql: State, qr: State) -> Residual:
    return ql - qr


@pointwise
def in_place(x: Residual) -> Residual:
    return 2.0 * x


@pointwise
def block_value(b: dsl.Jac) -> Residual:
    return b


@pointwise(compute="accum")
def scaled_in_accum(x: Geom) -> Residual:
    return 2.0 * x


def test_loop_declarations_are_checked() -> None:
    with pytest.raises(CodegenError) as info:
        build_loop("bad", dsl.face_loop(flux, ql=q.left, out=dsl.Inc(r, left=1, right=-1)))
    assert info.value.code == "E1003"
    with pytest.raises(CodegenError) as info:
        build_loop(
            "bad", dsl.face_loop(flux, ql=q.left, qr=q.right, out=dsl.Inc(q, left=1, right=-1))
        )
    assert info.value.code == "E1008"
    with pytest.raises(CodegenError) as info:
        dsl.face_loop(flux, ql=q.left, qr=q.right, out=dsl.Inc(r, left=1))
    assert info.value.code == "E1004"
    with pytest.raises(CodegenError) as info:
        build_loop("bad", dsl.cell_loop(in_place, x=r, out=dsl.Write(r)))
    assert info.value.code == "E1003"
    # A body computing in the accumulation type reads blocks and parameters only: no
    # derivatives, no geometry rounded to the compute type.
    for x in (q, cells.field("volume", Geom)):
        with pytest.raises(CodegenError) as info:
            build_loop("bad", dsl.cell_loop(scaled_in_accum, x=x, out=dsl.Write(r)))
        assert info.value.code == "E1003"
    # near and far take a half of a (2, ...) face dataset, in incidence reductions only.
    single, pair = dsl.faces.field("single", dsl.Jac), dsl.faces.field("pair", dsl.Jac.shape(2))
    for loop in (
        lambda: dsl.cell_reduce(block_value, b=single.near, acc=dsl.Sum(r)),
        lambda: dsl.face_loop(block_value, b=pair.far, out=dsl.Inc(r, left=1, right=-1)),
    ):
        with pytest.raises(CodegenError) as info:
            build_loop("bad", loop())
        assert info.value.code == "E1003"


@pointwise
def state_gradient_step(
    qc: State.Vec5, qn: State.Vec5, w: Geom.Vec3, r: Geom.Vec3
) -> tuple[Grad.Mat53, State.Vec5]:
    g = outer(qn - qc, w)
    return g, qc + g @ r


def test_matrix_gradients_of_vector_states() -> None:
    # outer(a, b)[i, j] = a_i b_j; (5 x 3) @ (3,) gives row-wise dot products.
    fn, sig = parse(state_gradient_step)
    assert sig.returns is not None
    assert sig.returns[0].shape == (5, 3)
    qc, qn = (1.0, 2.0, 3.0, 4.0, 5.0), (2.0, 4.0, 6.0, 8.0, 10.0)
    w, r = (1.0, 0.5, 0.0), (2.0, 2.0, 1.0)
    out = run(fn, qc, qn, w, r)
    d = np.subtract(qn, qc)
    np.testing.assert_array_equal(out[:15], np.outer(d, w).ravel())
    np.testing.assert_array_equal(out[15:], np.add(qc, np.outer(d, w) @ r))


@pointwise
def outer_of_one(v: State.Vec3) -> Grad.Mat3:
    return outer(v)


def test_intrinsic_arity_is_a_coded_error() -> None:
    with pytest.raises(CodegenError) as info:
        parse(outer_of_one)
    assert info.value.code == "E1006"


q_face = dsl.faces.field("q_face", State)


@pointwise
def copy_state(x: State) -> State:
    return x


def test_boundary_states_from_active_data_must_be_declared_frozen() -> None:
    with pytest.raises(CodegenError) as info:
        build_loop("bad", dsl.boundary_loop(copy_state, x=q.left, out=dsl.Write(q_face)))
    assert info.value.code == "E1012"
    lm = build_loop(
        "ok", dsl.boundary_loop(copy_state, x=q.left, out=dsl.Write(q_face), frozen=True)
    )
    assert lm.kind == "boundary_write"


@pointwise
def two_sided(ql: State.Vec5, qr: State.Vec5, n: Geom.Vec3) -> Residual.Vec5:
    return dot(ql, ql) * qr + n[0] * ql


u5 = cells.field("u5", State.Vec5)
r5 = cells.field("r5", Residual.Vec5)


@pointwise
def apply_block(a: dsl.JacAccum.Mat5, v: State.Vec5) -> Residual.Vec5:
    return a @ v


def test_jacobian_outputs_are_checked_against_the_body() -> None:
    good = dsl.faces.field("good", dsl.Jac.shape(2, 5, 5))
    out = [dsl.Inc(r5, left=1, right=-1), dsl.Jacobian(good, wrt=("ql", "qr"))]
    lm = build_loop(
        "ok", dsl.face_loop(two_sided, ql=u5.left, qr=u5.right, n=dsl.geometry.face_normal, out=out)
    )
    assert len(lm.jac_fns) == 1
    assert len(lm.jac_fns[0][1].outputs) == 50
    for dataset, wrt, code in (
        (dsl.faces.field("flat", dsl.Jac.shape(5, 5)), ("ql", "qr"), "E1004"),
        (dsl.faces.field("state", State.Mat5), ("ql",), "E1004"),
        (dsl.cells.field("on_cells", dsl.Jac.shape(5, 5)), ("ql",), "E1004"),
        (dsl.faces.field("geom", dsl.Jac.shape(5, 3)), ("n",), "E1003"),
    ):
        out = [dsl.Inc(r5, left=1, right=-1), dsl.Jacobian(dataset, wrt=wrt)]
        with pytest.raises(CodegenError) as info:
            build_loop(
                "bad",
                dsl.face_loop(
                    two_sided, ql=u5.left, qr=u5.right, n=dsl.geometry.face_normal, out=out
                ),
            )
        assert info.value.code == code, dataset.name
    # Blocks are passive: no Jacobian with respect to a block dataset.
    blocks = dsl.cells.field("blocks", dsl.JacAccum.Mat5)
    sink = dsl.cells.field("sink", dsl.Jac.shape(5, 25))
    with pytest.raises(CodegenError) as info:
        build_loop(
            "passive",
            dsl.cell_loop(
                apply_block, a=blocks, v=u5, out=[dsl.Write(r5), dsl.Jacobian(sink, wrt="a")]
            ),
        )
    assert info.value.code == "E1003"
    out = [
        dsl.Inc(r5, left=1, right=-1),
        dsl.Jacobian(good, wrt="ql"),
        dsl.Jacobian(good, wrt="qr"),
    ]
    with pytest.raises(CodegenError) as info:
        build_loop(
            "bad",
            dsl.face_loop(two_sided, ql=u5.left, qr=u5.right, n=dsl.geometry.face_normal, out=out),
        )
    assert info.value.code == "E1004"
    with pytest.raises(CodegenError) as info:
        dsl.boundary_loop(
            copy_state, x=q.left, out=[dsl.Write(q_face), dsl.Jacobian(good, wrt="x")], frozen=True
        )
    assert info.value.code == "E1004"


def test_jacobian_columns_are_the_partial_derivatives() -> None:
    # F = (ql . ql) qr + n0 ql: dF/dql = 2 qr ql^T + n0 I, dF/dqr = (ql . ql) I.
    fn, _ = parse(two_sided)
    jfn = jacobian(fn, [0, 1])
    ql, qr, n = (1.0, 2.0, -1.0, 0.5, 3.0), (0.5, -2.0, 1.0, 4.0, 1.0), (2.0, 0.0, 0.0)
    got = np.array(run(jfn, ql, qr, n)).reshape(2, 5, 5)
    a, b = np.array(ql), np.array(qr)
    np.testing.assert_allclose(got[0], 2.0 * np.outer(b, a) + 2.0 * np.eye(5), rtol=1e-15)
    np.testing.assert_allclose(got[1], (a @ a) * np.eye(5), rtol=1e-15)


SHADOWED_ABS = """
from autan_codegen.dsl import JacAccum, inverse, pointwise


@pointwise
def abs(x: JacAccum) -> JacAccum:
    return x


@pointwise
def invert(a: JacAccum.Mat5) -> JacAccum.Mat5:
    return inverse(a)
"""


def test_inverse_pivots_on_magnitudes_whatever_the_module_calls_abs(tmp_path: Path) -> None:
    src = tmp_path / "shadowed.py"
    src.write_text(SHADOWED_ABS)
    spec = importlib.util.spec_from_file_location("shadowed", src)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn, _ = parse(module.invert)
    a = -np.eye(5)
    a[4, 0] = 3.0  # pivoting on column 0 swaps rows 0 and 4
    with np.errstate(invalid="ignore"):  # the NaN constant of the singular branch
        got = run(fn, tuple(a.ravel()))
    np.testing.assert_allclose(got, np.linalg.inv(a).ravel())


ENTRIES_OF_ACTIVE_DATA = """
from autan_codegen.dsl import Jac, Residual, State, Sum, Write, cell_reduce, cells, incidences
from autan_codegen.dsl import pointwise

q = cells.field("q", State)
r = cells.field("r", Residual)
e = incidences.field("e", Jac)


@pointwise
def term(qo: State) -> tuple[Residual, Jac]:
    return qo, 2.0 * qo


bad = cell_reduce(term, qo=q.other(None), acc=Sum(r), entries=Write(e))
"""


def test_per_entry_outputs_go_to_the_entries_and_take_no_active_input(tmp_path: Path) -> None:
    # One value per entry, written once by its cell: a dataset of the iterated set, and no
    # derivatives to transpose (the adjoint of a per-entry write is not generated).
    with pytest.raises(CodegenError) as info:
        dsl.cell_reduce(
            block_value, b=dsl.faces.field("f", dsl.Jac), acc=dsl.Sum(r), entries=dsl.Write(r)
        )
    assert info.value.code == "E1004"
    src = tmp_path / "entries.py"
    src.write_text(ENTRIES_OF_ACTIVE_DATA)
    with pytest.raises(CodegenError) as info:
        generate(src, tmp_path / "gen", ["cpu"])
    assert info.value.code == "E1012"


@pointwise
def pass_on(x: Residual) -> Residual:
    return x


def test_a_chain_takes_two_reductions_and_reads_the_first_from_its_own_cell() -> None:
    # The first reduction's outputs are in registers for the cell only: another cell's value,
    # a dataset it does not write, or a loop that is not a cell reduction is refused.
    total = dsl.cells.field("total", Residual)
    first = dsl.cell_reduce(block_value, b=dsl.faces.field("fb", dsl.Jac), acc=dsl.Sum(total))
    across = dsl.cell_reduce(pass_on, x=total.other(None), acc=dsl.Sum(r))
    own = dsl.cell_reduce(pass_on, x=total.self_, acc=dsl.Sum(r))
    with pytest.raises(CodegenError) as info:
        dsl.chain(first, dsl.cell_loop(copy_state, x=q, out=dsl.Write(r)))
    assert info.value.code == "E1004"
    with pytest.raises(CodegenError) as info:
        dsl.chain(first, own, keep=[r])
    assert info.value.code == "E1004"
    with pytest.raises(CodegenError) as info:
        build_loop("bad", dsl.chain(first, across))
    assert info.value.code == "E1003"
    assert build_loop("ok", dsl.chain(first, own, keep=[total])).kind == "chain"


CHAIN_NAMES = """
from autan_codegen.dsl import Residual, Sum, Max, cell_reduce, cells, chain, pointwise

g = cells.field("g", Residual)
h = cells.field("h", Residual)
x = cells.field("x", Residual)


@pointwise
def chain_g(v: Residual) -> tuple[Residual, Residual]:
    return v, 2.0 * v


@pointwise
def pass_on(v: Residual) -> Residual:
    return v


first = cell_reduce(chain_g, v=x.other(None), acc=[Sum(g), Max(g)])
second = cell_reduce(pass_on, v=g.self_, acc=Sum(h))
fused = chain(first, second, keep=[g])
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_a_chain_compiles_whatever_its_names_and_refuses_writing_what_it_reads(
    tmp_path: Path,
) -> None:
    # A body named as a register would be, two accumulators on one dataset: the chain compiles.
    src = tmp_path / "chain_names.py"
    src.write_text(CHAIN_NAMES)
    generate(src, tmp_path / "gen", ["cpu"])
    unit = tmp_path / "unit.cpp"
    unit.write_text(
        '#include "chain_names_cpu.hpp"\n'
        "template void autan::kernels::chain_names::cpu::fused<autan::kernels::fp32_acc64>("
        "const autan::kernels::chain_names::Data<autan::kernels::fp32_acc64>&);\n"
    )
    subprocess.run(
        ["c++", "-std=c++20", "-fsyntax-only", "-Wall", "-Wextra", "-Werror",
         f"-I{ROOT / 'codegen' / 'include'}", f"-I{tmp_path / 'gen'}", str(unit)],
        check=True,
    )  # fmt: skip
    # The second writing what the first reads across cells: refused.
    q = dsl.cells.field("qq", Residual)
    g = dsl.cells.field("gg", Residual)
    first = dsl.cell_reduce(pass_on, x=q.other(None), acc=dsl.Sum(g))
    second = dsl.cell_reduce(pass_on, x=g.self_, acc=dsl.Sum(q))
    with pytest.raises(CodegenError) as info:
        build_loop("bad", dsl.chain(first, second, keep=[g]))
    assert info.value.code == "E1003"


CHAIN_STALE = """
from autan_codegen.dsl import Residual, Sum, cell_reduce, cell_loop, cells, chain, pointwise, Write

g = cells.field("g", Residual)
h = cells.field("h", Residual)
x = cells.field("x", Residual)
y = cells.field("y", Residual)


@pointwise
def pass_on(v: Residual) -> Residual:
    return v


first = cell_reduce(pass_on, v=x.other(None), acc=Sum(g))
second = cell_reduce(pass_on, v=g.self_, acc=Sum(h))
fused = chain(first, second)
REWRITE
reader = cell_loop(pass_on, v=g, out=Write(y))
"""


def test_a_loop_after_a_chain_does_not_read_what_the_chain_left_unstored(tmp_path: Path) -> None:
    # g is computed in registers only; a loop declared after the chain reading g, with nothing
    # writing g in between, would read a stale g. Writing g in between, or keeping it, is fine;
    # Euler's module, whose unfused limiter is declared before its chain, generates.
    for rewrite, code in (("", "E1003"), ("again = cell_loop(pass_on, v=x, out=Write(g))", None)):
        src = tmp_path / f"stale_{code}.py"
        src.write_text(CHAIN_STALE.replace("REWRITE", rewrite))
        if code is None:
            generate(src, tmp_path / "ok", ["cpu"])
        else:
            with pytest.raises(CodegenError) as info:
                generate(src, tmp_path / "bad", ["cpu"])
            assert info.value.code == code
    generate(ROOT / "kernels" / "euler.py", tmp_path / "euler", ["cpu"])
