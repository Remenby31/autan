"""DSL parsing: SSA semantics of names, branches and unrolled loops, and coded rejections."""

from __future__ import annotations

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
    outer,
    pointwise,
    rel,
    sqrt,
)
from autan_codegen.errors import CodegenError
from autan_codegen.frontend import parse

if TYPE_CHECKING:
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
