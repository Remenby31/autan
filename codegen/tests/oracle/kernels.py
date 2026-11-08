"""Kernel replay for D1 (ADR 0003, check 3): the gradient loops of `kernels/gradients.py`,
driven by the generator's own IR instruction list for each body.

Every value is a `Pair`: the value the build executes (exact emulation of IEEE 754
round-to-nearest-even in the format the emitted C++ uses) and its certificate. The formats follow
the precision policies of `runtime.hpp`; the conversions follow `emit_cpp.py`:
- a dataset is loaded in its storage type (DECLARED_STORAGE below, written from the spec and
  checked against `emit_cpp.storage_type`: built-in geometry no body
  reads through `rel` is stored in `geom_value`), converted to the parameter's role type, then to
  the body's compute type `C` on first use;
- `rel(x, y)` subtracts the two raw Geom parameters in their role type, then converts to `C`;
- every body operation is done in `C`;
and the accumulation order follows `emit_cpu.py`:
- `cell_reduce`: incidences in `cell_faces` order; Sums in `P::accum` (`acc += A(out)`), Min/Max
  in `C` (strict comparisons, the first incidence wins a tie); the finalise body converts its
  arguments to its role types, then to `C`; outputs stored with `static_cast` to the role type;
- face and boundary loops: faces in increasing order, `target += A(out)` (left) and
  `target -= A(out)` (right), in the target's storage type;
- cell loops: `static_cast` of the body's output to the written dataset's type.
The topology must have no reversed face and no `incidence_cell` (a serial mesh); else invalid.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from autan_codegen import dsl, emit_cpp
from autan_codegen.model import ROLE_TYPE, LoopModel, Module, load

from oracle import certified as c
from oracle.sequences import Pair

if TYPE_CHECKING:
    from autan_codegen.ir import Function

    from oracle.topology import ReadMesh

F = Fraction
B64, B32 = c.BINARY64, c.BINARY32

# runtime.hpp, the three policies of the gate: type name -> format.
POLICIES: dict[str, dict[str, c.Format]] = {
    "fp64": dict.fromkeys(
        ("state", "grad", "residual", "geom", "geom_value", "param", "compute", "accum"), B64
    ),
    "fp32+acc64": {
        "state": B32,
        "grad": B32,
        "residual": B64,
        "geom": B64,
        "geom_value": B32,
        "param": B64,
        "compute": B32,
        "accum": B64,
    },
    "fp32": dict.fromkeys(
        ("state", "grad", "residual", "geom", "geom_value", "param", "compute", "accum"), B32
    ),
}

# TEST ONLY (control 11): spike/spike.hpp's Fp32Acc32TestOnly, fp32+acc64 with its accumulation
# narrowed to binary32. Never a gate policy.
POLICIES["fp32+acc32-test-only"] = {**POLICIES["fp32+acc64"], "residual": B32, "accum": B32}

# The declared precision contract of the gradient loops, written BY HAND from the specification:
# runtime.hpp's policies, ADR 0003 ("Roles and casts per policy": geometry read through rel in
# P::geom, other geometry in P::geom_value), and the roles declared in kernels/gradients.py's
# signatures. It is never derived from the generator: the replay uses only this table, and
# `check_contract` makes the instance invalid where the generator disagrees with it (ADR 0003,
# amendment 2026-11-08; the common mode found in review).
DECLARED_STORAGE = {  # dataset -> policy type it is stored in
    "q": "state", "q_b": "state", "grad": "grad", "qmin": "state", "qmax": "state",
    "gg_sum": "residual",
    "lsq_weight": "geom_value", "face_area": "geom_value", "face_moment": "geom_value",
    "cell_volume": "geom_value",
    "cell_centroid": "geom", "face_centroid": "geom",  # read through rel()
}  # fmt: skip
DECLARED_COMPUTE = {  # body -> the policy type it computes in
    "lsq_bounds_term": "compute", "include_own_value": "compute", "gg_interior": "compute",
    "gg_boundary": "compute", "gg_divide": "compute",
}  # fmt: skip
DECLARED_ROLES = {  # body -> argument -> role type
    "lsq_bounds_term": {"qc": "state", "qn": "state", "w": "geom"},
    "include_own_value": {"g": "residual", "lo": "residual", "hi": "residual", "qc": "state"},
    "gg_interior": {
        "ql": "state", "gl": "grad", "xl": "geom", "qr": "state", "gr": "grad", "xr": "geom",
        "xf": "geom", "s": "geom", "m": "geom",
    },
    "gg_boundary": {"qb": "state", "gl": "grad", "s": "geom", "m": "geom"},
    "gg_divide": {"total": "residual", "vol": "geom"},
}  # fmt: skip
GRADIENT_LOOPS = (
    "lsq_gradient_bounds", "green_gauss_interior", "green_gauss_boundary", "green_gauss_finalise",
)  # fmt: skip


DECLARED_LOOPS = {  # loop -> (body, finalise body or None)
    "lsq_gradient_bounds": ("lsq_bounds_term", "include_own_value"),
    "green_gauss_interior": ("gg_interior", None),
    "green_gauss_boundary": ("gg_boundary", None),
    "green_gauss_finalise": ("gg_divide", None),
}


def check_contract(m: Module) -> list[str]:
    """Where the generator departs from the declared precision contract (empty if it agrees). The
    inventory is exhaustive: every declared loop, body, finalise and argument must exist with its
    declared role and compute type, and every dataset the loops read, write or reduce into
    (outputs included, e.g. qmin and qmax) must be stored as declared."""
    issues = []
    rel_read = emit_cpp.rel_datasets(m)
    loops = {lm.name: lm for lm in m.loops}
    seen: set[str] = set()
    for name, (body, fin) in DECLARED_LOOPS.items():
        lm = loops.get(name)
        if lm is None:
            issues.append(f"loop {name} is missing")
            continue
        declared = [(body, lm.fn, lm.sig)]
        if fin is not None or lm.fin_fn is not None:
            declared.append((fin, lm.fin_fn, lm.fin_sig))
        for want, fn, sig in declared:
            if fn is None or sig is None or fn.name != want:
                issues.append(f"{name}: body {getattr(fn, 'name', None)}, declared {want}")
                continue
            if fn.compute != DECLARED_COMPUTE.get(fn.name):
                issues.append(
                    f"{fn.name}: computes in {fn.compute}, declared {DECLARED_COMPUTE.get(fn.name)}"
                )
            roles = DECLARED_ROLES[fn.name]
            if list(sig.names) != list(roles):
                issues.append(f"{fn.name}: arguments {list(sig.names)}, declared {list(roles)}")
            for arg, t in zip(sig.names, sig.types, strict=True):
                if ROLE_TYPE[t.role] != roles.get(arg):
                    issues.append(
                        f"{fn.name}({arg}): role {ROLE_TYPE[t.role]}, declared {roles.get(arg)}"
                    )
        datasets = [b.dataset for b in lm.loop.bindings.values()]
        datasets += [
            b.boundary
            for b in lm.loop.bindings.values()
            if getattr(b, "boundary", None) is not None
        ]
        datasets += [b.dataset for b in lm.loop.finalise_bindings.values()]
        datasets += [o.dataset for o in [*lm.loop.outputs, *lm.loop.entries]]
        for d in datasets:
            if not isinstance(d, dsl.Dataset):
                continue
            seen.add(d.name)
            got = emit_cpp.storage_type(d, rel_read).split("::")[-1]
            if got != DECLARED_STORAGE.get(d.name):
                issues.append(f"{d.name}: stored in {got}, declared {DECLARED_STORAGE.get(d.name)}")
    issues.extend(
        f"dataset {name} is declared but no gradient loop uses it"
        for name in sorted(set(DECLARED_STORAGE) - seen)
    )
    return sorted(set(issues))


# The generated gradient code and runtime.hpp's policies, frozen as reviewed (ADR 0003, amendment
# 2026-11-08): the structural check above compares the generator's decisions with the declared
# table; these digests pin what the build compiles (accumulator declarations, conversions and their
# placement, loop order, policy aliases). A departure makes every instance invalid. Digests of the
# text without its #line directives, which carry absolute paths.
FROZEN_SHA256 = {
    "codegen/generated/gradients.hpp": "54252a30ead82da4399226cbc725d17570421c19b53cd649119b20f8b285efbb",
    "codegen/generated/gradients_cpu.hpp": "e17ff4c58b5b9babd9f98522f4b8fbec7067081c2836bd41253113345179a03b",
    "codegen/include/autan/kernels/runtime.hpp": "06cc008f37b0c6fa6cf179c21fd43e2e39000868ec4946515628eb33097cdb3f",
}


def _digest(path: Path) -> str:
    lines = [ln for ln in path.read_text().splitlines(keepends=True) if not ln.startswith("#line")]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def library_sha256() -> str | None:
    """SHA-256 of the spike library the oracle loads (AUTAN_SPIKE_LIB), or None if unset."""
    lib = os.environ.get("AUTAN_SPIKE_LIB")
    return (
        hashlib.sha256(Path(lib).read_bytes()).hexdigest() if lib and Path(lib).is_file() else None
    )


def check_library(expected: str) -> list[str]:
    """Where the loaded library is not the frozen one. The header digests pin reviewed source; this
    pins the executable: the library built from the freeze commit, whose SHA-256 the freeze manifest
    records (ADR 0003, amendment 2026-11-08). It covers what the header digests cannot: a library
    built from other source with the headers restored afterwards, and the spike's policy dispatch."""
    got = library_sha256()
    if got is None:
        return ["the spike library is not found (AUTAN_SPIKE_LIB)"]
    return (
        []
        if got == expected
        else [f"the loaded library {got[:12]} is not the frozen one {expected[:12]}"]
    )


def check_frozen() -> list[str]:
    """Where the compiled generated headers or runtime.hpp differ from their frozen digests."""
    lib = os.environ.get("AUTAN_SPIKE_LIB")
    generated = Path(lib).resolve().parent / "generated" if lib else None
    source = Path(__file__).resolve().parents[2]  # codegen/
    issues = []
    for key, want in FROZEN_SHA256.items():
        if key.startswith("codegen/generated/"):
            if generated is None:
                issues.append(f"{key}: no build to read (AUTAN_SPIKE_LIB unset)")
                continue
            path = generated / Path(key).name
        else:
            path = source / Path(key).relative_to("codegen")
        if not path.is_file():
            issues.append(f"{key}: missing")
        elif _digest(path) != want:
            issues.append(f"{key}: differs from the frozen, reviewed text")
    return issues


KERNEL_SOURCE = Path(__file__).resolve().parents[3] / "kernels" / "gradients.py"
SOURCES = (
    (
        "kernels/gradients.py: lsq_gradient_bounds, green_gauss_interior, green_gauss_boundary, "
        "green_gauss_finalise"
    ),
    "codegen/src/autan_codegen/emit_cpp.py: body_lines, call_args, storage_type",
    "codegen/src/autan_codegen/emit_cpu.py: reduce_body, inc_lines, cell_stores, emit_loop",
    "codegen/include/autan/kernels/runtime.hpp: fp64, fp32_acc64, fp32",
)


def module() -> Module:
    return load(KERNEL_SOURCE)


def convert(x: Pair, to: c.Format) -> Pair:
    """A C++ conversion of a value of format x.c.fmt to `to`: exact when it widens."""
    src = x.c.fmt
    if src is not None and src == to:
        return x
    if src is not None and src.p <= to.p:
        return Pair(x.e, c.cast(x.c, to, narrowing=False))
    return Pair(c.round_to(x.e, to), c.cast(x.c, to))


def stored(v: Pair, fmt: c.Format) -> Pair:
    """A binary64 construction (or an exact input) stored in `fmt`."""
    return convert(v, fmt)


def given(x: float, fmt: c.Format) -> Pair:
    """A value the kernel reads, given exactly in its format (radius 0)."""
    v = F(x)
    if not c.representable(v, fmt):
        msg = f"{x!r} is not a {fmt.name} value"
        raise c.Invalid(msg)
    return Pair(v, c.Cert(v, F(0), fmt))


def _op(op: str, x: Pair, y: Pair, t: c.Format) -> Pair:
    if op == "add":
        return Pair(c.fl_add(x.e, y.e, t), c.add(x.c, y.c, t))
    if op == "sub":
        return Pair(c.fl_add(x.e, -y.e, t), c.sub(x.c, y.c, t))
    if op == "mul":
        return Pair(c.fl_mul(x.e, y.e, t), c.mul(x.c, y.c, t))
    if op == "div":
        return Pair(c.fl_div(x.e, y.e, t), c.div(x.c, y.c, t))
    raise AssertionError(op)


def _less(x: Pair, y: Pair, strict: bool) -> bool:
    """A comparison the executed path and every value of the certificates decide alike."""
    lo_x, hi_x = x.c.value - x.c.radius, x.c.value + x.c.radius
    lo_y, hi_y = y.c.value - y.c.radius, y.c.value + y.c.radius
    exec_ = x.e < y.e if strict else x.e <= y.e
    certain = (hi_x < lo_y) if exec_ else (lo_x >= hi_y if strict else lo_x > hi_y)
    if x.c.radius == 0 and y.c.radius == 0:
        certain = True
    if not certain:
        msg = "a comparison is not decided by the certificates"
        raise c.Invalid(msg)
    return exec_


def _cone(fn: Function, outputs: list[int]) -> set[int]:
    """The instructions the given outputs depend on."""
    need = {fn.outputs[o] for o in outputs}
    for v in range(len(fn.instrs) - 1, -1, -1):
        if v in need:
            need.update(fn.instrs[v].args)
    return need


def run_body(
    fn: Function,
    args: list[list[Pair]],
    role_fmt: list[c.Format],
    pol: str,
    outputs: list[int] | None = None,
) -> list[Pair]:
    """Evaluate `fn`'s IR on arguments already converted to their role types; only the
    instructions the requested outputs (default: all) depend on are evaluated."""
    C = POLICIES[pol][DECLARED_COMPUTE[fn.name]]
    wanted = list(range(len(fn.outputs))) if outputs is None else outputs
    cone = _cone(fn, wanted)
    # emit_cpp.body_lines converts an input to C only if some non-rel instruction or an output uses it.
    used = {fn.outputs[o] for o in wanted}
    for v in cone:
        if fn.instrs[v].op != "rel":
            used.update(fn.instrs[v].args)
    vals: list[Pair | bool | None] = []
    for v, ins in enumerate(fn.instrs):
        op = ins.op
        if v not in cone:
            vals.append(None)
            continue
        if op == "input":
            k, comp = ins.attr  # type: ignore[misc]
            vals.append(convert(args[k][comp], C) if v in used else None)
        elif op == "const":
            vals.append(convert(given(float(ins.attr), B64), C))  # type: ignore[arg-type]
        elif op == "rel":
            (kx, cx), (ky, cy) = (fn.instrs[i].attr for i in ins.args)  # type: ignore[misc]
            t = role_fmt[kx]
            assert role_fmt[ky] == t
            vals.append(convert(_op("sub", args[kx][cx], args[ky][cy], t), C))
        elif op in ("add", "sub", "mul", "div"):
            x, y = (vals[i] for i in ins.args)
            assert isinstance(x, Pair)
            assert isinstance(y, Pair)
            vals.append(_op(op, x, y, C))
        elif op == "neg":
            x = vals[ins.args[0]]
            assert isinstance(x, Pair)
            vals.append(Pair(-x.e, c.Cert(-x.c.value, x.c.radius, x.c.fmt)))
        elif op in ("lt", "le", "gt", "ge"):
            x, y = (vals[i] for i in ins.args)
            assert isinstance(x, Pair)
            assert isinstance(y, Pair)
            if op in ("gt", "ge"):
                x, y = y, x
            vals.append(_less(x, y, strict=op in ("lt", "gt")))
        elif op == "select":
            cond, yes, no = (vals[i] for i in ins.args)
            vals.append(yes if cond else no)
        else:
            msg = f"{fn.name}: operation {op} is outside the declared gradient bodies"
            raise c.Invalid(msg)
    out = [vals[fn.outputs[o]] for o in wanted]
    assert all(isinstance(v, Pair) for v in out)
    return out  # type: ignore[return-value]


def _roles(fn: Function, names: list[str], pol: str) -> list[c.Format]:
    """The declared role formats of a body's arguments (DECLARED_ROLES, not the generator's)."""
    return [POLICIES[pol][DECLARED_ROLES[fn.name][n]] for n in names]


@dataclass
class Inputs:
    """What the gradient kernels read, in binary64 before storage: geometry as constructed (Pairs
    of `sequences.construct`), the field, its boundary values and the previous gradient."""

    geometry: dict[str, list[list[Pair]]]
    q: list[Pair]
    q_b: list[Pair]
    grad: list[list[Pair]] | None = None
    # D1 reads the kernel's stored field values as given exactly (radius 0): the field's rounding
    # is not propagated. Check 4 propagates it from the binary64 values (field_exact=False).
    field_exact: bool = True


def field_value(v: Pair, fmt: c.Format, exact: bool) -> Pair:
    """A field value stored in `fmt`: as given exactly (D1) or with its rounding (check 4)."""
    if exact:
        return given(float(c.round_to(v.e, fmt)), fmt)
    return stored(v, fmt)


class Rounded:
    """The executed arithmetic: Pairs, rounded in the policy's formats."""

    def conv(self, v: Pair, fmt: c.Format) -> Pair:
        return convert(v, fmt)

    def op(self, op: str, x: Pair, y: Pair, fmt: c.Format) -> Pair:
        return _op(op, x, y, fmt)

    def zero(self, fmt: c.Format) -> Pair:
        return given(0.0, fmt)

    def body(
        self,
        fn: Function,
        args: list[list[Pair]],
        roles: list[c.Format],
        pol: str,
        outputs: list[int] | None = None,
    ) -> list[Pair]:
        return run_body(fn, args, roles, pol, outputs)


class Exact:
    """The declared sequence with every rounding removed: exact rationals (correspondence)."""

    def conv(self, v: Fraction, fmt: c.Format) -> Fraction:  # noqa: ARG002
        return v

    def op(self, op: str, x: Fraction, y: Fraction, fmt: c.Format) -> Fraction:  # noqa: ARG002
        return {"add": x + y, "sub": x - y, "mul": x * y}[op] if op != "div" else x / y

    def zero(self, fmt: c.Format) -> Fraction:  # noqa: ARG002
        return F(0)

    def body(
        self,
        fn: Function,
        args: list[list[Fraction]],
        roles: list[c.Format],  # noqa: ARG002
        pol: str,  # noqa: ARG002
        outputs: list[int] | None = None,
    ) -> list[Fraction]:
        wanted = list(range(len(fn.outputs))) if outputs is None else outputs
        cone = _cone(fn, wanted)
        vals: list[object] = []
        for v, ins in enumerate(fn.instrs):
            if v not in cone:
                vals.append(None)
                continue
            a = [vals[i] for i in ins.args]
            if ins.op == "input":
                k, comp = ins.attr  # type: ignore[misc]
                vals.append(args[k][comp])
            elif ins.op == "const":
                vals.append(F(float(ins.attr)))  # type: ignore[arg-type]
            elif ins.op == "rel":
                (kx, cx), (ky, cy) = (fn.instrs[i].attr for i in ins.args)  # type: ignore[misc]
                vals.append(args[kx][cx] - args[ky][cy])
            elif ins.op in ("add", "sub", "mul", "div"):
                vals.append(self.op(ins.op, a[0], a[1], B64))  # type: ignore[arg-type]
            elif ins.op == "neg":
                vals.append(-a[0])  # type: ignore[operator]
            elif ins.op in ("lt", "le", "gt", "ge"):
                x, y = a
                vals.append({"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y}[ins.op])  # type: ignore[operator]
            elif ins.op == "select":
                vals.append(a[1] if a[0] else a[2])
            else:
                msg = f"{fn.name}: operation {ins.op} is outside the declared gradient bodies"
                raise c.Invalid(msg)
        return [vals[fn.outputs[o]] for o in wanted]  # type: ignore[misc]


def _storage(name: str, pol: str) -> c.Format:
    """The declared storage format of a dataset (DECLARED_STORAGE, not the generator's)."""
    return POLICIES[pol][DECLARED_STORAGE[name]]


def _check_serial(read: ReadMesh) -> None:
    if getattr(read, "face_reversed", None) is not None:
        msg = "a reversed face: the replay covers serial meshes only"
        raise c.Invalid(msg)


def _loop(m: Module, name: str) -> LoopModel:
    return next(lm for lm in m.loops if lm.name == name)


@dataclass
class Read:
    """The kernels' inputs as loaded, in their storage types: per policy (Rounded) or exact."""

    geo: dict[str, list[list[object]]]
    q: list[object]
    q_b: list[object]
    grad: list[list[object]] | None


def load_inputs(inp: Inputs, pol: str) -> Read:
    """Every dataset the gradient loops read, stored as the build stores it (Rounded)."""
    geo = {
        name: [[stored(v, _storage(name, pol)) for v in row] for row in inp.geometry[name]]
        for name in (
            "lsq_weight",
            "cell_centroid",
            "face_centroid",
            "face_area",
            "face_moment",
            "cell_volume",
        )
    }
    q = [field_value(v, _storage("q", pol), inp.field_exact) for v in inp.q]
    q_b = [field_value(v, _storage("q_b", pol), inp.field_exact) for v in inp.q_b]
    grad = (
        None
        if inp.grad is None
        else [
            [field_value(v, _storage("grad", pol), inp.field_exact) for v in row]
            for row in inp.grad
        ]
    )
    return Read(geo, q, q_b, grad)  # type: ignore[arg-type]


def lsq(read: ReadMesh, data: Read, pol: str, ar: Rounded | Exact) -> list[list[object]]:
    """`lsq_gradient_bounds`: the gradient of each cell (the bounds are selections, not replayed)."""
    _check_serial(read)
    m = module()
    lm = _loop(m, "lsq_gradient_bounds")
    assert lm.fin_fn is not None
    assert lm.fin_sig is not None
    P = POLICIES[pol]
    roles = _roles(lm.fn, lm.sig.names, pol)
    fin_roles = _roles(lm.fin_fn, lm.fin_sig.names, pol)
    n_int = len(read.neighbour)
    out: list[list[object]] = []
    for cell in range(read.n_cells):
        acc = [ar.zero(P["accum"])] * 3
        for k in range(int(read.cell_face_offsets[cell]), int(read.cell_face_offsets[cell + 1])):
            f = int(read.cell_faces[k])
            nb = (
                (int(read.neighbour[f]) if int(read.owner[f]) == cell else int(read.owner[f]))
                if f < n_int
                else -1
            )
            by_name = {
                "qc": [data.q[cell]],
                "qn": [data.q[nb] if nb >= 0 else data.q_b[f]],
                "w": data.geo["lsq_weight"][k],
            }
            args = [
                [ar.conv(v, roles[i]) for v in by_name[nm]] for i, nm in enumerate(lm.sig.names)
            ]
            res = ar.body(lm.fn, args, roles, pol, outputs=[0, 1, 2])
            acc = [ar.op("add", acc[u], ar.conv(res[u], P["accum"]), P["accum"]) for u in range(3)]
        # include_own_value(g, lo, hi, qc): lo and hi do not reach the gradient; given as qc.
        qc = data.q[cell]
        fin_args = [[ar.conv(a, fin_roles[0]) for a in acc]] + [
            [ar.conv(qc, fin_roles[i])] for i in (1, 2, 3)
        ]
        fin = ar.body(lm.fin_fn, fin_args, fin_roles, pol, outputs=[0, 1, 2])
        out.append([ar.conv(fin[u], _storage("grad", pol)) for u in range(3)])
    return out


GEOMETRY_ARGS = {  # body argument -> (dataset, which cell or the face)
    "xl": ("cell_centroid", "l"),
    "xr": ("cell_centroid", "r"),
    "xf": ("face_centroid", "f"),
    "s": ("face_area", "f"),
    "m": ("face_moment", "f"),
}


def green_gauss(
    read: ReadMesh, data: Read, pol: str, ar: Rounded | Exact, iterations: int = 1
) -> list[list[object]]:
    """`green_gauss_interior`, `green_gauss_boundary` and `green_gauss_finalise`, `iterations`
    times from the previous gradient `data.grad` (gg_sum zeroed before each iteration)."""
    _check_serial(read)
    assert data.grad is not None
    m = module()
    lms = {
        n: _loop(m, n)
        for n in ("green_gauss_interior", "green_gauss_boundary", "green_gauss_finalise")
    }
    P = POLICIES[pol]
    if P["residual"] != P["accum"]:
        msg = "gg_sum += A(out) assumes P::residual == P::accum"
        raise c.Invalid(msg)
    S = _storage("gg_sum", pol)
    grad = data.grad
    n_int, n_faces = len(read.neighbour), len(read.owner)

    def args_of(lm: LoopModel, f: int, l: int, r: int) -> tuple[list[list[object]], list[c.Format]]:
        roles = _roles(lm.fn, lm.sig.names, pol)
        out = []
        for i, nm in enumerate(lm.sig.names):
            if nm in GEOMETRY_ARGS:
                ds, at_ = GEOMETRY_ARGS[nm]
                raw = data.geo[ds][{"l": l, "r": r, "f": f}[at_]]
            elif nm in ("ql", "qr"):
                raw = [data.q[l if nm == "ql" else r]]
            elif nm in ("gl", "gr"):
                raw = grad[l if nm == "gl" else r]
            elif nm == "qb":
                raw = [data.q_b[f]]
            else:
                raise AssertionError(nm)
            out.append([ar.conv(v, roles[i]) for v in raw])
        return out, roles

    for _ in range(iterations):
        total = [[ar.zero(S)] * 3 for _ in range(read.n_cells)]
        lm = lms["green_gauss_interior"]
        for f in range(n_int):
            l, r = int(read.owner[f]), int(read.neighbour[f])
            a, roles = args_of(lm, f, l, r)
            res = ar.body(lm.fn, a, roles, pol)
            for u in range(3):
                v = ar.conv(res[u], P["accum"])
                total[l][u] = ar.op("add", total[l][u], v, S)
                total[r][u] = ar.op("sub", total[r][u], v, S)
        lm = lms["green_gauss_boundary"]
        for f in range(n_int, n_faces):
            l = int(read.owner[f])
            a, roles = args_of(lm, f, l, -1)
            res = ar.body(lm.fn, a, roles, pol)
            for u in range(3):
                total[l][u] = ar.op("add", total[l][u], ar.conv(res[u], P["accum"]), S)
        lm = lms["green_gauss_finalise"]
        roles = _roles(lm.fn, lm.sig.names, pol)
        new = []
        for cell in range(read.n_cells):
            a = [
                [ar.conv(v, roles[0]) for v in total[cell]],
                [ar.conv(data.geo["cell_volume"][cell][0], roles[1])],
            ]
            res = ar.body(lm.fn, a, roles, pol)
            new.append([ar.conv(res[u], _storage("grad", pol)) for u in range(3)])
        grad = new
    return grad


# ---------------------------------------------------------------- check 3 (D1)
# The intended operators, written from their definitions (ADR 0003, decision 3), on the exact
# node-defined geometry (`sequences.geometry_exact`) and the kernel's stored inputs given exactly.
# They never use the code under test.


def _moment_t(m9: list[Fraction], v: list[Fraction]) -> list[Fraction]:
    """M^T v, the face moment M stored row by row: M[j][u] is component 3 j + u."""
    return [sum((m9[3 * j + u] * v[j] for j in range(3)), F(0)) for u in range(3)]


def lsq_intended(
    read: ReadMesh, geo: dict[str, list[list[Fraction]]], q: list[Fraction], q_b: list[Fraction]
) -> list[list[Fraction]]:
    n_int = len(read.neighbour)
    out = []
    for cell in range(read.n_cells):
        g = [F(0)] * 3
        for k in range(int(read.cell_face_offsets[cell]), int(read.cell_face_offsets[cell + 1])):
            f = int(read.cell_faces[k])
            if f < n_int:
                nb = int(read.neighbour[f]) if int(read.owner[f]) == cell else int(read.owner[f])
                dq = q[nb] - q[cell]
            else:
                dq = q_b[f] - q[cell]
            g = [g[u] + geo["lsq_weight"][k][u] * dq for u in range(3)]
        out.append(g)
    return out


def gg_intended(
    read: ReadMesh,
    geo: dict[str, list[list[Fraction]]],
    q: list[Fraction],
    q_b: list[Fraction],
    grad: list[list[Fraction]],
) -> list[list[Fraction]]:
    n_int, n_faces = len(read.neighbour), len(read.owner)
    xc, xf, s, m = geo["cell_centroid"], geo["face_centroid"], geo["face_area"], geo["face_moment"]
    total = [[F(0)] * 3 for _ in range(read.n_cells)]

    def dot(a: list[Fraction], b: list[Fraction]) -> Fraction:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    for f in range(n_int):
        l, r = int(read.owner[f]), int(read.neighbour[f])
        rl = [xf[f][u] - xc[l][u] for u in range(3)]
        rr = [xf[f][u] - xc[r][u] for u in range(3)]
        qf = (q[l] + dot(grad[l], rl) + q[r] + dot(grad[r], rr)) / 2
        mt = _moment_t(m[f], [(grad[l][u] + grad[r][u]) / 2 for u in range(3)])
        for u in range(3):
            v = qf * s[f][u] + mt[u]
            total[l][u] += v
            total[r][u] -= v
    for f in range(n_int, n_faces):
        l = int(read.owner[f])
        mt = _moment_t(m[f], grad[l])
        for u in range(3):
            total[l][u] += q_b[f] * s[f][u] + mt[u]
    return [
        [total[cell][u] / geo["cell_volume"][cell][0] for u in range(3)]
        for cell in range(read.n_cells)
    ]


def _structural(m: Module, library: str | None) -> list[str]:
    """The structural validity of an instance, checked before any replay: the declared precision
    contract, the frozen generated code and runtime.hpp, and (held-out runs, given the frozen
    library digest) the loaded library. Any issue makes the instance invalid at once."""
    out = [f"invalid: precision contract: {x}" for x in check_contract(m)]
    out += [f"invalid: frozen code: {x}" for x in check_frozen()]
    if library is not None:
        out += [f"invalid: frozen library: {x}" for x in check_library(library)]
    return out


@dataclass
class D1:
    """Check 3 on one instance: issues ("invalid: ..." blocks PASS like a failure, "fail: ...")
    and the largest |v - g_int| / R, the share of R_D1 = R + |v - g_int| due to the centre."""

    issues: list[str]
    max_centre_ratio: Fraction
    max_use: Fraction  # largest |g_hat - g_int| / R_D1


def check_d1(
    read: ReadMesh,
    inp: Inputs,
    exact_geometry: dict[str, list[list[Fraction]]],
    kernel: list[list[float]],
    pol: str,
    method: str,
    iterations: int = 1,
    library: str | None = None,
) -> D1:
    """|g_hat - g_int| <= R_D1 = R + |v - g_int|, centred on the intended g_int; invalid if
    g_int lies outside [v - R, v + R] (interval consistency), if the declared sequence with every rounding
    removed is not the intended operator, or if the build does not execute the declared sequence
    (the replay differs from the kernel's output)."""
    m = module()
    structural = _structural(m, library)
    if structural:
        return D1(structural, F(0), F(0))
    data = load_inputs(inp, pol)
    q = [v.e for v in data.q]  # type: ignore[attr-defined]
    q_b = [v.e for v in data.q_b]  # type: ignore[attr-defined]
    grad = None if data.grad is None else [[v.e for v in row] for row in data.grad]  # type: ignore[attr-defined]
    exact_data = Read(exact_geometry, q, q_b, grad)  # type: ignore[arg-type]
    if method == "lsq":
        replay = lsq(read, data, pol, Rounded())
        removed = lsq(read, exact_data, pol, Exact())
        intended = lsq_intended(read, exact_geometry, q, q_b)
    else:
        assert grad is not None
        assert iterations == 1, "the intended GG is one iteration from grad"
        replay = green_gauss(read, data, pol, Rounded(), iterations)
        removed = green_gauss(read, exact_data, pol, Exact(), iterations)
        intended = gg_intended(read, exact_geometry, q, q_b, grad)
    issues: list[str] = []
    ratio, use = F(0), F(0)
    for cell in range(read.n_cells):
        for u in range(3):
            g_hat = F(float(kernel[cell][u]))
            g_int = intended[cell][u]
            pair = replay[cell][u]
            assert isinstance(pair, Pair)
            v, r = pair.c.value, pair.c.radius
            where = f"{method} cell {cell} component {u}"
            if removed[cell][u] != g_int:
                issues.append(
                    f"invalid: {where}: the declared sequence without rounding is not the intended operator"
                )
            if pair.e != g_hat:
                issues.append(f"invalid: {where}: the build does not execute the declared sequence")
            d = abs(v - g_int)
            if d > r:
                issues.append(
                    f"invalid: {where}: the intended value lies outside the certified interval"
                )
                continue
            if r > 0:
                ratio = max(ratio, d / r)
            bound = r + d
            err = abs(g_hat - g_int)
            if err > bound:
                issues.append(f"fail: {where}: |g_hat - g_int| exceeds R_D1")
            elif bound > 0:
                use = max(use, err / bound)
    return D1(issues, ratio, use)


# ---------------------------------------------------------------- check 4 (the bound)


def linear_field(points: list[list[Pair]], a: float, g: tuple[float, float, float]) -> list[Pair]:
    """The fixtures' declared field sequence: a + ((x0 G0 + x1 G1) + x2 G2) in binary64, at points
    given as Pairs (the constructed centroids, executed and certified)."""
    A = given(a, B64)
    G = [given(v, B64) for v in g]
    out = []
    for x in points:
        s01 = _op("add", _op("mul", x[0], G[0], B64), _op("mul", x[1], G[1], B64), B64)
        out.append(_op("add", A, _op("add", s01, _op("mul", x[2], G[2], B64), B64), B64))
    return out


@dataclass
class Bound:
    """Check 4 on one instance: issues as for D1, and the largest |v - G| / R."""

    issues: list[str]
    max_centre_ratio: Fraction
    max_use: Fraction  # largest |g_hat - G| / R4
    bounds: dict[tuple[int, int], Fraction]  # (cell, component) -> R4


def check_bound(
    read: ReadMesh,
    geometry: dict[str, list[list[Pair]]],
    exact_geometry: dict[str, list[list[Fraction]]],
    a: float,
    g: tuple[float, float, float],
    kernel: list[list[float]],
    pol: str,
    method: str,
    library: str | None = None,
) -> Bound:
    """|g_hat - G| <= R4 = R + |v - G|, R propagated from the exact data (nodes, a, G): the field's
    evaluation and storage, the geometry, the weights, the injected gradient (GG), the casts and the
    kernel arithmetic; invalid if G lies outside [v - R, v + R] or if the build does not execute the
    declared sequence. First, the intended operator must reproduce G exactly on the exact field with
    the exact geometry, per cell (a failure is reported as a property of the operator)."""
    n_int = len(read.neighbour)
    G = [F(v) for v in g]
    structural = _structural(module(), library)
    if structural:
        return Bound(structural, F(0), F(0), {})
    issues: list[str] = []
    # Premise: the intended operator on exact data.
    xc = exact_geometry["cell_centroid"]
    xf = exact_geometry["face_centroid"]
    q_ex = [F(a) + sum((x[u] * G[u] for u in range(3)), F(0)) for x in xc]
    qb_ex = [F(a) + sum((x[u] * G[u] for u in range(3)), F(0)) for x in xf]
    if method == "lsq":
        premise = lsq_intended(read, exact_geometry, q_ex, qb_ex)
    else:
        premise = gg_intended(read, exact_geometry, q_ex, qb_ex, [G] * read.n_cells)
    issues.extend(
        f"operator: {method} cell {cell} does not reproduce G on exact data"
        for cell, row in enumerate(premise)
        if row != G
    )
    # Correspondence on the exact fixture: the declared sequence with every rounding removed, on the
    # exact field, geometry and injected gradient, is the intended operator, per cell.
    qb_full = [F(0)] * n_int + qb_ex[n_int:]
    exact_read = Read(exact_geometry, q_ex, qb_full, [list(G)] * read.n_cells)  # type: ignore[arg-type]
    if method == "lsq":
        removed = lsq(read, exact_read, pol, Exact())
    else:
        removed = green_gauss(read, exact_read, pol, Exact())
    issues.extend(
        f"invalid: {method} cell {cell}: the declared sequence without rounding is not the "
        "intended operator on the exact fixture"
        for cell in range(read.n_cells)
        if [F(v) for v in removed[cell]] != premise[cell]  # type: ignore[arg-type]
    )
    # The replay from the exact data.
    q = linear_field(geometry["cell_centroid"], a, g)
    q_b = [given(0.0, B64)] * n_int + linear_field(geometry["face_centroid"][n_int:], a, g)
    inp = Inputs(geometry, q, q_b, [[given(v, B64) for v in g]] * read.n_cells, field_exact=False)
    data = load_inputs(inp, pol)
    replay = (
        lsq(read, data, pol, Rounded())
        if method == "lsq"
        else green_gauss(read, data, pol, Rounded())
    )
    ratio, use = F(0), F(0)
    bounds: dict[tuple[int, int], Fraction] = {}
    for cell in range(read.n_cells):
        for u in range(3):
            pair = replay[cell][u]
            assert isinstance(pair, Pair)
            g_hat = F(float(kernel[cell][u]))
            where = f"{method} cell {cell} component {u}"
            if pair.e != g_hat:
                issues.append(f"invalid: {where}: the build does not execute the declared sequence")
            v, r = pair.c.value, pair.c.radius
            d = abs(v - G[u])
            if d > r:
                issues.append(f"invalid: {where}: G lies outside the certified interval")
                continue
            if r > 0:
                ratio = max(ratio, d / r)
            bound = r + d
            bounds[(cell, u)] = bound
            err = abs(g_hat - G[u])
            if err > bound:
                issues.append(f"fail: {where}: |g_hat - G| exceeds R")
            elif bound > 0:
                use = max(use, err / bound)
    return Bound(issues, ratio, use, bounds)


def field_floats(
    geometry: dict[str, list[list[Pair]]], n_int: int, a: float, g: tuple[float, float, float]
) -> tuple[list[float], list[float]]:
    """The executed field the kernel is given: cell values and boundary face values (0 inside)."""
    q = [float(v.e) for v in linear_field(geometry["cell_centroid"], a, g)]
    qb = [0.0] * n_int + [float(v.e) for v in linear_field(geometry["face_centroid"][n_int:], a, g)]
    return q, qb
