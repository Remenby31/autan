"""Serial CPU loops: `<module>_cpu.hpp`. The reference backend and the host debugging path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.derive import ReduceAdjoint, adj_ds
from autan_codegen.emit_cpp import Emitter, at, call_args, ctype

if TYPE_CHECKING:
    from autan_codegen.model import LoopModel, Module


def offsets(outputs: list[dsl.Inc | dsl.Write | dsl.Accumulator]) -> list[int]:
    out, k = [], 0
    for o in outputs:
        out.append(k)
        k += o.dataset.type.size
    return out


def inc_lines(
    lm: LoopModel, side: str, index: str, var: str = "out", indent: str = "    "
) -> list[str]:
    lines = []
    for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
        assert isinstance(o, dsl.Inc)
        sign = o.left if side == "left" else o.right
        if not sign:
            continue
        for c in range(o.dataset.type.size):
            target = at(o.dataset, index, c)
            val = f"A({var}[{off + c}])"
            if sign == 1:
                lines.append(f"{indent}{target} += {val};")
            elif sign == -1:
                lines.append(f"{indent}{target} -= {val};")
            else:
                lines.append(f"{indent}{target} += A({sign}) * {val};")
    return lines


def n_out(lm: LoopModel) -> int:
    return sum(o.dataset.type.size for o in lm.loop.outputs)


def acc_init(o: dsl.Accumulator) -> str:
    if isinstance(o, dsl.Sum | dsl.Follow):
        return "A(0)"
    if isinstance(o, dsl.Min):
        return "::autan::kernels::infinity<A>()"
    return "-::autan::kernels::infinity<A>()"


def _updates(accs: list[dsl.Accumulator], offs: list[int], src: str, indent: str) -> list[str]:
    """Accumulator updates for one incidence; Follow accumulators move with their leader."""
    lines = []
    for o, off in zip(accs, offs, strict=True):
        if isinstance(o, dsl.Follow):
            continue
        for c in range(o.dataset.type.size):
            k = off + c
            if isinstance(o, dsl.Sum):
                lines.append(f"{indent}acc[{k}] += A({src}[{k}]);")
                continue
            # Ties keep the first incidence in list order (section 4.3, reductions).
            op = "lt" if isinstance(o, dsl.Min) else "gt"
            lines.append(f"{indent}if (::autan::kernels::{op}(A({src}[{k}]), acc[{k}])) {{")
            lines.append(f"{indent}  acc[{k}] = A({src}[{k}]);")
            for f, foff in zip(accs, offs, strict=True):
                if isinstance(f, dsl.Follow) and accs[f.leader] is o:
                    lines.append(f"{indent}  acc[{foff + c}] = A({src}[{foff + c}]);")
            lines.append(f"{indent}}}")
    return lines


def entry_loop(over: str, indent: str) -> list[str]:
    """Opening of the loop over the entries k of cell c: the faces of c (`incidences`) or its
    stencil (`stencil`). Defines `across` (the entry is a cell `nb`), else boundary face `f`."""
    i2 = indent + "  "
    if over == dsl.STENCIL:
        return [
            # CSR (stride 1) or sliced layout: entry j at stencil_offsets[c] + j * stride.
            (
                f"{indent}for (offset_t k = d.topo.stencil_offsets[c], kn = d.topo.stencil_stride == 1 "
                "? d.topo.stencil_offsets[c + 1] : d.topo.stencil_offsets[c] + "
                "offset_t(d.topo.stencil_length[c]) * d.topo.stencil_stride; "
                "k < kn; k += d.topo.stencil_stride) {"
            ),
            f"{i2}const index_t e = d.topo.stencil[k];",
            f"{i2}[[maybe_unused]] const bool across = e >= 0;",
            f"{i2}[[maybe_unused]] const index_t nb = e;",
            f"{i2}[[maybe_unused]] const index_t f = across ? -1 : -e - 1;",
        ]
    return [
        f"{indent}for (offset_t k = d.topo.cell_face_offsets[c]; k < d.topo.cell_face_offsets[c + 1]; ++k) {{",
        f"{i2}const index_t f = d.topo.cell_faces[k];",
        f"{i2}[[maybe_unused]] const bool across = f < d.topo.n_interior_faces;",
        f"{i2}[[maybe_unused]] const index_t nb = across ? (d.topo.owner[f] == c ? d.topo.neighbour[f] : d.topo.owner[f]) : -1;",
    ]


def reduce_body(lm: LoopModel, indent: str, layout: str = "soa") -> list[str]:
    """Statements of one cell's reduction, given `c`; shared by the CPU and GPU cell kernels."""
    loop = lm.loop
    accs = loop.outputs
    offs = offsets(accs)
    n = n_out(lm)
    lines = [f"{indent}A acc[{n}];"]
    lines.extend(
        f"{indent}acc[{off + c}] = {acc_init(o)};"  # type: ignore[arg-type]
        for o, off in zip(accs, offs, strict=True)
        for c in range(o.dataset.type.size)
    )
    lines.extend(entry_loop(loop.over, indent))
    i2 = indent + "  "
    lines.append(f"{i2}typename P::compute out[{n}];")
    args = call_args(
        lm.sig, loop.bindings, {"self": "c", "direct": "k", "face": "f"}, layout=layout
    )
    lines.append(f"{i2}{lm.fn.name}<P>({', '.join([*args, 'out'])});")
    lines.extend(_updates(accs, offs, "out", i2))  # type: ignore[arg-type]
    lines.append(f"{indent}}}")
    written = [(o, off) for o, off in zip(accs, offs, strict=True) if o.write]  # type: ignore[union-attr]
    if lm.fin_fn is not None and lm.fin_sig is not None:
        n_fin = len(lm.fin_fn.outputs)
        lines.append(f"{indent}typename P::compute fin[{n_fin}];")
        order = lm.fin_args or [("acc", k) for k in range(len(accs))] + [
            ("bind", nm) for nm in lm.fin_sig.names[len(accs) :]
        ]
        fargs: list[str] = []
        for kind, ref in order:
            if kind == "acc":
                assert isinstance(ref, int)
                fargs += [f"acc[{offs[ref] + c}]" for c in range(accs[ref].dataset.type.size)]
            else:
                assert isinstance(ref, str)
                b = loop.finalise_bindings[ref]
                t = lm.fin_sig.types[lm.fin_sig.names.index(ref)]
                fargs += [at(b.dataset, "c", c, layout) for c in range(t.size)]
        lines.append(f"{indent}{lm.fin_fn.name}<P>({', '.join([*fargs, 'fin'])});")
        k = 0
        for o, _ in written:
            for c in range(o.dataset.type.size):
                lines.append(
                    f"{indent}{at(o.dataset, 'c', c, layout)} = static_cast<{ctype(o.dataset.type.role)}>(fin[{k}]);"
                )
                k += 1
    else:
        lines.extend(
            f"{indent}{at(o.dataset, 'c', c, layout)} = static_cast<{ctype(o.dataset.type.role)}>(acc[{off + c}]);"
            for o, off in written
            for c in range(o.dataset.type.size)
        )
    return lines


def other_targets(lm: LoopModel) -> list[tuple[int, int]]:
    """(argument, component) of each active argument read across incidences (`other`): the
    per-incidence scratch columns of the deterministic reduction adjoint."""
    ra = lm.radj
    assert isinstance(ra, ReduceAdjoint)
    p = ra.primal
    return [
        (k, c)
        for k in ra.body_active
        if p.loop.bindings[p.sig.names[k]].via == "other"
        for c in range(p.sig.types[k].size)
    ]


def entries(over: str) -> str:
    """Number of entries of all cells, the stride of per-entry data and scratch."""
    return "d.topo.n_stencil" if over == dsl.STENCIL else "d.topo.n_incidences"


def twin_gather_body(lm: LoopModel, indent: str) -> list[str]:
    """Second pass of the deterministic reduction adjoint, for cell c (owned or ghost): add, in
    the order of c's entries, what each owned neighbour stored for c on its twin entry (the same
    face, or c in the neighbour's stencil)."""
    ra = lm.radj
    assert isinstance(ra, ReduceAdjoint)
    p = ra.primal
    over = p.loop.over
    i2 = indent + "  "
    twin = "d.topo.stencil_twin" if over == dsl.STENCIL else "d.topo.incidence_twin"
    lines = [
        *entry_loop(over, indent),
        f"{i2}if (!across || nb >= d.topo.n_owned_cells) {{",
        f"{i2}  continue;",
        f"{i2}}}",
        f"{i2}const offset_t kt = {twin}[k];",
    ]
    for m_other, (k, c) in enumerate(other_targets(lm)):
        a = adj_ds(p.loop.bindings[p.sig.names[k]].dataset)  # type: ignore[arg-type]
        lines.append(f"{i2}{at(a, 'c', c)} += scratch[{m_other} * {entries(over)} + kt];")
    lines.append(f"{indent}}}")
    return lines


def reduce_adjoint_body(lm: LoopModel, indent: str, scatter: str) -> list[str]:
    """Adjoint of one cell's reduction (given `c`): recompute the accumulators and the incidence
    each Min/Max selected, apply the finalise adjoint, then send each accumulator's adjoint to
    the incidences that fed it (all for Sum, the selected one for Min/Max) through the body's
    adjoint. `scatter` says how the increments of other cells' adjoints are made: "plain"
    (serial), "atomic", or "store" (to the per-incidence scratch, summed afterwards in a fixed
    order by twin_gather_body); in "store" mode the cell's own adjoints are plain increments, as
    no other thread writes them in that pass."""
    ra = lm.radj
    assert isinstance(ra, ReduceAdjoint)
    p = ra.primal
    loop = p.loop
    accs = loop.outputs
    offs = offsets(accs)
    n = n_out(p)
    i2 = indent + "  "

    def add(target: str, v: str, ind: str) -> str:
        if scatter == "store":
            return f"{ind}{target} += {v};"
        if scatter == "atomic":
            return f"{ind}atomicAdd(&{target}, {v});"
        return f"{ind}{target} += {v};"

    selects = any(isinstance(o, dsl.Min | dsl.Max) for o in accs)
    lines = [f"{indent}A acc[{n}];"]
    if selects:
        lines.append(f"{indent}offset_t sel[{n}];")
    for o, off in zip(accs, offs, strict=True):
        for c in range(o.dataset.type.size):
            lines.append(f"{indent}acc[{off + c}] = {acc_init(o)};")  # type: ignore[arg-type]
            if selects:
                lines.append(f"{indent}sel[{off + c}] = -1;")
    head = entry_loop(loop.over, indent)
    args = call_args(p.sig, loop.bindings, {"self": "c", "direct": "k", "face": "f"})
    lines += [*head, f"{i2}typename P::compute out[{n}];"]
    lines.append(f"{i2}{p.fn.name}<P>({', '.join([*args, 'out'])});")
    for o, off in zip(accs, offs, strict=True):
        for c in range(o.dataset.type.size):
            k = off + c
            if isinstance(o, dsl.Sum):
                lines.append(f"{i2}acc[{k}] += A(out[{k}]);")
            else:
                op = "lt" if isinstance(o, dsl.Min) else "gt"
                lines.append(f"{i2}if (::autan::kernels::{op}(A(out[{k}]), acc[{k}])) {{")
                lines.append(f"{i2}  acc[{k}] = A(out[{k}]);")
                lines.append(f"{i2}  sel[{k}] = k;")
                lines.append(f"{i2}}}")
    lines.append(f"{indent}}}")
    # Seeds: the adjoints of the written datasets, read then zeroed (they were overwritten).
    seeds = [at(adj_ds(o.dataset), "c", c) for o in accs for c in range(o.dataset.type.size)]
    lines.append(f"{indent}A accb[{n}];")
    if ra.fin_adj is not None and p.fin_sig is not None:
        n_acc = len(accs)
        fargs = [f"acc[{k}]" for k in range(n)]
        fargs += call_args(p.fin_sig, loop.finalise_bindings, {"direct": "c"}, skip=n_acc)
        lines.append(f"{indent}typename P::compute finb[{len(ra.fin_adj.outputs)}];")
        lines.append(f"{indent}{ra.fin_adj.name}<P>({', '.join([*fargs, *seeds, 'finb'])});")
        for k in range(n):
            lines.append(f"{indent}accb[{k}] = A(finb[{k}]);")
        j = n
        for k in ra.fin_active[n_acc:]:
            b = loop.finalise_bindings[p.fin_sig.names[k]]
            a = adj_ds(b.dataset)  # type: ignore[arg-type]
            for c in range(p.fin_sig.types[k].size):
                lines.append(add(at(a, "c", c), f"A(finb[{j}])", indent))
                j += 1
    else:
        for k, sd in enumerate(seeds):
            lines.append(f"{indent}accb[{k}] = A({sd});")
    for sd in seeds:
        lines.append(f"{indent}{sd} = 0;")
    # Reverse sweep over the incidences.
    lines += head
    seed_vals = []
    for o, off in zip(accs, offs, strict=True):
        for c in range(o.dataset.type.size):
            kk = off + c
            seed_vals.append(
                f"accb[{kk}]" if isinstance(o, dsl.Sum) else f"(sel[{kk}] == k ? accb[{kk}] : A(0))"
            )
    n_bar = len(ra.body_adj.outputs)
    lines.append(f"{i2}typename P::compute bar[{n_bar}];")
    lines.append(f"{i2}{ra.body_adj.name}<P>({', '.join([*args, *seed_vals, 'bar'])});")
    j = 0
    for k in ra.body_active:
        b = loop.bindings[p.sig.names[k]]
        a = adj_ds(b.dataset)  # type: ignore[arg-type]
        for c in range(p.sig.types[k].size):
            if b.via == "self":
                lines.append(add(at(a, "c", c), f"A(bar[{j}])", i2))
            elif scatter == "store":
                m_other = len(other_targets(lm)[: other_targets(lm).index((k, c))])
                lines.append(f"{i2}scratch[{m_other} * {entries(loop.over)} + k] = A(bar[{j}]);")
            else:
                lines.append(f"{i2}if (across) {{")
                lines.append(add(at(a, "nb", c), f"A(bar[{j}])", i2 + "  "))
                lines.append(f"{i2}}}")
            j += 1
    lines.append(f"{indent}}}")
    return lines


def cell_stores(lm: LoopModel, indent: str) -> list[str]:
    """Write (=) or Inc (+=) of a cell loop's outputs at cell c."""
    lines = []
    for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True):
        for c in range(o.dataset.type.size):
            target = at(o.dataset, "c", c)
            if isinstance(o, dsl.Inc):
                lines.append(f"{indent}{target} += A(out[{off + c}]);")
            else:
                lines.append(
                    f"{indent}{target} = static_cast<{ctype(o.dataset.type.role)}>(out[{off + c}]);"
                )
    return lines


def boundary_write_body(lm: LoopModel, indent: str) -> list[str]:
    """Body of a boundary loop writing face datasets, for boundary face f."""
    lines = [f"{indent}[[maybe_unused]] const index_t l = d.topo.owner[f];"]
    lines.append(f"{indent}typename P::compute out[{n_out(lm)}];")
    args = call_args(lm.sig, lm.loop.bindings, {"left": "l", "direct": "f"})
    lines.append(f"{indent}{lm.fn.name}<P>({', '.join([*args, 'out'])});")
    lines.extend(
        f"{indent}{at(o.dataset, 'f', c)} = static_cast<{ctype(o.dataset.type.role)}>(out[{off + c}]);"
        for o, off in zip(lm.loop.outputs, offsets(lm.loop.outputs), strict=True)
        for c in range(o.dataset.type.size)
    )
    return lines


def jacobian_lines(lm: LoopModel, args: list[str], index: str, indent: str) -> list[str]:
    """Evaluate each Jacobian output's derivative function on the same arguments as the body
    and write it at entity `index` (a face or a cell) in the Jac storage type."""
    lines = []
    for n, (j, jfn) in enumerate(lm.jac_fns):
        size = j.dataset.type.size
        lines.append(f"{indent}typename P::compute jac{n}[{size}];")
        lines.append(f"{indent}{jfn.name}<P>({', '.join([*args, f'jac{n}'])});")
        lines.extend(
            f"{indent}{at(j.dataset, index, c)} = static_cast<typename P::jac>(jac{n}[{c}]);"
            for c in range(size)
        )
    return lines


def emit_loop(e: Emitter, lm: LoopModel) -> None:
    loop = lm.loop
    e.w("template <class P>")
    e.w(f"void {lm.name}(const Data<P>& d) {{")
    e.w("  using A [[maybe_unused]] = typename P::accum;")
    n = n_out(lm) if lm.kind != "reduce_adjoint" else 0
    if lm.kind in ("face", "boundary"):
        lo, hi = (
            ("0", "d.topo.n_owned_interior_faces")
            if lm.kind == "face"
            else ("d.topo.n_interior_faces", "d.topo.n_owned_faces")
        )
        e.w(f"  for (index_t f = {lo}; f < {hi}; ++f) {{")
        e.w("    const index_t l = d.topo.owner[f];")
        if lm.kind == "face":
            e.w("    const index_t r = d.topo.neighbour[f];")
        e.w(f"    typename P::compute out[{n}];")
        args = call_args(lm.sig, loop.bindings, {"left": "l", "right": "r", "direct": "f"})
        e.w(f"    {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        for s in jacobian_lines(lm, args, "f", "    "):
            e.w(s)
        for s in inc_lines(lm, "left", "l"):
            e.w(s)
        if lm.kind == "face":
            for s in inc_lines(lm, "right", "r"):
                e.w(s)
        e.w("  }")
    elif lm.kind == "boundary_write":
        e.w("  for (index_t f = d.topo.n_interior_faces; f < d.topo.n_faces; ++f) {")
        for s in boundary_write_body(lm, "    "):
            e.w(s)
        e.w("  }")
    elif lm.kind == "cell":
        e.w("  for (index_t c = 0; c < d.topo.n_owned_cells; ++c) {")
        e.w(f"    typename P::compute out[{n}];")
        args = call_args(lm.sig, loop.bindings, {"direct": "c"})
        e.w(f"    {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        for s in jacobian_lines(lm, args, "c", "    "):
            e.w(s)
        for s in cell_stores(lm, "    "):
            e.w(s)
        e.w("  }")
    elif lm.kind == "reduce_adjoint":
        e.w("  for (index_t c = 0; c < d.topo.n_owned_cells; ++c) {")
        for s in reduce_adjoint_body(lm, "    ", "plain"):
            e.w(s)
        e.w("  }")
    else:
        e.w("  for (index_t c = 0; c < d.topo.n_owned_cells; ++c) {")
        for s in reduce_body(lm, "    "):
            e.w(s)
        e.w("  }")
    e.w("}")
    e.w()


def emit_cpu_header(m: Module) -> str:
    e = Emitter(f"{m.name}_cpu.hpp", line_directives=False)
    e.w(f"// Generated by autan_codegen from {m.path.name}. Do not edit.")
    e.w("#pragma once")
    e.w(f'#include "{m.name}.hpp"')
    e.w()
    e.w(f"namespace autan::kernels::{m.name}::cpu {{")
    e.w("using ::autan::kernels::index_t;")
    e.w("using ::autan::kernels::offset_t;")
    e.w()
    for lm in m.loops:
        emit_loop(e, lm)
    e.w(f"}}  // namespace autan::kernels::{m.name}::cpu")
    return e.text()
