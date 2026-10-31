"""Serial CPU loops: `<module>_cpu.hpp`. The reference backend and the host debugging path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autan_codegen import dsl
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
        assert sign is not None
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
    if isinstance(o, dsl.Sum):
        return "A(0)"
    if isinstance(o, dsl.Min):
        return "::autan::kernels::infinity<A>()"
    return "-::autan::kernels::infinity<A>()"


def acc_update(o: dsl.Accumulator, k: int, src: str) -> str:
    if isinstance(o, dsl.Sum):
        return f"acc[{k}] += A({src});"
    # Ties keep the first incidence in list order (section 4.3, reductions).
    op = "lt" if isinstance(o, dsl.Min) else "gt"
    return f"acc[{k}] = ::autan::kernels::{op}(A({src}), acc[{k}]) ? A({src}) : acc[{k}];"


def reduce_body(lm: LoopModel, indent: str, layout: str = "soa") -> list[str]:
    """Statements of one cell's reduction, given `c`; shared by the CPU and GPU cell kernels."""
    loop = lm.loop
    accs = loop.outputs
    n = n_out(lm)
    lines = [f"{indent}A acc[{n}];"]
    lines.extend(
        f"{indent}acc[{off + c}] = {acc_init(o)};"  # type: ignore[arg-type]
        for o, off in zip(accs, offsets(accs), strict=True)
        for c in range(o.dataset.type.size)
    )
    lines.append(
        f"{indent}for (offset_t k = d.topo.cell_face_offsets[c]; k < d.topo.cell_face_offsets[c + 1]; ++k) {{"
    )
    i2 = indent + "  "
    lines.append(f"{i2}const index_t f = d.topo.cell_faces[k];")
    lines.append(
        f"{i2}[[maybe_unused]] const index_t nb = f < d.topo.n_interior_faces ? (d.topo.owner[f] == c ? d.topo.neighbour[f] : d.topo.owner[f]) : -1;"
    )
    lines.append(f"{i2}typename P::compute out[{n}];")
    args = call_args(lm.sig, loop.bindings, {"self": "c", "direct": "k"}, layout=layout)
    lines.append(f"{i2}{lm.fn.name}<P>({', '.join([*args, 'out'])});")
    lines.extend(
        i2 + acc_update(o, off + c, f"out[{off + c}]")  # type: ignore[arg-type]
        for o, off in zip(accs, offsets(accs), strict=True)
        for c in range(o.dataset.type.size)
    )
    lines.append(f"{indent}}}")
    if lm.fin_fn is not None and lm.fin_sig is not None:
        n_fin = len(lm.fin_fn.outputs)
        lines.append(f"{indent}typename P::compute fin[{n_fin}];")
        acc_args = [f"acc[{k}]" for k in range(n)]
        fargs = call_args(
            lm.fin_sig, loop.finalise_bindings, {"direct": "c"}, skip=len(accs), layout=layout
        )
        lines.append(f"{indent}{lm.fin_fn.name}<P>({', '.join([*acc_args, *fargs, 'fin'])});")
        src = "fin"
    else:
        src = "acc"
    lines.extend(
        f"{indent}{at(o.dataset, 'c', c, layout)} = static_cast<{ctype(o.dataset.type.role)}>({src}[{off + c}]);"
        for o, off in zip(accs, offsets(accs), strict=True)
        for c in range(o.dataset.type.size)
    )
    return lines


def emit_loop(e: Emitter, lm: LoopModel) -> None:
    loop = lm.loop
    e.w("template <class P>")
    e.w(f"void {lm.name}(const Data<P>& d) {{")
    e.w("  using A [[maybe_unused]] = typename P::accum;")
    n = n_out(lm)
    if lm.kind in ("face", "boundary"):
        lo, hi = (
            ("0", "d.topo.n_interior_faces")
            if lm.kind == "face"
            else ("d.topo.n_interior_faces", "d.topo.n_faces")
        )
        e.w(f"  for (index_t f = {lo}; f < {hi}; ++f) {{")
        e.w("    const index_t l = d.topo.owner[f];")
        if lm.kind == "face":
            e.w("    const index_t r = d.topo.neighbour[f];")
        e.w(f"    typename P::compute out[{n}];")
        args = call_args(lm.sig, loop.bindings, {"left": "l", "right": "r", "direct": "f"})
        e.w(f"    {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        for s in inc_lines(lm, "left", "l"):
            e.w(s)
        if lm.kind == "face":
            for s in inc_lines(lm, "right", "r"):
                e.w(s)
        e.w("  }")
    elif lm.kind == "cell":
        e.w("  for (index_t c = 0; c < d.topo.n_owned_cells; ++c) {")
        e.w(f"    typename P::compute out[{n}];")
        args = call_args(lm.sig, loop.bindings, {"direct": "c"})
        e.w(f"    {lm.fn.name}<P>({', '.join([*args, 'out'])});")
        for o, off in zip(loop.outputs, offsets(loop.outputs), strict=True):
            for c in range(o.dataset.type.size):
                e.w(
                    f"    {at(o.dataset, 'c', c)} = static_cast<{ctype(o.dataset.type.role)}>(out[{off + c}]);"
                )
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
