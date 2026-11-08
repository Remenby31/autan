"""Minimum bytes each loop touches, derived from its bindings and its IR.

The convention (docs/infra/bench.md, bytes_per_rep): each element component the loop needs is
read once and written once, at the storage type of its Data<P> member. These are bytes touched,
not bytes moved: re-reads through the maps and cache misses are not counted, so a bandwidth
computed from them is a lower bound on the real traffic. A component counts when the optimised
IR of the loop's bodies uses it (the compiler drops the loads of the others). Counted:
- datasets read through bindings, over the elements the loop's range reaches;
- outputs: Inc read and written, Write and written accumulators written, Jacobian blocks and
  per-entry values written;
- the maps a loop reads: owner and neighbour per face, offsets and entries of a reduction.
Not counted: strategy scratch and the deterministic adjoint's twin pass (store-gather, atomics),
colour lists, Topology::face_reversed, and parameters.

Element counts come at run time from autan::kernels::Counts (autan/kernels/traffic.hpp), filled
from the real Topology: every term below is a linear combination of its fields. A face loop
reads its cells over `cells`: every cell has an interior face on a connected mesh of two cells
or more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.derive import ReduceAdjoint, adj_ds

if TYPE_CHECKING:
    from autan_codegen.ir import Function
    from autan_codegen.model import LoopModel

# Fields of autan::kernels::Counts, in declaration order.
COUNTS = (
    "cells",
    "owned_cells",
    "owned_interior_faces",
    "boundary_faces",
    "owned_boundary_faces",
    "boundary_cells",
    "faces",
    "incidences",
    "stencil",
    "stencil_boundary_faces",
)
# Layout flags of autan::kernels::Counts: which maps a reduction reads.
FLAGS = ("stencil_sliced", "incidence_sliced", "incidence_cell")

# Element sets each set contains: a component read over two sets of one dataset counts once,
# over the larger.
_WITHIN = {
    "owned_cells": {"cells"},
    "boundary_cells": {"owned_cells", "cells"},
    "owned_interior_faces": {"faces"},
    "owned_boundary_faces": {"boundary_faces", "faces"},
    "boundary_faces": set(),
    "stencil_boundary_faces": {"boundary_faces"},
    "faces": set(),
    "cells": set(),
    "incidences": set(),
    "stencil": set(),
}


@dataclass(frozen=True)
class Term:
    """`n` values of `size` (a Data member's name, "index_t" or "offset_t") per element of
    `count` (Counts fields with coefficients, "one" a constant), when the layout flag `when`
    holds ("!flag": when it does not; None: always)."""

    count: tuple[tuple[str, int], ...]
    size: str
    n: int
    when: str | None = None


def _count(**fields: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(fields.items()))


def used(fn: Function) -> set[tuple[int, int]]:
    """(argument, component) of every input the function's instructions or outputs use."""
    inputs = {v: i.attr for v, i in enumerate(fn.instrs) if i.op == "input"}
    out: set[tuple[int, int]] = set()
    for i in fn.instrs:
        for a in i.args:
            if a in inputs:
                out.add(inputs[a])  # type: ignore[arg-type]
    for o in fn.outputs:
        if o in inputs:
            out.add(inputs[o])  # type: ignore[arg-type]
    return out


def _sets(kind: str, b: dsl.Binding) -> str:
    """The element set a binding's loads reach in a loop of `kind`."""
    ds = b.dataset
    assert isinstance(ds, dsl.Dataset)
    cells = ds.set is dsl.MeshSet.CELLS
    if kind in ("face", "face_write"):
        return "cells" if cells else "owned_interior_faces"
    if kind == "boundary":
        return "boundary_cells" if cells else "owned_boundary_faces"
    if kind == "boundary_write":
        return "boundary_cells" if cells else "boundary_faces"
    if kind == "cell":
        return "owned_cells"
    # Reductions (and their adjoints): self over the owned cells, other across their entries.
    if cells:
        return "owned_cells" if b.via == "self" else "cells"
    if ds.set is dsl.MeshSet.STENCIL:
        return "stencil"
    if ds.set is dsl.MeshSet.INCIDENCES:
        return "incidences"
    if b.via == "on_boundary":
        return "owned_boundary_faces"
    return "faces"


class _Reads:
    """Components read per dataset and element set; a component counts over its largest set."""

    def __init__(self) -> None:
        self.sets: dict[str, dict[str, set[int]]] = {}

    def add(self, name: str, where: str, comps: set[int]) -> None:
        self.sets.setdefault(name, {}).setdefault(where, set()).update(comps)

    def terms(self) -> list[Term]:
        out = []
        for name, by_set in self.sets.items():
            for where, comps in by_set.items():
                larger = [s for s in by_set if s in _WITHIN[where]]
                n = len(comps - set().union(*(by_set[s] for s in larger)))
                if n:
                    out.append(Term(_count(**{where: 1}), name, n))
        return out


def _bound_reads(
    reads: _Reads, kind: str, over: str, fn: Function, names: list[str],
    bindings: dict[str, dsl.Binding], skip: int = 0, extra: Function | None = None,
) -> None:  # fmt: skip
    """The components of each bound argument that `fn` (and `extra`, on the same arguments)
    uses, from argument `skip` on."""
    use = used(fn) | (used(extra) if extra is not None else set())
    for k, name in enumerate(names):
        if k < skip:
            continue
        b = bindings.get(name)
        if b is None or isinstance(b.dataset, dsl.Parameter):
            continue
        comps = {c for (a, c) in use if a == k}
        if not comps:
            continue
        where = "owned_cells" if kind == "finalise" else _sets(kind, b)
        reads.add(b.dataset.name, where, comps)
        if b.via == "other" and b.boundary is not None:
            reads.add(b.boundary.name, _boundary_of(over), comps)


def _boundary_of(over: str) -> str:
    return "stencil_boundary_faces" if over == dsl.STENCIL else "owned_boundary_faces"


def _maps(kind: str, over: str, across: bool) -> list[Term]:
    """The Topology arrays a loop's range reads; `across`: a reduction that needs the entry's
    other cell or face."""
    if kind in ("face", "face_write"):
        return [Term(_count(owned_interior_faces=1), "index_t", 2)]
    if kind == "boundary":
        return [Term(_count(owned_boundary_faces=1), "index_t", 1)]
    if kind == "boundary_write":
        return [Term(_count(boundary_faces=1), "index_t", 1)]
    if kind == "cell":
        return []
    flag = "stencil_sliced" if over == dsl.STENCIL else "incidence_sliced"
    out = [
        Term(_count(owned_cells=1, one=1), "offset_t", 1, f"!{flag}"),
        Term(_count(owned_cells=1), "offset_t", 1, flag),
        Term(_count(owned_cells=1), "index_t", 1, flag),
    ]
    if over == dsl.STENCIL:
        if across:
            out.append(Term(_count(stencil=1), "index_t", 1))
        return out
    out.append(Term(_count(incidences=1), "index_t", 1))
    if across:
        out.append(Term(_count(incidences=1), "index_t", 1, "incidence_cell"))
        out.append(Term(_count(faces=1, owned_boundary_faces=-1), "index_t", 2, "!incidence_cell"))
    return out


def _across(lm: LoopModel) -> bool:
    return any(b.via in ("other", "near", "far", "on_boundary") for b in lm.loop.bindings.values())


def _write_set(kind: str) -> str:
    """The elements a loop of `kind` writes its outputs to (Inc: reads and writes)."""
    if kind == "face_write":
        return "owned_interior_faces"
    if kind == "boundary_write":
        return "boundary_faces"
    if kind == "boundary":
        return "boundary_cells"
    if kind == "face":
        return "cells"
    return "owned_cells"


def traffic(lm: LoopModel) -> list[Term]:
    """The terms of the minimum bytes loop `lm` touches."""
    loop = lm.loop
    kind = "reduce" if lm.kind == "reduce_adjoint" else lm.kind
    over = loop.over
    reads = _Reads()
    out: list[Term] = []
    ra = lm.radj if isinstance(lm.radj, ReduceAdjoint) else None
    p = ra.primal if ra is not None else lm
    extra = ra.body_adj if ra is not None else None
    _bound_reads(reads, kind, over, p.fn, p.sig.names, p.loop.bindings, extra=extra)
    for _, jfn in lm.jac_fns:
        _bound_reads(reads, kind, over, jfn, lm.sig.names, loop.bindings)
    fin_fn = ra.fin_adj if ra is not None and ra.fin_adj is not None else p.fin_fn
    if fin_fn is not None and p.fin_sig is not None:
        n_acc = len(p.loop.outputs)
        _bound_reads(
            reads, "finalise", over, fin_fn, p.fin_sig.names, p.loop.finalise_bindings, skip=n_acc
        )
    out += reads.terms()
    out += _maps(kind, over, _across(p))
    if ra is not None:
        # Seeds: the adjoints of the outputs, read then zeroed; the adjoints of the active
        # arguments incremented, of the owned cells (self, finalise) or across the entries.
        for o in p.loop.outputs:
            a = adj_ds(o.dataset)
            out.append(Term(_count(owned_cells=1), a.name, 2 * a.type.size))
        for k in ra.body_active:
            b = p.loop.bindings[p.sig.names[k]]
            a = adj_ds(b.dataset)  # type: ignore[arg-type]
            where = "owned_cells" if b.via == "self" else "cells"
            out.append(Term(_count(**{where: 1}), a.name, 2 * a.type.size))
        if p.fin_sig is not None:
            for k in ra.fin_active[len(p.loop.outputs) :]:
                b = p.loop.finalise_bindings[p.fin_sig.names[k]]
                a = adj_ds(b.dataset)  # type: ignore[arg-type]
                out.append(Term(_count(owned_cells=1), a.name, 2 * a.type.size))
        return out
    for o in loop.outputs:
        size = o.dataset.type.size
        if isinstance(o, dsl.Inc):
            out.append(Term(_count(**{_write_set(kind): 1}), o.dataset.name, 2 * size))
        elif isinstance(o, dsl.Write):
            out.append(Term(_count(**{_write_set(kind): 1}), o.dataset.name, size))
        elif o.write:
            out.append(Term(_count(owned_cells=1), o.dataset.name, size))
    for w in loop.entries:
        where = "stencil" if over == dsl.STENCIL else "incidences"
        out.append(Term(_count(**{where: 1}), w.dataset.name, w.dataset.type.size))
    for j in loop.jacobians:
        where = {"face": "owned_interior_faces", "boundary": "owned_boundary_faces"}.get(
            kind, "owned_cells"
        )
        out.append(Term(_count(**{where: 1}), j.dataset.name, j.dataset.type.size))
    return out


def evaluate(terms: list[Term], counts: dict[str, float], sizes: dict[str, int]) -> float:
    """Bytes of `terms` for `counts` (Counts fields and layout flags) and element `sizes`."""
    total = 0.0
    for t in terms:
        if t.when is not None:
            flag = t.when.lstrip("!")
            if bool(counts[flag]) == t.when.startswith("!"):
                continue
        n = sum(coeff * (1.0 if f == "one" else counts[f]) for f, coeff in t.count)
        total += n * t.n * sizes[t.size]
    return total


def emit(lm: LoopModel) -> list[str]:
    """`template <class P> double <loop>_bytes(const Counts& n)`, the terms in C++."""
    lines = ["template <class P>", f"double {lm.name}_bytes(const ::autan::kernels::Counts& n) {{"]
    lines.append("  [[maybe_unused]] constexpr double one = 1.0;")
    lines.append("  double b = 0.0;")
    for t in traffic(lm):
        count = " + ".join(f"{c} * {'one' if f == 'one' else 'n.' + f}" for f, c in t.count)
        size = (
            f"sizeof(::autan::kernels::{t.size})"
            if t.size in ("index_t", "offset_t")
            else f"::autan::kernels::element_size<decltype(Data<P>::{t.size})>()"
        )
        line = f"b += ({count}) * {t.n}.0 * {size};"
        if t.when is not None:
            flag = t.when.lstrip("!")
            cond = f"!n.{flag}" if t.when.startswith("!") else f"n.{flag}"
            line = f"if ({cond}) {line}"
        lines.append("  " + line)
    lines.append("  return b;")
    lines.append("}")
    return lines
