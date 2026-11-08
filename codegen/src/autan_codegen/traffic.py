"""Minimum bytes each loop touches, derived from its bindings and its IR.

The convention (docs/infra/bench.md, bytes_per_rep): each element component the loop needs is
read once and written once, at the storage type of its Data<P> member. These are bytes touched,
not bytes moved: re-reads through the maps and cache misses are not counted, so a bandwidth
computed from them is a lower bound on the real traffic.

- A component counts when it reaches a live output through the optimised IR of the loop's bodies
  (the compiler drops the loads of the others): an output a loop writes, a Min or Max that
  selects for a written Follow, an accumulator a finalise reads.
- Each dataset component counts once, over the union of the elements the loop reaches it on
  (`_UNION`): a cell read as `self` and across the entries counts once over the cells touched.
- Outputs: Inc read and written, Write and written accumulators written, Jacobian blocks and
  per-entry values written; a reduction adjoint reads and zeroes its seeds and increments the
  adjoints of its active arguments.
- The maps a loop reads: owner and neighbour per face, offsets (or lengths) and entries of a
  reduction, the column index or owner and neighbour of an incidence.

Not counted: strategy scratch and the deterministic adjoint's twin pass (store-gather, atomics),
colour lists, Topology::face_reversed (face loops are counted in the mesh's orientation), and
parameters.

Element counts come at run time from autan::kernels::Counts (autan/kernels/traffic.hpp), filled
from the real Topology (`counts`): every term is a linear combination of its fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autan_codegen import dsl
from autan_codegen.derive import ReduceAdjoint, adj_ds
from autan_codegen.errors import Code, CodegenError
from autan_codegen.ir import Function, dce

if TYPE_CHECKING:
    from autan_codegen.model import LoopModel

# Fields of autan::kernels::Counts, in declaration order.
COUNTS = (
    "owned_cells",
    "face_owner_cells",
    "face_neighbour_cells",
    "face_cells",
    "owned_interior_faces",
    "boundary_faces",
    "owned_boundary_faces",
    "boundary_cells",
    "all_boundary_cells",
    "faces",
    "incidences",
    "incidence_cells",
    "incidence_touched_cells",
    "stencil",
    "stencil_rows",
    "stencil_cells",
    "stencil_touched_cells",
    "stencil_owned_cells",
    "stencil_boundary_faces",
)
# Layout flags of autan::kernels::Counts: which maps a reduction reads.
FLAGS = ("stencil_sliced", "incidence_sliced", "incidence_cell")

# Element sets that are not fields: their counts as combinations of fields.
_DERIVED = {"incidence_interior_faces": {"faces": 1, "owned_boundary_faces": -1}}

# The set of elements two sets of one dataset reach together (each contains the other or is
# their union field); a pair absent here is refused rather than double counted.
_UNION = {
    frozenset({"owned_cells", "boundary_cells"}): "owned_cells",
    frozenset({"owned_cells", "stencil_rows"}): "owned_cells",
    frozenset({"face_owner_cells", "face_neighbour_cells"}): "face_cells",
    frozenset({"face_cells", "face_owner_cells"}): "face_cells",
    frozenset({"face_cells", "face_neighbour_cells"}): "face_cells",
    frozenset({"stencil_rows", "stencil_cells"}): "stencil_touched_cells",
    frozenset({"stencil_touched_cells", "stencil_rows"}): "stencil_touched_cells",
    frozenset({"stencil_touched_cells", "stencil_cells"}): "stencil_touched_cells",
    frozenset({"owned_cells", "stencil_cells"}): "stencil_owned_cells",
    frozenset({"owned_cells", "stencil_touched_cells"}): "stencil_owned_cells",
    frozenset({"stencil_owned_cells", "owned_cells"}): "stencil_owned_cells",
    frozenset({"stencil_owned_cells", "stencil_rows"}): "stencil_owned_cells",
    frozenset({"stencil_owned_cells", "stencil_cells"}): "stencil_owned_cells",
    frozenset({"stencil_owned_cells", "stencil_touched_cells"}): "stencil_owned_cells",
    frozenset({"owned_cells", "incidence_cells"}): "incidence_touched_cells",
    frozenset({"incidence_touched_cells", "owned_cells"}): "incidence_touched_cells",
    frozenset({"incidence_touched_cells", "incidence_cells"}): "incidence_touched_cells",
    frozenset({"boundary_cells", "all_boundary_cells"}): "all_boundary_cells",
    frozenset({"faces", "owned_interior_faces"}): "faces",
    frozenset({"faces", "owned_boundary_faces"}): "faces",
    frozenset({"faces", "incidence_interior_faces"}): "faces",
    frozenset({"incidence_interior_faces", "owned_boundary_faces"}): "faces",
    frozenset({"owned_boundary_faces", "boundary_faces"}): "boundary_faces",
    frozenset({"stencil_boundary_faces", "boundary_faces"}): "boundary_faces",
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


def _of(where: str) -> tuple[tuple[str, int], ...]:
    return _count(**_DERIVED.get(where, {where: 1}))


def used(fn: Function, live: list[int] | None = None) -> set[tuple[int, int]]:
    """(argument, component) of every input that reaches the outputs `live` (indices into
    fn.outputs; None: all) through the function's instructions."""
    if live is not None:
        kept = Function(
            fn.name, fn.arg_names, fn.arg_sizes, list(fn.instrs), [fn.outputs[i] for i in live]
        )
        fn = dce(kept)
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


class _Touched:
    """Per dataset component: the element sets it is reached on and whether it is read, written
    or both."""

    def __init__(self) -> None:
        self.sets: dict[str, dict[int, set[str]]] = {}
        self.modes: dict[str, dict[int, set[str]]] = {}

    def add(self, name: str, where: str, comps: set[int] | range, mode: str) -> None:
        for c in comps:
            self.sets.setdefault(name, {}).setdefault(c, set()).add(where)
            self.modes.setdefault(name, {}).setdefault(c, set()).update(mode)

    def terms(self) -> list[Term]:
        out: dict[tuple[str, str, int], int] = {}
        for name, by_comp in self.sets.items():
            for c, places in by_comp.items():
                where = _union(places)
                times = 2 if self.modes[name][c] == {"r", "w"} else 1
                key = (name, where, times)
                out[key] = out.get(key, 0) + 1
        return [Term(_of(where), name, n * times) for (name, where, times), n in out.items()]


def _union(places: set[str]) -> str:
    sets = set(places)
    while len(sets) > 1:
        a, b = sorted(sets)[:2]
        u = _UNION.get(frozenset({a, b}))
        if u is None:
            raise CodegenError(Code.INVALID_BINDING, f"no element count for {a} together with {b}")
        sets -= {a, b}
        sets.add(u)
    return sets.pop()


def _where(kind: str, over: str, b: dsl.Binding, finalise: bool = False) -> str:
    """The elements a binding's loads reach in a loop of `kind`."""
    ds = b.dataset
    assert isinstance(ds, dsl.Dataset)
    cells = ds.set is dsl.MeshSet.CELLS
    stencil = over == dsl.STENCIL
    if finalise or kind == "cell":
        return "owned_cells"
    if kind in ("face", "face_write"):
        if not cells:
            return "owned_interior_faces"
        return "face_owner_cells" if b.via == "left" else "face_neighbour_cells"
    if kind == "boundary":
        return "boundary_cells" if cells else "owned_boundary_faces"
    if kind == "boundary_write":
        return "all_boundary_cells" if cells else "boundary_faces"
    # Reductions: self in the cells whose rows have entries, other across the entries.
    if cells:
        if b.via == "self":
            return "stencil_rows" if stencil else "owned_cells"
        return "stencil_cells" if stencil else "incidence_cells"
    if ds.set is dsl.MeshSet.STENCIL:
        return "stencil"
    if ds.set is dsl.MeshSet.INCIDENCES:
        return "incidences"
    if b.via == "on_boundary":
        return "owned_boundary_faces"
    if b.via in ("near", "far"):
        return "incidence_interior_faces"
    return "faces"


def _components(b: dsl.Binding, comps: set[int], size: int) -> set[int]:
    """The dataset components argument components `comps` load: near and far read either half
    of a (2, ...) dataset, as the entry's cell is the face's left or right one."""
    if b.via in ("near", "far"):
        return comps | {c + size for c in comps}
    return comps


def _bound(
    touched: _Touched, kind: str, over: str, fn: Function, names: list[str],
    bindings: dict[str, dsl.Binding], sizes: list[int], live: list[int] | None = None,
    finalise: bool = False,
) -> None:  # fmt: skip
    """The components of each bound argument of `fn` that reach its outputs `live`."""
    use = used(fn, live)
    for k, name in enumerate(names):
        b = bindings.get(name)
        if b is None or isinstance(b.dataset, dsl.Parameter):
            continue
        comps = {c for (a, c) in use if a == k}
        if not comps:
            continue
        touched.add(
            b.dataset.name, _where(kind, over, b, finalise), _components(b, comps, sizes[k]), "r"
        )
        if b.via == "other" and b.boundary is not None:
            where = "stencil_boundary_faces" if over == dsl.STENCIL else "owned_boundary_faces"
            touched.add(b.boundary.name, where, comps, "r")


def _maps(kind: str, over: str, vias: set[str]) -> list[Term]:
    """The Topology arrays a loop's range reads; `vias`: how its bindings are loaded."""
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
    across = bool(vias & {"other", "near", "far", "on_boundary"})
    if over == dsl.STENCIL:
        if across:
            out.append(Term(_count(stencil=1), "index_t", 1))
        return out
    out.append(Term(_count(incidences=1), "index_t", 1))
    if across:
        # The cell across: the column index, or the owner and neighbour of interior faces.
        out.append(Term(_count(incidences=1), "index_t", 1, "incidence_cell"))
        out.append(Term(_of("incidence_interior_faces"), "index_t", 2, "!incidence_cell"))
        if vias & {"near", "far"}:
            # The half near and far take: the face's owner (read above without the column index).
            out.append(Term(_of("incidence_interior_faces"), "index_t", 1, "incidence_cell"))
    return out


def _write_set(kind: str, o: object = None) -> str:
    """The elements a loop of `kind` writes output `o` to (Inc: reads and writes)."""
    if kind == "face_write":
        return "owned_interior_faces"
    if kind == "boundary_write":
        return "boundary_faces"
    if kind == "boundary":
        return "boundary_cells"
    if kind == "face" and isinstance(o, dsl.Inc):
        if o.left and o.right:
            return "face_cells"
        return "face_owner_cells" if o.left else "face_neighbour_cells"
    return "owned_cells"


def _live(lm: LoopModel) -> list[int] | None:
    """The outputs of a reduction's body that reach what it writes (None: all)."""
    loop = lm.loop
    if lm.kind not in ("reduce", "reduce_stencil") or lm.fin_fn is not None:
        return None
    accs = loop.outputs
    leaders = {o.leader for o in accs if isinstance(o, dsl.Follow) and o.write}
    live, k = [], 0
    for i, o in enumerate(accs):
        size = o.dataset.type.size
        if o.write or i in leaders:
            live += range(k, k + size)
        k += size
    for w in loop.entries:
        live += range(k, k + w.dataset.type.size)
        k += w.dataset.type.size
    return live


def traffic(lm: LoopModel) -> list[Term]:
    """The terms of the minimum bytes loop `lm` touches."""
    ra = lm.radj if isinstance(lm.radj, ReduceAdjoint) else None
    p = ra.primal if ra is not None else lm
    loop = p.loop
    kind = "reduce" if lm.kind in ("reduce_adjoint", "reduce_stencil") else lm.kind
    over = loop.over
    t = _Touched()
    sizes = [ty.size for ty in p.sig.types]
    if ra is None:
        _bound(t, kind, over, lm.fn, lm.sig.names, loop.bindings, sizes, _live(lm))
        for _, jfn in lm.jac_fns:
            _bound(t, kind, over, jfn, lm.sig.names, loop.bindings, sizes)
        if lm.fin_fn is not None and lm.fin_sig is not None:
            fin_sizes = [ty.size for ty in lm.fin_sig.types]
            _bound(t, kind, over, lm.fin_fn, lm.fin_sig.names, loop.finalise_bindings, fin_sizes,
                   finalise=True)  # fmt: skip
    else:
        # The adjoint recomputes the primal only where the finalise's adjoint reads its
        # accumulators or a Min or Max selects; the body's adjoint needs every bound input.
        accs = loop.outputs
        if ra.fin_adj is not None:
            _bound(t, kind, over, p.fn, p.sig.names, loop.bindings, sizes)
        else:
            live, k = [], 0
            for o in accs:
                if isinstance(o, dsl.Min | dsl.Max):
                    live += range(k, k + o.dataset.type.size)
                k += o.dataset.type.size
            if live:
                _bound(t, kind, over, p.fn, p.sig.names, loop.bindings, sizes, live)
        _bound(t, kind, over, ra.body_adj, p.sig.names, loop.bindings, sizes)
        if ra.fin_adj is not None and p.fin_sig is not None:
            fin_sizes = [ty.size for ty in p.fin_sig.types]
            _bound(t, kind, over, ra.fin_adj, p.fin_sig.names, loop.finalise_bindings, fin_sizes,
                   finalise=True)  # fmt: skip
        for o in accs:
            a = adj_ds(o.dataset)
            t.add(a.name, "owned_cells", range(a.type.size), "rw")
        for k in ra.body_active:
            b = loop.bindings[p.sig.names[k]]
            a = adj_ds(b.dataset)  # type: ignore[arg-type]
            t.add(a.name, _where(kind, over, b), range(a.type.size), "rw")
        if p.fin_sig is not None:
            for k in ra.fin_active[len(accs) :]:
                b = loop.finalise_bindings[p.fin_sig.names[k]]
                a = adj_ds(b.dataset)  # type: ignore[arg-type]
                t.add(a.name, "owned_cells", range(a.type.size), "rw")
        vias = {b.via for b in loop.bindings.values()}
        return t.terms() + _maps(kind, over, vias)
    for o in loop.outputs:
        size = o.dataset.type.size
        if isinstance(o, dsl.Inc):
            t.add(o.dataset.name, _write_set(kind, o), range(size), "rw")
        elif isinstance(o, dsl.Write):
            t.add(o.dataset.name, _write_set(kind, o), range(size), "w")
        elif o.write:
            t.add(o.dataset.name, "owned_cells", range(size), "w")
    for w in loop.entries:
        where = "stencil" if over == dsl.STENCIL else "incidences"
        t.add(w.dataset.name, where, range(w.dataset.type.size), "w")
    for j in loop.jacobians:
        where = {"face": "owned_interior_faces", "boundary": "owned_boundary_faces"}.get(
            kind, "owned_cells"
        )
        t.add(j.dataset.name, where, range(j.dataset.type.size), "w")
    return t.terms() + _maps(kind, over, {b.via for b in loop.bindings.values()})


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
