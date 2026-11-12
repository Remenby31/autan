"""Minimum bytes per loop (autan_codegen.traffic) against counts made by hand from the kernels."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from autan_codegen import model, traffic
from autan_codegen.derive import derive
from autan_codegen.emit_cpp import rel_datasets, storage_type
from autan_codegen.errors import Code, CodegenError
from autan_codegen.generate import generate

ROOT = Path(__file__).resolve().parents[2]

# Every Counts field a distinct prime, so that a term counted over the wrong set changes the total.
PRIMES = dict(
    zip(
        traffic.COUNTS,
        (
            1031,
            1021,
            1019,
            1013,
            1009,
            997,
            991,
            983,
            977,
            971,
            967,
            953,
            947,
            941,
            937,
            929,
            919,
            911,
            907,
        ),
        strict=True,
    )
)
CSR = {**PRIMES, "stencil_sliced": False, "incidence_sliced": False, "incidence_cell": False}
# autan/kernels/runtime.hpp: index_t is int32_t, offset_t int64_t; the policies' storage types.
INDEX = {"index_t": 4, "offset_t": 8}
ROLE_BYTES = {
    "fp64": {"state": 8, "grad": 8, "residual": 8, "geom": 8, "geom_value": 8, "jac": 8},
    "fp32+acc64": {"state": 4, "grad": 4, "residual": 8, "geom": 8, "geom_value": 4, "jac": 4},
}

MODULE = """
from autan_codegen.dsl import (
    Geom, Inc, Jac, Jacobian, Max, Residual, State, Sum, Vec5, Write, boundary_loop, cell_loop,
    cell_reduce, cells, face_loop, faces, geometry, pointwise, rel,
)

u = cells.field("u", State.Vec5)
sc = cells.field("sc", State)
w = cells.field("w", State.Vec5)
res = cells.field("res", Residual.Vec5)
res2 = cells.field("res2", Residual.Vec5)
ub = faces.field("ub", State)
g = cells.field("g", State.Vec3)
top = cells.field("top", State)
jac = faces.field("jac", Jac.shape(2, 5, 5))
h = faces.field("h", State.Vec2)
hs = cells.field("hs", State)


@pointwise
def copy(q: State.Vec5) -> State.Vec5:
    return q


direct = cell_loop(copy, q=u, out=Write(w))


@pointwise
def flux(ql: State.Vec5, qr: State.Vec5, a: Geom) -> State.Vec5:
    s = (ql[0] + qr[1]) * a
    return Vec5(s, s, s, s, s)


partial = face_loop(flux, ql=u.left, qr=u.right, a=geometry.face_area_magnitude,
                    out=Inc(res, left=+1, right=-1))
with_blocks = face_loop(flux, ql=u.left, qr=u.right, a=geometry.face_area_magnitude,
                        out=[Inc(res2, left=+1, right=-1), Jacobian(jac, wrt=("ql", "qr"))])


@pointwise
def spread(xc: Geom.Vec3, xn: Geom.Vec3, v: Geom) -> State.Vec3:
    return rel(xn, xc) * v


mixed = cell_reduce(spread, xc=geometry.cell_centroid.self_,
                    xn=geometry.cell_centroid.other(boundary=geometry.face_centroid),
                    v=geometry.cell_volume.self_, acc=Sum(g))


@pointwise
def pick(q: State) -> State:
    return q


largest = cell_reduce(pick, q=sc.other(boundary=ub), acc=Max(top))

fb = faces.field("fb", State)


@pointwise
def first(q: State.Vec5) -> State:
    return q[0]


face_values = boundary_loop(first, q=u.left, out=Write(fb), frozen=True)


@pointwise
def half(a: State, b: State) -> State:
    return a


halves = cell_reduce(half, a=h.near, b=h.far, acc=Sum(hs))

tot = cells.field("tot", State)


@pointwise
def twice(a: Residual) -> State:
    return 2.0 * a


doubled = cell_reduce(pick, q=sc.other(boundary=ub), acc=Sum(tot), finalise=twice)
"""

# The same face loop reading one more component of the right cell (qr[2]).
WIDER = MODULE.replace("s = (ql[0] + qr[1]) * a", "s = (ql[0] + qr[1] + qr[2]) * a")


def _module(tmp_path: Path, text: str) -> model.Module:
    src = tmp_path / "traffic_test.py"
    src.write_text(text)
    m = model.load(src)
    derive(m)
    return m


def _sizes(m: model.Module, policy: str) -> dict[str, int]:
    rel_read = rel_datasets(m)
    roles = ROLE_BYTES[policy]
    out = dict(INDEX)
    for d in m.datasets:
        out[d.name] = roles[storage_type(d, rel_read).removeprefix("typename P::")]
    return out


def _bytes(m: model.Module, loop: str, policy: str, counts: dict[str, float] = CSR) -> float:
    lm = next(lm for lm in m.loops if lm.name == loop)
    return traffic.evaluate(traffic.traffic(lm), counts, _sizes(m, policy))


@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_a_cell_loop_reads_and_writes_each_component_of_its_owned_cells_once(
    tmp_path: Path, policy: str
) -> None:
    m = _module(tmp_path, MODULE)
    s = ROLE_BYTES[policy]["state"]
    assert _bytes(m, "direct", policy) == PRIMES["owned_cells"] * (5 + 5) * s


@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_a_face_loop_counts_the_components_its_body_reads_through_the_maps(
    tmp_path: Path, policy: str
) -> None:
    m = _module(tmp_path, MODULE)
    sz = ROLE_BYTES[policy]
    p = PRIMES
    # ql[0] of the faces' owners and qr[1] of their neighbours, |S_f| per owned interior face,
    # the residual read and written in the faces' cells, owner and neighbour per face.
    partial = (
        p["face_owner_cells"] * sz["state"]
        + p["face_neighbour_cells"] * sz["state"]
        + p["owned_interior_faces"] * sz["geom_value"]
        + p["face_cells"] * 2 * 5 * sz["residual"]
        + p["owned_interior_faces"] * 2 * INDEX["index_t"]
    )
    assert _bytes(m, "partial", policy) == partial
    # The Jacobian blocks (2, 5, 5) per owned interior face, written.
    blocks = p["owned_interior_faces"] * 50 * sz["jac"]
    assert _bytes(m, "with_blocks", policy) == partial + blocks


@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_a_reduction_counts_geometry_at_its_storage_and_the_incidence_maps(
    tmp_path: Path, policy: str
) -> None:
    m = _module(tmp_path, MODULE)
    sz = ROLE_BYTES[policy]
    p = PRIMES
    # Centroids are read through rel(): stored as geom (FP64 in both policies); the volume as a
    # value only: geom_value. Own centroid and the one across count once, over the owned cells and
    # the cells across.
    expected = (
        p["incidence_touched_cells"] * 3 * sz["geom"]
        + p["owned_boundary_faces"] * 3 * sz["geom"]
        + p["owned_cells"] * sz["geom_value"]
        + p["owned_cells"] * 3 * sz["state"]
        + (p["owned_cells"] + 1) * INDEX["offset_t"]
        + p["incidences"] * INDEX["index_t"]
        + (p["faces"] - p["owned_boundary_faces"]) * 2 * INDEX["index_t"]
    )
    assert _bytes(m, "mixed", policy) == expected
    # The sliced layout reads the lengths and the column index instead of owner and neighbour.
    sliced = {**CSR, "incidence_sliced": True, "incidence_cell": True}
    expected_sliced = (
        expected
        - (p["owned_cells"] + 1) * INDEX["offset_t"]
        - (p["faces"] - p["owned_boundary_faces"]) * 2 * INDEX["index_t"]
        + p["owned_cells"] * (INDEX["offset_t"] + INDEX["index_t"])
        + p["incidences"] * INDEX["index_t"]
    )
    assert _bytes(m, "mixed", policy, sliced) == expected_sliced


@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_a_boundary_loop_writing_face_data_runs_over_every_boundary_face(
    tmp_path: Path, policy: str
) -> None:
    m = _module(tmp_path, MODULE)
    s = ROLE_BYTES[policy]["state"]
    p = PRIMES
    # Its range is [n_interior_faces, n_faces), owned or not (emit_cpu.face_write_range): q[0] of
    # each owner, fb written and the owner index per boundary face.
    expected = (
        p["all_boundary_cells"] * s
        + p["boundary_faces"] * s
        + p["boundary_faces"] * INDEX["index_t"]
    )
    assert _bytes(m, "face_values", policy) == expected


@pytest.mark.parametrize("policy", ["fp64", "fp32+acc64"])
def test_near_reads_one_half_per_interior_incidence_of_the_owned_cells(
    tmp_path: Path, policy: str
) -> None:
    m = _module(tmp_path, MODULE)
    s = ROLE_BYTES[policy]["state"]
    p = PRIMES
    interior = p["faces"] - p["owned_boundary_faces"]
    # `a` (near) takes half 0 of a face for its left cell and half 1 for its right one: one half
    # per incidence of an owned cell on an interior face (on a partition, a face whose other cell
    # is a ghost gives one half only); `b` (far) reaches no output. hs written.
    halves = p["incidences"] - p["owned_boundary_faces"]
    common = halves * s + p["owned_cells"] * s + p["incidences"] * INDEX["index_t"]
    csr = common + (p["owned_cells"] + 1) * INDEX["offset_t"] + interior * 2 * INDEX["index_t"]
    assert _bytes(m, "halves", policy) == csr
    # Sliced, with the column index: lengths, the column index, and the owner choosing the half.
    sliced = {**CSR, "incidence_sliced": True, "incidence_cell": True}
    expected = (
        common
        + p["owned_cells"] * (INDEX["offset_t"] + INDEX["index_t"])
        + p["incidences"] * INDEX["index_t"]
        + interior * INDEX["index_t"]
    )
    assert _bytes(m, "halves", policy, sliced) == expected


def test_a_tangent_reads_no_primal_input_its_finalise_does_not_use(tmp_path: Path) -> None:
    # doubled's finalise 2 a is linear: its tangent needs the tangent sum only, so the tangent
    # loop reads sc_tan across the entries and writes tot_tan, and reads neither sc nor ub.
    m = _module(tmp_path, MODULE)
    p = PRIMES
    s = 8
    maps = (
        (p["owned_cells"] + 1) * INDEX["offset_t"]
        + p["incidences"] * INDEX["index_t"]
        + (p["faces"] - p["owned_boundary_faces"]) * 2 * INDEX["index_t"]
    )
    tangent = p["incidence_cells"] * s + p["owned_cells"] * s + maps
    assert _bytes(m, "doubled_tangent", "fp64") == tangent
    lm = next(lm for lm in m.loops if lm.name == "doubled_tangent")
    assert {t.size for t in traffic.traffic(lm)} == {"sc_tan", "tot_tan", "index_t", "offset_t"}


MIXED = """
from autan_codegen.dsl import State, Sum, cell_reduce, cells, faces, pointwise

h = faces.field("h", State.Vec2)
hs = cells.field("hs", State)


@pointwise
def half_and_whole(a: State, b: State.Vec2) -> State:
    return a + b[1]


mixed_halves = cell_reduce(half_and_whole, a=h.near, b=h, acc=Sum(hs))
"""


def test_a_dataset_read_by_halves_and_whole_in_one_loop_gets_no_byte_count(
    tmp_path: Path,
) -> None:
    # near's components index a half, the direct binding's the whole (2, ...) dataset: they are
    # not the same elements, so the derivation refuses rather than counting them apart.
    m = _module(tmp_path, MIXED)
    lm = next(lm for lm in m.loops if lm.name == "mixed_halves")
    with pytest.raises(CodegenError) as e:
        traffic.traffic(lm)
    assert e.value.code is Code.INVALID_BINDING


TOPOLOGY = """
#include <cstdio>
#include "autan/kernels/traffic.hpp"

int main() {
  using autan::kernels::index_t;
  using autan::kernels::offset_t;
  // Cells 0 | 1 | 2 in a row, 2 a halo cell: interior faces 0 (0|1) and 1 (1|2), boundary faces
  // 2 (of 0, owned) and 3 (of the halo cell 2). Stencils: 0 -> {1, face 2}, 1 -> {0, 2}, 2 -> {}.
  const index_t owner[] = {0, 1, 0, 2};
  const index_t neighbour[] = {1, 2};
  const offset_t cell_face_offsets[] = {0, 2, 4, 6};
  const index_t cell_faces[] = {0, 2, 0, 1, 1, 3};
  const offset_t stencil_offsets[] = {0, 2, 4, 4};
  const index_t stencil[] = {1, -3, 0, 2};
  autan::kernels::Topology t{};
  t.n_cells = 3;
  t.n_owned_cells = 2;
  t.n_faces = 4;
  t.n_interior_faces = 2;
  t.n_owned_interior_faces = 2;
  t.n_owned_faces = 3;
  t.n_incidences = 6;
  t.owner = owner;
  t.neighbour = neighbour;
  t.cell_face_offsets = cell_face_offsets;
  t.cell_faces = cell_faces;
  t.n_stencil = 4;
  t.stencil_offsets = stencil_offsets;
  t.stencil = stencil;
  const auto n = autan::kernels::counts(t);
  std::printf(
      "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\\n", n.owned_cells,
      n.face_owner_cells, n.face_neighbour_cells, n.face_cells, n.owned_interior_faces,
      n.boundary_faces, n.owned_boundary_faces, n.boundary_cells, n.all_boundary_cells, n.faces,
      n.incidences, n.incidence_cells, n.incidence_touched_cells, n.stencil, n.stencil_rows,
      n.stencil_cells, n.stencil_touched_cells, n.stencil_owned_cells, n.stencil_boundary_faces);
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
def test_the_counts_of_a_partition_are_those_counted_by_hand(tmp_path: Path) -> None:
    unit = tmp_path / "counts.cpp"
    unit.write_text(TOPOLOGY)
    exe = tmp_path / "counts"
    subprocess.run(
        ["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'codegen' / 'include'}",
         str(unit), "-o", str(exe)],
        check=True,
    )  # fmt: skip
    got = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    # Counted from the comment's mesh: owners {0, 1} and neighbours {1, 2} of the interior faces;
    # boundary owners {0} owned, {0, 2} in all; the owned cells' faces {0, 1, 2}; across them
    # {1} from 0 and {0, 2} from 1; stencil rows {0, 1}, its cells {0, 1, 2}, its face {2}.
    expected = {
        "owned_cells": 2,
        "face_owner_cells": 2,
        "face_neighbour_cells": 2,
        "face_cells": 3,
        "owned_interior_faces": 2,
        "boundary_faces": 2,
        "owned_boundary_faces": 1,
        "boundary_cells": 1,
        "all_boundary_cells": 2,
        "faces": 3,
        "incidences": 4,
        "incidence_cells": 3,
        "incidence_touched_cells": 3,
        "stencil": 4,
        "stencil_rows": 2,
        "stencil_cells": 3,
        "stencil_touched_cells": 3,
        "stencil_owned_cells": 3,
        "stencil_boundary_faces": 1,
    }
    assert dict(zip(traffic.COUNTS, map(float, got), strict=True)) == expected


def test_a_component_the_body_reads_more_or_less_changes_the_count_by_its_bytes(
    tmp_path: Path,
) -> None:
    narrow = _module(tmp_path, MODULE)
    (tmp_path / "wider").mkdir()
    wide = _module(tmp_path / "wider", WIDER)
    for policy in ("fp64", "fp32+acc64"):
        added = _bytes(wide, "partial", policy) - _bytes(narrow, "partial", policy)
        assert added == PRIMES["face_neighbour_cells"] * ROLE_BYTES[policy]["state"]


def test_derived_reductions_write_only_their_written_accumulators_and_read_no_twins(
    tmp_path: Path,
) -> None:
    m = _module(tmp_path, MODULE)
    p = PRIMES
    s = 8
    # The primal reads sc across the entries, ub on the boundary faces and the maps, and writes
    # top. The tangent recomputes the Max without writing it (write=False), reads sc_tan across
    # the entries and writes the tangent the Max selects (Follow).
    maps = (
        (p["owned_cells"] + 1) * INDEX["offset_t"]
        + p["incidences"] * INDEX["index_t"]
        + (p["faces"] - p["owned_boundary_faces"]) * 2 * INDEX["index_t"]
    )
    primal = p["incidence_cells"] * s + p["owned_boundary_faces"] * s + maps
    assert _bytes(m, "largest", "fp64") == primal + p["owned_cells"] * s
    tangent = primal + p["incidence_cells"] * s + p["owned_cells"] * s
    assert _bytes(m, "largest_tangent", "fp64") == tangent
    # The adjoint reads the primal's data once, reads and zeroes the seed (top_adj) and
    # increments sc_adj across the entries; the deterministic strategy's twins are not counted.
    adjoint = primal + p["owned_cells"] * 2 * s + p["incidence_cells"] * 2 * s
    assert _bytes(m, "largest_adjoint", "fp64") == adjoint
    lm = next(lm for lm in m.loops if lm.name == "largest_adjoint")
    assert all("twin" not in t.size for t in traffic.traffic(lm))


MAIN = """
#include <cstdio>
#include "{module}.hpp"

template <class P>
void print(const char* policy, const autan::kernels::Counts& n) {{
  using autan::kernels::{module}::Data;
{sizes}
  for (const auto& [name, bytes] : autan::kernels::{module}::traffic_table<P>) {{
    std::printf("loop %s %s %.17g\\n", policy, name, bytes(n));
  }}
}}

int main() {{
  for (int layout = 0; layout < 2; ++layout) {{
    autan::kernels::Counts n{{{primes}, layout == 1, layout == 1, layout == 1}};
    std::printf("layout %d\\n", layout);
    print<autan::kernels::fp64>("fp64", n);
    print<autan::kernels::fp32_acc64>("fp32+acc64", n);
  }}
}}
"""

MODULES = sorted((ROOT / "kernels").glob("*.py")) + sorted(
    (ROOT / "codegen" / "tests" / "kernels").glob("*.py")
)


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("src", MODULES, ids=lambda p: p.stem)
def test_every_generated_loop_has_the_bytes_the_derivation_gives(tmp_path: Path, src: Path) -> None:
    out = tmp_path / "gen"
    generate(src, out, ["cpu"])
    m = model.load(src)
    derive(m)
    sizes = "\n".join(
        f'  std::printf("size %s {d.name} %.17g\\n", policy, '
        f"autan::kernels::element_size<decltype(Data<P>::{d.name})>());"
        for d in m.datasets
    )
    unit = tmp_path / "main.cpp"
    unit.write_text(
        MAIN.format(
            module=m.name, sizes=sizes, primes=", ".join(str(PRIMES[f]) for f in traffic.COUNTS)
        )
    )
    exe = tmp_path / "main"
    subprocess.run(
        ["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'codegen' / 'include'}",
         f"-I{out}", str(unit), "-o", str(exe)],
        check=True,
    )  # fmt: skip
    lines = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    loops = {lm.name: lm for lm in m.loops}
    seen: dict[tuple[int, str], set[str]] = {}
    sizes_of: dict[str, dict[str, int]] = {}
    layout = 0
    for line in lines.splitlines():
        kind, *rest = line.split()
        if kind == "layout":
            layout = int(rest[0])
            sizes_of = {}
        elif kind == "size":
            policy, name, value = rest
            sizes_of.setdefault(policy, dict(INDEX))[name] = int(float(value))
        else:
            policy, name, value = rest
            flag = layout == 1
            counts = {
                **PRIMES,
                "stencil_sliced": flag,
                "incidence_sliced": flag,
                "incidence_cell": flag,
            }
            expected = traffic.evaluate(traffic.traffic(loops[name]), counts, sizes_of[policy])
            assert float(value) == pytest.approx(expected, rel=1e-15), (policy, name, layout)
            seen.setdefault((layout, policy), set()).add(name)
    for key, names in seen.items():
        assert names == set(loops), key
    assert len(seen) == 4
