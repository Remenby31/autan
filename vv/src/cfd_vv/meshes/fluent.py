"""Fluent ASCII mesh writer (sections 0, 2, 10, 12, 13, 45), for meshes Gmsh MSH cannot store.

Conventions agreed with the C++ core reader: section indices and connectivity in hexadecimal,
1-based; the right-hand-rule normal of a face's node list points toward c0 and away from c1;
boundary faces have c1 = 0. Faces use the mixed type (0), each line starting with its node count.
Cells use element type 7 (polyhedron) or 0 (mixed) with per-cell types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from cfd_vv.meshes.facemesh import FaceMesh

# Fluent boundary-condition type codes (section 13 header) and names (section 45).
BC_CODES = {"interior": 2, "wall": 3, "pressure-far-field": 9}
CELL_TYPE_CODES = {"tet": 2, "hex": 4, "pyramid": 5, "prism": 6, "polyhedron": 7}


def write_fluent(mesh: FaceMesh, path: Path, cell_zone: str) -> None:
    """Write `mesh` as a 3D Fluent ASCII mesh with one cell zone and its face zones."""
    lines: list[str] = ['(0 "cfd-vv median-dual mesh")', "(2 3)"]
    npts = mesh.points.shape[0]
    lines.append(f"(10 (0 1 {npts:x} 0 3))")
    lines.append(f"(10 (1 1 {npts:x} 1 3)(")
    lines += [f"{x!r} {y!r} {z!r}" for x, y, z in mesh.points.tolist()]
    lines.append("))")

    cell_zone_id, interior_id = 2, 3
    lines.append(f"(12 (0 1 {mesh.n_cells:x} 0))")
    kinds = set(mesh.cell_kinds) or {"polyhedron"}
    if len(kinds) == 1:
        lines.append(f"(12 ({cell_zone_id:x} 1 {mesh.n_cells:x} 1 {CELL_TYPE_CODES[kinds.pop()]:x}))")
    else:
        lines.append(f"(12 ({cell_zone_id:x} 1 {mesh.n_cells:x} 1 0)(")
        lines += [" ".join(f"{CELL_TYPE_CODES[k]:x}" for k in mesh.cell_kinds[i : i + 16])
                  for i in range(0, mesh.n_cells, 16)]
        lines.append("))")

    lines.append(f"(13 (0 1 {mesh.n_faces:x} 0))")
    zone_names = [(cell_zone_id, "fluid", cell_zone), (interior_id, "interior", "interior")]
    ranges = [(interior_id, "interior", 0, mesh.n_interior)]
    ranges += [(4 + i, z.bc_type, z.start, z.count) for i, z in enumerate(mesh.zones)]
    zone_names += [(4 + i, z.bc_type, z.name) for i, z in enumerate(mesh.zones)]
    for zone_id, bc, start, count in ranges:
        lines.append(f"(13 ({zone_id:x} {start + 1:x} {start + count:x} {BC_CODES[bc]:x} 0)(")
        for f in range(start, start + count):
            # Our normals point owner -> neighbour (outward on the boundary); Fluent's point to c0.
            nodes = mesh.face(f)[::-1] + 1
            c0 = mesh.owner[f] + 1
            c1 = mesh.neighbour[f] + 1 if f < mesh.n_interior else 0
            lines.append(f"{nodes.size:x} " + " ".join(f"{v:x}" for v in nodes) + f" {c0:x} {c1:x}")
        lines.append("))")
    lines += [f"(45 ({zid:x} {kind} {name})())" for zid, kind, name in zone_names]
    path.write_text("\n".join(lines) + "\n")
