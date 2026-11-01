"""Writes the Gmsh test meshes of tests/data/mesh with the Gmsh API.

Each mesh fills the unit cube; its six sides are physical surfaces named
xmin, xmax, ymin, ymax, zmin, zmax. Gmsh serialises every file, so the
readers are tested against an independent writer. Run with
`uv run python generate.py OUTPUT_DIR`; convert.sh then derives the
OpenFOAM and Fluent versions with OpenFOAM's own converters.
"""

import itertools
import sys
from pathlib import Path

import gmsh

TET, HEX, PRISM, PYRAMID, TRI, QUAD = 4, 5, 6, 7, 2, 3
SIDES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]

# Outward faces of the Gmsh reference hexahedron.
HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
# Element faces used to find boundary faces (orientation irrelevant there).
ELEMENT_FACES = {
    TET: [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    PYRAMID: [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
    PRISM: [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (0, 3, 5, 2)],
    HEX: HEX_FACES,
}


def structured(n, fill):
    """Returns (coords, cells) with cells as (gmsh type, nodes, volume tag)."""
    m = n + 1
    coords = [(i / n, j / n, k / n) for k in range(m) for j in range(m) for i in range(m)]
    half = n // 2
    cells = []
    for k, j, i in itertools.product(range(n), repeat=3):

        def v(bx, by, bz, i=i, j=j, k=k):
            return (i + bx) + m * ((j + by) + m * (k + bz))

        hexa = [
            v(0, 0, 0),
            v(1, 0, 0),
            v(1, 1, 0),
            v(0, 1, 0),
            v(0, 0, 1),
            v(1, 0, 1),
            v(1, 1, 1),
            v(0, 1, 1),
        ]
        local, volume = fill, 1
        if fill == "hybrid":
            volume = 1 if k < half else 2
            if i < half and k < half:
                local = "hex" if (i + j + k) % 2 == 0 else "pyramid"
            elif k < half:
                local = "prism"
            elif i >= half:
                local = "tet"
            else:
                local = "xprism"
        if local == "hex":
            cells.append((HEX, hexa, volume))
        elif local == "tet":
            # Kuhn subdivision; odd axis orders are mirrored.
            for parity, order in [
                (0, (0, 1, 2)),
                (0, (1, 2, 0)),
                (0, (2, 0, 1)),
                (1, (0, 2, 1)),
                (1, (2, 1, 0)),
                (1, (1, 0, 2)),
            ]:
                b = [0, 0, 0]
                tet = [v(0, 0, 0)]
                for axis in order:
                    b[axis] = 1
                    tet.append(v(*b))
                if parity:
                    tet[1], tet[2] = tet[2], tet[1]
                cells.append((TET, tet, volume))
        elif local == "prism":
            cells.append(
                (
                    PRISM,
                    [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 0, 1), v(1, 0, 1), v(1, 1, 1)],
                    volume,
                )
            )
            cells.append(
                (
                    PRISM,
                    [v(0, 0, 0), v(1, 1, 0), v(0, 1, 0), v(0, 0, 1), v(1, 1, 1), v(0, 1, 1)],
                    volume,
                )
            )
        elif local == "xprism":
            cells.append(
                (
                    PRISM,
                    [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(1, 0, 0), v(1, 1, 0), v(1, 1, 1)],
                    volume,
                )
            )
            cells.append(
                (
                    PRISM,
                    [v(0, 0, 0), v(0, 1, 1), v(0, 0, 1), v(1, 0, 0), v(1, 1, 1), v(1, 0, 1)],
                    volume,
                )
            )
        elif local == "pyramid":
            centre = len(coords)
            coords.append(tuple(sum(coords[p][a] for p in hexa) / 8 for a in range(3)))
            for face in HEX_FACES:
                base = [hexa[f] for f in reversed(face)]
                cells.append((PYRAMID, [*base, centre], volume))
    return coords, cells


def side_faces(coords, cells):
    """Boundary faces of the cells, grouped by box side."""
    count = {}
    for etype, nodes, _ in cells:
        for face in ELEMENT_FACES[etype]:
            key = tuple(sorted(nodes[f] for f in face))
            count.setdefault(key, [0, [nodes[f] for f in face]])[0] += 1
    sides = {s: [] for s in range(6)}
    for n, face in count.values():
        if n != 1:
            continue
        for s in range(6):
            value = 0.0 if s % 2 == 0 else 1.0
            if all(coords[p][s // 2] == value for p in face):
                sides[s].append(face)
    return sides


def write_structured(name, n, fill, out):
    coords, cells = structured(n, fill)
    gmsh.model.add(name)
    volumes = sorted({c[2] for c in cells})
    for vol in volumes:
        gmsh.model.addDiscreteEntity(3, vol)
    for s in range(6):
        gmsh.model.addDiscreteEntity(2, s + 1)
    gmsh.model.mesh.addNodes(
        3, volumes[0], list(range(1, len(coords) + 1)), [x for p in coords for x in p]
    )
    tag = 1
    for vol in volumes:
        for etype in (TET, PYRAMID, PRISM, HEX):
            group = [c[1] for c in cells if c[0] == etype and c[2] == vol]
            if group:
                gmsh.model.mesh.addElementsByType(
                    vol,
                    etype,
                    list(range(tag, tag + len(group))),
                    [p + 1 for nodes in group for p in nodes],
                )
                tag += len(group)
    for s, faces in side_faces(coords, cells).items():
        for size, etype in ((3, TRI), (4, QUAD)):
            group = [f for f in faces if len(f) == size]
            if group:
                gmsh.model.mesh.addElementsByType(
                    s + 1,
                    etype,
                    list(range(tag, tag + len(group))),
                    [p + 1 for f in group for p in f],
                )
                tag += len(group)
    names = {1: "fluid"} if len(volumes) == 1 else {1: "lower", 2: "upper"}
    for vol in volumes:
        gmsh.model.addPhysicalGroup(3, [vol], 100 + vol, names[vol])
    for s, side in enumerate(SIDES):
        gmsh.model.addPhysicalGroup(2, [s + 1], s + 1, side)
    save(name, out)


def write_unstructured_tet(name, out):
    gmsh.model.add(name)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.addPhysicalGroup(3, [1], 101, "fluid")
    surfaces = {}
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        for axis in range(3):
            if abs(box[axis] - box[axis + 3]) < 1e-6:
                surfaces[2 * axis + round(box[axis])] = tag
    for s, side in enumerate(SIDES):
        gmsh.model.addPhysicalGroup(2, [surfaces[s]], s + 1, side)
    save(name, out)


def write_periodic_box(name, out):
    """Unstructured tetrahedra, periodic in x and y: Gmsh meshes xmax and ymax
    as translates of xmin and ymin and records the links in $Periodic."""
    gmsh.model.add(name)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfaces = {}
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        for axis in range(3):
            if abs(box[axis] - box[axis + 3]) < 1e-6:
                surfaces[2 * axis + round(box[axis])] = tag
    for axis in (0, 1):
        affine = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        affine[4 * axis + 3] = 1
        gmsh.model.mesh.setPeriodic(2, [surfaces[2 * axis + 1]], [surfaces[2 * axis]], affine)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.addPhysicalGroup(3, [1], 101, "fluid")
    for s, side in enumerate(SIDES):
        gmsh.model.addPhysicalGroup(2, [surfaces[s]], s + 1, side)
    save(name, out)


def save(name, out):
    for suffix, version, binary in (("ascii", 4.1, 0), ("binary", 4.1, 1), ("v22", 2.2, 0)):
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.option.setNumber("Mesh.Binary", binary)
        gmsh.write(str(out / f"{name}.{suffix}.msh"))
    gmsh.write(str(out / f"{name}.cgns"))
    gmsh.model.remove()


FOAM_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}

"""


def write_hanging_node_polymesh(out):
    """Box [0,2]x[0,1]x[0,1]: the cube [0,1]^3 is one 105-face polyhedron
    facing a 10x10 layer of hexes across x = 1; four of its faces carry 9
    collinear hanging nodes. Written as an OpenFOAM polyMesh, which
    checkMesh validates before conversion to Fluent."""
    n = 10
    points = [(0.0, y, z) for y in (0, 1) for z in (0, 1)]  # q[y][z] = 2 y + z
    q = lambda y, z: 2 * y + z
    base = len(points)
    for plane in (0, 1):
        points += [(1.0 + plane, j / n, k / n) for k in range(n + 1) for j in range(n + 1)]
    g = lambda plane, j, k: base + plane * (n + 1) ** 2 + j + (n + 1) * k

    big = [[q(0, 0), q(0, 1), q(1, 1), q(1, 0)]]
    big += [
        [g(0, j, k), g(0, j + 1, k), g(0, j + 1, k + 1), g(0, j, k + 1)]
        for k in range(n)
        for j in range(n)
    ]
    big.append([q(0, 0)] + [g(0, 0, k) for k in range(n + 1)] + [q(0, 1)])
    big.append([q(1, 0), q(1, 1)] + [g(0, n, k) for k in range(n, -1, -1)])
    big.append([q(0, 0), q(1, 0)] + [g(0, j, 0) for j in range(n, -1, -1)])
    big.append([q(0, 1)] + [g(0, j, n) for j in range(n + 1)] + [q(1, 1)])
    cells = [big]
    for k in range(n):
        for j in range(n):
            hexa = [
                g(0, j, k),
                g(1, j, k),
                g(1, j + 1, k),
                g(0, j + 1, k),
                g(0, j, k + 1),
                g(1, j, k + 1),
                g(1, j + 1, k + 1),
                g(0, j + 1, k + 1),
            ]
            cells.append([[hexa[f] for f in face] for face in HEX_FACES])

    faces = {}
    for c, cell in enumerate(cells):
        for face in cell:
            faces.setdefault(tuple(sorted(face)), []).append((c, face))
    interior, boundary = [], {"inlet": [], "outlet": [], "walls": []}
    for sides in faces.values():
        if len(sides) == 2:
            (c0, f0), (c1, _) = sorted(sides)
            interior.append((c0, c1, f0))
        else:
            ((c0, f0),) = sides
            xs = {points[p][0] for p in f0}
            patch = "inlet" if xs == {0.0} else "outlet" if xs == {2.0} else "walls"
            boundary[patch].append((c0, f0))
    interior.sort(key=lambda t: (t[0], t[1]))
    ordered = [f for _, _, f in interior]
    owner = [c0 for c0, _, _ in interior]
    neighbour = [c1 for _, c1, _ in interior]
    patches = []
    for name, ptype in (("inlet", "patch"), ("outlet", "patch"), ("walls", "wall")):
        patches.append((name, ptype, len(boundary[name]), len(ordered)))
        for c0, f0 in sorted(boundary[name], key=lambda t: t[0]):
            ordered.append(f0)
            owner.append(c0)

    mesh = out / "hanging" / "constant" / "polyMesh"
    mesh.mkdir(parents=True, exist_ok=True)

    def write(obj, cls, body):
        (mesh / obj).write_text(FOAM_HEADER.format(cls=cls, obj=obj) + body)

    write(
        "points",
        "vectorField",
        f"{len(points)}\n(\n" + "".join(f"({x!r} {y!r} {z!r})\n" for x, y, z in points) + ")\n",
    )
    write(
        "faces",
        "faceList",
        f"{len(ordered)}\n(\n"
        + "".join(f"{len(f)}({' '.join(map(str, f))})\n" for f in ordered)
        + ")\n",
    )
    write("owner", "labelList", f"{len(owner)}\n(\n" + "".join(f"{c}\n" for c in owner) + ")\n")
    write(
        "neighbour",
        "labelList",
        f"{len(neighbour)}\n(\n" + "".join(f"{c}\n" for c in neighbour) + ")\n",
    )
    write(
        "boundary",
        "polyBoundaryMesh",
        f"{len(patches)}\n(\n"
        + "".join(
            f"    {name}\n    {{\n        type {ptype};\n        nFaces {count};\n        startFace {start};\n    }}\n"
            for name, ptype, count, start in patches
        )
        + ")\n",
    )
    zones = [("coarse", [0]), ("fine", list(range(1, len(cells))))]
    write(
        "cellZones",
        "regIOobject",
        f"{len(zones)}\n(\n"
        + "".join(
            f"{name}\n{{\n    type cellZone;\n    cellLabels List<label> {len(ids)}({' '.join(map(str, ids))});\n}}\n"
            for name, ids in zones
        )
        + ")\n",
    )


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    write_structured("hex", 3, "hex", out)
    write_structured("prism", 3, "prism", out)
    write_structured("pyramid", 2, "pyramid", out)
    write_structured("tet", 2, "tet", out)
    write_structured("hybrid", 4, "hybrid", out)
    write_unstructured_tet("tet-gmsh", out)
    write_periodic_box("periodic-box", out)
    write_hanging_node_polymesh(out)
    gmsh.finalize()


if __name__ == "__main__":
    main()
