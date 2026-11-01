"""A NACA 0012 slab for UI screenshots: chord 1 along x, thickness along z, one cell layer of
span 0.1 along y, in a circular farfield of radius 20 chords. Two transfinite blocks mirrored
about the chord, triangulated and extruded into prisms, so the mesh is symmetric and gives no
lift at zero incidence. Not a validation grid.

uv run --locked --project tests/data/mesh/generate python ui/scripts/naca0012_mesh.py OUT.msh
"""

import math
import sys

import gmsh

SPAN = 0.1
RADIUS = 20.0
CELLS_ALONG_EACH_SIDE = 80
CELLS_TO_FARFIELD = 48
OUTWARD_SIZE_RATIO = 1.12


def thickness(x: float) -> float:
    # NACA 0012 with the closed trailing edge (-0.1036 x^4).
    return 0.6 * (
        0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )


def main(out: str) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo
    xs = [
        0.5 * (1 - math.cos(math.pi * k / CELLS_ALONG_EACH_SIDE))
        for k in range(CELLS_ALONG_EACH_SIDE + 1)
    ]
    leading, trailing = geo.addPoint(0, 0, 0), geo.addPoint(1, 0, 0)
    centre = geo.addPoint(0.5, 0, 0)
    west, east = geo.addPoint(0.5 - RADIUS, 0, 0), geo.addPoint(0.5 + RADIUS, 0, 0)
    to_west, to_east = geo.addLine(leading, west), geo.addLine(trailing, east)
    for line in (to_west, to_east):
        geo.mesh.setTransfiniteCurve(line, CELLS_TO_FARFIELD + 1, "Progression", OUTWARD_SIZE_RATIO)

    # Above (side 1) or below (side -1) the chord: each curve is oriented and divided as its
    # mirror image, which makes the transfinite triangulation mirrored too.
    def block(side: float) -> int:
        inner = [geo.addPoint(x, 0, side * thickness(x)) for x in xs[1:-1]]
        airfoil = geo.addSpline([leading, *inner, trailing])
        pole = geo.addPoint(0.5, 0, side * RADIUS)
        arcs = [geo.addCircleArc(west, centre, pole), geo.addCircleArc(pole, centre, east)]
        geo.mesh.setTransfiniteCurve(airfoil, CELLS_ALONG_EACH_SIDE + 1, "Bump", 0.2)
        for arc in arcs:
            geo.mesh.setTransfiniteCurve(arc, CELLS_ALONG_EACH_SIDE // 2 + 1)
        loop = geo.addCurveLoop([airfoil, to_east, -arcs[1], -arcs[0], -to_west])
        surface = geo.addPlaneSurface([loop])
        geo.mesh.setTransfiniteSurface(surface, cornerTags=[leading, trailing, east, west])
        return surface

    bases = [block(1.0), block(-1.0)]
    extruded = geo.extrude([(2, s) for s in bases], 0, SPAN, 0, numElements=[1], recombine=True)
    geo.synchronize()
    # extrude() lists, for each base surface: its top, its volume, then one side per curve of
    # its loop, in loop order (airfoil, to_east, arc, arc, to_west).
    tops, volumes, airfoil, farfield = [], [], [], []
    for k in range(len(bases)):
        top, volume, *sides = extruded[k * 7 : (k + 1) * 7]
        tops.append(top[1])
        volumes.append(volume[1])
        airfoil.append(sides[0][1])
        farfield += [sides[2][1], sides[3][1]]
    gmsh.model.addPhysicalGroup(2, farfield, name="farfield")
    gmsh.model.addPhysicalGroup(2, airfoil, name="airfoil")
    gmsh.model.addPhysicalGroup(2, bases + tops, name="symmetry")
    gmsh.model.addPhysicalGroup(3, volumes, name="fluid")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    gmsh.finalize()


if __name__ == "__main__":
    main(sys.argv[1])
