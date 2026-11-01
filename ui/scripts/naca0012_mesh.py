"""A NACA 0012 slab for UI screenshots: chord 1 along x, thickness along z, one cell layer of
span 0.1 along y, farfield box 20 chords wide. Not a validation grid.

uv run --locked --project tests/data/mesh/generate python ui/scripts/naca0012_mesh.py OUT.msh
"""

import math
import sys

import gmsh

SPAN = 0.1
POINTS = 80


def thickness(x: float) -> float:
    # NACA 0012 with the closed trailing edge (-0.1036 x^4).
    return 0.6 * (
        0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )


def main(out: str) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo
    xs = [0.5 * (1 - math.cos(math.pi * k / POINTS)) for k in range(POINTS + 1)]
    upper = [geo.addPoint(x, 0, thickness(x), 0.01) for x in reversed(xs)]
    lower = [geo.addPoint(x, 0, -thickness(x), 0.01) for x in xs[1:-1]]
    airfoil = geo.addSpline([*upper, *lower, upper[0]])
    corners = [geo.addPoint(x, 0, z, 1.5) for x, z in ((-10, -10), (11, -10), (11, 10), (-10, 10))]
    outer = [geo.addLine(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    surface = geo.addPlaneSurface([geo.addCurveLoop(outer), geo.addCurveLoop([airfoil])])
    extruded = geo.extrude([(2, surface)], 0, SPAN, 0, numElements=[1], recombine=True)
    geo.synchronize()
    top, volume = extruded[0][1], extruded[1][1]
    sides = [tag for dim, tag in extruded[2:] if dim == 2]
    gmsh.model.addPhysicalGroup(2, sides[:4], name="farfield")
    gmsh.model.addPhysicalGroup(2, sides[4:], name="airfoil")
    gmsh.model.addPhysicalGroup(2, [surface, top], name="symmetry")
    gmsh.model.addPhysicalGroup(3, [volume], name="fluid")
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    gmsh.finalize()


if __name__ == "__main__":
    main(sys.argv[1])
