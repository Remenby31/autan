#pragma once

#include <array>
#include <string>

#include "autan/core/cellwise.hpp"

namespace autan::test {

enum class BoxFill { hexa, tetra, prism, pyramid, hybrid };

std::string to_string(BoxFill fill);

/// Unit cube [0,1]^3 cut into n^3 cubes, each filled with `fill`:
/// hexa (1 hex), tetra (6 Kuhn tetrahedra), prism (2 prisms along z),
/// pyramid (6 pyramids on a centre node), hybrid (hexes and pyramid-split
/// cubes in one corner block, z-prisms, x-prisms and tetrahedra in the
/// others, all conforming). The six sides are labelled zones xmin, xmax,
/// ymin, ymax, zmin, zmax (in that order) unless `label_sides` is false.
/// With `perturbation` > 0, interior grid nodes move by up to that fraction
/// of the spacing (deterministic), which warps the quadrilateral faces.
CellwiseMesh box(int n, BoxFill fill, double perturbation = 0.0, bool label_sides = true);

/// Box [0,2]x[0,1]x[0,1]: the unit cube [0,1]^3 is one polyhedron facing a
/// 10x10 layer of hexes across x = 1, so it has 105 faces, four of them
/// polygons with 9 hanging nodes each. Its x = 0 face is labelled "inlet".
CellwiseMesh hanging_node_box();

/// Single cell of the given shape: the reference cell of known volume and
/// centroid (unit tetra, unit cube, unit prism, pyramid on the unit square
/// with apex (1/2, 1/2, 1)) mapped by x -> A x + b.
struct ReferenceCell {
  CellwiseMesh mesh;
  double volume;
  std::array<double, 3> centroid;
};
ReferenceCell affine_cell(CellShape shape, const std::array<std::array<double, 3>, 3>& a,
                          const std::array<double, 3>& b);

}  // namespace autan::test
