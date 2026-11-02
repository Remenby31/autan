#pragma once

#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Lines of strong coupling of a mesh, CSR over its cells: each line holds at least two cells,
/// consecutive cells share a face, and a line touching the boundary at one end starts there.
/// Cells in no line are left out.
struct Lines {
  std::vector<offset_t> offsets{0};
  std::vector<index_t> cells;
};

/// The lines a line-implicit smoother solves along (Mavriplis 1998, J. Comput. Phys.
/// 145:141-165, sec. 4). A face's coupling is A_f / |x_N - x_O|, summed over the faces two cells
/// share; a line grows through the strongest coupling that is at least `ratio` times the weakest
/// coupling of both cells, in the same cell zone. Depends on the mesh, its geometry and its cell
/// order (ties go to the smaller index). Throws std::invalid_argument on a mesh with ghost cells.
Lines smoother_lines(const Mesh& mesh, const Geometry& geometry, double ratio = 4.0);

}  // namespace autan
