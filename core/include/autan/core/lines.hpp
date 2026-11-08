#pragma once

#include <cstddef>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Lines of strong coupling of a mesh, CSR over its cells: each line holds at least two cells and
/// consecutive cells share a face. Cells in no line are left out.
struct Lines {
  std::vector<offset_t> offsets{0};
  std::vector<index_t> cells;
};

/// The lines a line-implicit smoother solves along (Mavriplis 1998, J. Comput. Phys. 145:141-165,
/// sec. 4). A face's coupling is A_f / |x_N - x_O|, summed over the faces two cells share; a cell
/// with at most two neighbours also counts each of its boundary faces in its weakest and strongest
/// couplings, as A_f / (2 |x_f - x_c|), the distance to its mirror cell. A line grows through the
/// strongest coupling that is at least `ratio` times the weakest coupling of both cells, in the
/// same cell zone. Seeds are the cells whose strongest coupling is at least `ratio` times their
/// weakest, those with a boundary face first, then by decreasing strongest / weakest. Each seed not
/// yet in a line grows from itself one way, then the other, and keeps the line if it holds two
/// cells or more; a cell belongs to the first line that reaches it, and lines come in the order of
/// their seeds. A line runs from the end of its second growth to the end of its first, reversed
/// when only its last cell has a boundary face, so a line touching the boundary at one end starts
/// there. Depends on the mesh, its geometry and its cell order (ties go to the smaller index).
/// Throws std::invalid_argument on a mesh with ghost cells.
Lines smoother_lines(const Mesh& mesh, const Geometry& geometry, double ratio = 4.0);

/// Lines of at least this many cells are coarsened along themselves by the agglomeration
/// multigrid; shorter lines are smoothed as lines but agglomerated like the other cells.
inline constexpr std::size_t long_line = 4;

/// The number of cells in lines of at least `min_length` cells: every cell of `lines` with the
/// default, the cells the multigrid coarsens along lines with `long_line`.
std::size_t cells_in_lines(const Lines& lines, std::size_t min_length = 2);

}  // namespace autan
