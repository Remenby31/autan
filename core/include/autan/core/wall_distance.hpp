#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// The wall surface of a mesh: the fan triangles (around the mean of each
/// face's nodes, as compute_geometry splits faces) of the boundary faces of
/// the given zones. A face whose node mean lies outside it (strongly
/// non-convex) is represented by its fan, not by the polygon. On a mesh with
/// periodic pairs, the distance also covers the walls' periodic copies, as
/// far as the walls and cells span. Throws E0102 for a zone out of range or
/// holding interior faces, E0120 for a wall node that is not finite, and
/// E0118 for periodic translations that are not three independent
/// directions (translations on one line must be multiples of each other).
class WallSurface {
 public:
  WallSurface(const Mesh& mesh, std::span<const index_t> wall_zones);

  /// Distance from `point` to the nearest point of the surface (exact
  /// point-triangle projection, FP64); +infinity without wall faces, NaN for
  /// a point that is not finite.
  [[nodiscard]] double distance(const std::array<double, 3>& point) const;

  [[nodiscard]] std::size_t triangles() const noexcept { return a_.size(); }

 private:
  struct Node {
    std::array<double, 3> lo, hi;
    std::int64_t first;  ///< leaf: first triangle; inner: left child (right = first + 1)
    std::int64_t count;  ///< leaf: triangle count; inner: 0
  };
  void build(std::vector<std::int64_t>& order, std::int64_t self, std::int64_t begin,
             std::int64_t end);
  [[nodiscard]] double nearest2(const std::array<double, 3>& point, double best) const;

  std::vector<std::array<double, 3>> a_, b_, c_;
  std::vector<Node> nodes_;
  std::vector<std::array<double, 3>> directions_;
  std::vector<std::int64_t> reach_;
};

/// Distance from each cell centroid of `mesh` to the wall surface of
/// `wall_zones` (WallSurface), n_cells values; cells whose centroid is NaN
/// (ghosts before the geometry exchange) get NaN. Throws E0101 when
/// `geometry` does not have n_cells centroids. On a partition, build the
/// WallSurface from the global mesh and query the local centroids instead:
/// the walls nearest to a part may lie in another.
std::vector<double> wall_distance(const Mesh& mesh, const Geometry& geometry,
                                  std::span<const index_t> wall_zones);

}  // namespace autan
