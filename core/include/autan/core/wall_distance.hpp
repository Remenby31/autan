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
/// the given zones. On a mesh with periodic pairs it also holds the copies
/// shifted by -1, 0 or +1 times each periodic translation, so a nearest wall
/// across a periodic boundary is found. Throws E0102 for a zone out of range
/// or a zone that holds interior faces.
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
    std::int32_t first;  ///< leaf: first triangle; inner: left child (right = first + 1)
    std::int32_t count;  ///< leaf: triangle count; inner: 0
  };
  void build(std::int32_t self, std::int32_t begin, std::int32_t end);

  std::vector<std::array<double, 3>> a_, b_, c_;
  std::vector<Node> nodes_;
};

/// Distance from each cell centroid of `mesh` to the wall surface of
/// `wall_zones` (WallSurface), n_cells values; cells whose centroid is NaN
/// (ghosts before the geometry exchange) get NaN. On a partition, build the
/// WallSurface from the global mesh and query the local centroids instead:
/// the walls nearest to a part may lie in another.
std::vector<double> wall_distance(const Mesh& mesh, const Geometry& geometry,
                                  std::span<const index_t> wall_zones);

}  // namespace autan
