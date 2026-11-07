#pragma once

#include <cstdint>

#include "autan/core/mesh.hpp"

namespace autan {

/// Tolerances of the merge of boundary-layer tetrahedra into prisms (#337).
struct PrismOptions {
  /// Most angle, in degrees, between the normals of the two triangles of a side quadrilateral.
  /// The side must also be convex: walked from its wall-side edge, a convex side turns against its
  /// outward normal at every corner, and a side passes when its four corners turn the same way, a
  /// straight corner counting as turning against the normal (not_planar otherwise).
  double planarity_deg = 5.0;
  /// Most height of a prism (mean pillar length) per shortest edge of its wall-side cap.
  double aspect = 1.0;
  /// Most angle, in degrees, between a pillar and the normal of the wall-side cap: a set of
  /// tetrahedra leaning into a neighbouring column can fill an oblique prism, whose pillars
  /// cross a cap edge sideways (at least atan(1 / aspect) from the normal).
  double pillar_deg = 30.0;
};

/// What the merge did: the cells of the result, and why front triangles were left.
struct PrismReport {
  std::int64_t prisms = 0;       ///< merged cells: 8-face polyhedra, the union of 3 tetrahedra
  std::int64_t tetrahedra = 0;   ///< tetrahedra left
  std::int64_t other_cells = 0;  ///< cells neither tetrahedra nor merged
  std::int64_t layers = 0;       ///< most prisms stacked on one wall triangle
  /// Quality before and after (compute_quality): the largest non-orthogonality, in degrees, and
  /// the least volume ratio across an interior face.
  double max_non_orthogonality_in = 0.0, max_non_orthogonality_out = 0.0;
  double min_volume_ratio_in = 1.0, min_volume_ratio_out = 1.0;
  // Front triangles (a wall triangle or a merged prism's top cap) where a column stopped. When
  // several sets of three fill a prism there and none is valid, the cause counted is that of the
  // set that passed the most checks, taken in the order across_zones, not_planar, not_aligned,
  // not_layer, invalid.
  std::int64_t no_candidate = 0;     ///< no three tetrahedra form a prism on it
  std::int64_t ambiguous = 0;        ///< several sets of three do
  std::int64_t not_planar = 0;       ///< a side quadrilateral beyond the planarity tolerance
  std::int64_t not_aligned = 0;      ///< a pillar beyond the pillar tolerance
  std::int64_t not_layer = 0;        ///< a prism higher than `aspect` times its cap
  std::int64_t invalid = 0;          ///< a prism with a corner of non-positive volume
  std::int64_t across_zones = 0;     ///< tetrahedra of different cell zones
  std::int64_t not_tetrahedron = 0;  ///< the cell across it is not an unmerged tetrahedron
};

struct PrismMerge {
  Mesh mesh;
  PrismReport report;
};

/// Merges, in the boundary layer of `mesh` (unpartitioned, without periodic pairs), each set of
/// three tetrahedra that fills a triangular prism into that prism, advancing in layers from the
/// faces of the boundary zones of kind wall: the inverse of the advancing-layers construction
/// (Pirzadeh, AIAA J. 34(1), 1996), three tetrahedra recombined by pattern (Meshkat and Talmor,
/// IJNME 49, 2000). A front triangle abc takes the three face-connected tetrahedra not yet merged,
/// the first across abc, whose outer faces are exactly abc, a cap a'b'c' and two triangles
/// covering each side quadrilateral u v v' u' (aa', bb', cc' the pillars), when exactly one such
/// set exists and it is a valid layer cell within `options`; its top cap a'b'c' is then a front
/// triangle.
/// The faces inside a prism go and every other face is kept, each side as its two triangles, so
/// that a merged cell is exactly its tetrahedra's union. Nodes, boundary faces and zones are those
/// of `mesh`. Throws E0133 for a mesh with ghost cells or periodic pairs, and std::logic_error when
/// the result breaks an invariant: other nodes or boundary faces, or a merged cell whose volume
/// differs from its tetrahedra's by more than 1e-12 relative.
PrismMerge merge_layer_prisms(const Mesh& mesh, const PrismOptions& options = {});

}  // namespace autan
