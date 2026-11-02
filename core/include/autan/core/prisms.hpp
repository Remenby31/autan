#pragma once

#include <cstdint>

#include "autan/core/mesh.hpp"

namespace autan {

/// Tolerances of the merge of boundary-layer tetrahedra into prisms (#337).
struct PrismOptions {
  /// Most angle, in degrees, between the normals of the two triangles of a side quadrilateral.
  double planarity_deg = 5.0;
  /// Most height of a prism (mean pillar length) per shortest edge of its wall-side cap.
  double aspect = 1.0;
};

/// What the merge did: the cells of the result, and why front triangles were left.
struct PrismReport {
  std::int64_t prisms = 0;        ///< merged cells whose sides are quadrilaterals
  std::int64_t split_prisms = 0;  ///< merged cells keeping a side as two triangles
  std::int64_t tetrahedra = 0;    ///< tetrahedra left
  std::int64_t other_cells = 0;   ///< cells neither tetrahedra nor merged
  std::int64_t layers = 0;        ///< most prisms stacked on one wall triangle
  // Front triangles (a wall triangle or a merged prism's top cap) where a column stopped.
  std::int64_t no_candidate = 0;     ///< no three tetrahedra form a prism on it
  std::int64_t ambiguous = 0;        ///< several sets of three do
  std::int64_t not_planar = 0;       ///< a side quadrilateral beyond the planarity tolerance
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
/// IJNME 49, 2000). A front triangle abc takes the three face-connected tetrahedra, the first
/// across abc, whose outer faces are exactly abc, a cap a'b'c' and two triangles covering each
/// side quadrilateral u v v' u' (aa', bb', cc' the pillars), when exactly one such set exists
/// and it is a valid layer cell within `options`; its top cap a'b'c' is then a front triangle.
/// A side quadrilateral both of whose triangles face one merged cell becomes one face; every
/// other face, the boundary's included, is kept. Nodes, boundary faces and zones are those of
/// `mesh`. Throws E0133 for a mesh with ghost cells or periodic pairs, and std::logic_error when
/// the result breaks an invariant: other nodes or boundary faces, or a merged cell whose volume
/// differs from its tetrahedra's by more than 1e-12 relative.
PrismMerge merge_layer_prisms(const Mesh& mesh, const PrismOptions& options = {});

}  // namespace autan
