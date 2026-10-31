#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan {

/// Linear cell shapes with the node ordering of the Gmsh reference elements
/// (https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering).
enum class CellShape : std::uint8_t { tetra, pyramid, prism, hexa };

/// A mesh described cell by cell, as element-based formats store it: each cell
/// lists its own faces with outward normals, and shared faces appear once per
/// cell. Labelled faces carry the face zone of the cell face they coincide with.
struct CellwiseMesh {
  std::vector<double> x, y, z;
  std::vector<offset_t> cell_face_offsets{0};
  std::vector<offset_t> face_node_offsets{0};
  std::vector<index_t> face_nodes;
  std::vector<index_t> cell_zone;
  std::vector<offset_t> labelled_face_node_offsets{0};
  std::vector<index_t> labelled_face_nodes;
  std::vector<index_t> labelled_face_zone;
  std::vector<FaceZone> face_zones;
  std::vector<CellZone> cell_zones;
  MeshSource source;

  /// Appends a standard cell; `nodes` follows the Gmsh ordering of `shape`.
  void add_cell(CellShape shape, std::span<const index_t> nodes, index_t zone);
  /// Appends a polyhedron given as a local CSR of outward faces.
  void add_polyhedron(std::span<const offset_t> local_face_offsets,
                      std::span<const index_t> local_face_nodes, index_t zone);
  void add_labelled_face(std::span<const index_t> nodes, index_t zone);
};

/// Merges coincident cell faces (same node set) into face-based form. A face
/// seen from two cells must be listed with opposite orientations (E0114) and
/// by at most two cells (E0108); a labelled face must match exactly one cell
/// face (E0111) and be labelled once (E0115). A zone that labels faces takes
/// the role (interior or boundary) of those faces and may not mix them
/// (E0109); a zone that labels none keeps its declared role. Unlabelled interior faces go to
/// a synthesised zone "interior"; unlabelled boundary faces go to a
/// synthesised zone "unassigned", so that missing boundary conditions stay
/// visible downstream.
MeshInput to_face_based(CellwiseMesh mesh);

}  // namespace autan
