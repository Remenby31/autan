#pragma once

#include <vector>

#include "cfd/core/mesh.hpp"

namespace cfd {

/// FP64 geometry of a Mesh, structure of arrays indexed like the mesh.
/// Face area vectors point from owner to neighbour, outward on the boundary.
struct Geometry {
  std::vector<double> face_area_x, face_area_y, face_area_z;
  std::vector<double> face_area;  ///< magnitude of the area vector
  std::vector<double> face_centroid_x, face_centroid_y, face_centroid_z;
  std::vector<double> cell_volume;
  std::vector<double> cell_centroid_x, cell_centroid_y, cell_centroid_z;
};

/// Non-owning view of a Geometry for kernels (host or device pointers).
struct GeometryView {
  const double* face_area_x;
  const double* face_area_y;
  const double* face_area_z;
  const double* face_area;
  const double* face_centroid_x;
  const double* face_centroid_y;
  const double* face_centroid_z;
  const double* cell_volume;
  const double* cell_centroid_x;
  const double* cell_centroid_y;
  const double* cell_centroid_z;
};

/// Computes face and cell geometry in FP64. Each face is split into a fan of
/// triangles around the mean of its nodes; the cell is the polyhedron bounded
/// by these triangles, decomposed into tetrahedra (pyramids on each fan
/// triangle). Ghost cells get NaN volume and centroid: the partitioner fills
/// them from the owning partition. Throws E0120 for a face of zero or
/// non-finite area and E0121 for an owned cell of non-positive or non-finite
/// volume.
Geometry compute_geometry(const Mesh& mesh);

[[nodiscard]] GeometryView view(const Geometry& geometry) noexcept;

/// Closure error |sum_f S_f| / sum_f |S_f| of each owned cell, with outward
/// area vectors; zero for a closed cell up to round-off. `geometry` must come
/// from compute_geometry(mesh); a geometry whose sizes do not match throws
/// E0101.
std::vector<double> cell_closure_error(const Mesh& mesh, const Geometry& geometry);

}  // namespace cfd
