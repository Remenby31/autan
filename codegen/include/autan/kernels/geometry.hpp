#pragma once

#include <cstdint>
#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"
#include "autan/kernels/runtime.hpp"

namespace autan::kernels {

/// FP64 geometry in the layout of the generated `geometry.*` datasets: structure of arrays,
/// component c of entity i at c * n + i, matrices row-major.
struct PackedGeometry {
  std::vector<double> face_area;      ///< 3 x n_faces, owner -> neighbour
  std::vector<double> face_centroid;  ///< 3 x n_faces
  std::vector<double> face_normal;    ///< 3 x n_faces, unit
  std::vector<double> face_area_magnitude;
  std::vector<double> face_r_left;  ///< 3 x n_faces: face centroid - owner centroid
  std::vector<double>
      face_r_right;  ///< 3 x n_faces: face centroid - neighbour centroid, 0 on boundary
  /// 9 x n_faces: sum over the face's fan triangles t of (c_t - x_f) (x) S_t. Zero on a planar
  /// face; with it the flux of a linear field u through any face is u(x_f) S_f + M_f^T grad(u).
  std::vector<double> face_moment;
  std::vector<double> cell_centroid;  ///< 3 x n_cells
  std::vector<double> cell_volume;    ///< n_cells
  /// 3 x n_incidences: weighted least-squares gradient weights w_k = M^-1 d_k / |d_k|^2 with
  /// M = sum_k d_k d_k^T / |d_k|^2, d_k from the cell centroid to the neighbour centroid
  /// (interior face) or the face centroid (boundary face). Zero on ghost cells.
  std::vector<double> lsq_weight;
  /// 3 x stencil size: LSQ weights over the cell stencil, when one is set (CellStencil).
  std::vector<double> stencil_lsq_weight;
};

PackedGeometry pack_geometry(const Mesh& mesh, const Geometry& geometry);

/// Topology of `mesh`; `incidence_twin` stays null (set it from incidence_twins()).
[[nodiscard]] Topology topology(const MeshView& mesh) noexcept;

/// For each incidence k (entry of the cell -> face CSR) of an interior face f, the incidence of f
/// in the other cell's list; -1 for boundary faces.
std::vector<offset_t> incidence_twins(const Mesh& mesh);

/// Cell stencils for least-squares gradients wider than the face neighbours, which make the
/// unlimited second-order scheme linearly stable on tetrahedra (Diskin and Thomas, AIAA J.
/// 49(4), 2011): `vertex` holds the cells sharing a node with the cell and the boundary faces
/// sharing a node with it; `face2` the face neighbours and their face neighbours, and the
/// boundary faces of the cell and of its face neighbours.
enum class StencilKind : std::uint8_t { vertex, face2 };

/// A cell -> entries CSR (layout of Topology::stencil) with its twins and the LSQ weights over
/// it (3 x size, structure of arrays, as PackedGeometry::lsq_weight). Entries of a cell are its
/// stencil cells in increasing order, then its boundary faces in increasing order. Owned cells
/// only; ghost cells get empty lists (a wider halo is the partitioner's job).
struct CellStencil {
  std::vector<offset_t> offsets;
  std::vector<index_t> entries;
  std::vector<offset_t> twin;
  std::vector<double> lsq_weight;
  /// 3 x stencil size: LSQ weights over the cell stencil, when one is set (CellStencil).
  std::vector<double> stencil_lsq_weight;
};

CellStencil cell_stencil(const Mesh& mesh, const PackedGeometry& geometry, StencilKind kind);

/// Copy of `v` converted to T (the storage type of a precision role).
template <class T>
std::vector<T> converted(const std::vector<double>& v) {
  return std::vector<T>(v.begin(), v.end());
}

}  // namespace autan::kernels
