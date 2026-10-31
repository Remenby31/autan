#pragma once

#include <vector>

#include "autan/core/geometry.hpp"
#include "autan/core/mesh.hpp"

namespace autan {

/// Mesh quality metrics with the definitions of OpenFOAM's checkMesh
/// (primitiveMeshCheck), indexed like the mesh. Ghost cells must carry their
/// geometry (halo exchange) before faces that touch them are meaningful.
struct MeshQuality {
  /// Per face, degrees: angle between S_f and C_N - C_P (interior) or
  /// c_f - C_P (boundary).
  std::vector<double> non_orthogonality;
  /// Per face: distance from c_f to where the line C_P -> C_N crosses the
  /// face plane, divided by |C_N - C_P|; on the boundary the line is the
  /// face normal through C_P. Infinite when the line does not cross forward.
  std::vector<double> skewness;
  /// Per interior face: max(V_P, V_N) / min(V_P, V_N).
  std::vector<double> volume_ratio;
  /// Per owned cell: max(a_max / a_min, sum_k a_k / (6 V^(2/3))) with
  /// a_k = sum_f |S_f,k|, so 1 for a cube and the edge ratio for a box.
  std::vector<double> aspect_ratio;
};

MeshQuality compute_quality(const Mesh& mesh, const Geometry& geometry);

}  // namespace autan
