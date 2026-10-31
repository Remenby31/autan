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
  /// Per face: |s| / max(k |d|, extent of the face along s), where s goes
  /// from the point the line C_P + t d crosses the face plane to c_f, with
  /// d = C_N - C_P and k = 0.2 (interior), or d the normal projection of
  /// c_f - C_P and k = 0.4 (boundary). Infinite when d · S_f <= 0.
  std::vector<double> skewness;
  /// Per interior face: min(V_P, V_N) / max(V_P, V_N), in (0, 1].
  std::vector<double> volume_ratio;
  /// Per owned cell: max(a_max / a_min, sum_k a_k / (6 V^(2/3))) with
  /// a_k = sum_f |S_f,k|, so 1 for a cube and the edge ratio for a box.
  std::vector<double> aspect_ratio;
};

MeshQuality compute_quality(const Mesh& mesh, const Geometry& geometry);

}  // namespace autan
