#pragma once

#include <filesystem>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Writes `mesh` (unpartitioned, without periodic pairs) as an OpenFOAM polyMesh directory that
/// read_polymesh reads back to the same mesh: binary points, faces (faceCompactList), owner,
/// neighbour and cellZones, 32-bit labels and 64-bit scalars, and the boundary patches in zone
/// order, typed `wall` for a wall zone, `symmetryPlane` for a symmetry zone and `patch`
/// otherwise. Creates `directory`. Throws E0130 when a file cannot be written, E0113 when a count
/// exceeds 2^31 - 1 and E0133 for a partitioned mesh or one with periodic pairs.
void write_polymesh(const std::filesystem::path& directory, const Mesh& mesh);

}  // namespace autan::io
