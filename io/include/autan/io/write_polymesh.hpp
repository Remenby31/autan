#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Writes `mesh` (unpartitioned, without periodic pairs) as an OpenFOAM polyMesh directory that
/// read_polymesh reads back to the same mesh: binary points, faces (faceCompactList), owner,
/// neighbour and cellZones, 32-bit labels and 64-bit scalars, and the boundary patches in zone
/// order, typed `wall` for a wall zone, `symmetry` for a symmetry zone and `patch` otherwise.
/// A zone name is written as a word of letters, digits and `_ - . : +`: each run of other
/// characters becomes `_`, a leading digit gets a `_` before it, and `interior` (the zone the
/// reader gives the internal faces) becomes `interior_`; the renamed zones are returned, old name
/// to new. Creates `directory`. Throws E0130 when a file cannot be
/// written, E0113 when a count exceeds 2^31 - 1, E0110 when two zones of a kind get one name and
/// E0133 for a partitioned mesh or one with periodic pairs.
std::map<std::string, std::string> write_polymesh(const std::filesystem::path& directory,
                                                  const Mesh& mesh);

}  // namespace autan::io
