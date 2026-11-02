#pragma once

#include <filesystem>
#include <map>
#include <string>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Writes `mesh` (unpartitioned, without periodic pairs) as an OpenFOAM polyMesh directory that
/// read_polymesh reads back to the same mesh: binary points, faces (faceCompactList), owner,
/// neighbour and cellZones, 32-bit labels and 64-bit scalars, and the boundary patches in zone
/// order, typed `wall` for a wall zone, `symmetryPlane` for a symmetry zone and `patch`
/// otherwise. A zone name that is not an OpenFOAM word (no whitespace, quotes, `/`, `;`, `{`,
/// `}`, `(`, `)`) is written with each run of those characters replaced by `_`; the renamed zones
/// are returned, old name to new. Creates `directory`. Throws E0130 when a file cannot be
/// written, E0113 when a count exceeds 2^31 - 1, E0110 when two zones of a kind get one name and
/// E0133 for a partitioned mesh or one with periodic pairs.
std::map<std::string, std::string> write_polymesh(const std::filesystem::path& directory,
                                                  const Mesh& mesh);

}  // namespace autan::io
