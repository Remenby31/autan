#pragma once

#include <filesystem>
#include <string>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Mesh hash recorded by every run, as 64 lowercase hex characters: SHA-256
/// of the file bytes, or for a polyMesh directory of the concatenation, over
/// points, faces, owner, neighbour, boundary and cellZones (when present), of
/// "<name>\n<byte length>\n<bytes>" (docs/contracts/mesh-info.md).
std::string mesh_hash(const std::filesystem::path& path);

/// MeshInfo document (docs/contracts/mesh-info.md, schema autan.mesh-info/1)
/// of an assembled, unpartitioned mesh read from its source, as JSON text.
std::string mesh_info_json(const Mesh& mesh, const std::string& sha256);

}  // namespace autan::io
