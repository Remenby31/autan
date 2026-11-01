#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Mesh hash recorded by every run, as 64 lowercase hex characters: SHA-256
/// of the file bytes, or for a polyMesh directory of the concatenation, over
/// points, faces, owner, neighbour, boundary and cellZones (when present), of
/// "<name>\n<byte length>\n<bytes>" (docs/contracts/mesh-info.md).
std::string mesh_hash(const std::filesystem::path& path);

/// Zones in MeshInfo order: boundary zones are the face zones with
/// `boundary` set, cell zones those holding at least one cell; indices into
/// Mesh::face_zones() and Mesh::cell_zones().
struct MeshInfoZones {
  std::vector<index_t> boundary;
  std::vector<index_t> cell;
};
MeshInfoZones meshinfo_zones(const Mesh& mesh);

/// MeshInfo document (docs/contracts/mesh-info.md, schema autan.mesh-info/1)
/// of an assembled, unpartitioned mesh read from its source, as JSON text.
std::string mesh_info_json(const Mesh& mesh, const std::string& sha256);

/// The boundary surface of an assembled, unpartitioned mesh as an
/// autan.surface/1 file (docs/contracts/mesh.md#boundary-surface): its
/// boundary faces by zone, in MeshInfo zone order, polygons as stored, nodes
/// in order of first use. Throws E0113 if a count exceeds 2^32 - 1.
std::string surface_bytes(const Mesh& mesh, const std::string& sha256);

}  // namespace autan::io
