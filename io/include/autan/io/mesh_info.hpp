#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "autan/core/mesh.hpp"
#include "autan/io/cgns.hpp"

namespace autan::io {

/// Mesh hash recorded by every run, as 64 lowercase hex characters: SHA-256
/// of the file bytes, or for a polyMesh directory of the concatenation, over
/// points, faces, owner, neighbour, boundary and cellZones (when present), of
/// "<name>\n<byte length>\n<bytes>" (docs/contracts/mesh-info.md).
std::string mesh_hash(const std::filesystem::path& path);

/// The size in bytes of what mesh_hash covers: the file, or the sum of a polyMesh's files.
std::int64_t mesh_bytes(const std::filesystem::path& path);

/// What reading `path` costs in bytes (cgns_logical_bytes for CGNS); for every other format,
/// mesh_bytes and no descriptors (mesh-info.md, source).
LogicalBytes mesh_logical_bytes(const std::filesystem::path& path);

/// Zones in MeshInfo order: boundary zones are the face zones with
/// `boundary` set, cell zones those holding at least one cell; indices into
/// Mesh::face_zones() and Mesh::cell_zones().
struct MeshInfoZones {
  std::vector<index_t> boundary;
  std::vector<index_t> cell;
};
MeshInfoZones meshinfo_zones(const Mesh& mesh);

/// Groups of boundary zones from a mesh's sidecar (docs/contracts/mesh-info.md,
/// Zone groups): each name with its zones, sorted by name, zones in MeshInfo
/// order; `path` the sidecar's, symlinks of the mesh path followed, `sha256` of its bytes.
struct ZoneGroups {
  std::string path, sha256;
  std::vector<std::pair<std::string, std::string>> provenance;
  std::vector<std::pair<std::string, std::vector<std::string>>> groups;
};

/// The sidecar of the mesh at `mesh_path` (`<mesh file>.zones.json`, for a
/// polyMesh directory `<directory>.zones.json`, symlinks followed), checked against `mesh` and
/// its hash `sha256`; nullopt when there is none. Throws E0136 for a sidecar
/// that cannot be read or breaks autan.zone-groups/1, E0137 for another mesh's
/// hash, E0138 for an unknown zone, an empty group, a bad name or a zone twice
/// in a group, E0139 for a name given twice.
std::optional<ZoneGroups> read_zone_groups(const std::filesystem::path& mesh_path, const Mesh& mesh,
                                           const std::string& sha256);

/// What a mesh costs the solver that runs on it (docs/contracts/mesh-info.md):
/// its smoother's lines in a cell order, the stencil and periodic ghosts of a
/// single-rank run, its multigrid's levels in that cell order, and with `partition` those of
/// each rank of a run on several (autan::io::run_counts computes them).
struct MeshInfoRun {
  std::string line_order;   // "rcm" or "input"
  double line_scale = 1.0;  // metres per mesh unit the counts were computed with
  std::int64_t line_cells = 0;
  std::int64_t long_line_cells = 0;
  /// The multigrid's levels, finest (the mesh) first: their nodes and edges.
  std::vector<std::int64_t> multigrid_cells, multigrid_edges;
  std::int64_t stencil_entries = 0;
  std::int64_t periodic_ghosts = 0;
  struct Partition {
    std::int32_t ranks = 0;
    std::string groups;  // "none" or "lines"
    std::vector<std::int64_t> cells, ghosts, faces, interior_faces, stencil_entries;
    std::vector<std::int64_t> nodes, face_nodes, sent_cells;
  };
  std::optional<Partition> partition;
};

/// MeshInfo document (docs/contracts/mesh-info.md, schema autan.mesh-info/1)
/// of an assembled, unpartitioned mesh read from its source, as JSON text, with the zone
/// groups of its sidecar when it has one.
std::string mesh_info_json(const Mesh& mesh, const std::string& sha256, std::int64_t bytes,
                           const LogicalBytes& logical, const MeshInfoRun& run,
                           const std::optional<ZoneGroups>& groups = std::nullopt);

/// The boundary surface of an assembled, unpartitioned mesh as an
/// autan.surface/1 file (docs/contracts/mesh.md#boundary-surface): its
/// boundary faces by zone, in MeshInfo zone order, polygons as stored, nodes
/// in order of first use. Throws E0120 or E0121 for a face or cell whose
/// geometry is not valid (an inverted cell would turn normals inward), E0113
/// if a count exceeds 2^32 - 1.
std::string surface_bytes(const Mesh& mesh, const std::string& sha256);

}  // namespace autan::io
