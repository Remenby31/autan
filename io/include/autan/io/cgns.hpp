#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Reads a CGNS mesh: one base, one unstructured zone, cells as NFACE_n over
/// NGON_n faces or as linear standard elements (MIXED included). Boundary
/// zones come from BC_t nodes, cell zones from cell-centred BC_t,
/// ZoneSubRegion_t or the families of element sections; supported layouts
/// and rejections are listed in docs/contracts/mesh.md, "Readers".
Mesh read_cgns(const std::filesystem::path& file);

/// What reading a mesh file costs beyond its arrays' counts, in bytes at their stored element count
/// times element size whatever the compression (mesh-info.md, source): `total` the datasets the
/// reader reads and every descriptor, `descriptors` every Descriptor_t (the CGNS library loads them
/// all at cg_open), `zone_names` the ZoneNames descriptor the reader parses. 0 descriptors and zone
/// names for a format without them.
struct LogicalBytes {
  std::int64_t total = 0;
  std::int64_t descriptors = 0;
  std::int64_t zone_names = 0;
  /// Of the filtered (compressed) datasets, the largest twice-chunk-logical plus stored size: what
  /// HDF5 holds beside the destination while it reads one. 0 for an unfiltered file.
  std::int64_t filtered = 0;
  /// Every node of the file's tree, for each of which the CGNS library allocates at cg_open.
  std::int64_t nodes = 0;
  /// The entries of every IndexArray_t under the zone's BCs, subregions and connectivities, for
  /// whose ids the reader holds 8 bytes each to know which faces they name. 0 for other formats.
  std::int64_t point_list_ids = 0;
  /// The data cg_open reads beyond descriptors (mesh.md, Memory of a CGNS read): at most 1 MiB.
  std::int64_t eager = 0;
};

/// LogicalBytes of a CGNS file, from its node metadata only, through cgio (no cg_open).
LogicalBytes cgns_logical_bytes(const std::filesystem::path& file);

struct Field {
  std::string name;  ///< SIDS name, e.g. "Density"
  std::vector<double> values;
};

/// Content of a solution.cgns beyond the mesh (docs/contracts/results.md).
/// Zone lists follow the MeshInfo order: boundary zones are the mesh's face
/// zones with `boundary` set, cell zones those holding cells.
struct Solution {
  std::string run_metadata;  ///< JSON text of the RunMetadata descriptor
  /// BCType name of the resolved condition of each boundary zone
  /// (e.g. "BCWallViscousHeatFlux"); empty for a zone without faces and for
  /// a periodic zone, which is a connectivity, not a boundary condition.
  std::vector<std::string> boundary_bc_types;
  std::vector<Field> fields;        ///< cell-centred, one value per cell
  std::vector<Field> mean_fields;   ///< time averages; empty unless averaging
  std::vector<index_t> wall_faces;  ///< ascending face indices of wall and slip faces
  std::vector<Field> wall_fields;   ///< one value per wall face
  /// Cell-centred state a physics needs to restart beyond `fields` (e.g.
  /// earlier time levels), written as the FlowSolution_t "RestartState";
  /// readers of results ignore it. Empty: no such node.
  std::vector<Field> restart_fields;
};

/// Cell fields of a solution.cgns written by write_solution, to restart from:
/// the zone's cell count, the RunMetadata text, and FlowSolution,
/// FlowSolutionMean and RestartState (each empty when absent, the last two
/// not read without `whole_state`: a warm start needs only FlowSolution).
/// Throws E0130 to E0133 for a file outside that layout.
struct StoredSolution {
  index_t cells = 0;
  std::string run_metadata;
  std::vector<Field> fields;
  std::vector<Field> mean_fields;
  std::vector<Field> restart_fields;
};
StoredSolution read_solution(const std::filesystem::path& file, bool whole_state = true);

/// The arrays of each FlowSolution_t of the zone of start solution `file`, by solution name, read
/// through cgio without their values (after check_start_solution: the walk is bounded).
std::map<std::string, std::vector<std::string>> solution_field_names(
    const std::filesystem::path& file);

/// The run's mesh a start solution was written for: its boundary and cell zones and its periodic
/// pairs (results.md, Sizes a reader accepts).
struct StartLimits {
  std::int64_t zones = 0;
  std::int64_t periodic_pairs = 0;
};

/// Checks through cgio alone, before the CGNS library loads anything, that `file` holds no
/// descriptor or family beyond what results.md's writer writes for a mesh of `limits`:
/// RunMetadata of at most 64 KiB and ZoneNames of at most (zones + 1) × 25 600 bytes in the base,
/// at most 2 × periodic_pairs PeriodicTranslation of at most 4096 bytes, no other descriptor,
/// at most `zones` Family_t, no filtered (compressed) dataset, at most 16 384 nodes. Throws E0132
/// otherwise.
void check_start_solution(const std::filesystem::path& file, const StartLimits& limits);

/// The cells of a solution's zone, through cgio alone (no cg_open).
std::int64_t solution_cells(const std::filesystem::path& file);

/// Writes `solution` and `mesh` (unpartitioned) as solution.cgns with the
/// layout of docs/contracts/results.md. Throws E0101 when the arrays do not
/// match the mesh and E0130 when the file cannot be written.
void write_solution(const std::filesystem::path& file, const Mesh& mesh, const Solution& solution);

/// Node names of the MeshInfo boundary and cell zones by the rule of
/// results.md: the zone name when it is a safe, unique CGNS node name, else
/// boundary-NNNN / cell-zone-NNNN.
struct ZoneNodeNames {
  std::vector<std::string> boundary;
  std::vector<std::string> cell;
};
ZoneNodeNames zone_node_names(const std::vector<std::string>& boundary,
                              const std::vector<std::string>& cell);

}  // namespace autan::io
