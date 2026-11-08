#pragma once

#include <filesystem>

namespace autan::io {

/// Writes the run solution `solution` (a solution.cgns of docs/contracts/results.md) as the VTK
/// XML UnstructuredGrid `out` of docs/contracts/vtu.md: every cell a polyhedron with outward
/// faces, and as cell data the FlowSolution fields, the FlowSolutionMean fields as <name>Mean and
/// CellZone, the index of the cell's zone in the MeshInfo cell zones. `out` is written under a
/// temporary name and renamed, so it is whole or absent. Throws E0130 when a file cannot be
/// opened or written, E0147 when `solution` holds no FlowSolution, and the readers' codes for a
/// file outside results.md's layout.
void write_vtu(const std::filesystem::path& solution, const std::filesystem::path& out);

}  // namespace autan::io
