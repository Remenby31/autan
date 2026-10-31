#pragma once

#include <filesystem>

#include "autan/core/mesh.hpp"

namespace autan::io {

/// Reads a mesh, detecting the format from the content: a Gmsh MSH 4.1 file
/// (ASCII or binary), a Fluent mesh file (ASCII or binary sections), or an
/// OpenFOAM polyMesh directory (the directory itself or a case directory
/// holding constant/polyMesh). Returns the assembled mesh with zones and
/// provenance; throws autan::Error with an E01xx code (docs/contracts/mesh.md).
Mesh read_mesh(const std::filesystem::path& path);

Mesh read_gmsh(const std::filesystem::path& file);
Mesh read_fluent(const std::filesystem::path& file);
Mesh read_polymesh(const std::filesystem::path& directory);

}  // namespace autan::io
