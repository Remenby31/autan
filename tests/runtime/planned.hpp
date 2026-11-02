#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "autan/io/mesh_counts.hpp"
#include "autan/io/mesh_info.hpp"
#include "autan/io/read_mesh.hpp"

namespace autan_test {

// The order autan-run computes in, as this process would start it (AUTAN_CELL_ORDER).
inline std::string run_cell_order() {
  const char* order = std::getenv("AUTAN_CELL_ORDER");
  return order == nullptr ? "rcm" : order;
}

// `doc` with the mesh_info `autan prepare` plans it with (solver-input.md): autan-mesh-info's
// document for its mesh at its mesh.scale, lines in `order`. An input whose mesh cannot be read
// takes the document of `readable`, so that the run reaches the mesh and fails there.
inline nlohmann::json planned(nlohmann::json doc, const std::filesystem::path& readable,
                              const std::string& order = run_cell_order()) {
  std::filesystem::path mesh = doc["mesh"]["path"].get<std::string>();
  if (!std::filesystem::exists(mesh)) mesh = readable;
  const autan::Mesh m = autan::io::read_mesh(mesh);
  // An input whose scale the reader refuses is planned at 1: it fails at mesh.scale first.
  const nlohmann::json given = doc["mesh"].value("scale", nlohmann::json());
  const double scale = given.is_number() && given.get<double>() > 0.0 ? given.get<double>() : 1.0;
  const auto counts = autan::io::run_counts(
      m, scale, order == "input" ? autan::io::CellOrder::input : autan::io::CellOrder::rcm);
  doc["mesh_info"] = nlohmann::json::parse(autan::io::mesh_info_json(
      m, autan::io::mesh_hash(mesh), autan::io::mesh_bytes(mesh), counts));
  return doc;
}

}  // namespace autan_test
