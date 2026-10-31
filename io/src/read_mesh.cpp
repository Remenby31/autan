#include "autan/io/read_mesh.hpp"

#include <fstream>
#include <iterator>
#include <string>

#include "cursor.hpp"

namespace autan::io {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error(ErrorCode::io_cannot_open, "cannot open '" + path + "'");
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) throw Error(ErrorCode::io_cannot_open, "cannot read '" + path + "'");
  return data;
}

Mesh read_mesh(const std::filesystem::path& path) {
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) return read_polymesh(path);
  if (!std::filesystem::exists(path, ec)) {
    throw Error(ErrorCode::io_cannot_open, "'" + path.string() + "' does not exist");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error(ErrorCode::io_cannot_open, "cannot open '" + path.string() + "'");
  char c = ' ';
  while (in.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
  }
  if (c == '$') return read_gmsh(path);
  if (c == '(') return read_fluent(path);
  throw Error(ErrorCode::io_unknown_format,
              "'" + path.string() + "' is neither a Gmsh, a Fluent nor an OpenFOAM mesh");
}

}  // namespace autan::io
