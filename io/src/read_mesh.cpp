#include "autan/io/read_mesh.hpp"

#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include "autan/io/cgns.hpp"
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
  // CGNS files are HDF5 files: signature \x89HDF\r\n\x1a\n at offset 0.
  char signature[8] = {};
  in.read(signature, sizeof signature);
  if (in.gcount() == 8 &&
      std::string_view(signature, 8) == std::string_view("\x89HDF\r\n\x1a\n", 8)) {
#ifdef AUTAN_HAVE_CGNS
    return read_cgns(path);
#else
    throw Error(ErrorCode::io_unsupported,
                "'" + path.string() +
                    "' is a CGNS/HDF5 file and this build has no CGNS support "
                    "(AUTAN_ENABLE_CGNS=OFF)");
#endif
  }
  in.clear();
  in.seekg(0);
  char c = ' ';
  while (in.get(c) && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
  }
  if (c == '$') return read_gmsh(path);
  if (c == '(') return read_fluent(path);
  throw Error(ErrorCode::io_unknown_format,
              "'" + path.string() + "' is neither a Gmsh, Fluent, CGNS nor OpenFOAM mesh");
}

}  // namespace autan::io
