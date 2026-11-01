#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "autan/io/read_mesh.hpp"
#ifdef AUTAN_HAVE_CGNS
#include "autan/io/cgns.hpp"
#endif

namespace autan_fuzz {

namespace fs = std::filesystem;

enum class Reader { gmsh, fluent, polymesh, cgns };

inline void write(const fs::path& path, std::string_view bytes) {
  std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/// Reads `bytes` as a file of `reader` through its public reader, using
/// `scratch` for the files. Returns, or throws what the reader throws: an
/// autan::Error with an E01xx code is the only accepted failure. A polyMesh
/// input is its five files (points, faces, owner, neighbour, boundary)
/// joined by NUL bytes.
inline void read_input(Reader reader, std::string_view bytes, const fs::path& scratch) {
  fs::create_directories(scratch);
  switch (reader) {
    case Reader::gmsh:
      write(scratch / "input.msh", bytes);
      (void)autan::io::read_gmsh(scratch / "input.msh");
      return;
    case Reader::fluent:
      write(scratch / "input.msh", bytes);
      (void)autan::io::read_fluent(scratch / "input.msh");
      return;
    case Reader::polymesh: {
      const fs::path dir = scratch / "polyMesh";
      fs::create_directories(dir);
      for (const char* name : {"points", "faces", "owner", "neighbour", "boundary"}) {
        const auto cut = bytes.find('\0');
        write(dir / name, bytes.substr(0, cut));
        bytes = cut == std::string_view::npos ? std::string_view() : bytes.substr(cut + 1);
      }
      (void)autan::io::read_polymesh(dir);
      return;
    }
    case Reader::cgns:
#ifdef AUTAN_HAVE_CGNS
      write(scratch / "input.cgns", bytes);
      (void)autan::io::read_cgns(scratch / "input.cgns");
#endif
      return;
  }
}

}  // namespace autan_fuzz
