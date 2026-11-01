// libFuzzer harness for the mesh readers (issue #75). Each input is written
// to a temporary file (or a polyMesh directory) and read through the public
// reader; reading must succeed or end with an autan::Error carrying an
// E01xx code. Anything else (a crash, a sanitizer report, another
// exception) is a finding.

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "autan/core/error.hpp"
#include "autan/io/read_mesh.hpp"
#ifdef AUTAN_HAVE_CGNS
#include "autan/io/cgns.hpp"
#endif

namespace {

namespace fs = std::filesystem;

const fs::path& scratch() {
  static const fs::path dir = [] {
    const fs::path d = fs::temp_directory_path() / ("autan-fuzz-" + std::to_string(::getpid()));
    fs::create_directories(d);
    return d;
  }();
  return dir;
}

void write(const fs::path& path, std::string_view bytes) {
  std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A polyMesh input is its five files separated by the byte 0x00: points,
// faces, owner, neighbour, boundary (missing parts are empty files).
fs::path polymesh(std::string_view bytes) {
  const fs::path dir = scratch() / "polyMesh";
  fs::create_directories(dir);
  const char* names[] = {"points", "faces", "owner", "neighbour", "boundary"};
  for (const char* name : names) {
    const auto cut = bytes.find('\0');
    write(dir / name, bytes.substr(0, cut));
    bytes = cut == std::string_view::npos ? std::string_view() : bytes.substr(cut + 1);
  }
  return dir;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view bytes(reinterpret_cast<const char*>(data), size);
  try {
#if defined(AUTAN_FUZZ_GMSH)
    const fs::path file = scratch() / "input.msh";
    write(file, bytes);
    (void)autan::io::read_gmsh(file);
#elif defined(AUTAN_FUZZ_FLUENT)
    const fs::path file = scratch() / "input.msh";
    write(file, bytes);
    (void)autan::io::read_fluent(file);
#elif defined(AUTAN_FUZZ_POLYMESH)
    (void)autan::io::read_polymesh(polymesh(bytes));
#elif defined(AUTAN_FUZZ_CGNS)
    const fs::path file = scratch() / "input.cgns";
    write(file, bytes);
    (void)autan::io::read_cgns(file);
#else
#error "define the reader to fuzz"
#endif
  } catch (const autan::Error& e) {
    const auto code = static_cast<unsigned>(e.code());
    if (code < 101 || code > 199) std::abort();
  }
  return 0;
}
