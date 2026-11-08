// autan-vtu SOLUTION OUT: a run's solution.cgns as one VTK XML UnstructuredGrid
// (docs/contracts/vtu.md).
#include <cstdio>
#include <exception>
#include <string_view>

#include "autan/core/error.hpp"
#include "autan/io/vtu.hpp"

int main(int argc, const char* const* argv) {
  constexpr const char* usage = "usage: autan-vtu SOLUTION.cgns OUT.vtu\n";
  if (argc != 3 || std::string_view(argv[1]).starts_with('-')) {
    std::fputs(usage, stderr);
    return 2;
  }
  try {
    autan::io::write_vtu(argv[1], argv[2]);
  } catch (const autan::Error& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "autan-vtu: %s\n", e.what());
    return 1;
  }
  return 0;
}
