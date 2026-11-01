#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <sys/resource.h>

#include "autan/core/error.hpp"
#include "read_input.hpp"

namespace {

// Peak resident memory of this process so far, in bytes.
long long peak_memory() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  return static_cast<long long>(usage.ru_maxrss);  // bytes on macOS
#else
  return static_cast<long long>(usage.ru_maxrss) * 1024;  // KiB on Linux
#endif
}

}  // namespace

// Inputs the fuzzers once broke the readers with, kept in tests/data/fuzz/:
// each must now read or end with a coded mesh error, without the memory
// spike some of them caused (a size field taken from the file).
TEST_CASE("Former fuzzer findings read or end with an E01xx error", "[fuzz]") {
  const autan_fuzz::fs::path data{AUTAN_FUZZ_DATA_DIR};
  std::random_device random;
  const auto scratch = autan_fuzz::fs::temp_directory_path() /
                       ("autan-fuzz-regressions-" + std::to_string(random()));
  const std::pair<const char*, autan_fuzz::Reader> readers[] = {
      {"gmsh", autan_fuzz::Reader::gmsh},
      {"fluent", autan_fuzz::Reader::fluent},
      {"polymesh", autan_fuzz::Reader::polymesh},
#ifdef AUTAN_HAVE_CGNS
      {"cgns", autan_fuzz::Reader::cgns},
#endif
  };
  int files = 0;
  for (const auto& [name, reader] : readers) {
    if (!autan_fuzz::fs::exists(data / name)) continue;
    for (const auto& entry : autan_fuzz::fs::directory_iterator(data / name)) {
      CAPTURE(entry.path().string());
      std::ifstream in(entry.path(), std::ios::binary);
      std::ostringstream bytes;
      bytes << in.rdbuf();
      const long long before = peak_memory();
      try {
        autan_fuzz::read_input(reader, bytes.str(), scratch);
      } catch (const autan::Error& e) {
        const auto code = static_cast<unsigned>(e.code());
        CHECK(code >= 101);
        CHECK(code <= 199);
      }
      CHECK(peak_memory() - before < 256LL * 1024 * 1024);
      ++files;
    }
  }
  CHECK(files > 0);
}
