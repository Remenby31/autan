#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "autan/core/error.hpp"
#include "read_input.hpp"

// Inputs the fuzzers once broke the readers with, kept in tests/data/fuzz/:
// each must now read or end with a coded mesh error.
TEST_CASE("Former fuzzer findings read or end with an E01xx error", "[fuzz]") {
  const autan_fuzz::fs::path data{AUTAN_FUZZ_DATA_DIR};
  std::random_device random;
  const auto scratch = autan_fuzz::fs::temp_directory_path() / ("autan-fuzz-regressions-" + std::to_string(random()));
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
      try {
        autan_fuzz::read_input(reader, bytes.str(), scratch);
      } catch (const autan::Error& e) {
        const auto code = static_cast<unsigned>(e.code());
        CHECK(code >= 101);
        CHECK(code <= 199);
      }
      ++files;
    }
  }
  CHECK(files > 0);
}
