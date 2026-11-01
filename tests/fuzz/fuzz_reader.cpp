// libFuzzer entry point for one mesh reader (AUTAN_FUZZ_READER): every input
// must read or end with an autan::Error carrying an E01xx code. A crash, a
// sanitizer report, an uncaught exception of another type or another code
// is a finding.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <unistd.h>

#include "autan/core/error.hpp"
#include "read_input.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto scratch =
      autan_fuzz::fs::temp_directory_path() / ("autan-fuzz-" + std::to_string(::getpid()));
  try {
    autan_fuzz::read_input(autan_fuzz::Reader::AUTAN_FUZZ_READER,
                           {reinterpret_cast<const char*>(data), size}, scratch);
  } catch (const autan::Error& e) {
    const auto code = static_cast<unsigned>(e.code());
    if (code < 101 || code > 199) std::abort();
  }
  return 0;
}
