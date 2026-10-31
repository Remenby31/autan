#include "autan/core/error.hpp"

#include <cstdio>

namespace autan {

std::string error_code_string(ErrorCode code) {
  char buffer[8];
  std::snprintf(buffer, sizeof buffer, "E%04u", static_cast<unsigned>(code));
  return buffer;
}

Error::Error(ErrorCode code, std::string_view message)
    : std::runtime_error(error_code_string(code) + ": " + std::string(message)), code_(code) {}

}  // namespace autan
