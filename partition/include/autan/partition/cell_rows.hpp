#pragma once

#include <cstdint>
#include <vector>

#include "autan/core/mesh.hpp"

namespace autan {

/// Variable-length rows of cells as CSR: row c holds the entries offsets[c] .. offsets[c + 1],
/// each a key and `stride` values (values[stride * e ..]); a row may be empty.
struct CellRows {
  std::vector<offset_t> offsets{0};
  std::vector<std::int64_t> keys;
  std::vector<double> values;
  int stride = 0;
};

}  // namespace autan
