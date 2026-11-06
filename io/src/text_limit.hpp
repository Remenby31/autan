#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "autan/core/error.hpp"

namespace autan::io {

/// The longest name a reader keeps, in UTF-8 bytes (mesh.md, E0122).
inline constexpr std::size_t max_name_bytes = 4096;

/// Throws E0122 when `bytes` exceeds `limit`, before the caller copies the text: `what` names it
/// ("zone name", "ZoneNames descriptor"...).
inline void check_text(const std::string& path, std::string_view what, std::size_t bytes,
                       std::size_t limit = max_name_bytes) {
  if (bytes <= limit) return;
  throw Error(ErrorCode::mesh_text_too_long,
              path + ": a " + std::string(what) + " of " + std::to_string(bytes) +
                  " bytes exceeds " + std::to_string(limit),
              R"({"what":")" + std::string(what) + R"(","bytes":)" + std::to_string(bytes) +
                  R"(,"limit":)" + std::to_string(limit) + "}");
}

}  // namespace autan::io
