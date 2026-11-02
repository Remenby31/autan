#pragma once

#include <cstdint>

namespace autan::runtime {

/// The bytes this process can still allocate on its host: on Linux, the
/// lower of MemAvailable (/proc/meminfo) and, for every cgroup v2 above the
/// process that has a limit, memory.max minus memory.current; on macOS, the
/// free, inactive and purgeable pages. Throws std::runtime_error when the
/// host does not say.
std::int64_t host_available_bytes();

}  // namespace autan::runtime
