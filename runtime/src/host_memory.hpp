#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

namespace autan::runtime {

/// The bytes this process can still allocate on its host. On Linux, the
/// lower of MemAvailable and, for every memory cgroup above the process with
/// a limit, the limit less its usage: cgroup v2 memory.max and
/// memory.current, cgroup v1 memory.limit_in_bytes and memory.usage_in_bytes,
/// found through `proc`/self/cgroup and the mounts of `proc`/self/mountinfo.
/// On macOS, the free and inactive pages. Throws std::runtime_error when
/// MemAvailable is missing or a limit file exists but cannot be read.
std::int64_t host_available_bytes(const std::filesystem::path& proc = "/proc");

/// What a host can still give the ranks it runs: one pool, or the host's and
/// a discrete device's (`device_free`, shared by the ranks on that device).
struct MemoryPools {
  std::int64_t host_available = 0;
  std::optional<std::int64_t> device_free;
  std::int64_t devices = 1;  ///< the discrete devices the host's ranks share
};

/// The memory the ranks of a host need beyond what one of its pools can give
/// less `margin`, as {required, available} of that pool; nullopt when every
/// pool holds its share. Each of `ranks` ranks needs `bytes_host` and
/// `bytes_device`, both from one pool when it is the only one. Sums saturate.
struct Shortfall {
  std::int64_t required = 0;
  std::int64_t available = 0;
};
std::optional<Shortfall> memory_shortfall(std::int64_t bytes_host, std::int64_t bytes_device,
                                          std::int64_t ranks, const MemoryPools& pools,
                                          std::int64_t margin);

}  // namespace autan::runtime
