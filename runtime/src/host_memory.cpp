#include "host_memory.hpp"

#include <algorithm>
#include <stdexcept>

#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#endif

namespace autan::runtime {

#if defined(__APPLE__)

std::int64_t host_available_bytes() {
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
                        &count) != KERN_SUCCESS) {
    throw std::runtime_error("host_statistics64 failed");
  }
  const auto pages =
      static_cast<std::int64_t>(vm.free_count) + vm.inactive_count + vm.purgeable_count;
  return pages * static_cast<std::int64_t>(vm_page_size);
}

#else

namespace {

namespace fs = std::filesystem;

// A cgroup v2 file holding one number; nullopt when absent or "max".
std::optional<std::int64_t> cgroup_value(const fs::path& file) {
  std::ifstream in(file);
  std::string text;
  if (!(in >> text) || text == "max") return std::nullopt;
  return std::stoll(text);
}

}  // namespace

std::int64_t host_available_bytes() {
  std::ifstream meminfo("/proc/meminfo");
  std::optional<std::int64_t> available;
  for (std::string line; std::getline(meminfo, line);) {
    std::istringstream fields(line);
    std::string key;
    std::int64_t kib = 0;
    if (fields >> key >> kib && key == "MemAvailable:") available = kib * 1024;
  }
  if (!available) throw std::runtime_error("/proc/meminfo gives no MemAvailable");
  // cgroup v2: the line "0::<path>"; every cgroup from it up to the root may limit memory.
  std::ifstream self("/proc/self/cgroup");
  for (std::string line; std::getline(self, line);) {
    if (!line.starts_with("0::")) continue;
    for (fs::path group = fs::path(line.substr(3)).relative_path();; group = group.parent_path()) {
      const fs::path dir = fs::path("/sys/fs/cgroup") / group;
      const auto limit = cgroup_value(dir / "memory.max");
      const auto used = cgroup_value(dir / "memory.current");
      if (limit && used)
        available = std::min(*available, std::max<std::int64_t>(0, *limit - *used));
      if (group.empty()) break;
    }
  }
  return *available;
}

#endif

}  // namespace autan::runtime
