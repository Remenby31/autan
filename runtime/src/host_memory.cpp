#include "host_memory.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <fstream>
#include <sstream>
#include <vector>
#endif

namespace autan::runtime {

namespace {

constexpr std::int64_t top = std::numeric_limits<std::int64_t>::max();

std::int64_t times(std::int64_t n, std::int64_t bytes) {
  return n > 0 && bytes > top / n ? top : n * bytes;
}

std::int64_t plus(std::int64_t a, std::int64_t b) { return b > top - a ? top : a + b; }

}  // namespace

std::optional<Shortfall> memory_shortfall(std::int64_t bytes_host, std::int64_t bytes_device,
                                          std::int64_t ranks, const MemoryPools& pools,
                                          std::int64_t margin) {
  const auto short_of = [&](std::int64_t required,
                            std::int64_t available) -> std::optional<Shortfall> {
    if (required > available - margin)
      return Shortfall{.required = required, .available = available};
    return std::nullopt;
  };
  if (pools.device_free) {
    const std::int64_t sharing = (ranks + pools.devices - 1) / pools.devices;
    if (auto s = short_of(times(sharing, bytes_device), *pools.device_free)) return s;
    return short_of(times(ranks, bytes_host), pools.host_available);
  }
  return short_of(times(ranks, plus(bytes_host, bytes_device)), pools.host_available);
}

#if defined(__APPLE__)

HostMemory host_memory(const std::filesystem::path& /*proc*/) {
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
                        &count) != KERN_SUCCESS) {
    throw std::runtime_error("host_statistics64 failed");
  }
  // Purgeable pages are counted among the free and inactive ones already.
  const auto pages = static_cast<std::int64_t>(vm.free_count) + vm.inactive_count;
  return {.available = pages * static_cast<std::int64_t>(vm_page_size), .unseen_cgroups = {}};
}

#else

namespace {

namespace fs = std::filesystem;

// A limit or usage file holding one number; nullopt when absent, or for a v2
// limit of "max". A file that exists but holds no number is not a limit we can
// trust: an error.
std::optional<std::int64_t> number_in(const fs::path& file) {
  std::error_code ec;
  if (!fs::exists(file, ec)) return std::nullopt;
  std::ifstream in(file);
  std::string text;
  if (!(in >> text)) throw std::runtime_error("cannot read " + file.string());
  if (text == "max") return std::nullopt;
  try {
    return std::stoll(text);
  } catch (const std::exception&) {
    throw std::runtime_error(file.string() + " holds '" + text + "', not a number of bytes");
  }
}

struct Mount {
  fs::path root, point;
};

// proc(5): mountinfo writes a space, tab, newline and backslash of a path as \040, \011, \012 and
// \134 (the kernel's show_path); every \ followed by three octal digits is one byte.
std::string unescaped(const std::string& field) {
  std::string out;
  for (std::size_t i = 0; i < field.size(); ++i) {
    const auto octal = [&](std::size_t k) { return field[k] >= '0' && field[k] <= '7'; };
    if (field[i] == '\\' && i + 3 < field.size() && octal(i + 1) && octal(i + 2) && octal(i + 3)) {
      out.push_back(static_cast<char>((field[i + 1] - '0') * 64 + (field[i + 2] - '0') * 8 +
                                      (field[i + 3] - '0')));
      i += 3;
    } else {
      out.push_back(field[i]);
    }
  }
  return out;
}

// proc(5), /proc/self/mountinfo: "id parent major:minor root mountpoint options... - fstype
// source superoptions"; the mounts of cgroup v2, or of the cgroup v1 memory controller.
std::vector<Mount> cgroup_mounts(const fs::path& proc, bool v2) {
  std::vector<Mount> mounts;
  std::ifstream in(proc / "self" / "mountinfo");
  for (std::string line; std::getline(in, line);) {
    const auto dash = line.find(" - ");
    if (dash == std::string::npos) continue;
    std::istringstream left(line.substr(0, dash)), right(line.substr(dash + 3));
    std::string id, parent, device, root, point, type, source, options;
    if (!(left >> id >> parent >> device >> root >> point) || !(right >> type >> source >> options))
      continue;
    const bool memory = ("," + options + ",").find(",memory,") != std::string::npos;
    if (v2 ? type == "cgroup2" : type == "cgroup" && memory)
      mounts.push_back({unescaped(root), unescaped(point)});
  }
  return mounts;
}

// The least limit less usage of `group` and of each of its ancestors, each read in the mount
// that shows it: a parent's limit binds the group even when another mount shows the group.
std::optional<std::int64_t> cgroup_available(const fs::path& proc, const fs::path& group, bool v2,
                                             std::vector<std::string>& unseen) {
  const auto mounts = cgroup_mounts(proc, v2);
  std::optional<std::int64_t> available;
  for (fs::path cg = group.lexically_normal();; cg = cg.parent_path()) {
    const Mount* shown = nullptr;
    for (const auto& m : mounts) {
      const bool contains = std::ranges::mismatch(m.root, cg).in1 == m.root.end();
      if (contains &&
          (shown == nullptr || std::distance(m.root.begin(), m.root.end()) >
                                   std::distance(shown->root.begin(), shown->root.end())))
        shown = &m;
    }
    if (shown == nullptr && cg != cg.root_path()) unseen.push_back(cg.string());
    if (shown != nullptr) {
      const fs::path relative = cg.lexically_relative(shown->root);
      const fs::path dir = relative == "." ? shown->point : shown->point / relative;
      const auto limit = number_in(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
      // cgroup v1 writes an unlimited group's limit as the largest page-aligned count.
      if (limit && (v2 || *limit < (std::int64_t{1} << 62))) {
        const auto used = number_in(dir / (v2 ? "memory.current" : "memory.usage_in_bytes"));
        if (!used) throw std::runtime_error("no memory usage beside the limit in " + dir.string());
        const std::int64_t left = std::max<std::int64_t>(0, *limit - *used);
        available = std::min(available.value_or(left), left);
      }
    }
    if (cg == cg.parent_path() || cg.empty()) break;
  }
  return available;
}

}  // namespace

HostMemory host_memory(const std::filesystem::path& proc) {
  std::ifstream meminfo(proc / "meminfo");
  std::optional<std::int64_t> available;
  for (std::string line; std::getline(meminfo, line);) {
    std::istringstream fields(line);
    std::string key;
    std::int64_t kib = 0;
    if (fields >> key >> kib && key == "MemAvailable:") available = kib * 1024;
  }
  if (!available) throw std::runtime_error((proc / "meminfo").string() + " gives no MemAvailable");
  std::vector<std::string> unseen;
  // proc(5), /proc/self/cgroup: "0::path" for cgroup v2, "id:controllers:path" for v1.
  std::ifstream self(proc / "self" / "cgroup");
  for (std::string line; std::getline(self, line);) {
    const auto first = line.find(':');
    const auto second = line.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos) continue;
    const std::string controllers = line.substr(first + 1, second - first - 1);
    const fs::path group = line.substr(second + 1);
    const bool v2 = line.substr(0, first) == "0" && controllers.empty();
    const bool v1 = ("," + controllers + ",").find(",memory,") != std::string::npos;
    if (!v2 && !v1) continue;
    if (const auto limited = cgroup_available(proc, group, v2, unseen))
      available = std::min(*available, *limited);
  }
  return {.available = *available, .unseen_cgroups = std::move(unseen)};
}

#endif

}  // namespace autan::runtime
