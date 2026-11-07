#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "host_memory.hpp"

namespace fs = std::filesystem;
using autan::runtime::memory_shortfall;
using autan::runtime::MemoryPools;

namespace {

constexpr std::int64_t gib = std::int64_t{1} << 30;

#if !defined(__APPLE__)
// For the /proc tests only: macOS has no /proc, and its tests below say so.
fs::path fresh_dir() {
  std::random_device random;
  const fs::path dir =
      fs::temp_directory_path() / ("autan_host_memory-" + std::to_string(random()));
  fs::create_directories(dir);
  return dir;
}

void write(const fs::path& file, const std::string& text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file) << text;
}
#endif

}  // namespace

TEST_CASE("The memory a host can still give is the least of MemAvailable and its cgroup limits",
          "[runtime]") {
#if defined(__APPLE__)
  SKIP("macOS has no /proc: its free and inactive pages are read through Mach");
#else
  // proc(5): /proc/meminfo in kB, /proc/self/cgroup and /proc/self/mountinfo; the limits in
  // memory.max / memory.current (cgroup v2) or memory.limit_in_bytes / memory.usage_in_bytes (v1),
  // less the inactive page cache of memory.stat.
  const fs::path dir = fresh_dir();
  const fs::path proc = dir / "proc";
  write(proc / "meminfo", "MemTotal: 134217728 kB\nMemAvailable: 67108864 kB\n");
  const std::int64_t host = std::int64_t{67108864} * 1024;  // 64 GiB
  SECTION("no cgroup") { CHECK(autan::runtime::host_memory(proc).available == host); }
  SECTION("cgroup v2: the tightest of the group and its ancestors") {
    const fs::path cg = dir / "sys" / "cgroup";
    write(proc / "self" / "cgroup", "0::/tenant/job\n");
    write(proc / "self" / "mountinfo",
          "24 1 0:22 / " + cg.string() + " rw,nosuid - cgroup2 cgroup2 rw,nsdelegate\n");
    write(cg / "tenant" / "memory.max", std::to_string(4 * gib) + "\n");
    write(cg / "tenant" / "memory.current", std::to_string(gib) + "\n");
    write(cg / "tenant" / "job" / "memory.max", "max\n");
    write(cg / "tenant" / "job" / "memory.current", std::to_string(gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 3 * gib);
    // A group above its limit has nothing left.
    write(cg / "tenant" / "job" / "memory.max", std::to_string(gib / 2) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 0);
    // A limit that cannot be read is not taken for no limit.
    write(cg / "tenant" / "job" / "memory.max", "a lot\n");
    CHECK_THROWS_AS(autan::runtime::host_memory(proc).available, std::runtime_error);
  }
  SECTION("cgroup v2: the inactive page cache counts as available") {
    // A capped container after builds and file I/O: memory.current includes the page cache,
    // whose inactive part reclaim takes first (memory.stat inactive_file); the active part may
    // not be reclaimable and does not count (#582).
    const fs::path cg = dir / "sys" / "cgroup";
    write(proc / "self" / "cgroup", "0::/job\n");
    write(proc / "self" / "mountinfo", "24 1 0:22 / " + cg.string() + " rw - cgroup2 cgroup2 rw\n");
    write(cg / "job" / "memory.max", std::to_string(6 * gib) + "\n");
    write(cg / "job" / "memory.current", std::to_string(6 * gib - gib / 4) + "\n");
    const std::int64_t need = 2 * gib, margin = 2 * gib;
    const auto pools = [&] {
      return MemoryPools{.host_available = autan::runtime::host_memory(proc).available,
                         .device_free = {},
                         .devices = 1,
                         .unseen_cgroups = {}};
    };
    // Mostly cache, 5 GiB of it inactive: 6 - 5.75 + 5 GiB left, and the run fits.
    write(cg / "job" / "memory.stat", "anon 262144\nfile 6174015488\nactive_file " +
                                          std::to_string(gib / 2) + "\ninactive_file " +
                                          std::to_string(5 * gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 5 * gib + gib / 4);
    CHECK_FALSE(memory_shortfall(need, 0, 1, pools(), margin));
    // The same usage held by the processes: no inactive cache, and the run is refused.
    write(cg / "job" / "memory.stat", "anon 6174015488\nfile 0\ninactive_file 0\n");
    CHECK(autan::runtime::host_memory(proc).available == gib / 4);
    CHECK(memory_shortfall(need, 0, 1, pools(), margin));
    // A memory.stat without inactive_file, or none: the limit less the usage.
    write(cg / "job" / "memory.stat", "anon 262144\n");
    CHECK(autan::runtime::host_memory(proc).available == gib / 4);
    fs::remove(cg / "job" / "memory.stat");
    CHECK(autan::runtime::host_memory(proc).available == gib / 4);
    // No limit: the cache does not raise the host's MemAvailable.
    write(cg / "job" / "memory.stat", "inactive_file " + std::to_string(5 * gib) + "\n");
    write(cg / "job" / "memory.max", "max\n");
    CHECK(autan::runtime::host_memory(proc).available == host);
    // Never more than the limit, whatever memory.stat says.
    write(cg / "job" / "memory.max", std::to_string(6 * gib) + "\n");
    write(cg / "job" / "memory.stat", "inactive_file " + std::to_string(8 * gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 6 * gib);
    // A limit near the largest count: the credit saturates at the limit instead of overflowing.
    write(cg / "job" / "memory.max",
          std::to_string(std::numeric_limits<std::int64_t>::max() - 8191) + "\n");
    write(cg / "job" / "memory.current", "4096\n");
    write(cg / "job" / "memory.stat", "inactive_file 16384\n");
    CHECK(autan::runtime::host_memory(proc).available == host);
  }
  SECTION("cgroup v2: an ancestor's page cache does not count") {
    // The parent's 5 GiB of cache may be a sibling's that memory.min protects from reclaim: the
    // parent's limit less its usage binds the job (Linux mm/vmscan.c, protected cgroups skipped).
    const fs::path cg = dir / "sys" / "cgroup";
    write(proc / "self" / "cgroup", "0::/tenant/job\n");
    write(proc / "self" / "mountinfo", "24 1 0:22 / " + cg.string() + " rw - cgroup2 cgroup2 rw\n");
    write(cg / "tenant" / "memory.max", std::to_string(8 * gib) + "\n");
    write(cg / "tenant" / "memory.current", std::to_string(7 * gib) + "\n");
    write(cg / "tenant" / "memory.stat", "inactive_file " + std::to_string(5 * gib) + "\n");
    write(cg / "tenant" / "job" / "memory.max", "max\n");
    write(cg / "tenant" / "job" / "memory.current", std::to_string(2 * gib) + "\n");
    write(cg / "tenant" / "job" / "memory.stat",
          "anon " + std::to_string(2 * gib) + "\ninactive_file 0\n");
    CHECK(autan::runtime::host_memory(proc).available == gib);
  }
  SECTION("cgroup v2 whose parent limits it through another mount") {
    // The root hierarchy at cg and the group /tenant/job also at job: the job has no limit, but
    // /tenant, seen only through cg, has 4 GiB with 1 GiB used.
    const fs::path cg = dir / "sys" / "cgroup";
    const fs::path job = dir / "job";
    write(proc / "self" / "cgroup", "0::/tenant/job\n");
    write(proc / "self" / "mountinfo", "24 1 0:22 / " + cg.string() + " rw - cgroup2 cgroup2 rw\n" +
                                           "31 24 0:22 /tenant/job " + job.string() +
                                           " rw - cgroup2 cgroup2 rw\n");
    write(job / "memory.max", "max\n");
    write(job / "memory.current", std::to_string(gib) + "\n");
    write(cg / "tenant" / "memory.max", std::to_string(4 * gib) + "\n");
    write(cg / "tenant" / "memory.current", std::to_string(gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 3 * gib);
  }
  SECTION("cgroup v2 whose ancestors no mount shows") {
    // Only the group /tenant/job is mounted: its own limit counts, /tenant's cannot be read and is
    // reported (the root has none to read). The check then rests on what it could read.
    const fs::path job = dir / "job";
    write(proc / "self" / "cgroup", "0::/tenant/job\n");
    write(proc / "self" / "mountinfo",
          "31 24 0:22 /tenant/job " + job.string() + " rw - cgroup2 cgroup2 rw\n");
    write(job / "memory.max", std::to_string(4 * gib) + "\n");
    write(job / "memory.current", std::to_string(gib) + "\n");
    const auto memory = autan::runtime::host_memory(proc);
    CHECK(memory.available == 3 * gib);
    CHECK(memory.unseen_cgroups == std::vector<std::string>{"/tenant"});
  }
  SECTION("a private cgroup namespace: limits above its root cannot be seen") {
    // Docker on cgroup v2: the namespace's root, "/", mounted at cg. Its inode differs from the
    // initial namespace's (4026531835, checked on Linux 6.17).
    const fs::path cg = dir / "ns";
    fs::create_directories(proc / "self" / "ns");
    write(proc / "self" / "cgroup", "0::/\n");
    write(proc / "self" / "mountinfo",
          "7575 7574 0:32 / " + cg.string() + " ro - cgroup2 cgroup rw,nsdelegate\n");
    write(cg / "memory.current", std::to_string(gib) + "\n");
    const auto link = proc / "self" / "ns" / "cgroup";
    const auto namespace_link = [&](const std::string& target) {
      std::error_code ec;
      fs::remove(link, ec);
      fs::create_symlink(target, link);
    };
    namespace_link("cgroup:[4026535872]");
    // A limit at the root (docker --memory): it counts, scoped to the namespace.
    write(cg / "memory.max", std::to_string(4 * gib) + "\n");
    auto memory = autan::runtime::host_memory(proc);
    CHECK(memory.available == 3 * gib);
    CHECK(memory.unseen_cgroups.empty());
    CHECK(memory.cgroup_scope == "namespace_root");
    // No limit in sight: what is above the namespace root is unseen.
    write(cg / "memory.max", "max\n");
    memory = autan::runtime::host_memory(proc);
    CHECK(memory.available == host);
    CHECK(memory.unseen_cgroups == std::vector<std::string>{"(above the cgroup namespace root)"});
    CHECK(memory.cgroup_scope == "namespace_root");
    // The initial namespace sees the whole hierarchy.
    namespace_link("cgroup:[4026531835]");
    memory = autan::runtime::host_memory(proc);
    CHECK(memory.unseen_cgroups.empty());
    CHECK(memory.cgroup_scope == "host");
  }
  SECTION("cgroup v2 mounted from inside the hierarchy") {
    // /tenant is mounted at cg: the group /tenant/job is cg/job, and /tenant is cg itself.
    const fs::path cg = dir / "mnt";
    write(proc / "self" / "cgroup", "0::/tenant/job\n");
    write(proc / "self" / "mountinfo",
          "30 24 0:22 /tenant " + cg.string() + " rw - cgroup2 cgroup2 rw\n");
    write(cg / "memory.max", std::to_string(8 * gib) + "\n");
    write(cg / "memory.current", std::to_string(2 * gib) + "\n");
    write(cg / "job" / "memory.max", std::to_string(4 * gib) + "\n");
    write(cg / "job" / "memory.current", std::to_string(3 * gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == gib);
  }
  SECTION("cgroup v2 mounted at a path mountinfo escapes") {
    // proc(5): a space in the mount point is written \040.
    const fs::path cg = dir / "cgroup memory";
    std::string escaped = cg.string();
    escaped.replace(escaped.find(' '), 1, "\\040");
    write(proc / "self" / "cgroup", "0::/job\n");
    write(proc / "self" / "mountinfo", "24 1 0:22 / " + escaped + " rw - cgroup2 cgroup2 rw\n");
    write(cg / "job" / "memory.max", std::to_string(4 * gib) + "\n");
    write(cg / "job" / "memory.current", std::to_string(gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 3 * gib);
  }
  SECTION("cgroup v1, whose unlimited groups write the largest page-aligned count") {
    const fs::path cg = dir / "v1";
    write(proc / "self" / "cgroup", "5:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n");
    write(proc / "self" / "mountinfo",
          "40 24 0:35 /docker/abc " + cg.string() +
              " rw - cgroup cgroup rw,memory\n41 24 0:36 / /cpu rw - cgroup cgroup rw,cpu\n");
    write(cg / "memory.limit_in_bytes", std::to_string(4 * gib) + "\n");
    write(cg / "memory.usage_in_bytes", std::to_string(gib) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 3 * gib);
    // Its usage is hierarchical, and so is the inactive cache it counts (total_inactive_file).
    write(cg / "memory.stat", "inactive_file " + std::to_string(gib / 4) +
                                  "\ntotal_inactive_file " + std::to_string(gib / 2) + "\n");
    CHECK(autan::runtime::host_memory(proc).available == 3 * gib + gib / 2);
    write(cg / "memory.limit_in_bytes", "9223372036854771712\n");
    CHECK(autan::runtime::host_memory(proc).available == host);
  }
  SECTION("no MemAvailable") {
    write(proc / "meminfo", "MemTotal: 134217728 kB\n");
    CHECK_THROWS_AS(autan::runtime::host_memory(proc).available, std::runtime_error);
  }
  fs::remove_all(dir);
#endif
}

TEST_CASE("The ranks of a host fit its memory up to the margin, to the byte", "[runtime]") {
  // solver-input.md, `memory`: the need of the host's ranks against what a pool can give less the
  // margin; at exactly that, it fits.
  const MemoryPools one{
      .host_available = 10 * gib, .device_free = {}, .devices = 1, .unseen_cgroups = {}};
  const std::int64_t margin = 2 * gib;
  CHECK_FALSE(memory_shortfall(4 * gib, 0, 2, one, margin));
  CHECK_FALSE(memory_shortfall(3 * gib, gib, 2, one, margin));  // one pool: host and device
  const auto over = memory_shortfall(4 * gib + 1, 0, 2, one, margin);
  REQUIRE(over);
  CHECK(over->required == 8 * gib + 2);
  CHECK(over->available == 10 * gib);
  // A discrete device holds the device shares of the ranks that run on it, the host the rest.
  const MemoryPools split{
      .host_available = 10 * gib, .device_free = 6 * gib, .devices = 2, .unseen_cgroups = {}};
  CHECK_FALSE(memory_shortfall(2 * gib, 2 * gib, 4, split, margin));  // 2 ranks per device
  const auto device = memory_shortfall(gib, 2 * gib + 1, 3, split, margin);
  REQUIRE(device);
  CHECK(device->required == 4 * gib + 2);
  CHECK(device->available == 6 * gib);
  const auto host = memory_shortfall(3 * gib, 0, 4, split, margin);
  REQUIRE(host);
  CHECK(host->available == 10 * gib);
  // Needs too large to count saturate rather than wrap.
  constexpr std::int64_t top = std::numeric_limits<std::int64_t>::max();
  const auto huge = memory_shortfall(top / 2, top / 2, 4, one, margin);
  REQUIRE(huge);
  CHECK(huge->required == top);
}
