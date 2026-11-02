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

}  // namespace

TEST_CASE("The memory a host can still give is the least of MemAvailable and its cgroup limits",
          "[runtime]") {
#if defined(__APPLE__)
  SKIP("macOS has no /proc: its free and inactive pages are read through Mach");
#else
  // proc(5): /proc/meminfo in kB, /proc/self/cgroup and /proc/self/mountinfo; the limits in
  // memory.max / memory.current (cgroup v2) or memory.limit_in_bytes / memory.usage_in_bytes (v1).
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
  const MemoryPools one{.host_available = 10 * gib, .device_free = {}, .devices = 1};
  const std::int64_t margin = 2 * gib;
  CHECK_FALSE(memory_shortfall(4 * gib, 0, 2, one, margin));
  CHECK_FALSE(memory_shortfall(3 * gib, gib, 2, one, margin));  // one pool: host and device
  const auto over = memory_shortfall(4 * gib + 1, 0, 2, one, margin);
  REQUIRE(over);
  CHECK(over->required == 8 * gib + 2);
  CHECK(over->available == 10 * gib);
  // A discrete device holds the device shares of the ranks that run on it, the host the rest.
  const MemoryPools split{.host_available = 10 * gib, .device_free = 6 * gib, .devices = 2};
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
