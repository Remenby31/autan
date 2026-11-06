#include "executable_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <picosha2.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace autan::runtime {
namespace {

namespace fs = std::filesystem;

std::string executable_path() {
#ifdef __APPLE__
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::string path(size, '\0');
  if (_NSGetExecutablePath(path.data(), &size) != 0)
    throw std::runtime_error("cannot locate the running executable");
  path.resize(std::char_traits<char>::length(path.c_str()));
  return path;
#else
  return "/proc/self/exe";
#endif
}

std::string hash_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read the running executable '" + path + "'");
  picosha2::hash256_one_by_one hasher;
  std::vector<char> buffer(1 << 20);
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hasher.process(buffer.begin(), buffer.begin() + in.gcount());
  }
  if (in.bad()) throw std::runtime_error("cannot read the running executable '" + path + "'");
  hasher.finish();
  return picosha2::get_hash_hex_string(hasher);
}

struct FileTimes {
  std::string key;  // device, inode, size, mtime and ctime in nanoseconds
  std::int64_t ctime_ns = 0;
};

std::optional<FileTimes> file_times(const std::string& path) {
  struct stat st{};
  if (stat(path.c_str(), &st) != 0) return std::nullopt;
#ifdef __APPLE__
  const timespec mt = st.st_mtimespec, ct = st.st_ctimespec;
#else
  const timespec mt = st.st_mtim, ct = st.st_ctim;
#endif
  const auto ns = [](const timespec& t) {
    return static_cast<std::int64_t>(t.tv_sec) * 1'000'000'000 + t.tv_nsec;
  };
  return FileTimes{.key = std::to_string(st.st_dev) + "-" + std::to_string(st.st_ino) + "-" +
                          std::to_string(st.st_size) + "-" + std::to_string(ns(mt)) + "-" +
                          std::to_string(ns(ct)),
                   .ctime_ns = ns(ct)};
}

bool is_sha256_hex(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

// A file whose ctime is this recent may still change within the same timestamp tick on a file
// system with coarse timestamps, so its hash is not stored (git's racy index entries).
constexpr std::int64_t settled_ns = 2'000'000'000;

std::string hash_executable() {
  const std::string path = executable_path();
  const char* dir = std::getenv("AUTAN_BINARY_HASH_CACHE");
  if (dir == nullptr || *dir == '\0') return hash_file(path);

  // Opt-in cache, for test suites that start many processes: an entry is keyed by what any
  // change to the file changes. A write changes mtime and ctime, a rename or chmod ctime, a copy
  // the inode; ctime cannot be set from user space.
  const auto before = file_times(path);
  if (!before) return hash_file(path);
  const fs::path entry = fs::path(dir) / (before->key + ".sha256");
  {
    std::ifstream in(entry);
    std::string stored;
    if (in && std::getline(in, stored) && is_sha256_hex(stored)) return stored;
  }
  const std::string hash = hash_file(path);
  const auto after = file_times(path);
  const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  if (after && after->key == before->key && now - after->ctime_ns >= settled_ns) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path tmp = entry.string() + "." + std::to_string(getpid()) + ".tmp";
    bool written = false;
    {
      std::ofstream out(tmp);
      out << hash << '\n';
      out.close();
      written = !out.fail();
    }
    if (written) fs::rename(tmp, entry, ec);
    fs::remove(tmp, ec);
  }
  return hash;
}

}  // namespace

std::string executable_sha256() {
  // A static whose initialiser throws is initialised again by the next call.
  static const std::string hash = hash_executable();
  return hash;
}

}  // namespace autan::runtime
