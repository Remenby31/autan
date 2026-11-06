#include "executable_hash.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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

struct FileStat {
  std::string key;  // device, inode, size, mtime and ctime in nanoseconds
  std::uint64_t device = 0;
  std::int64_t mtime_ns = 0;
  std::int64_t ctime_ns = 0;
};

std::optional<FileStat> file_stat(const std::string& path) {
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
  return FileStat{.key = std::to_string(st.st_dev) + "-" + std::to_string(st.st_ino) + "-" +
                         std::to_string(st.st_size) + "-" + std::to_string(ns(mt)) + "-" +
                         std::to_string(ns(ct)),
                  .device = static_cast<std::uint64_t>(st.st_dev),
                  .mtime_ns = ns(mt),
                  .ctime_ns = ns(ct)};
}

// An entry is exactly 64 lower-case hex digits and a newline.
std::optional<std::string> read_entry(const fs::path& entry) {
  std::ifstream in(entry, std::ios::binary);
  if (!in) return std::nullopt;
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (text.size() != 65 || text.back() != '\n') return std::nullopt;
  text.pop_back();
  const bool hex = std::all_of(text.begin(), text.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
  return hex ? std::optional<std::string>(text) : std::nullopt;
}

// A file whose ctime is this recent may still change within the same timestamp tick on a file
// system with coarse timestamps, so its hash is not stored (git's racy index entries). "Now" is
// the cache file system's own clock, read from the entry written for it, so a skew between that
// clock and the process's cannot shorten the margin.
constexpr std::int64_t settled_ns = 2'000'000'000;

ExecutableHash hash_executable() {
  const std::string path = executable_path();
  const char* dir = std::getenv("AUTAN_BINARY_HASH_CACHE");
  if (dir == nullptr || *dir == '\0') return {hash_file(path), ""};

  // Opt-in cache, for test suites that start many processes: an entry is keyed by what any
  // change to the file changes. A write changes mtime and ctime, a rename or chmod ctime, a copy
  // the inode; ctime cannot be set from user space.
  const auto before = file_stat(path);
  if (!before) return {hash_file(path), "cannot stat the executable"};
  const fs::path entry = fs::path(dir) / (before->key + ".sha256");
  if (const auto stored = read_entry(entry)) return {*stored, ""};

  ExecutableHash h{hash_file(path), ""};
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    h.cache_problem = "cannot create " + std::string(dir) + ": " + ec.message();
    return h;
  }
  const fs::path tmp = entry.string() + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary);
    out << h.sha256 << '\n';
    out.close();
    if (out.fail()) {
      fs::remove(tmp, ec);
      h.cache_problem = "cannot write in " + std::string(dir);
      return h;
    }
  }
  const auto after = file_stat(path);
  const auto now = file_stat(tmp.string());
  if (!after || !now) {
    h.cache_problem = "cannot stat the executable or the cache";
  } else if (after->device != now->device) {
    h.cache_problem = "the executable and the cache are on different file systems";
  } else if (after->key == before->key && now->mtime_ns - after->ctime_ns >= settled_ns) {
    fs::rename(tmp, entry, ec);
    if (ec) h.cache_problem = "cannot store the entry: " + ec.message();
  }
  fs::remove(tmp, ec);
  return h;
}

}  // namespace

const ExecutableHash& executable_hash() {
  // A static whose initialiser throws is initialised again by the next call.
  static const ExecutableHash hash = hash_executable();
  return hash;
}

}  // namespace autan::runtime
