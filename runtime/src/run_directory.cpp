#include "run_directory.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace autan::runtime {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& what, const fs::path& path) {
  throw std::runtime_error("cannot " + what + " '" + path.string() +
                           "': " + std::system_category().message(errno));
}

void write_file(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
  out.flush();
  if (!out) fail("write", path);
}

// rename(2) replaces an empty target directory; publishing must not.
void rename_no_replace(const fs::path& from, const fs::path& to) {
#if defined(__APPLE__)
  if (renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) != 0) fail("publish", to);
#elif defined(__linux__)
  if (renameat2(AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), RENAME_NOREPLACE) != 0)
    fail("publish", to);
#else
#error "rename without replacement is implemented for Linux and macOS"
#endif
}

}  // namespace

void write_atomically(const fs::path& path, const std::string& text) {
  const fs::path staging = path.parent_path() / ("." + path.filename().string() + ".tmp");
  write_file(staging, text);
  if (std::rename(staging.c_str(), path.c_str()) != 0) fail("replace", path);
}

RunDirectory::RunDirectory(const fs::path& runs, const std::string& run_id,
                           const std::string& run_json, const std::string& case_toml,
                           const std::string& mesh_info_json, const std::string& first_log_line,
                           const std::optional<std::string>& manufactured_json) {
  std::error_code ec;
  fs::create_directories(runs, ec);
  if (ec) fail("create", runs);
  const fs::path staging = runs / ("." + run_id);
  if (!fs::create_directory(staging, ec) || ec) fail("create", staging);
  write_file(staging / "run.json", run_json);
  write_file(staging / "case.toml", case_toml);
  write_file(staging / "mesh-info.json", mesh_info_json);
  if (manufactured_json) write_file(staging / "manufactured.json", *manufactured_json);
  write_file(staging / "log.jsonl", first_log_line + "\n");
  path_ = runs / run_id;
  rename_no_replace(staging, path_);
}

RunDirectory::~RunDirectory() {
  if (history_ >= 0) ::close(history_);
  if (log_ >= 0) ::close(log_);
}

void RunDirectory::write_run_json(const std::string& text) {
  write_atomically(path_ / "run.json", text);
}

void RunDirectory::append_history(const std::string& line) {
  append(history_, "history.jsonl", line);
}

void RunDirectory::append_log(const std::string& line) { append(log_, "log.jsonl", line); }

// Readers see whole lines, except possibly the last one of a running run; a
// line that cannot be completed is cut off again so the file stays whole once
// the run is finalised.
void RunDirectory::append(int& fd, const char* name, const std::string& line) {
  if (fd < 0) {
    fd = ::open((path_ / name).c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) fail("open", path_ / name);
  }
  const std::string whole = line + "\n";
  const off_t start = ::lseek(fd, 0, SEEK_END);
  std::size_t done = 0;
  while (done < whole.size()) {
    const ssize_t n = ::write(fd, whole.data() + done, whole.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      const int saved = errno;
      // Best effort: if the cut fails too, the run fails anyway.
      if (start >= 0 && ::ftruncate(fd, start) != 0) {
      }
      errno = saved;
      fail("append to", path_ / name);
    }
    done += static_cast<std::size_t>(n);
  }
}

fs::path RunDirectory::solution_staging() const { return path_ / ".solution.cgns.tmp"; }

void RunDirectory::publish_solution() {
  if (std::rename(solution_staging().c_str(), (path_ / "solution.cgns").c_str()) != 0) {
    fail("publish", path_ / "solution.cgns");
  }
}

}  // namespace autan::runtime
