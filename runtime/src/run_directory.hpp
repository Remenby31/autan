#pragma once

#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>

namespace autan::runtime {

/// The run directory of docs/contracts/results.md: prepared under a name
/// starting with '.', published by an atomic rename that fails if the target
/// exists; run.json is replaced atomically, history lines are appended one
/// whole line per write.
class RunDirectory {
 public:
  /// Creates `.<run_id>` under `runs`, writes the initial files (and
  /// manufactured.json, byte for byte, for a verification case), renames it
  /// to `<run_id>`. Throws std::runtime_error when a step fails.
  RunDirectory(const std::filesystem::path& runs, const std::string& run_id,
               const std::string& run_json, const std::string& case_toml,
               const std::string& mesh_info_json,
               const std::optional<std::string>& manufactured_json = std::nullopt);
  RunDirectory(const RunDirectory&) = delete;
  RunDirectory& operator=(const RunDirectory&) = delete;
  ~RunDirectory();

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
  void write_run_json(const std::string& text);
  void append_history(const std::string& line);
  [[nodiscard]] std::filesystem::path solution_staging() const;
  /// Renames the staged solution to solution.cgns.
  void publish_solution();

 private:
  std::filesystem::path path_;
  int history_ = -1;
};

/// Writes `text` to `path` through a temporary file and a rename.
void write_atomically(const std::filesystem::path& path, const std::string& text);

}  // namespace autan::runtime
