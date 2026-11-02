#pragma once

#include <filesystem>
#include <string>

#include "autan/runtime/physics.hpp"

namespace autan::runtime {

/// Build identity recorded in run.json and the run identity.
struct SolverBuild {
  std::string version;  ///< version.txt
  std::string commit;   ///< full 40-character git commit
};

/// Exit statuses of autan-run (docs/contracts/solver-input.md, Invocation).
enum class ExitStatus : int { finished = 0, failed = 1, invalid_input = 2, signalled = 130 };

/// Runs the case of `input_file` and writes its run directory under `runs`
/// as docs/contracts/results.md specifies; prints the directory on `out`
/// (one line) and invalid-input diagnostics as JSON lines on `err`. The run
/// stops as on SIGINT (status stopped, exit 130) at the first iteration
/// boundary where `stop_file` (if not empty) exists.
ExitStatus run_case(const std::filesystem::path& input_file, const std::filesystem::path& runs,
                    const PhysicsRegistry& registry, const SolverBuild& build, std::FILE* out,
                    std::FILE* err, const std::filesystem::path& stop_file = {});

/// Entry point of autan-run: parses `--input FILE --runs DIR`.
int run_main(int argc, char* const* argv, const PhysicsRegistry& registry);

}  // namespace autan::runtime
