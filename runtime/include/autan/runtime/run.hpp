#pragma once

#include <cstdint>
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

/// How a run weighs the input's memory estimate (solver-input.md, `memory`)
/// against the memory its host can still give: it refuses to start (E0711)
/// when the estimate exceeds that memory minus `margin_bytes`, or starts
/// with a warning (W0712) when `allow_oversubscription`.
struct MemoryPolicy {
  std::int64_t margin_bytes = std::int64_t{2} << 30;
  bool allow_oversubscription = false;
};

/// Runs the case of `input_file` and writes its run directory under `runs`
/// as docs/contracts/results.md specifies; prints the directory on `out`
/// (one line) and invalid-input diagnostics as JSON lines on `err`. The run
/// stops as on SIGINT (status stopped, exit 130) at the first iteration
/// boundary where `stop_file` (if not empty) exists.
ExitStatus run_case(const std::filesystem::path& input_file, const std::filesystem::path& runs,
                    const PhysicsRegistry& registry, const SolverBuild& build, std::FILE* out,
                    std::FILE* err, const std::filesystem::path& stop_file = {},
                    const MemoryPolicy& memory = {});

/// Entry point of autan-run: parses `--input FILE --runs DIR [--stop-file PATH]
/// [--memory-margin BYTES] [--allow-oversubscription]`.
int run_main(int argc, char* const* argv, const PhysicsRegistry& registry);

}  // namespace autan::runtime
