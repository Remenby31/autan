#include <cstdio>
#include <string_view>

#include "autan/runtime/run.hpp"
#include "autan_build_info.hpp"

namespace autan::runtime {

int run_main(int argc, char* const* argv, const PhysicsRegistry& registry) {
  constexpr const char* usage =
      "usage: autan-run --input SOLVER_INPUT.json --runs DIR\n"
      "\n"
      "Runs the solver input written by `autan prepare` and writes DIR/<run_id>/\n"
      "(docs/contracts/results.md); prints that directory. Exit status: 0 when\n"
      "the run ended converged, completed, stopped or diverged, 1 when it failed,\n"
      "2 when the input is invalid (diagnostics as JSON on stderr), 130 when\n"
      "stopped by SIGINT or SIGTERM.\n";
  const char* input = nullptr;
  const char* runs = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::fputs(usage, stdout);
      return 0;
    }
    if ((arg == "--input" || arg == "--runs") && i + 1 < argc) {
      (arg == "--input" ? input : runs) = argv[++i];
    } else {
      std::fputs(usage, stderr);
      return static_cast<int>(ExitStatus::invalid_input);
    }
  }
  if (input == nullptr || runs == nullptr) {
    std::fputs(usage, stderr);
    return static_cast<int>(ExitStatus::invalid_input);
  }
  const SolverBuild build{.version = AUTAN_BUILD_VERSION, .commit = AUTAN_BUILD_COMMIT};
  return static_cast<int>(run_case(input, runs, registry, build, stdout, stderr));
}

}  // namespace autan::runtime
