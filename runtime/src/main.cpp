#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <system_error>

#include "autan/runtime/run.hpp"
#include "autan_build_info.hpp"
#include "comm.hpp"

#ifdef AUTAN_HAVE_MPI
#include <mpi.h>
#endif

namespace autan::runtime {

namespace {

// A count of bytes: digits only.
bool parse_bytes(std::string_view text, std::int64_t& bytes) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
  return error == std::errc{} && end == text.data() + text.size() && bytes >= 0;
}

}  // namespace

int run_main(int argc, char* const* argv, const PhysicsRegistry& registry) {
  constexpr const char* usage =
      "usage: autan-run --input SOLVER_INPUT.json --runs DIR [--stop-file PATH]\n"
      "                 [--memory-margin BYTES] [--allow-oversubscription]\n"
      "\n"
      "Runs the solver input written by `autan prepare` and writes DIR/<run_id>/\n"
      "(docs/contracts/results.md); prints that directory. Exit status: 0 when\n"
      "the run ended converged, completed, stopped or diverged, 1 when it failed,\n"
      "2 when the input is invalid (diagnostics as JSON on stderr), 130 when\n"
      "stopped by SIGINT or SIGTERM, or when PATH exists at an iteration boundary.\n"
      "Before reading the mesh, it refuses (E0711) a run whose memory estimate exceeds\n"
      "the memory available minus BYTES (default 2 GiB); --allow-oversubscription\n"
      "starts it with a warning (W0712).\n";
  const char* input = nullptr;
  const char* runs = nullptr;
  const char* stop_file = nullptr;
  MemoryPolicy memory;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::fputs(usage, stdout);
      return 0;
    }
    if ((arg == "--input" || arg == "--runs" || arg == "--stop-file") && i + 1 < argc) {
      (arg == "--input" ? input : arg == "--runs" ? runs : stop_file) = argv[++i];
    } else if (arg == "--allow-oversubscription") {
      memory.allow_oversubscription = true;
    } else if (arg == "--memory-margin" && i + 1 < argc &&
               parse_bytes(argv[i + 1], memory.margin_bytes)) {
      ++i;
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
#ifdef AUTAN_HAVE_MPI
  // MPI starts only under a launcher: a plain autan-run is one rank.
  int argc_mpi = argc;
  char** argv_mpi = const_cast<char**>(
      argv);  // NOLINT(cppcoreguidelines-pro-type-const-cast): MPI_Init's signature
  if (launched_by_mpi() && MPI_Init(&argc_mpi, &argv_mpi) != MPI_SUCCESS) {
    std::fputs("autan-run: MPI_Init failed\n", stderr);
    return static_cast<int>(ExitStatus::failed);
  }
  const int status = static_cast<int>(run_case(input, runs, registry, build, stdout, stderr,
                                               stop_file != nullptr ? stop_file : "", memory));
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) MPI_Finalize();
  return status;
#else
  return static_cast<int>(run_case(input, runs, registry, build, stdout, stderr,
                                   stop_file != nullptr ? stop_file : "", memory));
#endif
}

}  // namespace autan::runtime
