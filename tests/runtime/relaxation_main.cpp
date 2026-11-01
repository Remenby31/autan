// autan-run with the test relaxation physics, for tests that need a real
// process (signals): options come from AUTAN_TEST_* environment variables.
#include <cstdlib>
#include <fstream>
#include <string>

#include "autan/runtime/run.hpp"
#include "relaxation.hpp"

namespace {

// AUTAN_TEST_EXIT_DIR: each rank that returns from main, rather than being
// ended by MPI_Abort (which runs no exit handler), writes exit-<rank> there.
std::string exit_marker;

void mark_exit() { std::ofstream{exit_marker}.put('\n'); }

}  // namespace

int main(int argc, char** argv) {
  if (const char* dir = std::getenv("AUTAN_TEST_EXIT_DIR")) {
    const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (rank == nullptr) rank = std::getenv("PMI_RANK");
    exit_marker = std::string(dir) + "/exit-" + (rank != nullptr ? rank : "0");
    std::atexit(mark_exit);
  }
  const auto registry =
      autan_test::relaxation_registry(autan_test::RelaxationOptions::from_environment());
  return autan::runtime::run_main(argc, argv, registry);
}
