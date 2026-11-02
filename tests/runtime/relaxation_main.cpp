// autan-run with the test relaxation physics, for tests that need a real
// process (signals): options come from AUTAN_TEST_* environment variables.
#include <cstdlib>
#include <fstream>
#include <string>

#include "autan/runtime/run.hpp"
#include "relaxation.hpp"

int main(int argc, char** argv) {
  const auto registry =
      autan_test::relaxation_registry(autan_test::RelaxationOptions::from_environment());
  const int status = autan::runtime::run_main(argc, argv, registry);
  // AUTAN_TEST_EXIT_DIR: each rank that returns from run_main, rather than
  // being ended in an abort, writes exit-<rank> there.
  if (const char* dir = std::getenv("AUTAN_TEST_EXIT_DIR")) {
    const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
    if (rank == nullptr) rank = std::getenv("PMI_RANK");
    std::ofstream{std::string(dir) + "/exit-" + (rank != nullptr ? rank : "0")}.put('\n');
  }
  return status;
}
