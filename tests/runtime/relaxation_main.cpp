// autan-run with the test relaxation physics, for tests that need a real
// process (signals): options come from AUTAN_TEST_* environment variables.
#include "autan/runtime/run.hpp"
#include "relaxation.hpp"

int main(int argc, char** argv) {
  const auto registry =
      autan_test::relaxation_registry(autan_test::RelaxationOptions::from_environment());
  return autan::runtime::run_main(argc, argv, registry);
}
