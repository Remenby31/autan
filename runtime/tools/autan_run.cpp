// autan-run: runs a solver input and writes its run directory
// (docs/contracts/solver-input.md, docs/contracts/results.md).

#include "autan/runtime/run.hpp"

int main(int argc, char** argv) {  // NOLINT(misc-const-correctness): main's signature
  // Physics implementations register here as the solver lanes provide them.
  const autan::runtime::PhysicsRegistry registry;
  return autan::runtime::run_main(argc, argv, registry);
}
