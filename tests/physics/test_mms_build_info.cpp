#include <regex>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "autan_build_info.hpp"

extern "C" {
const char* autan_mms_commit();
int autan_mms_dirty();
}

TEST_CASE("The study's library reports the commit and the dirty flag it was built with", "[mms]") {
  // results.md, solver.commit and solver.dirty: the build's own record, as autan-run's run.json.
  const std::string commit = autan_mms_commit();
  CHECK(std::regex_match(commit, std::regex("[0-9a-f]{40}")));
  CHECK(commit == AUTAN_BUILD_COMMIT);
  CHECK(autan_mms_dirty() == (AUTAN_BUILD_DIRTY ? 1 : 0));
}
