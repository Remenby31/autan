#pragma once

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

namespace autan_test {

// A run.json's diagnostics about the run itself: every entry but the host's memory warnings
// (W0712, W0713), which the runtime records whatever the case (solver-input.md); W0710, W0714 and
// every other code stay.
inline nlohmann::json run_diagnostics(const nlohmann::json& info) {
  const nlohmann::json& diagnostics = info.at("diagnostics");
  REQUIRE(diagnostics.is_array());
  nlohmann::json kept = nlohmann::json::array();
  for (const auto& d : diagnostics)
    if (d.at("code") != "W0712" && d.at("code") != "W0713") kept.push_back(d);
  return kept;
}

}  // namespace autan_test
