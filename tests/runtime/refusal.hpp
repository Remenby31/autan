#pragma once

#include <optional>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

namespace autan_test {

// The error `code` that refused an autan-run, read from its stderr. solver-input.md, exit 2:
// one diagnostic per line, the host's memory warnings written before the refusal (W0712, W0713)
// and the error; a launcher merging several ranks' streams may put a warning after another rank's
// error, and each process may write the same error. The calling test fails when no line holds
// that error, or when any other line is not W0712 or W0713; under an MPI launcher, `launched`
// skips the launcher's own lines, those that are not JSON objects.
inline nlohmann::json refusal_in(const std::string& err, const std::string& code,
                                 bool launched = false) {
  std::optional<nlohmann::json> refusal;
  std::istringstream lines(err);
  for (std::string line; std::getline(lines, line);) {
    if (line.empty() || (launched && !line.starts_with('{'))) continue;
    INFO(line);
    REQUIRE(line.starts_with('{'));
    const nlohmann::json diagnostic = nlohmann::json::parse(line);
    if (diagnostic.at("code") == code) {
      CHECK(diagnostic.at("severity") == "error");
      if (!refusal) refusal = diagnostic;
    } else {
      CHECK((diagnostic.at("code") == "W0712" || diagnostic.at("code") == "W0713"));
    }
  }
  INFO(err);
  REQUIRE(refusal.has_value());
  return *refusal;
}

}  // namespace autan_test
