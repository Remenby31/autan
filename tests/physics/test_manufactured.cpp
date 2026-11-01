// Manufactured solutions read from autan.mms/1 documents (autan-vv's export of `euler`).
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "autan/physics/manufactured.hpp"

namespace {

std::string read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

const std::filesystem::path euler_mms{AUTAN_EULER_MMS};

}  // namespace

TEST_CASE("the euler solution evaluates to the fields of Roy et al. (2004)", "[manufactured]") {
  const autan::physics::Manufactured mms(read(euler_mms));
  CHECK(mms.physics() == "compressible");
  CHECK(mms.turbulence() == "inviscid");
  const std::vector<std::array<double, 3>> points = {{0.3, 0.6, 0.9}, {0.0, 1.0, 0.25}};
  const std::array<std::string, 2> names = {"rho", "p"};
  const auto w = mms.evaluate("fields", names, points);
  const double pi = std::numbers::pi;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto [x, y, z] = points[i];
    // vv/src/autan_vv/mms/solutions.py, ROY_SUBSONIC_3D.
    const double rho = 1.0 + 0.15 * std::sin(pi * x) - 0.1 * std::cos(0.5 * pi * y) +
                       0.08 * std::sin(0.75 * pi * z);
    const double p = 1e5 - 3e4 * std::cos(pi * x) + 2e4 * std::sin(1.25 * pi * y) +
                     1e4 * std::cos(0.75 * pi * z);
    CHECK(std::abs(w[0][i] - rho) <= 16 * 2.2e-16 * rho);
    CHECK(std::abs(w[1][i] - p) <= 16 * 2.2e-16 * p);
  }
}

TEST_CASE("the euler mass source is the divergence of the momentum", "[manufactured]") {
  const autan::physics::Manufactured mms(read(euler_mms));
  const std::array<double, 3> x0 = {0.37, 0.52, 0.81};
  const double h = 1e-4;
  std::vector<std::array<double, 3>> points;
  for (std::size_t a = 0; a < 3; ++a) {
    for (const double s : {-h, h}) {
      auto x = x0;
      x[a] += s;
      points.push_back(x);
    }
  }
  const std::array<std::string, 3> momentum = {"rhou", "rhov", "rhow"};
  const auto m = mms.evaluate("conserved", momentum, points);
  double divergence = 0.0;
  for (std::size_t a = 0; a < 3; ++a) divergence += (m[a][2 * a + 1] - m[a][2 * a]) / (2 * h);
  const std::array<std::string, 1> mass = {"rho"};
  const std::vector<std::array<double, 3>> at = {x0};
  const double source = mms.evaluate("sources", mass, at)[0][0];
  // Central differences: O(h^2) truncation, relative to the momentum gradient's scale.
  CHECK(std::abs(source - divergence) <= 1e-6 * 100.0 * std::numbers::pi * 20.0);
}

TEST_CASE("an expression outside the autan.mms/1 grammar is refused", "[manufactured]") {
  auto doc = [](const std::string& expr) {
    return R"({"schema": "autan.mms/1", "model": {"physics": "compressible", "turbulence": null},
               "parameters": {}, "fields": {"subexpressions": [], "outputs": {"rho": ")" +
           expr + R"("}}})";
  };
  CHECK_NOTHROW(autan::physics::Manufactured(doc("pow(x, 2) + exp(-y) / (1 + pi)")));
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("system(1)")), std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("x ** 2")), std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(doc("t0 + 1")), std::invalid_argument);
  auto with_subexpressions = [](const std::string& subexpressions, const std::string& expr) {
    return R"({"schema": "autan.mms/1", "model": {"physics": "compressible", "turbulence": null},
               "parameters": {}, "fields": {"subexpressions": )" +
           subexpressions + R"(, "outputs": {"rho": ")" + expr + R"("}}})";
  };
  const std::array<std::string, 1> rho = {"rho"};
  const std::vector<std::array<double, 3>> origin = {{0.0, 0.0, 0.0}};
  const autan::physics::Manufactured chained(
      with_subexpressions(R"([["a", "1"], ["b", "a + 2"], ["c", "b * 3"]])", "c - a"));
  CHECK(chained.evaluate("fields", rho, origin)[0][0] == 8.0);
  CHECK_THROWS_AS(
      autan::physics::Manufactured(with_subexpressions(R"([["a", "1"], ["a", "2"]])", "a")),
      std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(with_subexpressions(R"([["a", "1", "2"]])", "a")),
                  std::invalid_argument);
  CHECK_THROWS_AS(autan::physics::Manufactured(with_subexpressions(R"({"a": "1"})", "a")),
                  std::invalid_argument);
}
